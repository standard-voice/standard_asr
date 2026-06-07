# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Full-duplex streaming transcription protocol (spec, section "Streaming").

This module defines the streaming event model and session machinery:

* :class:`TranscriptionEvent` -- the 6-type event (``partial`` / ``final`` /
  ``supersede`` / ``progress`` / ``done`` / ``error``) carrying a stable
  ``segment_id``, cumulative ``text``, a conservative ``stable_until`` codepoint
  frontier, and an ``audio_processed_until`` cursor.
* :func:`validate_stable_until` -- enforces the combining-character invariant
  (``stable_until`` MUST NOT split a combining sequence) using stdlib
  ``unicodedata`` only.
* :func:`reduce_event` / :class:`StreamReducer` -- the canonical application-side
  reduce (including the core ``supersede`` handling) and reduction of a session
  to a :class:`~standard_asr.results.TranscriptionResult`.
* :class:`TranscriptionSession` -- an async-first, full-duplex session base.
  Authors implement the async ``_open`` / ``_produce`` / ``_close`` hooks; the
  base provides ``feed`` vs manual ``send_audio`` / ``end_audio`` single
  ownership, bounded backpressure-aware iteration, lifecycle enforcement, a
  bounded rolling audio buffer + reconnect scaffolding, an overall idle/wall
  termination deadline, and result reduction.
* :class:`SyncSession` -- the standard sync bridge (one background event loop in
  a thread, owned by the session), so authors only ever write async. Lifecycle
  submits carry a timeout so a hanging adapter can never deadlock the caller.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
import unicodedata
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .exceptions import StreamClosedError
from .results import Diagnostic, Segment, TranscriptionResult, Word

EventType = Literal["partial", "final", "supersede", "progress", "done", "error"]

#: Default seconds to wait for *any* further event before synthesizing a
#: terminal error. This bounds the gap between consecutive events; see
#: :data:`DEFAULT_MAX_IDLE` for the content-progress deadline and
#: :data:`DEFAULT_MAX_SESSION_SECONDS` for the absolute wall-clock cap.
DEFAULT_DONE_TIMEOUT = 30.0

#: Default seconds without a *content* event (``partial`` / ``final`` /
#: ``supersede``) before the session is force-terminated. Unlike the
#: per-event ``done_timeout`` this is NOT reset by ``progress`` heartbeats, so a
#: chatty-but-stuck engine (e.g. a DSM model emitting only heartbeats) is still
#: guaranteed to terminate (spec ST.6.1: the iterator ALWAYS terminates).
DEFAULT_MAX_IDLE = 120.0

#: Default absolute session wall-clock cap in seconds. ``None`` disables it.
#: A finite default guarantees termination even if both content and heartbeat
#: events keep arriving forever.
DEFAULT_MAX_SESSION_SECONDS: float | None = None

#: Default capacity (number of non-coalesced events) of the send-side event
#: buffer before overflow (spec ST.6.4: bounded send-side buffer, overflow emits
#: an error). Coalesced partials do not count against this bound.
DEFAULT_EVENT_BUFFER_CAPACITY = 1024

#: Default capacity (number of pending audio chunks) of the audio queue, so
#: ``feed`` / ``send_audio`` exert real backpressure on a slow engine.
DEFAULT_AUDIO_QUEUE_MAXSIZE = 256

#: Default capacity (number of chunks) of the bounded rolling audio buffer used
#: to replay recent audio after an internal reconnect (spec ST.6.3 / D10.7).
DEFAULT_AUDIO_HISTORY_MAXLEN = 256


async def _cancel_all_tasks() -> None:
    """Cancel and await every task on the running loop except the caller.

    Used by the sync bridge during teardown so no task is destroyed while
    pending (which would emit a warning under ``-W error``).
    """
    current = asyncio.current_task()
    tasks = [task for task in asyncio.all_tasks() if task is not current]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def validate_stable_until(text: str, stable_until: int) -> bool:
    """Return whether ``stable_until`` is a valid frozen-prefix boundary.

    ``stable_until`` is a codepoint count; ``text[:stable_until]`` is the frozen
    prefix. It MUST NOT split a Unicode combining sequence -- i.e. the codepoint
    at the cut (if any) must not be a combining mark (spec ST.4.2). Validated
    with stdlib ``unicodedata`` only.

    Args:
        text: The segment text.
        stable_until: The proposed frozen-prefix length in codepoints.

    Returns:
        ``True`` if the boundary is valid.
    """
    if stable_until < 0 or stable_until > len(text):
        return False
    if stable_until == 0 or stable_until == len(text):
        return True
    return unicodedata.combining(text[stable_until]) == 0


class TranscriptionEvent(BaseModel):
    """A single streaming transcription event.

    Args:
        type: The event type.
        segment_id: Stable id of the segment this event concerns.
        text: The segment's complete current text (cumulative/replace).
        stable_until: Frozen-prefix length in codepoints (monotonic per segment).
        finality: For ``final`` events, ``"final"`` or ``"closed"``.
        words: Optional word-level detail (shares the batch ``Word`` model).
        start: Segment start time in seconds (origin = first session sample).
        end: Segment end time in seconds.
        audio_processed_until: Monotonic audio-time cursor in seconds.
        old_ids: For ``supersede``, the retired segment ids.
        new_ids: For ``supersede``, the replacement segment ids.
        code: For ``error``, the error code.
        recoverable: For ``error``, whether the session may continue.
        retriable_after: For ``error``, suggested retry delay in seconds.
        reconnect: For ``progress``, whether this marks a reconnect.
        gap_start: For a reconnect ``progress``, the gap start time.
        gap_end: For a reconnect ``progress``, the gap end time.
        extra: Engine-specific extra data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: EventType
    segment_id: str | None = None
    text: str | None = None
    stable_until: int | None = None
    finality: Literal["final", "closed"] = "final"
    words: list[Word] | None = None
    start: float | None = None
    end: float | None = None
    audio_processed_until: float | None = None
    old_ids: list[str] = Field(default_factory=list)
    new_ids: list[str] = Field(default_factory=list)
    code: str | None = None
    recoverable: bool | None = None
    retriable_after: float | None = None
    reconnect: bool | None = None
    gap_start: float | None = None
    gap_end: float | None = None
    detected_language: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def stable_text(self) -> str:
        """The frozen prefix of ``text`` (``text[:stable_until]``).

        Guards against an invalid (negative or out-of-range) ``stable_until`` so
        a malformed frontier never produces a wrong or oversized prefix.

        Returns:
            The frozen prefix, or ``""`` if nothing is validly frozen.
        """
        if self.text is None or self.stable_until is None:
            return ""
        if self.stable_until <= 0:
            return ""
        return self.text[: self.stable_until]

    @property
    def is_content(self) -> bool:
        """Whether this event advances transcription content.

        Returns:
            ``True`` for ``partial`` / ``final`` / ``supersede`` (events that
            move the transcription forward, as opposed to ``progress``
            heartbeats / ``done`` / ``error``).
        """
        return self.type in ("partial", "final", "supersede")

    @property
    def is_terminal(self) -> bool:
        """Whether this event ends the session.

        Returns:
            ``True`` for ``done`` or a non-recoverable ``error``.
        """
        if self.type == "done":
            return True
        return self.type == "error" and self.recoverable is False

    @classmethod
    def partial(cls, segment_id: str, text: str, **kw: Any) -> TranscriptionEvent:
        """Build a ``partial`` event.

        Args:
            segment_id: The segment id.
            text: The segment's complete current text.
            **kw: Additional event fields.

        Returns:
            A ``partial`` event.
        """
        return cls(type="partial", segment_id=segment_id, text=text, **kw)

    @classmethod
    def final(cls, segment_id: str, text: str, **kw: Any) -> TranscriptionEvent:
        """Build a ``final`` event.

        Args:
            segment_id: The segment id.
            text: The segment's final text.
            **kw: Additional event fields.

        Returns:
            A ``final`` event.
        """
        return cls(type="final", segment_id=segment_id, text=text, **kw)

    @classmethod
    def closed(cls, segment_id: str, text: str, **kw: Any) -> TranscriptionEvent:
        """Build a ``closed`` finality event (a ``final`` with finality=closed).

        Args:
            segment_id: The segment id.
            text: The segment's possibly post-processed text.
            **kw: Additional event fields.

        Returns:
            A ``final`` event marked ``finality="closed"``.
        """
        return cls(type="final", segment_id=segment_id, text=text, finality="closed", **kw)

    @classmethod
    def supersede(cls, old_ids: list[str], new_ids: list[str], **kw: Any) -> TranscriptionEvent:
        """Build a ``supersede`` event replacing old segments with new ones.

        Args:
            old_ids: The retired segment ids.
            new_ids: The replacement segment ids (must be disjoint from old).
            **kw: Additional event fields.

        Returns:
            A ``supersede`` event.

        Raises:
            ValueError: If ``old_ids`` and ``new_ids`` intersect.
        """
        if set(old_ids) & set(new_ids):
            raise ValueError("supersede old_ids and new_ids MUST be disjoint.")
        return cls(type="supersede", old_ids=old_ids, new_ids=new_ids, **kw)

    @classmethod
    def progress(cls, **kw: Any) -> TranscriptionEvent:
        """Build a ``progress`` event (heartbeat / cursor / reconnect notice).

        Args:
            **kw: Event fields (e.g. ``audio_processed_until``, ``reconnect``).

        Returns:
            A ``progress`` event.
        """
        return cls(type="progress", **kw)

    @classmethod
    def done(cls, **kw: Any) -> TranscriptionEvent:
        """Build a terminal ``done`` event.

        Args:
            **kw: Additional event fields.

        Returns:
            A ``done`` event.
        """
        return cls(type="done", **kw)

    @classmethod
    def make_error(cls, code: str, *, recoverable: bool = False, **kw: Any) -> TranscriptionEvent:
        """Build an ``error`` event.

        Args:
            code: The error code.
            recoverable: Whether the session may continue.
            **kw: Additional event fields.

        Returns:
            An ``error`` event.
        """
        return cls(type="error", code=code, recoverable=recoverable, **kw)


def reduce_event(segments: dict[str, str], event: TranscriptionEvent) -> None:
    """Apply the canonical streaming reduce to a ``{segment_id: text}`` map.

    This is the core reduce every compliant application implements, including
    ``supersede`` handling (spec ST.5.2). Non-text events are ignored.

    Args:
        segments: The mutable segment-text map to update in place.
        event: The event to apply.
    """
    if event.type in ("partial", "final") and event.segment_id is not None:
        segments[event.segment_id] = event.text or ""
    elif event.type == "supersede":
        for old_id in event.old_ids:
            segments.pop(old_id, None)


class StreamReducer:
    """Reduces a stream of events into a :class:`TranscriptionResult`.

    Tracks finalized segments in arrival order, honouring ``supersede`` removals,
    so :meth:`result` reflects the session's committed transcription.

    Timestamp handling: many engines (e.g. Qwen3 streaming) emit no timestamps.
    Rather than fabricate ``start=0.0`` / ``end=0.0`` (which would collapse and
    mis-sort the transcript), the reducer preserves arrival order and only sorts
    by ``start`` when *every* retained segment carries a real timestamp.
    """

    def __init__(self) -> None:
        """Initialize an empty reducer."""
        self._segments: dict[str, Segment] = {}
        self._has_timestamp: dict[str, bool] = {}
        self._order: list[str] = []
        self._detected_language: str | None = None

    def add(self, event: TranscriptionEvent) -> None:
        """Incorporate one event into the running result.

        Args:
            event: The event to incorporate.
        """
        if event.detected_language is not None:
            self._detected_language = event.detected_language
        if event.type == "final" and event.segment_id is not None:
            if event.segment_id not in self._segments:
                self._order.append(event.segment_id)
            has_ts = event.start is not None
            # Do NOT fabricate 0.0 timestamps for timestamp-less engines: keep
            # a sentinel-free Segment whose start/end are 0.0 only when the
            # engine genuinely had none, and remember that fact so result()
            # never sorts on a fabricated value.
            self._segments[event.segment_id] = Segment(
                start=event.start if event.start is not None else 0.0,
                end=event.end if event.end is not None else (event.start or 0.0),
                text=event.text or "",
                words=event.words,
            )
            self._has_timestamp[event.segment_id] = has_ts
        elif event.type == "supersede":
            for old_id in event.old_ids:
                if old_id in self._segments:
                    del self._segments[old_id]
                    self._has_timestamp.pop(old_id, None)
                    self._order.remove(old_id)

    def result(self) -> TranscriptionResult:
        """Build the reduced transcription result.

        Returns:
            A :class:`TranscriptionResult` from the committed segments. Ordered
            by ``start`` only when every segment carries a real timestamp;
            otherwise arrival order is preserved.
        """
        order = list(self._order)
        if order and all(self._has_timestamp.get(sid, False) for sid in order):
            order.sort(key=lambda sid: self._segments[sid].start)
        segments = [self._segments[sid] for sid in order]
        text = " ".join(s.text for s in segments).strip()
        return TranscriptionResult(
            text=text,
            segments=segments or None,
            detected_language=self._detected_language,
        )


class EventBufferOverflow(Exception):
    """Internal signal: the bounded send-side event buffer overflowed.

    Never propagates to applications; the producer converts it into a terminal
    ``error(code="backpressure")`` event (spec ST.6.4).
    """


class _CoalescingBuffer:
    """An async event buffer with partial coalescing (spec ST.6.4 backpressure).

    Pending ``partial`` events are merged per ``segment_id`` (latest wins); a
    same-segment ``final`` / ``closed`` / ``supersede`` invalidates and DROPS the
    pending partial so a replaced/finalized segment can never revive (spec
    ST.6.4: "合并 MUST 被同 segment 的 final/closed/supersede 作废 ... 该 partial
    MUST 丢弃"). ``final`` / ``supersede`` / ``done`` / ``error`` are never
    dropped or reordered.

    The buffer is **bounded**: at most ``capacity`` non-coalesced events may be
    pending. Coalesced partials reuse their existing slot and never grow the
    buffer. Enqueuing past capacity raises :class:`EventBufferOverflow`, which
    the producer turns into a terminal ``backpressure`` error.
    """

    def __init__(self, capacity: int = DEFAULT_EVENT_BUFFER_CAPACITY) -> None:
        """Initialize the buffer.

        Args:
            capacity: Maximum number of pending non-coalesced events.
        """
        self._capacity = capacity
        # deque of (event, alive) where alive=False marks a coalesced partial
        # that was invalidated in place (lazily skipped on get) so we keep O(1)
        # amortized put/get without an O(n) reindex.
        self._items: deque[_Slot] = deque()
        self._partial_slot: dict[str, _Slot] = {}
        self._live_count = 0
        self._event = asyncio.Event()
        self._closed = False

    def put(self, event: TranscriptionEvent) -> None:
        """Add an event, coalescing superseded partials.

        Args:
            event: The event to enqueue.

        Raises:
            EventBufferOverflow: If the buffer is at capacity and the event is
                not a coalescible partial reusing an existing slot.
        """
        if event.type == "partial" and event.segment_id is not None:
            slot = self._partial_slot.get(event.segment_id)
            if slot is not None and slot.alive:
                # Coalesce in place: latest partial wins, no growth.
                slot.event = event
                self._event.set()
                return
            self._reserve()
            slot = _Slot(event)
            self._partial_slot[event.segment_id] = slot
            self._items.append(slot)
            self._live_count += 1
            self._event.set()
            return

        # A terminal-for-segment event invalidates and DROPS any pending
        # partial for that/those segment(s) so a dead segment never revives.
        if event.type in ("final", "supersede"):
            targets = [event.segment_id, *event.old_ids]
            for sid in targets:
                if sid is None:
                    continue
                stale = self._partial_slot.pop(sid, None)
                if stale is not None and stale.alive:
                    stale.alive = False
                    self._live_count -= 1
        self._reserve()
        self._items.append(_Slot(event))
        self._live_count += 1
        self._event.set()

    def put_forced(self, event: TranscriptionEvent) -> None:
        """Append a terminal event bypassing the capacity bound.

        Terminal events (``done`` / ``error``) MUST never be dropped (spec
        ST.6.4), even when the buffer overflowed because the consumer was slow.
        They are few (the producer stops after one) so bypassing the bound is
        safe, and -- crucially -- they go into the *same* buffer the iterator is
        already awaiting, so the terminal event is delivered promptly.

        Args:
            event: The terminal event to append.
        """
        self._items.append(_Slot(event))
        self._live_count += 1
        self._event.set()

    def _reserve(self) -> None:
        """Ensure room for one more live event.

        Raises:
            EventBufferOverflow: If already at capacity.
        """
        if self._live_count >= self._capacity:
            raise EventBufferOverflow

    def close(self) -> None:
        """Signal that no further events will be added."""
        self._closed = True
        self._event.set()

    async def get(self) -> TranscriptionEvent | None:
        """Pop the next live event, awaiting one if necessary.

        Returns:
            The next event, or ``None`` once closed and drained.
        """
        while True:
            while self._items:
                slot = self._items.popleft()
                if not slot.alive:
                    continue  # invalidated (coalesced-away) partial: skip.
                self._live_count -= 1
                event = slot.event
                if (
                    event.type == "partial"
                    and event.segment_id is not None
                    and self._partial_slot.get(event.segment_id) is slot
                ):
                    del self._partial_slot[event.segment_id]
                return event
            if self._closed:
                return None
            self._event.clear()
            await self._event.wait()


class _Slot:
    """A mutable buffer slot allowing in-place coalescing and invalidation."""

    __slots__ = ("alive", "event")

    def __init__(self, event: TranscriptionEvent) -> None:
        """Initialize a live slot.

        Args:
            event: The event held by the slot.
        """
        self.event = event
        self.alive = True


class _LifecycleGuard:
    """Enforces segment lifecycle + ``stable_until`` invariants (spec ST.5.1).

    Defense in depth: the spec assigns suppression of illegal transitions to the
    adapter (MUST), but a wrong transcript is the cardinal sin, so the base
    independently guards. By default illegal events are SUPPRESSED and a
    structured :class:`Diagnostic` is recorded; in ``strict`` mode the guard
    raises instead. A ``stable_until`` decrease is CLAMPED to its prior value
    (it MUST only increase, spec ST.4.2) with a diagnostic.

    States per segment id: ``open`` -> ``final`` -> ``closed`` (terminal), or
    ``superseded`` (terminal). ``new_ids`` from a supersede start ``open``.
    """

    def __init__(self, *, strict: bool = False) -> None:
        """Initialize the guard.

        Args:
            strict: If ``True``, raise on an illegal transition instead of
                suppressing it.
        """
        self._strict = strict
        self._state: dict[str, str] = {}
        self._stable_until: dict[str, int] = {}
        self.diagnostics: list[Diagnostic] = []

    def _reject(self, code: str, message: str) -> None:
        """Record a suppression diagnostic or raise in strict mode.

        Args:
            code: Diagnostic code.
            message: Human-readable explanation.

        Raises:
            ValueError: In strict mode.
        """
        if self._strict:
            raise ValueError(message)
        self.diagnostics.append(Diagnostic(level="warning", code=code, message=message))

    def admit(self, event: TranscriptionEvent) -> TranscriptionEvent | None:
        """Validate (and possibly clamp) an event before it is forwarded.

        Args:
            event: The raw event from the producer.

        Returns:
            The event to forward (possibly with a clamped ``stable_until``), or
            ``None`` if the event is an illegal transition and was suppressed.
        """
        sid = event.segment_id
        if event.type == "supersede":
            for old in event.old_ids:
                if self._state.get(old) == "closed":
                    self._reject(
                        "lifecycle_closed_superseded",
                        f"supersede old_ids contains closed segment {old!r}; "
                        "suppressed (spec ST.5.3: closed MUST NOT be superseded).",
                    )
                    return None
            for old in event.old_ids:
                self._state[old] = "superseded"
            for new in event.new_ids:
                self._state.setdefault(new, "open")
            return event

        if event.type in ("partial", "final") and sid is not None:
            state = self._state.get(sid, "open")
            if state in ("superseded", "closed"):
                self._reject(
                    "lifecycle_after_terminal",
                    f"{event.type} for segment {sid!r} after it became {state}; "
                    "suppressed (spec ST.5.1 illegal transition).",
                )
                return None
            if event.type == "partial" and state == "final":
                self._reject(
                    "lifecycle_partial_after_final",
                    f"partial for segment {sid!r} after final; suppressed "
                    "(spec ST.5.1 illegal transition).",
                )
                return None
            event = self._clamp_stable_until(event, sid)
            if event.type == "final":
                self._state[sid] = "closed" if event.finality == "closed" else "final"
            else:
                self._state[sid] = "open"
            return event

        return event

    def _clamp_stable_until(self, event: TranscriptionEvent, sid: str) -> TranscriptionEvent:
        """Clamp a decreasing or invalid ``stable_until`` (spec ST.4.2).

        Args:
            event: The partial/final event.
            sid: The segment id.

        Returns:
            The event, with ``stable_until`` clamped if it decreased or was an
            invalid boundary; otherwise the event unchanged.
        """
        su = event.stable_until
        if su is None:
            return event
        prior = self._stable_until.get(sid, 0)
        text = event.text or ""
        clamped = su
        reason = ""
        if su < prior:
            clamped = prior
            reason = (
                f"stable_until decreased {su} -> clamped to {prior} "
                "(spec ST.4.2: MUST only increase)"
            )
        if not validate_stable_until(text, clamped):
            # Fall back to the largest valid boundary <= clamped, else prior/0.
            safe = clamped
            while safe > 0 and not validate_stable_until(text, safe):
                safe -= 1
            if reason:
                reason += "; "
            reason += f"stable_until {clamped} invalid boundary -> {safe}"
            clamped = safe
        if clamped != su:
            self._reject("stable_until_clamped", reason)
            event = event.model_copy(update={"stable_until": clamped})
        self._stable_until[sid] = clamped
        return event


class TranscriptionSession(ABC):
    """Async-first, full-duplex streaming session base.

    Authors implement :meth:`_produce` (and optionally :meth:`_open` /
    :meth:`_close`), reading fed audio via :meth:`audio_chunks` and yielding
    :class:`TranscriptionEvent` objects. The base manages input ownership,
    bounded backpressure, lifecycle enforcement, a bounded rolling audio buffer
    + reconnect scaffolding, termination deadlines, and result reduction.

    Reconnect contract (spec ST.6.3 / D10.7) -- base vs adapter:

    * The **base** owns a bounded rolling audio buffer (the most recent fed
      chunks) and exposes :meth:`replay_buffer`, the source ``replayable``
      classification, and :meth:`note_reconnect`. It cannot detect a
      reconnect itself (it owns no network connection).
    * The **adapter** detects the disconnect, re-establishes the connection,
      replays :meth:`replay_buffer` audio, keeps ``segment_id`` / timestamps /
      detected language continuous, then calls :meth:`note_reconnect`. The base
      then emits the ``progress(reconnect=True, gap_start, gap_end)`` event and,
      for a non-replayable source whose buffer overflowed during the gap, a
      trailing ``error(code="content_lost", recoverable=False)``.
    """

    def __init__(
        self,
        *,
        done_timeout: float = DEFAULT_DONE_TIMEOUT,
        max_idle: float | None = DEFAULT_MAX_IDLE,
        max_session_seconds: float | None = DEFAULT_MAX_SESSION_SECONDS,
        event_buffer_capacity: int = DEFAULT_EVENT_BUFFER_CAPACITY,
        audio_queue_maxsize: int = DEFAULT_AUDIO_QUEUE_MAXSIZE,
        audio_history_maxlen: int = DEFAULT_AUDIO_HISTORY_MAXLEN,
        strict_lifecycle: bool = False,
    ) -> None:
        """Initialize the session.

        Args:
            done_timeout: Seconds to await *any* further event before
                synthesizing a ``done_timeout`` error. Bounds the per-event gap;
                reset by every event (including ``progress`` heartbeats).
            max_idle: Seconds without a *content* event (``partial`` / ``final``
                / ``supersede``) before force-terminating with a
                ``stream_stalled`` error. NOT reset by ``progress`` heartbeats,
                so a heartbeat-only engine still terminates. ``None`` disables.
            max_session_seconds: Absolute wall-clock cap; ``None`` disables.
            event_buffer_capacity: Max pending non-coalesced events before
                overflow emits a ``backpressure`` error.
            audio_queue_maxsize: Max pending audio chunks; bounds ``feed`` /
                ``send_audio`` so a slow engine exerts real backpressure.
            audio_history_maxlen: Capacity of the bounded rolling audio buffer
                used to replay recent audio after a reconnect.
            strict_lifecycle: If ``True``, raise on illegal lifecycle
                transitions instead of suppressing + diagnosing them.
        """
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=audio_queue_maxsize)
        self._buffer = _CoalescingBuffer(capacity=event_buffer_capacity)
        self._guard = _LifecycleGuard(strict=strict_lifecycle)
        self._mode: Literal["feed", "manual"] | None = None
        self._ended = False
        self._done_timeout = done_timeout
        self._max_idle = max_idle
        self._max_session_seconds = max_session_seconds
        self._feed_task: asyncio.Task[None] | None = None
        self._producer_task: asyncio.Task[None] | None = None
        self._reducer = StreamReducer()
        # Reconnect scaffolding (spec ST.6.3 / D10.7).
        self._audio_history: deque[bytes] = deque(maxlen=audio_history_maxlen)
        self._replayable = False
        self._history_overflowed = False
        self._pending_reconnects: list[TranscriptionEvent] = []
        self._monotonic = time.monotonic

    # ----- author hooks ---------------------------------------------------- #
    async def _open(self) -> None:
        """Open engine resources bound to the event loop (override as needed)."""

    async def _close(self) -> None:
        """Tear down engine resources (override as needed)."""

    @abstractmethod
    def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        """Yield raw transcription events for the session.

        Implementations consume fed audio via :meth:`audio_chunks` and yield
        events. Returning normally ends the stream (the base appends ``done``).

        Returns:
            An async iterator of events.
        """
        raise NotImplementedError

    async def audio_chunks(self) -> AsyncIterator[bytes]:
        """Async-iterate fed audio chunks until the input ends.

        Each yielded chunk is also retained in the bounded rolling audio buffer
        for possible replay after a reconnect.

        Yields:
            Raw audio chunks in the session's declared format.
        """
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                return
            # Appending to a full bounded deque evicts the oldest chunk: that
            # chunk can no longer be replayed. For a non-replayable (live)
            # source an eviction means a later reconnect may have lost content.
            if (
                self._audio_history.maxlen is not None
                and len(self._audio_history) >= self._audio_history.maxlen
            ):
                self._history_overflowed = True
            self._audio_history.append(chunk)
            yield chunk

    # ----- reconnect scaffolding ------------------------------------------- #
    @property
    def replayable(self) -> bool:
        """Whether the audio source can be re-read after a reconnect.

        Returns:
            ``True`` if the fed source is replayable (a finite, re-iterable
            collection), ``False`` for live / one-shot sources.
        """
        return self._replayable

    def replay_buffer(self) -> list[bytes]:
        """Return the bounded rolling buffer of recent audio for re-feeding.

        Adapters call this on reconnect to re-send the most recent audio to a
        freshly re-established engine connection.

        Returns:
            The retained recent audio chunks, oldest first.
        """
        return list(self._audio_history)

    def note_reconnect(self, gap_start: float | None = None, gap_end: float | None = None) -> None:
        """Record that an internal reconnect bridged a gap (adapter-driven).

        The base queues a ``progress(reconnect=True, gap_start, gap_end)`` event
        to be emitted in order with produced events, and -- for a non-replayable
        source whose rolling buffer overflowed during the session (audio was
        evicted and thus cannot be replayed) -- a trailing
        ``error(code="content_lost", recoverable=False)`` (spec ST.6.3).

        ``segment_id`` / timestamps / detected language continuity across the
        reconnect is the adapter's responsibility (the base never rewrites them).

        Args:
            gap_start: Start time (seconds) of the lossy gap, if known.
            gap_end: End time (seconds) of the lossy gap, if known.
        """
        self._pending_reconnects.append(
            TranscriptionEvent.progress(reconnect=True, gap_start=gap_start, gap_end=gap_end)
        )
        if not self._replayable and self._history_overflowed:
            self._pending_reconnects.append(
                TranscriptionEvent.make_error(
                    code="content_lost",
                    recoverable=False,
                    gap_start=gap_start,
                    gap_end=gap_end,
                )
            )

    # ----- input ownership ------------------------------------------------- #
    def _claim_mode(self, mode: Literal["feed", "manual"]) -> None:
        """Atomically claim single input ownership for the first input call.

        Args:
            mode: The mode being claimed by this call.

        Raises:
            StreamClosedError: If the other mode was already claimed.
        """
        if self._mode is None:
            self._mode = mode
            return
        if self._mode != mode:
            other = "manual" if mode == "feed" else "feed"
            raise StreamClosedError(f"{other} input already in use; cannot mix with {mode}.")

    def feed(self, source: Iterable[bytes] | AsyncIterator[bytes]) -> None:
        """Feed audio from a managed source (mutually exclusive with manual).

        A non-async, non-iterator collection (``list`` / ``tuple`` / ``bytes``
        sequence) is classified as **replayable** for reconnect purposes; an
        async iterator or a one-shot generator/iterator is **non-replayable**.

        Args:
            source: A sync or async iterable of audio chunks.

        Raises:
            StreamClosedError: If manual input or a prior feed was already used.
        """
        self._claim_mode("feed")
        if self._feed_task is not None:
            raise StreamClosedError("feed() already called once.")
        # Replayable iff a re-iterable collection (not a one-shot iterator and
        # not an async source). list/tuple/bytes/bytearray qualify.
        self._replayable = isinstance(source, (list, tuple, bytes, bytearray))
        self._feed_task = asyncio.ensure_future(self._drain_source(source))

    async def _drain_source(self, source: Iterable[bytes] | AsyncIterator[bytes]) -> None:
        """Drain a fed source into the audio queue, ending on exhaustion.

        Args:
            source: The audio source.
        """
        try:
            if isinstance(source, AsyncIterator):
                async for chunk in source:
                    await self._put_audio(chunk)
            else:
                for chunk in source:
                    await self._put_audio(chunk)
        finally:
            await self._audio_queue.put(None)
            self._ended = True

    async def _put_audio(self, chunk: bytes) -> None:
        """Enqueue one audio chunk (bounded queue exerts backpressure).

        Args:
            chunk: The audio chunk.
        """
        await self._audio_queue.put(chunk)

    async def send_audio(self, chunk: bytes) -> None:
        """Manually send one audio chunk (mutually exclusive with ``feed``).

        Manual sources are always treated as **non-replayable** (live input).

        Args:
            chunk: The audio chunk.

        Raises:
            StreamClosedError: If ``feed`` was used or the input was ended.
        """
        if self._ended:
            raise StreamClosedError("Cannot send_audio after end_audio().")
        self._claim_mode("manual")
        await self._put_audio(chunk)

    async def end_audio(self) -> None:
        """Mark the end of manual audio input (idempotent in manual mode).

        Claims manual ownership if this is the first input call, so a later
        ``feed`` is correctly rejected as mixing (spec ST.3.3).

        Raises:
            StreamClosedError: If ``feed`` was used.
        """
        self._claim_mode("manual")
        if self._ended:
            return
        self._ended = True
        await self._audio_queue.put(None)

    # ----- iteration ------------------------------------------------------- #
    async def __aenter__(self) -> TranscriptionSession:
        """Open resources and start the producer.

        Returns:
            The session.
        """
        await self._open()
        self._producer_task = asyncio.ensure_future(self._run_producer())
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Tear down the producer, feed task, and engine resources.

        Cancels the producer and feed tasks and AWAITS them (so no coroutine is
        still touching engine state when :meth:`_close` runs), then closes.
        """
        tasks = [t for t in (self._producer_task, self._feed_task) if t is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._close()

    def _drain_pending_reconnects(self) -> None:
        """Flush any queued reconnect ``progress`` / ``content_lost`` events.

        Raises:
            EventBufferOverflow: Propagated from the buffer (handled by caller).
        """
        if not self._pending_reconnects:
            return
        pending = self._pending_reconnects
        self._pending_reconnects = []
        for ev in pending:
            self._reducer.add(ev)
            self._buffer.put(ev)

    async def _run_producer(self) -> None:
        """Drive ``_produce``, appending a terminal ``done`` or ``error``.

        Enforces lifecycle invariants (suppressing illegal transitions), flushes
        adapter-driven reconnect events in order, and converts a send-side
        buffer overflow into a terminal ``backpressure`` error.
        """
        try:
            async for event in self._produce():
                self._drain_pending_reconnects()
                admitted = self._guard.admit(event)
                if admitted is None:
                    continue  # illegal transition: suppressed (diagnosed).
                self._reducer.add(admitted)
                self._buffer.put(admitted)
                if admitted.is_terminal:
                    return
            self._drain_pending_reconnects()
            # done MUST never be dropped: bypass the bound so it always lands.
            self._buffer.put_forced(TranscriptionEvent.done())
        except asyncio.CancelledError:  # pragma: no cover - teardown path
            raise
        except EventBufferOverflow:
            self._force_error(
                "backpressure",
                "Send-side event buffer overflowed; consumer too slow.",
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as an error event
            self._force_error("engine_error", str(exc))
        finally:
            self._buffer.close()

    def _force_error(self, code: str, detail: str) -> None:
        """Append a terminal error bypassing the buffer bound (drop-proof path).

        Used when the normal bounded buffer cannot accept more events (overflow)
        or the producer crashed: ``final`` / ``done`` / ``error`` MUST never be
        silently lost (spec ST.6.4), so the terminal error bypasses the capacity
        check and lands in the same buffer the iterator is already awaiting.

        Args:
            code: The error code.
            detail: Human-readable detail stored under ``extra["detail"]``.
        """
        self._buffer.put_forced(
            TranscriptionEvent.make_error(code=code, recoverable=False, extra={"detail": detail})
        )

    def __aiter__(self) -> AsyncIterator[TranscriptionEvent]:
        """Return the event async iterator.

        Returns:
            ``self`` as an async iterator.
        """
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[TranscriptionEvent]:
        """Yield events with per-event, idle, and wall-clock termination.

        Termination is guaranteed by three independent deadlines (spec ST.6.1):

        * ``done_timeout`` -- max gap between consecutive events of any kind.
        * ``max_idle`` -- max time without a *content* event (heartbeats do not
          reset it), so a heartbeat-only stuck engine still terminates.
        * ``max_session_seconds`` -- absolute wall-clock cap.

        Yields:
            Events until a terminal event or a deadline fires.
        """
        start = self._monotonic()
        last_content = start
        while True:
            now = self._monotonic()
            timeout = self._done_timeout
            if self._max_idle is not None:
                timeout = min(timeout, max(0.0, self._max_idle - (now - last_content)))
            if self._max_session_seconds is not None:
                remaining = self._max_session_seconds - (now - start)
                if remaining <= 0:
                    yield TranscriptionEvent.make_error(code="session_timeout", recoverable=False)
                    return
                timeout = min(timeout, remaining)
            try:
                event = await asyncio.wait_for(self._buffer.get(), timeout=timeout)
            except asyncio.TimeoutError:
                now = self._monotonic()
                if self._max_idle is not None and (now - last_content) >= self._max_idle:
                    yield TranscriptionEvent.make_error(code="stream_stalled", recoverable=False)
                    return
                if (
                    self._max_session_seconds is not None
                    and (now - start) >= self._max_session_seconds
                ):
                    yield TranscriptionEvent.make_error(code="session_timeout", recoverable=False)
                    return
                yield TranscriptionEvent.make_error(code="done_timeout", recoverable=False)
                return
            if event is None:
                return
            if event.is_content:
                last_content = self._monotonic()
            yield event
            if event.is_terminal:
                return

    def diagnostics(self) -> list[Diagnostic]:
        """Return lifecycle-suppression diagnostics accumulated so far.

        Returns:
            The structured diagnostics for any suppressed illegal transitions
            or clamped ``stable_until`` values.
        """
        return list(self._guard.diagnostics)

    @property
    def done_timeout(self) -> float:
        """The configured per-event done-timeout in seconds.

        Returns:
            Max seconds awaited between consecutive events before a
            ``done_timeout`` error is synthesized.
        """
        return self._done_timeout

    def result(self) -> TranscriptionResult:
        """Reduce the session so far into a transcription result.

        Returns:
            The reduced result.
        """
        return self._reducer.result()


class SyncSession:
    """Synchronous bridge over an async :class:`TranscriptionSession`.

    Runs a single background event loop in a dedicated thread (owned by this
    object and torn down on close), so applications can drive an async adapter
    synchronously and authors only ever write async code (spec ST.6.5).

    Lifecycle submits (``__enter__`` / input calls / ``__exit__``) carry a
    timeout: a hanging adapter ``_open`` / ``_close`` can never deadlock the
    calling thread, and the background loop + thread are always torn down even on
    timeout (spec ST.6.5: from an external thread, no deadlock, no leak).
    """

    def __init__(
        self,
        session: TranscriptionSession,
        *,
        submit_timeout: float | None = 30.0,
    ) -> None:
        """Wrap an async session.

        Args:
            session: The async session to drive.
            submit_timeout: Seconds to wait for a lifecycle submit (enter /
                feed / send / end / exit) before raising ``TimeoutError`` and
                tearing the loop down. ``None`` waits forever (not recommended).
        """
        self._session = session
        self._submit_timeout = submit_timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._aiter: AsyncIterator[TranscriptionEvent] | None = None
        self._closed = False

    def _run_loop(self) -> None:
        """Run the owned event loop until stopped."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _shutdown(self) -> None:
        """Cancel pending tasks, stop the loop, join the thread, and close it.

        Cancelling + awaiting outstanding tasks (e.g. a hung ``__aenter__``)
        before stopping avoids "Task was destroyed but it is pending" warnings,
        and closing the loop avoids the ``BaseEventLoop.__del__`` resource
        warning — both of which would otherwise surface (and fail ``-W error``).
        """
        if self._closed:
            return
        self._closed = True
        # Best-effort: cancel and await all outstanding tasks on the loop thread
        # so nothing is destroyed while pending. A truly blocking (non-awaiting)
        # adapter can't be cancelled cooperatively; the join timeout is the
        # backstop for that pathological case.
        try:
            future = asyncio.run_coroutine_threadsafe(_cancel_all_tasks(), self._loop)
            future.result(timeout=5.0)
        except Exception:  # noqa: BLE001 - teardown is best-effort  # pragma: no cover
            # Defensive: the cancel-coroutine submit only fails if the owned loop
            # is already torn down, which the _closed guard above prevents on the
            # normal path. Kept so a pathological double-teardown cannot escape.
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        # ``is_alive()`` is False on every non-pathological path (a cooperative
        # adapter always returns within the 5s join); a truly blocking adapter
        # that never yields would leave the thread alive and the loop unclosed.
        if not self._thread.is_alive():  # pragma: no branch
            self._loop.close()

    def _submit(self, coro: Any, *, timeout: float | None) -> Any:
        """Run a coroutine on the owned loop and wait, bounded by ``timeout``.

        Args:
            coro: The coroutine to run.
            timeout: Seconds to wait; ``None`` waits forever.

        Returns:
            The coroutine's result.

        Raises:
            TimeoutError: If the coroutine does not complete within ``timeout``.
                The background loop + thread are torn down before raising.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            # On Python 3.10 concurrent.futures.TimeoutError is distinct from the
            # builtin TimeoutError; re-raise as the builtin for a stable API.
            future.cancel()
            self._shutdown()
            raise TimeoutError(
                f"SyncSession lifecycle call timed out after {timeout}s; "
                "the async adapter hung (spec ST.6.5 no-hang contract)."
            ) from exc

    def __enter__(self) -> SyncSession:
        """Enter the async session's context.

        Returns:
            The sync session.

        Raises:
            TimeoutError: If the adapter ``_open`` hangs past ``submit_timeout``.
        """
        self._submit(self._session.__aenter__(), timeout=self._submit_timeout)
        self._aiter = self._session.__aiter__()
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the async context and stop the owned loop (never leaks)."""
        try:
            self._submit(self._session.__aexit__(*exc), timeout=self._submit_timeout)
        finally:
            self._shutdown()

    def feed(self, source: Iterable[bytes]) -> None:
        """Feed audio from a managed source.

        Args:
            source: A sync iterable of audio chunks.
        """

        async def _do_feed() -> None:
            self._session.feed(source)

        self._submit(_do_feed(), timeout=self._submit_timeout)

    def send_audio(self, chunk: bytes) -> None:
        """Manually send one audio chunk.

        Args:
            chunk: The audio chunk.
        """
        self._submit(self._session.send_audio(chunk), timeout=self._submit_timeout)

    def end_audio(self) -> None:
        """Mark the end of manual audio input."""
        self._submit(self._session.end_audio(), timeout=self._submit_timeout)

    def __iter__(self) -> Iterator[TranscriptionEvent]:
        """Iterate events synchronously.

        Event-pump submits use ``done_timeout`` as their bound (plus slack) so a
        stuck engine surfaces as a terminal event from the iterator rather than
        hanging the caller.

        Yields:
            Events from the underlying async session.
        """
        assert self._aiter is not None
        # Event waits are bounded by the session's own deadlines, which always
        # synthesize a terminal event; add slack so the submit never trips first.
        pump_timeout: float | None
        if self._submit_timeout is None:
            pump_timeout = None
        else:
            pump_timeout = max(self._submit_timeout, self._session.done_timeout) + 5.0
        while True:
            try:
                yield self._submit(self._aiter.__anext__(), timeout=pump_timeout)
            except StopAsyncIteration:
                return

    def result(self) -> TranscriptionResult:
        """Reduce the session so far into a transcription result.

        Returns:
            The reduced result.
        """
        return self._session.result()


__all__ = [
    "DEFAULT_AUDIO_HISTORY_MAXLEN",
    "DEFAULT_AUDIO_QUEUE_MAXSIZE",
    "DEFAULT_DONE_TIMEOUT",
    "DEFAULT_EVENT_BUFFER_CAPACITY",
    "DEFAULT_MAX_IDLE",
    "DEFAULT_MAX_SESSION_SECONDS",
    "EventType",
    "StreamReducer",
    "SyncSession",
    "TranscriptionEvent",
    "TranscriptionSession",
    "reduce_event",
    "validate_stable_until",
]
