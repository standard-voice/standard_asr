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
import logging
import threading
import time
import unicodedata
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exceptions import StreamClosedError
from .results import Diagnostic, Segment, TranscriptionResult, Word

LOGGER = logging.getLogger(__name__)

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

_INPUT_SOURCE_ERROR_DETAIL = "Audio input source failed during streaming."


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

    @model_validator(mode="after")
    def _check_invariants(self) -> TranscriptionEvent:
        """Reject structurally illegal events at construction (spec ST.5).

        A content event with no segment it can address, an error with no code, or
        a supersede whose retired/replacement ids overlap (or that retires
        nothing) is malformed -- a wrong or unattributable transcript is the
        cardinal sin, so the event model refuses to represent one. Sequence-level
        invariants (monotonic ``stable_until`` / ``audio_processed_until``,
        frozen-prefix immutability, illegal lifecycle transitions) are enforced
        across events by :class:`_LifecycleGuard`, not here.

        Returns:
            The validated event.

        Raises:
            ValueError: If the event is structurally illegal for its type.
        """
        if self.type in ("partial", "final"):
            if self.segment_id is None or self.text is None:
                raise ValueError(f"{self.type} event MUST carry both segment_id and text.")
        elif self.type == "supersede":
            if not self.old_ids:
                raise ValueError("supersede event MUST retire at least one segment (old_ids).")
            if set(self.old_ids) & set(self.new_ids):
                raise ValueError("supersede old_ids and new_ids MUST be disjoint.")
        elif self.type == "error" and self.code is None:
            raise ValueError("error event MUST carry a code.")
        return self

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

        Lineage is **set-to-set**: ``old_ids``/``new_ids`` express the
        *cardinality* of the re-segmentation (which ids retire, which appear)
        but not a per-old->per-new mapping. On a merge+split (many->many) a UI
        cannot tell which specific old segment a given new segment descends
        from. This is a documented v1 limitation (X-ST-6); the spec does not
        require a pairwise mapping, and the frozen-prefix-preservation invariant
        (§ST 5.2) is enforced over the concatenated prefixes, not per pair.
        Per-pair edit-ops/diffs are the deferred §10 direction (additive later).

        Args:
            old_ids: The retired segment ids, in reading (time) order.
            new_ids: The replacement segment ids, in reading (time) order
                (must be disjoint from ``old_ids``).
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


class _InputSourceError(Exception):
    """Internal signal: the fed audio source failed mid-stream."""


class _InputSourceFailure:
    """Queue marker preserving audio ordering before an input-source failure."""


_INPUT_SOURCE_FAILURE = _InputSourceFailure()


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
    buffer. Only a NEW partial slot (a not-yet-pending segment) may push past
    ``capacity`` and raise :class:`EventBufferOverflow`, which the producer turns
    into a terminal ``backpressure`` error. ``final`` / ``supersede`` (like
    ``done`` / ``error`` via :meth:`put_forced`) bypass the bound and are
    appended drop-proof: they MUST never be dropped (spec ST.6.4) and are
    bounded per segment, so only distinct-segment *partials* trigger
    backpressure.
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

        ``final`` / ``supersede`` are appended drop-proof (bypassing the bound),
        so only a NEW partial slot for a not-yet-pending segment can overflow.

        Args:
            event: The event to enqueue.

        Raises:
            EventBufferOverflow: If the buffer is at capacity and the event is a
                NEW partial for a not-yet-pending segment (growing the buffer).
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
            # final / supersede MUST never be dropped (spec ST.6.4): append
            # drop-proof, bypassing the capacity bound. Only a NEW partial slot
            # (which GROWS the buffer for a not-yet-pending segment) may overflow
            # -- finals/supersedes are bounded per segment (each invalidates its
            # own pending partial above), so bypassing the bound is safe. The
            # residual: a pathological flood of distinct-segment finals can grow
            # memory unboundedly -- accepted, because the spec forbids dropping
            # them; only distinct-segment *partials* trigger backpressure.
            self._items.append(_Slot(event))
            self._live_count += 1
            self._event.set()
            return
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


class _SupersedeObligation:
    """A pending frozen-prefix-preservation obligation for one supersede group.

    A ``supersede`` MUST preserve the concatenated frozen text of the retired
    segments across the replacement (spec ST.5.2). This records, for one
    ``new_ids`` group, the concatenated frozen prefix of the retired old
    segments (``f_old``, in ``old_ids`` order) and the running concatenated
    frozen prefix of the new segments (in ``new_ids`` order), so the guard can
    eagerly reject the cardinal-sin direction (a new segment rewriting text the
    user already saw frozen).
    """

    __slots__ = ("f_old", "frozen", "new_ids")

    def __init__(self, f_old: str, new_ids: list[str]) -> None:
        """Initialize the obligation.

        Args:
            f_old: Concatenated frozen prefix of the retired segments.
            new_ids: The replacement segment ids, in reading (temporal) order.
        """
        self.f_old = f_old
        self.new_ids = new_ids
        #: Per-new-id current frozen prefix, accumulated as each new segment
        #: freezes more text.
        self.frozen: dict[str, str] = {}

    def f_new(self) -> str:
        """Return the replacement's *contiguous* frozen prefix.

        A frozen prefix is contiguous from position 0, so the replacement's
        frozen prefix is the concatenation of the new segments' frozen prefixes
        in ``new_ids`` order **only up to the first new segment that has not yet
        frozen any text**. The streaming protocol does not forbid freezing the
        new segments of a split out of order; a later ``new_id`` that freezes
        before an earlier one does NOT yet contribute to position 0, so its
        text must not be counted until the gap to its left is filled (otherwise
        it would be misplaced and falsely flagged as rewriting ``f_old``).

        Returns:
            The new segments' frozen prefixes joined in ``new_ids`` order,
            truncated at the first not-yet-frozen (missing or empty) new id.
        """
        parts: list[str] = []
        for nid in self.new_ids:
            frozen = self.frozen.get(nid, "")
            if not frozen:
                break
            parts.append(frozen)
        return "".join(parts)


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
        self._frozen_text: dict[str, str] = {}
        self._audio_cursor: float = 0.0
        #: Maps each ``new_id`` of an active supersede group to its shared
        #: frozen-prefix-preservation obligation (spec ST.5.2).
        self._supersede_obligations: dict[str, _SupersedeObligation] = {}
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
        event = self._clamp_audio_cursor(event)
        sid = event.segment_id
        if event.type == "supersede":
            for old in event.old_ids:
                if old not in self._state:
                    self._reject(
                        "supersede_unknown_old_id",
                        f"supersede old_ids contains never-announced segment "
                        f"{old!r}; suppressed (spec ST.5.2: old_ids MUST have "
                        "received at least one partial/final).",
                    )
                    return None
            for old in event.old_ids:
                if self._state.get(old) == "closed":
                    self._reject(
                        "lifecycle_closed_superseded",
                        f"supersede old_ids contains closed segment {old!r}; "
                        "suppressed (spec ST.5.3: closed MUST NOT be superseded).",
                    )
                    return None
            for new in event.new_ids:
                if new in self._state:
                    self._reject(
                        "supersede_reintroduces_segment",
                        f"supersede new_ids reintroduces already-known segment "
                        f"{new!r} (state {self._state[new]!r}); suppressed "
                        "(spec ST.5.2: a new_id MUST be fresh).",
                    )
                    return None
            # Concatenate the retired segments' frozen prefixes, in old_ids
            # (reading) order: this is the text the user already saw frozen and
            # which the replacement MUST preserve (spec ST.5.2).
            f_old = "".join(self._frozen_text.get(old, "") for old in event.old_ids)
            if not event.new_ids and f_old:
                # Pure deletion (empty new_ids) cannot preserve any frozen text;
                # it MUST NOT silently destroy a prefix the user saw frozen.
                self._reject(
                    "supersede_deletes_frozen_text",
                    "supersede with empty new_ids would delete the frozen prefix "
                    f"of {event.old_ids!r}; suppressed (spec ST.5.2: frozen text "
                    "MUST be preserved -- pure deletion is allowed only for "
                    "segments with no frozen prefix).",
                )
                return None
            for old in event.old_ids:
                self._state[old] = "superseded"
            for new in event.new_ids:
                self._state.setdefault(new, "open")
            if f_old:
                obligation = _SupersedeObligation(f_old, list(event.new_ids))
                for new in event.new_ids:
                    self._supersede_obligations[new] = obligation
            return event

        if event.type in ("partial", "final") and sid is not None:
            is_closed_final = event.type == "final" and event.finality == "closed"
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
            if event.type == "final" and state == "final" and event.finality != "closed":
                # From state final the only legal transitions are supersede or a
                # closed event; a plain final re-freezing/rewriting the segment
                # is illegal (spec ST.5.1).
                self._reject(
                    "lifecycle_final_after_final",
                    f"non-closed final for segment {sid!r} already in state final; "
                    "suppressed (spec ST.5.1: from final only supersede or a "
                    "closed event is legal).",
                )
                return None
            # ``closed`` is the terminal post-processing correction and may
            # replace previously frozen text in place.
            if not is_closed_final and self._frozen_prefix_rewritten(event, sid):
                self._reject(
                    "frozen_prefix_rewritten",
                    f"segment {sid!r} rewrote its already-frozen prefix "
                    "(text[:stable_until] changed); suppressed (spec ST.4.2: the "
                    "frozen prefix is immutable).",
                )
                return None
            event = self._clamp_stable_until(event, sid)
            su = event.stable_until or 0
            if su > 0 and event.text is not None:
                self._frozen_text[sid] = event.text[:su]
                if not is_closed_final and not self._supersede_preserves_frozen(sid):
                    self._reject(
                        "frozen_prefix_rewritten_supersede",
                        f"segment {sid!r} froze text that rewrites the frozen "
                        "prefix of the segment(s) it superseded; suppressed "
                        "(spec ST.5.2: supersede MUST preserve frozen text).",
                    )
                    # Undo the freeze so subsequent state is not corrupted.
                    del self._frozen_text[sid]
                    return None
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

    def _frozen_prefix_rewritten(self, event: TranscriptionEvent, sid: str) -> bool:
        """Return whether ``event`` rewrites segment ``sid``'s frozen prefix.

        The frozen prefix (``text[:stable_until]`` at the last accepted frontier)
        is immutable: an engine may extend the text but MUST NOT alter what it
        has already frozen (spec ST.4.2). Returns ``False`` for the first event
        of a segment or when nothing is frozen yet.

        Args:
            event: The incoming partial/final event.
            sid: The segment id.

        Returns:
            ``True`` if the previously-frozen prefix would change.
        """
        prior_su = self._stable_until.get(sid, 0)
        if prior_su <= 0 or event.text is None:
            return False
        return event.text[:prior_su] != self._frozen_text.get(sid, "")

    def _supersede_preserves_frozen(self, sid: str) -> bool:
        """Return whether ``sid``'s freeze keeps a supersede obligation intact.

        When ``sid`` is one of the ``new_ids`` of an active supersede group, the
        concatenated frozen prefix of the new segments (``F_new``, in ``new_ids``
        order) MUST agree with the retired segments' concatenated frozen prefix
        (``F_old``) on their common prefix -- neither may rewrite the other
        (spec ST.5.2). Only the *contradiction* (divergence on the common
        prefix) is checked, and eagerly: it is the cardinal-sin direction. The
        opposite case (``F_new`` still strictly shorter than ``F_old``) is the
        safe, conservative direction (the new segmentation has simply not yet
        re-frozen everything) and is permitted to remain pending.

        Args:
            sid: The segment id that just froze (more) text.

        Returns:
            ``True`` if no obligation is violated (including when ``sid`` is not
            part of any supersede group).
        """
        obligation = self._supersede_obligations.get(sid)
        if obligation is None:
            return True
        obligation.frozen[sid] = self._frozen_text.get(sid, "")
        f_new = obligation.f_new()
        common = min(len(f_new), len(obligation.f_old))
        return f_new[:common] == obligation.f_old[:common]

    def _clamp_audio_cursor(self, event: TranscriptionEvent) -> TranscriptionEvent:
        """Clamp a decreasing ``audio_processed_until`` cursor (spec ST.4.1).

        The audio-time cursor is monotonic across the whole session (it never
        moves backwards), independent of segment. A decrease is clamped to the
        prior value with a diagnostic (or raises in strict mode).

        Args:
            event: Any event (the cursor may appear on content or progress).

        Returns:
            The event, with ``audio_processed_until`` clamped if it decreased.
        """
        cursor = event.audio_processed_until
        if cursor is None:
            return event
        if cursor < self._audio_cursor:
            self._reject(
                "audio_cursor_decreased",
                f"audio_processed_until decreased {cursor} -> clamped to "
                f"{self._audio_cursor} (spec ST.4.1: the cursor is monotonic).",
            )
            event = event.model_copy(update={"audio_processed_until": self._audio_cursor})
        else:
            self._audio_cursor = cursor
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
      iff the adapter passed ``content_lost=True`` (its own determination that
      the reconnect + replay could not cover the gap), a trailing
      ``error(code="content_lost", recoverable=False)``.
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
        self._audio_queue: asyncio.Queue[bytes | _InputSourceFailure | None] = asyncio.Queue(
            maxsize=audio_queue_maxsize
        )
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
        # The FULL fed source, retained only for a truly replayable (list/tuple)
        # source so replay_buffer() can offer loss-free replay even past the ring
        # length. None for non-replayable/live sources (which use the ring).
        self._replay_source: tuple[bytes, ...] | None = None
        self._pending_reconnects: list[TranscriptionEvent] = []
        self._monotonic = time.monotonic
        # Standard-layer diagnostics (parameter gating / language resolution)
        # attached by the base ``start_transcription`` template before the
        # session is handed to the application, so they surface through the
        # session's existing ``diagnostics()`` channel.
        self._initial_diagnostics: list[Diagnostic] = []

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
            if isinstance(chunk, _InputSourceFailure):
                raise _InputSourceError(_INPUT_SOURCE_ERROR_DETAIL)
            # The bounded deque drops its oldest chunk when full: it is only the
            # most-recent-audio replay window for replay_buffer(). Whether a
            # reconnect gap actually lost unreplayable audio is the adapter's
            # determination (passed to note_reconnect(content_lost=...)), not an
            # eviction count -- a live ring is always evicting, so eviction is
            # the wrong content-loss signal.
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
        """Return audio for re-feeding to a freshly re-established connection.

        Adapters call this on reconnect to re-send audio. For a **replayable**
        (list / tuple) source the COMPLETE source is returned -- replayability
        promises loss-free replay, so it must not be silently truncated to the
        rolling ring. For a **non-replayable** / live source only the bounded
        rolling window of recent chunks is available (older audio was evicted).

        Returns:
            The audio chunks to replay, oldest first: the full source if
            replayable, otherwise the bounded rolling window.
        """
        if self._replay_source is not None:
            return list(self._replay_source)
        return list(self._audio_history)

    def note_reconnect(
        self,
        gap_start: float | None = None,
        gap_end: float | None = None,
        *,
        content_lost: bool = False,
    ) -> None:
        """Record that an internal reconnect bridged a gap (adapter-driven).

        The base ALWAYS queues a ``progress(reconnect=True, gap_start, gap_end)``
        event to be emitted in order with produced events. It queues a trailing
        ``error(code="content_lost", recoverable=False, gap_start, gap_end)`` --
        IMMEDIATELY following the progress (spec ST.6.3) -- IFF the adapter passes
        ``content_lost=True``.

        Content loss is an **explicit adapter determination**, not something the
        base infers from rolling-buffer eviction: a live ring is always evicting,
        so eviction is the wrong signal (it would falsely claim permanent loss on
        every long live session). The adapter -- which alone knows whether its
        reconnect + :meth:`replay_buffer` replay actually covered the gap -- sets
        ``content_lost=True`` only when audio the engine had not yet processed
        could not be replayed and is therefore truly lost. This mirrors the
        existing contract where ``segment_id`` / timestamps / detected language
        continuity across the reconnect is likewise the adapter's responsibility
        (the base never rewrites them).

        Args:
            gap_start: Start time (seconds) of the lossy gap, if known.
            gap_end: End time (seconds) of the lossy gap, if known.
            content_lost: ``True`` if the reconnect could not cover the gap and
                unreplayable audio was permanently lost; queues a terminal
                ``content_lost`` error after the progress (spec ST.6.3).
        """
        self._pending_reconnects.append(
            TranscriptionEvent.progress(reconnect=True, gap_start=gap_start, gap_end=gap_end)
        )
        if content_lost:
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

    def feed(self, source: Iterable[bytes] | AsyncIterator[bytes] | bytes | bytearray) -> None:
        """Feed audio from a managed source (mutually exclusive with manual).

        A bare ``bytes`` / ``bytearray`` is treated as a **single** audio chunk
        (not an iterable of chunks -- iterating it would yield ``int`` byte
        values). A non-async, re-iterable collection (``list`` / ``tuple``, or a
        wrapped bytes-like) is classified as **replayable** for reconnect
        purposes; an async iterator or a one-shot generator/iterator is
        **non-replayable**.

        Args:
            source: A sync or async iterable of audio chunks, or a single
                ``bytes`` / ``bytearray`` chunk.

        Raises:
            StreamClosedError: If manual input or a prior feed was already used.
        """
        self._claim_mode("feed")
        if self._feed_task is not None:
            raise StreamClosedError("feed() already called once.")
        if isinstance(source, (bytes, bytearray)):
            # A bare bytes-like is ONE chunk: wrap it so draining yields the
            # whole chunk rather than iterating it into individual int values.
            source = [bytes(source)]
        # Replayable iff a re-iterable collection (not a one-shot iterator and
        # not an async source); list/tuple (incl. a wrapped bytes-like) qualify.
        self._replayable = isinstance(source, (list, tuple))
        if self._replayable:
            # Retain the FULL source so replay_buffer() can offer loss-free
            # replay even when the source is longer than the rolling ring --
            # "replayable" must mean the whole source, not merely the tail.
            self._replay_source = tuple(source)  # type: ignore[arg-type]
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
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Fed audio source failed during streaming.")
            await self._audio_queue.put(_INPUT_SOURCE_FAILURE)
            self._ended = True
        else:
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
        # Claim manual ownership FIRST so mixing with an active feed always
        # raises the deterministic mixing error -- otherwise the feed task
        # setting _ended on exhaustion would race the _ended check below and
        # sometimes surface the "after end_audio" message instead (spec ST.3.3).
        self._claim_mode("manual")
        if self._ended:
            raise StreamClosedError("Cannot send_audio after end_audio().")
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

        Reconnect events are appended drop-proof (bypassing the capacity bound,
        like ``final`` / terminal events) so a ``progress(reconnect=True)`` can
        neither be dropped nor split from its immediately-following
        ``content_lost`` error under backpressure (spec ST.6.3 requires the
        ``content_lost`` to IMMEDIATELY follow the ``progress``; spec ST.6.4
        forbids dropping ``error``). They are few and bounded per reconnect, so
        bypassing the bound is safe.
        """
        if not self._pending_reconnects:
            return
        pending = self._pending_reconnects
        self._pending_reconnects = []
        for ev in pending:
            self._reducer.add(ev)
            self._buffer.put_forced(ev)

    async def _run_producer(self) -> None:
        """Drive ``_produce``, appending a terminal ``done`` or ``error``.

        Enforces lifecycle invariants (suppressing illegal transitions), flushes
        adapter-driven reconnect events in order, and converts a send-side
        buffer overflow into a terminal ``backpressure`` error.

        Pending adapter reconnect events are drained drop-proof BEFORE every
        terminal append (so a queued ``progress`` + ``content_lost`` precedes,
        and is delivered ahead of, the terminal) and once more in the ``finally``
        as a guard, so no exit path -- normal, early-terminal, exception, or
        overflow -- can lose them (spec ST.6.3 / ST.6.4).
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
            self._drain_pending_reconnects()
            self._force_error(
                "backpressure",
                "Send-side event buffer overflowed; consumer too slow.",
            )
        except _InputSourceError:
            self._drain_pending_reconnects()
            self._force_error("input_source_error", _INPUT_SOURCE_ERROR_DETAIL)
        except Exception as exc:  # noqa: BLE001 - surfaced as an error event
            self._drain_pending_reconnects()
            self._force_error("engine_error", str(exc))
        finally:
            # Guard: any path that skipped the drains above (e.g. an early
            # terminal return while reconnects were still queued) still flushes
            # them before the buffer closes -- queued reconnect events MUST never
            # be silently lost (spec ST.6.4).
            self._drain_pending_reconnects()
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

    def _attach_initial_diagnostics(self, diagnostics: list[Diagnostic]) -> None:
        """Record standard-layer diagnostics produced before the session ran.

        Called once by the base :meth:`~standard_asr.asr_interface.EngineBase.\
start_transcription` template with the parameter-gating and language-axis
        diagnostics, so they surface through :meth:`diagnostics` alongside the
        runtime's lifecycle-suppression diagnostics.

        Args:
            diagnostics: The gating / language diagnostics to attach.

        Returns:
            None.
        """
        self._initial_diagnostics = list(diagnostics)

    def diagnostics(self) -> list[Diagnostic]:
        """Return the diagnostics accumulated for this session so far.

        Returns:
            The standard-layer parameter-gating / language diagnostics attached
            at session establishment, followed by the runtime's
            lifecycle-suppression diagnostics (suppressed illegal transitions or
            clamped ``stable_until`` values).
        """
        return [*self._initial_diagnostics, *self._guard.diagnostics]

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

    def feed(self, source: Iterable[bytes] | bytes | bytearray) -> None:
        """Feed audio from a managed source.

        Args:
            source: A sync iterable of audio chunks, or a single ``bytes`` /
                ``bytearray`` chunk.
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
