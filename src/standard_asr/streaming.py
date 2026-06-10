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
from collections.abc import AsyncIterable, AsyncIterator, Coroutine, Iterable, Iterator
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .exceptions import InvalidSessionUseError, StreamClosedError
from .results import Diagnostic, Segment, TranscriptionResult, Word

LOGGER = logging.getLogger(__name__)

EventType = Literal["partial", "final", "supersede", "progress", "done", "error"]

#: Default seconds of total pipeline inactivity -- no event arriving AND no fed
#: audio being consumed via :meth:`TranscriptionSession.audio_chunks` -- before
#: the session synthesizes a terminal ``done_timeout`` error. This is a hang
#: backstop against stuck adapters, NOT engine-liveness detection: engines that
#: legitimately emit nothing during user silence stay alive for as long as the
#: adapter keeps consuming audio (industry anchors stream liveness to audio
#: flow, not result flow -- see docs/research/5). After ``end_audio()`` there is
#: nothing left to consume, so this bounds the engine's flush-and-``done``
#: window (spec ST.6.1: ``done`` MUST arrive, bounded by a timeout).
DEFAULT_DONE_TIMEOUT: float | None = 300.0

#: Default seconds without a *content* event (``partial`` / ``final`` /
#: ``supersede``) before the session is force-terminated with
#: ``stream_stalled``. ``None`` (the default) disables it: silence is a normal
#: state for a live session, and no surveyed engine emits content on a schedule
#: (docs/research/5). Opt in where continuous speech is expected, or to detect
#: a pathological engine that keeps consuming audio (or heartbeating) without
#: ever producing content -- this deadline is NOT reset by ``progress``
#: heartbeats or audio consumption.
DEFAULT_MAX_IDLE: float | None = None

#: Default absolute session wall-clock cap in seconds. ``None`` (the default)
#: disables the cap -- long-lived sessions are legal out of the box. When set
#: to a finite value it guarantees termination even if content and heartbeat
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

#: Upper bound on the number of lifecycle-suppression diagnostics a single
#: session's :class:`_LifecycleGuard` retains. Like every other session
#: resource (event buffer, audio queue, audio history -- all bounded), the
#: diagnostic channel is bounded so a misbehaving-but-usable engine that trips a
#: clamp on (nearly) every event -- e.g. a perpetually jittering audio cursor --
#: cannot grow the list without limit over a multi-hour live session and slowly
#: exhaust memory (spec ST.6.4 bounded-buffer philosophy). On overflow the guard
#: stops retaining individual entries and instead keeps a single aggregated
#: ``diagnostics_truncated`` summary (per-code counts) as the final entry, so the
#: overflow is reported honestly rather than silently dropped.
DEFAULT_MAX_GUARD_DIAGNOSTICS = 1000

#: Seconds per wait slice of the sync bridge's event pump. Event waits are
#: unbounded by design (a live, fed session may legitimately go arbitrarily
#: long between events); the pump polls in slices purely to detect the death
#: of its owned event-loop thread.
_SYNC_PUMP_POLL_SECONDS = 5.0

#: Stable name for the SyncSession's owned event-loop thread. A fixed name makes
#: the thread identifiable in a traceback / debugger, and documents that the
#: bridge owns exactly one background loop thread whose teardown the sync-bridge
#: compliance check asserts on directly (via ``is_loop_alive``) rather than by
#: diffing the whole process thread set -- a diff would mis-flag a dependency's
#: benign daemon thread (e.g. tqdm's monitor) as a bridge leak.
_SYNC_BRIDGE_LOOP_THREAD_NAME = "standard-asr-sync-bridge-loop"

_INPUT_SOURCE_ERROR_DETAIL = "Audio input source failed during streaming."


class StreamDeadlines(BaseModel):
    """Application-side overrides for a session's termination deadlines.

    Pass to ``StandardASR.start_transcription(deadlines=...)``. Only fields you
    explicitly set are applied; unset fields keep whatever the adapter (or the
    standard default) chose. Precedence: application explicit > adapter
    construction choice > standard default.

    The fields mirror the :class:`TranscriptionSession` deadline parameters --
    see there for full semantics. In short: ``done_timeout`` is the
    pipeline-inactivity hang backstop (reset by events AND by audio
    consumption), ``max_idle`` the opt-in content-stall detector, and
    ``max_session_seconds`` the opt-in absolute wall-clock cap. Each accepts
    ``None`` to explicitly disable that deadline.
    """

    model_config = ConfigDict(frozen=True)

    done_timeout: float | None = Field(default=DEFAULT_DONE_TIMEOUT, gt=0.0)
    max_idle: float | None = Field(default=DEFAULT_MAX_IDLE, gt=0.0)
    max_session_seconds: float | None = Field(default=DEFAULT_MAX_SESSION_SECONDS, gt=0.0)


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

    # allow_inf_nan=False matches Word/Segment: a NaN/Inf time is rejected at
    # construction, so a malformed adapter timestamp fails loudly on the event
    # rather than silently flowing over the wire or deferring the crash to result
    # reduction (a silent wrong timestamp is the cardinal sin; spec TR.2).
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    type: EventType
    segment_id: str | None = None
    text: str | None = None
    stable_until: int | None = None
    finality: Literal["final", "closed"] = "final"
    words: list[Word] | None = None
    # Time-frame fields share Word/Segment's TR.2 invariant: non-negative finite
    # seconds (origin = first session sample). ``ge=0`` rejects a negative time
    # here instead of letting it pass event validation only to crash later when
    # StreamReducer builds a Segment from it.
    start: float | None = Field(default=None, ge=0.0)
    end: float | None = Field(default=None, ge=0.0)
    audio_processed_until: float | None = Field(default=None, ge=0.0)
    old_ids: list[str] = Field(default_factory=list)
    new_ids: list[str] = Field(default_factory=list)
    code: str | None = None
    recoverable: bool | None = None
    retriable_after: float | None = Field(default=None, ge=0.0)
    reconnect: bool | None = None
    gap_start: float | None = Field(default=None, ge=0.0)
    gap_end: float | None = Field(default=None, ge=0.0)
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

    @model_validator(mode="before")
    @classmethod
    def _default_error_recoverability(cls, data: Any) -> Any:
        """Normalize an ``error`` event's unset ``recoverable`` to ``False``.

        ``recoverable=None`` on an error would be an undefined third state:
        :attr:`is_terminal` checks ``recoverable is False``, so ``None`` would
        silently read as "recoverable" and leave consumers waiting on a stream
        that may never continue. Unknown recoverability MUST fail safe to
        terminal (the :meth:`make_error` factory already defaults ``False``;
        this covers direct construction).

        Args:
            data: The raw constructor input.

        Returns:
            The input, with ``recoverable=False`` filled in for error events.
        """
        if not isinstance(data, dict):
            return data
        mapping = cast("dict[str, Any]", data)
        if mapping.get("type") == "error" and mapping.get("recoverable") is None:
            updated: dict[str, Any] = {**mapping, "recoverable": False}
            return updated
        return mapping

    @field_validator("detected_language")
    @classmethod
    def _check_detected_language(cls, value: str | None) -> str | None:
        """Validate and canonicalize ``detected_language`` (mirrors spec TR.1).

        Delegates to the shared
        :func:`~standard_asr.language.validate_detected_language` -- the same
        rule as ``TranscriptionResult.detected_language``, because the event
        field is the §6.3 reconnect-continuity mechanism and the two sides MUST
        accept exactly the same tags. The import is deferred because
        :mod:`standard_asr.language` imports from :mod:`standard_asr.results`.

        Args:
            value: The candidate detected-language tag, or ``None``.

        Returns:
            The canonicalized tag, or ``None`` when not applicable.

        Raises:
            ValueError: If ``value`` is the reserved ``auto`` token or is not a
                well-formed BCP-47 tag.
        """
        from .language import validate_detected_language

        return validate_detected_language(value)

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
            su = self.stable_until
            if su is not None and not 0 <= su <= len(self.text):
                # Structural bound only: ``text[:stable_until]`` must be a real
                # prefix, or every consumer (UI, wire client) receives an
                # unsatisfiable frozen-prefix claim. The *combining-character*
                # boundary rule stays at the guard/compliance layer (spec ST.4.2
                # keeps it a sequence-level concern with clamp-and-diagnose
                # semantics, not a construction error).
                raise ValueError(
                    f"stable_until {su} is out of range for text of length "
                    f"{len(self.text)} (text[:stable_until] must be a real prefix)."
                )
        elif self.type == "supersede":
            if not self.old_ids:
                raise ValueError("supersede event MUST retire at least one segment (old_ids).")
            if len(set(self.old_ids)) != len(self.old_ids):
                raise ValueError(
                    "supersede old_ids MUST NOT repeat a segment id "
                    "(an id retires the moment it appears in old_ids; spec §ST 5.2)."
                )
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
        from. This is a documented v1 limitation; the spec does not
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
            ValueError: If ``old_ids`` and ``new_ids`` intersect, or if ``old_ids``
                repeats a segment id.
        """
        if len(set(old_ids)) != len(old_ids):
            raise ValueError(
                "supersede old_ids MUST NOT repeat a segment id "
                "(an id retires the moment it appears in old_ids; spec §ST 5.2)."
            )
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
        # Strip each segment and drop empties so a segment carrying edge
        # whitespace (or an empty committed segment) does not inject a double
        # space / stray separator into the reduced transcript (AW-3). Segments are
        # space-joined as the v1 default; a no-space-language (CJK) separator is a
        # separate spec question, not silently changed here.
        text = " ".join(part for part in (segment.text.strip() for segment in segments) if part)
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
    ST.6.4: "a merge MUST be invalidated by the same segment's final/closed/supersede
    ... that partial MUST be dropped"). ``final`` / ``supersede`` / ``done`` / ``error`` are never
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

        ``final`` / ``supersede`` / ``error`` / ``done`` are appended drop-proof
        (bypassing the bound, spec ST.6.4), so only a NEW partial slot for a
        not-yet-pending segment or a ``progress`` event can overflow.

        Args:
            event: The event to enqueue.

        Raises:
            EventBufferOverflow: If the buffer is at capacity and the event is a
                NEW partial for a not-yet-pending segment (growing the buffer)
                or a ``progress`` heartbeat.
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

        if event.type in ("error", "done"):
            # error MUST never be dropped either (spec ST.6.4 does not
            # distinguish recoverable from terminal). Terminal error/done
            # normally arrive via put_forced; this branch covers
            # adapter-yielded recoverable errors, which previously fell into
            # the bounded path below and could be replaced by a backpressure
            # error on overflow -- losing the original error's semantics. No
            # stale-partial invalidation here: a recoverable error does not
            # end its segment. The unbounded residual matches the accepted
            # finals/supersedes case above (the spec forbids dropping them).
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

    def _pop_live(self) -> TranscriptionEvent | None:
        """Pop the next live pending event without waiting.

        Returns:
            The next live event, or ``None`` when nothing live is pending.
        """
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
        return None

    async def get(self) -> TranscriptionEvent | None:
        """Pop the next live event, awaiting one if necessary.

        Returns:
            The next event, or ``None`` once closed and drained.
        """
        while True:
            event = self._pop_live()
            if event is not None:
                return event
            if self._closed:
                return None
            self._event.clear()
            await self._event.wait()

    def drain(self) -> list[TranscriptionEvent]:
        """Synchronously pop every live pending event (deadline-drain path).

        Used when an iteration deadline fires: events already admitted to the
        reducer (and therefore part of ``result()``) but still buffered MUST
        reach the consumer ahead of the synthesized terminal, or ``result()``
        would diverge from the delivered stream.

        Returns:
            The pending live events, in delivery order.
        """
        drained: list[TranscriptionEvent] = []
        while (event := self._pop_live()) is not None:
            drained.append(event)
        return drained


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

    #: Diagnostic code of the aggregated overflow summary appended once the cap
    #: is reached. Stable so a reader (and the compliance suite) can recognize a
    #: truncated diagnostic stream.
    OVERFLOW_CODE = "diagnostics_truncated"

    def __init__(
        self, *, strict: bool = False, max_diagnostics: int = DEFAULT_MAX_GUARD_DIAGNOSTICS
    ) -> None:
        """Initialize the guard.

        Args:
            strict: If ``True``, raise on an illegal transition instead of
                suppressing it.
            max_diagnostics: Upper bound on retained diagnostics before the
                guard switches to an aggregated overflow summary (the bounded
                diagnostic channel; spec ST.6.4). MUST be > 0.

        Raises:
            ValueError: If ``max_diagnostics`` is not positive.
        """
        if max_diagnostics <= 0:
            raise ValueError("max_diagnostics must be > 0.")
        self._strict = strict
        self._max_diagnostics = max_diagnostics
        self._state: dict[str, str] = {}
        self._stable_until: dict[str, int] = {}
        self._frozen_text: dict[str, str] = {}
        self._audio_cursor: float = 0.0
        #: Maps each ``new_id`` of an active supersede group to its shared
        #: frozen-prefix-preservation obligation (spec ST.5.2).
        self._supersede_obligations: dict[str, _SupersedeObligation] = {}
        #: Set once :meth:`finalize` has run, so the end-of-session obligation
        #: sweep is emitted at most once per guard.
        self._finalized = False
        self.diagnostics: list[Diagnostic] = []
        #: Per-code count of diagnostics dropped after the cap was reached;
        #: surfaced through the single overflow-summary entry. Empty until the
        #: list first overflows.
        self._overflow_counts: dict[str, int] = {}

    def _record(self, diagnostic: Diagnostic) -> None:
        """Append a diagnostic, enforcing the bounded-channel cap.

        Below the cap the diagnostic is retained verbatim. Once the cap is
        reached the guard stops retaining individual entries and instead keeps a
        single aggregated :attr:`OVERFLOW_CODE` summary as the final list entry,
        tallying how many diagnostics of each code overflowed -- so a
        misbehaving engine cannot grow the list without limit (spec ST.6.4) yet
        the overflow is still reported honestly (never silently dropped).

        Args:
            diagnostic: The diagnostic to record.
        """
        # Reserve the final slot for the overflow summary: retain individual
        # entries only up to (cap - 1), so the list -- real entries plus at most
        # one summary -- never exceeds the cap.
        if not self._overflow_counts and len(self.diagnostics) < self._max_diagnostics - 1:
            self.diagnostics.append(diagnostic)
            return
        # At or past the cap: aggregate by code and (re)write the single summary
        # entry occupying the last slot.
        self._overflow_counts[diagnostic.code] = self._overflow_counts.get(diagnostic.code, 0) + 1
        summary = Diagnostic(
            level="warning",
            code=self.OVERFLOW_CODE,
            message=(
                f"diagnostics truncated at {self._max_diagnostics} entries; "
                f"further suppressions aggregated by code: "
                f"{dict(sorted(self._overflow_counts.items()))}."
            ),
        )
        if (
            self._overflow_counts
            and self.diagnostics
            and self.diagnostics[-1].code == self.OVERFLOW_CODE
        ):
            self.diagnostics[-1] = summary
        else:
            self.diagnostics.append(summary)

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
        self._record(Diagnostic(level="warning", code=code, message=message))

    def record_diagnostic(self, diagnostic: Diagnostic) -> None:
        """Record an author-emitted diagnostic into the bounded channel (SF-2).

        The public entry point for :meth:`TranscriptionSession.emit_diagnostic`: a
        producer's mid-stream diagnostic flows through the same bounded,
        overflow-capped channel (spec ST.6.4) the guard uses for its own
        suppression diagnostics, so the streaming layer has the diagnostics channel
        the batch path has via ``result.diagnostics`` -- rather than authors having
        no sanctioned way to surface a non-fatal note from ``_produce``.

        Args:
            diagnostic: The diagnostic to record.
        """
        self._record(diagnostic)

    def admit(self, event: TranscriptionEvent) -> TranscriptionEvent | None:
        """Validate (and possibly clamp) an event before it is forwarded.

        Args:
            event: The raw event from the producer.

        Returns:
            The event to forward (possibly with a clamped ``stable_until``), or
            ``None`` if the event is an illegal transition and was suppressed.
        """
        event, pending_cursor = self._clamp_audio_cursor(event)
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
                old_state = self._state.get(old)
                if old_state == "closed":
                    self._reject(
                        "lifecycle_closed_superseded",
                        f"supersede old_ids contains closed segment {old!r}; "
                        "suppressed (spec ST.5.3: closed MUST NOT be superseded).",
                    )
                    return None
                if old_state == "superseded":
                    # superseded is a terminal state (spec ST.5.1): an id retires
                    # the moment it appears in old_ids. Without this check a
                    # second retirement would copy the retired segment's frozen
                    # text into a SECOND independent replacement lineage.
                    self._reject(
                        "lifecycle_retired_resuperseded",
                        f"supersede old_ids contains already-superseded segment "
                        f"{old!r}; suppressed (spec ST.5.1/ST.5.2: superseded is "
                        "terminal -- an id MUST NOT be retired twice).",
                    )
                    return None
            for new in event.new_ids:
                if new in self._state:
                    self._reject(
                        "supersede_reintroduces_segment",
                        f"supersede new_ids contains already-known segment "
                        f"{new!r} (state {self._state[new]!r}); suppressed. "
                        "Either the id is being reused (spec ST.5.2: a new_id "
                        "MUST be fresh) or this supersede was delivered out of "
                        "order (spec ST.5.2: a supersede MUST precede any "
                        "partial/final of its new_ids).",
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
            self._commit_audio_cursor(pending_cursor)
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
            had_stable_until = sid in self._stable_until
            prior_stable_until = self._stable_until.get(sid, 0)
            had_frozen_text = sid in self._frozen_text
            prior_frozen_text = self._frozen_text.get(sid, "")
            obligation = self._supersede_obligations.get(sid)
            had_obligation_frozen = obligation is not None and sid in obligation.frozen
            prior_obligation_frozen = (
                obligation.frozen.get(sid, "") if obligation is not None else ""
            )

            event = self._clamp_stable_until(event, sid, allow_decrease=is_closed_final)
            su = event.stable_until or 0
            if su > 0 and event.text is not None:
                self._frozen_text[sid] = event.text[:su]
                if is_closed_final and obligation is not None:
                    # Bookkeeping only: a closed final may legally rewrite frozen
                    # text (the divergence rejection below is exempted for it),
                    # but its freeze still fulfils the segment's supersede
                    # obligation. Without recording it here -- the rejection
                    # path's _supersede_preserves_frozen is the ledger's only
                    # other writer -- a closed final that is a replacement
                    # group's sole freeze never registers, and finalize() emits
                    # a false supersede_obligation_unfulfilled for fully
                    # preserved text (a lying diagnostic).
                    obligation.frozen[sid] = self._frozen_text[sid]
                if not is_closed_final and not self._supersede_preserves_frozen(sid):
                    assert obligation is not None
                    # Capture the diverging comparison BEFORE the rollback below
                    # restores ``obligation.frozen[sid]``. The contradiction is a
                    # property of the supersede GROUP, not of ``sid`` alone: a
                    # later (e.g. out-of-order) freeze on ``sid`` can simply
                    # complete the contiguous run and expose an EARLIER new id's
                    # divergence, so blaming ``sid`` mis-attributes the rewrite.
                    # Report the group and the F_old-vs-F_new comparison instead.
                    f_old = obligation.f_old
                    f_new = obligation.f_new()
                    group = obligation.new_ids
                    if had_stable_until:
                        self._stable_until[sid] = prior_stable_until
                    else:
                        self._stable_until.pop(sid, None)
                    if had_frozen_text:
                        self._frozen_text[sid] = prior_frozen_text
                    else:
                        self._frozen_text.pop(sid, None)
                    if had_obligation_frozen:
                        obligation.frozen[sid] = prior_obligation_frozen
                    else:
                        obligation.frozen.pop(sid, None)
                    self._reject(
                        "frozen_prefix_rewritten_supersede",
                        f"supersede replacement group {group!r} froze a "
                        f"concatenated prefix {f_new!r} that diverges from the "
                        f"retired frozen text {f_old!r} it MUST preserve; the freeze "
                        f"on segment {sid!r} (the latest in the group to freeze) "
                        "completed the contiguous run that exposed the divergence. "
                        "Suppressed (spec ST.5.2: supersede MUST preserve frozen "
                        "text).",
                    )
                    return None
            if event.type == "final":
                self._state[sid] = "closed" if event.finality == "closed" else "final"
            else:
                self._state[sid] = "open"
            self._commit_audio_cursor(pending_cursor)
            return event

        self._commit_audio_cursor(pending_cursor)
        return event

    def _clamp_stable_until(
        self, event: TranscriptionEvent, sid: str, *, allow_decrease: bool = False
    ) -> TranscriptionEvent:
        """Clamp a decreasing or invalid ``stable_until`` (spec ST.4.2).

        With ``allow_decrease`` (the terminal ``closed`` event), a *smaller*
        ``stable_until`` is spec-legal: the post-processing rewrite (ITN /
        punctuation / casing, spec ST.5.3) may shorten the text -- e.g.
        "twenty twenty" -> "2020" -- so the monotonic-increase rule MUST NOT
        clamp it back up above the new text. Only the structural bounds
        (``0 <= stable_until <= len(text)``, non-combining cut) are repaired,
        and the closed frontier is **not** recorded as the segment's running
        frontier (the segment is terminal; recording it would poison nothing
        but means nothing).

        Args:
            event: The partial/final event.
            sid: The segment id.
            allow_decrease: ``True`` for a terminal ``closed`` event, whose
                ``stable_until`` may legally shrink along with the text.

        Returns:
            The event, with ``stable_until`` clamped if it decreased illegally
            or was an invalid boundary; otherwise the event unchanged.
        """
        su = event.stable_until
        if su is None:
            return event
        prior = self._stable_until.get(sid, 0)
        text = event.text or ""
        clamped = su
        reason = ""
        if su < prior and not allow_decrease:
            clamped = prior
            reason = (
                f"stable_until decreased {su} -> clamped to {prior} "
                "(spec ST.4.2: MUST only increase)"
            )
        if not validate_stable_until(text, clamped):
            # Fall back to the largest valid boundary <= clamped without moving
            # below the previously-published frozen frontier (for a closed
            # event the frontier constraint is void, so the floor is 0).
            floor = 0 if allow_decrease else prior
            safe = min(clamped, len(text))
            while safe > floor and not validate_stable_until(text, safe):
                safe -= 1
            if reason:
                reason += "; "
            reason += f"stable_until {clamped} invalid boundary -> {safe}"
            clamped = safe
        if reason:
            self._reject("stable_until_clamped", reason)
        if clamped != su:
            event = event.model_copy(update={"stable_until": clamped})
        if not allow_decrease:
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

    def finalize(self) -> list[Diagnostic]:
        """Sweep for supersede obligations left unfulfilled at session end.

        The supersede frozen-prefix rule is asymmetric (spec ST.5.2): the
        *contradiction* direction is rejected eagerly in :meth:`admit`, but the
        *conservative* direction -- the replacement's concatenated frozen prefix
        ``F_new`` remaining strictly SHORTER than the retired ``F_old`` -- is
        permitted to stay pending, on the bet that later events will re-freeze the
        rest. If the session ends with that bet unsettled, frozen text the user
        saw was effectively dropped from the lineage; the spec allows it but
        wants it reported "at most with a soft diagnostic". This emits exactly
        that: one **soft** (``info``) ``supersede_obligation_unfulfilled``
        diagnostic per still-short obligation, naming the affected ``new_ids``.
        It is NOT an error and does not reject anything -- the supersede stands.

        Call this once when the session reaches its terminal (the base appends
        ``done``) and once when :func:`~standard_asr.compliance.check_event_sequence`
        finishes replaying. Idempotent: the sweep runs at most once per guard.

        Returns:
            The newly emitted obligation diagnostics (also recorded into
            :attr:`diagnostics`, subject to the bounded-channel cap); empty if
            every obligation reconciled.
        """
        if self._finalized:
            return []
        self._finalized = True
        emitted: list[Diagnostic] = []
        seen: set[int] = set()
        for obligation in self._supersede_obligations.values():
            # new_ids of one supersede share a single obligation object; emit
            # once per obligation, not once per new_id.
            if id(obligation) in seen:
                continue
            seen.add(id(obligation))
            if len(obligation.f_new()) < len(obligation.f_old):
                emitted.append(
                    Diagnostic(
                        level="info",
                        code="supersede_obligation_unfulfilled",
                        message=(
                            f"supersede replacement {obligation.new_ids!r} ended with its "
                            f"concatenated frozen prefix shorter than the retired frozen text "
                            f"({obligation.f_new()!r} vs {obligation.f_old!r}); the unre-frozen "
                            "tail was dropped from the lineage (spec ST.5.2: permitted, "
                            "reported as a soft diagnostic)."
                        ),
                    )
                )
        for diagnostic in emitted:
            self._record(diagnostic)
        return emitted

    def _clamp_audio_cursor(
        self, event: TranscriptionEvent
    ) -> tuple[TranscriptionEvent, float | None]:
        """Clamp a decreasing ``audio_processed_until`` cursor (spec ST.4.1).

        The audio-time cursor is monotonic across the whole session (it never
        moves backwards), independent of segment. A decrease is clamped to the
        prior value with a diagnostic (or raises in strict mode).

        This method does NOT advance the session cursor itself: the advance is
        committed via :meth:`_commit_audio_cursor` only on the paths where
        :meth:`admit` actually forwards the event. Committing up front would
        let a suppressed illegal event poison the cursor, clamping every later
        legal event up to the rejected value.

        Args:
            event: Any event (the cursor may appear on content or progress).

        Returns:
            A ``(event, pending_cursor)`` pair: the event (with
            ``audio_processed_until`` clamped if it decreased) and the cursor
            value to commit if the event is admitted (``None`` when there is
            nothing to commit).
        """
        cursor = event.audio_processed_until
        if cursor is None:
            return event, None
        if cursor < self._audio_cursor:
            self._reject(
                "audio_cursor_decreased",
                f"audio_processed_until decreased {cursor} -> clamped to "
                f"{self._audio_cursor} (spec ST.4.1: the cursor is monotonic).",
            )
            return (
                event.model_copy(update={"audio_processed_until": self._audio_cursor}),
                None,
            )
        return event, cursor

    def _commit_audio_cursor(self, cursor: float | None) -> None:
        """Advance the session audio cursor for an admitted event.

        Args:
            cursor: The pending cursor from :meth:`_clamp_audio_cursor`, or
                ``None`` when the admitted event carried nothing to commit.
        """
        if cursor is not None and cursor > self._audio_cursor:
            self._audio_cursor = cursor


#: Private attribute names owned by :class:`TranscriptionSession`'s base machinery
#: (input ownership, backpressure buffer, lifecycle guard, deadlines, reconnect
#: scaffolding, ...). A subclass that rebinds any of these clobbers base state and
#: would otherwise crash deep in the producer with a cryptic error far from the
#: cause; the reserved-attribute guard (snapshotted in ``__init__``, validated in
#: ``__aenter__``) turns that into a loud, named error at session open (SF-1). Kept
#: in sync with ``__init__`` by ``test_reserved_session_attrs_matches_base_init``.
_RESERVED_SESSION_ATTRS: frozenset[str] = frozenset(
    {
        "_audio_queue",
        "_buffer",
        "_guard",
        "_mode",
        "_ended",
        "_input_released",
        "_done_timeout",
        "_max_idle",
        "_max_session_seconds",
        "_feed_task",
        "_producer_task",
        "_reducer",
        "_audio_history",
        "_replayable",
        "_replay_source",
        "_pending_reconnects",
        "_monotonic",
        "_last_audio_activity",
        "_session_started_at",
        "_iterating",
        "_initial_diagnostics",
    }
)


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
      ``error(code="content_lost", recoverable=True)`` fidelity warning.
    """

    def __init__(
        self,
        *,
        done_timeout: float | None = DEFAULT_DONE_TIMEOUT,
        max_idle: float | None = DEFAULT_MAX_IDLE,
        max_session_seconds: float | None = DEFAULT_MAX_SESSION_SECONDS,
        event_buffer_capacity: int = DEFAULT_EVENT_BUFFER_CAPACITY,
        audio_queue_maxsize: int = DEFAULT_AUDIO_QUEUE_MAXSIZE,
        audio_history_maxlen: int = DEFAULT_AUDIO_HISTORY_MAXLEN,
        strict_lifecycle: bool = False,
        max_guard_diagnostics: int = DEFAULT_MAX_GUARD_DIAGNOSTICS,
    ) -> None:
        """Initialize the session.

        Args:
            done_timeout: Seconds of total pipeline inactivity -- no event
                arriving AND no fed audio consumed via :meth:`audio_chunks` --
                before synthesizing a ``done_timeout`` error. A hang backstop,
                not engine-liveness detection: a silently-listening engine
                stays alive while the adapter keeps consuming audio. After
                ``end_audio()`` it bounds the engine's flush-and-``done``
                window. ``None`` disables (explicit opt-out of the backstop).
            max_idle: Seconds without a *content* event (``partial`` / ``final``
                / ``supersede``) before force-terminating with a
                ``stream_stalled`` error. NOT reset by ``progress`` heartbeats
                or audio consumption, so it detects an engine that consumes (or
                chats) without ever producing content. ``None`` (the default)
                disables: silence is a normal state for a live session.
            max_session_seconds: Absolute wall-clock cap; ``None`` disables.
            event_buffer_capacity: Max pending non-coalesced events before
                overflow emits a ``backpressure`` error.
            audio_queue_maxsize: Max pending audio chunks; bounds ``feed`` /
                ``send_audio`` so a slow engine exerts real backpressure.
            audio_history_maxlen: Capacity of the bounded rolling audio buffer
                used to replay recent audio after a reconnect.
            strict_lifecycle: If ``True``, raise on illegal lifecycle
                transitions instead of suppressing + diagnosing them.
            max_guard_diagnostics: Cap on the bounded lifecycle-suppression
                diagnostics channel (spec ST.6.4); further diagnostics are
                aggregated into a single overflow summary rather than growing
                without bound. Exposed alongside the other bounds so a session
                can size it; defaults to ``DEFAULT_MAX_GUARD_DIAGNOSTICS``.

        Raises:
            ValueError: If a deadline is not positive (or ``None`` where
                allowed) or a buffer/queue bound is not positive. In particular
                ``audio_queue_maxsize=0`` would mean an UNBOUNDED
                ``asyncio.Queue`` -- silently disabling the documented feed
                backpressure -- so it is rejected rather than passed through.
                Also if ``max_guard_diagnostics`` is not positive.
        """
        if done_timeout is not None and done_timeout <= 0:
            raise ValueError("done_timeout must be > 0 seconds, or None to disable.")
        if max_idle is not None and max_idle <= 0:
            raise ValueError("max_idle must be > 0 seconds, or None to disable.")
        if max_session_seconds is not None and max_session_seconds <= 0:
            raise ValueError("max_session_seconds must be > 0 seconds, or None to disable.")
        if event_buffer_capacity <= 0:
            raise ValueError("event_buffer_capacity must be > 0 events.")
        if audio_queue_maxsize <= 0:
            raise ValueError(
                "audio_queue_maxsize must be > 0 chunks (0 means an unbounded "
                "asyncio.Queue, which would silently disable feed backpressure)."
            )
        # Validated here (not only in _LifecycleGuard) so the error names the public
        # parameter and is raised from the documented location, like every other
        # bound above (the guard re-checks defensively for its own direct callers).
        if max_guard_diagnostics <= 0:
            raise ValueError("max_guard_diagnostics must be > 0 entries.")
        self._audio_queue: asyncio.Queue[bytes | _InputSourceFailure | None] = asyncio.Queue(
            maxsize=audio_queue_maxsize
        )
        self._buffer = _CoalescingBuffer(capacity=event_buffer_capacity)
        self._guard = _LifecycleGuard(
            strict=strict_lifecycle, max_diagnostics=max_guard_diagnostics
        )
        self._mode: Literal["feed", "manual"] | None = None
        self._ended = False
        #: Set by :meth:`_terminate` once a terminal event is on its way to the
        #: consumer: the audio queue has no consumer anymore, so the put side is
        #: released (blocked putters wake; new ``send_audio`` calls raise).
        self._input_released = False
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
        #: Liveness anchor (spec ST.6.1): advanced on every audio chunk the
        #: adapter consumes via :meth:`audio_chunks`, so ``done_timeout``
        #: measures pipeline inactivity rather than mere event silence.
        self._last_audio_activity: float = self._monotonic()
        #: Absolute wall-clock origin for ``max_session_seconds`` (spec ST.6.1:
        #: an *absolute session* cap, not a per-iteration one). Anchored at
        #: session establishment (:meth:`__aenter__`) -- NOT at the first
        #: :meth:`__anext__` -- so the time between ``__aenter__`` and the first
        #: iteration counts, and a re-entered iterator cannot reset the cap and
        #: (unintentionally) renew the session indefinitely. ``None`` until the
        #: session is opened.
        self._session_started_at: float | None = None
        #: Guards the single-consumer contract: a second :meth:`__aiter__`
        #: raises rather than racing the first iterator for events on the shared
        #: buffer (competing consumers would silently split the stream).
        self._iterating = False
        # Standard-layer diagnostics (parameter gating / language resolution)
        # attached by the base ``start_transcription`` template before the
        # session is handed to the application, so they surface through the
        # session's existing ``diagnostics()`` channel.
        self._initial_diagnostics: list[Diagnostic] = []
        # Reserved-attribute guard (SF-1): snapshot the base-owned objects now, so
        # the guard can fail loudly if a subclass rebinds one of these reserved
        # private names (almost always by accident -- e.g. using ``self._buffer``
        # for its own state) and silently clobbers base machinery. References (not
        # ``id``) compared with ``is`` so a recycled id cannot fake a match; the
        # store + the once-flag are name-mangled so they cannot themselves be the
        # clobbered attribute. Validation runs exactly once, at the earliest entry
        # point (``start_transcription`` on the engine path, else the first of
        # ``feed`` / ``send_audio`` / ``end_audio`` / ``__aenter__``), BEFORE the
        # base legitimately rebinds any reserved attribute -- so the comparison is
        # always pristine-snapshot vs current and never mistakes a base rebind for
        # a subclass clobber.
        self.__reserved_attrs: dict[str, object] = {
            name: getattr(self, name) for name in _RESERVED_SESSION_ATTRS
        }
        self.__reserved_checked = False

    def _ensure_reserved_attrs_checked(self) -> None:
        """Validate the reserved-attribute guard once, before the base mutates state.

        Idempotent: the first call (whichever entry point runs first) performs the
        check; later calls are a cheap no-op. Placed at the top of every entry
        point that could run first, so the check always sees the pristine
        post-``__init__`` snapshot versus the current attributes -- catching a
        subclass clobber before any legitimate base rebind muddies the comparison.

        Raises:
            TypeError: If a subclass rebound a reserved base attribute (SF-1).
        """
        if not self.__reserved_checked:
            self._check_reserved_attrs()
            self.__reserved_checked = True

    def _replace_reserved_attr(self, name: str, value: object) -> None:
        """Rebind a reserved base attribute, keeping the SF-1 guard snapshot in sync.

        The reserved-attribute guard treats any post-``__init__`` rebind of a
        reserved name as a subclass clobber. This is the one supported way to
        override one afterwards: it updates the snapshot so the override is tracked
        rather than flagged. It exists for the library's own white-box tests (e.g.
        injecting a deterministic clock into ``_monotonic``); it is NOT part of the
        engine-author contract -- adapters configure via the ``__init__`` bounds,
        and an accidental ``self._buffer = ...`` never routes through here, so the
        guard still catches it.

        Args:
            name: A reserved attribute name (must be in
                :data:`_RESERVED_SESSION_ATTRS`).
            value: The replacement value.

        Raises:
            KeyError: If ``name`` is not a reserved base attribute.
        """
        if name not in self.__reserved_attrs:
            raise KeyError(name)
        setattr(self, name, value)
        self.__reserved_attrs[name] = value

    def _check_reserved_attrs(self) -> None:
        """Fail loudly if a subclass rebound a reserved base attribute (SF-1).

        Compares the current reserved attributes against the snapshot taken in
        ``__init__``. A subclass that assigns one of :data:`_RESERVED_SESSION_ATTRS`
        (e.g. its own ``self._buffer``) corrupts the base session machinery and
        would otherwise surface only as a cryptic crash deep inside the producer,
        far from the cause. Turning that into a named error at session start is the
        explicit-over-silent contract (a loud error the author can fix beats a
        mangled transcript).

        Raises:
            TypeError: If any reserved base attribute was rebound (or removed) by a
                subclass, naming the offending attribute(s).
        """
        missing = object()
        clobbered = sorted(
            name
            for name, original in self.__reserved_attrs.items()
            if getattr(self, name, missing) is not original
        )
        if clobbered:
            raise TypeError(
                f"{type(self).__name__} overwrote reserved TranscriptionSession "
                f"attribute(s) {clobbered}: these private names belong to the base "
                "session machinery and rebinding one corrupts it. Rename your own "
                "attributes (e.g. prefix them with your engine name) so they do not "
                "collide with the reserved set."
            )

    def _apply_deadline_overrides(self, deadlines: StreamDeadlines) -> None:
        """Apply application-chosen deadline overrides (friend API).

        Called by the base ``start_transcription`` template after the adapter
        constructed the session, so the application's explicitly-set fields win
        over the adapter's construction-time choices without relying on every
        adapter to forward them (a forwarding obligation could be silently
        missed). Only fields the application explicitly set are applied.

        Args:
            deadlines: The validated override model.

        Raises:
            TypeError: If a subclass rebound a reserved base attribute (SF-1).
        """
        self._ensure_reserved_attrs_checked()
        if "done_timeout" in deadlines.model_fields_set:
            self._done_timeout = deadlines.done_timeout
        if "max_idle" in deadlines.model_fields_set:
            self._max_idle = deadlines.max_idle
        if "max_session_seconds" in deadlines.model_fields_set:
            self._max_session_seconds = deadlines.max_session_seconds

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

        Consuming from this iterator is the session's liveness anchor (spec
        ST.6.1): every dequeue resets the ``done_timeout`` backstop, so an
        engine that legitimately emits no events during user silence stays
        alive for as long as the adapter keeps reading audio here.

        Yields:
            Raw audio chunks in the session's declared format.
        """
        while True:
            chunk = await self._audio_queue.get()
            # Reset the liveness anchor before the sentinel checks: the final
            # (end-of-input) dequeue counts too, giving the engine a full
            # ``done_timeout`` window from end-of-input to flush and ``done``.
            self._last_audio_activity = self._monotonic()
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

    def emit_diagnostic(
        self,
        *,
        code: str,
        message: str,
        level: Literal["info", "warning"] = "info",
        param: str | None = None,
        provided: object | None = None,
        effective: object | None = None,
    ) -> None:
        """Surface a structured diagnostic from ``_produce`` (the streaming channel).

        The streaming counterpart of the batch path's ``result.diagnostics``: call
        this from your :meth:`_produce` to report a non-fatal note -- a best-effort
        degradation, an assumed parameter, a lossy fallback -- through the session's
        own :meth:`diagnostics` channel, so a WS client (mid-stream ``diagnostics``
        frame) or a sync caller (``diagnostics()``) sees WHY (SF-2; spec G.5.2). For
        a *fatal* condition, emit an ``error`` event instead -- this is for
        non-fatal notes that must not end the stream.

        Recorded through the same **bounded** channel (spec ST.6.4) as the guard's
        own diagnostics: past the cap entries aggregate into the overflow summary
        rather than growing without bound, so a chatty producer cannot exhaust
        memory.

        Security -- this is **engine-authored, client-facing output**, like the
        transcript text itself. Every field (``message`` / ``param`` / ``provided``
        / ``effective``) is forwarded to clients VERBATIM -- to a (possibly
        unauthenticated) WebSocket client as a ``diagnostics`` frame, and, for the
        batch counterpart ``result.diagnostics``, in the REST response -- and is
        **NOT** redacted. The standard layer scrubs only the detail IT auto-captures
        (an ``error`` event's ``extra``, which may hold ``str(exc)``); content you
        pass here is your contract to keep. NEVER pass a credential, API key, a URL
        with embedded auth, or raw exception text -- route sensitive operator detail
        to ``logging`` instead. (That asymmetry is deliberate: an ``error`` event's
        ``extra`` is auto-captured so the server drops it; this note you chose, so
        you own its safety.)

        Args:
            code: Stable, machine-readable diagnostic code (e.g. ``"vad_fallback"``).
            message: Human-readable explanation (client-facing; no secrets).
            level: ``"info"`` or ``"warning"`` (default ``"info"``).
            param: The parameter the diagnostic concerns, if any.
            provided: The value the application provided, if relevant (client-facing).
            effective: The value that took effect, if relevant (client-facing).
        """
        self._guard.record_diagnostic(
            Diagnostic(
                level=level,
                code=code,
                message=message,
                param=param,
                provided=provided,
                effective=effective,
            )
        )

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

        The base ALWAYS emits a ``progress(reconnect=True, gap_start, gap_end)``
        event in order with produced events, followed -- IMMEDIATELY (spec
        ST.6.3) -- by a trailing ``error(code="content_lost", recoverable=True,
        gap_start, gap_end)`` IFF the adapter passes ``content_lost=True``.

        The events are flushed **promptly**, the moment this is called, through
        the drop-proof path (:meth:`_CoalescingBuffer.put_forced`, the same sync
        primitive the base uses for synthesized terminals). This matters because
        an adapter typically calls :meth:`note_reconnect` and then BLOCKS while
        it re-establishes the connection without yielding another event; deferring
        the flush to the next produced event would leave the consumer staring at a
        timeout/silence during a slow reconnect -- the opposite of "transparent
        but honest" (spec ST.6.3). :meth:`note_reconnect` runs in the
        producer-task coroutine on the session loop, so a synchronous flush is
        correct and ordered (the events land after already-emitted events and
        before any subsequent one). The :meth:`_run_producer` / ``finally`` drains
        remain as a harmless safety net -- they will simply find nothing pending.

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
                unreplayable audio was permanently lost; queues a non-terminal
                ``content_lost`` fidelity warning after the progress (spec ST.6.3).
        """
        self._pending_reconnects.append(
            TranscriptionEvent.progress(reconnect=True, gap_start=gap_start, gap_end=gap_end)
        )
        if content_lost:
            self._pending_reconnects.append(
                TranscriptionEvent.make_error(
                    code="content_lost",
                    recoverable=True,
                    gap_start=gap_start,
                    gap_end=gap_end,
                )
            )
        # Flush promptly (drop-proof, in order) so the notification reaches the
        # consumer even if the adapter now blocks on a slow reconnect without
        # yielding another event. The progress + content_lost pair is drained
        # together, preserving their required adjacency (spec ST.6.3).
        self._drain_pending_reconnects()

    # ----- input ownership ------------------------------------------------- #
    def _claim_mode(self, mode: Literal["feed", "manual"]) -> None:
        """Atomically claim single input ownership for the first input call.

        Args:
            mode: The mode being claimed by this call.

        Raises:
            InvalidSessionUseError: If the other mode was already claimed.
                Mixing input modes is a usage error against a still-live
                session, NOT a lifecycle close (spec ST.3.3).
        """
        if self._mode is None:
            self._mode = mode
            return
        if self._mode != mode:
            other = "manual" if mode == "feed" else "feed"
            raise InvalidSessionUseError(f"{other} input already in use; cannot mix with {mode}.")

    def feed(self, source: Iterable[bytes] | AsyncIterable[bytes] | bytes | bytearray) -> None:
        """Feed audio from a managed source (mutually exclusive with manual).

        A bare ``bytes`` / ``bytearray`` is treated as a **single** audio chunk
        (not an iterable of chunks -- iterating it would yield ``int`` byte
        values). A non-async, re-iterable collection (``list`` / ``tuple``, or a
        wrapped bytes-like) is classified as **replayable** for reconnect
        purposes; an async source or a one-shot generator/iterator is
        **non-replayable**. Any ``AsyncIterable`` (the Pythonic ``__aiter__``
        protocol, not only the stricter ``AsyncIterator``) is accepted and
        normalized via :func:`aiter`.

        Args:
            source: A sync or async **iterable of byte chunks** (``Iterable`` /
                ``AsyncIterable`` of ``bytes``), or a single ``bytes`` /
                ``bytearray`` chunk.

        Raises:
            TypeError: If ``source`` is a ``str``. A ``str`` satisfies
                ``Iterable[str]`` and would be silently consumed one character
                at a time (or fail deep inside an adapter as a confusing
                ``engine_error``); passing a **file path** here is a common
                slip. A whole audio file goes to ``start_transcription(audio=...)``
                (spec ST.3.1), and incremental input is raw PCM byte chunks.
            InvalidSessionUseError: If manual input was already used (mixing) or
                ``feed`` was already called once -- a usage error against a
                still-live session, NOT a lifecycle close.
            TypeError: If a subclass rebound a reserved base attribute (SF-1).
        """
        self._ensure_reserved_attrs_checked()
        # Reject str up front: it is Iterable[str] and would otherwise fall into
        # the sync-iterable path and be consumed character by character (or blow
        # up far downstream), turning a likely "I passed a file path" slip into a
        # baffling engine_error rather than a fail-loud call-site error.
        if isinstance(source, str):
            raise TypeError(
                "feed() takes byte chunks, not a str. To transcribe a whole "
                "file pass start_transcription(audio=...); to stream incrementally "
                "feed raw PCM bytes (e.g. bytes chunks or an iterator of them)."
            )
        self._claim_mode("feed")
        if self._feed_task is not None:
            raise InvalidSessionUseError("feed() already called once.")
        drain_source: Iterable[bytes] | AsyncIterator[bytes]
        if isinstance(source, (bytes, bytearray)):
            # A bare bytes-like is ONE chunk: wrap it so draining yields the
            # whole chunk rather than iterating it into individual int values.
            replayable_list = [bytes(source)]
            self._replayable = True
            self._replay_source = tuple(replayable_list)
            drain_source = replayable_list
        elif isinstance(source, AsyncIterable):
            # Accept the broader AsyncIterable (only __aiter__), not just
            # AsyncIterator: aiter() normalizes it to an async iterator so a
            # custom async audio source (the most Pythonic shape) is consumed
            # via `async for` instead of falling into the sync branch and
            # failing as a misleading input_source_error. Async sources are
            # never replayable.
            self._replayable = False
            drain_source = aiter(source)
        else:
            # A sync iterable of chunks. Replayable iff a re-iterable collection
            # (list / tuple) -- a one-shot generator/iterator is not.
            self._replayable = isinstance(source, (list, tuple))
            if self._replayable:
                # Retain the FULL source so replay_buffer() can offer loss-free
                # replay even when the source is longer than the rolling ring --
                # "replayable" must mean the whole source, not merely the tail.
                self._replay_source = tuple(source)
            drain_source = source
        self._feed_task = asyncio.ensure_future(self._drain_source(drain_source))

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
        if self._input_released:
            # A terminal was already delivered: the queue has no consumer.
            # Discard instead of blocking forever on a dead queue.
            return
        await self._audio_queue.put(chunk)
        if self._input_released:
            # The session terminated while this put was blocked; it only
            # completed because _release_audio_input() drained the queue.
            # Re-drain so any putter still blocked behind this one wakes too
            # (cascade), discarding the chunk just queued (nothing consumes it).
            self._release_audio_input()

    async def send_audio(self, chunk: bytes) -> None:
        """Manually send one audio chunk (mutually exclusive with ``feed``).

        Manual sources are always treated as **non-replayable** (live input).

        Args:
            chunk: The audio chunk.

        Raises:
            InvalidSessionUseError: If ``feed`` was already used (mixing input
                modes is a usage error against a still-live session).
            StreamClosedError: If the input was ended (``end_audio()``) or the
                session already delivered a terminal event -- a genuine
                lifecycle close: the audio queue has no consumer anymore, so
                raising beats blocking forever on a dead queue.
            TypeError: If a subclass rebound a reserved base attribute (SF-1).
        """
        self._ensure_reserved_attrs_checked()
        # Claim manual ownership FIRST so mixing with an active feed always
        # raises the deterministic mixing error -- otherwise the feed task
        # setting _ended on exhaustion would race the _ended check below and
        # sometimes surface the "after end_audio" message instead (spec ST.3.3).
        self._claim_mode("manual")
        if self._ended:
            raise StreamClosedError("Cannot send_audio after end_audio().")
        if self._input_released:
            raise StreamClosedError(
                "Cannot send_audio after the session terminated (a terminal "
                "event ended the stream; no consumer reads the audio queue)."
            )
        await self._put_audio(chunk)

    async def end_audio(self) -> None:
        """Mark the end of manual audio input (idempotent in manual mode).

        Claims manual ownership if this is the first input call, so a later
        ``feed`` is correctly rejected as mixing (spec ST.3.3).

        Raises:
            InvalidSessionUseError: If ``feed`` was used (mixing input modes is
                a usage error against a still-live session).
            TypeError: If a subclass rebound a reserved base attribute (SF-1).
        """
        self._ensure_reserved_attrs_checked()
        self._claim_mode("manual")
        if self._ended or self._input_released:
            # Already ended -- or the session terminated and released the
            # input side, in which case the end-marker would land in a dead
            # queue (and could block on a full one).
            return
        self._ended = True
        await self._audio_queue.put(None)

    # ----- iteration ------------------------------------------------------- #
    async def __aenter__(self) -> TranscriptionSession:
        """Open resources and start the producer.

        Anchors the ``max_session_seconds`` wall-clock origin here, at session
        establishment, so the absolute cap measures from when the session
        opened -- not from the first :meth:`__anext__` -- and a re-entered
        iterator cannot reset it (spec ST.6.1).

        Returns:
            The session.

        Raises:
            TypeError: If a subclass rebound a reserved base attribute (SF-1).
        """
        self._ensure_reserved_attrs_checked()
        self._session_started_at = self._monotonic()
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
                if admitted.is_terminal:
                    # Engine-emitted terminal: run the shared terminal funnel,
                    # then deliver drop-proof.
                    self._buffer.put_forced(self._terminate(admitted))
                    return
                self._buffer.put(admitted)
            self._drain_pending_reconnects()
            # Session ended cleanly: the funnel sweeps for supersede obligations
            # whose replacement never re-froze all the retired frozen text. Any
            # such lineage loss is permitted but MUST be reported honestly as a
            # soft diagnostic (spec ST.5.2); it surfaces through diagnostics().
            # done MUST never be dropped: bypass the bound so it always lands.
            self._buffer.put_forced(self._terminate(TranscriptionEvent.done()))
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
            self._terminate(
                TranscriptionEvent.make_error(
                    code=code, recoverable=False, extra={"detail": detail}
                )
            )
        )

    def __aiter__(self) -> AsyncIterator[TranscriptionEvent]:
        """Return the event async iterator (single-consumer).

        A session has exactly ONE event stream and exactly ONE consumer: the
        events live in a shared bounded buffer, and a second concurrent
        iterator would race the first for :meth:`_CoalescingBuffer.get`,
        silently splitting the stream between them (each iterator would see an
        arbitrary subset, so neither's ``result`` would match the stream --
        breaking the stream == result invariant). Re-iterating would also reset
        the per-iteration deadline anchors. Both are programming errors, so the
        second call fails loudly instead of returning a competing iterator
        (spec ST.6.1).

        Returns:
            The session's event async iterator.

        Raises:
            InvalidSessionUseError: If the session is already being iterated --
                a usage error against a still-live session, not a lifecycle
                close.
            TypeError: If a subclass rebound a reserved base attribute (SF-1).
        """
        self._ensure_reserved_attrs_checked()
        if self._iterating:
            raise InvalidSessionUseError(
                "This session is already being iterated; a streaming session "
                "has a single event stream with a single consumer. Iterate it "
                "once (one `async for`), not concurrently or repeatedly."
            )
        self._iterating = True
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[TranscriptionEvent]:
        """Yield events with hang-backstop, idle, and wall-clock termination.

        Three independent deadlines (spec ST.6.1):

        * ``done_timeout`` -- pipeline-inactivity backstop. Anchored to the
          most recent *activity* = max(last event, last audio chunk consumed
          via :meth:`audio_chunks`), so a live session in a silent stretch
          (engine listening, emitting nothing) never trips it while the
          adapter keeps consuming audio. A stuck adapter -- or an engine that
          fails to deliver ``done`` after end-of-input, when no audio is left
          to consume -- does trip it.
        * ``max_idle`` -- max time without a *content* event (heartbeats and
          audio consumption do not reset it); opt-in.
        * ``max_session_seconds`` -- absolute wall-clock cap; opt-in.

        Termination guarantee: once input has ended (or the application stops
        feeding), a terminal event is synthesized within ``done_timeout`` of
        the last pipeline activity -- unless the application explicitly
        disabled every deadline.

        Yields:
            Events until a terminal event or a deadline fires.
        """
        # Per-iteration anchors for the activity-relative deadlines
        # (``done_timeout`` / ``max_idle``): these measure silence within the
        # active iteration, so they legitimately re-arm each time iteration
        # (re)starts.
        start = self._monotonic()
        last_event = start
        last_content = start
        # Absolute wall-clock origin for ``max_session_seconds``: anchored at
        # session establishment (:meth:`__aenter__`), NOT here, so the cap is a
        # true *session* cap (spec ST.6.1) -- the ``__aenter__``-to-first-event
        # gap counts, and a re-entered iterator cannot reset and renew it.
        # ``__aiter__`` enforces single iteration, so ``_session_started_at`` is
        # set on every reachable path; the ``or start`` is a defensive fallback
        # for a direct ``_iterate`` call outside ``async with``.
        session_start = self._session_started_at
        if session_start is None:  # pragma: no cover - defensive (always set)
            session_start = start
        while True:
            now = self._monotonic()
            remaining: list[float] = []
            if self._max_session_seconds is not None:
                session_left = self._max_session_seconds - (now - session_start)
                if session_left <= 0:
                    for tail in self._deadline_events("session_timeout"):
                        yield tail
                    return
                remaining.append(session_left)
            if self._done_timeout is not None:
                activity = max(last_event, self._last_audio_activity)
                remaining.append(self._done_timeout - (now - activity))
            if self._max_idle is not None:
                remaining.append(self._max_idle - (now - last_content))
            timeout = max(0.0, min(remaining)) if remaining else None
            try:
                event = await asyncio.wait_for(self._buffer.get(), timeout=timeout)
            except asyncio.TimeoutError:
                now = self._monotonic()
                if self._max_idle is not None and (now - last_content) >= self._max_idle:
                    code = "stream_stalled"
                elif (
                    self._max_session_seconds is not None
                    and (now - session_start) >= self._max_session_seconds
                ):
                    code = "session_timeout"
                elif (
                    self._done_timeout is not None
                    and (now - max(last_event, self._last_audio_activity)) >= self._done_timeout
                ):
                    code = "done_timeout"
                else:
                    # Audio consumption advanced the done anchor while we
                    # waited (live session in a silent stretch): nothing has
                    # actually expired -- re-arm against the fresh anchor.
                    continue
                for tail in self._deadline_events(code):
                    yield tail
                return
            if event is None:
                return
            last_event = self._monotonic()
            if event.is_content:
                last_content = last_event
            yield event
            if event.is_terminal:
                return

    def _terminate(self, terminal: TranscriptionEvent) -> TranscriptionEvent:
        """Run the bookkeeping every terminal-emission path MUST share.

        The single funnel for all terminal sites (engine-emitted terminal,
        clean-end ``done``, :meth:`_force_error`, and the deadline synthesis
        in :meth:`_deadline_events`), so a future terminal path cannot
        silently skip a step:

        * run the supersede-obligation sweep (idempotent) so
          :meth:`diagnostics` is complete on every exit path; and
        * release the audio-input side: a terminal also ends the producer --
          the audio queue's only drainer -- so without a release, an
          application feeder blocked in :meth:`send_audio` on the bounded
          queue would never wake (a deadlock inside ``async with session:``).

        Args:
            terminal: The terminal event about to be delivered.

        Returns:
            The terminal event, for delivery by the caller.
        """
        self._guard.finalize()
        self._release_audio_input()
        return terminal

    def _release_audio_input(self) -> None:
        """Release the put side of the bounded audio queue (terminal teardown).

        After a terminal, nothing consumes the audio queue anymore (the
        producer -- :meth:`audio_chunks`'s ``get`` loop -- is finished or
        cancelled), but a feeder may be blocked in ``await queue.put(...)``.
        Mark the input released and drain the queue: each drained item wakes
        one blocked putter (whose now-completing chunk is discarded -- the
        stream already carries the terminal), and the putter path
        (:meth:`_put_audio`) re-drains after its put completes, so a cascade
        of blocked putters all wake. Subsequent :meth:`send_audio` calls
        raise :class:`StreamClosedError` instead of blocking on a dead queue.
        """
        self._input_released = True
        while True:
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _deadline_events(self, code: str) -> list[TranscriptionEvent]:
        """Build the deadline-termination tail: drained backlog + terminal.

        A deadline (``done_timeout`` / ``stream_stalled`` /
        ``session_timeout``) ends iteration while the producer may still be
        running. Three things MUST happen, in order, for the stream == result
        invariant to hold:

        1. Cancel the producer/feed tasks so no FURTHER event reaches the
           reducer (cancellation is idempotent; ``__aexit__`` still awaits
           the tasks).
        2. Drain events already admitted to the reducer but still undelivered
           in the buffer: they are part of :meth:`result`, so the consumer
           must see them ahead of the terminal (the deadline is about engine
           silence, not about suppressing legally admitted work). The drain
           is synchronous and the producer is cancellation-pending, so the
           backlog cannot grow underneath it.
        3. Append the synthesized terminal through the shared
           :meth:`_terminate` funnel -- unless the backlog already ends in a
           REAL terminal (possible when the wall-clock cap fires at the top
           of the iteration loop with a terminal still buffered), which ends
           the stream itself; a second terminal after it would violate the
           single-terminal contract.

        Args:
            code: The synthesized terminal error code.

        Returns:
            The events to deliver, ending in exactly one terminal.
        """
        for task in (self._producer_task, self._feed_task):
            if task is not None and not task.done():
                task.cancel()
        events: list[TranscriptionEvent] = []
        for event in self._buffer.drain():
            events.append(event)
            if event.is_terminal:
                # The producer already ran the _terminate funnel for it.
                return events
        events.append(self._terminate(TranscriptionEvent.make_error(code=code, recoverable=False)))
        return events

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

        Raises:
            TypeError: If a subclass rebound a reserved base attribute (SF-1).
        """
        self._ensure_reserved_attrs_checked()
        self._initial_diagnostics = list(diagnostics)

    def diagnostics(self) -> list[Diagnostic]:
        """Return the diagnostics accumulated for this session so far.

        The lifecycle-suppression channel is **bounded** (spec ST.6.4): a
        misbehaving engine that trips a clamp on (nearly) every event cannot
        grow this list without limit over a long session. Once
        :data:`DEFAULT_MAX_GUARD_DIAGNOSTICS` entries accumulate, further
        suppressions are aggregated into a single trailing
        ``diagnostics_truncated`` summary (per-code counts) instead of being
        retained individually -- so the overflow is reported honestly, never
        silently dropped.

        Returns:
            The standard-layer parameter-gating / language diagnostics attached
            at session establishment, followed by the runtime's
            lifecycle-suppression diagnostics (suppressed illegal transitions or
            clamped ``stable_until`` values), capped as described above.
        """
        return [*self._initial_diagnostics, *self._guard.diagnostics]

    @property
    def done_timeout(self) -> float | None:
        """The configured pipeline-inactivity backstop in seconds.

        Returns:
            Max seconds without any pipeline activity (an event arriving or an
            audio chunk consumed via :meth:`audio_chunks`) before a
            ``done_timeout`` error is synthesized; ``None`` if disabled.
        """
        return self._done_timeout

    @property
    def max_idle(self) -> float | None:
        """The configured content-stall deadline in seconds.

        Returns:
            Max seconds without a content event before a ``stream_stalled``
            error is synthesized; ``None`` (the default) if disabled.
        """
        return self._max_idle

    @property
    def max_session_seconds(self) -> float | None:
        """The configured absolute wall-clock cap in seconds.

        Returns:
            Max session wall-clock seconds before a ``session_timeout`` error
            is synthesized; ``None`` (the default) if disabled.
        """
        return self._max_session_seconds

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
        self._thread = threading.Thread(
            target=self._run_loop, name=_SYNC_BRIDGE_LOOP_THREAD_NAME, daemon=True
        )
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

        Two dispatch paths by loop state. When the loop is still RUNNING on its
        thread (the normal case), outstanding tasks are cancelled cross-thread and
        the loop is stopped before the join. When the loop is already
        STOPPED-but-not-closed (the pump's "died" branch, where ``run_forever`` has
        returned), a cross-thread submit would never execute -- it would stall the
        join timeout and leave the cancel coroutine un-awaited and tasks pending
        (``-W error`` failures) -- so the thread is joined first and the still
        pending tasks are then drained synchronously on this thread via
        ``run_until_complete``.
        """
        if self._closed:
            return
        self._closed = True
        # Best-effort: cancel and await all outstanding tasks so nothing is
        # destroyed while pending. A truly blocking (non-awaiting) adapter can't be
        # cancelled cooperatively; the join timeout is the backstop for that case.
        if self._loop.is_running():
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
        else:
            # The loop already stopped (the pump's "died" branch): a cross-thread
            # submit would never run, so join the now-returning thread and drain the
            # pending tasks synchronously on this thread.
            self._thread.join(timeout=5.0)
            if not self._thread.is_alive():  # pragma: no branch
                self._loop.run_until_complete(_cancel_all_tasks())
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
            StreamClosedError: If the bridge was already torn down (a prior
                lifecycle call timed out); the coroutine cannot run anymore.
            TimeoutError: If the coroutine does not complete within ``timeout``.
                The background loop + thread are torn down before raising.
        """
        if self._closed:
            # The owned loop is already stopped/closed (a prior submit timed
            # out). Submitting would raise an unrelated RuntimeError("Event
            # loop is closed") and leave the coroutine un-awaited (a
            # RuntimeWarning under -W error). Close it and fail with the
            # contracted error instead.
            coro.close()
            raise StreamClosedError(
                "SyncSession is already torn down (a prior lifecycle call "
                "timed out and closed the bridge loop); this call cannot run."
            )
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

        Exception-safe: a context manager whose ``__enter__`` raises never
        receives ``__exit__``, so a failed enter MUST tear down the owned loop +
        thread started in ``__init__`` itself -- otherwise an adapter whose
        ``_open`` raises (bad credentials, unreachable host) would leak the
        bridge's background thread (spec ST.6.5: no leak). The timeout path is
        already torn down inside :meth:`_submit`; the ``entered`` guard also
        covers a non-timeout raise (``_open`` raising a regular exception).

        Returns:
            The sync session.

        Raises:
            TimeoutError: If the adapter ``_open`` hangs past ``submit_timeout``.
        """
        entered = False
        try:
            self._submit(self._session.__aenter__(), timeout=self._submit_timeout)
            self._aiter = self._session.__aiter__()
            entered = True
            return self
        finally:
            if not entered:
                self._shutdown()

    def __exit__(self, *exc: object) -> None:
        """Exit the async context and stop the owned loop (never leaks)."""
        if self._closed:
            # A prior lifecycle call timed out and already tore the loop down;
            # nothing is left that could run __aexit__. Returning here lets the
            # ORIGINAL TimeoutError propagate out of the ``with`` block instead
            # of masking it with an unrelated "Event loop is closed" error
            # (and avoids creating a never-awaited __aexit__ coroutine).
            return
        try:
            self._submit(self._session.__aexit__(*exc), timeout=self._submit_timeout)
        finally:
            self._shutdown()

    def feed(self, source: Iterable[bytes] | bytes | bytearray) -> None:
        """Feed audio from a managed source.

        Args:
            source: A sync iterable of byte chunks, or a single ``bytes`` /
                ``bytearray`` chunk.

        Raises:
            TypeError: If ``source`` is a ``str`` (forwarded from the async
                session: a ``str`` is byte-chunks-shaped only by accident -- a
                whole file goes to ``start_transcription(audio=...)``).
            InvalidSessionUseError: If manual input or a prior feed was already
                used (forwarded from the async session).
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

    def _loop_responsive(self) -> bool:
        """Whether the owned loop can still schedule work.

        A dead thread is NOT the only way the bridge stops making progress: an
        adapter that calls blocking (non-async) code mid-session freezes the
        loop while its thread stays alive, and the in-loop deadlines (spec
        ST.6.1) cannot fire on a frozen loop. The pump must detect that
        out-of-band or the sync caller hangs forever (spec ST.6.5 no-hang
        contract). The probe is a threadsafe no-op callback: an idle (merely
        quiet) loop runs it immediately; a frozen loop never does.

        Returns:
            ``True`` if the loop ran the probe within one poll slice.
        """
        probe = threading.Event()
        try:
            self._loop.call_soon_threadsafe(probe.set)
        except RuntimeError:  # pragma: no cover - loop closed mid-check
            return False
        return probe.wait(_SYNC_PUMP_POLL_SECONDS)

    def __iter__(self) -> Iterator[TranscriptionEvent]:
        """Iterate events synchronously.

        Event waits are unbounded by design: a live, fed session may
        legitimately go arbitrarily long between events (user silence), so the
        pump must never manufacture a timeout for it -- a stuck pipeline
        already surfaces as a terminal event from the async side's own
        deadlines (spec ST.6.1). The pump waits in short slices purely to
        detect the two failures those in-loop deadlines cannot report: the
        owned event-loop thread dying, and the loop being frozen by an
        adapter running blocking (non-async) code. Brief blocking stalls are
        tolerated (several consecutive unresponsive probes are required), so
        only a persistently frozen loop tears the bridge down.

        Yields:
            Events from the underlying async session.

        Raises:
            StreamClosedError: If the bridge was already torn down by a prior
                timed-out lifecycle call, or if the session was never entered
                (no event stream exists to pump).
            TimeoutError: If the owned event-loop thread died or stayed
                unresponsive, so no further event (or in-loop deadline) can
                ever be delivered.
        """
        if self._aiter is None:
            raise StreamClosedError(
                "SyncSession has no event stream to pump: enter the session before "
                "iterating; if a prior lifecycle call timed out, the bridge is "
                "already torn down."
            )
        while True:
            if self._closed:
                raise StreamClosedError(
                    "SyncSession is already torn down (a prior lifecycle call "
                    "timed out and closed the bridge loop); cannot pump events."
                )
            # The session's __aiter__ is an async generator, so __anext__ IS a
            # coroutine; the AsyncIterator protocol only promises Awaitable.
            anext_coro = cast("Coroutine[Any, Any, TranscriptionEvent]", self._aiter.__anext__())
            future = asyncio.run_coroutine_threadsafe(anext_coro, self._loop)
            unresponsive_probes = 0
            while True:
                try:
                    event = future.result(timeout=_SYNC_PUMP_POLL_SECONDS)
                    break
                except concurrent.futures.TimeoutError as exc:
                    if self._thread.is_alive() and self._loop_responsive():
                        unresponsive_probes = 0
                        continue
                    if self._thread.is_alive() and unresponsive_probes < 2:
                        # Tolerate a brief blocking stall (e.g. a sync model
                        # load inside the adapter): require consecutive failed
                        # probes before declaring the loop frozen.
                        unresponsive_probes += 1
                        continue
                    # Capture the failure mode BEFORE teardown: _shutdown joins
                    # the thread, so is_alive() afterwards always reports dead.
                    frozen = self._thread.is_alive()
                    future.cancel()
                    self._shutdown()
                    raise TimeoutError(
                        "SyncSession event pump aborted: the bridge's "
                        "event-loop thread "
                        + ("is frozen by blocking adapter code" if frozen else "died")
                        + ", so no further event or in-loop deadline can be "
                        "delivered."
                    ) from exc
                except StopAsyncIteration:
                    return
            yield event

    def result(self) -> TranscriptionResult:
        """Reduce the session so far into a transcription result.

        Returns:
            The reduced result.
        """
        return self._session.result()

    def diagnostics(self) -> list[Diagnostic]:
        """Return the session's standard-layer and lifecycle diagnostics.

        Mirrors :meth:`TranscriptionSession.diagnostics` (forwarded directly, like
        :meth:`result`) so a synchronously-driven session exposes the same
        parameter-gating / language-resolution / lifecycle-suppression
        diagnostics as the async one. Without this the sync bridge would silently
        drop a first-class, compliance-checked part of the session surface (spec
        ST.6.5: the sync bridge is a faithful mirror of the async session).

        Returns:
            The accumulated diagnostics.
        """
        return self._session.diagnostics()

    def is_loop_alive(self) -> bool:
        """Whether the owned background event-loop thread is still running.

        The bridge starts a dedicated event-loop thread in ``__init__`` and MUST
        tear it down on close (``__exit__``) or on a failed ``__enter__`` (spec
        ST.6.5: from an external thread, no leak). After a clean lifecycle this
        returns ``False``; a ``True`` here once the session is closed is a leaked
        loop thread. Exposed so the sync-bridge compliance check can assert on
        *this* bridge's own thread rather than diffing the whole process thread
        set (which would mis-flag a dependency's benign daemon thread as a leak).

        Returns:
            ``True`` if the owned loop thread is alive.
        """
        return self._thread.is_alive()


__all__ = [
    "DEFAULT_AUDIO_HISTORY_MAXLEN",
    "DEFAULT_AUDIO_QUEUE_MAXSIZE",
    "DEFAULT_DONE_TIMEOUT",
    "DEFAULT_EVENT_BUFFER_CAPACITY",
    "DEFAULT_MAX_GUARD_DIAGNOSTICS",
    "DEFAULT_MAX_IDLE",
    "DEFAULT_MAX_SESSION_SECONDS",
    "EventType",
    "StreamDeadlines",
    "StreamReducer",
    "SyncSession",
    "TranscriptionEvent",
    "TranscriptionSession",
    "reduce_event",
    "validate_stable_until",
]
