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
  ownership, backpressure-aware iteration, a done-timeout, and result reduction.
* :class:`SyncSession` -- the standard sync bridge (one background event loop in
  a thread, owned by the session), so authors only ever write async.
"""

from __future__ import annotations

import asyncio
import threading
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .exceptions import StreamClosedError
from .results import Segment, TranscriptionResult, Word

EventType = Literal["partial", "final", "supersede", "progress", "done", "error"]

#: Default seconds to wait for a terminal ``done`` event before synthesizing an
#: error (the iterator MUST always terminate; spec ST.6.1).
DEFAULT_DONE_TIMEOUT = 30.0


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
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def stable_text(self) -> str:
        """The frozen prefix of ``text`` (``text[:stable_until]``).

        Returns:
            The frozen prefix, or ``""`` if nothing is frozen.
        """
        if self.stable_until and self.text is not None:
            return self.text[: self.stable_until]
        return ""

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
        return cls(
            type="final", segment_id=segment_id, text=text, finality="closed", **kw
        )

    @classmethod
    def supersede(
        cls, old_ids: list[str], new_ids: list[str], **kw: Any
    ) -> TranscriptionEvent:
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
    def make_error(
        cls, code: str, *, recoverable: bool = False, **kw: Any
    ) -> TranscriptionEvent:
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
    """

    def __init__(self) -> None:
        """Initialize an empty reducer."""
        self._segments: dict[str, Segment] = {}
        self._order: list[str] = []
        self._detected_language: str | None = None

    def add(self, event: TranscriptionEvent) -> None:
        """Incorporate one event into the running result.

        Args:
            event: The event to incorporate.
        """
        if event.type == "final" and event.segment_id is not None:
            if event.segment_id not in self._segments:
                self._order.append(event.segment_id)
            self._segments[event.segment_id] = Segment(
                start=event.start or 0.0,
                end=event.end or (event.start or 0.0),
                text=event.text or "",
                words=event.words,
            )
        elif event.type == "supersede":
            for old_id in event.old_ids:
                if old_id in self._segments:
                    del self._segments[old_id]
                    self._order.remove(old_id)

    def result(self) -> TranscriptionResult:
        """Build the reduced transcription result.

        Returns:
            A :class:`TranscriptionResult` from the committed segments, ordered
            by start time.
        """
        segments = [self._segments[sid] for sid in self._order]
        segments.sort(key=lambda s: s.start)
        text = " ".join(s.text for s in segments).strip()
        return TranscriptionResult(
            text=text,
            segments=segments or None,
            detected_language=self._detected_language,
        )


class _CoalescingBuffer:
    """An async event buffer with partial coalescing (spec ST.6.4 backpressure).

    Pending ``partial`` events are merged per ``segment_id`` (latest wins); a
    same-segment ``final`` / ``closed`` / ``supersede`` invalidates a pending
    partial so a replaced segment never revives. ``final`` / ``supersede`` /
    ``done`` / ``error`` are never dropped or reordered.
    """

    def __init__(self) -> None:
        """Initialize the buffer."""
        self._items: list[TranscriptionEvent] = []
        self._partial_index: dict[str, int] = {}
        self._event = asyncio.Event()
        self._closed = False

    def put(self, event: TranscriptionEvent) -> None:
        """Add an event, coalescing superseded partials.

        Args:
            event: The event to enqueue.
        """
        if event.type == "partial" and event.segment_id is not None:
            idx = self._partial_index.get(event.segment_id)
            if idx is not None and idx < len(self._items):
                self._items[idx] = event
                self._event.set()
                return
            self._partial_index[event.segment_id] = len(self._items)
        else:
            # A terminal-for-segment event invalidates any pending partial.
            sid = event.segment_id
            if event.type in ("final", "supersede"):
                for old in [sid, *event.old_ids]:
                    if old is not None:
                        self._partial_index.pop(old, None)
        self._items.append(event)
        self._event.set()

    def close(self) -> None:
        """Signal that no further events will be added."""
        self._closed = True
        self._event.set()

    async def get(self) -> TranscriptionEvent | None:
        """Pop the next event, awaiting one if necessary.

        Returns:
            The next event, or ``None`` once closed and drained.
        """
        while True:
            if self._items:
                item = self._items.pop(0)
                # Reindex remaining pending partials.
                self._partial_index = {
                    e.segment_id: i
                    for i, e in enumerate(self._items)
                    if e.type == "partial" and e.segment_id is not None
                }
                return item
            if self._closed:
                return None
            self._event.clear()
            await self._event.wait()


class TranscriptionSession(ABC):
    """Async-first, full-duplex streaming session base.

    Authors implement :meth:`_produce` (and optionally :meth:`_open` /
    :meth:`_close`), reading fed audio via :meth:`audio_chunks` and yielding
    :class:`TranscriptionEvent` objects. The base manages input ownership,
    backpressure, the done-timeout, and result reduction.
    """

    def __init__(self, *, done_timeout: float = DEFAULT_DONE_TIMEOUT) -> None:
        """Initialize the session.

        Args:
            done_timeout: Seconds to await a terminal event before synthesizing
                a ``done_timeout`` error.
        """
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._buffer = _CoalescingBuffer()
        self._mode: Literal["feed", "manual"] | None = None
        self._ended = False
        self._done_timeout = done_timeout
        self._feed_task: asyncio.Task[None] | None = None
        self._producer_task: asyncio.Task[None] | None = None
        self._reducer = StreamReducer()

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

        Yields:
            Raw audio chunks in the session's declared format.
        """
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                return
            yield chunk

    # ----- input ownership ------------------------------------------------- #
    def feed(self, source: Iterable[bytes] | AsyncIterator[bytes]) -> None:
        """Feed audio from a managed source (mutually exclusive with manual).

        Args:
            source: A sync or async iterable of audio chunks.

        Raises:
            StreamClosedError: If manual input was already used.
        """
        if self._mode == "manual":
            raise StreamClosedError("send_audio/end_audio already used; cannot feed().")
        self._mode = "feed"
        self._feed_task = asyncio.ensure_future(self._drain_source(source))

    async def _drain_source(
        self, source: Iterable[bytes] | AsyncIterator[bytes]
    ) -> None:
        """Drain a fed source into the audio queue, ending on exhaustion.

        Args:
            source: The audio source.
        """
        try:
            if isinstance(source, AsyncIterator):
                async for chunk in source:
                    await self._audio_queue.put(chunk)
            else:
                for chunk in source:
                    await self._audio_queue.put(chunk)
        finally:
            await self._audio_queue.put(None)
            self._ended = True

    async def send_audio(self, chunk: bytes) -> None:
        """Manually send one audio chunk (mutually exclusive with ``feed``).

        Args:
            chunk: The audio chunk.

        Raises:
            StreamClosedError: If ``feed`` was used or the input was ended.
        """
        if self._mode == "feed":
            raise StreamClosedError("feed() already used; cannot send_audio().")
        if self._ended:
            raise StreamClosedError("Cannot send_audio after end_audio().")
        self._mode = "manual"
        await self._audio_queue.put(chunk)

    async def end_audio(self) -> None:
        """Mark the end of manual audio input (idempotent in manual mode).

        Raises:
            StreamClosedError: If ``feed`` was used.
        """
        if self._mode == "feed":
            raise StreamClosedError("feed() manages end_audio(); do not call it.")
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
        """Tear down the producer, feed task, and engine resources."""
        if self._producer_task is not None:
            self._producer_task.cancel()
        if self._feed_task is not None:
            self._feed_task.cancel()
        await self._close()

    async def _run_producer(self) -> None:
        """Drive ``_produce``, appending a terminal ``done`` or ``error``."""
        try:
            async for event in self._produce():
                self._reducer.add(event)
                self._buffer.put(event)
            self._buffer.put(TranscriptionEvent.done())
        except asyncio.CancelledError:  # pragma: no cover - teardown path
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as an error event
            self._buffer.put(
                TranscriptionEvent.make_error(
                    code="engine_error", recoverable=False, extra={"detail": str(exc)}
                )
            )
        finally:
            self._buffer.close()

    def __aiter__(self) -> AsyncIterator[TranscriptionEvent]:
        """Return the event async iterator.

        Returns:
            ``self`` as an async iterator.
        """
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[TranscriptionEvent]:
        """Yield events with a done-timeout safeguard.

        Yields:
            Events until a terminal event or the done-timeout fires.
        """
        while True:
            try:
                event = await asyncio.wait_for(
                    self._buffer.get(), timeout=self._done_timeout
                )
            except asyncio.TimeoutError:
                yield TranscriptionEvent.make_error(
                    code="done_timeout", recoverable=False
                )
                return
            if event is None:
                return
            yield event
            if event.is_terminal:
                return

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
    """

    def __init__(self, session: TranscriptionSession) -> None:
        """Wrap an async session.

        Args:
            session: The async session to drive.
        """
        self._session = session
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._aiter: AsyncIterator[TranscriptionEvent] | None = None

    def _run_loop(self) -> None:
        """Run the owned event loop until stopped."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro: Any) -> Any:
        """Run a coroutine on the owned loop and wait for its result.

        Args:
            coro: The coroutine to run.

        Returns:
            The coroutine's result.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def __enter__(self) -> SyncSession:
        """Enter the async session's context.

        Returns:
            The sync session.
        """
        self._submit(self._session.__aenter__())
        self._aiter = self._session.__aiter__()
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the async context and stop the owned loop."""
        self._submit(self._session.__aexit__(*exc))
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    def feed(self, source: Iterable[bytes]) -> None:
        """Feed audio from a managed source.

        Args:
            source: A sync iterable of audio chunks.
        """

        async def _do_feed() -> None:
            self._session.feed(source)

        self._submit(_do_feed())

    def send_audio(self, chunk: bytes) -> None:
        """Manually send one audio chunk.

        Args:
            chunk: The audio chunk.
        """
        self._submit(self._session.send_audio(chunk))

    def end_audio(self) -> None:
        """Mark the end of manual audio input."""
        self._submit(self._session.end_audio())

    def __iter__(self) -> Iterator[TranscriptionEvent]:
        """Iterate events synchronously.

        Yields:
            Events from the underlying async session.
        """
        assert self._aiter is not None
        while True:
            try:
                yield self._submit(self._aiter.__anext__())
            except StopAsyncIteration:
                return

    def result(self) -> TranscriptionResult:
        """Reduce the session so far into a transcription result.

        Returns:
            The reduced result.
        """
        return self._session.result()


__all__ = [
    "DEFAULT_DONE_TIMEOUT",
    "EventType",
    "StreamReducer",
    "SyncSession",
    "TranscriptionEvent",
    "TranscriptionSession",
    "reduce_event",
    "validate_stable_until",
]
