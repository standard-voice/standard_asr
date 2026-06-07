# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the streaming protocol: events, reduce, session, sync bridge."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from standard_asr.exceptions import StreamClosedError
from standard_asr.streaming import (
    EventBufferOverflow,
    StreamReducer,
    SyncSession,
    TranscriptionEvent,
    TranscriptionSession,
    _CoalescingBuffer,  # pyright: ignore[reportPrivateUsage]
    _LifecycleGuard,  # pyright: ignore[reportPrivateUsage]
    reduce_event,
    validate_stable_until,
)


# --------------------------------------------------------------------------- #
# stable_until invariant
# --------------------------------------------------------------------------- #
def test_validate_stable_until_bounds() -> None:
    assert validate_stable_until("hello", 0) is True
    assert validate_stable_until("hello", 5) is True
    assert validate_stable_until("hello", 3) is True
    assert validate_stable_until("hello", -1) is False
    assert validate_stable_until("hello", 6) is False


def test_validate_stable_until_combining() -> None:
    # "e" + combining acute accent: cutting before the accent is invalid.
    text = "éx"
    assert validate_stable_until(text, 1) is False
    assert validate_stable_until(text, 2) is True


def test_stable_text_property() -> None:
    ev = TranscriptionEvent.partial("s0", "hello world", stable_until=5)
    assert ev.stable_text == "hello"
    assert TranscriptionEvent.partial("s0", "x").stable_text == ""


# --------------------------------------------------------------------------- #
# event model
# --------------------------------------------------------------------------- #
def test_supersede_disjoint_enforced() -> None:
    with pytest.raises(ValueError):
        TranscriptionEvent.supersede(["a"], ["a"])


def test_is_terminal() -> None:
    assert TranscriptionEvent.done().is_terminal is True
    assert TranscriptionEvent.make_error("x", recoverable=False).is_terminal is True
    assert TranscriptionEvent.make_error("x", recoverable=True).is_terminal is False
    assert TranscriptionEvent.partial("s", "t").is_terminal is False


def test_closed_finality() -> None:
    ev = TranscriptionEvent.closed("s0", "Hello.")
    assert ev.type == "final"
    assert ev.finality == "closed"


# --------------------------------------------------------------------------- #
# reduce
# --------------------------------------------------------------------------- #
def test_reduce_event_partial_final_supersede() -> None:
    segs: dict[str, str] = {}
    reduce_event(segs, TranscriptionEvent.partial("s1", "hel"))
    reduce_event(segs, TranscriptionEvent.final("s1", "hello"))
    assert segs == {"s1": "hello"}
    reduce_event(segs, TranscriptionEvent.final("s2", "world"))
    reduce_event(segs, TranscriptionEvent.supersede(["s1", "s2"], ["s3"]))
    assert segs == {}
    reduce_event(segs, TranscriptionEvent.final("s3", "hello world"))
    assert segs == {"s3": "hello world"}


def test_stream_reducer_result() -> None:
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "hello", start=0.0, end=1.0))
    reducer.add(TranscriptionEvent.final("s2", "world", start=1.0, end=2.0))
    result = reducer.result()
    assert result.text == "hello world"
    assert result.segments is not None and len(result.segments) == 2


def test_stream_reducer_supersede_removes() -> None:
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "wrong", start=0.0, end=1.0))
    reducer.add(TranscriptionEvent.supersede(["s1"], ["s2"]))
    reducer.add(TranscriptionEvent.final("s2", "right", start=0.0, end=1.0))
    assert reducer.result().text == "right"


# --------------------------------------------------------------------------- #
# session
# --------------------------------------------------------------------------- #
class _EchoSession(TranscriptionSession):
    """Emits a partial then final per fed chunk; supports backpressure tests."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        index = 0
        async for chunk in self.audio_chunks():
            sid = f"seg-{index}"
            text = chunk.decode()
            yield TranscriptionEvent.partial(sid, text[:1])
            yield TranscriptionEvent.final(sid, text, start=float(index), end=float(index + 1))
            index += 1


async def _collect(session: TranscriptionSession) -> list[TranscriptionEvent]:
    events: list[TranscriptionEvent] = []
    async with session:
        async for event in session:
            events.append(event)
    return events


def test_session_feed_mode() -> None:
    async def run() -> list[TranscriptionEvent]:
        session = _EchoSession()
        session.feed([b"abc", b"de"])
        return await _collect(session)

    events = asyncio.run(run())
    types = [e.type for e in events]
    assert types[-1] == "done"
    finals = [e for e in events if e.type == "final"]
    assert {f.text for f in finals} == {"abc", "de"}


def test_session_manual_mode_and_result() -> None:
    async def run() -> tuple[list[TranscriptionEvent], str]:
        session = _EchoSession()
        async with session:
            await session.send_audio(b"hello")
            await session.send_audio(b"world")
            await session.end_audio()
            events = [e async for e in session]
        return events, session.result().text

    events, text = asyncio.run(run())
    assert events[-1].type == "done"
    assert text == "hello world"


def test_session_feed_then_manual_raises() -> None:
    async def run() -> None:
        session = _EchoSession()
        session.feed([b"x"])
        async with session:
            await session.send_audio(b"y")

    with pytest.raises(StreamClosedError):
        asyncio.run(run())


def test_session_manual_then_feed_raises() -> None:
    async def run() -> None:
        session = _EchoSession()
        async with session:
            await session.send_audio(b"y")
            session.feed([b"x"])

    with pytest.raises(StreamClosedError):
        asyncio.run(run())


def test_session_send_after_end_raises() -> None:
    async def run() -> None:
        session = _EchoSession()
        async with session:
            await session.send_audio(b"y")
            await session.end_audio()
            await session.send_audio(b"z")

    with pytest.raises(StreamClosedError):
        asyncio.run(run())


def test_session_end_audio_idempotent_manual() -> None:
    async def run() -> None:
        session = _EchoSession()
        async with session:
            await session.send_audio(b"y")
            await session.end_audio()
            await session.end_audio()  # idempotent
            _ = [e async for e in session]

    asyncio.run(run())


def test_session_done_timeout() -> None:
    class _HangSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(10)
            yield TranscriptionEvent.done()  # pragma: no cover

    async def run() -> list[TranscriptionEvent]:
        session = _HangSession(done_timeout=0.05)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "done_timeout"


def test_session_producer_error_surfaced() -> None:
    class _BoomSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            raise RuntimeError("boom")
            yield TranscriptionEvent.done()  # pragma: no cover

    async def run() -> list[TranscriptionEvent]:
        session = _BoomSession()
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "engine_error"


# --------------------------------------------------------------------------- #
# sync bridge
# --------------------------------------------------------------------------- #
def test_sync_bridge_feed() -> None:
    with SyncSession(_EchoSession()) as sync:
        sync.feed([b"abc", b"de"])
        events = list(sync)
    assert events[-1].type == "done"
    assert sync.result().text in ("abc de", "de abc") or "abc" in sync.result().text


def test_sync_bridge_manual() -> None:
    with SyncSession(_EchoSession()) as sync:
        sync.send_audio(b"hi")
        sync.end_audio()
        events = list(sync)
    finals = [e for e in events if e.type == "final"]
    assert finals[0].text == "hi"


# --------------------------------------------------------------------------- #
# C5 -- coalescing buffer: stale partial dropped by terminal-for-segment event
# --------------------------------------------------------------------------- #
async def _drain_buffer(buf: _CoalescingBuffer) -> list[TranscriptionEvent]:
    out: list[TranscriptionEvent] = []
    while True:
        ev = await buf.get()
        if ev is None:
            return out
        out.append(ev)


def test_coalescing_partial_dropped_by_same_segment_final() -> None:
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s0", "hel"))  # pending, not delivered
        buf.put(TranscriptionEvent.final("s0", "hello"))  # invalidates the partial
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    # The stale partial MUST be dropped; only the final survives. No partial
    # may be delivered AFTER the final (would revive a dead segment).
    assert [e.type for e in events] == ["final"]
    assert events[0].text == "hello"


def test_coalescing_partial_dropped_by_supersede_old_ids() -> None:
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s1", "aaa"))
        buf.put(TranscriptionEvent.partial("s2", "bbb"))
        # supersede retires s1 and s2 -> both pending partials MUST be dropped.
        buf.put(TranscriptionEvent.supersede(["s1", "s2"], ["s3"]))
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    assert [e.type for e in events] == ["supersede"]


def test_coalescing_partial_dropped_by_closed() -> None:
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s0", "draft"))
        buf.put(TranscriptionEvent.closed("s0", "Final."))  # closed = final variant
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    assert [e.type for e in events] == ["final"]
    assert events[0].finality == "closed"


def test_coalescing_latest_partial_wins() -> None:
    async def run() -> list[TranscriptionEvent]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s0", "a"))
        buf.put(TranscriptionEvent.partial("s0", "ab"))
        buf.put(TranscriptionEvent.partial("s0", "abc"))
        buf.close()
        return await _drain_buffer(buf)

    events = asyncio.run(run())
    assert [e.text for e in events] == ["abc"]


def test_coalescing_partial_after_delivery_starts_fresh_slot() -> None:
    async def run() -> list[str | None]:
        buf = _CoalescingBuffer()
        buf.put(TranscriptionEvent.partial("s0", "a"))
        first = await buf.get()  # delivers and frees the slot
        buf.put(TranscriptionEvent.partial("s0", "b"))  # new pending slot
        buf.put(TranscriptionEvent.final("s0", "final"))  # drops the "b" partial
        buf.close()
        rest = await _drain_buffer(buf)
        return [first.text if first else None, *[e.text for e in rest]]

    texts = asyncio.run(run())
    assert texts == ["a", "final"]


# --------------------------------------------------------------------------- #
# C6 -- bounded buffers
# --------------------------------------------------------------------------- #
def test_event_buffer_overflow_raises() -> None:
    buf = _CoalescingBuffer(capacity=2)
    buf.put(TranscriptionEvent.final("s0", "a"))
    buf.put(TranscriptionEvent.final("s1", "b"))
    with pytest.raises(EventBufferOverflow):
        buf.put(TranscriptionEvent.final("s2", "c"))


def test_event_buffer_coalesced_partial_does_not_grow() -> None:
    buf = _CoalescingBuffer(capacity=1)
    buf.put(TranscriptionEvent.partial("s0", "a"))
    # Re-coalescing the same segment reuses the slot, never overflows.
    buf.put(TranscriptionEvent.partial("s0", "ab"))
    buf.put(TranscriptionEvent.partial("s0", "abc"))


def test_put_forced_bypasses_capacity() -> None:
    buf = _CoalescingBuffer(capacity=1)
    buf.put(TranscriptionEvent.final("s0", "a"))
    # Terminal events must always land even at capacity.
    buf.put_forced(TranscriptionEvent.done())


def test_session_backpressure_overflow_emits_error() -> None:
    class _FloodSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            # Many distinct finals (never coalesced) overflow a tiny buffer.
            for i in range(50):
                yield TranscriptionEvent.final(f"s{i}", "x")

    async def run() -> list[TranscriptionEvent]:
        # Consumer never reads until producer is done -> buffer fills.
        session = _FloodSession(event_buffer_capacity=4)
        session.feed([])
        async with session:
            await asyncio.sleep(0.05)  # let producer run ahead and overflow
            return [e async for e in session]

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "backpressure"
    assert events[-1].recoverable is False


def test_audio_queue_is_bounded() -> None:
    async def run() -> None:
        session = _EchoSession(audio_queue_maxsize=2)
        assert session._audio_queue.maxsize == 2  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# C7 + teardown -- sync bridge timeout / no leak
# --------------------------------------------------------------------------- #
class _HangOpenSession(TranscriptionSession):
    async def _open(self) -> None:
        await asyncio.sleep(100)

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        yield TranscriptionEvent.done()  # pragma: no cover


def test_sync_bridge_open_timeout_no_deadlock() -> None:
    sync = SyncSession(_HangOpenSession(), submit_timeout=0.1)
    with pytest.raises(TimeoutError):
        sync.__enter__()
    # Background thread must be torn down, not leaked.
    assert sync._thread.is_alive() is False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_sync_bridge_exit_always_shuts_down() -> None:
    sync = SyncSession(_EchoSession(), submit_timeout=5.0)
    sync.__enter__()
    sync.feed([b"hi"])
    list(sync)
    sync.__exit__(None, None, None)
    assert sync._thread.is_alive() is False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------------------------------- #
# H10 -- reconnect scaffolding
# --------------------------------------------------------------------------- #
class _ReconnectSession(TranscriptionSession):
    """Drains all audio, notes a reconnect, then finalizes -- continuity test."""

    def __init__(self, gap: tuple[float, float], **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._gap = gap

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        chunks: list[bytes] = []
        async for chunk in self.audio_chunks():
            chunks.append(chunk)
        # Simulate the adapter re-establishing, replaying the rolling buffer,
        # and signalling the bridged gap after audio has been processed.
        _ = self.replay_buffer()
        self.note_reconnect(self._gap[0], self._gap[1])
        yield TranscriptionEvent.final("seg-0", b"".join(chunks).decode(), start=0.0)


def test_reconnect_emits_progress_replayable_no_content_lost() -> None:
    async def run() -> list[TranscriptionEvent]:
        session = _ReconnectSession((1.0, 2.0))
        session.feed([b"ab", b"cd"])  # list -> replayable
        assert session.replayable is True
        return await _collect(session)

    events = asyncio.run(run())
    progress = [e for e in events if e.type == "progress"]
    assert len(progress) == 1
    assert progress[0].reconnect is True
    assert progress[0].gap_start == 1.0 and progress[0].gap_end == 2.0
    # Replayable source -> NO content_lost error.
    assert not any(e.type == "error" for e in events)
    assert events[-1].type == "done"


def test_reconnect_nonreplayable_overflow_emits_content_lost() -> None:
    async def run() -> list[TranscriptionEvent]:
        # Async generator source -> non-replayable; tiny history -> overflow.
        async def gen() -> AsyncIterator[bytes]:
            for _ in range(5):
                yield b"x"

        session = _ReconnectSession((1.0, 2.0), audio_history_maxlen=1)
        session.feed(gen())
        assert session.replayable is False
        return await _collect(session)

    events = asyncio.run(run())
    progress = [e for e in events if e.type == "progress" and e.reconnect]
    errors = [e for e in events if e.type == "error"]
    assert len(progress) == 1
    # content_lost MUST follow the reconnect progress for a lossy live source.
    assert any(e.code == "content_lost" for e in errors)
    cl = next(e for e in errors if e.code == "content_lost")
    assert cl.recoverable is False
    # progress precedes the content_lost error in delivery order.
    assert events.index(progress[0]) < events.index(cl)


# --------------------------------------------------------------------------- #
# H11 -- lifecycle enforcement + stable_until monotonicity
# --------------------------------------------------------------------------- #
def test_guard_suppresses_partial_after_final() -> None:
    guard = _LifecycleGuard()
    assert guard.admit(TranscriptionEvent.final("s0", "done")) is not None
    assert guard.admit(TranscriptionEvent.partial("s0", "oops")) is None
    assert any(d.code == "lifecycle_partial_after_final" for d in guard.diagnostics)


def test_guard_suppresses_events_after_closed() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("s0", "x"))
    guard.admit(TranscriptionEvent.closed("s0", "X."))
    assert guard.admit(TranscriptionEvent.partial("s0", "y")) is None
    assert guard.admit(TranscriptionEvent.final("s0", "z")) is None


def test_guard_suppresses_closed_in_supersede_old_ids() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("s0", "x"))
    guard.admit(TranscriptionEvent.closed("s0", "X."))
    # closed segment MUST NOT appear in a later supersede old_ids.
    assert guard.admit(TranscriptionEvent.supersede(["s0"], ["s1"])) is None
    assert any(d.code == "lifecycle_closed_superseded" for d in guard.diagnostics)


def test_guard_strict_raises() -> None:
    guard = _LifecycleGuard(strict=True)
    guard.admit(TranscriptionEvent.final("s0", "x"))
    with pytest.raises(ValueError):
        guard.admit(TranscriptionEvent.partial("s0", "y"))


def test_guard_clamps_decreasing_stable_until() -> None:
    guard = _LifecycleGuard()
    ev1 = guard.admit(TranscriptionEvent.partial("s0", "hello", stable_until=4))
    assert ev1 is not None and ev1.stable_until == 4
    ev2 = guard.admit(TranscriptionEvent.partial("s0", "hello", stable_until=2))
    assert ev2 is not None and ev2.stable_until == 4  # clamped up to prior
    assert any(d.code == "stable_until_clamped" for d in guard.diagnostics)


def test_guard_clamps_invalid_combining_boundary() -> None:
    guard = _LifecycleGuard()
    # "é" as e + combining accent; cutting at 1 splits the combining sequence.
    ev = guard.admit(TranscriptionEvent.partial("s0", "éx", stable_until=1))
    assert ev is not None and ev.stable_until == 0


def test_guard_supersede_new_ids_open_then_partial_allowed() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("s0", "x"))
    guard.admit(TranscriptionEvent.supersede(["s0"], ["s1"]))
    # s1 was started open by supersede; a partial for it is legal.
    assert guard.admit(TranscriptionEvent.partial("s1", "new")) is not None


def test_session_suppresses_illegal_transition_in_stream() -> None:
    class _BadSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.final("s0", "good")
            yield TranscriptionEvent.partial("s0", "revived")  # illegal

    async def run() -> tuple[list[TranscriptionEvent], int]:
        session = _BadSession()
        session.feed([])
        events = await _collect(session)
        return events, len(session.diagnostics())

    events, ndiag = asyncio.run(run())
    # The revived partial must NOT be forwarded.
    assert not any(e.type == "partial" for e in events)
    assert ndiag >= 1


def test_stable_text_guards_invalid_stable_until() -> None:
    # Negative / out-of-range stable_until must not produce a wrong prefix.
    assert TranscriptionEvent.partial("s", "hello", stable_until=-2).stable_text == ""
    assert TranscriptionEvent.partial("s", "hi", stable_until=99).stable_text == "hi"


# --------------------------------------------------------------------------- #
# H12 -- termination guarantees (idle / wall clock) beyond per-event gap
# --------------------------------------------------------------------------- #
def test_heartbeat_only_engine_still_terminates() -> None:
    class _HeartbeatSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            while True:
                await asyncio.sleep(0.01)
                yield TranscriptionEvent.progress(audio_processed_until=1.0)

    async def run() -> list[TranscriptionEvent]:
        # Frequent heartbeats keep done_timeout alive, but max_idle (no content)
        # MUST still terminate the iterator.
        session = _HeartbeatSession(done_timeout=5.0, max_idle=0.1)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "stream_stalled"


def test_max_session_seconds_caps_wall_time() -> None:
    class _ChattySession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            i = 0
            while True:
                await asyncio.sleep(0.01)
                yield TranscriptionEvent.final(f"s{i}", "x")
                i += 1

    async def run() -> list[TranscriptionEvent]:
        # Continuous content events keep both done_timeout and max_idle alive;
        # only the wall-clock cap guarantees termination.
        session = _ChattySession(done_timeout=5.0, max_idle=5.0, max_session_seconds=0.1)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "session_timeout"


def test_done_timeout_still_fires_on_total_silence() -> None:
    class _SilentSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(10)
            yield TranscriptionEvent.done()  # pragma: no cover

    async def run() -> list[TranscriptionEvent]:
        session = _SilentSession(done_timeout=0.05, max_idle=5.0)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].code == "done_timeout"


# --------------------------------------------------------------------------- #
# MEDIUM -- end_audio() first then feed() rejected (mode claimed atomically)
# --------------------------------------------------------------------------- #
def test_end_audio_first_then_feed_raises() -> None:
    async def run() -> None:
        session = _EchoSession()
        async with session:
            await session.end_audio()  # claims manual mode
            session.feed([b"x"])  # mixing -> must raise

    with pytest.raises(StreamClosedError):
        asyncio.run(run())


def test_feed_twice_raises() -> None:
    async def run() -> None:
        session = _EchoSession()
        session.feed([b"a"])
        session.feed([b"b"])

    with pytest.raises(StreamClosedError):
        asyncio.run(run())


# --------------------------------------------------------------------------- #
# MEDIUM -- StreamReducer: no fabricated 0.0 timestamps; arrival order kept
# --------------------------------------------------------------------------- #
def test_reducer_preserves_arrival_order_without_timestamps() -> None:
    reducer = StreamReducer()
    # No start/end given (timestamp-less engine like Qwen streaming).
    reducer.add(TranscriptionEvent.final("s1", "world"))
    reducer.add(TranscriptionEvent.final("s2", "hello"))
    result = reducer.result()
    # Arrival order preserved (NOT re-sorted to 0.0 == 0.0 ambiguity).
    assert result.text == "world hello"


def test_reducer_sorts_when_all_have_timestamps() -> None:
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "second", start=5.0, end=6.0))
    reducer.add(TranscriptionEvent.final("s2", "first", start=1.0, end=2.0))
    assert reducer.result().text == "first second"


def test_reducer_no_sort_when_mixed_timestamps() -> None:
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "b", start=5.0, end=6.0))
    reducer.add(TranscriptionEvent.final("s2", "a"))  # no timestamp
    # Mixed -> preserve arrival order, do not sort on a fabricated 0.0.
    assert reducer.result().text == "b a"


# --------------------------------------------------------------------------- #
# LOW -- coalescing buffer drains O(n) without per-pop reindex
# --------------------------------------------------------------------------- #
def test_coalescing_buffer_large_drain_order() -> None:
    async def run() -> list[str | None]:
        buf = _CoalescingBuffer(capacity=10_000)
        for i in range(5000):
            buf.put(TranscriptionEvent.final(f"s{i}", str(i)))
        buf.close()
        events = await _drain_buffer(buf)
        return [e.text for e in events]

    texts = asyncio.run(run())
    assert texts == [str(i) for i in range(5000)]


# --------------------------------------------------------------------------- #
# SURVEY -- WeNet two-pass supersede preserves frozen prefix; DSM; FireRed
# --------------------------------------------------------------------------- #
def test_survey_wenet_two_pass_supersede_reduce() -> None:
    # First pass finalizes seg-3/seg-4; second pass merges into seg-5.
    segs: dict[str, str] = {}
    reduce_event(segs, TranscriptionEvent.final("seg-3", "hello"))
    reduce_event(segs, TranscriptionEvent.final("seg-4", "world"))
    reduce_event(segs, TranscriptionEvent.supersede(["seg-3", "seg-4"], ["seg-5"]))
    assert segs == {}
    reduce_event(segs, TranscriptionEvent.final("seg-5", "hello world"))
    assert segs == {"seg-5": "hello world"}


def test_survey_fireredasr_no_interim_only_finals() -> None:
    # no_interim engine: each segment emits exactly one final, no partial.
    class _NoInterim(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            async for chunk in self.audio_chunks():
                yield TranscriptionEvent.final("seg-0", chunk.decode(), start=0.0)

    async def run() -> list[TranscriptionEvent]:
        session = _NoInterim()
        session.feed([b"sentence"])
        return await _collect(session)

    events = asyncio.run(run())
    assert not any(e.type == "partial" for e in events)
    assert any(e.type == "final" for e in events)


def test_survey_dsm_heartbeat_progress_does_not_reset_idle() -> None:
    # A DSM-style heartbeat (progress only) is not a content event.
    assert TranscriptionEvent.progress(audio_processed_until=1.0).is_content is False
    assert TranscriptionEvent.partial("s", "x").is_content is True
    assert TranscriptionEvent.final("s", "x").is_content is True
    assert TranscriptionEvent.supersede(["a"], ["b"]).is_content is True
