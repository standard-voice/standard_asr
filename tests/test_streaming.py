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
    _cancel_all_tasks,  # pyright: ignore[reportPrivateUsage]
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


def test_event_model_rejects_structurally_illegal_events() -> None:
    from pydantic import ValidationError

    # partial/final MUST carry segment_id and text.
    with pytest.raises(ValidationError, match="segment_id and text"):
        TranscriptionEvent(type="partial", segment_id="s0")
    with pytest.raises(ValidationError, match="segment_id and text"):
        TranscriptionEvent(type="final", text="hi")
    # error MUST carry a code.
    with pytest.raises(ValidationError, match="MUST carry a code"):
        TranscriptionEvent(type="error")
    # supersede MUST retire something and keep old/new disjoint.
    with pytest.raises(ValidationError, match="retire at least one"):
        TranscriptionEvent(type="supersede", new_ids=["s1"])
    with pytest.raises(ValidationError, match="disjoint"):
        TranscriptionEvent(type="supersede", old_ids=["s1"], new_ids=["s1"])
    # progress / done need no segment fields.
    assert TranscriptionEvent(type="progress").type == "progress"


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


def test_session_feed_bytes_is_a_single_chunk() -> None:
    # A bare bytes-like is ONE chunk, not an iterable of int byte values:
    # feed(b"abc") must yield the chunk b"abc", not 97/98/99.
    captured: dict[str, bool] = {}

    async def run() -> list[TranscriptionEvent]:
        session = _EchoSession()
        session.feed(b"abc")
        captured["replayable"] = session.replayable
        return await _collect(session)

    events = asyncio.run(run())
    finals = [e for e in events if e.type == "final"]
    assert {f.text for f in finals} == {"abc"}
    assert events[-1].type == "done"
    # a wrapped bytes-like is a re-iterable collection -> replayable
    assert captured["replayable"] is True


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


def test_sync_bridge_shutdown_is_idempotent() -> None:
    # A second __exit__ / _shutdown must be a no-op (the already-closed guard),
    # never re-tearing-down a stopped loop.
    sync = SyncSession(_EchoSession(), submit_timeout=5.0)
    sync.__enter__()
    sync.feed([b"hi"])
    list(sync)
    sync.__exit__(None, None, None)
    # Second teardown returns immediately via the _closed guard.
    sync._shutdown()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert sync._thread.is_alive() is False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_sync_bridge_unbounded_submit_timeout() -> None:
    # submit_timeout=None means the pump never imposes its own deadline (it relies
    # on the session's own terminal-event guarantees). The bridge still completes.
    sync = SyncSession(_EchoSession(), submit_timeout=None)
    with sync:
        sync.feed([b"hello"])
        events = list(sync)
    assert events[-1].type == "done"


def test_cancel_all_tasks_cancels_outstanding() -> None:
    # The teardown helper cancels and awaits every task on the loop except the
    # caller, so no task is destroyed while pending.
    async def run() -> bool:
        async def _forever() -> None:
            await asyncio.sleep(100)

        task = asyncio.ensure_future(_forever())
        await asyncio.sleep(0)  # let it start
        await _cancel_all_tasks()
        return task.cancelled()

    assert asyncio.run(run()) is True


def test_aexit_without_aenter_has_no_tasks_to_cancel() -> None:
    # __aexit__ before __aenter__ ran: there is no producer/feed task, so the
    # cancel/gather is skipped and _close still runs cleanly.
    async def run() -> None:
        session = _EchoSession()
        await session.__aexit__(None, None, None)

    asyncio.run(run())


def test_iterate_stops_on_closed_empty_buffer() -> None:
    # Teardown race guard: if the event buffer is closed with no terminal event
    # ever landing (e.g. the producer was cancelled mid-flight), the iterator
    # must end cleanly when get() returns None rather than hang.
    async def run() -> list[TranscriptionEvent]:
        session = _EchoSession(done_timeout=5.0)
        # Close the buffer directly without any events -> get() yields None.
        session._buffer.close()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        events: list[TranscriptionEvent] = []
        async for ev in session:
            events.append(ev)
        return events

    assert asyncio.run(run()) == []


def test_cancel_all_tasks_no_other_tasks_is_noop() -> None:
    # With no outstanding tasks the gather branch is skipped without error.
    asyncio.run(_cancel_all_tasks())


def test_abstract_produce_raises_not_implemented() -> None:
    # The abstract base _produce body raises NotImplementedError when invoked
    # directly (e.g. a subclass that delegates to super() instead of overriding).
    class _Concrete(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.done()  # pragma: no cover

    with pytest.raises(NotImplementedError):
        # Deliberately invoke the abstract base body (which raises) to prove the
        # contract for a subclass that wrongly delegates to super().
        TranscriptionSession._produce(_Concrete())  # pyright: ignore[reportPrivateUsage, reportAbstractUsage]


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


def test_guard_clamp_decreased_then_invalid_boundary_combines_reasons() -> None:
    # A decreased stable_until is first clamped UP to the prior; if that prior is
    # itself an invalid boundary for the new text, a second clamp DOWN to a valid
    # boundary applies and both reasons are reported together.
    guard = _LifecycleGuard()
    # prior = 2 on text whose boundary 2 is valid; freezes the prefix "ae".
    first = guard.admit(TranscriptionEvent.partial("s0", "ae", stable_until=2))
    assert first is not None and first.stable_until == 2
    # New text "a" + "e" + combining accent extends "ae" (frozen prefix preserved)
    # but boundary 2 now splits the combining sequence. A decreased request
    # (0 < prior 2) clamps up to 2, which is then invalid for this text -> clamp
    # down to the largest valid boundary (1).
    combining = "a" + "e" + "́"  # "ae" + COMBINING ACUTE ACCENT over the e
    second = guard.admit(TranscriptionEvent.partial("s0", combining, stable_until=0))
    assert second is not None and second.stable_until == 1
    msgs = [d.message for d in guard.diagnostics if d.code == "stable_until_clamped"]
    # The second clamp records BOTH the decrease and the invalid-boundary reason
    # in one combined message (the "; " join under test).
    assert any("decreased" in m and "invalid boundary" in m and "; " in m for m in msgs)


def test_guard_clamps_decreasing_audio_cursor() -> None:
    # audio_processed_until is monotonic across the whole session; a decrease is
    # clamped to the prior value with a diagnostic (spec ST.4.1).
    guard = _LifecycleGuard()
    e1 = guard.admit(TranscriptionEvent.progress(audio_processed_until=2.0))
    assert e1 is not None and e1.audio_processed_until == 2.0
    e2 = guard.admit(TranscriptionEvent.progress(audio_processed_until=1.0))
    assert e2 is not None and e2.audio_processed_until == 2.0
    assert any(d.code == "audio_cursor_decreased" for d in guard.diagnostics)


def test_guard_raises_on_decreasing_audio_cursor_strict() -> None:
    guard = _LifecycleGuard(strict=True)
    guard.admit(TranscriptionEvent.progress(audio_processed_until=2.0))
    with pytest.raises(ValueError, match="cursor is monotonic"):
        guard.admit(TranscriptionEvent.progress(audio_processed_until=1.0))


def test_guard_suppresses_frozen_prefix_rewrite() -> None:
    # The frozen prefix (text[:stable_until]) is immutable: extending text is
    # fine, but rewriting an already-frozen region is suppressed (spec ST.4.2).
    guard = _LifecycleGuard()
    first = guard.admit(TranscriptionEvent.partial("s0", "the cat", stable_until=4))
    assert first is not None  # freezes "the "
    extend = guard.admit(TranscriptionEvent.partial("s0", "the cattle", stable_until=4))
    assert extend is not None  # extends, prefix preserved
    rewrite = guard.admit(TranscriptionEvent.partial("s0", "a dog runs", stable_until=4))
    assert rewrite is None
    assert any(d.code == "frozen_prefix_rewritten" for d in guard.diagnostics)


def test_guard_supersede_new_ids_open_then_partial_allowed() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("s0", "x"))
    guard.admit(TranscriptionEvent.supersede(["s0"], ["s1"]))
    # s1 was started open by supersede; a partial for it is legal.
    assert guard.admit(TranscriptionEvent.partial("s1", "new")) is not None


# --------------------------------------------------------------------------- #
# C1 -- supersede MUST preserve concatenated frozen text (spec ST.5.2)
# --------------------------------------------------------------------------- #
def test_guard_supersede_2to1_merge_preserves_frozen_text() -> None:
    # Two retired segments froze "你好" and "世界"; the single replacement MUST
    # carry the concatenation "你好世界" as its frozen prefix.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("a", "你好", stable_until=2))
    guard.admit(TranscriptionEvent.final("b", "世界", stable_until=2))
    guard.admit(TranscriptionEvent.supersede(["a", "b"], ["c"]))
    accepted = guard.admit(TranscriptionEvent.partial("c", "你好世界！", stable_until=4))
    assert accepted is not None
    assert not guard.diagnostics


def test_guard_supersede_1to2_split_preserves_frozen_text() -> None:
    # "你好世界" frozen on one segment, split into "你好" + "世界…": the two new
    # segments' concatenated frozen prefix reconstructs F_old and MUST be
    # accepted (the conservative split case).
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("a", "你好世界", stable_until=4))
    guard.admit(TranscriptionEvent.supersede(["a"], ["b", "c"]))
    # First new segment freezes "你好" -- strictly shorter than F_old, the safe
    # (pending) direction: accepted with no diagnostic.
    first = guard.admit(TranscriptionEvent.partial("b", "你好", stable_until=2))
    assert first is not None
    assert not guard.diagnostics
    # Second new segment freezes "世界"; F_new now == "你好世界" == F_old.
    second = guard.admit(TranscriptionEvent.final("c", "世界呀", stable_until=2))
    assert second is not None
    assert not guard.diagnostics


def test_guard_supersede_rewrite_frozen_prefix_suppressed() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("a", "你好世界", stable_until=4))
    guard.admit(TranscriptionEvent.supersede(["a"], ["b"]))
    # New segment freezes "再见" -- rewrites the user-visible frozen "你好世界".
    rejected = guard.admit(TranscriptionEvent.partial("b", "再见", stable_until=2))
    assert rejected is None
    assert any(d.code == "frozen_prefix_rewritten_supersede" for d in guard.diagnostics)


def test_guard_supersede_rewrite_frozen_prefix_strict_raises() -> None:
    guard = _LifecycleGuard(strict=True)
    guard.admit(TranscriptionEvent.final("a", "你好世界", stable_until=4))
    guard.admit(TranscriptionEvent.supersede(["a"], ["b"]))
    with pytest.raises(ValueError, match="preserve frozen text"):
        guard.admit(TranscriptionEvent.partial("b", "再见", stable_until=2))


def test_guard_supersede_no_frozen_old_text_has_no_obligation() -> None:
    # An old segment with no frozen prefix imposes no preservation obligation;
    # the replacement may freeze whatever it likes.
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("a", "draft"))  # stable_until None -> 0
    guard.admit(TranscriptionEvent.supersede(["a"], ["b"]))
    accepted = guard.admit(TranscriptionEvent.partial("b", "different", stable_until=4))
    assert accepted is not None
    assert not guard.diagnostics


# --------------------------------------------------------------------------- #
# STRE-2 / X-ST-2 -- supersede ordering & disjointness invariants (spec ST.5.2)
# --------------------------------------------------------------------------- #
def test_supersede_disjoint_enforced_at_construction() -> None:
    # old_ids n new_ids = empty MUST hold; the event model refuses to build one.
    with pytest.raises(ValueError, match="disjoint"):
        TranscriptionEvent.supersede(["a"], ["a"])
    with pytest.raises(ValueError, match="disjoint"):
        TranscriptionEvent(type="supersede", old_ids=["a"], new_ids=["a"])


def test_guard_supersede_unknown_old_id_suppressed() -> None:
    guard = _LifecycleGuard()
    rejected = guard.admit(TranscriptionEvent.supersede(["never-seen"], ["b"]))
    assert rejected is None
    assert any(d.code == "supersede_unknown_old_id" for d in guard.diagnostics)


def test_guard_supersede_unknown_old_id_strict_raises() -> None:
    guard = _LifecycleGuard(strict=True)
    with pytest.raises(ValueError, match="never-announced"):
        guard.admit(TranscriptionEvent.supersede(["never-seen"], ["b"]))


def test_guard_supersede_reintroduces_known_new_id_suppressed() -> None:
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.partial("a", "x"))
    guard.admit(TranscriptionEvent.partial("b", "y"))  # b already open
    rejected = guard.admit(TranscriptionEvent.supersede(["a"], ["b"]))
    assert rejected is None
    assert any(d.code == "supersede_reintroduces_segment" for d in guard.diagnostics)


def test_guard_supersede_reintroduces_known_new_id_strict_raises() -> None:
    guard = _LifecycleGuard(strict=True)
    guard.admit(TranscriptionEvent.partial("a", "x"))
    guard.admit(TranscriptionEvent.partial("b", "y"))
    with pytest.raises(ValueError, match="MUST be fresh"):
        guard.admit(TranscriptionEvent.supersede(["a"], ["b"]))


# --------------------------------------------------------------------------- #
# STRE-3/4 -- illegal final-after-final (spec ST.5.1)
# --------------------------------------------------------------------------- #
def test_guard_suppresses_final_after_final() -> None:
    guard = _LifecycleGuard()
    assert guard.admit(TranscriptionEvent.final("s0", "done")) is not None
    rejected = guard.admit(TranscriptionEvent.final("s0", "rewritten"))
    assert rejected is None
    assert any(d.code == "lifecycle_final_after_final" for d in guard.diagnostics)


def test_guard_final_after_final_strict_raises() -> None:
    guard = _LifecycleGuard(strict=True)
    guard.admit(TranscriptionEvent.final("s0", "done"))
    with pytest.raises(ValueError, match="only supersede or a"):
        guard.admit(TranscriptionEvent.final("s0", "again"))


def test_guard_closed_after_final_is_legal() -> None:
    # A closed event (finality="closed") after a plain final is the legal
    # in-place post-processing correction (spec ST.5.1/5.4).
    guard = _LifecycleGuard()
    guard.admit(TranscriptionEvent.final("s0", "hello"))
    closed = guard.admit(TranscriptionEvent.closed("s0", "Hello."))
    assert closed is not None
    assert not any(d.code == "lifecycle_final_after_final" for d in guard.diagnostics)


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


def test_max_session_seconds_without_max_idle() -> None:
    # max_idle is None (1041 False branch); only the wall-clock cap terminates a
    # continuously-chatty session.
    class _ChattySession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            i = 0
            while True:
                await asyncio.sleep(0.01)
                yield TranscriptionEvent.final(f"s{i}", "x")
                i += 1

    async def run() -> list[TranscriptionEvent]:
        session = _ChattySession(done_timeout=5.0, max_idle=None, max_session_seconds=0.1)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "session_timeout"


def test_session_timeout_checked_at_loop_top_with_buffered_events() -> None:
    # The wall-clock cap is detected at the TOP of the loop (remaining <= 0)
    # before any wait, when the clock has already advanced past the budget. A
    # deterministic fake clock removes the timing race.
    class _OneShotSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.final("s0", "x")
            await asyncio.sleep(10)  # then go silent

    async def run() -> list[TranscriptionEvent]:
        session = _OneShotSession(done_timeout=5.0, max_idle=None, max_session_seconds=1.0)
        # Deterministic clock: start at 0, then jump past the 1.0s budget so the
        # second loop iteration's top-of-loop check sees remaining <= 0.
        ticks = iter([0.0, 0.0, 2.0, 2.0, 2.0, 2.0])

        def _clock() -> float:
            try:
                return next(ticks)
            except StopIteration:  # pragma: no cover - safety for extra reads
                return 2.0

        session._monotonic = _clock  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
    assert events[-1].type == "error"
    assert events[-1].code == "session_timeout"


def test_session_timeout_on_silence_in_timeout_handler() -> None:
    # Total silence: the per-event wait times out exactly at the wall-clock cap so
    # the TimeoutError handler synthesizes session_timeout (not done_timeout).
    class _SilentSession(TranscriptionSession):
        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            await asyncio.sleep(10)
            yield TranscriptionEvent.done()  # pragma: no cover

    async def run() -> list[TranscriptionEvent]:
        # max_session < done_timeout, no idle cap: the wait is bounded by the
        # remaining wall-clock budget and the handler picks session_timeout.
        session = _SilentSession(done_timeout=0.2, max_idle=None, max_session_seconds=0.05)
        session.feed([])
        return await _collect(session)

    events = asyncio.run(run())
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


def test_reduce_event_ignores_non_text_events() -> None:
    # done / error / heartbeat carry no segment text -> the map is untouched.
    segs: dict[str, str] = {"s1": "kept"}
    reduce_event(segs, TranscriptionEvent.done())
    reduce_event(segs, TranscriptionEvent.make_error("x", recoverable=False))
    assert segs == {"s1": "kept"}


def test_reduce_event_supersede_unknown_old_id_is_noop() -> None:
    # Superseding an id that was never committed must not raise.
    segs: dict[str, str] = {"s1": "a"}
    reduce_event(segs, TranscriptionEvent.supersede(["ghost"], ["s2"]))
    assert segs == {"s1": "a"}


def test_reducer_records_detected_language() -> None:
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.partial("s0", "hola", detected_language="es"))
    reducer.add(TranscriptionEvent.final("s0", "hola amigo"))
    result = reducer.result()
    assert result.detected_language == "es"


def test_reducer_refinalize_same_segment_keeps_single_slot() -> None:
    # A second final for the same segment_id overwrites in place (it is already in
    # _order), it must not append a duplicate ordering entry.
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s0", "first"))
    reducer.add(TranscriptionEvent.final("s0", "second"))
    result = reducer.result()
    assert result.text == "second"
    assert result.segments is not None
    assert len(result.segments) == 1


def test_reducer_supersede_removes_committed_segment() -> None:
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s0", "old", start=0.0, end=1.0))
    reducer.add(TranscriptionEvent.supersede(["s0"], ["s1"]))
    reducer.add(TranscriptionEvent.final("s1", "new", start=0.0, end=1.0))
    assert reducer.result().text == "new"


def test_reducer_supersede_unknown_id_is_noop() -> None:
    # Superseding an id the reducer never committed must be skipped silently (the
    # `if old_id in self._segments` guard), leaving committed segments intact.
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s0", "kept", start=0.0, end=1.0))
    reducer.add(TranscriptionEvent.supersede(["never-seen"], ["s9"]))
    assert reducer.result().text == "kept"


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
