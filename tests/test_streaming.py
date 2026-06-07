# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the streaming protocol: events, reduce, session, sync bridge."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from standard_asr.exceptions import StreamClosedError
from standard_asr.streaming import (
    StreamReducer,
    SyncSession,
    TranscriptionEvent,
    TranscriptionSession,
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
