# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Canceled routes retain their engines until work and cleanup finish."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException, WebSocket
from fastapi.routing import APIRoute

from standard_asr.audio.input import AudioBytes, AudioInput
from standard_asr.contract.artifacts import ArtifactReport
from standard_asr.contract.params import RuntimeParams
from standard_asr.contract.results import TranscriptionResult
from standard_asr.plugins.discovery import ModelRegistry
from standard_asr.runtime.engine_pool import EnginePool
from standard_asr.runtime.streaming import TranscriptionEvent, TranscriptionSession
from standard_asr.toolchain import server


class _Engine:
    provider_params_type = None

    def __init__(self, failure: BaseException | None = None) -> None:
        self.started = threading.Event()
        self.unblock = threading.Event()
        self.finished = threading.Event()
        self.failure = failure
        self.close_calls = 0

    def _work(self) -> None:
        self.started.set()
        assert self.unblock.wait(timeout=5), "test did not release the engine operation"
        self.finished.set()
        if self.failure is not None:
            raise self.failure

    def transcribe(self, _audio: AudioInput, _params: RuntimeParams) -> TranscriptionResult:
        self._work()
        return TranscriptionResult(text="finished")

    def artifact_status(self) -> ArtifactReport:
        self._work()
        return ArtifactReport.from_requirements(mode="batch", applicable=False)

    def close(self) -> None:
        assert self.finished.is_set(), "engine closed before its work finished"
        self.close_calls += 1


class _Registry:
    def __init__(self, engine: _Engine) -> None:
        self.engine = engine

    def names(self) -> list[str]:
        return ["dummy/model"]

    def create(self, _model: str, **_config: Any) -> _Engine:
        return self.engine


def _registry(engine: _Engine) -> ModelRegistry:
    return cast(ModelRegistry, _Registry(engine))


@pytest.mark.parametrize("operation", ["transcribe", "readiness"])
@pytest.mark.parametrize("fails", [False, True])
def test_canceled_http_route_drains_native_work_before_releasing_engine(
    operation: str, fails: bool, caplog: pytest.LogCaptureFixture
) -> None:
    async def scenario() -> None:
        engine = _Engine(RuntimeError("late engine failure") if fails else None)
        app = server.create_app(registry=_registry(engine))
        pool: EnginePool = app.state.engine_pool
        if operation == "readiness":
            endpoint = next(
                route.endpoint
                for route in app.routes
                if isinstance(route, APIRoute) and route.path == "/v1/readiness/{model:path}"
            )
            request = endpoint("dummy/model")
        else:
            request = server._run_transcription(  # pyright: ignore[reportPrivateUsage]
                pool, "dummy/model", AudioBytes(b"fixture"), None, HTTPException
            )
        route_task = asyncio.create_task(request)
        try:
            assert await asyncio.to_thread(engine.started.wait, 2)
            route_task.cancel("first cancellation")
            await asyncio.sleep(0)
            closing = asyncio.create_task(pool.aclose())
            await asyncio.sleep(0)
            assert not route_task.done()
            assert not closing.done()
            route_task.cancel("second cancellation")
            await asyncio.sleep(0)
            assert not route_task.done()
            assert not closing.done()
            assert engine.close_calls == 0
        finally:
            engine.unblock.set()
        with pytest.raises(asyncio.CancelledError, match="first cancellation"):
            await route_task
        await asyncio.wait_for(closing, timeout=2)
        assert engine.close_calls == 1

    asyncio.run(scenario())
    assert ("late engine failure" in caplog.text) is fails


def test_native_operation_can_itself_report_cancellation() -> None:
    async def scenario() -> None:
        engine = _Engine(asyncio.CancelledError("engine stopped"))
        engine.unblock.set()
        pool = EnginePool(_registry(engine))
        with pytest.raises(asyncio.CancelledError):
            await server._run_transcription(  # pyright: ignore[reportPrivateUsage]
                pool, "dummy/model", AudioBytes(b"fixture"), None, HTTPException
            )
        await pool.aclose()
        assert engine.close_calls == 1

    asyncio.run(scenario())


class _CleanupSession(TranscriptionSession):
    def __init__(self, engine: _Engine) -> None:
        super().__init__()
        self.engine = engine
        self.close_started = asyncio.Event()
        self.finish_close = asyncio.Event()

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        yield TranscriptionEvent.done()

    async def _close(self) -> None:
        self.close_started.set()
        await self.finish_close.wait()
        self.engine.finished.set()


class _Socket:
    async def receive(self) -> dict[str, Any]:
        await asyncio.Event().wait()
        return {}

    async def send_json(self, _frame: Any) -> None:
        pass


def test_canceled_websocket_route_finishes_session_cleanup_before_releasing_engine() -> None:
    async def scenario() -> None:
        engine = _Engine()
        session = _CleanupSession(engine)
        pool = EnginePool(_registry(engine))

        async def route() -> None:
            lease = await pool.acquire("dummy/model")
            server._release_lease_when_task_finishes(lease)  # pyright: ignore[reportPrivateUsage]
            await server._bridge_stream(  # pyright: ignore[reportPrivateUsage]
                cast(WebSocket, _Socket()),
                session,
                max_frame_bytes=1024,
                max_session_bytes=1024,
            )

        route_task = asyncio.create_task(route())
        try:
            await asyncio.wait_for(session.close_started.wait(), timeout=2)
            route_task.cancel("first cancellation")
            await asyncio.sleep(0)
            closing = asyncio.create_task(pool.aclose())
            await asyncio.sleep(0)
            route_task.cancel("second cancellation")
            await asyncio.sleep(0)
            assert not route_task.done()
            assert not closing.done()
            assert engine.close_calls == 0
        finally:
            session.finish_close.set()
        with pytest.raises(asyncio.CancelledError, match="first cancellation"):
            await route_task
        await asyncio.wait_for(closing, timeout=2)
        assert engine.close_calls == 1

    asyncio.run(scenario())
