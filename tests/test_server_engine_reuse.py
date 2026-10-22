# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""The reference server keeps one idle engine instance per model key.

The design is in ``docs/internal/feat_plan/server-engine-reuse.md``. Each test
engine records its constructions, so a test can say how many engines the
server built for a sequence of calls.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from importlib.metadata import EntryPoint
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import pytest

from standard_asr.contract.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    FlagCap,
    LanguageCaps,
    StreamingCapabilities,
)
from standard_asr.contract.exceptions import AudioProcessingError
from standard_asr.contract.params import RuntimeParams
from standard_asr.contract.results import TranscriptionResult
from standard_asr.engine import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    InputKind,
    PreparedAudio,
    SampleRateRange,
)
from standard_asr.plugins.discovery import ModelRegistry, discover_models
from standard_asr.runtime.streaming import TranscriptionEvent, TranscriptionSession
from standard_asr.toolchain import server as server_module

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

#: Every engine the factories below construct, in order.
CONSTRUCTED: list[object] = []

#: How the next calls end: set by each test, read by the engines.
BEHAVIOR: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    CONSTRUCTED.clear()
    BEHAVIOR.clear()
    BEHAVIOR.update(transcribe="ok", session="done")
    yield
    _BatchEngine.gate = None


class _Config(BaseConfig[str]):
    engine: str = "reuse"


class _BatchProperties(BaseProperties):
    engine_id: str = "reuse-batch"
    model_name: str = "echo"
    protocol_version: str = "0.2.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    selectable_languages: list[str] = ["en"]


class _BatchEngine:
    """A structural engine that ignores its audio, as the server tests' dummy does."""

    properties: ClassVar[_BatchProperties] = _BatchProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        batch=BatchCapabilities(language=LanguageCaps(runtime_override=FlagCap(supported=True)))
    )
    provider_params_type: ClassVar[None] = None

    #: Set by a test to hold ``transcribe`` until the test releases it.
    gate: ClassVar[threading.Barrier | threading.Event | None] = None

    def __init__(self) -> None:
        self.config = _Config(engine="reuse-batch")
        CONSTRUCTED.append(self)

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        if self.gate is not None:
            self.gate.wait(timeout=5)
        if BEHAVIOR["transcribe"] == "engine_error":
            raise RuntimeError("engine fault")
        if BEHAVIOR["transcribe"] == "caller_error":
            raise AudioProcessingError("the audio does not decode")
        return TranscriptionResult(text="ok")


def _batch_factory() -> _BatchEngine:  # pyright: ignore[reportUnusedFunction]
    return _BatchEngine()


class _StreamProperties(_BatchProperties):
    engine_id: str = "reuse-stream"
    selectable_languages: list[str] = []
    wire_encodings: list[str] | None = ["pcm_s16le"]


class _Session(TranscriptionSession):
    """Ends the way ``BEHAVIOR["session"]`` says."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        async for chunk in self.audio_chunks():
            yield TranscriptionEvent.final("s0", chunk.decode(), start=0.0, end=1.0)
        if BEHAVIOR["session"] == "engine_error":
            yield TranscriptionEvent.make_error("engine_error", recoverable=False)
        elif BEHAVIOR["session"] == "linger":
            # The client leaves before done: the producer never finishes.
            yield TranscriptionEvent.final("s1", "late", start=1.0, end=2.0)
            await asyncio.Event().wait()

    async def _close(self) -> None:
        if BEHAVIOR["session"] == "close_fault":
            raise RuntimeError("close fault")


class _StreamEngine(EngineBase):
    properties: ClassVar[BaseProperties] = _StreamProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
    )

    def __init__(self) -> None:
        self.config = _Config(engine="reuse-stream")
        CONSTRUCTED.append(self)

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        return TranscriptionResult(text="")  # pragma: no cover - the batch path is unused

    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        return _Session()


def _stream_factory() -> _StreamEngine:  # pyright: ignore[reportUnusedFunction]
    return _StreamEngine()


def _registry() -> ModelRegistry:
    eps = [
        EntryPoint(
            name=f"reuse-{name}/echo",
            value=f"tests.test_server_engine_reuse:_{name}_factory",
            group="standard_asr.models",
        )
        for name in ("batch", "stream")
    ]
    return discover_models(eps=eps, strict=True)


def _client() -> TestClient:
    from fastapi.testclient import TestClient

    return TestClient(server_module.create_app(registry=_registry()))


def _transcribe(client: TestClient) -> int:
    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    return client.post(
        "/v1/transcribe", data={"model": "reuse-batch/echo"}, files=files
    ).status_code


def _stream(client: TestClient) -> list[str]:
    """Run one session and return the types of the frames the server sent."""
    types: list[str] = []
    with client.websocket_connect("/v1/stream/reuse-stream/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        ws.send_bytes(b"hi")
        ws.send_text("end")
        while not types or types[-1] not in ("done", "error"):
            types.append(ws.receive_json()["type"])
    return types


def test_calls_in_sequence_share_one_engine(caplog: pytest.LogCaptureFixture) -> None:
    client = _client()
    with caplog.at_level(logging.INFO, logger="standard_asr.toolchain.server"):
        assert [_transcribe(client) for _ in range(3)] == [200, 200, 200]
    assert len(CONSTRUCTED) == 1
    assert caplog.text.count("Constructed an engine for model 'reuse-batch/echo'") == 1


def test_overlapping_calls_get_their_own_engines_and_one_stays_idle(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Two calls in flight at once each need an instance; when both end, one
    # instance stays idle, so later calls construct nothing more.
    client = _client()
    _BatchEngine.gate = threading.Barrier(2)
    statuses: list[int] = []
    try:
        with caplog.at_level(logging.INFO, logger="standard_asr.toolchain.server"):
            threads = [threading.Thread(target=lambda: statuses.append(_transcribe(client)))]
            threads.append(threading.Thread(target=lambda: statuses.append(_transcribe(client))))
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
    finally:
        _BatchEngine.gate = None
    assert statuses == [200, 200]
    assert len(CONSTRUCTED) == 2
    assert "another instance is idle" in caplog.text
    assert [_transcribe(client) for _ in range(2)] == [200, 200]
    assert len(CONSTRUCTED) == 2


@pytest.mark.parametrize(("behavior", "status"), [("engine_error", 500), ("caller_error", 400)])
def test_a_call_that_raises_drops_its_engine(
    behavior: str, status: int, caplog: pytest.LogCaptureFixture
) -> None:
    # A caller's error drops the engine too: an engine can raise one after it
    # changed state, and no exception promises that the instance still works.
    client = _client()
    BEHAVIOR["transcribe"] = behavior
    with caplog.at_level(logging.INFO, logger="standard_asr.toolchain.server"):
        assert _transcribe(client) == status
    assert (
        "Dropped the engine for model 'reuse-batch/echo': its call did not succeed" in caplog.text
    )
    BEHAVIOR["transcribe"] = "ok"
    assert _transcribe(client) == 200
    assert len(CONSTRUCTED) == 2


def test_a_canceled_call_drops_its_engine() -> None:
    # The worker thread can outlive a canceled handler, so the instance is
    # never handed to another call.
    from fastapi import HTTPException

    from standard_asr.audio.input import AudioBytes

    idle = server_module._IdleEngines()  # pyright: ignore[reportPrivateUsage]
    release = threading.Event()
    _BatchEngine.gate = release

    async def run() -> None:
        call = asyncio.ensure_future(
            server_module._run_transcription(  # pyright: ignore[reportPrivateUsage]
                _registry(), idle, "reuse-batch/echo", AudioBytes(data=b"x"), None, HTTPException
            )
        )
        while not CONSTRUCTED:
            await asyncio.sleep(0.01)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert idle.take("reuse-batch/echo") is None
        # Let the worker thread finish, which asyncio.run waits for.
        release.set()

    asyncio.run(run())
    assert len(CONSTRUCTED) == 1


def test_a_stream_that_ends_with_done_keeps_its_engine() -> None:
    client = _client()
    assert _stream(client) == ["final", "done"]
    assert _stream(client) == ["final", "done"]
    assert len(CONSTRUCTED) == 1


@pytest.mark.parametrize("behavior", ["engine_error", "close_fault"])
def test_a_stream_that_does_not_end_cleanly_drops_its_engine(behavior: str) -> None:
    client = _client()
    BEHAVIOR["session"] = behavior
    first = _stream(client)
    assert first[-1] == ("error" if behavior == "engine_error" else "done")
    BEHAVIOR["session"] = "done"
    assert _stream(client) == ["final", "done"]
    assert len(CONSTRUCTED) == 2


def test_a_client_that_leaves_before_done_drops_the_engine() -> None:
    client = _client()
    BEHAVIOR["session"] = "linger"
    with client.websocket_connect("/v1/stream/reuse-stream/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        ws.send_bytes(b"hi")
        ws.send_text("end")
        assert ws.receive_json()["type"] == "final"
    BEHAVIOR["session"] = "done"
    assert _stream(client) == ["final", "done"]
    assert len(CONSTRUCTED) == 2


def test_a_failed_establishment_drops_the_engine() -> None:
    # An audio format the engine does not accept fails before the bridge.
    client = _client()
    with client.websocket_connect("/v1/stream/reuse-stream/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 8000}})
        assert ws.receive_json()["type"] == "error"
    assert _stream(client) == ["final", "done"]
    assert len(CONSTRUCTED) == 2


def test_shutdown_drops_the_idle_engines(caplog: pytest.LogCaptureFixture) -> None:
    from fastapi.testclient import TestClient

    with caplog.at_level(logging.INFO, logger="standard_asr.toolchain.server"):
        with TestClient(server_module.create_app(registry=_registry())) as client:
            assert _transcribe(client) == 200
        assert "Dropped 1 idle engine instance(s) at shutdown" in caplog.text
        # An application that served nothing has nothing to drop.
        with TestClient(server_module.create_app(registry=_registry())):
            pass
    assert caplog.text.count("at shutdown") == 1
