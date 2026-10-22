# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""The reference server keeps at most one idle engine instance per model key.

The design is in ``docs/internal/feat_plan/server-engine-reuse.md``. Each test
engine records its constructions, so a test can say how many engines the
server built for a sequence of calls.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import wave
from collections.abc import AsyncIterator, Iterator
from importlib.metadata import EntryPoint
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
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

#: One entry per call on the stream engine: the instance, the call type, the input.
CALLS: list[tuple[int, str, str]] = []


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    CONSTRUCTED.clear()
    CALLS.clear()
    BEHAVIOR.clear()
    BEHAVIOR.update(transcribe="ok", session="done")
    yield
    _BatchEngine.gate = None
    _BatchEngine.entered = None
    _Session.closing = None
    _Session.release = None


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

    #: Set by a test to learn that ``transcribe`` has started.
    entered: ClassVar[threading.Event | None] = None

    def __init__(self) -> None:
        self.config = _Config(engine="reuse-batch")
        CONSTRUCTED.append(self)

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        if self.entered is not None:
            self.entered.set()
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

    #: Set by a test to learn that the server is closing the session.
    closing: ClassVar[threading.Event | None] = None

    #: Set by a test to hold ``_close`` until the test releases it.
    release: ClassVar[threading.Event | None] = None

    def __init__(self, engine: _StreamEngine) -> None:
        super().__init__()
        self._engine = engine

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        try:
            async for chunk in self.audio_chunks():
                CALLS.append((id(self._engine), "stream", chunk.decode()))
                yield TranscriptionEvent.final("s0", chunk.decode(), start=0.0, end=1.0)
            if BEHAVIOR["session"] == "engine_error":
                yield TranscriptionEvent.make_error("engine_error", recoverable=False)
            elif BEHAVIOR["session"] == "linger":
                # The client leaves before done: the producer never finishes.
                yield TranscriptionEvent.final("s1", "late", start=1.0, end=2.0)
                await asyncio.Event().wait()
            elif BEHAVIOR["session"] == "cleanup_fault":
                yield TranscriptionEvent.done()
        finally:
            if BEHAVIOR["session"] == "cleanup_fault":
                raise RuntimeError("cleanup fault")

    async def _close(self) -> None:
        if self.closing is not None:
            self.closing.set()
        while self.release is not None and not self.release.is_set():
            await asyncio.sleep(0.01)
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
        samples = 0 if prepared.array is None else prepared.array.size
        CALLS.append((id(self), "rest", str(samples)))
        return TranscriptionResult(text=f"{samples} samples")

    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        return _Session(self)


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


def _client(**caps: int) -> TestClient:
    from fastapi.testclient import TestClient

    return TestClient(server_module.create_app(registry=_registry(), **caps))


def _wav(samples: int) -> bytes:
    """Return a mono 16-bit PCM WAV at 16 kHz with ``samples`` samples."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros(samples, dtype=np.int16).tobytes())
    return buffer.getvalue()


def _transcribe_stream_engine(client: TestClient, samples: int) -> str:
    """Send a REST call to the stream engine and return its transcript."""
    files = {"file": ("audio.wav", _wav(samples), "audio/wav")}
    response = client.post("/v1/transcribe", data={"model": "reuse-stream/echo"}, files=files)
    assert response.status_code == 200, response.text
    return response.json()["result"]["text"]


def _transcribe(client: TestClient) -> int:
    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    return client.post(
        "/v1/transcribe", data={"model": "reuse-batch/echo"}, files=files
    ).status_code


def _stream(client: TestClient, audio: bytes = b"hi") -> list[str]:
    """Run one session and return the types of the frames the server sent."""
    types: list[str] = []
    with client.websocket_connect("/v1/stream/reuse-stream/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        ws.send_bytes(audio)
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
    # instance stays idle, so later calls construct nothing more. The client
    # context runs every request on one event loop, as uvicorn does.
    _BatchEngine.gate = threading.Barrier(2)
    statuses: list[int] = []
    with _client() as client:
        with caplog.at_level(logging.INFO, logger="standard_asr.toolchain.server"):
            threads = [threading.Thread(target=lambda: statuses.append(_transcribe(client)))]
            threads.append(threading.Thread(target=lambda: statuses.append(_transcribe(client))))
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
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
    assert "Dropped the engine for model 'reuse-batch/echo': its call failed" in caplog.text
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
    _BatchEngine.entered = threading.Event()

    async def run() -> None:
        call = asyncio.ensure_future(
            server_module._run_transcription(  # pyright: ignore[reportPrivateUsage]
                _registry(), idle, "reuse-batch/echo", AudioBytes(data=b"x"), None, HTTPException
            )
        )
        while not _BatchEngine.entered.is_set():  # pyright: ignore[reportOptionalMemberAccess]
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


def test_a_stream_whose_audio_exceeds_the_cap_drops_the_engine() -> None:
    # The session can still end with done, but the stream failed: the server
    # sent payload_too_large, and every error drops the engine.
    client = _client(max_ws_session_bytes=4)
    with client.websocket_connect("/v1/stream/reuse-stream/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        ws.send_bytes(b"0123456789")
        frames: list[dict[str, Any]] = []
        while not frames or frames[-1]["type"] != "error":
            frames.append(ws.receive_json())
    assert frames[-1]["code"] == "payload_too_large"
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
        assert "Dropped the idle engine for model 'reuse-batch/echo' at shutdown" in caplog.text
        # An application that served nothing has nothing to drop.
        with TestClient(server_module.create_app(registry=_registry())):
            pass
    assert caplog.text.count("at shutdown") == 1


def test_an_applications_shutdown_handler_still_runs() -> None:
    # The idle engines are dropped by a shutdown handler, not a custom
    # lifespan, which would skip the handlers an application registers.
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    ran: list[str] = []
    app.router.add_event_handler("shutdown", lambda: ran.append("shutdown"))
    with TestClient(app):
        pass
    assert ran == ["shutdown"]


@pytest.mark.parametrize("rest_first", [True, False])
def test_one_instance_alternates_rest_calls_and_sessions(rest_first: bool) -> None:
    # The server keeps one instance for one model key across call types: REST
    # calls run on worker threads and sessions on the event loop. Each call
    # sees its own input, none of the previous one.
    steps = [("rest", "8000"), ("stream", "one"), ("rest", "4000"), ("stream", "two")]
    if not rest_first:
        steps = [steps[1], steps[0], steps[3], steps[2]]
    with _client() as client:
        for kind, value in steps:
            if kind == "rest":
                assert _transcribe_stream_engine(client, int(value)) == f"{value} samples"
            else:
                assert _stream(client, value.encode()) == ["final", "done"]
    assert len(CONSTRUCTED) == 1
    assert {instance for instance, _, _ in CALLS} == {id(CONSTRUCTED[0])}
    assert [(kind, value) for _, kind, value in CALLS] == steps


def test_an_instance_stays_held_until_its_session_closes() -> None:
    # While the server waits in a session's _close, the session still holds
    # its instance, so a REST call constructs another one.
    closing, release = threading.Event(), threading.Event()
    _Session.closing, _Session.release = closing, release
    with _client() as client:
        with client.websocket_connect("/v1/stream/reuse-stream/echo") as ws:
            ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
            ws.send_bytes(b"hi")
            ws.send_text("end")
            while ws.receive_json()["type"] != "done":
                pass
            assert closing.wait(timeout=5)
            _Session.closing = None
            assert _transcribe_stream_engine(client, 1600) == "1600 samples"
            release.set()
    assert len(CONSTRUCTED) == 2
    stream_instance = next(instance for instance, kind, _ in CALLS if kind == "stream")
    rest_instance = next(instance for instance, kind, _ in CALLS if kind == "rest")
    assert stream_instance != rest_instance


def test_a_producer_cleanup_that_raises_after_done_drops_the_engine() -> None:
    # The session closes the engine's producer before it appends the engine's
    # done. The cleanup raises, so the client gets an error and never done,
    # and the server does not keep the instance.
    client = _client()
    BEHAVIOR["session"] = "cleanup_fault"
    frames = _stream(client)
    assert frames[-1] == "error"
    assert "done" not in frames
    BEHAVIOR["session"] = "done"
    assert _stream(client) == ["final", "done"]
    assert len(CONSTRUCTED) == 2
