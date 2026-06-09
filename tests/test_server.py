"""Tests for FastAPI server helpers.

The transcription endpoints deliberately do **not** pre-decode uploads: they
hand the encoded payload to the engine's own standard negotiation. The tests
below therefore exercise real :class:`EngineBase` engines so that decoding,
resampling and encoded-passthrough are proven end-to-end (a bare stub that
ignored the audio would mask the very contract the server must honour).
"""

from __future__ import annotations

import base64
import builtins
import io
import json
import wave
from collections.abc import AsyncIterator
from importlib.metadata import EntryPoint
from typing import Any, ClassVar, Literal

import httpx
import numpy as np
import pytest

from standard_asr import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    InputKind,
    PreparedAudio,
    RuntimeParams,
    TranscriptionResult,
)
from standard_asr import server as server_module
from standard_asr.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    FlagCap,
    LanguageCaps,
    StreamingCapabilities,
)
from standard_asr.discovery import discover_models
from standard_asr.runtime_params import ProviderParams
from standard_asr.streaming import TranscriptionEvent, TranscriptionSession


class _DummyConfig(BaseConfig[str]):
    engine: str = "dummy"


class _DummyProperties(BaseProperties):
    engine_id: str = "dummy"
    model_name: str = "echo"
    protocol_version: str = "0.2.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | Literal["any"] = [16000]
    selectable_languages: list[str] = ["en"]


class _DummyParams(ProviderParams):
    beam: int = 1


_DUMMY_CAPS = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
    )
)


class _DummyASR:
    """Bare structural engine (not EngineBase): ignores audio, returns a fixed
    transcript. Used for the error-mapping, capabilities and params-schema
    tests, none of which depend on audio negotiation."""

    properties: ClassVar[_DummyProperties] = _DummyProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _DUMMY_CAPS
    provider_params_type: ClassVar[type[ProviderParams] | None] = _DummyParams

    def __init__(self) -> None:
        self.config = _DummyConfig(engine="dummy")

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="dummy")


def _dummy_factory() -> _DummyASR:  # pyright: ignore[reportUnusedFunction]
    return _DummyASR()


class _FailASR(_DummyASR):
    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        raise RuntimeError("boom: /secret/internal/path leaked")


def _fail_factory() -> _FailASR:  # pyright: ignore[reportUnusedFunction]
    return _FailASR()


class _ClientErrorASR(_DummyASR):
    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        from standard_asr.exceptions import UnsupportedFeatureError

        raise UnsupportedFeatureError("word_timestamps not supported")


def _client_error_factory() -> _ClientErrorASR:  # pyright: ignore[reportUnusedFunction]
    return _ClientErrorASR()


class _NoInstantiateASR(_DummyASR):
    def __init__(self) -> None:
        raise RuntimeError("instantiation forbidden (would resolve credentials)")


def _no_instantiate_factory() -> _NoInstantiateASR:  # pyright: ignore[reportUnusedFunction]
    return _NoInstantiateASR()


class _ConfigErrorOnConstructASR(_DummyASR):
    """Construction raises a client-config error (e.g. missing credential)."""

    def __init__(self) -> None:
        from standard_asr.exceptions import ConfigError

        raise ConfigError("missing API key for /secret/internal/path")


def _config_error_construct_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _ConfigErrorOnConstructASR
):
    return _ConfigErrorOnConstructASR()


# --- Real EngineBase engines that record what negotiation hands them ----------

#: Set by the recording engines' ``_transcribe`` so tests can assert on the
#: shape/rate/bytes the standard negotiation actually produced.
_RECORDED: dict[str, Any] = {}

_REC_CAPS = DeclaredCapabilities(batch=BatchCapabilities())


def _wav_bytes(rate: int, samples: int = 1600) -> bytes:
    """Return a minimal mono 16-bit PCM WAV at ``rate`` Hz."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(np.zeros(samples, dtype=np.int16).tobytes())
    return buf.getvalue()


class _Array8kProperties(BaseProperties):
    engine_id: str = "rec"
    model_name: str = "array8k"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 8000
    accepted_sample_rates: list[int] | Literal["any"] = [8000]
    selectable_languages: list[str] = []


class _RecordingArray8kASR(EngineBase):
    """8 kHz-native engine: an 8 kHz upload must reach it at 8 kHz, never
    silently up-sampled to 16 kHz (spec R7)."""

    properties: ClassVar[BaseProperties] = _Array8kProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _REC_CAPS

    def __init__(self) -> None:
        self.config = _DummyConfig(engine="rec")

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        _RECORDED["kind"] = prepared.kind
        _RECORDED["sample_rate"] = prepared.sample_rate
        _RECORDED["array_len"] = int(prepared.array.size) if prepared.array is not None else None
        return TranscriptionResult(text="array8k")


def _recording_array8k_factory() -> _RecordingArray8kASR:  # pyright: ignore[reportUnusedFunction]
    return _RecordingArray8kASR()


class _EncodedProperties(BaseProperties):
    engine_id: str = "rec"
    model_name: str = "bytes"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ENCODED_BYTES}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | Literal["any"] = "any"
    selectable_languages: list[str] = []


class _RecordingEncodedASR(EngineBase):
    """Encoded-only engine: must be servable at all (mission G.2.2) and must
    receive the original encoded bytes byte-for-byte (passthrough)."""

    properties: ClassVar[BaseProperties] = _EncodedProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _REC_CAPS

    def __init__(self) -> None:
        self.config = _DummyConfig(engine="rec")

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        _RECORDED["kind"] = prepared.kind
        _RECORDED["data"] = prepared.data
        return TranscriptionResult(text="bytes")


def _recording_encoded_factory() -> _RecordingEncodedASR:  # pyright: ignore[reportUnusedFunction]
    return _RecordingEncodedASR()


def _registry():
    eps = [
        EntryPoint(
            name="dummy/echo",
            value="tests.test_server:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    return discover_models(eps=eps, strict=True)


def _registry_for(factory: str):
    eps = [
        EntryPoint(
            name="dummy/echo",
            value=f"tests.test_server:{factory}",
            group="standard_asr.models",
        )
    ]
    return discover_models(eps=eps, strict=True)


def test_create_app_missing_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "fastapi":
            raise ImportError("fastapi not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        server_module.create_app()


def test_create_app_endpoints() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    response: httpx.Response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    models: httpx.Response = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()[0]["key"] == "dummy/echo"

    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode("utf-8"),
    }
    transcribe: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert transcribe.status_code == 200
    assert transcribe.json()["result"]["text"] == "dummy"


def test_server_array_engine_keeps_native_rate_through_negotiation() -> None:
    # An 8 kHz upload to an 8 kHz-native engine must arrive as an ARRAY at
    # 8000 Hz -- proving the server routes through negotiation and never forces
    # the old unconditional 16 kHz resample (spec R7).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _RECORDED.clear()
    app = server_module.create_app(registry=_registry_for("_recording_array8k_factory"))
    client = TestClient(app)

    files = {"file": ("audio.wav", _wav_bytes(rate=8000), "audio/wav")}
    resp: httpx.Response = client.post("/v1/transcribe", data={"model": "dummy/echo"}, files=files)
    assert resp.status_code == 200
    assert _RECORDED["kind"] is InputKind.ARRAY
    assert _RECORDED["sample_rate"] == 8000


def test_server_encoded_engine_receives_original_bytes_multipart() -> None:
    # An encoded-only engine must be servable (mission G.2.2) and receive the
    # uploaded bytes verbatim (passthrough, no lossy decode/re-encode).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _RECORDED.clear()
    wav = _wav_bytes(rate=16000)
    app = server_module.create_app(registry=_registry_for("_recording_encoded_factory"))
    client = TestClient(app)

    files = {"file": ("audio.wav", wav, "audio/wav")}
    resp: httpx.Response = client.post("/v1/transcribe", data={"model": "dummy/echo"}, files=files)
    assert resp.status_code == 200
    assert _RECORDED["kind"] is InputKind.ENCODED_BYTES
    assert _RECORDED["data"] == wav


def test_server_encoded_engine_receives_original_bytes_json() -> None:
    # The JSON (base64) endpoint feeds the same negotiation path.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _RECORDED.clear()
    wav = _wav_bytes(rate=16000)
    app = server_module.create_app(registry=_registry_for("_recording_encoded_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(wav).decode()}
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 200
    assert _RECORDED["kind"] is InputKind.ENCODED_BYTES
    assert _RECORDED["data"] == wav


def test_transcribe_json_decode_error_maps_to_400() -> None:
    # Invalid base64 reaching a real engine fails inside negotiation and maps
    # to 400 (no pre-decode in the endpoint any more).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_recording_array8k_factory"))
    client = TestClient(app)

    resp: httpx.Response = client.post(
        "/v1/transcribe:json", json={"model": "dummy/echo", "audio": "not-valid-base64!!!"}
    )
    assert resp.status_code == 400


def test_transcribe_file_decode_error_maps_to_400() -> None:
    # Undecodable upload bytes fail in negotiation -> 400.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_recording_array8k_factory"))
    client = TestClient(app)

    files = {"file": ("audio.wav", b"this is not audio", "audio/wav")}
    resp: httpx.Response = client.post("/v1/transcribe", data={"model": "dummy/echo"}, files=files)
    assert resp.status_code == 400


def test_transcribe_json_internal_error_maps_to_500() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_fail_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500


def test_transcribe_file_success_and_internal_error() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    data = {"model": "dummy/echo"}
    response: httpx.Response = client.post("/v1/transcribe", data=data, files=files)
    assert response.status_code == 200
    assert response.json()["result"]["text"] == "dummy"

    app_fail = server_module.create_app(registry=_registry_for("_fail_factory"))
    client_fail = TestClient(app_fail)
    response = client_fail.post("/v1/transcribe", data=data, files=files)
    assert response.status_code == 500


def test_run_handles_missing_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "uvicorn":
            raise ImportError("uvicorn not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        server_module.run()


def test_run_calls_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    uvicorn_stub = types.ModuleType("uvicorn")
    setattr(uvicorn_stub, "called", False)
    setattr(uvicorn_stub, "kwargs", {})

    def _run(app: Any, **kwargs: Any) -> None:
        setattr(uvicorn_stub, "called", True)
        setattr(uvicorn_stub, "kwargs", kwargs)

    uvicorn_stub.run = _run  # type: ignore[attr-defined]

    monkeypatch.setitem(__import__("sys").modules, "uvicorn", uvicorn_stub)

    def _create_app() -> str:
        return "app"

    monkeypatch.setattr(server_module, "create_app", _create_app)

    server_module.run(host="127.0.0.1", port=9999, reload=False, log_level="warning")

    assert getattr(uvicorn_stub, "called") is True
    kwargs = getattr(uvicorn_stub, "kwargs")
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9999


def test_capabilities_endpoint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    resp: httpx.Response = client.get("/v1/capabilities/dummy/echo")
    assert resp.status_code == 200
    body = resp.json()
    assert body["batch"]["language"]["runtime_override"]["supported"] is True


def test_capabilities_endpoint_unknown_model() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    resp: httpx.Response = client.get("/v1/capabilities/nope/missing")
    assert resp.status_code == 404


def test_params_schema_endpoint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    resp: httpx.Response = client.get("/v1/params-schema/dummy/echo")
    assert resp.status_code == 200
    schema = resp.json()
    assert "beam" in schema.get("properties", {})


def test_transcribe_client_error_maps_to_422() -> None:
    """UnsupportedFeatureError (client-caused) must map to 422, not 500."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_client_error_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422
    assert "word_timestamps" in resp.json()["detail"]


def test_transcribe_unknown_model_maps_to_404() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    payload = {"model": "nope/missing", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 404


def test_transcribe_500_does_not_leak_internal_detail() -> None:
    """Unexpected errors return a generic message; raw text stays server-side."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_fail_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "/secret/internal/path" not in detail
    assert "See server logs" in detail


def test_transcribe_construction_config_error_maps_to_422() -> None:
    """A ConfigError raised during engine construction is client-caused -> 422.

    Construction errors must be mapped exactly like transcription errors; a bad
    config / missing credential must not escape as a non-spec 500.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_config_error_construct_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422
    # The safe message is surfaced; it is not a traceback.
    assert "missing API key" in resp.json()["detail"]
    assert "Traceback" not in resp.json()["detail"]


def test_transcribe_construction_unexpected_error_maps_to_500_no_leak() -> None:
    """An unexpected construction fault -> generic 500 with no internal detail."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # _NoInstantiateASR.__init__ raises a RuntimeError carrying internal text.
    app = server_module.create_app(registry=_registry_for("_no_instantiate_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "instantiation forbidden" not in detail
    assert "See server logs" in detail


def test_transcribe_file_construction_config_error_maps_to_422() -> None:
    # The multipart endpoint maps construction config errors identically.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_config_error_construct_factory"))
    client = TestClient(app)

    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    resp: httpx.Response = client.post("/v1/transcribe", data={"model": "dummy/echo"}, files=files)
    assert resp.status_code == 422


def test_ws_stream_construction_config_error_reports_bad_request() -> None:
    # A client config error during engine construction (e.g. missing credential)
    # is surfaced as a pre-bridge bad_request frame, not a route crash.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_config_error_construct_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "bad_request"
    assert "missing API key" in err["message"]


def test_ws_stream_construction_unexpected_error_reports_internal_no_leak() -> None:
    # An unexpected construction fault must not crash the route or leak detail:
    # a single generic internal_error frame is sent instead.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_no_instantiate_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "internal_error"
    assert "instantiation forbidden" not in err["message"]
    assert "See server logs" in err["message"]


def test_body_size_limit_returns_413() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry(), max_body_bytes=64)
    client = TestClient(app)

    big = base64.b64encode(b"x" * 1024).decode()
    payload = {"model": "dummy/echo", "audio": big}
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 413


def test_create_app_rejects_nonpositive_max_body() -> None:
    pytest.importorskip("fastapi")
    with pytest.raises(ValueError):
        server_module.create_app(registry=_registry(), max_body_bytes=0)


class _AudioErrorASR(_DummyASR):
    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        from standard_asr.exceptions import AudioProcessingError

        raise AudioProcessingError("bad audio frames")


def _audio_error_factory() -> _AudioErrorASR:  # pyright: ignore[reportUnusedFunction]
    return _AudioErrorASR()


class _NoCapsASR(_DummyASR):
    declared_capabilities: ClassVar[DeclaredCapabilities | None] = None  # type: ignore[assignment]
    provider_params_type: ClassVar[type[ProviderParams] | None] = None


def _no_caps_factory() -> _NoCapsASR:  # pyright: ignore[reportUnusedFunction]
    return _NoCapsASR()


def test_transcribe_audio_error_maps_to_400() -> None:
    # AudioProcessingError raised inside transcribe maps to 400.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_audio_error_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 400
    assert "bad audio frames" in resp.json()["detail"]


def test_transcribe_json_with_options_builds_params() -> None:
    # A non-null options object is parsed into RuntimeParams (the _build_params
    # validate path).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode(),
        "options": {"language": "en"},
    }
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 200


def test_transcribe_json_with_bad_options_maps_to_422() -> None:
    # A semantically invalid options object in the JSON body (a malformed
    # language tag) is an unprocessable entity (422), not a malformed-syntax 400.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode(),
        "options": {"language": "english"},
    }
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422


def test_transcribe_file_with_bad_options_maps_to_400() -> None:
    # A malformed options JSON string in the multipart form is a client error.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    data = {"model": "dummy/echo", "options": "{not json}"}
    resp: httpx.Response = client.post("/v1/transcribe", data=data, files=files)
    assert resp.status_code == 400


def test_body_validation_error_does_not_echo_input() -> None:
    # A body-validation failure (here: wrong type for `audio`) must NOT reflect
    # the offending submitted value -- FastAPI's default handler would echo it.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    # `audio` must be a string; send a recognisable sentinel as the wrong type.
    resp: httpx.Response = client.post(
        "/v1/transcribe:json", json={"model": "dummy/echo", "audio": 1234567890}
    )
    assert resp.status_code == 422
    assert "1234567890" not in resp.text


def test_body_validation_error_redacts_credential_field_value() -> None:
    # A mis-placed secret (an `api_key` put at the top level of the JSON body)
    # is rejected by extra="forbid"; its value must be redacted, never bounced
    # back to the client / proxy / a copied bug report.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    secret = "sk-LEAKED-SECRET-VALUE"
    resp: httpx.Response = client.post(
        "/v1/transcribe:json",
        json={"model": "dummy/echo", "audio": "Zm9v", "api_key": secret},
    )
    assert resp.status_code == 422
    assert secret not in resp.text
    detail = resp.json()["detail"]
    # The credential entry is present (so the caller knows what to fix) but its
    # message is redacted.
    assert any("api_key" in entry["loc"] for entry in detail)
    assert any(entry["msg"] == "[redacted]" for entry in detail)


def test_options_validation_error_does_not_echo_secret() -> None:
    # A secret mis-placed inside `options` reaches _build_params, whose pydantic
    # str(exc) would otherwise echo input_value=. The sanitized message must not
    # contain it (and the offending field is named).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    secret = "sk-OPTIONS-SECRET"
    payload = {
        "model": "dummy/echo",
        "audio": "Zm9v",
        "options": {"api_key": secret},
    }
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422
    assert secret not in resp.text
    assert "api_key" in resp.json()["detail"]


def test_options_validation_error_message_omits_input_value() -> None:
    # A malformed language tag in options must surface a useful message but never
    # the raw input_value pydantic would otherwise embed.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    payload = {
        "model": "dummy/echo",
        "audio": "Zm9v",
        "options": {"language": "definitely-not-a-tag-XYZ"},
    }
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "language" in detail
    assert "input_value" not in detail


def test_transcribe_file_options_validation_error_is_sanitized() -> None:
    # The multipart endpoint's options (valid JSON, invalid RuntimeParams) take
    # the sanitized ValidationError branch (a mis-placed secret is not echoed).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    secret = "sk-MULTIPART-SECRET"
    data = {"model": "dummy/echo", "options": json.dumps({"api_key": secret})}
    resp: httpx.Response = client.post("/v1/transcribe", data=data, files=files)
    assert resp.status_code == 422
    assert secret not in resp.text
    assert "api_key" in resp.json()["detail"]


def test_transcribe_json_rejects_provider_params_over_wire() -> None:
    # D5: provider_params is discover-only, never sendable. A request whose
    # options carry it must be rejected with a clear 422 (not silently dropped
    # or mis-routed into the internal model).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    payload = {
        "model": "dummy/echo",
        "audio": "Zm9v",
        "options": {"language": "en", "provider_params": {"beam": 5}},
    }
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422
    assert "provider_params" in resp.json()["detail"]


def test_transcribe_file_rejects_provider_params_over_wire() -> None:
    # The multipart endpoint enforces the same portable-only wire contract.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    data = {"model": "dummy/echo", "options": json.dumps({"provider_params": {"beam": 5}})}
    resp: httpx.Response = client.post("/v1/transcribe", data=data, files=files)
    assert resp.status_code == 422
    assert "provider_params" in resp.json()["detail"]


def test_transcribe_json_portable_params_still_work() -> None:
    # A request carrying only portable params validates and transcribes normally.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode(),
        "options": {
            "language": "en",
            "word_timestamps": "word",
            "prompt": "hello",
            "phrase_hints": ["foo"],
            "on_unsupported": "degrade_to_prompt",
        },
    }
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 200
    assert resp.json()["result"]["text"] == "dummy"


def test_ws_rejects_provider_params_over_wire() -> None:
    # The WS config-frame path shares _build_params; provider_params in its
    # options must be rejected (bad_request), never reach the session.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json(
            {
                "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                "options": {"provider_params": {"beam": 5}},
            }
        )
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "bad_request"
    assert "provider_params" in err["message"]


def test_loc_to_list_wraps_a_scalar() -> None:
    # Defensive: a scalar (non tuple/list) loc is wrapped into a single-element
    # list so the redaction scan can iterate it uniformly.
    assert server_module._loc_to_list("api_key") == ["api_key"]  # pyright: ignore[reportPrivateUsage]
    assert server_module._loc_to_list(("a", 0)) == ["a", 0]  # pyright: ignore[reportPrivateUsage]


def test_validation_error_with_non_string_loc_index_is_handled() -> None:
    # A bad element inside a list field yields a loc with an int index
    # (e.g. ["candidate_languages", 0]); the redaction scan must skip the
    # non-string component without error.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    payload = {
        "model": "dummy/echo",
        "audio": "Zm9v",
        "options": {"candidate_languages": [123]},
    }
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422
    assert "candidate_languages" in resp.json()["detail"]


def test_ws_options_validation_error_does_not_echo_secret() -> None:
    # The WS config-frame path shares _build_params; a mis-placed secret in
    # options must not be echoed in the bad_request frame.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    secret = "sk-WS-OPTIONS-SECRET"
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json(
            {
                "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                "options": {"api_key": secret},
            }
        )
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "bad_request"
    assert secret not in json.dumps(err)
    assert "api_key" in err["message"]


def test_capabilities_endpoint_none_caps_returns_404() -> None:
    # An engine class with declared_capabilities=None has no caps to serve.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_no_caps_factory"))
    client = TestClient(app)
    resp: httpx.Response = client.get("/v1/capabilities/dummy/echo")
    assert resp.status_code == 404
    assert "No capabilities" in resp.json()["detail"]


def test_params_schema_endpoint_none_returns_empty() -> None:
    # An engine with no provider_params_type publishes an empty schema.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_no_caps_factory"))
    client = TestClient(app)
    resp: httpx.Response = client.get("/v1/params-schema/dummy/echo")
    assert resp.status_code == 200
    assert resp.json() == {}


def test_invalid_content_length_returns_400() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    # A non-integer Content-Length must be rejected by the body-size middleware.
    resp: httpx.Response = client.post(
        "/v1/transcribe:json",
        content=b"{}",
        headers={"Content-Length": "not-a-number", "Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert "Invalid Content-Length" in resp.json()["detail"]


def test_body_size_middleware_passes_non_http_scope() -> None:
    # Non-HTTP scopes (websocket / lifespan) must pass straight through to the
    # wrapped app without the Content-Length inspection.
    import asyncio

    forwarded: list[str] = []

    async def _inner(scope: Any, receive: Any, send: Any) -> None:
        forwarded.append(scope["type"])

    mw = server_module._BodySizeLimitMiddleware(_inner, max_body_bytes=10)  # pyright: ignore[reportPrivateUsage]

    async def _noop() -> dict[str, Any]:
        return {}

    async def _send(_: Any) -> None:
        return None

    asyncio.run(mw({"type": "lifespan", "headers": []}, _noop, _send))
    assert forwarded == ["lifespan"]


def test_body_size_middleware_counts_streamed_bytes_and_suppresses_app_response() -> None:
    # The true-cap layer: an oversize body delivered as multiple chunks (no
    # honest Content-Length) is rejected with 413 the moment the cumulative count
    # exceeds the cap; the app keeps reading past the breach (covering the
    # already-rejected branch and the disconnect passthrough) and its own late
    # response is suppressed so it cannot clobber the 413.
    import asyncio

    # Two over-cap body chunks (the cap is 4): the first breaches and triggers
    # the 413; a (deliberately stubborn) app keeps reading, so the wrapper is
    # re-entered on a second over-cap chunk and must NOT emit a second 413 (the
    # already-rejected branch), then yields a disconnect.
    incoming: list[dict[str, Any]] = [
        {"type": "http.request", "body": b"aaaaa", "more_body": True},
        {"type": "http.request", "body": b"bbbbb", "more_body": False},
    ]

    async def _receive() -> dict[str, Any]:
        if incoming:
            return incoming.pop(0)
        return {"type": "http.disconnect"}

    sent: list[dict[str, Any]] = []

    async def _send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def _app(scope: Any, receive: Any, send: Any) -> None:
        # A stubborn body-reading app: pull a couple of frames even past a
        # disconnect (forcing the wrapper to re-enter after rejection), then try
        # to respond (this late response must be suppressed).
        for _ in range(3):
            await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"late"})

    mw = server_module._BodySizeLimitMiddleware(_app, max_body_bytes=4)  # pyright: ignore[reportPrivateUsage]
    asyncio.run(mw({"type": "http", "headers": []}, _receive, _send))

    # Exactly one response was emitted: the middleware's 413 (the app's 200 +
    # body were suppressed).
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1
    assert starts[0]["status"] == 413
    assert not any(m.get("body") == b"late" for m in sent)


def test_body_size_middleware_within_cap_streamed_passes_through() -> None:
    # A streamed body within the cap passes through untouched: the app reads all
    # frames and its own response is delivered (no 413, no suppression).
    import asyncio

    incoming: list[dict[str, Any]] = [
        {"type": "http.request", "body": b"ab", "more_body": True},
        {"type": "http.request", "body": b"c", "more_body": False},
    ]

    async def _receive() -> dict[str, Any]:
        if incoming:
            return incoming.pop(0)
        return {"type": "http.disconnect"}

    sent: list[dict[str, Any]] = []

    async def _send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def _app(scope: Any, receive: Any, send: Any) -> None:
        total = 0
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                break
            total += len(message.get("body", b""))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": str(total).encode()})

    mw = server_module._BodySizeLimitMiddleware(_app, max_body_bytes=4)  # pyright: ignore[reportPrivateUsage]
    asyncio.run(mw({"type": "http", "headers": []}, _receive, _send))

    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1 and starts[0]["status"] == 200
    # The app saw the full 3-byte body.
    assert any(m.get("body") == b"3" for m in sent)


def test_transcribe_file_over_limit_without_content_length() -> None:
    # A chunked upload (no Content-Length) bypasses the early middleware guard;
    # the handler still rejects the materialised oversize body with 413.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry(), max_body_bytes=8)
    client = TestClient(app)

    # Build a multipart body by hand and stream it via an iterator so httpx omits
    # Content-Length (Transfer-Encoding: chunked), defeating the early guard.
    boundary = "----stdasrboundary"
    big_file = b"x" * 64
    parts = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="model"\r\n\r\n'
            "dummy/echo\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + big_file
        + f"\r\n--{boundary}--\r\n".encode()
    )

    def _gen() -> Any:
        yield parts

    resp: httpx.Response = client.post(
        "/v1/transcribe",
        content=_gen(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"]


def test_transcribe_json_over_limit_without_content_length() -> None:
    # The JSON endpoint must reject an over-limit encoded payload too, even when
    # a chunked request (no Content-Length) slips past the early middleware guard.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry(), max_body_bytes=8)
    client = TestClient(app)

    body = json.dumps({"model": "dummy/echo", "audio": "x" * 64}).encode()

    def _gen() -> Any:
        yield body

    resp: httpx.Response = client.post(
        "/v1/transcribe:json",
        content=_gen(),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"]


def test_capabilities_no_instantiation() -> None:
    """capabilities/params-schema must NOT instantiate the engine (DoS / auth)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # _NoInstantiateASR.__init__ raises; reading ClassVars must still work.
    app = server_module.create_app(registry=_registry_for("_no_instantiate_factory"))
    client = TestClient(app)

    caps: httpx.Response = client.get("/v1/capabilities/dummy/echo")
    assert caps.status_code == 200
    assert caps.json()["batch"]["language"]["runtime_override"]["supported"] is True

    schema: httpx.Response = client.get("/v1/params-schema/dummy/echo")
    assert schema.status_code == 200
    assert "beam" in schema.json().get("properties", {})


# --- WebSocket streaming surface ---------------------------------------------


class _StreamProperties(BaseProperties):
    engine_id: str = "stream"
    model_name: str = "echo"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | Literal["any"] = [16000]
    selectable_languages: list[str] = []
    wire_encodings: list[str] | None = ["pcm_s16le"]


class _StreamEchoSession(TranscriptionSession):
    """Emits one final per fed chunk (its decoded text), then the base done."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        index = 0
        async for chunk in self.audio_chunks():
            yield TranscriptionEvent.final(
                f"seg-{index}",
                chunk.decode("utf-8", "replace"),
                start=float(index),
                end=float(index + 1),
            )
            index += 1


class _StreamEchoEngine(EngineBase):
    properties: ClassVar[BaseProperties] = _StreamProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
    )

    def __init__(self) -> None:
        self.config = _DummyConfig(engine="stream")

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        return TranscriptionResult(text="")  # batch path unused by these tests

    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        return _StreamEchoSession()


def _stream_echo_factory() -> _StreamEchoEngine:  # pyright: ignore[reportUnusedFunction]
    return _StreamEchoEngine()


def test_ws_stream_happy_path() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json(
            {"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}, "options": None}
        )
        ws.send_bytes(b"abc")
        ws.send_bytes(b"de")
        ws.send_text("end")  # any text frame signals end-of-audio
        events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "done":
                break
    finals = {e["text"] for e in events if e["type"] == "final"}
    assert finals == {"abc", "de"}


def test_ws_stream_bad_config_reports_error() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"no_audio_format": True})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "bad_request"


def test_ws_stream_unknown_model_reports_error() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/nope/missing") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "unknown_model"


def test_ws_stream_non_streaming_engine_reports_unsupported() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # _RecordingArray8kASR is a batch-only EngineBase: start_transcription raises.
    app = server_module.create_app(registry=_registry_for("_recording_array8k_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "unsupported"


def test_ws_stream_client_disconnect_is_handled() -> None:
    # A client that leaves mid-stream must not crash the server: the bridge ends
    # the session and stops forwarding (covers the disconnect + send-failure path).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        ws.send_bytes(b"abc")
        first = ws.receive_json()
        assert first["type"] == "final"
    # Exiting the context closes the socket without an end frame.


class _StreamErrorSession(TranscriptionSession):
    """Raises a detail-bearing exception so the base synthesizes an
    ``engine_error`` event whose ``extra['detail']`` carries the raw text."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        async for _chunk in self.audio_chunks():
            raise RuntimeError("boom: /secret/internal/path leaked")
        yield TranscriptionEvent.done()  # pragma: no cover - never reached


class _StreamErrorEngine(_StreamEchoEngine):
    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        return _StreamErrorSession()


def _stream_error_factory() -> _StreamErrorEngine:  # pyright: ignore[reportUnusedFunction]
    return _StreamErrorEngine()


def test_ws_stream_error_event_does_not_leak_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An engine that raises mid-stream surfaces an `error` event. Its raw
    # exception text MUST stay server-side (logged), never reach the client --
    # matching the REST 500 non-leak contract (server.md §3.7 / §4.2).
    import logging

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_error_factory"))
    client = TestClient(app)
    with caplog.at_level(logging.ERROR, logger="standard_asr.server"):
        with client.websocket_connect("/v1/stream/dummy/echo") as ws:
            ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
            ws.send_bytes(b"abc")
            ws.send_text("end")
            events: list[dict[str, Any]] = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] == "error":
                    break

    error = next(e for e in events if e["type"] == "error")
    assert error["code"] == "engine_error"
    # The structured fields survive; the raw detail is scrubbed from the frame.
    assert error["recoverable"] is False
    assert error["extra"] == {}
    assert "/secret/internal/path" not in json.dumps(error)
    # The dropped detail is logged server-side for operators.
    assert any("/secret/internal/path" in rec.getMessage() for rec in caplog.records)


def test_ws_stream_oversize_frame_rejected() -> None:
    # A single binary frame larger than the per-frame cap is rejected with a
    # policy error and the session is torn down (the HTTP body guard does not
    # cover the WS scope, so the bridge must cap frames itself).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(
        registry=_registry_for("_stream_echo_factory"), max_ws_frame_bytes=4
    )
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        ws.send_bytes(b"way too big")  # 11 bytes > 4-byte per-frame cap
        events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "error":
                break
    err = events[-1]
    assert err["code"] == "payload_too_large"
    assert "per-frame limit" in err["message"]


def test_ws_stream_cumulative_cap_rejected() -> None:
    # Each frame is within the per-frame cap, but their cumulative total exceeds
    # the per-session cap: the bridge rejects with the policy error.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(
        registry=_registry_for("_stream_echo_factory"),
        max_ws_frame_bytes=8,
        max_ws_session_bytes=5,
    )
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        ws.send_bytes(b"abc")  # 3 bytes (ok)
        ws.send_bytes(b"def")  # cumulative 6 > 5-byte session cap
        events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "error":
                break
    err = events[-1]
    assert err["code"] == "payload_too_large"
    assert "per-session limit" in err["message"]


def test_ws_stream_within_caps_still_works() -> None:
    # Within both caps, audio still flows and the session completes normally.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(
        registry=_registry_for("_stream_echo_factory"),
        max_ws_frame_bytes=16,
        max_ws_session_bytes=64,
    )
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        ws.send_bytes(b"abc")
        ws.send_text("end")
        events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "done":
                break
    assert {e["text"] for e in events if e["type"] == "final"} == {"abc"}


@pytest.mark.parametrize("kwargs", [{"max_ws_frame_bytes": 0}, {"max_ws_session_bytes": 0}])
def test_create_app_rejects_nonpositive_ws_caps(kwargs: dict[str, int]) -> None:
    pytest.importorskip("fastapi")
    with pytest.raises(ValueError):
        server_module.create_app(registry=_registry(), **kwargs)


def test_bridge_stream_pump_failure_is_logged_and_signalled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A protocol violation on the pump side (send_audio raising, e.g.
    # StreamClosedError) must NOT be silently swallowed: it is logged
    # server-side and surfaced to the client as a single generic, non-leaking
    # error frame (SERV-3).
    import asyncio
    import logging

    from standard_asr.exceptions import StreamClosedError

    class _FakeWS:
        def __init__(self) -> None:
            self._frames: list[dict[str, Any]] = [{"type": "websocket.receive", "bytes": b"abc"}]
            self.sent: list[Any] = []

        async def receive(self) -> dict[str, Any]:
            if self._frames:
                return self._frames.pop(0)
            return {"type": "websocket.disconnect"}

        async def send_json(self, data: Any) -> None:
            self.sent.append(data)

    class _FakeSession:
        def __init__(self) -> None:
            # Set when input ends so the producer terminates (mirrors a real
            # session: it does not emit `done` until input is ended).
            self._ended = asyncio.Event()

        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def send_audio(self, chunk: bytes) -> None:
            raise StreamClosedError("session already ended: /secret/path")

        async def end_audio(self) -> None:
            self._ended.set()

        def __aiter__(self) -> AsyncIterator[TranscriptionEvent]:
            async def _gen() -> AsyncIterator[TranscriptionEvent]:
                await self._ended.wait()
                yield TranscriptionEvent.done()

            return _gen()

    websocket = _FakeWS()
    with caplog.at_level(logging.ERROR, logger="standard_asr.server"):
        asyncio.run(
            server_module._bridge_stream(  # pyright: ignore[reportPrivateUsage]
                websocket,  # pyright: ignore[reportArgumentType]
                _FakeSession(),  # pyright: ignore[reportArgumentType]
                max_frame_bytes=1024,
                max_session_bytes=1024,
            )
        )
    # The failure was logged (with detail) and a generic error frame was sent.
    assert any("audio pump failed" in rec.getMessage().lower() for rec in caplog.records)
    error_frames = [f for f in websocket.sent if f.get("type") == "error"]
    assert error_frames and error_frames[-1]["code"] == "stream_input_error"
    assert "/secret/path" not in json.dumps(websocket.sent)


def test_bridge_stream_tolerates_send_failure() -> None:
    # If the client vanishes mid-stream, a failing send must not propagate: the
    # bridge swallows it, ends input, and tears the session down cleanly.
    import asyncio

    class _FakeWS:
        def __init__(self) -> None:
            self.send_attempted = False

        async def receive(self) -> dict[str, Any]:
            return {"type": "websocket.disconnect"}

        async def send_json(self, data: Any) -> None:
            self.send_attempted = True
            raise RuntimeError("client gone")

    class _FakeSession:
        def __init__(self) -> None:
            self.ended = False

        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def send_audio(self, chunk: bytes) -> None:  # pragma: no cover - unused
            return None

        async def end_audio(self) -> None:
            self.ended = True

        def __aiter__(self) -> AsyncIterator[TranscriptionEvent]:
            async def _gen() -> AsyncIterator[TranscriptionEvent]:
                yield TranscriptionEvent.done()

            return _gen()

    websocket = _FakeWS()
    asyncio.run(
        server_module._bridge_stream(  # pyright: ignore[reportPrivateUsage]
            websocket,  # pyright: ignore[reportArgumentType]
            _FakeSession(),  # pyright: ignore[reportArgumentType]
            max_frame_bytes=1024,
            max_session_bytes=1024,
        )
    )
    # The send was attempted and its failure was swallowed (the run completed).
    assert websocket.send_attempted is True
