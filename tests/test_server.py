"""Tests for FastAPI server helpers."""

from __future__ import annotations

import base64
import builtins
from importlib.metadata import EntryPoint
from typing import Any, ClassVar, Literal

import httpx
import numpy as np
import pytest
from numpy.typing import NDArray

from standard_asr import BaseConfig, BaseProperties, TranscriptionResult
from standard_asr import server as server_module
from standard_asr.audio_input import InputKind
from standard_asr.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    FlagCap,
    LanguageCaps,
)
from standard_asr.discovery import discover_models
from standard_asr.runtime_params import ProviderParams


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


def _fake_audio_bytes(
    _: bytes, target_sr: int = 16000, target_channels: int | None = 1
) -> NDArray[np.float32]:
    return np.zeros(target_sr, dtype=np.float32)


def _fake_audio(_: str) -> NDArray[np.float32]:
    return np.zeros(16000, dtype=np.float32)


def _raise_value_error_bytes(
    _: bytes, target_sr: int = 16000, target_channels: int | None = 1
) -> NDArray[np.float32]:
    raise ValueError("bad")


def _raise_value_error_str(_: str) -> NDArray[np.float32]:
    raise ValueError("bad")


def _registry():
    eps = [
        EntryPoint(
            name="dummy/echo",
            value="tests.test_server:_dummy_factory",
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


def test_create_app_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())

    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)

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


def test_transcribe_json_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    real_decode = server_module._decode_audio_payload  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(server_module, "_decode_audio_payload", _raise_value_error_str)
    payload = {"model": "dummy/echo", "audio": "bad"}
    response: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert response.status_code == 400

    monkeypatch.setattr(server_module, "_decode_audio_payload", real_decode)

    eps = [
        EntryPoint(
            name="dummy/echo",
            value="tests.test_server:_fail_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    app_fail = server_module.create_app(registry=registry)
    client_fail = TestClient(app_fail)

    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)
    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode("utf-8"),
    }
    response: httpx.Response = client_fail.post("/v1/transcribe:json", json=payload)
    assert response.status_code == 500


def test_transcribe_file_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)

    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    data = {"model": "dummy/echo"}
    response: httpx.Response = client.post("/v1/transcribe", data=data, files=files)
    assert response.status_code == 200
    assert response.json()["result"]["text"] == "dummy"

    monkeypatch.setattr(server_module, "load_audio_from_bytes", _raise_value_error_bytes)
    response: httpx.Response = client.post("/v1/transcribe", data=data, files=files)
    assert response.status_code == 400

    eps = [
        EntryPoint(
            name="dummy/echo",
            value="tests.test_server:_fail_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    app_fail = server_module.create_app(registry=registry)
    client_fail = TestClient(app_fail)

    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)
    response: httpx.Response = client_fail.post("/v1/transcribe", data=data, files=files)
    assert response.status_code == 500


def test_decode_audio_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)
    monkeypatch.setattr(server_module, "load_audio", _fake_audio)
    raw = base64.b64encode(b"fake-data").decode("utf-8")
    audio = server_module._decode_audio_payload(  # pyright: ignore[reportPrivateUsage]
        raw
    )
    assert isinstance(audio, np.ndarray)

    uri = f"data:audio/wav;base64,{raw}"
    audio_uri = server_module._decode_audio_payload(  # pyright: ignore[reportPrivateUsage]
        uri
    )
    assert isinstance(audio_uri, np.ndarray)

    with pytest.raises(ValueError):
        server_module._decode_audio_payload(  # pyright: ignore[reportPrivateUsage]
            "not-base64!!!"
        )


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


def _registry_for(factory: str):
    eps = [
        EntryPoint(
            name="dummy/echo",
            value=f"tests.test_server:{factory}",
            group="standard_asr.models",
        )
    ]
    return discover_models(eps=eps, strict=True)


def test_transcribe_client_error_maps_to_422(monkeypatch: pytest.MonkeyPatch) -> None:
    """UnsupportedFeatureError (client-caused) must map to 422, not 500."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_client_error_factory"))
    client = TestClient(app)
    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422
    assert "word_timestamps" in resp.json()["detail"]


def test_transcribe_unknown_model_maps_to_404(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)

    payload = {"model": "nope/missing", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 404


def test_transcribe_500_does_not_leak_internal_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected errors return a generic message; raw text stays server-side."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_fail_factory"))
    client = TestClient(app)
    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "/secret/internal/path" not in detail
    assert "See server logs" in detail


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


def test_transcribe_audio_error_maps_to_400(monkeypatch: pytest.MonkeyPatch) -> None:
    # AudioProcessingError raised inside transcribe (not decode) maps to 400.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_audio_error_factory"))
    client = TestClient(app)
    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 400
    assert "bad audio frames" in resp.json()["detail"]


def test_transcribe_json_with_options_builds_params(monkeypatch: pytest.MonkeyPatch) -> None:
    # A non-null options object is parsed into RuntimeParams (the _build_params
    # validate path).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)

    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode(),
        "options": {"language": "en"},
    }
    resp: httpx.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 200


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


def test_transcribe_file_over_limit_without_content_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A chunked upload (no Content-Length) bypasses the early middleware guard;
    # the handler still rejects the materialised oversize body with 413.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry(), max_body_bytes=8)
    client = TestClient(app)
    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)

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


def test_capabilities_no_instantiation(monkeypatch: pytest.MonkeyPatch) -> None:
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
