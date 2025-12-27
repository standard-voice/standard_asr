"""Tests for FastAPI server helpers."""

from __future__ import annotations

import base64
import builtins
from importlib.metadata import EntryPoint
from typing import Any, ClassVar

import numpy as np
import pytest

from standard_asr import BaseConfig, BaseProperties, TranscriptionResult
from standard_asr.discovery import discover_models
from standard_asr.server import _decode_audio_payload, create_app, run


class _DummyConfig(BaseConfig[str]):
    engine: str = "dummy"


class _DummyProperties(BaseProperties):
    engine_id: str = "dummy"
    model_name: str = "echo"
    protocol_version: str = "0.2.0"
    supported_languages: list[str] = ["en"]
    supported_devices: list[str] = ["cpu"]
    supported_sample_rates: list[int] = [16000]
    supported_channels: list[int] = [1]
    audio_dtype: str = "float32"


class _DummyASR:
    properties: ClassVar[_DummyProperties] = _DummyProperties()

    def __init__(self) -> None:
        self.config = _DummyConfig(engine="dummy")

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="dummy")


def _dummy_factory() -> _DummyASR:
    return _DummyASR()


class _FailASR(_DummyASR):
    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        raise RuntimeError("boom")


def _fail_factory() -> _FailASR:
    return _FailASR()


def _registry():
    eps = [
        EntryPoint(
            name="dummy/echo",
            value="tests.test_server:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    return discover_models(eps=eps, strict=True)


def test_create_app_missing_fastapi(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "fastapi":
            raise ImportError("fastapi not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        create_app()


def test_create_app_endpoints(monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = create_app(registry=_registry())

    monkeypatch.setattr(
        "standard_asr.server.load_audio_from_bytes",
        lambda _: np.zeros(16000, dtype=np.float32),
    )

    client = TestClient(app)

    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()[0]["key"] == "dummy/echo"

    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode("utf-8"),
    }
    transcribe = client.post("/v1/transcribe:json", json=payload)
    assert transcribe.status_code == 200
    assert transcribe.json()["result"]["text"] == "dummy"


def test_transcribe_json_error_paths(monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = create_app(registry=_registry())
    client = TestClient(app)

    from standard_asr import server as server_module

    real_decode = server_module._decode_audio_payload
    monkeypatch.setattr(
        server_module,
        "_decode_audio_payload",
        lambda _: (_ for _ in ()).throw(ValueError("bad")),
    )
    payload = {"model": "dummy/echo", "audio": "bad"}
    response = client.post("/v1/transcribe:json", json=payload)
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
    app_fail = create_app(registry=registry)
    client_fail = TestClient(app_fail)

    monkeypatch.setattr(
        "standard_asr.server.load_audio_from_bytes",
        lambda _: np.zeros(16000, dtype=np.float32),
    )
    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode("utf-8"),
    }
    response = client_fail.post("/v1/transcribe:json", json=payload)
    assert response.status_code == 500


def test_transcribe_file_paths(monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = create_app(registry=_registry())
    client = TestClient(app)

    monkeypatch.setattr(
        "standard_asr.server.load_audio_from_bytes",
        lambda _: np.zeros(16000, dtype=np.float32),
    )

    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    data = {"model": "dummy/echo"}
    response = client.post("/v1/transcribe", data=data, files=files)
    assert response.status_code == 200
    assert response.json()["result"]["text"] == "dummy"

    monkeypatch.setattr(
        "standard_asr.server.load_audio_from_bytes",
        lambda _: (_ for _ in ()).throw(ValueError("bad")),
    )
    response = client.post("/v1/transcribe", data=data, files=files)
    assert response.status_code == 400

    eps = [
        EntryPoint(
            name="dummy/echo",
            value="tests.test_server:_fail_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    app_fail = create_app(registry=registry)
    client_fail = TestClient(app_fail)

    monkeypatch.setattr(
        "standard_asr.server.load_audio_from_bytes",
        lambda _: np.zeros(16000, dtype=np.float32),
    )
    response = client_fail.post("/v1/transcribe", data=data, files=files)
    assert response.status_code == 500
def test_decode_audio_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        "standard_asr.server.load_audio_from_bytes",
        lambda _: np.zeros(16000, dtype=np.float32),
    )
    monkeypatch.setattr(
        "standard_asr.server.load_audio",
        lambda _: np.zeros(16000, dtype=np.float32),
    )
    raw = base64.b64encode(b"fake-data").decode("utf-8")
    audio = _decode_audio_payload(raw)
    assert isinstance(audio, np.ndarray)

    uri = f"data:audio/wav;base64,{raw}"
    audio_uri = _decode_audio_payload(uri)
    assert isinstance(audio_uri, np.ndarray)

    with pytest.raises(ValueError):
        _decode_audio_payload("not-base64!!!")


def test_run_handles_missing_uvicorn(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "uvicorn":
            raise ImportError("uvicorn not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        run()


def test_run_calls_uvicorn(monkeypatch) -> None:
    import types

    uvicorn_stub = types.ModuleType("uvicorn")
    uvicorn_stub.called = False
    uvicorn_stub.kwargs = {}

    def _run(app, **kwargs):
        uvicorn_stub.called = True
        uvicorn_stub.kwargs = kwargs

    uvicorn_stub.run = _run  # type: ignore[attr-defined]

    monkeypatch.setitem(__import__("sys").modules, "uvicorn", uvicorn_stub)

    monkeypatch.setattr(
        "standard_asr.server.create_app",
        lambda: "app",
    )

    run(host="127.0.0.1", port=9999, reload=False, log_level="warning")

    assert uvicorn_stub.called is True
    assert uvicorn_stub.kwargs["host"] == "127.0.0.1"
    assert uvicorn_stub.kwargs["port"] == 9999
