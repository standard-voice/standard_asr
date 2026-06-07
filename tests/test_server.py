"""Tests for FastAPI server helpers."""

from __future__ import annotations

import base64
import builtins
from importlib.metadata import EntryPoint
from typing import Any, ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray

from standard_asr import BaseConfig, BaseProperties, TranscriptionResult
from standard_asr import server as server_module
from standard_asr.audio_input import InputKind
from standard_asr.discovery import discover_models


class _DummyConfig(BaseConfig[str]):
    engine: str = "dummy"


class _DummyProperties(BaseProperties):
    engine_id: str = "dummy"
    model_name: str = "echo"
    protocol_version: str = "0.2.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] = [16000]
    selectable_languages: list[str] = ["en"]


class _DummyASR:
    properties: ClassVar[_DummyProperties] = _DummyProperties()

    def __init__(self) -> None:
        self.config = _DummyConfig(engine="dummy")

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="dummy")


def _dummy_factory() -> _DummyASR:  # pyright: ignore[reportUnusedFunction]
    return _DummyASR()


class _FailASR(_DummyASR):
    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        raise RuntimeError("boom")


def _fail_factory() -> _FailASR:  # pyright: ignore[reportUnusedFunction]
    return _FailASR()


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


def test_transcribe_json_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    real_decode = server_module._decode_audio_payload  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(server_module, "_decode_audio_payload", _raise_value_error_str)
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
    app_fail = server_module.create_app(registry=registry)
    client_fail = TestClient(app_fail)

    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)
    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode("utf-8"),
    }
    response = client_fail.post("/v1/transcribe:json", json=payload)
    assert response.status_code == 500


def test_transcribe_file_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)

    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    data = {"model": "dummy/echo"}
    response = client.post("/v1/transcribe", data=data, files=files)
    assert response.status_code == 200
    assert response.json()["result"]["text"] == "dummy"

    monkeypatch.setattr(
        server_module, "load_audio_from_bytes", _raise_value_error_bytes
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
    app_fail = server_module.create_app(registry=registry)
    client_fail = TestClient(app_fail)

    monkeypatch.setattr(server_module, "load_audio_from_bytes", _fake_audio_bytes)
    response = client_fail.post("/v1/transcribe", data=data, files=files)
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
