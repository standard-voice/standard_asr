# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for CLI and server error boundaries."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from standard_asr.contract.exceptions import ArtifactStatusError
from standard_asr.runtime.engine_pool import EnginePool
from standard_asr.toolchain import cli
from standard_asr.toolchain import server as server_module


class _CliRegistry:
    """Return one engine with a deliberately invalid provider declaration."""

    class _Engine:
        provider_params_type = dict

    def create(self, _model: str, **_config: object) -> _Engine:
        return self._Engine()


def test_cli_transcribe_rejects_an_invalid_provider_params_declaration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI classifies a malformed provider type as an engine fault."""

    def discover(**_kwargs: object) -> _CliRegistry:
        return _CliRegistry()

    monkeypatch.setattr(cli, "discover_models", discover)

    exit_code = cli.main(["transcribe", "dummy/model", "audio.wav"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "provider_params_type" in captured.err


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (None, "readable UTF-8 JSON file"),
        ("[]", "mapping model keys to config objects"),
    ],
)
def test_cli_serve_rejects_unreadable_or_non_object_engine_configs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    contents: str | None,
    message: str,
) -> None:
    """The serve command rejects invalid config documents before startup."""
    path = tmp_path / "engines.json"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")

    exit_code = cli.main(["serve", "--engine-configs", str(path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert message in captured.err


class _StatusEngine:
    """Minimal pooled engine whose readiness inspection fails."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.close_calls = 0

    def artifact_status(self) -> None:
        raise self._error

    def close(self) -> None:
        self.close_calls += 1


class _ServerRegistry:
    """Expose one engine to the routes under test."""

    def __init__(self, create: Callable[[], Any] | None = None) -> None:
        self._create = create

    def names(self) -> list[str]:
        return ["dummy/model"]

    def create(self, model: str, **_config: object) -> Any:
        assert model == "dummy/model"
        if self._create is None:
            raise AssertionError("The route must not construct an engine")
        return self._create()


@pytest.mark.parametrize(
    "error",
    [ArtifactStatusError("native status failed"), RuntimeError("unexpected status failure")],
)
def test_readiness_scrubs_expected_and_unexpected_inspection_failures(error: Exception) -> None:
    """Readiness faults return the same stable, non-leaking server response."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    engine = _StatusEngine(error)
    app = server_module.create_app(
        registry=_ServerRegistry(lambda: engine),  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        response = client.get("/v1/readiness/dummy/model")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Internal artifact readiness error. See server logs for details."
    )
    assert "native status failed" not in response.text
    assert "unexpected status failure" not in response.text
    assert engine.close_calls == 1


def test_websocket_handshake_internal_failure_is_scrubbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected receive failures produce one generic internal error frame."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    async def fail_receive(*_args: object) -> dict[str, object]:
        raise RuntimeError("handshake leaked /private/path")

    monkeypatch.setattr(server_module, "_receive_config_frame", fail_receive)
    app = server_module.create_app(registry=_ServerRegistry())  # type: ignore[arg-type]

    with TestClient(app) as client:
        with client.websocket_connect("/v1/stream/dummy/model") as websocket:
            frame = websocket.receive_json()

    assert frame["type"] == "error"
    assert frame["code"] == "internal_error"
    assert "See server logs" in frame["message"]
    assert "/private/path" not in frame["message"]


def test_websocket_refuses_new_work_after_pool_shutdown() -> None:
    """Streaming receives a stable availability frame once shutdown starts."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_ServerRegistry())  # type: ignore[arg-type]
    with TestClient(app) as client:
        pool: EnginePool = app.state.engine_pool
        pool._closing = True  # pyright: ignore[reportPrivateUsage]
        with client.websocket_connect("/v1/stream/dummy/model") as websocket:
            websocket.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
            frame = websocket.receive_json()

    assert frame == {
        "type": "error",
        "code": "service_unavailable",
        "message": "The server is shutting down and is not accepting new streaming work.",
    }


def test_rest_refuses_new_work_after_pool_shutdown() -> None:
    """Batch requests receive 503 once the application starts shutting down."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_ServerRegistry())  # type: ignore[arg-type]
    with TestClient(app) as client:
        pool: EnginePool = app.state.engine_pool
        pool._closing = True  # pyright: ignore[reportPrivateUsage]
        response = client.post(
            "/v1/transcribe:json",
            json={"model": "dummy/model", "audio": "ZmFrZQ=="},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The server is shutting down and is not accepting new transcription work."
    )
