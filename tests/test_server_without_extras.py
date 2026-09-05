# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0
"""Server behavior that holds without the ``[server]`` extra installed.

``tests/test_server.py`` skips itself when FastAPI is absent, because its tests
build the app or drive it through Starlette's test client. This module holds
the tests that need no extra: a missing FastAPI, the uvicorn launcher, and the
packaging contract of the extra itself. The core-floor CI lane, which installs
no extras, runs them.
"""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from standard_asr.toolchain import server as server_module


def test_create_app_missing_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "fastapi":
            raise ImportError("fastapi not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        server_module.create_app()


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

    create_app_kwargs: dict[str, Any] = {}

    def _create_app(**kwargs: Any) -> str:
        create_app_kwargs.update(kwargs)
        return "app"

    monkeypatch.setattr(server_module, "create_app", _create_app)

    server_module.run(
        host="127.0.0.1",
        port=9999,
        log_level="warning",
        max_ws_frame_bytes=4096,
    )

    assert getattr(uvicorn_stub, "called") is True
    kwargs = getattr(uvicorn_stub, "kwargs")
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9999
    # The WS per-frame cap is wired to uvicorn's transport ws_max_size so the
    # app-level bound and the transport bound match.
    assert kwargs["ws_max_size"] == 4096
    # The same cap is propagated to the app it builds.
    assert create_app_kwargs["max_ws_frame_bytes"] == 4096


def test_server_extra_declares_a_websocket_library() -> None:
    # Drift guard: server.md promises a WebSocket
    # streaming endpoint, but bare uvicorn ships no WS protocol implementation.
    # The documented `pip install standard-asr[server]` must therefore pull one
    # in, or /v1/stream answers 404 on upgrade in every user install while the
    # in-process TestClient suite stays green.
    import re
    from pathlib import Path

    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^server\s*=\s*\[(?P<deps>[^\]]*)\]", pyproject, re.MULTILINE)
    assert match is not None, "pyproject.toml must declare the [server] extra"
    deps = match.group("deps")
    assert "websockets" in deps or "wsproto" in deps, (
        "The [server] extra must include a WebSocket protocol library "
        "(websockets or wsproto); bare uvicorn cannot serve /v1/stream."
    )
