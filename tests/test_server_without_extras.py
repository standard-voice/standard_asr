# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0
"""Server behavior that holds without the ``[server]`` extra installed.

``tests/test_server.py`` skips itself when FastAPI is absent, because its tests
drive the app through Starlette's test client. This module holds the test that
is about that absence, so the core-floor CI lane, which installs no extras,
still runs it.
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
