# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for runtime helpers (download policy and cache dirs)."""

from pathlib import Path

import pytest

from standard_asr.runtime import allow_downloads, ensure_cache_dir, resolve_cache_dir


def test_allow_downloads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STANDARD_ASR_ALLOW_DOWNLOAD", raising=False)
    assert allow_downloads() is True

    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    assert allow_downloads() is False

    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "yes")
    assert allow_downloads() is True


def test_cache_dir_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STANDARD_ASR_MODEL_DIR", str(tmp_path))
    resolved = resolve_cache_dir()
    assert resolved == tmp_path

    ensured = ensure_cache_dir()
    assert ensured.exists()


def test_cache_dir_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
    resolved = resolve_cache_dir()
    assert isinstance(resolved, Path)


def test_cache_dir_windows_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    resolved = resolve_cache_dir(os_name="nt")
    assert resolved == tmp_path / "standard-asr"


def test_cache_dir_windows_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)

    resolved = resolve_cache_dir(os_name="nt")
    assert resolved.name == "standard-asr"
