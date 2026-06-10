# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for runtime helpers (download policy and cache dirs)."""

from pathlib import Path

import pytest

from standard_asr.runtime import (
    allow_downloads,
    ensure_cache_dir,
    resolve_cache_dir,
    resolve_download_root,
)


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


def test_cache_dir_whitespace_only_env_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # R3-DISCOVERY-03: a whitespace-only override carries no path; it MUST fall
    # through to the default, never resolve to a whitespace-named directory.
    monkeypatch.setenv("STANDARD_ASR_MODEL_DIR", "   ")
    with_whitespace = resolve_cache_dir()
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR")
    assert with_whitespace == resolve_cache_dir()


def test_cache_dir_relative_env_resolves_against_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # R3-DISCOVERY-03: a relative override resolves against the cwd at call time
    # so the returned path is absolute (not dependent on a later chdir).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STANDARD_ASR_MODEL_DIR", "models")
    resolved = resolve_cache_dir()
    assert resolved.is_absolute()
    assert resolved == Path.cwd() / "models"


# --------------------------------------------------------------------------- #
# resolve_download_root: the four-level spec IC.9 precedence chain.
# --------------------------------------------------------------------------- #
def test_download_root_explicit_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STANDARD_ASR_MODEL_DIR", str(tmp_path / "env"))
    explicit = tmp_path / "explicit"
    resolved = resolve_download_root(explicit, has_library_default=True)
    assert resolved == explicit


def test_download_root_env_beats_library_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STANDARD_ASR_MODEL_DIR", str(tmp_path / "env"))
    resolved = resolve_download_root(has_library_default=True)
    assert resolved == tmp_path / "env"


def test_download_root_library_default_passthrough_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FV-3: unconfigured + env unset on an engine whose library has its own
    # default cache resolves to the LIBRARY tier -- a None passthrough the
    # adapter forwards (e.g. WhisperModel(download_root=None) -> the HF hub
    # cache) -- never a forced concrete directory that would relocate every
    # unconfigured install's models.
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
    assert resolve_download_root(has_library_default=True) is None


def test_download_root_whitespace_env_falls_to_library_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The env tier reads the variable like resolve_cache_dir: whitespace-only is
    # unset, so the chain continues to the library-default passthrough.
    monkeypatch.setenv("STANDARD_ASR_MODEL_DIR", "   ")
    assert resolve_download_root(has_library_default=True) is None


def test_download_root_falls_back_to_standard_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without a library default the chain ends at the shared standard cache.
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
    assert resolve_download_root() == resolve_cache_dir()
