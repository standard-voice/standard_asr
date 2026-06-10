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


def test_allow_downloads_unrecognized_value_warns_and_disables(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A non-affirmative typo (e.g. ``on``) must fail safe to disabled
    # AND log once -- the engine that later raises DiscoveryError only sees the
    # boolean, so without this diagnostic the operator cannot trace the cause.
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "on")
    caplog.set_level("WARNING")
    assert allow_downloads() is False
    assert any("not a recognized value" in r.message for r in caplog.records)


def test_allow_downloads_empty_value_warns_and_disables(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # An empty string (a common ``VAR=`` docker-compose artifact) is
    # an unrecognized value -> fail safe to disabled, with a diagnostic. This is
    # the deliberate empty-string asymmetry with the cache path override.
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "")
    caplog.set_level("WARNING")
    assert allow_downloads() is False
    assert any("not a recognized value" in r.message for r in caplog.records)


def test_allow_downloads_recognized_disable_value_is_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A recognized disable value must NOT warn (only unrecognized values do).
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "false")
    caplog.set_level("WARNING")
    assert allow_downloads() is False
    assert not any("not a recognized value" in r.message for r in caplog.records)


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
    # With LOCALAPPDATA unset, the cache must derive the non-roaming
    # ~/AppData/Local/standard-asr (LOCALAPPDATA's standard location), NEVER the
    # roaming %APPDATA% (synced across domain logins -- wrong for GB weights) and
    # never the previous ~/.cache fallback.
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("APPDATA", str(Path.home() / "Roaming"))

    resolved = resolve_cache_dir(os_name="nt")
    assert resolved == Path.home() / "AppData" / "Local" / "standard-asr"
    # The roaming profile must NOT appear anywhere in the resolved path.
    assert "Roaming" not in resolved.parts


def test_cache_dir_windows_ignores_roaming_appdata(monkeypatch: pytest.MonkeyPatch) -> None:
    # APPDATA (roaming) is no longer consulted at all -- even when
    # present, only LOCALAPPDATA (or its derived default) is used.
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("APPDATA", "/should/not/be/used")

    resolved = resolve_cache_dir(os_name="nt")
    assert "should" not in resolved.parts


def test_cache_dir_xdg_cache_home_honored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # On POSIX an absolute XDG_CACHE_HOME is honoured (parity with
    # HuggingFace hub / pip / uv), so weights follow a user's deliberately
    # relocated cache instead of always landing in ~/.cache.
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    resolved = resolve_cache_dir(os_name="posix")
    assert resolved == tmp_path / "standard-asr"


def test_cache_dir_xdg_relative_value_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The XDG Base Directory spec requires a non-absolute
    # XDG_CACHE_HOME to be ignored; resolution falls through to ~/.cache.
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")

    resolved = resolve_cache_dir(os_name="posix")
    assert resolved == Path.home() / ".cache" / "standard-asr"


def test_cache_dir_posix_default_without_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no override and no XDG_CACHE_HOME, the POSIX default is
    # ~/.cache/standard-asr (unchanged, HF-ecosystem-aligned).
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    resolved = resolve_cache_dir(os_name="posix")
    assert resolved == Path.home() / ".cache" / "standard-asr"


def test_cache_dir_whitespace_only_env_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # A whitespace-only override carries no path; it MUST fall
    # through to the default, never resolve to a whitespace-named directory.
    monkeypatch.setenv("STANDARD_ASR_MODEL_DIR", "   ")
    with_whitespace = resolve_cache_dir()
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR")
    assert with_whitespace == resolve_cache_dir()


def test_cache_dir_relative_env_resolves_against_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A relative override resolves against the cwd at call time
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
    # Unconfigured + env unset on an engine whose library has its own
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
