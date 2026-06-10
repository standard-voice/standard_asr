# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Runtime helpers for Standard ASR engines.

Download-policy and cache-directory resolution helpers used by engines during
lazy model loading (spec, section "Init Config", rule IC.9). Audio input
validation now lives in the negotiation layer
(:mod:`standard_asr.audio_negotiation`), not here.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_override(env_var: str) -> str:
    """Read an environment path override, treating whitespace-only as unset.

    The single reading of the "is the override set" rule shared by the IC.9
    consumers (:func:`resolve_cache_dir` and :func:`resolve_download_root`),
    so the two can never drift on it.

    Args:
        env_var: The environment variable name.

    Returns:
        The stripped value, or ``""`` when unset or whitespace-only.
    """
    return (os.getenv(env_var) or "").strip()


def allow_downloads(env_var: str = "STANDARD_ASR_ALLOW_DOWNLOAD") -> bool:
    """Return whether model downloads are allowed at runtime.

    Args:
        env_var: Environment variable name that controls download policy.

    Returns:
        ``True`` when downloads are allowed, otherwise ``False``.
    """
    value = os.getenv(env_var)
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "yes"}


def resolve_cache_dir(
    env_var: str = "STANDARD_ASR_MODEL_DIR", *, os_name: str | None = None
) -> Path:
    """Resolve the Standard ASR model cache directory.

    The ``env_var`` override is read first. A whitespace-only value is treated
    as unset (falls through to the platform default), and a relative value
    resolves against the current working directory *at call time*, so the
    override is always returned as an absolute path.

    Args:
        env_var: Environment variable that overrides the cache directory.
        os_name: Optional OS name override (useful for testing).

    Returns:
        Path to the cache directory.

    Raises:
        OSError: If the path cannot be resolved.
    """
    override = _env_override(env_var)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    name = os_name if os_name is not None else os.name

    if name == "nt":
        root = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        if root:
            return Path(root) / "standard-asr"
    return Path.home() / ".cache" / "standard-asr"


def resolve_download_root(
    explicit: Path | None = None, *, has_library_default: bool = False
) -> Path | None:
    """Resolve an engine's model download root per the spec IC.9 precedence.

    Implements the normative four-level chain engines MUST follow when picking
    where model artifacts land: **explicit** config (the engine's
    ``download_root`` field) > the ``STANDARD_ASR_MODEL_DIR`` environment
    override > the engine library's **own default cache** (when it has one;
    expressed as a ``None`` passthrough, see below) > the shared Standard ASR
    cache directory (:func:`resolve_cache_dir`, ``~/.cache/standard-asr`` or
    the platform equivalent). The environment tier inherits
    :func:`resolve_cache_dir`'s reading of the variable: a whitespace-only
    value is unset and a relative value resolves against the current working
    directory at call time.

    The library tier is a **passthrough**: engine libraries typically express
    "use my own default cache" as an unset download path (e.g. faster-whisper's
    ``WhisperModel(download_root=None)`` resolves via the HuggingFace hub
    cache), so when the chain lands on that tier this returns ``None`` and the
    adapter forwards it unchanged. Substituting a concrete directory here would
    delete the spec's third tier: every unconfigured install's models would
    relocate away from the library's existing cache, breaking offline loads of
    already-downloaded models and silently re-downloading them.

    Args:
        explicit: Explicitly configured download root (highest priority); a
            leading ``~`` is expanded.
        has_library_default: Whether the engine library has its own default
            model cache (the spec's third tier). When ``True`` and neither an
            explicit value nor the environment override is present, ``None``
            is returned for the adapter to forward; an adapter that knows the
            library's concrete cache path may substitute it for the ``None``.

    Returns:
        The resolved download root, or ``None`` when the engine library's own
        default cache applies (only possible with ``has_library_default``).
    """
    if explicit is not None:
        return explicit.expanduser()
    if _env_override("STANDARD_ASR_MODEL_DIR"):
        return resolve_cache_dir()
    if has_library_default:
        return None
    return resolve_cache_dir()


def ensure_cache_dir(
    env_var: str = "STANDARD_ASR_MODEL_DIR", *, os_name: str | None = None
) -> Path:
    """Ensure the Standard ASR cache directory exists.

    Args:
        env_var: Environment variable that overrides the cache directory.
        os_name: Optional OS name override (useful for testing).

    Returns:
        Path to the existing cache directory.

    Raises:
        OSError: If the directory cannot be created.
    """
    cache_dir = resolve_cache_dir(env_var=env_var, os_name=os_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


__all__ = [
    "allow_downloads",
    "ensure_cache_dir",
    "resolve_cache_dir",
    "resolve_download_root",
]
