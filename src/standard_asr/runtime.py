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
    override = (os.getenv(env_var) or "").strip()
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
    explicit: Path | None = None, *, library_default: Path | None = None
) -> Path:
    """Resolve an engine's model download root per the spec IC.9 precedence.

    Implements the normative four-level chain engines MUST follow when picking
    where model artifacts land: **explicit** config (the engine's
    ``download_root`` field) > the ``STANDARD_ASR_MODEL_DIR`` environment
    override > the engine library's own default cache (when it has one) > the
    shared Standard ASR cache directory (:func:`resolve_cache_dir`,
    ``~/.cache/standard-asr`` or the platform equivalent). The environment tier
    inherits :func:`resolve_cache_dir`'s reading of the variable: a
    whitespace-only value is unset and a relative value resolves against the
    current working directory at call time.

    Args:
        explicit: Explicitly configured download root (highest priority); a
            leading ``~`` is expanded.
        library_default: The engine library's own default cache directory, used
            only when neither an explicit value nor the environment override is
            present; a leading ``~`` is expanded.

    Returns:
        The resolved download root.
    """
    if explicit is not None:
        return explicit.expanduser()
    if (os.getenv("STANDARD_ASR_MODEL_DIR") or "").strip():
        return resolve_cache_dir()
    if library_default is not None:
        return library_default.expanduser()
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
