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

    Args:
        env_var: Environment variable that overrides the cache directory.
        os_name: Optional OS name override (useful for testing).

    Returns:
        Path to the cache directory.

    Raises:
        OSError: If the path cannot be resolved.
    """
    override = os.getenv(env_var)
    if override:
        return Path(override).expanduser()

    name = os_name if os_name is not None else os.name

    if name == "nt":
        root = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        if root:
            return Path(root) / "standard-asr"
    return Path.home() / ".cache" / "standard-asr"


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
]
