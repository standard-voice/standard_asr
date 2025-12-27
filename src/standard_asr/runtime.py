"""Runtime helpers for Standard ASR engines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .asr_properties import BaseProperties
from .exceptions import AudioProcessingError


def allow_downloads(env_var: str = "STANDARD_ASR_ALLOW_DOWNLOAD") -> bool:
    """Return whether model downloads are allowed at runtime.

    Args:
        env_var: Environment variable name that controls download policy.

    Returns:
        ``True`` when downloads are allowed, otherwise ``False``.

    Raises:
        None.
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


def validate_audio_input(
    audio: NDArray[Any], properties: BaseProperties
) -> NDArray[np.float32]:
    """Validate audio input against Standard ASR properties.

    Args:
        audio: Input audio array.
        properties: Engine properties that describe supported formats.

    Returns:
        The same audio array cast to ``np.float32`` if needed.

    Raises:
        AudioProcessingError: If dtype or channel count is unsupported.
    """
    array = np.asarray(audio)
    if array.dtype != properties.numpy_dtype:
        try:
            array = array.astype(properties.numpy_dtype, copy=False)
        except (TypeError, ValueError) as exc:
            raise AudioProcessingError(
                "Audio dtype is not compatible with engine requirements."
            ) from exc

    if array.ndim == 1:
        channels = 1
    elif array.ndim == 2:
        channels = int(array.shape[1])
    else:
        raise AudioProcessingError("Audio must be 1D (mono) or 2D (multi-channel).")

    if channels not in properties.supported_channels:
        raise AudioProcessingError(
            f"Audio has {channels} channel(s); supported channels are {properties.supported_channels}."
        )

    return array.astype(np.float32, copy=False)


__all__ = [
    "allow_downloads",
    "ensure_cache_dir",
    "resolve_cache_dir",
    "validate_audio_input",
]
