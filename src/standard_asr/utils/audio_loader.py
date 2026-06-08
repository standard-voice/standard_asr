# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Standard ASR Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Audio loading and normalization utilities for the Standard ASR audio contract.

This module focuses on turning a wide range of audio inputs into a normalized
NumPy array suitable for ASR models. It deliberately prefers light-weight,
standard-library options first, then optional libraries, and finally a
system-level FFmpeg fallback for maximum compatibility.

Key points and contract:
- Output dtype is always `np.float32`.
- Output value range is clipped to [-1.0, 1.0].
- Output shape is `(n_samples,)` for mono, `(n_samples, n_channels)` for multi.
- Resampling is performed with ``scipy.signal.resample_poly`` if sample rates
  differ (requires `scipy`).
- Channel handling:
  - Downmix to mono uses arithmetic mean.
  - Upmix replicates channels (e.g., 1 -> 2).
  - Multi->fewer channels down-mix by truncation (first N channels) unless
    loading is delegated to FFmpeg, which can provide higher-quality mixing.
- NaN/Inf are sanitized to safe values (NaN->0.0, +Inf->1.0, -Inf->-1.0).

Dependencies and fallbacks:
1) WAV via stdlib `wave` (8/16-bit PCM)
2) `soundfile` (if installed) for many formats and in-memory bytes
3) FFmpeg subprocess fallback (requires ffmpeg in PATH)

If `soundfile` succeeds but resampling requires `scipy` and it is missing, the
loader falls back to FFmpeg (with a warning) so decoding can still succeed when
FFmpeg is available.

All functions emit clear exceptions with actionable messages and log helpful
warnings when quality-affecting fallbacks are used.
"""

import base64
import io
import logging
import math
import pathlib
import shutil
import subprocess
import wave
from typing import Any, BinaryIO, Literal, TypeGuard, cast, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray

from ..exceptions import (
    AudioProcessingError,
    FFmpegNotFoundError,
)

logger = logging.getLogger(__name__)

#: Hard ceiling (bytes) on a single buffered decode, guarding against
#: decompression bombs / oversize inputs when no engine limit is supplied
#: (spec R9). 2 GiB comfortably covers multi-hour PCM while bounding memory.
_DEFAULT_MAX_DECODE_BYTES = 2 * 1024 * 1024 * 1024

# --- Public API ---


def decode_base64_audio(value: str) -> bytes:
    """Decode a base64 audio payload, optionally wrapped in a ``data:`` URI.

    Single source of truth for the ``data:``-URI/base64 parse rules, shared by
    the convenience loaders, :func:`decode_audio`, and the conversion layer so
    every entry point accepts and rejects exactly the same forms (AUDI-4).

    Rules:

    * A ``data:`` URI MUST carry the explicit ``;base64,`` marker; the bytes
      after it are base64-decoded. A ``data:`` URI without ``;base64,`` (e.g. a
      percent-encoded ``data:audio/wav,...``) is rejected rather than silently
      treated as base64.
    * Any other string is treated as a bare base64 payload.

    Validation is strict (``validate=True``): non-base64 characters fail loudly
    rather than being silently dropped.

    Args:
        value: A ``data:...;base64,...`` URI or a bare base64 string.

    Returns:
        The decoded bytes.

    Raises:
        AudioProcessingError: If a ``data:`` URI lacks the ``;base64,`` marker,
            or the payload is not valid base64.
    """
    if value.startswith("data:"):
        marker = ";base64,"
        if marker not in value:
            raise AudioProcessingError(
                "Malformed data: URI -- only base64-encoded data URIs are "
                "supported; the ';base64,' marker is required."
            )
        payload = value.split(marker, 1)[1]
    else:
        payload = value
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise AudioProcessingError("Invalid base64 audio payload.") from exc


def _validate_local_source_path(path: str) -> str:
    """Validate and absolutize a local file path before handing it to ffmpeg.

    This is the file-input half of the bare-str-never-URL defense (design
    decision D1): ffmpeg must only ever open a real local file, never a network
    URL (``http://``, ``tcp://``), a protocol-chaining input (``concat:``,
    ``data:``) or an option-injection string (a leading ``-``).

    Args:
        path: The candidate local file path.

    Returns:
        The resolved absolute path as a string.

    Raises:
        AudioProcessingError: If the path is empty, looks like a CLI option, or
            does not resolve to an existing regular file.
    """
    if not path or path.startswith("-"):
        raise AudioProcessingError(
            "Refusing to decode an audio source that begins with '-' "
            "(possible option injection). Pass a real local file path."
        )
    resolved = pathlib.Path(path).expanduser()
    if not resolved.is_file():
        raise AudioProcessingError(
            f"Audio file not found or not a regular file: {path!r}. "
            "Bare strings are always treated as local file paths; wrap URLs in "
            "AudioUrl and base64 in AudioBase64."
        )
    return str(resolved.resolve())


def _enforce_decode_size(num_bytes: int, max_bytes: int | None) -> None:
    """Raise if a buffered payload exceeds the allowed decode size (spec R9).

    Args:
        num_bytes: Size of the payload about to be buffered, in bytes.
        max_bytes: Engine-declared limit, or ``None`` to use the default cap.

    Raises:
        AudioProcessingError: If ``num_bytes`` exceeds the effective limit.
    """
    limit = max_bytes if max_bytes is not None else _DEFAULT_MAX_DECODE_BYTES
    if num_bytes > limit:
        raise AudioProcessingError(
            f"Audio payload is {num_bytes} bytes, exceeding the decode limit of "
            f"{limit} bytes. Provide a smaller input or an engine without this "
            "limit."
        )


# numpy datatype check
@overload
def ensure_datatype(audio: NDArray[Any]) -> NDArray[np.float32]: ...


@overload
def ensure_datatype(audio: NDArray[Any], data_type: Literal["float32"]) -> NDArray[np.float32]: ...


@overload
def ensure_datatype(
    audio: NDArray[Any], data_type: np.dtype[np.float32]
) -> NDArray[np.float32]: ...


@overload
def ensure_datatype(audio: NDArray[Any], data_type: type[np.float32]) -> NDArray[np.float32]: ...


@overload
def ensure_datatype(audio: NDArray[Any], data_type: DTypeLike) -> NDArray[Any]: ...


def ensure_datatype(audio: NDArray[Any], data_type: DTypeLike = np.float32) -> NDArray[Any]:
    """Convert a NumPy array to the specified dtype (default: ``float32``).

    Args:
        audio: Input NumPy array of any dtype.
        data_type: Target dtype (e.g., ``"float32"``, ``np.float32``). Default: ``np.float32``.

    Returns:
        NumPy array with the requested dtype. Returns a view (no copy) if already matching.

    Raises:
        TypeError: If ``data_type`` is not a valid NumPy dtype.

    Example:
        >>> audio = ensure_datatype(raw_audio)  # -> float32
        >>> audio = ensure_datatype(raw_audio, "float64")  # -> float64
    """
    # Compute the target dtype for runtime comparison; helps static checkers.
    # Use np.asarray (not astype(copy=False)) per spec DEP.2 / D4: copy=False is
    # banned because its no-copy guarantee differs subtly across numpy 1.x/2.x.
    target_dtype: np.dtype[np.generic] = np.dtype(data_type)
    if audio.dtype != target_dtype:
        audio = np.asarray(audio, dtype=target_dtype)
    return audio


def normalize_audio(
    audio: NDArray[Any],
    original_sr: int,
    target_sr: int = 16000,
    target_channels: int | None = 1,
) -> NDArray[np.float32]:
    """Normalize a raw waveform to the Standard ASR audio format.

    **Standard ASR Audio Format:**

    - **dtype:** ``np.float32``
    - **Sample rate:** ``16000`` Hz (configurable)
    - **Channels:** Mono ``(n_samples,)`` or multi-channel ``(n_samples, n_channels)``
    - **Value range:** ``[-1.0, 1.0]``

    Args:
        audio: Input waveform, 1D ``(n_samples,)`` or 2D ``(n_samples, n_channels)``.
            Any dtype (will be converted to ``float32``).
        original_sr: Sample rate of the input audio (Hz). Must be > 0.
        target_sr: Target sample rate. Default: ``16000`` Hz.
        target_channels: Target channel count. ``1`` = mono (default), ``2`` = stereo,
            ``None`` = preserve original.

    Returns:
        Normalized waveform: ``np.float32``, resampled to ``target_sr``, with
        ``target_channels`` channels, values clipped to ``[-1.0, 1.0]``.

    Raises:
        AudioProcessingError: Invalid parameters or empty audio.

    Note:
        **Resampling:** Uses ``scipy.signal.resample_poly`` for high-quality
        conversion when ``scipy`` is installed (the ``[audio]`` extra), and
        degrades to the built-in numpy-only anti-aliasing Fourier resampler
        otherwise (a missing extra is never fatal, spec AI R8).

        **Channel conversion:**

        - Stereo → Mono: arithmetic mean of channels.
        - Mono → Stereo: channel replication.
        - Multi-channel down-mix: truncates to first N channels (for better quality,
          use FFmpeg path via ``load_audio``).

        **Invalid values:** NaN/Inf are sanitized (NaN→0.0, ±Inf→±1.0) with a warning.
    """
    processed_audio: NDArray[np.float32] = ensure_datatype(audio, "float32")
    # Basic parameter validation

    if original_sr <= 0:
        raise AudioProcessingError(f"original_sr must be > 0, got {original_sr}")
    if target_sr <= 0:
        raise AudioProcessingError(f"target_sr must be > 0, got {target_sr}")
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(f"target_channels must be None or > 0, got {target_channels}")

    # Check for empty audio
    if processed_audio.size == 0:
        raise AudioProcessingError("Cannot process empty audio array")

    # 1. Resample if necessary. Prefer scipy's polyphase resampler (high quality,
    #    available via the [audio] extra); fall back to the core numpy-only
    #    anti-aliasing Fourier resampler so a missing extra is never fatal
    #    (spec AI R8).
    if original_sr != target_sr:
        try:
            from math import gcd

            from scipy.signal import (  # pyright: ignore[reportMissingTypeStubs]
                resample_poly as _resample_poly,  # pyright: ignore[reportUnknownVariableType]
            )

            g = gcd(original_sr, target_sr)
            up, down = target_sr // g, original_sr // g
            processed_audio = np.asarray(
                _resample_poly(processed_audio, up=up, down=down, axis=0),  # pyright: ignore[reportUnknownArgumentType]
                dtype=np.float32,
            )
        except ImportError:
            from ..resampling import resample as _fallback_resample

            logger.warning(
                "scipy not installed; using the built-in anti-aliasing fallback "
                "resampler. Install standard-asr[audio] for higher quality."
            )
            processed_audio = _fallback_resample(processed_audio, original_sr, target_sr)

    # Ensure at least 2D for uniform channel processing
    if processed_audio.ndim == 1:
        processed_audio = processed_audio[:, np.newaxis]

    # Help static type checker with tuple[int, ...] for shape
    current_channels = int(processed_audio.shape[1])

    # 2. Channel conversion
    if target_channels is not None and current_channels != target_channels:
        if target_channels == 1:
            # Downmix to mono using average across channels
            processed_audio = processed_audio.mean(axis=1, dtype=np.float32)[:, np.newaxis]
        else:
            # Note: For multi-to-multi down-mixing (e.g., 6->2), this implementation performs a
            # simple channel selection/truncation instead of a perceptually accurate mix.
            # For higher quality down-mix, ensure FFmpeg is installed and prefer the FFmpeg path.
            if target_channels > current_channels:  # Upscale (e.g., mono to stereo)
                reps = int(math.ceil(target_channels / current_channels))
                processed_audio = np.tile(processed_audio, (1, reps))[:, :target_channels]
            else:  # Down-mix by truncation
                logger.warning(
                    "Down-mixing from %d to %d channels by taking the first %d channels. "
                    "This may result in information loss. For high-quality down-mixing, "
                    "ensure your audio is processed via the FFmpeg backend.",
                    current_channels,
                    target_channels,
                    target_channels,
                )
                processed_audio = processed_audio[:, :target_channels]

    # Clean up any NaN/Inf values that might have been introduced during processing
    if not np.isfinite(processed_audio).all():
        bad_count = (~np.isfinite(processed_audio)).sum()
        logger.warning(
            "Detected %d invalid samples (NaN/Inf) in audio; replacing with safe values.",
            int(bad_count),
        )
        processed_audio = np.nan_to_num(processed_audio, nan=0.0, posinf=1.0, neginf=-1.0)

    # Clip to contract range, then cast. Clip BEFORE cast (DEP.2 defensive
    # ordering) and use np.asarray instead of astype(copy=False) (DEP.2 ban).
    processed_audio = np.asarray(np.clip(processed_audio, -1.0, 1.0), dtype=np.float32)
    # Respect contract: mono->1D, multi->2D even if n_samples==1
    if int(processed_audio.shape[1]) == 1:
        return processed_audio[:, 0]
    return processed_audio


def load_audio(
    source: str | bytes | bytearray | memoryview | pathlib.Path | BinaryIO,
    target_sr: int = 16000,
    target_channels: int | None = 1,
) -> NDArray[np.float32]:
    """Load audio from given source and convert to Standard ASR format.

    **Accepts:** File path, ``bytes``, ``pathlib.Path``, base64 data URI, or file-like object.

    **Returns:** ``np.float32`` array, 16kHz mono by default, values in ``[-1.0, 1.0]``.

    Convenience loader for application code that wants a decoded array directly
    (e.g. to plot or pre-process audio). It handles format detection, decoding,
    resampling, and channel conversion automatically, and for a ``str`` it
    auto-detects a base64 ``data:`` URI.

    This is **not** the engine input boundary. When transcribing, pass an
    :data:`~standard_asr.audio_input.AudioInput` to ``transcribe`` and let the
    standard negotiation layer decode/convert per the engine's ``accepted_input``
    -- there a bare ``str`` is **always** a file path and is never sniffed (a
    security boundary against SSRF / data-URI confusion). This helper's
    convenience sniffing is intentional and local to it.

    Args:
        source: Audio input. Supported types:

            - ``str``: File path (``"audio.mp3"``) or base64 data URI
              (``"data:audio/wav;base64,..."``).
            - ``bytes`` / ``bytearray`` / ``memoryview``: Raw audio file bytes.
            - ``pathlib.Path``: File path object.
            - ``BinaryIO``: File-like object opened in binary mode.

        target_sr: Output sample rate in Hz. Default: ``16000``.
        target_channels: Output channels. ``1`` = mono (default), ``2`` = stereo,
            ``None`` = preserve.

    Returns:
        Waveform as ``np.float32`` array:

        - Shape: ``(n_samples,)`` for mono, ``(n_samples, n_channels)`` for multi-channel.
        - Sample rate: ``target_sr`` Hz.
        - Values: Clipped to ``[-1.0, 1.0]``.

    Raises:
        AudioProcessingError: Invalid parameters or decoding/processing failures,
            including missing or unreadable paths.
        FFmpegNotFoundError: FFmpeg fallback needed but not installed.
        TypeError: Unsupported source type.

    Example:
        >>> audio = load_audio("speech.mp3")  # Load from file
        >>> audio = load_audio(audio_bytes)   # Load from bytes
        >>> audio = load_audio(Path("~/audio.wav"), target_sr=8000)  # Custom sample rate

    Note:
        **Decoding priority:**
        - File paths: stdlib ``wave`` → ``soundfile`` → FFmpeg subprocess.
        - Bytes/data URIs/BinaryIO: ``soundfile`` → FFmpeg subprocess.

        **Base64:** Only data URIs (``data:...;base64,...``) are auto-detected.
        For raw base64 strings (eg. ``YmFT...Y0``), decode manually:
        ``load_audio(base64.b64decode(s))``.

        **BinaryIO:** Reads from the current stream position; does not seek to the beginning.

        **Formats:** WAV, MP3, FLAC, OGG, and any format supported by FFmpeg.
    """
    if target_sr <= 0:
        raise AudioProcessingError(f"target_sr must be > 0, got {target_sr}")
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(f"target_channels must be None or > 0, got {target_channels}")
    if isinstance(source, str):
        # Improved base64 detection logic to avoid false positives
        s = source.strip()

        # First check: if it has explicit base64 data URI prefix, treat as base64
        if s.lower().startswith("data:") and ";base64," in s:
            source_bytes = decode_base64_audio(s)
            return load_audio_from_bytes(source_bytes, target_sr, target_channels)

        # Second check: if it exists as a file path, prioritize as path
        try:
            path = pathlib.Path(s).expanduser()
            if path.exists():
                return load_audio_from_path(str(path), target_sr, target_channels)
        except (OSError, ValueError):
            # Path operations failed, continue to treat as file path anyway
            pass

        # Default: treat as file path (will raise FileNotFoundError if not found)
        return load_audio_from_path(s, target_sr, target_channels)

    # Handle pathlib.Path by converting to string
    if isinstance(source, pathlib.Path):
        return load_audio_from_path(str(source), target_sr, target_channels)

    # Bytes-like objects
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = source.tobytes() if isinstance(source, memoryview) else bytes(source)
        return load_audio_from_bytes(data, target_sr, target_channels)

    # File-like object that returns bytes
    if _is_binary_io(source):
        data = source.read()
        return load_audio_from_bytes(data, target_sr, target_channels)

    raise TypeError(f"Unsupported audio source type: {type(source)}")


# --- Public Loaders ---


def _is_binary_io(obj: Any) -> TypeGuard[BinaryIO]:
    """Check if an object is a binary IO stream (internal helper).

    Args:
        obj: Any object to check.

    Returns:
        ``True`` if ``obj`` is a binary IO (returns bytes on read), ``False`` otherwise.

    Raises:
        None.
    """
    try:
        if isinstance(obj, (io.BufferedIOBase, io.BytesIO, io.RawIOBase)):
            return True
        read_attr = getattr(obj, "read", None)
        if not callable(read_attr):
            return False
        # Probe a zero-length read without consuming data. We avoid peek/read1/seek
        # for broad compatibility and to not alter stream position/state. Some custom
        # streams may not support read(0); in such case we return False.
        sample = read_attr(0)  # pyright: ignore[reportCallIssue]
        return isinstance(sample, (bytes, bytearray, memoryview))
    except Exception:
        return False


def load_audio_from_path(
    path: str, target_sr: int = 16000, target_channels: int | None = 1
) -> NDArray[np.float32]:
    """Load audio from a file path and convert to Standard ASR format.

    **Accepts:** File path as string (e.g., ``"speech.mp3"``, ``"~/audio.wav"``).

    **Returns:** ``np.float32`` array, resampled and channel-converted.

    Args:
        path: Path to audio file. Supports ``~`` expansion.
        target_sr: Output sample rate (Hz). Default: ``16000``.
        target_channels: Output channels. ``1`` = mono, ``2`` = stereo, ``None`` = preserve.

    Returns:
        Waveform as ``np.float32``, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.

    Raises:
        AudioProcessingError: Decoding failed, including missing or unreadable files.
        FFmpegNotFoundError: FFmpeg fallback needed but not installed.

    Note:
        **Decoding priority:** stdlib ``wave`` (WAV) → ``soundfile`` → FFmpeg.

        For broader format support, install ``soundfile`` or ensure FFmpeg is in PATH.
    """
    # Basic parameter validation
    if target_sr <= 0:
        raise AudioProcessingError(f"target_sr must be > 0, got {target_sr}")
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(f"target_channels must be None or > 0, got {target_channels}")

    # Expand user (~) to avoid surprises across platforms
    from os import fspath

    path = fspath(pathlib.Path(path).expanduser())
    # Layer 1: WAV files with Python standard library `wave`
    if path.lower().endswith(".wav"):
        try:
            with wave.open(path, "rb") as wf:
                orig_sr = wf.getframerate()
                sampwidth = wf.getsampwidth()
                n_channels = wf.getnchannels()
                # Only handle 8-bit (unsigned) and 16-bit PCM via stdlib; others fallback
                if sampwidth not in (1, 2):
                    raise AudioProcessingError(
                        f"Unsupported WAV sample width via stdlib: {sampwidth * 8} bits"
                    )

                frames = wf.readframes(wf.getnframes())
                # 16-bit PCM is little-endian by the WAV/canonical contract (spec R4):
                # read it with an explicit "<i2" dtype so a big-endian host does
                # not silently byte-swap samples (AUDI-2). uint8 has no endianness.
                dtype_map: dict[int, np.dtype[np.unsignedinteger | np.signedinteger]] = {
                    1: np.dtype(np.uint8),
                    2: np.dtype("<i2"),
                }
                audio = np.frombuffer(frames, dtype=dtype_map[sampwidth]).astype(np.float32)
                if sampwidth == 1:
                    # 8-bit unsigned PCM: convert to [-1, 1]
                    audio = audio - 128.0
                    audio = audio / 128.0
                else:
                    # 16-bit signed PCM
                    audio = audio / 32768.0

                # Re-affirm dtype for static checker after arithmetic
                audio = np.asarray(audio, dtype=np.float32)

                if n_channels > 1:
                    audio = audio.reshape(-1, n_channels)

                return normalize_audio(audio, orig_sr, target_sr, target_channels)
        except (wave.Error, AudioProcessingError, OSError, ValueError) as e:
            logger.debug(
                f"Could not load WAV with stdlib `wave` (unsupported format or corrupted file), "
                f"falling back to soundfile/ffmpeg... Error: {e}"
            )

    # Layer 2: Use `soundfile` for formats like FLAC, OGG, etc.
    try:
        import soundfile as sf  # pyright: ignore[reportMissingTypeStubs]

        sf_read: Any = getattr(sf, "read")
        audio, orig_sr = cast(tuple[NDArray[np.float32], int], sf_read(path, dtype="float32"))
        # normalize_audio never raises ImportError: a missing scipy degrades to
        # the built-in anti-aliasing fallback resampler internally (spec AI R8).
        return normalize_audio(audio, orig_sr, target_sr, target_channels)
    except ImportError:
        logger.debug("`soundfile` not installed, cannot load non-WAV formats without FFmpeg.")
    except Exception as e:
        logger.debug(f"Could not load with `soundfile`, falling back... Error: {e}")

    # Layer 3: Final fallback to FFmpeg
    return _load_with_ffmpeg(path, target_sr, target_channels)


def load_audio_from_bytes(
    data: bytes, target_sr: int = 16000, target_channels: int | None = 1
) -> NDArray[np.float32]:
    """Load audio from raw bytes and convert to Standard ASR format.

    **Accepts:** Audio file content as ``bytes`` (e.g., from ``file.read()``, HTTP response).

    **Returns:** ``np.float32`` array, resampled and channel-converted.

    Args:
        data: Raw bytes of an audio file (any format: WAV, MP3, FLAC, etc.).
        target_sr: Output sample rate (Hz). Default: ``16000``.
        target_channels: Output channels. ``1`` = mono, ``2`` = stereo, ``None`` = preserve.

    Returns:
        Waveform as ``np.float32``, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.

    Raises:
        AudioProcessingError: Decoding failed or empty audio.
        FFmpegNotFoundError: FFmpeg fallback needed but not installed.

    Note:
        **Decoding priority:** ``soundfile`` → FFmpeg. Install one for format support.
    """
    # Basic parameter validation
    if target_sr <= 0:
        raise AudioProcessingError(f"target_sr must be > 0, got {target_sr}")
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(f"target_channels must be None or > 0, got {target_channels}")
    # Layer 2: `soundfile` is the best primary method for bytes
    try:
        import soundfile as sf  # pyright: ignore[reportMissingTypeStubs]

        sf_read: Any = getattr(sf, "read")
        audio, orig_sr = cast(
            tuple[NDArray[np.float32], int], sf_read(io.BytesIO(data), dtype="float32")
        )
        # normalize_audio never raises ImportError: a missing scipy degrades to
        # the built-in anti-aliasing fallback resampler internally (spec AI R8).
        return normalize_audio(audio, orig_sr, target_sr, target_channels)
    except ImportError:
        logger.debug("`soundfile` not installed, cannot load from bytes without FFmpeg.")
    except Exception as e:
        logger.debug(f"Could not load bytes with `soundfile`, falling back... Error: {e}")

    # Layer 3: Final fallback to FFmpeg
    return _load_with_ffmpeg(data, target_sr, target_channels)


def decode_audio(
    source: str | bytes | bytearray | memoryview | pathlib.Path,
    *,
    target_channels: int | None = 1,
    max_bytes: int | None = None,
) -> tuple[NDArray[np.float32], int]:
    """Decode audio to a waveform at its **native** sample rate.

    Unlike :func:`load_audio`, this primitive does **not** resample: it returns
    the decoded waveform together with the source's original sample rate, so the
    caller can make the single authoritative resampling decision (spec R7). This
    is what the conversion layer needs to honour 8 kHz telephony and 24 kHz
    realtime engines without a spurious round-trip through 16 kHz (spec R7, the
    "MUST NOT upsample native-rate input" clause).

    Args:
        source: File path, ``pathlib.Path``, raw bytes, or a base64 ``data:``
            URI string. A bare ``str`` is always a local file path.
        target_channels: Output channels. ``1`` = mono (default), ``None`` =
            preserve the source channel layout.
        max_bytes: Optional cap on the buffered payload size (spec R9). Defaults
            to a generous internal ceiling when ``None``.

    Returns:
        A ``(array, native_sample_rate)`` pair. The array is ``float32`` in
        ``[-1, 1]`` at the source's original sample rate.

    Raises:
        AudioProcessingError: Decoding failed, the input is missing/oversize, or
            looks like an injected option.
        FFmpegNotFoundError: FFmpeg fallback needed but not installed.
        TypeError: Unsupported source type.
    """
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(f"target_channels must be None or > 0, got {target_channels}")

    if isinstance(source, str):
        s = source.strip()
        if s.lower().startswith("data:") and ";base64," in s:
            decoded = decode_base64_audio(s)
            _enforce_decode_size(len(decoded), max_bytes)
            return _decode_bytes_native(decoded, target_channels)
        path = _validate_local_source_path(s)
        _enforce_decode_size(pathlib.Path(path).stat().st_size, max_bytes)
        return _decode_path_native(path, target_channels)

    if isinstance(source, pathlib.Path):
        path = _validate_local_source_path(str(source))
        _enforce_decode_size(pathlib.Path(path).stat().st_size, max_bytes)
        return _decode_path_native(path, target_channels)

    # Defensive: the annotation narrows to bytes-like here, but this is a public
    # boundary that must reject mistyped runtime input gracefully.
    if not isinstance(source, (bytes, bytearray, memoryview)):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"Unsupported audio source type: {type(source)}")
    data = source.tobytes() if isinstance(source, memoryview) else bytes(source)
    _enforce_decode_size(len(data), max_bytes)
    return _decode_bytes_native(data, target_channels)


def _decode_path_native(path: str, target_channels: int | None) -> tuple[NDArray[np.float32], int]:
    """Decode a (validated, existing) file path to ``(array, native_sr)``.

    Args:
        path: A validated absolute local file path.
        target_channels: Output channels, or ``None`` to preserve.

    Returns:
        The decoded ``float32`` waveform and its native sample rate.

    Raises:
        AudioProcessingError: Decoding failed.
        FFmpegNotFoundError: FFmpeg fallback needed but not installed.
    """
    if path.lower().endswith(".wav"):
        try:
            with wave.open(path, "rb") as wf:
                orig_sr = wf.getframerate()
                sampwidth = wf.getsampwidth()
                n_channels = wf.getnchannels()
                if sampwidth not in (1, 2):
                    raise AudioProcessingError(
                        f"Unsupported WAV sample width via stdlib: {sampwidth * 8} bits"
                    )
                frames = wf.readframes(wf.getnframes())
                # 16-bit PCM is little-endian by the WAV/canonical contract (spec R4):
                # read it with an explicit "<i2" dtype so a big-endian host does
                # not silently byte-swap samples (AUDI-2). uint8 has no endianness.
                dtype_map: dict[int, np.dtype[np.unsignedinteger | np.signedinteger]] = {
                    1: np.dtype(np.uint8),
                    2: np.dtype("<i2"),
                }
                audio = np.frombuffer(frames, dtype=dtype_map[sampwidth]).astype(np.float32)
                if sampwidth == 1:
                    audio = (audio - 128.0) / 128.0
                else:
                    audio = audio / 32768.0
                audio = np.asarray(audio, dtype=np.float32)
                if n_channels > 1:
                    audio = audio.reshape(-1, n_channels)
                return normalize_audio(audio, orig_sr, orig_sr, target_channels), orig_sr
        except (wave.Error, AudioProcessingError, OSError, ValueError) as e:
            logger.debug("stdlib wave decode failed, falling back. Error: %s", e)

    try:
        import soundfile as sf  # pyright: ignore[reportMissingTypeStubs]

        sf_read: Any = getattr(sf, "read")
        audio, orig_sr = cast(tuple[NDArray[np.float32], int], sf_read(path, dtype="float32"))
        return normalize_audio(audio, orig_sr, orig_sr, target_channels), orig_sr
    except ImportError:
        logger.debug("`soundfile` not installed; using FFmpeg for native decode.")
    except Exception as e:
        logger.debug("soundfile decode failed, falling back. Error: %s", e)

    return _decode_with_ffmpeg_native(path, target_channels)


def _decode_bytes_native(
    data: bytes, target_channels: int | None
) -> tuple[NDArray[np.float32], int]:
    """Decode raw bytes to ``(array, native_sr)`` without resampling.

    Args:
        data: Encoded audio bytes.
        target_channels: Output channels, or ``None`` to preserve.

    Returns:
        The decoded ``float32`` waveform and its native sample rate.

    Raises:
        AudioProcessingError: Decoding failed.
        FFmpegNotFoundError: FFmpeg fallback needed but not installed.
    """
    try:
        import soundfile as sf  # pyright: ignore[reportMissingTypeStubs]

        sf_read: Any = getattr(sf, "read")
        audio, orig_sr = cast(
            tuple[NDArray[np.float32], int], sf_read(io.BytesIO(data), dtype="float32")
        )
        return normalize_audio(audio, orig_sr, orig_sr, target_channels), orig_sr
    except ImportError:
        logger.debug("`soundfile` not installed; using FFmpeg for native decode.")
    except Exception as e:
        logger.debug("soundfile decode failed, falling back. Error: %s", e)

    return _decode_with_ffmpeg_native(data, target_channels)


def _decode_with_ffmpeg_native(
    source: str | bytes, target_channels: int | None
) -> tuple[NDArray[np.float32], int]:
    """Decode via FFmpeg preserving the native sample rate.

    Args:
        source: A validated local file path, or raw bytes.
        target_channels: Output channels, or ``None`` to preserve.

    Returns:
        The decoded ``float32`` waveform and its native sample rate.

    Raises:
        AudioProcessingError: If the native rate cannot be determined or decoding
            fails.
        FFmpegNotFoundError: FFmpeg not in PATH.
    """
    native_sr = _probe_sample_rate_with_ffprobe(source)
    if native_sr is None:
        raise AudioProcessingError(
            "Could not determine the native sample rate via ffprobe; install "
            "ffprobe or the [audio] extra (soundfile) for native-rate decoding."
        )
    array = _load_with_ffmpeg(source, native_sr, target_channels)
    return array, native_sr


def _load_with_ffmpeg(
    source: str | bytes,
    target_sr: int,
    target_channels: int | None,
    timeout: float = 120.0,
) -> NDArray[np.float32]:
    """Decode audio via FFmpeg subprocess (internal fallback).

    Args:
        source: File path or raw bytes.
        target_sr: Output sample rate (Hz).
        target_channels: Output channels, or ``None`` to auto-detect via ffprobe.
        timeout: Max seconds before aborting. Default: ``120.0``.

    Returns:
        Waveform as ``np.float32``, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.

    Raises:
        FFmpegNotFoundError: FFmpeg not in PATH.
        AudioProcessingError: Decoding failed, timeout, or empty output.
    """
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError(
            "FFmpeg not found in PATH. Install via: 'brew install ffmpeg' (macOS), "
            "'sudo apt-get install ffmpeg' (Debian/Ubuntu), 'winget install ffmpeg' or "
            "'choco install ffmpeg' (Windows)."
        )

    # Security (D1 / spec R5 rationale): a string source is a *local file* and
    # nothing else. Validate + absolutize it up front (before probing), and
    # constrain ffmpeg/ffprobe to the file/pipe protocols so they can never be
    # coerced into fetching http(s)://, tcp://, concat:, data:, etc. via a
    # crafted input string.
    if isinstance(source, bytes):
        input_arg = "pipe:0"
        probe_source: str | bytes = source
    else:
        input_arg = _validate_local_source_path(source)
        probe_source = input_arg

    # If target_channels is None, attempt to preserve original channels via ffprobe.
    if target_channels is None:
        detected_channels = _probe_channels_with_ffprobe(probe_source)
        if detected_channels is None:
            logger.warning(
                "ffprobe not available or failed to detect channels. "
                "Defaulting to mono (1 channel)."
            )
            final_target_channels = 1
        else:
            final_target_channels = detected_channels
    else:
        final_target_channels = target_channels

    cmd = [
        "ffmpeg",
        "-nostdin",  # Prevent FFmpeg from waiting for stdin
        "-nostats",
        "-loglevel",
        "error",
        "-threads",
        "0",  # Use optimal number of threads
        "-protocol_whitelist",
        "file,pipe",  # Disallow network/chaining protocols (LFI/SSRF defense).
        "-i",
        input_arg,
        # Output options (must follow the input):
        "-vn",
        "-sn",
        "-dn",
        "-map",
        "a:0",  # Explicitly select the first audio stream
        "-f",
        "f32le",  # Output format: 32-bit floating-point, little-endian
        "-ac",
        str(final_target_channels),  # Set number of audio channels
        "-ar",
        str(target_sr),  # Set audio sample rate
        "-",  # Pipe output to stdout
    ]

    input_data = source if isinstance(source, bytes) else None

    try:
        proc = subprocess.run(
            cmd, capture_output=True, input=input_data, check=True, timeout=timeout
        )
        if not proc.stdout:
            raise AudioProcessingError("FFmpeg produced no audio data.")
        audio: NDArray[np.float32] = np.frombuffer(proc.stdout, dtype=np.float32)

        # Contract guarantee: check for empty decoded audio
        if audio.size == 0:
            raise AudioProcessingError("FFmpeg decoded audio is empty (no audio samples).")

        # Reshape the flat array into (n_samples, n_channels) if multi-channel
        if final_target_channels > 1:
            n = (audio.size // final_target_channels) * final_target_channels
            if n != audio.size:
                logger.warning("Dropping %d trailing samples to align channels.", audio.size - n)
            audio = audio[:n].reshape(-1, final_target_channels)

            # Check for empty array after channel alignment
            if audio.size == 0:
                raise AudioProcessingError(
                    "FFmpeg produced too few samples to form a complete multi-channel frame."
                )

        # Contract guarantee: clean up any NaN/Inf values from FFmpeg
        if not np.isfinite(audio).all():
            bad_count = (~np.isfinite(audio)).sum()
            logger.warning(
                "Detected %d invalid samples (NaN/Inf) from FFmpeg; replacing with safe values.",
                int(bad_count),
            )
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        # Contract guarantee: ensure values are in [-1, 1] range
        audio = np.clip(audio, -1.0, 1.0)

        # Respect contract: mono->1D, multi->2D even if n_samples==1
        if final_target_channels == 1:
            # Ensure we return 1D array (n_samples,) not scalar for single sample
            return audio.reshape(-1)
        return audio

    except subprocess.TimeoutExpired as e:
        raise AudioProcessingError(
            f"FFmpeg timed out after {timeout} seconds while processing audio. "
            "This may indicate corrupted input or very large file."
        ) from e
    except subprocess.CalledProcessError as e:
        # Limit stderr output to prevent overwhelming error messages
        stderr_msg = e.stderr.decode(errors="ignore")[:2000]
        if len(stderr_msg) == 2000:
            stderr_msg += "... (truncated)"

        raise AudioProcessingError(
            f"FFmpeg failed to process audio: {stderr_msg} | "
            "Install via 'brew install ffmpeg', 'sudo apt-get install ffmpeg', "
            "'winget install ffmpeg' or 'choco install ffmpeg'."
        ) from e


def _probe_stream_entry(source: str | bytes, entry: str, timeout: float = 5.0) -> int | None:
    """Query a single integer ``stream=<entry>`` value via ffprobe (guarded).

    Like the ffmpeg decode path, this constrains ffprobe to the ``file,pipe``
    protocols, so a crafted path can never trigger a network fetch (D1 / spec R5
    rationale). String sources are forwarded verbatim; callers that accept
    untrusted paths MUST validate them first via :func:`_validate_local_source_path`.

    Args:
        source: A local file path, or raw bytes.
        entry: The ``stream=`` field to read (e.g. ``"channels"``,
            ``"sample_rate"``).
        timeout: Max seconds to wait. Default: ``5.0``.

    Returns:
        The integer value, or ``None`` if ffprobe is unavailable or detection
        failed.

    Raises:
        None.
    """
    if shutil.which("ffprobe") is None:
        return None

    if isinstance(source, bytes):
        input_arg = "pipe:0"
        input_data: bytes | None = source
    else:
        input_arg = source
        input_data = None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-select_streams",
        "a:0",
        "-show_entries",
        f"stream={entry}",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_arg,
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, input=input_data, check=True, timeout=timeout
        )
        text = proc.stdout.decode().strip()
        return int(text) if text.isdigit() else None
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError:
        return None


def _probe_channels_with_ffprobe(source: str | bytes, timeout: float = 5.0) -> int | None:
    """Detect audio channel count via ffprobe (internal helper).

    Args:
        source: File path or raw bytes.
        timeout: Max seconds to wait. Default: ``5.0``.

    Returns:
        Number of channels, or ``None`` if ffprobe unavailable or detection failed.

    Raises:
        AudioProcessingError: If a string source is not a valid local file.
    """
    return _probe_stream_entry(source, "channels", timeout)


def _probe_sample_rate_with_ffprobe(source: str | bytes, timeout: float = 5.0) -> int | None:
    """Detect the native sample rate via ffprobe (internal helper).

    Args:
        source: File path or raw bytes.
        timeout: Max seconds to wait. Default: ``5.0``.

    Returns:
        Native sample rate in Hz, or ``None`` if ffprobe unavailable or detection
        failed.

    Raises:
        AudioProcessingError: If a string source is not a valid local file.
    """
    return _probe_stream_entry(source, "sample_rate", timeout)
