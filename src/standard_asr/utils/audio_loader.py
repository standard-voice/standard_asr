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
from typing import Any, BinaryIO, TypeGuard, cast
from typing import overload, Literal

import numpy as np
from numpy.typing import DTypeLike, NDArray

from ..exceptions import (
    AudioProcessingError,
    FFmpegNotFoundError,
)

logger = logging.getLogger(__name__)

# --- Public API ---


# numpy datatype check
@overload
def ensure_datatype(audio: NDArray[Any]) -> NDArray[np.float32]: ...


@overload
def ensure_datatype(
    audio: NDArray[Any], data_type: Literal["float32"]
) -> NDArray[np.float32]: ...


@overload
def ensure_datatype(
    audio: NDArray[Any], data_type: np.dtype[np.float32]
) -> NDArray[np.float32]: ...


@overload
def ensure_datatype(
    audio: NDArray[Any], data_type: type[np.float32]
) -> NDArray[np.float32]: ...


@overload
def ensure_datatype(audio: NDArray[Any], data_type: DTypeLike) -> NDArray[Any]: ...


def ensure_datatype(
    audio: NDArray[Any], data_type: DTypeLike = np.float32
) -> NDArray[Any]:
    """Ensure a NumPy array uses the specified dtype, returning a view when possible.

    This helper normalizes the dtype of an input array. If the array already has
    the requested dtype, the same array (not a copy) is returned. Otherwise a
    cast is performed with ``copy=False`` semantics when feasible.

    Args:
        audio (NDArray[Any]): The input array.
        data_type (DTypeLike): Target dtype. Accepts forms like ``"float32"``,
            ``np.float32``, ``np.dtype(np.float32)``. Defaults to ``np.float32``.

    Returns:
        NDArray[Any]: The same array if dtypes already match; otherwise a view or
        minimally-copied array with the requested dtype.
    """
    # Compute the target dtype for runtime comparison; helps static checkers
    target_dtype: np.dtype[np.generic] = np.dtype(data_type)
    if audio.dtype != target_dtype:
        audio = audio.astype(target_dtype, copy=False)
    return audio


def normalize_audio(
    audio: NDArray[Any],
    original_sr: int,
    target_sr: int = 16000,
    target_channels: int | None = 1,
) -> NDArray[np.float32]:
    """Normalize a raw waveform to the Standard ASR audio contract.

    This function validates basic parameters, optionally resamples, converts
    channels, cleans up invalid values, and enforces the output dtype and
    amplitude range. Input ``audio`` may be 1D (mono) or 2D with shape
    ``(n_samples, n_channels)``. The returned array is:

    - dtype: ``np.float32``
    - shape: ``(n_samples,)`` for mono, ``(n_samples, n_channels)`` for multi
    - values: clipped to ``[-1.0, 1.0]``

    Resampling:
    - Performed with ``scipy.signal.resample_poly`` when ``original_sr != target_sr``.
    - If ``scipy`` is not installed and resampling is required, an ``ImportError``
      is raised with guidance on how to install the dependency.

    Channel handling:
    - ``target_channels is None``: preserve the input channel count.
    - ``target_channels == 1``: downmix by arithmetic mean across channels.
    - ``target_channels > 1`` and input has fewer channels: upmix by replication.
    - ``target_channels < input_channels``: truncate to the first ``target_channels``
      (for high-quality mixing, load via FFmpeg instead).

    Invalid values:
    - Any NaN/Inf values encountered are replaced by safe values (NaN->0.0,
      +Inf->1.0, -Inf->-1.0), and a warning is logged.

    Args:
        audio (NDArray[Any]): Input waveform (1D mono or 2D multi-channel). The
            dtype is converted to ``np.float32``.
        original_sr (int): The original sample rate of ``audio``. Must be > 0.
        target_sr (int): Desired sample rate (default ``16000``). Must be > 0.
        target_channels (int | None): Desired channel count. ``1`` for mono,
            ``2`` for stereo, or ``None`` to preserve channels.

    Returns:
        NDArray[np.float32]: A waveform matching the Standard ASR audio contract.

    Raises:
        AudioProcessingError: If parameters are invalid or the input is empty.
        ImportError: If resampling is needed but ``scipy`` is not installed.
    """
    processed_audio: NDArray[np.float32] = ensure_datatype(audio, "float32")
    # Basic parameter validation

    if original_sr <= 0:
        raise AudioProcessingError(f"original_sr must be > 0, got {original_sr}")
    if target_sr <= 0:
        raise AudioProcessingError(f"target_sr must be > 0, got {target_sr}")
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(
            f"target_channels must be None or > 0, got {target_channels}"
        )

    # Check for empty audio
    if processed_audio.size == 0:
        raise AudioProcessingError("Cannot process empty audio array")

    # 1. Resample if necessary using resample_poly (more stable)
    if original_sr != target_sr:
        try:
            from math import gcd
            from scipy.signal import resample_poly as _resample_poly  # pyright: ignore[reportMissingTypeStubs,reportUnknownVariableType]

            g = gcd(original_sr, target_sr)
            up, down = target_sr // g, original_sr // g
            processed_audio = _resample_poly(
                processed_audio, up=up, down=down, axis=0
            ).astype(np.float32, copy=False)  # pyright: ignore[reportUnknownVariableType]
        except ImportError:
            raise ImportError(
                "Resampling requires `scipy`. Please install it (`pip install scipy`) "
                "or install standard-asr with audio support (`pip install 'standard-asr[audio]'`)"
            )

    # Ensure at least 2D for uniform channel processing
    if processed_audio.ndim == 1:
        processed_audio = processed_audio[:, np.newaxis]

    # Help static type checker with tuple[int, ...] for shape
    current_channels = int(processed_audio.shape[1])

    # 2. Channel conversion
    if target_channels is not None and current_channels != target_channels:
        if target_channels == 1:
            # Downmix to mono using average across channels
            processed_audio = processed_audio.mean(axis=1, dtype=np.float32)[
                :, np.newaxis
            ]
        else:
            # Note: For multi-to-multi down-mixing (e.g., 6->2), this implementation performs a
            # simple channel selection/truncation instead of a perceptually accurate mix.
            # For higher quality down-mix, ensure FFmpeg is installed and prefer the FFmpeg path.
            if target_channels > current_channels:  # Upscale (e.g., mono to stereo)
                reps = int(math.ceil(target_channels / current_channels))
                processed_audio = np.tile(processed_audio, (1, reps))[
                    :, :target_channels
                ]
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
            "Detected %d invalid samples (NaN/Inf) in audio; "
            "replacing with safe values.",
            int(bad_count),
        )
        processed_audio = np.nan_to_num(
            processed_audio, copy=False, nan=0.0, posinf=1.0, neginf=-1.0
        )

    # Clip to contract range
    processed_audio = np.clip(processed_audio, -1.0, 1.0).astype(np.float32, copy=False)
    # Respect contract: mono->1D, multi->2D even if n_samples==1
    if int(processed_audio.shape[1]) == 1:
        return processed_audio[:, 0]
    return processed_audio


def load_audio(
    source: str | bytes | bytearray | memoryview | pathlib.Path | BinaryIO,
    target_sr: int = 16000,
    target_channels: int | None = 1,
) -> NDArray[np.float32]:
    """Load audio from many sources and return a normalized waveform array.

    Loading strategy (in order):
    1) When a string is provided and starts with ``data:...;base64,``, decode the
       base64 payload and treat it as bytes.
    2) If the string represents an existing path, open it as a file path.
    3) Otherwise, treat the string as a file path (which may raise ``FileNotFoundError``).
    4) For bytes-like (``bytes``, ``bytearray``, ``memoryview``) and BinaryIO,
       first try ``soundfile`` (if installed), then fall back to FFmpeg.

    Contract guarantees for the returned array:
    - dtype: ``np.float32``
    - sr: ``target_sr`` (default 16000 Hz)
    - range: within ``[-1.0, 1.0]``
    - shape: ``(n_samples,)`` for mono; ``(n_samples, n_channels)`` for multi

    Args:
        source (str | bytes | bytearray | memoryview | pathlib.Path | BinaryIO):
            The audio source. Supports filesystem paths, bytes-like objects, base64
            data URIs, and binary file-like streams.
        target_sr (int): Target sample rate. Audio will be resampled if necessary.
        target_channels (int | None): Target number of channels: ``1`` for mono,
            ``2`` for stereo, or ``None`` to preserve input channels.

    Returns:
        NDArray[np.float32]: Waveform normalized to the Standard ASR audio contract.

    Raises:
        FileNotFoundError: If a provided path cannot be found/opened.
        AudioProcessingError: If loading or processing fails.
        ImportError: If a required optional dependency is missing for the chosen path.
        FFmpegNotFoundError: If FFmpeg fallback is required but not available.
        TypeError: If ``source`` is of an unsupported type.

    Notes:
        - Only data URIs (``data:...;base64,``) are auto-detected as base64. Raw base64
          strings are not implicitly decoded. To load raw base64, decode it manually and
          pass the resulting bytes to this function or use ``load_audio_from_bytes``.
        - For BinaryIO sources, this function reads from the current stream position and
          does not seek to the beginning.
        - FFmpeg fallback buffers the entire decoded audio in memory. For very long audio
          consider chunked or streaming approaches.
    """
    if target_sr <= 0:
        raise AudioProcessingError(f"target_sr must be > 0, got {target_sr}")
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(
            f"target_channels must be None or > 0, got {target_channels}"
        )
    if isinstance(source, str):
        # Improved base64 detection logic to avoid false positives
        s = source.strip()

        # First check: if it has explicit base64 data URI prefix, treat as base64
        if s.lower().startswith("data:") and ";base64," in s:
            try:
                encoded_data = s.split(";base64,", 1)[1]
                source_bytes = base64.b64decode(encoded_data, validate=True)
                return load_audio_from_bytes(source_bytes, target_sr, target_channels)
            except (ValueError, TypeError) as e:
                raise AudioProcessingError(f"Invalid base64 data URI: {e}") from e

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
    """Best-effort check that an object is a binary IO returning bytes-like data.

    The check succeeds for common binary IO classes or for any object whose
    ``read(0)`` returns a bytes-like object. Text IO is rejected. If a stream does
    not support zero-length reads or raises during probing, the function returns
    ``False`` to avoid misclassification.

    Args:
        obj (Any): An arbitrary object to evaluate.

    Returns:
        TypeGuard[BinaryIO]: ``True`` if ``obj`` appears to be a binary IO stream,
        otherwise ``False``.

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
    """Load audio from a filesystem path using layered fallbacks.

    Order of loaders:
    1) WAV via stdlib ``wave`` for 8-bit unsigned PCM and 16-bit signed PCM
    2) ``soundfile`` if available (supports many formats)
    3) FFmpeg subprocess fallback

    Args:
        path (str): Filesystem path to an audio file. ``~`` is expanded.
        target_sr (int): Desired sample rate for output, must be > 0.
        target_channels (int | None): Desired channel count. ``None`` preserves
            the original number of channels.

    Returns:
        NDArray[np.float32]: Standardized waveform adhering to the audio contract.

    Raises:
        FileNotFoundError: If the file does not exist or is inaccessible.
        AudioProcessingError: For unsupported/invalid WAV encodings via stdlib or
            for failures in decoding downstream.
        FFmpegNotFoundError: If FFmpeg is required but not present in the system ``PATH``.
    """
    # Basic parameter validation
    if target_sr <= 0:
        raise AudioProcessingError(f"target_sr must be > 0, got {target_sr}")
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(
            f"target_channels must be None or > 0, got {target_channels}"
        )

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
                dtype_map = {1: np.uint8, 2: np.int16}
                audio = np.frombuffer(frames, dtype=dtype_map[sampwidth]).astype(
                    np.float32
                )
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
        audio, orig_sr = cast(
            tuple[NDArray[np.float32], int], sf_read(path, dtype="float32")
        )
        try:
            return normalize_audio(audio, orig_sr, target_sr, target_channels)
        except ImportError:
            logger.warning(
                "Resampling requires scipy; falling back to FFmpeg for %s.", path
            )
            return _load_with_ffmpeg(path, target_sr, target_channels)
    except ImportError:
        logger.debug(
            "`soundfile` not installed, cannot load non-WAV formats without FFmpeg."
        )
    except Exception as e:
        logger.debug(f"Could not load with `soundfile`, falling back... Error: {e}")

    # Layer 3: Final fallback to FFmpeg
    return _load_with_ffmpeg(path, target_sr, target_channels)


def load_audio_from_bytes(
    data: bytes, target_sr: int = 16000, target_channels: int | None = 1
) -> NDArray[np.float32]:
    """Load audio from bytes and return a normalized waveform.

    Strategy:
    - Prefer ``soundfile`` (if installed) for robust container/codec support.
    - Fall back to FFmpeg subprocess when necessary.

    Args:
        data (bytes): Raw bytes containing an audio file/stream.
        target_sr (int): Target sample rate; must be > 0.
        target_channels (int | None): Target channel count, or ``None`` to preserve.

    Returns:
        NDArray[np.float32]: Standardized waveform adhering to the audio contract.

    Raises:
        AudioProcessingError: If decoding fails or produces empty audio.
        FFmpegNotFoundError: If FFmpeg is required but not available.
    """
    # Basic parameter validation
    if target_sr <= 0:
        raise AudioProcessingError(f"target_sr must be > 0, got {target_sr}")
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(
            f"target_channels must be None or > 0, got {target_channels}"
        )
    # Layer 2: `soundfile` is the best primary method for bytes
    try:
        import soundfile as sf  # pyright: ignore[reportMissingTypeStubs]

        sf_read: Any = getattr(sf, "read")
        audio, orig_sr = cast(
            tuple[NDArray[np.float32], int], sf_read(io.BytesIO(data), dtype="float32")
        )
        try:
            return normalize_audio(audio, orig_sr, target_sr, target_channels)
        except ImportError:
            logger.warning(
                "Resampling requires scipy; falling back to FFmpeg for byte input."
            )
            return _load_with_ffmpeg(data, target_sr, target_channels)
    except ImportError:
        logger.debug(
            "`soundfile` not installed, cannot load from bytes without FFmpeg."
        )
    except Exception as e:
        logger.debug(
            f"Could not load bytes with `soundfile`, falling back... Error: {e}"
        )

    # Layer 3: Final fallback to FFmpeg
    return _load_with_ffmpeg(data, target_sr, target_channels)


def _load_with_ffmpeg(
    source: str | bytes,
    target_sr: int,
    target_channels: int | None,
    timeout: float = 120.0,
) -> NDArray[np.float32]:
    """Load audio using an FFmpeg subprocess as the ultimate fallback.

    This function invokes FFmpeg to decode audio and emit 32-bit float
    little-endian PCM (``f32le``) to stdout. If ``target_channels`` is ``None``,
    it attempts to detect the input channel count via ``ffprobe`` first and fall
    back to mono when probing is unavailable.

    The output is normalized to the Standard ASR audio contract. Multi-channel
    output is reshaped to ``(n_samples, n_channels)`` from the flat buffer.

    Args:
        source (str | bytes): A filesystem path or raw bytes of an audio file.
        target_sr (int): Target sample rate (Hz).
        target_channels (int | None): Target number of channels or ``None`` to
            preserve when possible.
        timeout (float): Maximum seconds to allow FFmpeg to run before aborting.

    Returns:
        NDArray[np.float32]: Waveform array with dtype ``float32`` and shape per
        the audio contract.

    Raises:
        FFmpegNotFoundError: If FFmpeg is not found in ``PATH``.
        AudioProcessingError: On FFmpeg errors, timeouts, empty output or shape
            inconsistencies.
    """
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError(
            "FFmpeg not found in PATH. Install via: 'brew install ffmpeg' (macOS), "
            "'sudo apt-get install ffmpeg' (Debian/Ubuntu), 'winget install ffmpeg' or "
            "'choco install ffmpeg' (Windows)."
        )

    # If target_channels is None, attempt to preserve original channels via ffprobe.
    if target_channels is None:
        detected_channels = _probe_channels_with_ffprobe(source)
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
        "-vn",
        "-sn",
        "-dn",
        "-nostats",
        "-loglevel",
        "error",
        "-threads",
        "0",  # Use optimal number of threads
        "-map",
        "a:0",  # Explicitly select the first audio stream
        "-i",
        "pipe:0" if isinstance(source, bytes) else source,
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
            raise AudioProcessingError(
                "FFmpeg decoded audio is empty (no audio samples)."
            )

        # Reshape the flat array into (n_samples, n_channels) if multi-channel
        if final_target_channels > 1:
            n = (audio.size // final_target_channels) * final_target_channels
            if n != audio.size:
                logger.warning(
                    "Dropping %d trailing samples to align channels.", audio.size - n
                )
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
            audio = np.nan_to_num(audio, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

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


def _probe_channels_with_ffprobe(
    source: str | bytes, timeout: float = 5.0
) -> int | None:
    """Probe the input's channel count using ``ffprobe``.

    Args:
        source (str | bytes): The same input accepted by FFmpeg: path or bytes.
        timeout (float): Maximum seconds to wait for ffprobe.

    Returns:
        int | None: Detected number of channels, or ``None`` if ffprobe is not
        available or probing fails for any reason.

    Raises:
        None.
    """
    if shutil.which("ffprobe") is None:
        return None

    if isinstance(source, bytes):
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            "pipe:0",
        ]
        input_data = source
    else:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            source,
        ]
        input_data = None

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
