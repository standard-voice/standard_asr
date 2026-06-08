# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Utility helpers for encoding audio.

Two helpers live here:

* :func:`nparray_to_audio_file` -- write a waveform to a WAV file on disk,
  preserving channel count (legacy convenience).
* :func:`encode_array_to_wav_bytes` -- encode a waveform to an in-memory WAV
  byte buffer in canonical form (16-bit PCM LE, **mono**), used by the audio
  negotiation layer when an array must be delivered to a file/bytes-only engine
  (spec, section "Audio Input & Sample Rate", rule R4).
"""

from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..exceptions import AudioProcessingError

logger = logging.getLogger(__name__)


def _to_int16_pcm(audio: NDArray[np.floating]) -> NDArray[np.int16]:
    """Convert a float waveform in ``[-1, 1]`` to signed 16-bit PCM.

    Non-finite samples are sanitized *first* (NaN->0, +Inf->+1, -Inf->-1),
    because ``np.clip`` does NOT replace NaN (``np.clip(nan, -1, 1) == nan``) and
    casting NaN to ``int16`` is undefined behavior that emits garbage PCM -- a
    silent-wrong-result. This mirrors the decode paths, which already sanitize.
    Clipping then happens *before* the cast (NumPy 1.x/2.x defensive; see the
    dependencies spec section DEP.2).

    Args:
        audio: Float waveform array.

    Returns:
        Signed 16-bit PCM array.
    """
    finite = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    clipped = np.clip(finite, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


@dataclass(frozen=True)
class WavEncodeResult:
    """Result of encoding a waveform to canonical WAV bytes.

    Args:
        data: The encoded WAV byte payload.
        sample_rate: Sample rate written, in Hz.
        downmixed: Whether multi-channel input was downmixed to mono.
        quantized: Whether a lossy float->int16 quantization was applied.
    """

    data: bytes
    sample_rate: int
    downmixed: bool
    quantized: bool


def encode_array_to_wav_bytes(
    audio: NDArray[np.floating],
    sample_rate: int,
    *,
    max_file_size: int | None = None,
) -> WavEncodeResult:
    """Encode a waveform to an in-memory canonical WAV byte buffer.

    The output is always WAV / 16-bit PCM LE / **mono** (the canonical encoded
    form). Multi-channel input is downmixed by averaging channels. The encode
    never touches disk. When ``max_file_size`` is given, the encoded size is
    pre-checked and a clear local error is raised before returning if exceeded.

    Args:
        audio: Float waveform. Mono is 1D; multi-channel is
            ``(n_samples, n_channels)``.
        sample_rate: Sample rate of the waveform in Hz.
        max_file_size: Optional maximum encoded size in bytes.

    Returns:
        A :class:`WavEncodeResult` with the encoded bytes and conversion flags.

    Raises:
        AudioProcessingError: If the audio is not 1D/2D, or the encoded size
            exceeds ``max_file_size``.
    """
    array = np.asarray(audio)
    if array.ndim == 1:
        downmixed = False
        mono = array
    elif array.ndim == 2:
        downmixed = array.shape[1] > 1
        mono = array.mean(axis=1) if downmixed else array.reshape(-1)
    else:
        raise AudioProcessingError("Audio must be 1D (mono) or 2D (multi-channel).")

    pcm = _to_int16_pcm(mono)
    frames = pcm.tobytes()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes = 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(frames)
    data = buffer.getvalue()

    if max_file_size is not None and len(data) > max_file_size:
        raise AudioProcessingError(
            f"Encoded audio is {len(data)} bytes, which exceeds the engine's "
            f"max_file_size of {max_file_size} bytes. Provide a shorter clip or "
            "use an engine without this limit."
        )

    return WavEncodeResult(data=data, sample_rate=sample_rate, downmixed=downmixed, quantized=True)


def nparray_to_audio_file(
    audio: NDArray[np.float32], file_path: str, sample_rate: int = 16000
) -> None:
    """Write a float32 waveform to a WAV file as 16-bit PCM.

    The input is treated as a Standard ASR-normalized waveform (dtype
    ``np.float32``, values in roughly ``[-1.0, 1.0]``). Values are clipped to
    ``[-1, 1]`` and linearly mapped to signed 16-bit PCM for storage. Mono input
    uses 1 channel; 2D input is interpreted as ``(n_samples, n_channels)`` and
    its channel count is preserved.

    Args:
        audio: Waveform array to save. Mono can be 1D.
        file_path: Destination path for the ``.wav`` file.
        sample_rate: Sample rate to write (Hz). Defaults to ``16000``.

    Raises:
        OSError: If writing to ``file_path`` fails (permissions, disk, etc.).
    """
    audio_integer = _to_int16_pcm(audio)

    if audio_integer.ndim == 1:
        channels = 1
        frames = audio_integer.tobytes()
    else:
        channels = int(audio_integer.shape[1])
        frames = audio_integer.reshape(-1, channels).tobytes()

    try:
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 2 bytes = 16 bits
            wf.setframerate(sample_rate)
            wf.writeframes(frames)
    except OSError as e:
        logger.error("Error writing audio to file %s: %s", file_path, e)
        raise


__all__ = [
    "WavEncodeResult",
    "encode_array_to_wav_bytes",
    "nparray_to_audio_file",
]
