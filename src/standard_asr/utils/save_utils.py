# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Utility helpers for encoding audio.

Two helpers live here:

* :func:`encode_array_to_wav_bytes` -- encode a waveform to an in-memory WAV
  byte buffer in canonical form (16-bit PCM LE, **mono**), used by the audio
  negotiation layer when an array must be delivered to a file/bytes-only engine
  (spec, section "Audio Input & Sample Rate", rule R4).
* :func:`save_wav` -- write a waveform to a WAV file on disk, preserving the
  channel count.
"""

from __future__ import annotations

import io
import logging
import os
import wave
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..exceptions import AudioProcessingError
from ..wire import require_float_waveform, to_int16_pcm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WavEncodeResult:
    """Result of encoding a waveform to canonical WAV bytes.

    Args:
        data: The encoded WAV byte payload.
        sample_rate: Sample rate written, in Hz.
        downmixed: Whether multi-channel input was downmixed to mono.
        sanitized_non_finite: Number of non-finite samples (NaN/Inf) that were
            sanitized to ``0``/``+-1`` during the float->int16 cast. ``0`` when
            the input was already all-finite. Lets the conversion layer emit a
            ``non_finite_audio`` diagnostic so the sanitize is visible to the
            caller, matching the array-delivery path (spec R3).
    """

    data: bytes
    sample_rate: int
    downmixed: bool
    sanitized_non_finite: int


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

    The waveform MUST be a floating-point array in roughly ``[-1, 1]`` (an
    amplitude, not raw PCM codes): a non-floating dtype, an empty array, or a
    non-positive ``sample_rate`` fail loudly here rather than producing a
    corrupted or degenerate WAV that an engine would reject opaquely later.

    Args:
        audio: Float waveform. Mono is 1D; multi-channel is
            ``(n_samples, n_channels)``.
        sample_rate: Sample rate of the waveform in Hz. MUST be ``> 0``.
        max_file_size: Optional maximum encoded size in bytes.

    Returns:
        A :class:`WavEncodeResult` with the encoded bytes and conversion flags.

    Raises:
        AudioProcessingError: If ``sample_rate`` is not positive, the audio is
            not a floating-point dtype, the audio is empty, the audio is not
            1D/2D, or the encoded size exceeds ``max_file_size``.
    """
    if sample_rate <= 0:
        raise AudioProcessingError(f"sample_rate must be > 0, got {sample_rate}")
    array = require_float_waveform(audio)
    if array.size == 0:
        raise AudioProcessingError("Cannot encode empty audio array.")
    if array.ndim == 1:
        downmixed = False
        mono = array
    elif array.ndim == 2:
        downmixed = array.shape[1] > 1
        mono = array.mean(axis=1) if downmixed else array.reshape(-1)
    else:
        raise AudioProcessingError("Audio must be 1D (mono) or 2D (multi-channel).")

    pcm, sanitized_non_finite = to_int16_pcm(mono)
    # Serialize explicitly little-endian: WAV defines 16-bit PCM as LE, but
    # ``tobytes()`` uses the array's NATIVE byte order. On a big-endian host that
    # would silently emit byte-swapped samples under an LE-declaring header
    # (spec R4: canonical = WAV/16-bit PCM LE). ``<i2`` pins the contract.
    frames = pcm.astype("<i2").tobytes()

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

    return WavEncodeResult(
        data=data,
        sample_rate=sample_rate,
        downmixed=downmixed,
        sanitized_non_finite=sanitized_non_finite,
    )


def save_wav(
    audio: NDArray[np.floating],
    path: str | os.PathLike[str],
    sample_rate: int = 16000,
) -> None:
    """Write a float waveform to a WAV file on disk as 16-bit PCM.

    The disk-writing companion to :func:`encode_array_to_wav_bytes`. Unlike that
    canonical encoder (which always downmixes to mono), this **preserves the
    channel count**: mono input uses 1 channel; 2D input is interpreted as
    ``(n_samples, n_channels)`` and written with that many channels.

    The input is treated as a Standard ASR-normalized waveform (a floating
    amplitude in roughly ``[-1.0, 1.0]``). Values are clipped to ``[-1, 1]`` and
    mapped to signed 16-bit PCM. A non-floating dtype, an array that is not
    1D/2D, or a non-positive ``sample_rate`` fail loudly rather than writing a
    corrupted or degenerate file.

    Args:
        audio: Waveform array to save. Mono can be 1D; multi-channel is
            ``(n_samples, n_channels)``.
        path: Destination path for the ``.wav`` file. Accepts ``str`` or any
            ``os.PathLike`` (e.g. :class:`pathlib.Path`).
        sample_rate: Sample rate to write (Hz). MUST be ``> 0``. Defaults to
            ``16000``.

    Raises:
        AudioProcessingError: If ``sample_rate`` is not positive, the audio is
            not a floating-point dtype, or the audio is not 1D/2D.
        OSError: If writing to ``path`` fails (permissions, disk, etc.).
    """
    if sample_rate <= 0:
        raise AudioProcessingError(f"sample_rate must be > 0, got {sample_rate}")
    array = require_float_waveform(audio)
    if array.ndim not in (1, 2):
        raise AudioProcessingError("Audio must be 1D (mono) or 2D (multi-channel).")

    # Pin little-endian (``<i2``) for canonical WAV PCM regardless of host byte
    # order (spec R4); ``tobytes()`` would otherwise use native order.
    pcm, _sanitized = to_int16_pcm(array)
    audio_integer = pcm.astype("<i2")

    if audio_integer.ndim == 1:
        channels = 1
        frames = audio_integer.tobytes()
    else:
        channels = int(audio_integer.shape[1])
        frames = audio_integer.reshape(-1, channels).tobytes()

    file_path = os.fspath(path)
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
    "save_wav",
]
