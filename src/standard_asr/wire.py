# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical streaming wire encoding and PCM codec.

This module is the **single source of truth** for the ``float32`` <-> 16-bit PCM
conversion the streaming wire protocol is pinned to (spec "Audio Input & Sample
Rate", rule R4). The conversion is byte-pinned so that every language's wire
implementation produces identical PCM and a cross-language conformance test sees
no ``+-1`` LSB noise; defining it once here (rather than letting each engine
re-derive it) is what keeps the Python and wire layers isomorphic (goal G.5.2).

Public surface:

* :data:`CANONICAL_WIRE_ENCODING` -- the canonical encoding identifier
  (``"pcm_s16le"``), so applications and plugins stop hardcoding the string and
  risking a silent mismatch against an engine's ``wire_encodings``.
* :func:`pcm16_encode` -- float waveform -> canonical ``pcm_s16le`` bytes.
* :func:`pcm16_decode` -- canonical ``pcm_s16le`` bytes -> float32 waveform.
* :func:`require_float_waveform` / :func:`to_int16_pcm` -- the lower-level
  canonical-quantization primitives (a dtype guard + the round-half float->int16
  conversion with a non-finite count), shared by :func:`pcm16_encode` and the WAV
  encoders so the spec-pinned quantization is defined exactly once.

The encode/decode pair deliberately uses the spec's asymmetric scale factors
(``x 32767`` on encode, ``/ 32768`` on decode); the ~-0.00027 dB round-trip
attenuation is intentional and identical across the WAV encoder, the WAV reader,
and this codec because all three call the helpers defined here.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .exceptions import AudioProcessingError

logger = logging.getLogger(__name__)

#: The canonical streaming wire encoding: signed 16-bit PCM, little-endian.
#: Use this instead of hardcoding the ``"pcm_s16le"`` string. It is the one
#: encoding for which the standard ships a built-in codec (:func:`pcm16_encode` /
#: :func:`pcm16_decode`); ``AudioFormat.encoding`` and ``wire_encodings`` remain
#: open strings so engines MAY declare other encodings (e.g. ``"mulaw"``) that
#: carry their own transport.
CANONICAL_WIRE_ENCODING: Final = "pcm_s16le"


def require_float_waveform(audio: NDArray[np.floating]) -> NDArray[np.floating]:
    """Reject a non-floating array before it is treated as a ``[-1, 1]`` waveform.

    The public encoders are typed ``NDArray[np.floating]``, but ``np.asarray``
    does not enforce dtype at runtime, so a dynamic / un-type-checked caller can
    pass an **integer** PCM array. ``np.clip(int_codes, -1, 1)`` would crush every
    sample to a square wave (e.g. ``1000 -> 1.0``), encoding completely corrupted
    PCM that still 'succeeds' -- a silent wrong result. The engine input boundary
    (``AudioArray.__post_init__``) already rejects non-floating dtypes; this gives
    the standalone codec/encoder helpers the same guard. The conversion is
    ``audio.astype(np.float32) / 32768.0`` for int16 codes (mirrors the decode
    scaling).

    Args:
        audio: The array to validate.

    Returns:
        The array unchanged, narrowed to a floating dtype.

    Raises:
        AudioProcessingError: If ``audio`` is not a floating-point dtype.
    """
    array = np.asarray(audio)
    if not np.issubdtype(array.dtype, np.floating):
        raise AudioProcessingError(
            f"Audio must be a floating-point waveform in [-1, 1], got dtype "
            f"{array.dtype}. Integer PCM is not an amplitude: convert it first, "
            "e.g. samples.astype(np.float32) / 32768.0 for int16."
        )
    return array


def to_int16_pcm(audio: NDArray[np.floating]) -> tuple[NDArray[np.int16], int]:
    """Convert a float waveform in ``[-1, 1]`` to signed 16-bit PCM.

    Non-finite samples are sanitized *first* (NaN->0, +Inf->+1, -Inf->-1),
    because ``np.clip`` does NOT replace NaN (``np.clip(nan, -1, 1) == nan``) and
    casting NaN to ``int16`` is undefined behavior that emits garbage PCM -- a
    silent-wrong-result. This mirrors the decode paths, which already sanitize.
    The count of replaced samples is returned so the caller can emit a
    ``non_finite_audio`` diagnostic (the sanitize is correct and necessary, but
    the *fact* that it happened MUST be visible to the caller -- spec R3 /
    explicit > implicit). Clipping then happens *before* the cast (NumPy 1.x/2.x
    defensive; see the dependencies spec section DEP.2), and quantization uses
    round-half (``np.rint``) rather than truncation so the canonical encoder's
    quantization error stays bounded by 0.5 LSB instead of 1 LSB.

    Args:
        audio: Float waveform array.

    Returns:
        A ``(pcm, sanitized_non_finite)`` pair: the signed 16-bit PCM array and
        the number of non-finite samples that were replaced.
    """
    array = np.asarray(audio)
    sanitized_non_finite = int(np.count_nonzero(~np.isfinite(array)))
    finite = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
    clipped = np.clip(finite, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype(np.int16)
    return pcm, sanitized_non_finite


def pcm16_encode(samples: NDArray[np.floating]) -> bytes:
    """Encode a float waveform in ``[-1, 1]`` to canonical ``pcm_s16le`` bytes.

    The canonical streaming wire encoding (spec R4): clip to ``[-1, 1]``,
    quantize with round-half (``x 32767``), serialize **little-endian**. Non-finite
    samples are sanitized (NaN->0, +-Inf->+-full-scale) so the cast never emits
    garbage. An empty array encodes to ``b""``. Use this in a streaming engine
    instead of re-implementing the quantization (``astype(np.int16)`` truncation
    silently doubles the quantization error and diverges from the wire contract).

    Args:
        samples: Float waveform (an amplitude in roughly ``[-1, 1]``), 1D for
            mono. Multi-channel interleaving is the caller's responsibility.

    Returns:
        The little-endian signed 16-bit PCM byte payload.

    Raises:
        AudioProcessingError: If ``samples`` is not a floating-point dtype.
    """
    array = require_float_waveform(samples)
    pcm, _sanitized = to_int16_pcm(array)
    # Pin little-endian (`<i2`) regardless of host byte order; `tobytes()` would
    # otherwise use native order and a big-endian host would emit byte-swapped
    # samples under a pcm_s16le label (spec R4).
    return pcm.astype("<i2").tobytes()


def pcm16_decode(data: bytes) -> NDArray[np.float32]:
    """Decode canonical ``pcm_s16le`` bytes to a float32 waveform in ``[-1, 1]``.

    The inverse of :func:`pcm16_encode`: read the bytes as little-endian signed
    16-bit codes and scale by ``/ 32768`` (the deliberate ``32767``/``32768``
    round-trip asymmetry the spec pins, shared with the WAV reader). An empty
    ``data`` decodes to an empty array. Frames whose length is not a whole number
    of 16-bit samples are rejected loudly rather than silently dropping a byte.

    Args:
        data: Little-endian signed 16-bit PCM bytes (interleaved if
            multi-channel; de-interleaving is the caller's responsibility).

    Returns:
        The decoded ``float32`` waveform.

    Raises:
        AudioProcessingError: If ``len(data)`` is not a multiple of 2 bytes.
    """
    if len(data) % 2 != 0:
        raise AudioProcessingError(
            f"pcm_s16le frame length must be a multiple of 2 bytes (16-bit "
            f"samples), got {len(data)} bytes."
        )
    # `<i2` pins little-endian regardless of host byte order; the chained
    # `.astype(np.float32)` narrows the otherwise-unknown frombuffer dtype, and
    # `/ 32768.0` applies the spec's deliberate decode scaling.
    waveform = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return np.asarray(waveform, dtype=np.float32)


__all__ = [
    "CANONICAL_WIRE_ENCODING",
    "pcm16_decode",
    "pcm16_encode",
    "require_float_waveform",
    "to_int16_pcm",
]
