# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the canonical wire codec (spec "Audio Input & Sample Rate" R4).

These cover the public ``standard_asr.wire`` surface: the canonical encoding
identifier and the ``float32`` <-> ``pcm_s16le`` codec pair that every streaming
engine and the WAV reader now share, so the spec-pinned quantization is defined
and verified exactly once.
"""

from __future__ import annotations

import numpy as np
import pytest

from standard_asr.exceptions import AudioProcessingError
from standard_asr.wire import CANONICAL_WIRE_ENCODING, pcm16_decode, pcm16_encode


def test_canonical_encoding_value() -> None:
    # The one canonical wire encoding apps/plugins should reference by name.
    assert CANONICAL_WIRE_ENCODING == "pcm_s16le"


def test_encode_little_endian_full_scale() -> None:
    # +1.0 -> 32767 == 0x7FFF, serialized little-endian as (0xFF, 0x7F).
    assert pcm16_encode(np.array([1.0], dtype=np.float32)) == b"\xff\x7f"


def test_encode_clips_out_of_range() -> None:
    codes = np.frombuffer(pcm16_encode(np.array([2.0, -2.0], dtype=np.float32)), dtype="<i2")
    assert codes[0] == 32767
    assert codes[1] == -32767


def test_encode_sanitizes_non_finite() -> None:
    # NaN -> 0, +Inf -> +full-scale, -Inf -> -full-scale (never garbage PCM).
    codes = np.frombuffer(
        pcm16_encode(np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)), dtype="<i2"
    )
    assert list(codes) == [0, 32767, -32767, 0]


def test_encode_uses_round_half_not_truncation() -> None:
    # A value mapping to 5000.6 codes MUST round to 5001 (round-half), not
    # truncate to 5000 (truncation doubles the quantization error to 1 LSB).
    x = np.array([5000.6 / 32767.0], dtype=np.float32)
    assert np.frombuffer(pcm16_encode(x), dtype="<i2")[0] == 5001


def test_encode_rejects_integer_pcm() -> None:
    # Integer PCM is not an amplitude; clipping it to [-1, 1] would crush every
    # sample to a square wave -- reject loudly instead of silently corrupting.
    with pytest.raises(AudioProcessingError, match="floating-point"):
        pcm16_encode(np.array([1000, -1000], dtype=np.int16))  # pyright: ignore[reportArgumentType]


def test_encode_empty_array_is_empty_bytes() -> None:
    assert pcm16_encode(np.array([], dtype=np.float32)) == b""


def test_decode_full_scale_and_sign() -> None:
    # 0x7FFF -> 32767/32768 (the deliberate attenuation); 0x8000 -> -1.0 exactly.
    waveform = pcm16_decode(b"\xff\x7f\x00\x80")
    assert waveform.dtype == np.float32
    assert waveform[0] == pytest.approx(32767 / 32768, abs=1e-6)
    assert waveform[1] == pytest.approx(-1.0)


def test_decode_rejects_odd_length() -> None:
    # A frame that is not a whole number of 16-bit samples is malformed: fail
    # loudly rather than silently dropping a byte.
    with pytest.raises(AudioProcessingError, match="multiple of 2"):
        pcm16_decode(b"\x00\x00\x01")


def test_decode_empty_bytes_is_empty_array() -> None:
    out = pcm16_decode(b"")
    assert out.dtype == np.float32
    assert out.size == 0


def test_round_trip_within_quantization_tolerance() -> None:
    original = np.array([0.0, 0.25, -0.25, 0.5, -0.5, 0.999], dtype=np.float32)
    restored = pcm16_decode(pcm16_encode(original))
    assert restored.dtype == np.float32
    # Round-trip error is bounded by one quantization step (~1/32768 ~= 3.05e-5)
    # plus the intentional 32767/32768 attenuation -- comfortably under 1e-3.
    assert np.max(np.abs(restored - original)) < 1e-3
