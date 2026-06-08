# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-memory WAV encoder (spec AI R4)."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from standard_asr.exceptions import AudioProcessingError
from standard_asr.utils.save_utils import encode_array_to_wav_bytes


def test_encode_mono_roundtrip() -> None:
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    result = encode_array_to_wav_bytes(audio, 16000)
    assert result.downmixed is False
    assert result.quantized is True
    with wave.open(io.BytesIO(result.data), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2


def test_encode_downmixes_multichannel_to_mono() -> None:
    audio = np.array([[0.2, 0.4], [-0.2, -0.4]], dtype=np.float32)
    result = encode_array_to_wav_bytes(audio, 8000)
    assert result.downmixed is True
    with wave.open(io.BytesIO(result.data), "rb") as wf:
        assert wf.getnchannels() == 1


def test_encode_clips_out_of_range() -> None:
    audio = np.array([2.0, -2.0], dtype=np.float32)
    result = encode_array_to_wav_bytes(audio, 16000)
    with wave.open(io.BytesIO(result.data), "rb") as wf:
        frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    assert frames[0] == 32767
    assert frames[1] == -32767


def test_encode_precheck_max_file_size() -> None:
    audio = np.zeros(100000, dtype=np.float32)
    with pytest.raises(AudioProcessingError, match="max_file_size"):
        encode_array_to_wav_bytes(audio, 16000, max_file_size=1024)


def test_encode_rejects_3d() -> None:
    audio = np.zeros((2, 2, 2), dtype=np.float32)
    with pytest.raises(AudioProcessingError):
        encode_array_to_wav_bytes(audio, 16000)


def test_encode_sanitizes_nan_and_inf() -> None:
    # AUDI-1: NaN/Inf must be sanitized BEFORE the int16 cast (np.clip does not
    # replace NaN, and casting NaN to int16 yields garbage PCM). NaN->0,
    # +Inf->+full-scale, -Inf->-full-scale, and the result must round-trip sanely.
    audio = np.array([np.nan, np.inf, -np.inf, 0.5], dtype=np.float32)
    result = encode_array_to_wav_bytes(audio, 16000)
    with wave.open(io.BytesIO(result.data), "rb") as wf:
        frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    assert frames[0] == 0
    assert frames[1] == 32767
    assert frames[2] == -32767
    # No garbage: every sample is a bounded int16.
    assert frames.min() >= -32767
    assert frames.max() <= 32767
