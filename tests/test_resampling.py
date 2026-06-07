# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in anti-aliasing fallback resampler."""

from __future__ import annotations

import numpy as np
import pytest

from standard_asr.resampling import resample


def test_identity_when_rates_match() -> None:
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = resample(x, 16000, 16000)
    assert out.dtype == np.float32
    assert np.allclose(out, x)


def test_downsample_length_and_tone() -> None:
    t = np.arange(48000) / 48000.0
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    y = resample(x, 48000, 16000)
    assert y.shape[0] == 16000
    freqs = np.fft.rfftfreq(len(y), 1 / 16000)
    peak = freqs[np.argmax(np.abs(np.fft.rfft(y)))]
    assert abs(peak - 1000) < 5


def test_upsample_length() -> None:
    x = np.zeros(8000, dtype=np.float32)
    y = resample(x, 8000, 16000)
    assert y.shape[0] == 16000


def test_multichannel() -> None:
    x = np.zeros((8000, 2), dtype=np.float32)
    y = resample(x, 8000, 16000)
    assert y.shape == (16000, 2)


def test_invalid_rate_raises() -> None:
    with pytest.raises(ValueError):
        resample(np.zeros(4, dtype=np.float32), 0, 16000)


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        resample(np.zeros(0, dtype=np.float32), 8000, 16000)
