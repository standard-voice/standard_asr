# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in anti-aliasing fallback resampler."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from standard_asr.resampling import resample, resample_with_backend


def _require_scipy_signal() -> Any:
    """Import ``scipy.signal`` or skip the test.

    Skips on a plain missing dependency (``ImportError``) and also on the
    ``TypeError`` that scipy's import can raise under coverage when numpy has been
    reloaded (a known coverage/numpy-C-tracer interaction unrelated to the code
    under test).
    """
    try:
        import scipy.signal as scipy_signal  # pyright: ignore[reportMissingTypeStubs]
    except ImportError:
        pytest.skip("scipy not installed")
    except TypeError:  # pragma: no cover - coverage+numpy-reload artifact
        pytest.skip("scipy.signal unimportable in this environment")
    return scipy_signal


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


def test_dc_signal_preserved_upsample() -> None:
    """Analytic case (no scipy): a constant signal stays constant after resample."""
    x = np.full(101, 0.5, dtype=np.float64)
    y = resample(x, 16000, 24000)
    assert y.shape[0] == int(round(101 * 24000 / 16000))
    # A pure DC signal has no high-frequency content; every sample must remain
    # the original constant to band-limited precision.
    assert np.allclose(y, 0.5, atol=1e-6)


def test_dc_signal_preserved_downsample_odd() -> None:
    """Analytic case (no scipy): DC preserved when min(num, n_in) is odd."""
    x = np.full(100, -0.25, dtype=np.float64)
    # 100 -> 51 makes min(num, n_in) = 51 (odd): the previously buggy path.
    y = resample(x, 100, 51)
    assert y.shape[0] == 51
    assert np.allclose(y, -0.25, atol=1e-6)


def test_single_tone_below_nyquist_roundtrip() -> None:
    """Analytic: a low-frequency cosine survives up- then down-sampling."""
    n = 200
    t = np.arange(n) / n
    # 3 cycles over the window: well below Nyquist at every stage.
    x = np.cos(2 * np.pi * 3 * t).astype(np.float64)
    up = resample(x, 1000, 3000)
    back = resample(up, 3000, 1000)
    # Edge effects from circular convolution are largest at the boundaries; the
    # interior must match the original closely.
    assert np.allclose(back[20:-20], x[20:-20], atol=1e-2)


@pytest.mark.parametrize("n_in", [2, 3, 4, 5, 100, 101, 127, 128, 16000, 16001])
@pytest.mark.parametrize("num", [2, 3, 50, 51, 64, 65, 150, 151])
def test_matches_scipy_resample(n_in: int, num: int) -> None:
    """Exhaustive regression: match scipy.signal.resample across odd/even sizes.

    This is the C3 guard: the hand-written Fourier resampler must agree with the
    reference implementation to floating-point precision for *both* parities of
    ``min(num, n_in)``. The previous implementation diverged by ~1e-1 whenever
    that value was odd.
    """
    scipy_signal = _require_scipy_signal()
    rng = np.random.default_rng(n_in * 1000 + num)
    x = rng.standard_normal(n_in)
    expected = scipy_signal.resample(x, num)
    actual = resample(x.astype(np.float64), n_in, num).astype(np.float64)
    scale = float(np.max(np.abs(expected))) + 1e-12
    assert np.max(np.abs(expected - actual)) / scale < 1e-6


def test_matches_scipy_realtime_16k_to_24k_odd_frame() -> None:
    """C3 hot path: 16k->24k on an odd-length realtime frame matches scipy."""
    scipy_signal = _require_scipy_signal()
    rng = np.random.default_rng(7)
    # 321 samples at 16 kHz -> min(num, n_in) is odd, the broken case.
    x = rng.standard_normal(321)
    num = int(round(321 * 24000 / 16000))
    expected = scipy_signal.resample(x, num)
    actual = resample(x.astype(np.float64), 16000, 24000).astype(np.float64)
    scale = float(np.max(np.abs(expected))) + 1e-12
    assert np.max(np.abs(expected - actual)) / scale < 1e-6


def test_matches_scipy_multichannel() -> None:
    """2D resampling matches scipy column-wise."""
    scipy_signal = _require_scipy_signal()
    rng = np.random.default_rng(11)
    x = rng.standard_normal((101, 2))
    expected = scipy_signal.resample(x, 151, axis=0)
    actual = resample(x.astype(np.float64), 101, 151).astype(np.float64)
    scale = float(np.max(np.abs(expected))) + 1e-12
    assert np.max(np.abs(expected - actual)) / scale < 1e-6


def test_backend_identity_when_rates_match() -> None:
    out, backend = resample_with_backend(
        np.array([0.1, 0.2], dtype=np.float32), 16000, 16000
    )
    assert out.dtype == np.float32
    assert backend == "fallback"


def test_backend_reports_scipy_when_available() -> None:
    _require_scipy_signal()
    out, backend = resample_with_backend(
        np.zeros(16000, dtype=np.float32), 16000, 24000
    )
    assert backend == "scipy"
    assert out.shape[0] == 24000


# Note: the scipy-absent fallback branch of resample_with_backend is exercised by
# the no-[audio] CI lane (where scipy is not importable). We deliberately do NOT
# break the scipy import in-process here: purging scipy from sys.modules forces a
# numpy reload that corrupts scipy state for sibling tests under coverage.
