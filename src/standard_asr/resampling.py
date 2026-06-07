# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Anti-aliasing sample-rate conversion.

The standard never hard-fails on a sample-rate mismatch: it always has a
numpy-only, anti-aliasing fallback resampler (spec, section "Audio Input &
Sample Rate", rule R8). The fallback is a clean-room Fourier-domain resampler
(no vendored SoX/soxr), which is inherently band-limited and so anti-aliasing --
never naive linear interpolation or decimation. When the optional ``[audio]``
extra provides a higher-quality resampler, engines/the loader may use it on the
hot path instead; the fallback exists so a missing extra is never fatal.

Algorithm note: :func:`_resample_fourier` re-derives the standard band-limited
Fourier-resampling algorithm (the same approach ``scipy.signal.resample`` uses).
The algorithm is not copyrightable; this is an independent implementation and no
SoX/soxr/libsamplerate code is vendored (spec R8, design decision D3). It matches
``scipy.signal.resample`` to floating-point precision for both even- and
odd-length transforms.
"""

from __future__ import annotations

from math import gcd
from typing import Literal

import numpy as np
from numpy.typing import NDArray

#: Identifies which resampler produced a result: the high-quality scipy
#: polyphase resampler (``[audio]`` extra), or the built-in numpy fallback.
ResampleBackend = Literal["scipy", "fallback"]


def resample_with_backend(
    audio: NDArray[np.floating], orig_sr: int, target_sr: int
) -> tuple[NDArray[np.float32], ResampleBackend]:
    """Resample, preferring scipy's polyphase resampler when ``[audio]`` is present.

    The high-quality path (``scipy.signal.resample_poly``) is used when scipy is
    importable; otherwise the built-in numpy anti-aliasing Fourier fallback is
    used. The chosen backend is returned so callers can emit a truthful
    ``resampled_with`` diagnostic (only label it ``fallback`` when the fallback
    actually ran -- spec R8).

    Args:
        audio: Float waveform array (1D or 2D along axis 0).
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz.

    Returns:
        A ``(resampled_float32, backend)`` pair.

    Raises:
        ValueError: If a sample rate is non-positive or the input is empty.
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive.")
    array = np.asarray(audio, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot resample empty audio.")
    if orig_sr == target_sr:
        return array.astype(np.float32), "fallback"

    try:
        from scipy.signal import (  # pyright: ignore[reportMissingTypeStubs]
            resample_poly as _resample_poly,  # pyright: ignore[reportUnknownVariableType]
        )
    except Exception:
        # ImportError when [audio] is absent; other import-time errors (e.g. a
        # broken/partially-initialized scipy build) are also non-fatal here -- a
        # crashing optional dependency MUST degrade to the built-in fallback, not
        # propagate (battery-included DX, spec R8).
        return resample(array, orig_sr, target_sr), "fallback"

    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    out = np.asarray(
        _resample_poly(array, up=up, down=down, axis=0),  # pyright: ignore[reportUnknownArgumentType]
        dtype=np.float32,
    )
    return out, "scipy"


def resample(audio: NDArray[np.floating], orig_sr: int, target_sr: int) -> NDArray[np.float32]:
    """Resample a waveform between sample rates (anti-aliasing, FFT-based).

    Works on mono (1D) or multi-channel (2D ``(n_samples, n_channels)``) input
    along the sample axis. Identity when the rates match.

    Args:
        audio: Float waveform array.
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz.

    Returns:
        The resampled waveform as ``float32``.

    Raises:
        ValueError: If a sample rate is non-positive or the input is empty.
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive.")
    array = np.asarray(audio, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot resample empty audio.")
    if orig_sr == target_sr:
        return array.astype(np.float32)

    n_in = array.shape[0]
    n_out = int(round(n_in * target_sr / orig_sr))
    if n_out < 1:
        n_out = 1
    resampled = _resample_fourier(array, n_out)
    return resampled.astype(np.float32)


def _resample_fourier(x: NDArray[np.float64], num: int) -> NDArray[np.float64]:
    """Resample ``x`` to ``num`` samples along axis 0 via the Fourier method.

    Re-derivation of the standard band-limited Fourier resampling algorithm
    (as used by ``scipy.signal.resample``): take the forward FFT, copy the
    ``N = min(num, n_in)`` lowest-magnitude frequency bins into an output
    spectrum of length ``num`` (zero-padding for upsampling, truncating for
    downsampling), and -- crucially -- split the Nyquist bin **only when ``N`` is
    even** so the real-valued symmetry is preserved. Inverting then rescaling by
    ``num / n_in`` yields the resampled signal. Band-limiting in the frequency
    domain provides the anti-aliasing guarantee.

    For odd ``N`` there is no single Nyquist bin to split: the highest retained
    positive frequency and its conjugate are copied unchanged. Splitting in that
    case (the previous bug) introduced aliasing/energy errors of order 1e-1.

    Args:
        x: Input array (1D or 2D), real-valued.
        num: Desired number of output samples along axis 0.

    Returns:
        The resampled real array.
    """
    n_in = x.shape[0]
    spectrum = np.fft.fft(x, axis=0)

    out_shape = (num,) + x.shape[1:]
    new_spectrum = np.zeros(out_shape, dtype=complex)

    n = min(num, n_in)
    # Number of positive-frequency bins to copy, INCLUDING DC, EXCLUDING the
    # Nyquist bin. For even n this is n // 2; for odd n it is (n + 1) // 2 - 1,
    # i.e. (n - 1) // 2, but copying through index n // 2 (inclusive) is correct
    # for both parities because of how the negative side is mirrored below.
    half = (n + 1) // 2

    # Low (positive) frequencies including DC: indices [0, half).
    new_spectrum[:half] = spectrum[:half]
    # High (negative) frequencies: the last (n - half) bins of the input wrap to
    # the last (n - half) bins of the output.
    n_neg = n - half
    if n_neg > 0:
        new_spectrum[num - n_neg :] = spectrum[n_in - n_neg :]

    # Nyquist handling applies only when N is even: the single Nyquist bin at
    # index n // 2 must carry the energy of both the positive and negative
    # Nyquist images so the output stays real and energy is preserved.
    if n % 2 == 0:
        nyq = n // 2
        if num < n_in:
            # Downsampling: fold the two input Nyquist images into one bin.
            new_spectrum[nyq] = spectrum[nyq] + spectrum[n_in - nyq]
        elif num > n_in:
            # Upsampling: the lone input Nyquist bin splits symmetrically.
            new_spectrum[nyq] = spectrum[nyq] * 0.5
            new_spectrum[num - nyq] = np.conj(spectrum[nyq]) * 0.5
        else:  # num == n_in handled by the rate-equality fast path; defensive.
            new_spectrum[nyq] = spectrum[nyq]  # pragma: no cover

    result = np.fft.ifft(new_spectrum, axis=0).real
    result *= float(num) / float(n_in)
    return result


__all__ = ["ResampleBackend", "resample", "resample_with_backend"]
