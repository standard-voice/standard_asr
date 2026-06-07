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
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def resample(
    audio: NDArray[np.floating], orig_sr: int, target_sr: int
) -> NDArray[np.float32]:
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

    This mirrors the standard band-limited Fourier resampling algorithm:
    transform to the frequency domain, truncate (downsample) or zero-pad
    (upsample) about the Nyquist bin, then invert. Band-limiting in the
    frequency domain provides the anti-aliasing guarantee.

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
    half = n // 2

    # Positive frequencies (including DC), up to but excluding Nyquist.
    new_spectrum[: half + 1] = spectrum[: half + 1]
    # Negative frequencies.
    if n > 1:
        new_spectrum[num - half + (1 if n % 2 else 0):] = spectrum[
            n_in - half + (1 if n % 2 else 0):
        ]

    # Split the Nyquist component for even-length transforms so energy is
    # preserved symmetrically across positive/negative Nyquist bins.
    if n % 2 == 0:
        nyq = n // 2
        if num < n_in:
            # Downsampling: combine the two Nyquist images.
            new_spectrum[nyq] = spectrum[nyq] + spectrum[n_in - nyq]
        elif num > n_in:
            # Upsampling: halve and mirror the Nyquist bin.
            new_spectrum[nyq] = spectrum[nyq] * 0.5
            new_spectrum[num - nyq] = spectrum[nyq] * 0.5

    result = np.fft.ifft(new_spectrum, axis=0).real
    result *= float(num) / float(n_in)
    return result


__all__ = ["resample"]
