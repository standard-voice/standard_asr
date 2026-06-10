# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for audio saving utilities."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from standard_asr.exceptions import AudioProcessingError
from standard_asr.utils.save_utils import encode_array_to_wav_bytes, save_wav


def test_save_wav_writes_multichannel(tmp_path: Path) -> None:
    audio = np.array([[0.0, 0.5], [-0.5, 0.25]], dtype=np.float32)
    path = tmp_path / "multi.wav"

    save_wav(audio, str(path), sample_rate=8000)

    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getframerate() == 8000


def test_save_wav_accepts_pathlike(tmp_path: Path) -> None:
    # Save_wav accepts os.PathLike (a Path), not only str.
    audio = np.zeros(8, dtype=np.float32)
    path = tmp_path / "pathlike.wav"
    save_wav(audio, path, sample_rate=16000)  # pass the Path directly
    assert path.is_file()


def test_save_wav_writes_little_endian_pcm(tmp_path: Path) -> None:
    # save_wav must serialize 16-bit PCM little-endian regardless of host byte
    # order. +1.0 -> 32767 -> bytes (0xFF, 0x7F) LE.
    audio = np.array([1.0], dtype=np.float32)
    path = tmp_path / "le.wav"
    save_wav(audio, str(path), sample_rate=16000)
    with wave.open(str(path), "rb") as wf:
        assert wf.readframes(wf.getnframes()) == b"\xff\x7f"


def test_save_wav_raises_on_write_error() -> None:
    audio = np.zeros(4, dtype=np.float32)

    with patch("wave.open", side_effect=OSError("boom")):
        with pytest.raises(OSError):
            save_wav(audio, "missing.wav")


# --- input validation on the encode helpers ---


@pytest.mark.parametrize("bad_rate", [0, -1, -16000])
def test_encode_array_rejects_nonpositive_sample_rate(bad_rate: int) -> None:
    with pytest.raises(AudioProcessingError, match="sample_rate must be > 0"):
        encode_array_to_wav_bytes(np.zeros(8, dtype=np.float32), bad_rate)


@pytest.mark.parametrize("bad_rate", [0, -8000])
def test_save_wav_rejects_nonpositive_sample_rate(tmp_path: Path, bad_rate: int) -> None:
    with pytest.raises(AudioProcessingError, match="sample_rate must be > 0"):
        save_wav(np.zeros(8, dtype=np.float32), tmp_path / "x.wav", bad_rate)


def test_encode_array_rejects_empty_audio() -> None:
    # Decode-side rejects empty audio ("Cannot process empty audio array"); the
    # encode side must be symmetric rather than emitting a 0-frame WAV.
    with pytest.raises(AudioProcessingError, match="Cannot encode empty audio array"):
        encode_array_to_wav_bytes(np.zeros(0, dtype=np.float32), 16000)


def test_encode_array_rejects_integer_dtype() -> None:
    # An int16 PCM array passed to the float encoder would be crushed
    # to a square wave by np.clip(-1, 1); reject it loudly instead.
    pcm = np.array([1000, -2000, 32767], dtype=np.int16)
    with pytest.raises(AudioProcessingError, match="floating-point waveform"):
        encode_array_to_wav_bytes(pcm, 16000)  # type: ignore[arg-type]


def test_save_wav_rejects_integer_dtype(tmp_path: Path) -> None:
    pcm = np.array([1000, -2000], dtype=np.int16)
    with pytest.raises(AudioProcessingError, match="floating-point waveform"):
        save_wav(pcm, tmp_path / "x.wav")  # type: ignore[arg-type]


def test_encode_array_rejects_3d() -> None:
    with pytest.raises(AudioProcessingError, match="1D .* or 2D"):
        encode_array_to_wav_bytes(np.zeros((4, 2, 3), dtype=np.float32), 16000)


def test_save_wav_rejects_3d(tmp_path: Path) -> None:
    # A 3D array previously took shape[1] as the channel count and
    # silently wrote an interleaved, misaligned WAV. Now it is rejected.
    with pytest.raises(AudioProcessingError, match="1D .* or 2D"):
        save_wav(np.zeros((100, 2, 3), dtype=np.float32), tmp_path / "x.wav")


def test_encode_array_reports_sanitized_non_finite() -> None:
    # The count of sanitized NaN/Inf samples is surfaced so the
    # conversion layer can emit a diagnostic (the sanitize itself is necessary).
    audio = np.array([0.0, np.nan, np.inf, -np.inf, 0.5], dtype=np.float32)
    result = encode_array_to_wav_bytes(audio, 16000)
    assert result.sanitized_non_finite == 3  # NaN, +Inf, -Inf


def test_encode_array_finite_input_reports_zero_sanitized() -> None:
    result = encode_array_to_wav_bytes(np.array([0.0, 0.5, -0.5], dtype=np.float32), 16000)
    assert result.sanitized_non_finite == 0


@pytest.mark.parametrize(
    ("amplitude", "rounded", "truncated"),
    [
        (np.float32(16383.5 / 32767.0), 16384, 16383),
        (np.float32(-16383.5 / 32767.0), -16384, -16383),
    ],
)
def test_encode_array_quantizes_round_half_not_truncate(
    amplitude: np.float32, rounded: int, truncated: int
) -> None:
    # RR-006: round-half (np.rint) quantization, not toward-zero
    # truncation. Each amplitude's clipped * 32767 product lands on a true half-LSB
    # (+/-16383.5), where round-half and truncation give DIFFERENT int16 codes; the
    # negative case pins the toward-zero magnitude bias the spec calls out. The old
    # test used 16384.0/32767 -> an INTEGER product, where both agree, so it could
    # not catch a truncation regression.
    assert rounded != truncated  # self-guard: the value must actually discriminate
    result = encode_array_to_wav_bytes(np.array([amplitude], dtype=np.float32), 16000)
    with wave.open(io.BytesIO(result.data), "rb") as wf:
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    assert int(pcm[0]) == rounded
