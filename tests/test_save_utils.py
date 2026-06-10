"""Tests for audio saving utilities."""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from standard_asr.utils.save_utils import nparray_to_audio_file


def test_save_utils_writes_multichannel(tmp_path: Path) -> None:
    audio = np.array([[0.0, 0.5], [-0.5, 0.25]], dtype=np.float32)
    path = tmp_path / "multi.wav"

    nparray_to_audio_file(audio, str(path), sample_rate=8000)

    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getframerate() == 8000


def test_save_utils_writes_little_endian_pcm(tmp_path: Path) -> None:
    # nparray_to_audio_file must serialize 16-bit PCM little-endian
    # regardless of host byte order. +1.0 -> 32767 -> bytes (0xFF, 0x7F) LE.
    audio = np.array([1.0], dtype=np.float32)
    path = tmp_path / "le.wav"
    nparray_to_audio_file(audio, str(path), sample_rate=16000)
    with wave.open(str(path), "rb") as wf:
        assert wf.readframes(wf.getnframes()) == b"\xff\x7f"


def test_save_utils_raises_on_write_error() -> None:
    audio = np.zeros(4, dtype=np.float32)

    with patch("wave.open", side_effect=OSError("boom")):
        with pytest.raises(OSError):
            nparray_to_audio_file(audio, "missing.wav")
