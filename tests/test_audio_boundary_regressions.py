# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for audio conversion and decode failure boundaries."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import standard_asr.audio.conversion as conversion
import standard_asr.audio.loader as loader
from standard_asr.audio.input import AudioArray, InputKind
from standard_asr.audio.negotiation import ConversionPlan, negotiate
from standard_asr.contract.exceptions import AudioProcessingError


def test_resampler_value_error_uses_the_audio_error_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing resampling backend must not leak a bare ``ValueError``."""

    def fail_resample(*_args: object) -> tuple[np.ndarray, str]:
        raise ValueError("backend rejected the waveform")

    monkeypatch.setattr(conversion, "resample_with_backend", fail_resample)
    audio = AudioArray(np.zeros(8, dtype=np.float32), 48000)
    plan = negotiate(audio, {InputKind.ARRAY})
    assert isinstance(plan, ConversionPlan)

    with pytest.raises(AudioProcessingError, match="Cannot resample") as exc_info:
        conversion.execute_plan(
            audio,
            plan,
            accepted_sample_rates=[16000],
            native_sample_rate=16000,
        )

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_array_decode_from_path_uses_unprocessed_ffmpeg_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Path decoding keeps FFmpeg repairs visible to canonicalization."""
    path = tmp_path / "encoded.bin"
    path.write_bytes(b"encoded")
    decoded = np.array([np.nan, 2.0], dtype=np.float32)

    def no_decode(*_args: object) -> None:
        return None

    monkeypatch.setattr(loader, "_read_wav_stdlib", no_decode)
    monkeypatch.setattr(loader, "_read_with_soundfile", no_decode)

    def decode_with_ffmpeg(
        source: str | bytes,
        channels: int | None,
        *,
        canonicalize_output: bool,
    ) -> tuple[np.ndarray, int]:
        assert source == str(path)
        assert channels is None
        assert not canonicalize_output
        return decoded, 24000

    monkeypatch.setattr(loader, "_decode_with_ffmpeg_native", decode_with_ffmpeg)

    array, sample_rate, report = loader.decode_audio_for_array(str(path))

    np.testing.assert_array_equal(array, np.array([0.0, 1.0], dtype=np.float32))
    assert sample_rate == 24000
    assert report.sanitized_non_finite == 1
    assert report.clipped_samples == 1


def test_array_decode_rejects_a_non_bytes_dynamic_value() -> None:
    """The public decode boundary reports mistyped runtime input clearly."""
    with pytest.raises(TypeError, match="Unsupported audio source type"):
        loader.decode_audio_for_array(42)  # type: ignore[arg-type]


def test_ffmpeg_repairs_are_reported_by_the_array_decode_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FFmpeg output stays raw until the array boundary records its repairs."""
    raw = np.array([np.nan, np.inf, -np.inf, 1.5], dtype=np.float32)

    def no_soundfile_decode(*_args: object) -> None:
        return None

    def find_ffmpeg(_name: str) -> str:
        return "/usr/bin/ffmpeg"

    def probe_sample_rate(
        _source: str | bytes,
    ) -> loader._ProbeOutcome:  # pyright: ignore[reportPrivateUsage]
        return loader._ProbeOutcome(16000)  # pyright: ignore[reportPrivateUsage]

    def probe_channels(
        _source: str | bytes,
    ) -> loader._ProbeOutcome:  # pyright: ignore[reportPrivateUsage]
        return loader._ProbeOutcome(1)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(loader, "_read_with_soundfile", no_soundfile_decode)
    monkeypatch.setattr(loader.shutil, "which", find_ffmpeg)
    monkeypatch.setattr(loader, "_probe_sample_rate_with_ffprobe", probe_sample_rate)
    monkeypatch.setattr(loader, "_probe_channels_with_ffprobe", probe_channels)

    def run_ffmpeg(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=raw.tobytes(), stderr=b"")

    monkeypatch.setattr(subprocess, "run", run_ffmpeg)

    array, sample_rate, report = loader.decode_audio_for_array(b"encoded")

    np.testing.assert_array_equal(array, np.array([0.0, 1.0, -1.0, 1.0], dtype=np.float32))
    assert sample_rate == 16000
    assert report.sanitized_non_finite == 3
    assert report.clipped_samples == 1
