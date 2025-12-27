"""Tests for runtime helpers."""

import os
from pathlib import Path

import numpy as np
import pytest

from standard_asr.asr_properties import BaseProperties
from standard_asr.exceptions import AudioProcessingError
from standard_asr.runtime import allow_downloads, ensure_cache_dir, resolve_cache_dir, validate_audio_input


class _Props(BaseProperties):
    engine_id: str = "dummy"
    model_name: str = "demo"
    protocol_version: str = "0.2.0"
    supported_languages: list[str] = ["en"]
    supported_devices: list[str] = ["cpu"]
    supported_sample_rates: list[int] = [16000]
    supported_channels: list[int] = [1]
    audio_dtype: str = "float32"


def test_allow_downloads_env(monkeypatch) -> None:
    monkeypatch.delenv("STANDARD_ASR_ALLOW_DOWNLOAD", raising=False)
    assert allow_downloads() is True

    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    assert allow_downloads() is False

    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "yes")
    assert allow_downloads() is True


def test_cache_dir_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STANDARD_ASR_MODEL_DIR", str(tmp_path))
    resolved = resolve_cache_dir()
    assert resolved == tmp_path

    ensured = ensure_cache_dir()
    assert ensured.exists()


def test_cache_dir_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
    resolved = resolve_cache_dir()
    assert isinstance(resolved, Path)


def test_cache_dir_windows_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(os, "name", "nt", raising=False)

    resolved = resolve_cache_dir()
    assert resolved == tmp_path / "standard-asr"


def test_validate_audio_input_channels() -> None:
    props = _Props()
    audio = np.zeros((10, 2), dtype=np.float32)

    with pytest.raises(AudioProcessingError):
        validate_audio_input(audio, props)


def test_validate_audio_input_casts_dtype() -> None:
    props = _Props()
    audio = np.zeros(10, dtype=np.float64)

    out = validate_audio_input(audio, props)
    assert out.dtype == np.float32


def test_validate_audio_input_invalid_shape() -> None:
    props = _Props()
    audio = np.zeros((1, 2, 3), dtype=np.float32)

    with pytest.raises(AudioProcessingError):
        validate_audio_input(audio, props)
