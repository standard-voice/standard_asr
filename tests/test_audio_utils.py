# Copyright 2025 The Standard ASR Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Essential tests for audio_loader module focusing on key functionality and regressions."""

import base64
import io
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typing import Any, cast
from numpy.typing import NDArray

from standard_asr.utils.audio_loader import (
    load_audio,
    normalize_audio,
)
from standard_asr.utils.audio_loader import _load_with_ffmpeg  # pyright: ignore[reportPrivateUsage]
from standard_asr.utils.save_utils import nparray_to_audio_file
from standard_asr.exceptions import AudioProcessingError, FFmpegNotFoundError


def create_sine_wave(
    sr: int = 16000, duration: float = 0.1, channels: int = 1
) -> NDArray[np.float32]:
    """Create a test sine wave."""
    n_samples = int(sr * duration)
    t: NDArray[np.float32] = np.arange(n_samples, dtype=np.float32) / np.float32(sr)
    omega: np.float32 = np.float32(2.0 * np.pi * 440.0)
    raw = np.sin(omega * t)
    sine = raw.astype(np.float32, copy=False)

    if channels == 1:
        return sine
    else:
        stacked = np.stack([sine] * channels, axis=1)
        return stacked.astype(np.float32, copy=False)


def test_audio_contract_dtype():
    """All audio output should be np.float32."""
    audio = create_sine_wave()
    result = normalize_audio(audio, 16000, 16000, 1)
    assert result.dtype == np.float32


def test_audio_contract_range():
    """Audio values should be clipped to [-1, 1]."""
    audio = np.array([2.0, -3.0, 0.5], dtype=np.float32)
    result = normalize_audio(audio, 16000, 16000, 1)

    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)
    assert np.all(np.isfinite(result))


def test_single_sample_not_scalar():
    """Single sample mono should be (1,) not scalar - regression test for squeeze bug."""
    audio = np.array([0.5], dtype=np.float32)
    result = normalize_audio(audio, 16000, 16000, 1)

    assert result.ndim == 1
    assert result.shape == (1,)
    assert not np.isscalar(result)
    assert result[0] == 0.5


def test_empty_audio_raises_error():
    """Empty audio should raise AudioProcessingError."""
    empty_audio = np.array([], dtype=np.float32)

    try:
        normalize_audio(empty_audio, 16000, 16000, 1)
        assert False, "Should have raised AudioProcessingError"
    except AudioProcessingError as e:
        assert "Cannot process empty audio" in str(e)


def test_nan_inf_cleanup():
    """Audio with NaN/Inf should be cleaned up."""
    bad_audio = np.array([0.5, np.nan, np.inf, -np.inf, 0.3], dtype=np.float32)

    cleaned = normalize_audio(bad_audio, 16000, 16000, 1)

    assert np.all(np.isfinite(cleaned))
    assert cleaned[0] == 0.5  # Good value preserved
    assert cleaned[1] == 0.0  # NaN -> 0
    assert cleaned[2] == 1.0  # +Inf -> 1
    assert cleaned[3] == -1.0  # -Inf -> -1
    assert cleaned[4] == 0.3  # Good value preserved


def test_mono_stereo_shapes():
    """Test mono and stereo output shapes."""
    audio = create_sine_wave(channels=2)

    # Convert to mono
    mono = normalize_audio(audio, 16000, 16000, 1)
    assert mono.ndim == 1

    # Keep stereo
    stereo = normalize_audio(audio, 16000, 16000, 2)
    assert stereo.ndim == 2
    assert stereo.shape[1] == 2


def test_load_from_bytes():
    """Test loading from bytes."""
    # Create test audio and save to bytes
    audio = create_sine_wave()
    temp_path = "temp_test.wav"
    nparray_to_audio_file(audio, temp_path, 16000)

    try:
        with open(temp_path, "rb") as f:
            data = f.read()

        loaded = load_audio(data)
        assert isinstance(loaded, np.ndarray)
        assert loaded.dtype == np.float32
    finally:
        # Cleanup
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass


def test_load_from_base64():
    """Test loading from base64 data URI."""
    # Create test audio and convert to base64
    audio = create_sine_wave()
    temp_path = "temp_test.wav"
    nparray_to_audio_file(audio, temp_path, 16000)

    try:
        with open(temp_path, "rb") as f:
            data = f.read()

        b64_data = base64.b64encode(data).decode()
        uri = f"data:audio/wav;base64,{b64_data}"

        loaded = load_audio(uri)
        assert isinstance(loaded, np.ndarray)
        assert loaded.dtype == np.float32
    finally:
        # Cleanup
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass


def test_channel_downmix():
    """Test channel down-mixing."""
    stereo = np.array([[0.6, 0.4], [0.8, 0.2]], dtype=np.float32)

    mono = normalize_audio(stereo, 16000, 16000, 1)

    expected = np.array([0.5, 0.5], dtype=np.float32)  # Average
    np.testing.assert_array_almost_equal(mono, expected)


def test_channel_upmix():
    """Test channel up-mixing."""
    mono = np.array([0.5, 0.3], dtype=np.float32)

    stereo = normalize_audio(mono, 16000, 16000, 2)

    expected = np.array([[0.5, 0.5], [0.3, 0.3]], dtype=np.float32)
    np.testing.assert_array_almost_equal(stereo, expected)


def test_invalid_parameters():
    """Test various invalid parameter scenarios."""
    audio = np.array([0.1, 0.2], dtype=np.float32)

    # Invalid target_sr
    try:
        normalize_audio(audio, 16000, 0, 1)
        assert False, "Should have raised AudioProcessingError"
    except AudioProcessingError as e:
        assert "target_sr must be > 0" in str(e)

    # Invalid target_channels
    try:
        normalize_audio(audio, 16000, 16000, 0)
        assert False, "Should have raised AudioProcessingError"
    except AudioProcessingError as e:
        assert "target_channels must be None or > 0" in str(e)


def test_nonexistent_file():
    """Non-existent file should raise appropriate error."""
    try:
        load_audio("/path/that/does/not/exist.wav")
        assert False, "Should have raised an error"
    except (FileNotFoundError, AudioProcessingError):
        pass  # Expected


def test_unsupported_source_type():
    """Unsupported source type should raise TypeError."""
    try:
        load_audio(123)  # type: ignore[arg-type]
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Unsupported audio source type" in str(e)


@patch("shutil.which")
def test_missing_ffmpeg(mock_which: Any):
    """Missing FFmpeg should raise FFmpegNotFoundError."""
    mock_which.return_value = None

    try:
        _load_with_ffmpeg(b"fake_data", 16000, 1)
        assert False, "Should have raised FFmpegNotFoundError"
    except FFmpegNotFoundError as e:
        assert "FFmpeg not found in PATH" in str(e)


@patch("subprocess.run")
@patch("shutil.which")
def test_ffmpeg_timeout(mock_which: Any, mock_run: Any):
    """FFmpeg timeout should raise AudioProcessingError."""
    mock_which.return_value = "/usr/bin/ffmpeg"
    mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 120.0)

    try:
        _load_with_ffmpeg(b"fake_data", 16000, 1, timeout=120.0)
        assert False, "Should have raised AudioProcessingError"
    except AudioProcessingError as e:
        assert "FFmpeg timed out" in str(e)


def test_resampling_length():
    """Resampled audio length should be approximately correct."""
    # Create 0.5 second of audio at 8kHz
    original_sr = 8000
    target_sr = 16000
    duration = 0.5

    audio = create_sine_wave(original_sr, duration, 1)
    resampled = normalize_audio(audio, original_sr, target_sr, 1)

    expected_length = int(duration * target_sr)
    # Allow ±2 sample tolerance due to resampling
    assert abs(len(resampled) - expected_length) <= 2


def test_downmix_dtype_float32():
    """Downmix from stereo to mono should keep dtype float32."""
    stereo = np.stack([create_sine_wave(), create_sine_wave()], axis=1).astype(
        np.float32, copy=False
    )
    out = normalize_audio(stereo, 16000, 16000, 1)

    assert out.dtype == np.float32


def test_resample_dtype_float32():
    """Resampling path should keep dtype float32 (requires scipy)."""
    pytest.importorskip("scipy", reason="scipy required for resampling test")

    audio = create_sine_wave(sr=8000)
    out = normalize_audio(audio, 8000, 16000, 1)

    assert out.dtype == np.float32


def test_bytes_like_inputs_variants():
    """bytearray/memoryview inputs should load successfully with float32 dtype."""
    audio = create_sine_wave()
    temp_path = "temp_test_variants.wav"
    nparray_to_audio_file(audio, temp_path, 16000)

    try:
        with open(temp_path, "rb") as f:
            raw = f.read()

        for blob in (raw, bytearray(raw), memoryview(raw)):
            loaded = load_audio(blob)
            assert isinstance(loaded, np.ndarray)
            assert loaded.dtype == np.float32
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass


def test_text_io_not_supported():
    """Text IO (StringIO) should be rejected as unsupported type (not binary)."""
    with pytest.raises(TypeError):
        load_audio(cast(Any, io.StringIO("abc")))


def test_invalid_base64_data_uri():
    """Invalid base64 data URI should raise AudioProcessingError."""
    uri = "data:audio/wav;base64,@@not_valid_base64@@"
    try:
        load_audio(uri)
        assert False, "Should have raised AudioProcessingError"
    except AudioProcessingError:
        pass


def main():
    """Run basic tests manually if pytest not available."""
    print("Running basic audio_utils tests...")

    test_audio_contract_dtype()
    print("✓ Audio contract dtype test passed")

    test_single_sample_not_scalar()
    print("✓ Single sample regression test passed")

    test_empty_audio_raises_error()
    print("✓ Empty audio error test passed")

    test_nan_inf_cleanup()
    print("✓ NaN/Inf cleanup test passed")

    test_invalid_parameters()
    print("✓ Invalid parameters test passed")

    print("All basic tests passed! ✅")


if __name__ == "__main__":
    main()
