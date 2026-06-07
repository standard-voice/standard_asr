"""Coverage-oriented tests for audio loader branches."""

from __future__ import annotations

import builtins
import io
import pathlib
import subprocess
import types
import wave
from pathlib import Path
import numpy as np
import pytest
from numpy.typing import NDArray

import standard_asr.utils.audio_loader as audio_loader
from standard_asr.exceptions import AudioProcessingError


def _write_wav(path: Path, sampwidth: int, channels: int = 1) -> None:
    sample_rate = 16000
    frames = 4

    if sampwidth == 1:
        data = np.full((frames, channels), 128, dtype=np.uint8)
    else:
        data = np.zeros((frames, channels), dtype=np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def test_ensure_datatype_casts() -> None:
    audio = np.array([1, 2], dtype=np.int16)

    out = audio_loader.ensure_datatype(audio, np.float32)

    assert out.dtype == np.float32


def test_normalize_audio_invalid_original_sr() -> None:
    audio = np.array([0.1], dtype=np.float32)

    with pytest.raises(AudioProcessingError):
        audio_loader.normalize_audio(audio, 0, 16000, 1)


def test_normalize_audio_resample_fallback_without_scipy(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    audio = np.zeros(8, dtype=np.float32)
    real_import = builtins.__import__

    def _import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("scipy"):
            raise ImportError("no scipy")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    caplog.set_level("WARNING")

    # Without scipy the built-in anti-aliasing fallback resampler is used
    # (spec AI R8): never a hard failure.
    out = audio_loader.normalize_audio(audio, 8000, 16000, 1)

    assert out.shape[0] == 16  # 8 samples at 8 kHz -> 16 at 16 kHz
    assert any("fallback resampler" in record.message for record in caplog.records)


def test_normalize_audio_truncates_channels(caplog: pytest.LogCaptureFixture) -> None:
    audio = np.zeros((2, 3), dtype=np.float32)

    caplog.set_level("WARNING")
    out = audio_loader.normalize_audio(audio, 16000, 16000, 2)

    assert out.shape[1] == 2
    assert any("Down-mixing" in record.message for record in caplog.records)


def test_load_audio_invalid_params() -> None:
    with pytest.raises(AudioProcessingError):
        audio_loader.load_audio(b"data", target_sr=0)

    with pytest.raises(AudioProcessingError):
        audio_loader.load_audio(b"data", target_channels=0)


def test_load_audio_existing_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "exists.wav"
    path.write_bytes(b"placeholder")
    sentinel: NDArray[np.float32] = np.zeros(1, dtype=np.float32)

    def _load_audio_from_path(
        path_str: str, target_sr: int = 16000, target_channels: int | None = 1
    ) -> NDArray[np.float32]:
        assert path_str == str(path)
        return sentinel

    monkeypatch.setattr(audio_loader, "load_audio_from_path", _load_audio_from_path)

    out = audio_loader.load_audio(str(path))

    assert out is sentinel


def test_load_audio_pathlib(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "pathlike.wav"
    path.write_bytes(b"placeholder")
    sentinel: NDArray[np.float32] = np.zeros(1, dtype=np.float32)

    def _load_audio_from_path(
        path_str: str, target_sr: int = 16000, target_channels: int | None = 1
    ) -> NDArray[np.float32]:
        assert path_str == str(path)
        return sentinel

    monkeypatch.setattr(audio_loader, "load_audio_from_path", _load_audio_from_path)

    out = audio_loader.load_audio(path)

    assert out is sentinel


def test_load_audio_path_exists_probe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel: NDArray[np.float32] = np.zeros(1, dtype=np.float32)

    def _load_audio_from_path(
        path_str: str, target_sr: int = 16000, target_channels: int | None = 1
    ) -> NDArray[np.float32]:
        return sentinel

    def _raise_exists(self: pathlib.Path) -> bool:
        raise OSError("boom")

    monkeypatch.setattr(audio_loader, "load_audio_from_path", _load_audio_from_path)
    monkeypatch.setattr(pathlib.Path, "exists", _raise_exists)

    out = audio_loader.load_audio("/tmp/missing.wav")

    assert out is sentinel


def test_load_audio_binary_io(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel: NDArray[np.float32] = np.zeros(2, dtype=np.float32)

    def _load_audio_from_bytes(
        data: bytes, target_sr: int = 16000, target_channels: int | None = 1
    ) -> NDArray[np.float32]:
        assert data == b"abc"
        return sentinel

    monkeypatch.setattr(audio_loader, "load_audio_from_bytes", _load_audio_from_bytes)

    out = audio_loader.load_audio(io.BytesIO(b"abc"))

    assert out is sentinel


def test_is_binary_io_variants() -> None:
    assert audio_loader._is_binary_io(io.BytesIO(b"abc")) is True  # pyright: ignore[reportPrivateUsage]

    class _BadRead:
        def read(self, _: int = 0) -> bytes:
            raise RuntimeError("boom")

    assert audio_loader._is_binary_io(_BadRead()) is False  # pyright: ignore[reportPrivateUsage]


def test_load_audio_from_path_validates_params() -> None:
    with pytest.raises(AudioProcessingError):
        audio_loader.load_audio_from_path("/tmp/test.wav", target_sr=0)

    with pytest.raises(AudioProcessingError):
        audio_loader.load_audio_from_path("/tmp/test.wav", target_channels=0)


def test_load_audio_from_path_wav_8bit(tmp_path: Path) -> None:
    path = tmp_path / "audio8.wav"
    _write_wav(path, sampwidth=1, channels=1)

    out = audio_loader.load_audio_from_path(str(path))

    assert out.dtype == np.float32


def test_load_audio_from_path_wav_16bit_stereo(tmp_path: Path) -> None:
    path = tmp_path / "audio16.wav"
    _write_wav(path, sampwidth=2, channels=2)

    out = audio_loader.load_audio_from_path(str(path), target_channels=None)

    assert out.ndim == 2
    assert out.shape[1] == 2


def test_load_audio_from_path_wav_unsupported_sampwidth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel: NDArray[np.float32] = np.zeros(1, dtype=np.float32)

    class _FakeWave:
        def __enter__(self) -> "_FakeWave":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def getframerate(self) -> int:
            return 16000

        def getsampwidth(self) -> int:
            return 3

        def getnchannels(self) -> int:
            return 1

        def getnframes(self) -> int:
            return 0

        def readframes(self, _: int) -> bytes:
            return b""

    real_import = builtins.__import__

    def _import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("soundfile"):
            raise ImportError("no soundfile")
        return real_import(name, globals, locals, fromlist, level)

    def _open(*_args: object, **_kwargs: object) -> _FakeWave:
        return _FakeWave()

    def _fake_load(*_: object) -> NDArray[np.float32]:
        return sentinel

    monkeypatch.setattr(wave, "open", _open)
    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.setattr(audio_loader, "_load_with_ffmpeg", _fake_load)

    out = audio_loader.load_audio_from_path("/tmp/sample.wav")

    assert out is sentinel


def test_load_audio_from_path_soundfile_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("soundfile")

    def _read(path: str, dtype: str = "float32") -> tuple[NDArray[np.float32], int]:
        return np.zeros(4, dtype=np.float32), 16000

    setattr(module, "read", _read)
    monkeypatch.setitem(__import__("sys").modules, "soundfile", module)

    out = audio_loader.load_audio_from_path("dummy.flac")

    assert out.dtype == np.float32


def test_load_audio_from_path_soundfile_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("soundfile")

    def _read(path: str, dtype: str = "float32") -> tuple[NDArray[np.float32], int]:
        raise RuntimeError("boom")

    setattr(module, "read", _read)
    monkeypatch.setitem(__import__("sys").modules, "soundfile", module)

    def _fake_load(*_: object) -> NDArray[np.float32]:
        return np.zeros(1, dtype=np.float32)

    monkeypatch.setattr(audio_loader, "_load_with_ffmpeg", _fake_load)

    out = audio_loader.load_audio_from_path("dummy.flac")

    assert isinstance(out, np.ndarray)


def test_load_audio_from_path_soundfile_missing_scipy_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module = types.ModuleType("soundfile")

    def _read(path: str, dtype: str = "float32") -> tuple[NDArray[np.float32], int]:
        return np.zeros(4, dtype=np.float32), 8000

    setattr(module, "read", _read)
    monkeypatch.setitem(__import__("sys").modules, "soundfile", module)

    real_import = builtins.__import__

    def _import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("scipy"):
            raise ImportError("missing scipy")
        return real_import(name, globals, locals, fromlist, level)

    def _fake_load(*_: object) -> NDArray[np.float32]:
        raise AssertionError("FFmpeg must not be used when the fallback resampler works")

    caplog.set_level("WARNING")
    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.setattr(audio_loader, "_load_with_ffmpeg", _fake_load)

    # soundfile decodes; the built-in fallback resampler handles 8k -> 16k
    # without scipy and without falling back to FFmpeg (spec AI R8).
    out = audio_loader.load_audio_from_path("dummy.flac", target_sr=16000)

    assert out.shape[0] == 8  # 4 samples at 8 kHz -> 8 at 16 kHz
    assert any("fallback resampler" in record.message for record in caplog.records)


def test_load_audio_from_path_soundfile_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("soundfile"):
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    def _fake_load(*_: object) -> NDArray[np.float32]:
        return np.zeros(1, dtype=np.float32)

    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.setattr(audio_loader, "_load_with_ffmpeg", _fake_load)

    out = audio_loader.load_audio_from_path("dummy.flac")

    assert isinstance(out, np.ndarray)


def test_load_audio_from_bytes_validates_params() -> None:
    with pytest.raises(AudioProcessingError):
        audio_loader.load_audio_from_bytes(b"data", target_sr=0)

    with pytest.raises(AudioProcessingError):
        audio_loader.load_audio_from_bytes(b"data", target_channels=0)


def test_load_audio_from_bytes_soundfile_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("soundfile")

    def _read(
        handle: io.BytesIO, dtype: str = "float32"
    ) -> tuple[NDArray[np.float32], int]:
        return np.zeros(4, dtype=np.float32), 16000

    setattr(module, "read", _read)
    monkeypatch.setitem(__import__("sys").modules, "soundfile", module)

    out = audio_loader.load_audio_from_bytes(b"data")

    assert out.dtype == np.float32


def test_load_audio_from_bytes_soundfile_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("soundfile"):
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    def _fake_load(*_: object) -> NDArray[np.float32]:
        return np.zeros(1, dtype=np.float32)

    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.setattr(audio_loader, "_load_with_ffmpeg", _fake_load)

    out = audio_loader.load_audio_from_bytes(b"data")

    assert isinstance(out, np.ndarray)


def test_load_audio_from_bytes_soundfile_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("soundfile")

    def _read(
        handle: io.BytesIO, dtype: str = "float32"
    ) -> tuple[NDArray[np.float32], int]:
        raise RuntimeError("boom")

    setattr(module, "read", _read)
    monkeypatch.setitem(__import__("sys").modules, "soundfile", module)

    def _fake_load(*_: object) -> NDArray[np.float32]:
        return np.zeros(1, dtype=np.float32)

    monkeypatch.setattr(audio_loader, "_load_with_ffmpeg", _fake_load)

    out = audio_loader.load_audio_from_bytes(b"data")

    assert isinstance(out, np.ndarray)


def test_load_audio_from_bytes_soundfile_missing_scipy_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module = types.ModuleType("soundfile")

    def _read(
        handle: io.BytesIO, dtype: str = "float32"
    ) -> tuple[NDArray[np.float32], int]:
        return np.zeros(4, dtype=np.float32), 8000

    setattr(module, "read", _read)
    monkeypatch.setitem(__import__("sys").modules, "soundfile", module)

    real_import = builtins.__import__

    def _import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("scipy"):
            raise ImportError("missing scipy")
        return real_import(name, globals, locals, fromlist, level)

    def _fake_load(*_: object) -> NDArray[np.float32]:
        raise AssertionError("FFmpeg must not be used when the fallback resampler works")

    caplog.set_level("WARNING")
    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.setattr(audio_loader, "_load_with_ffmpeg", _fake_load)

    out = audio_loader.load_audio_from_bytes(b"data", target_sr=16000)

    assert out.shape[0] == 8  # 4 samples at 8 kHz -> 8 at 16 kHz
    assert any("fallback resampler" in record.message for record in caplog.records)


def test_load_with_ffmpeg_empty_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(_: str) -> str:
        return "/usr/bin/ffmpeg"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(["ffmpeg"], 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    with pytest.raises(AudioProcessingError, match="no audio data"):
        audio_loader._load_with_ffmpeg(b"data", 16000, 1)  # pyright: ignore[reportPrivateUsage]


def test_load_with_ffmpeg_zero_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(_: str) -> str:
        return "/usr/bin/ffmpeg"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    class _TruthyBytes(bytes):
        def __bool__(self) -> bool:
            return True

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            ["ffmpeg"], 0, stdout=_TruthyBytes(b""), stderr=b""
        )

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    with pytest.raises(AudioProcessingError, match="decoded audio is empty"):
        audio_loader._load_with_ffmpeg(b"data", 16000, 1)  # pyright: ignore[reportPrivateUsage]


def test_load_with_ffmpeg_defaults_to_mono(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = np.array([0.0, np.nan, np.inf, -0.5], dtype=np.float32)
    stdout = bytearray(audio.tobytes())

    def _which(_: str) -> str:
        return "/usr/bin/ffmpeg"

    def _probe(_: object) -> int | None:
        return None

    monkeypatch.setattr(audio_loader.shutil, "which", _which)
    monkeypatch.setattr(audio_loader, "_probe_channels_with_ffprobe", _probe)

    def _run(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes | bytearray]:
        return subprocess.CompletedProcess(["ffmpeg"], 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    out = audio_loader._load_with_ffmpeg(b"data", 16000, None)  # pyright: ignore[reportPrivateUsage]

    assert out.ndim == 1
    assert np.all(np.isfinite(out))


def test_load_with_ffmpeg_multichannel_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = np.array([0.0, np.nan, np.inf, -0.5, 0.25], dtype=np.float32)
    stdout = bytearray(audio.tobytes())

    def _which(_: str) -> str:
        return "/usr/bin/ffmpeg"

    def _probe(_: object) -> int | None:
        return 2

    monkeypatch.setattr(audio_loader.shutil, "which", _which)
    monkeypatch.setattr(audio_loader, "_probe_channels_with_ffprobe", _probe)

    def _run(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes | bytearray]:
        return subprocess.CompletedProcess(["ffmpeg"], 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    out = audio_loader._load_with_ffmpeg(b"data", 16000, None)  # pyright: ignore[reportPrivateUsage]

    assert out.ndim == 2
    assert out.shape[1] == 2
    assert np.all(np.isfinite(out))


def test_load_with_ffmpeg_multichannel_no_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32)
    stdout = bytearray(audio.tobytes())

    def _which(_: str) -> str:
        return "/usr/bin/ffmpeg"

    def _run(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes | bytearray]:
        return subprocess.CompletedProcess(["ffmpeg"], 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(audio_loader.shutil, "which", _which)
    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    out = audio_loader._load_with_ffmpeg(b"data", 16000, 2)  # pyright: ignore[reportPrivateUsage]

    assert out.shape == (2, 2)


def test_load_with_ffmpeg_multichannel_empty_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = np.array([0.25], dtype=np.float32)
    stdout = audio.tobytes()

    def _which(_: str) -> str:
        return "/usr/bin/ffmpeg"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(["ffmpeg"], 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    with pytest.raises(AudioProcessingError, match="too few samples"):
        audio_loader._load_with_ffmpeg(b"data", 16000, 2)  # pyright: ignore[reportPrivateUsage]


def test_load_with_ffmpeg_called_process_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(_: str) -> str:
        return "/usr/bin/ffmpeg"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    stderr = b"x" * 2001

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, "ffmpeg", output=b"", stderr=stderr)

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    with pytest.raises(AudioProcessingError) as excinfo:
        audio_loader._load_with_ffmpeg(b"data", 16000, 1)  # pyright: ignore[reportPrivateUsage]

    assert "truncated" in str(excinfo.value)


def test_probe_channels_ffprobe_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(_: str) -> None:
        return None

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    assert audio_loader._probe_channels_with_ffprobe(b"data") is None  # pyright: ignore[reportPrivateUsage]


def test_probe_channels_ffprobe_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(_: str) -> str:
        return "/usr/bin/ffprobe"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(["ffprobe"], 0, stdout=b"2\n", stderr=b"")

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    assert audio_loader._probe_channels_with_ffprobe(b"data") == 2  # pyright: ignore[reportPrivateUsage]


def test_probe_channels_ffprobe_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(_: str) -> str:
        return "/usr/bin/ffprobe"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired("ffprobe", 5.0)

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    assert audio_loader._probe_channels_with_ffprobe("/tmp/audio.wav") is None  # pyright: ignore[reportPrivateUsage]


def test_probe_channels_ffprobe_path_non_digit(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(_: str) -> str:
        return "/usr/bin/ffprobe"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(["ffprobe"], 0, stdout=b"abc", stderr=b"")

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    assert audio_loader._probe_channels_with_ffprobe("/tmp/audio.wav") is None  # pyright: ignore[reportPrivateUsage]


def test_probe_channels_ffprobe_called_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _which(_: str) -> str:
        return "/usr/bin/ffprobe"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, "ffprobe", output=b"", stderr=b"")

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    assert audio_loader._probe_channels_with_ffprobe("/tmp/audio.wav") is None  # pyright: ignore[reportPrivateUsage]
