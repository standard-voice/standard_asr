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


def _ffmpeg_native_returning(array: NDArray[np.float32], rate: int) -> "object":
    """Build a typed stand-in for ``_decode_with_ffmpeg_native``."""

    def _decode(
        source: str | bytes, target_channels: int | None
    ) -> tuple[NDArray[np.float32], int]:
        return array, rate

    return _decode


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


def test_normalize_audio_scales_int16_pcm() -> None:
    # int16 PCM is raw codes, not amplitudes: full-scale must map to ~+-1, not
    # clip to 1.0 across the board. -32768 maps to exactly -1.0; 32767 ~ 0.99997.
    audio = np.array([32767, -32768, 16384], dtype=np.int16)

    out = audio_loader.normalize_audio(audio, 16000, 16000, 1)

    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [32767 / 32768, -1.0, 0.5], atol=1e-6)


def test_normalize_audio_scales_int32_pcm() -> None:
    audio = np.array([2**31 - 1, -(2**31), 2**30], dtype=np.int32)

    out = audio_loader.normalize_audio(audio, 16000, 16000, 1)

    np.testing.assert_allclose(out, [(2**31 - 1) / 2**31, -1.0, 0.5], atol=1e-6)


def test_normalize_audio_scales_uint8_pcm() -> None:
    # uint8 PCM centers at 128: 128 -> 0.0, 255 -> ~+1, 0 -> -1, 192 -> 0.5.
    audio = np.array([128, 255, 0, 192], dtype=np.uint8)

    out = audio_loader.normalize_audio(audio, 16000, 16000, 1)

    np.testing.assert_allclose(out, [0.0, 127 / 128, -1.0, 0.5], atol=1e-6)


def test_normalize_audio_float_unchanged_modulo_clip() -> None:
    # Floating input is treated as already-normalized amplitude: in-range values
    # are preserved exactly, only out-of-range values are clipped for safety.
    audio = np.array([0.25, -0.5, 2.0, -3.0], dtype=np.float32)

    out = audio_loader.normalize_audio(audio, 16000, 16000, 1)

    np.testing.assert_allclose(out, [0.25, -0.5, 1.0, -1.0], atol=1e-7)


def test_normalize_audio_scales_int_stereo() -> None:
    # 2D integer PCM scales per-sample before any channel handling; preserving
    # both channels keeps the scaled amplitudes.
    audio = np.array([[16384, -16384], [32767, -32768]], dtype=np.int16)

    out = audio_loader.normalize_audio(audio, 16000, 16000, None)

    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, [[0.5, -0.5], [32767 / 32768, -1.0]], atol=1e-6)


def test_normalize_audio_exotic_dtype_plain_cast() -> None:
    # A non-integer, non-floating dtype (bool) is neither PCM nor an amplitude:
    # it falls through to a plain float cast (clipped by the contract), rather
    # than failing the rare caller.
    audio = np.array([True, False, True], dtype=bool)

    out = audio_loader.normalize_audio(audio, 16000, 16000, 1)

    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [1.0, 0.0, 1.0], atol=1e-7)


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


def test_load_audio_existing_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "exists.wav"
    path.write_bytes(b"placeholder")
    sentinel: NDArray[np.float32] = np.zeros(1, dtype=np.float32)

    def _load_audio_from_path(
        path_str: str,
        target_sr: int = 16000,
        target_channels: int | None = 1,
        *,
        max_bytes: int | None = None,
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
        path_str: str,
        target_sr: int = 16000,
        target_channels: int | None = 1,
        *,
        max_bytes: int | None = None,
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
        path_str: str,
        target_sr: int = 16000,
        target_channels: int | None = 1,
        *,
        max_bytes: int | None = None,
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
        data: bytes,
        target_sr: int = 16000,
        target_channels: int | None = 1,
        *,
        max_bytes: int | None = None,
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

    def _read(handle: io.BytesIO, dtype: str = "float32") -> tuple[NDArray[np.float32], int]:
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

    def _read(handle: io.BytesIO, dtype: str = "float32") -> tuple[NDArray[np.float32], int]:
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

    def _read(handle: io.BytesIO, dtype: str = "float32") -> tuple[NDArray[np.float32], int]:
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


def test_load_with_ffmpeg_rejects_oversized_output(monkeypatch: pytest.MonkeyPatch) -> None:
    # Spec R9: a crafted long-duration input could emit far more PCM than its
    # encoded size implies. If ffmpeg's stdout exceeds the output ceiling, it is
    # rejected (defense in depth) rather than buffered into a multi-GB array.
    def _which(_: str) -> str:
        return "/usr/bin/ffmpeg"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    oversized = np.zeros(64, dtype=np.float32).tobytes()  # 256 bytes > the cap

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(["ffmpeg"], 0, stdout=oversized, stderr=b"")

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    with pytest.raises(AudioProcessingError, match="exceeds the .* ceiling"):
        audio_loader._load_with_ffmpeg(  # pyright: ignore[reportPrivateUsage]
            b"data", 16000, 1, max_output_bytes=16
        )


def test_load_with_ffmpeg_passes_fs_limit_to_command(monkeypatch: pytest.MonkeyPatch) -> None:
    # The decode command must carry ``-fs <ceiling>`` so ffmpeg self-limits its
    # output (capture_output cannot then buffer past the ceiling).
    captured: dict[str, list[str]] = {}

    def _which(_: str) -> str:
        return "/usr/bin/ffmpeg"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout=np.zeros(4, np.float32).tobytes(), stderr=b""
        )

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    audio_loader._load_with_ffmpeg(b"data", 16000, 1, max_output_bytes=4096)  # pyright: ignore[reportPrivateUsage]
    cmd = captured["cmd"]
    assert "-fs" in cmd
    assert cmd[cmd.index("-fs") + 1] == "4096"
    assert cmd[-1] == "-"  # the pipe-output arg stays last


def test_load_with_ffmpeg_none_output_bound_omits_fs(monkeypatch: pytest.MonkeyPatch) -> None:
    # max_output_bytes=None disables the ceiling: no ``-fs`` flag, no rejection
    # even for a large stdout.
    captured: dict[str, list[str]] = {}

    def _which(_: str) -> str:
        return "/usr/bin/ffmpeg"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    big = np.zeros(1024, dtype=np.float32).tobytes()

    def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=big, stderr=b"")

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    out = audio_loader._load_with_ffmpeg(  # pyright: ignore[reportPrivateUsage]
        b"data", 16000, 1, max_output_bytes=None
    )
    assert out.shape[0] == 1024
    assert "-fs" not in captured["cmd"]


def test_load_with_ffmpeg_zero_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(_: str) -> str:
        return "/usr/bin/ffmpeg"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    class _TruthyBytes(bytes):
        def __bool__(self) -> bool:
            return True

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(["ffmpeg"], 0, stdout=_TruthyBytes(b""), stderr=b"")

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

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes | bytearray]:
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

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes | bytearray]:
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

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes | bytearray]:
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


def test_load_with_ffmpeg_called_process_error_short_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A short stderr (< 2000 bytes) must NOT be tagged as truncated (the
    # truncation guard's False branch).
    def _which(_: str) -> str:
        return "/usr/bin/ffmpeg"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, "ffmpeg", output=b"", stderr=b"short error")

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)

    with pytest.raises(AudioProcessingError) as excinfo:
        audio_loader._load_with_ffmpeg(b"data", 16000, 1)  # pyright: ignore[reportPrivateUsage]
    assert "truncated" not in str(excinfo.value)
    assert "short error" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# decode_audio: native-rate decoding (no resampling) across source variants.
# --------------------------------------------------------------------------- #
def test_decode_audio_rejects_nonpositive_channels() -> None:
    with pytest.raises(AudioProcessingError, match="target_channels"):
        audio_loader.decode_audio(b"data", target_channels=0)


def test_decode_audio_bad_base64_data_uri() -> None:
    with pytest.raises(AudioProcessingError, match="Invalid base64 audio payload"):
        audio_loader.decode_audio("data:audio/wav;base64,!!!notb64!!!")


def test_shared_base64_decoder_accepts_data_uri_and_bare() -> None:
    # AUDI-4: one shared decoder for both entry points. A base64 data URI and the
    # equivalent bare base64 string decode to the same bytes.
    import base64 as _b64

    raw = b"hello-audio"
    encoded = _b64.b64encode(raw).decode()
    assert audio_loader.decode_base64_audio(f"data:audio/wav;base64,{encoded}") == raw
    assert audio_loader.decode_base64_audio(encoded) == raw


def test_shared_base64_decoder_rejects_data_uri_without_base64_marker() -> None:
    # AUDI-4: a data: URI without the ';base64,' marker is rejected, not silently
    # treated as base64 (the old conversion._decode_b64 split on ',' and accepted
    # percent-encoded data URIs). Both entry points now share this strict rule.
    with pytest.raises(AudioProcessingError, match="';base64,' marker is required"):
        audio_loader.decode_base64_audio("data:audio/wav,not-base64-payload")


@pytest.mark.parametrize("scheme", ["data:", "DATA:", "Data:", "dAtA:"])
def test_shared_base64_decoder_scheme_is_case_insensitive(scheme: str) -> None:
    # The data: scheme is detected case-insensitively to match the case-
    # insensitive dispatch in load_audio/decode_audio. A mixed/upper-case DATA:
    # URI must decode its payload, not be mis-parsed as a raw base64 string (the
    # old case-sensitive prefix produced a misleading "Invalid base64 payload").
    import base64 as _b64

    raw = b"hello-audio"
    encoded = _b64.b64encode(raw).decode()
    assert audio_loader.decode_base64_audio(f"{scheme}audio/wav;base64,{encoded}") == raw


def test_decode_path_native_wav_8bit_mono(tmp_path: Path) -> None:
    path = tmp_path / "a8.wav"
    _write_wav(path, sampwidth=1, channels=1)
    arr, sr = audio_loader.decode_audio(str(path), target_channels=1)
    assert arr.dtype == np.float32
    assert sr == 16000


def test_decode_path_native_wav_16bit_stereo(tmp_path: Path) -> None:
    path = tmp_path / "a16.wav"
    _write_wav(path, sampwidth=2, channels=2)
    arr, sr = audio_loader.decode_audio(str(path), target_channels=None)
    assert arr.ndim == 2
    assert arr.shape[1] == 2
    assert sr == 16000


def test_decode_path_native_wav_unsupported_sampwidth_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 24-bit WAV is unsupported by the stdlib reader; decode falls through to
    # soundfile/ffmpeg. Here soundfile is absent and ffmpeg is stubbed.
    class _FakeWave:
        def __enter__(self) -> "_FakeWave":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def getframerate(self) -> int:
            return 16000

        def getsampwidth(self) -> int:
            return 3  # 24-bit -> unsupported via stdlib

        def getnchannels(self) -> int:
            return 1

        def getnframes(self) -> int:
            return 0

        def readframes(self, _: int) -> bytes:
            return b""

    real_import = builtins.__import__

    def _import(name: str, *a: object, **k: object) -> object:
        if name.startswith("soundfile"):
            raise ImportError("no soundfile")
        return real_import(name, *a, **k)  # type: ignore[arg-type]

    def _open(*_a: object, **_k: object) -> _FakeWave:
        return _FakeWave()

    monkeypatch.setattr(wave, "open", _open)
    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.setattr(
        audio_loader,
        "_decode_with_ffmpeg_native",
        _ffmpeg_native_returning(np.zeros(4, dtype=np.float32), 16000),
    )
    # Call the decode helper directly: decode_audio would reject the synthetic
    # path before reaching the WAV reader.
    arr, sr = audio_loader._decode_path_native("/tmp/u.wav", 1)  # pyright: ignore[reportPrivateUsage]
    assert arr.shape[0] == 4
    assert sr == 16000


def test_decode_path_native_soundfile_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # A non-WAV path decodes via soundfile, preserving the native rate.
    module = types.ModuleType("soundfile")

    def _read(path: str, dtype: str = "float32") -> tuple[NDArray[np.float32], int]:
        return np.zeros(10, dtype=np.float32), 22050

    setattr(module, "read", _read)
    monkeypatch.setitem(__import__("sys").modules, "soundfile", module)
    arr, sr = audio_loader._decode_path_native("/tmp/a.flac", 1)  # pyright: ignore[reportPrivateUsage]
    assert sr == 22050
    assert arr.shape[0] == 10


def test_decode_path_native_soundfile_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("soundfile")

    def _read(path: str, dtype: str = "float32") -> tuple[NDArray[np.float32], int]:
        raise RuntimeError("decode boom")

    setattr(module, "read", _read)
    monkeypatch.setitem(__import__("sys").modules, "soundfile", module)
    monkeypatch.setattr(
        audio_loader,
        "_decode_with_ffmpeg_native",
        _ffmpeg_native_returning(np.zeros(3, dtype=np.float32), 8000),
    )
    arr, sr = audio_loader._decode_path_native("/tmp/a.flac", 1)  # pyright: ignore[reportPrivateUsage]
    assert sr == 8000
    assert arr.shape[0] == 3


def test_decode_bytes_native_soundfile_success(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("soundfile")

    def _read(buf: object, dtype: str = "float32") -> tuple[NDArray[np.float32], int]:
        return np.zeros(6, dtype=np.float32), 44100

    setattr(module, "read", _read)
    monkeypatch.setitem(__import__("sys").modules, "soundfile", module)
    arr, sr = audio_loader.decode_audio(b"rawbytes", target_channels=1)
    assert sr == 44100
    assert arr.shape[0] == 6


def test_decode_bytes_native_soundfile_import_error_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _import(name: str, *a: object, **k: object) -> object:
        if name.startswith("soundfile"):
            raise ImportError("no soundfile")
        return real_import(name, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.setattr(
        audio_loader,
        "_decode_with_ffmpeg_native",
        _ffmpeg_native_returning(np.zeros(2, dtype=np.float32), 16000),
    )
    arr, sr = audio_loader.decode_audio(b"rawbytes", target_channels=1)
    assert sr == 16000
    assert arr.shape[0] == 2


def test_decode_bytes_native_soundfile_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("soundfile")

    def _read(buf: object, dtype: str = "float32") -> tuple[NDArray[np.float32], int]:
        raise RuntimeError("bytes decode boom")

    setattr(module, "read", _read)
    monkeypatch.setitem(__import__("sys").modules, "soundfile", module)
    monkeypatch.setattr(
        audio_loader,
        "_decode_with_ffmpeg_native",
        _ffmpeg_native_returning(np.zeros(7, dtype=np.float32), 16000),
    )
    arr, sr = audio_loader.decode_audio(b"rawbytes", target_channels=1)
    assert arr.shape[0] == 7
    assert sr == 16000


def test_decode_with_ffmpeg_native_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # The native ffmpeg decode probes the rate then decodes at that rate.
    def _probe(_source: str | bytes) -> int | None:
        return 32000

    def _load(_source: str | bytes, _sr: int, _ch: int | None) -> NDArray[np.float32]:
        return np.zeros(5, dtype=np.float32)

    monkeypatch.setattr(audio_loader, "_probe_sample_rate_with_ffprobe", _probe)
    monkeypatch.setattr(audio_loader, "_load_with_ffmpeg", _load)
    arr, sr = audio_loader._decode_with_ffmpeg_native(b"data", 1)  # pyright: ignore[reportPrivateUsage]
    assert sr == 32000
    assert arr.shape[0] == 5


def test_decode_with_ffmpeg_native_no_rate_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _probe(_source: str | bytes) -> int | None:
        return None

    monkeypatch.setattr(audio_loader, "_probe_sample_rate_with_ffprobe", _probe)
    with pytest.raises(AudioProcessingError, match="native sample rate"):
        audio_loader._decode_with_ffmpeg_native(b"data", 1)  # pyright: ignore[reportPrivateUsage]


def test_probe_sample_rate_delegates_to_stream_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(_: str) -> str:
        return "/usr/bin/ffprobe"

    monkeypatch.setattr(audio_loader.shutil, "which", _which)

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(["ffprobe"], 0, stdout=b"48000\n", stderr=b"")

    monkeypatch.setattr(audio_loader.subprocess, "run", _run)
    assert audio_loader._probe_sample_rate_with_ffprobe(b"data") == 48000  # pyright: ignore[reportPrivateUsage]


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


# --------------------------------------------------------------------------- #
# R3-AUDIO-LOADER-01 -- the soundfile paths enforce the decoded-output ceiling
# (a hard rejection that must NOT fall back to the FFmpeg decoder).
# --------------------------------------------------------------------------- #
def _install_big_soundfile_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a soundfile stub whose decode exceeds a tiny ceiling."""
    module = types.ModuleType("soundfile")

    def _read(source: object, dtype: str = "float32") -> tuple[NDArray[np.float32], int]:
        return np.zeros(64, dtype=np.float32), 16000

    setattr(module, "read", _read)
    monkeypatch.setitem(__import__("sys").modules, "soundfile", module)
    monkeypatch.setattr(audio_loader, "_DEFAULT_MAX_DECODE_BYTES", 8)


def test_load_audio_from_path_decoded_ceiling_rejects_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_big_soundfile_stub(monkeypatch)
    with pytest.raises(AudioProcessingError, match="decoded-output ceiling"):
        audio_loader.load_audio_from_path("dummy.flac")


def test_load_audio_from_bytes_decoded_ceiling_rejects_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_big_soundfile_stub(monkeypatch)
    with pytest.raises(AudioProcessingError, match="decoded-output ceiling"):
        audio_loader.load_audio_from_bytes(b"fLaC not really")


def test_decode_path_native_decoded_ceiling_rejects_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_big_soundfile_stub(monkeypatch)
    with pytest.raises(AudioProcessingError, match="decoded-output ceiling"):
        audio_loader._decode_path_native("dummy.flac", 1)  # pyright: ignore[reportPrivateUsage]


def test_decode_bytes_native_decoded_ceiling_rejects_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_big_soundfile_stub(monkeypatch)
    with pytest.raises(AudioProcessingError, match="decoded-output ceiling"):
        audio_loader._decode_bytes_native(b"fLaC not really", 1)  # pyright: ignore[reportPrivateUsage]
