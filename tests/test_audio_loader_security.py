# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Security and native-decode tests for the audio loader.

Covers the bare-str-never-URL / LFI defense (ffmpeg ``-protocol_whitelist`` and
local-file validation), the decompression-bomb size guard (spec R9), and the
native-rate :func:`decode_audio` primitive (spec R7/C4).
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from standard_asr.exceptions import AudioProcessingError
from standard_asr.utils import audio_loader
from standard_asr.utils.audio_loader import (
    _validate_local_source_path,  # pyright: ignore[reportPrivateUsage]
    decode_audio,
    load_audio,
    load_audio_from_bytes,
    load_audio_from_path,
)


def _wav_bytes(samples: int = 100, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(np.zeros(samples, dtype=np.int16).tobytes())
    return buf.getvalue()


# --- C2: ffmpeg/ffprobe LFI/SSRF defense (path validation) ---


@pytest.mark.parametrize(
    "bad",
    [
        "-i",
        "--evil",
        "http://evil.example.com/a.wav",
        "https://evil.example.com/a.wav",
        "concat:a.wav|b.wav",
        "data:audio/wav;base64,AAAA",
        "tcp://127.0.0.1:1234",
        "/definitely/not/a/real/file.wav",
        "",
    ],
)
def test_validate_local_source_path_rejects_unsafe(bad: str) -> None:
    with pytest.raises(AudioProcessingError):
        _validate_local_source_path(bad)


def test_validate_local_source_path_accepts_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "ok.wav"
    f.write_bytes(_wav_bytes())
    resolved = _validate_local_source_path(str(f))
    assert resolved == str(f.resolve())


def test_validate_local_source_path_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(AudioProcessingError):
        _validate_local_source_path(str(tmp_path))


def test_decode_audio_rejects_url_string() -> None:
    # A bare string that looks like a URL is always a (missing) local path.
    with pytest.raises(AudioProcessingError):
        decode_audio("https://evil.example.com/a.wav")


# --- R9: decompression-bomb / size guard ---


def test_decode_audio_size_guard_bytes() -> None:
    with pytest.raises(AudioProcessingError):
        decode_audio(_wav_bytes(samples=5000), max_bytes=10)


def test_decode_audio_size_guard_file(tmp_path: Path) -> None:
    f = tmp_path / "big.wav"
    f.write_bytes(_wav_bytes(samples=5000))
    with pytest.raises(AudioProcessingError):
        decode_audio(f, max_bytes=10)


def test_decode_audio_within_size_limit(tmp_path: Path) -> None:
    f = tmp_path / "ok.wav"
    data = _wav_bytes(samples=100)
    f.write_bytes(data)
    arr, sr = decode_audio(f, max_bytes=len(data) + 1000)
    assert sr == 16000
    assert arr.shape == (100,)


# --- AUDI-2 (loader-security): convenience loaders thread max_bytes ---


def test_load_audio_from_bytes_size_guard() -> None:
    with pytest.raises(AudioProcessingError, match="decode limit"):
        load_audio_from_bytes(_wav_bytes(samples=5000), max_bytes=10)


def test_load_audio_from_path_size_guard(tmp_path: Path) -> None:
    f = tmp_path / "big.wav"
    f.write_bytes(_wav_bytes(samples=5000))
    with pytest.raises(AudioProcessingError, match="decode limit"):
        load_audio_from_path(str(f), max_bytes=10)


def test_load_audio_size_guard_bytes() -> None:
    with pytest.raises(AudioProcessingError, match="decode limit"):
        load_audio(_wav_bytes(samples=5000), max_bytes=10)


def test_load_audio_within_size_limit_bytes() -> None:
    data = _wav_bytes(samples=100)
    arr = load_audio(data, max_bytes=len(data) + 1000)
    assert arr.shape == (100,)


def test_load_audio_from_path_within_limit(tmp_path: Path) -> None:
    f = tmp_path / "ok.wav"
    data = _wav_bytes(samples=100)
    f.write_bytes(data)
    arr = load_audio_from_path(str(f), max_bytes=len(data) + 1000)
    assert arr.shape == (100,)


# --- C4: native-rate decode (no forced 16k) ---


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
def test_decode_audio_preserves_native_rate(rate: int) -> None:
    arr, sr = decode_audio(_wav_bytes(samples=200, rate=rate))
    assert sr == rate
    assert arr.shape == (200,)


def test_decode_audio_from_data_uri_native_rate() -> None:
    import base64 as _b64

    uri = "data:audio/wav;base64," + _b64.b64encode(_wav_bytes(rate=8000)).decode()
    arr, sr = decode_audio(uri)
    assert sr == 8000
    assert arr.ndim == 1


def test_decode_audio_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError):
        decode_audio(12345)  # type: ignore[arg-type]


# --- ffmpeg command construction includes the protocol whitelist ---


def test_ffmpeg_command_has_protocol_whitelist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "a.wav"
    f.write_bytes(_wav_bytes())
    captured: dict[str, list[str]] = {}

    class _Proc:
        stdout = np.ones(100, dtype=np.float32).tobytes()
        stderr = b""

    def _fake_which(name: str) -> str | None:
        return "/usr/bin/" + name

    def _fake_run(cmd: list[str], **_kw: object) -> _Proc:
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(audio_loader.shutil, "which", _fake_which)
    monkeypatch.setattr(audio_loader.subprocess, "run", _fake_run)
    audio_loader._load_with_ffmpeg(str(f), 16000, 1)  # pyright: ignore[reportPrivateUsage]
    cmd = captured["cmd"]
    assert "-protocol_whitelist" in cmd
    wl = cmd[cmd.index("-protocol_whitelist") + 1]
    assert wl == "file,pipe"
    # The input arg is the resolved absolute path, never a raw URL/option.
    input_arg = cmd[cmd.index("-i") + 1]
    assert input_arg == str(f.resolve())
