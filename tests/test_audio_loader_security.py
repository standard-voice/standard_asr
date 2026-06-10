# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Security and native-decode tests for the audio loader.

Covers the bare-str-never-URL / LFI defense (ffmpeg ``-protocol_whitelist`` and
local-file validation), the decompression-bomb size guard (spec R9), and the
native-rate :func:`decode_audio` primitive (spec R7).
"""

from __future__ import annotations

import io
import struct
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


def _wav_with_inflated_nframes(
    claimed_frames: int,
    real_frames: int = 4,
    channels: int = 1,
    sampwidth: int = 2,
    rate: int = 16000,
) -> bytes:
    """Hand-craft a WAV whose header claims far more frames than really follow.

    ``getnframes()`` is read from the ``data`` chunk size in the header, so a
    crafted inflated chunk size makes ``readframes(getnframes())`` derive a huge
    read size from attacker-controlled metadata (only a few real frames follow).
    """
    block_align = channels * sampwidth
    claimed_data_bytes = claimed_frames * block_align
    real_data = np.zeros(real_frames * channels, dtype="<i2").tobytes()
    riff = b"RIFF" + struct.pack("<I", 36 + claimed_data_bytes) + b"WAVE"
    fmt = b"fmt " + struct.pack(
        "<IHHIIHH", 16, 1, channels, rate, rate * block_align, block_align, sampwidth * 8
    )
    data_hdr = b"data" + struct.pack("<I", claimed_data_bytes)
    return riff + fmt + data_hdr + real_data


# --- ffmpeg/ffprobe LFI/SSRF defense (path validation) ---


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


# --- R9: stdlib WAV header-declared nframes allocation guard ---


def test_load_audio_from_path_rejects_inflated_wav_nframes(tmp_path: Path) -> None:
    # A tiny file whose WAV header claims a billion frames must be rejected by the
    # header guard (the encoded stat precheck passes -- the file really is tiny --
    # so without the guard readframes() would derive a 2 GB read from the header).
    f = tmp_path / "bomb.wav"
    f.write_bytes(_wav_with_inflated_nframes(claimed_frames=1_000_000_000))
    with pytest.raises(AudioProcessingError, match="WAV header declares"):
        load_audio_from_path(str(f), max_bytes=64)


def test_decode_audio_rejects_inflated_wav_nframes(tmp_path: Path) -> None:
    f = tmp_path / "bomb.wav"
    f.write_bytes(_wav_with_inflated_nframes(claimed_frames=1_000_000_000))
    with pytest.raises(AudioProcessingError, match="WAV header declares"):
        decode_audio(f, max_bytes=64)


def test_inflated_wav_nframes_rejected_against_default_ceiling(tmp_path: Path) -> None:
    # With max_bytes=None the guard still bounds against the module default
    # ceiling, so a header claiming > 2 GiB of PCM is rejected even uncapped.
    huge_frames = (audio_loader._DEFAULT_MAX_DECODE_BYTES // 2) + 1  # pyright: ignore[reportPrivateUsage]
    f = tmp_path / "huge.wav"
    f.write_bytes(_wav_with_inflated_nframes(claimed_frames=huge_frames))
    with pytest.raises(AudioProcessingError, match="WAV header declares"):
        load_audio_from_path(str(f), max_bytes=None)


def test_valid_wav_with_honest_header_loads(tmp_path: Path) -> None:
    # A regression guard: an honest header within the cap still decodes fine.
    f = tmp_path / "ok.wav"
    f.write_bytes(_wav_bytes(samples=100))
    arr = load_audio_from_path(str(f), max_bytes=10_000)
    assert arr.shape == (100,)


# --- loader security: convenience loaders thread max_bytes ---


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


# --- max_bytes=None means truly unbounded (no cap), not the default ceiling ---


def test_load_audio_from_bytes_none_disables_cap() -> None:
    # An input that a small positive cap WOULD reject must load when max_bytes is
    # None: None disables the cap entirely rather than mapping to a default.
    data = _wav_bytes(samples=5000)
    # Sanity: this same input is rejected under a tiny explicit cap.
    with pytest.raises(AudioProcessingError, match="decode limit"):
        load_audio_from_bytes(data, max_bytes=10)
    # With None there is no cap, so the >cap input decodes successfully.
    arr = load_audio_from_bytes(data, max_bytes=None)
    assert arr.shape == (5000,)


def test_load_audio_from_path_none_disables_cap(tmp_path: Path) -> None:
    f = tmp_path / "big.wav"
    f.write_bytes(_wav_bytes(samples=5000))
    with pytest.raises(AudioProcessingError, match="decode limit"):
        load_audio_from_path(str(f), max_bytes=10)
    arr = load_audio_from_path(str(f), max_bytes=None)
    assert arr.shape == (5000,)


def test_enforce_decode_size_none_is_unbounded() -> None:
    # The shared guard treats None as no-check even for an enormous notional size.
    audio_loader._enforce_decode_size(10**18, None)  # pyright: ignore[reportPrivateUsage]


# --- BinaryIO stream reads are bounded by max_bytes (no unbounded buffering) ---


class _SpyStream(io.BytesIO):
    """A BytesIO that records every read() request size it received."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.read_sizes: list[int | None] = []

    def read(self, size: int | None = -1, /) -> bytes:  # type: ignore[override]
        self.read_sizes.append(size)
        return super().read(size)


def test_load_audio_stream_exceeding_max_bytes_raises_without_unbounded_read() -> None:
    # A stream larger than the cap must raise, and the loader must never request
    # more than max_bytes + 1 bytes (no read-everything-then-check blow-up).
    stream = _SpyStream(_wav_bytes(samples=5000))
    with pytest.raises(AudioProcessingError, match="decode limit"):
        load_audio(stream, max_bytes=64)
    # The capping read asked for exactly max_bytes + 1 and never an unbounded
    # read(-1)/read(None) over the whole stream.
    assert 65 in stream.read_sizes
    positive = [n for n in stream.read_sizes if n is not None and n >= 0]
    assert max(positive) == 65
    assert -1 not in stream.read_sizes
    assert None not in stream.read_sizes


def test_load_audio_stream_within_cap_loads() -> None:
    data = _wav_bytes(samples=100)
    stream = _SpyStream(data)
    arr = load_audio(stream, max_bytes=len(data) + 1000)
    assert arr.shape == (100,)


def test_load_audio_stream_none_reads_whole_stream() -> None:
    # With None the cap is disabled and the whole stream is read at once.
    data = _wav_bytes(samples=100)
    stream = _SpyStream(data)
    arr = load_audio(stream, max_bytes=None)
    assert arr.shape == (100,)
    assert -1 in stream.read_sizes  # bare read() of the entire stream


# --- native-rate decode (no forced 16k) ---


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


def test_decode_audio_from_mixed_case_data_uri() -> None:
    # End-to-end: a mixed-case DATA: scheme is routed in by the case-insensitive
    # dispatch AND decoded by the (now) case-insensitive base64 decoder, instead
    # of failing with a misleading "Invalid base64 audio payload".
    import base64 as _b64

    uri = "DATA:audio/wav;base64," + _b64.b64encode(_wav_bytes(rate=8000)).decode()
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
