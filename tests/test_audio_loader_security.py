# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Security and native-decode tests for the audio loader.

Covers the bare-str-never-URL / LFI defense (ffmpeg ``-protocol_whitelist`` and
local-file validation), the decompression-bomb size guard (spec R9), and the
native-rate :func:`decode_audio` primitive (spec R7).
"""

from __future__ import annotations

import io
import os
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from standard_asr.exceptions import AudioProcessingError, FFmpegNotFoundError
from standard_asr.utils import audio_loader
from standard_asr.utils.audio_loader import (
    _validate_local_source_path,  # pyright: ignore[reportPrivateUsage]
    decode_audio,
    decode_audio_from_data_uri,
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


def test_decode_audio_wraps_stat_toctou_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # (minor): the file is validated as existing, then a concurrent
    # delete races the stat() size precheck. The bare OSError MUST surface as the
    # documented AudioProcessingError, not leak past the Raises contract.
    f = tmp_path / "racy.wav"
    f.write_bytes(_wav_bytes())
    resolved = str(f.resolve())

    # Validation passes (the file exists at validation time)...
    def _passthrough_validate(_path: str) -> str:
        return resolved

    monkeypatch.setattr(audio_loader, "_validate_local_source_path", _passthrough_validate)

    # ...but the size precheck's stat() then fails (the TOCTOU window).
    real_stat = Path.stat

    def _racy_stat(self: Path, *args: object, **kwargs: object) -> object:
        if str(self) == resolved:
            raise FileNotFoundError("vanished between validation and stat")
        return real_stat(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(audio_loader.pathlib.Path, "stat", _racy_stat)

    with pytest.raises(AudioProcessingError, match="became unreadable before decoding"):
        decode_audio(str(f), max_bytes=10_000)


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


# --- load_audio_from_path rejects existing non-regular files ---


def test_load_audio_from_path_rejects_existing_directory(tmp_path: Path) -> None:
    # An existing path that is not a regular file (here a directory) is rejected
    # up front, never handed to stdlib wave/soundfile (which would block on a
    # FIFO with no timeout -- a hang/DoS vector).
    d = tmp_path / "a_dir.wav"
    d.mkdir()
    with pytest.raises(AudioProcessingError, match="not a regular file"):
        load_audio_from_path(str(d))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo is POSIX-only")
def test_load_audio_from_path_rejects_fifo(tmp_path: Path) -> None:
    # The concrete DoS vector: a FIFO would block wave.open/soundfile forever.
    # It must be rejected at the entry, before any blocking decode.
    fifo = tmp_path / "pipe.wav"
    os.mkfifo(fifo)  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AudioProcessingError, match="not a regular file"):
        load_audio_from_path(str(fifo))


def test_load_audio_from_path_missing_path_still_defers(tmp_path: Path) -> None:
    # boundary: a genuinely MISSING path is NOT rejected by the
    # non-regular-file guard (it does not exist); it falls through to the decode
    # ladder, which surfaces a clear not-found / ffmpeg error -- preserving the
    # convenience-loader deferral semantics the guard must not break.
    missing = tmp_path / "nope.wav"
    with pytest.raises((AudioProcessingError, FFmpegNotFoundError)):
        load_audio_from_path(str(missing))


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


class _ShortReadStream(io.BytesIO):
    """A stream whose ``read(n)`` returns at most ``chunk`` bytes per call.

    Models the IO contract for ``RawIOBase`` / pipes / sockets, where a single
    ``read(n)`` may legally return fewer than ``n`` bytes without being at EOF.
    Subclasses ``io.BytesIO`` (a ``BinaryIO``) and merely throttles each read so
    the capping loop must iterate.
    """

    def __init__(self, data: bytes, chunk: int) -> None:
        super().__init__(data)
        self._chunk = chunk

    def read(self, size: int | None = -1, /) -> bytes:  # type: ignore[override]
        if size is None or size < 0:
            return super().read(size)
        return super().read(min(size, self._chunk))


def test_load_audio_capped_stream_reads_to_eof_despite_short_reads() -> None:
    # A stream that short-reads (returns < requested per call) must
    # still be read in full under the cap -- a single read() would have silently
    # truncated the audio. The 100-sample WAV is ~244 bytes; 32-byte short reads
    # force many iterations, all of which the loop must follow.
    data = _wav_bytes(samples=100)
    stream = _ShortReadStream(data, chunk=32)
    arr = load_audio(stream, max_bytes=len(data) + 1000)
    assert arr.shape == (100,)  # full audio decoded, not a truncated prefix


def test_load_audio_capped_short_read_stream_still_enforces_cap() -> None:
    # The loop preserves the spec R9 memory ceiling -- a short-reading
    # stream that exceeds the cap is still rejected (it never holds more than
    # max_bytes + 1 bytes before aborting).
    data = _wav_bytes(samples=5000)
    stream = _ShortReadStream(data, chunk=16)
    with pytest.raises(AudioProcessingError, match="decode limit"):
        load_audio(stream, max_bytes=64)


# --- native-rate decode (no forced 16k) ---


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
def test_decode_audio_preserves_native_rate(rate: int) -> None:
    arr, sr = decode_audio(_wav_bytes(samples=200, rate=rate))
    assert sr == rate
    assert arr.shape == (200,)


def test_decode_audio_never_sniffs_data_uri_as_base64() -> None:
    # Decode_audio is the engine-input boundary; a bare str is ALWAYS
    # a local file path and MUST NOT be content-sniffed as a data: URI (spec R1 /
    # §3.1). A string literally named like a data: URI is therefore opened as a
    # (non-existent) file and fails "not found", NOT decoded as inline base64.
    # This is the security boundary the conversion layer relies on: an AudioPath
    # whose value happens to read "data:..." must not be decoded as base64.
    import base64 as _b64

    uri = "data:audio/wav;base64," + _b64.b64encode(_wav_bytes(rate=8000)).decode()
    with pytest.raises(AudioProcessingError, match="not found or not a regular file"):
        decode_audio(uri)


def test_decode_audio_does_not_strip_whitespace_from_path() -> None:
    # Removing the data: URI sniff also removed the strip that
    # would silently rewrite a path with surrounding whitespace. A bare str is
    # forwarded verbatim to path validation (a leading-space path is a legitimate
    # if unusual path and is reported as missing, never silently trimmed).
    with pytest.raises(AudioProcessingError, match="not found or not a regular file"):
        decode_audio("  data:audio/wav;base64,AAAA")


def test_decode_audio_from_data_uri_native_rate() -> None:
    # The EXPLICIT data-URI decode entry point preserves the
    # base64/data: decode convenience without any content sniffing.
    import base64 as _b64

    uri = "data:audio/wav;base64," + _b64.b64encode(_wav_bytes(rate=8000)).decode()
    arr, sr = decode_audio_from_data_uri(uri)
    assert sr == 8000
    assert arr.ndim == 1


def test_decode_audio_from_data_uri_mixed_case_scheme() -> None:
    # A mixed-case DATA: scheme is accepted by the case-insensitive parse and
    # decoded, instead of failing with a misleading "Invalid base64 audio
    # payload".
    import base64 as _b64

    uri = "DATA:audio/wav;base64," + _b64.b64encode(_wav_bytes(rate=8000)).decode()
    arr, sr = decode_audio_from_data_uri(uri)
    assert sr == 8000
    assert arr.ndim == 1


def test_decode_audio_from_data_uri_bare_base64() -> None:
    # A bare base64 payload (no data: wrapper) is accepted by the explicit entry
    # point too.
    import base64 as _b64

    payload = _b64.b64encode(_wav_bytes(rate=16000)).decode()
    arr, sr = decode_audio_from_data_uri(payload)
    assert sr == 16000
    assert arr.ndim == 1


def test_decode_audio_from_data_uri_rejects_oversize() -> None:
    # The gate-and-decode size cap (spec R9) is enforced on the explicit entry
    # point: an under-cap estimate never falsely rejects, an over-cap payload is
    # refused before allocation.
    import base64 as _b64

    uri = "data:audio/wav;base64," + _b64.b64encode(_wav_bytes(rate=8000)).decode()
    with pytest.raises(AudioProcessingError, match="exceeding the decode limit"):
        decode_audio_from_data_uri(uri, max_bytes=8)


def test_decode_audio_from_data_uri_rejects_non_str() -> None:
    with pytest.raises(TypeError):
        decode_audio_from_data_uri(b"not-a-str")  # type: ignore[arg-type]


def test_decode_audio_from_data_uri_rejects_bad_channels() -> None:
    uri = "data:audio/wav;base64,AAAA"
    with pytest.raises(AudioProcessingError, match="target_channels must be None or > 0"):
        decode_audio_from_data_uri(uri, target_channels=0)


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


def test_load_with_ffmpeg_missing_path_reports_not_found_even_without_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A missing/mistyped path must surface the clear "not found"
    # error even when ffmpeg is absent -- path validation now runs BEFORE the
    # ffmpeg-presence check, so a beginner who typo'd a filename is not misled
    # into installing an unrelated system dependency.
    def _no_ffmpeg(_name: str) -> str | None:
        return None

    monkeypatch.setattr(audio_loader.shutil, "which", _no_ffmpeg)
    with pytest.raises(AudioProcessingError, match="not found or not a regular file"):
        audio_loader._load_with_ffmpeg(  # pyright: ignore[reportPrivateUsage]
            "/tmp/definitely_missing_xyz.mp3", 16000, 1
        )


def test_load_with_ffmpeg_bytes_still_reports_missing_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The reorder must not regress the genuine FFmpegNotFoundError path: a bytes
    # source (which skips path validation) still reports the missing decoder.
    def _no_ffmpeg(_name: str) -> str | None:
        return None

    monkeypatch.setattr(audio_loader.shutil, "which", _no_ffmpeg)
    with pytest.raises(FFmpegNotFoundError, match="FFmpeg not found"):
        audio_loader._load_with_ffmpeg(b"fake audio", 16000, 1)  # pyright: ignore[reportPrivateUsage]
