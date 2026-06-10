# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

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


"""
Audio loading and normalization utilities for the Standard ASR audio contract.

This module focuses on turning a wide range of audio inputs into a normalized
NumPy array suitable for ASR models. It deliberately prefers light-weight,
standard-library options first, then optional libraries, and finally a
system-level FFmpeg fallback for maximum compatibility.

Key points and contract:
- Output dtype is always `np.float32`.
- Output value range is clipped to [-1.0, 1.0].
- Output shape is `(n_samples,)` for mono, `(n_samples, n_channels)` for multi.
- Resampling prefers ``scipy.signal.resample_poly`` when scipy is installed
  (the ``[audio]`` extra); when it is missing, it degrades to the built-in
  numpy-only anti-aliasing Fourier resampler (a missing extra is never fatal,
  spec AI R8) -- it does NOT fall back to FFmpeg for resampling.
- Channel handling:
  - Downmix to mono uses arithmetic mean.
  - Upmix replicates channels (e.g., 1 -> 2).
  - Multi->fewer channels down-mix by truncation (first N channels) unless
    loading is delegated to FFmpeg, which can provide higher-quality mixing.
- NaN/Inf are sanitized to safe values (NaN->0.0, +Inf->1.0, -Inf->-1.0).

Dependencies and fallbacks:
1) WAV via stdlib `wave` (8/16-bit PCM)
2) `soundfile` (if installed) for many formats and in-memory bytes
3) FFmpeg subprocess fallback (requires ffmpeg in PATH)

Decoding and resampling are independent fallbacks. A missing `scipy` only
affects resampling quality -- the built-in numpy anti-aliasing fallback keeps
resampling working (spec AI R8), so the loader never falls back to FFmpeg merely
because `scipy` is absent. FFmpeg is reached only when neither stdlib `wave` nor
`soundfile` can decode the container.

All functions emit clear exceptions with actionable messages and log helpful
warnings when quality-affecting fallbacks are used.
"""

import base64
import io
import logging
import math
import pathlib
import shutil
import subprocess
import wave
from typing import Any, BinaryIO, Literal, TypeGuard, cast, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray

from ..exceptions import (
    AudioProcessingError,
    FFmpegNotFoundError,
)
from ..wire import pcm16_decode

logger = logging.getLogger(__name__)

#: Hard ceiling (bytes) on a single buffered payload when no engine limit is
#: supplied (spec R9). 2 GiB comfortably covers multi-hour PCM while bounding
#: memory. Serves two distinct bounds: the default cap on a buffered **encoded**
#: input (see :func:`_enforce_decode_size`), and the dedicated **decoded-output**
#: ceiling applied by every decode backend (the stdlib WAV header guard, the
#: soundfile ceiling in :func:`_enforce_decoded_output_ceiling`, and the FFmpeg
#: ``-fs`` limit in :func:`_load_with_ffmpeg`).
_DEFAULT_MAX_DECODE_BYTES = 2 * 1024 * 1024 * 1024

#: FFmpeg writes its output in whole blocks of this size. ``-fs`` therefore
#: stops at the first block boundary at or past the limit, and a limit that is
#: itself block-aligned can be hit EXACTLY by a truncated stream. One block of
#: headroom on top of the caller's ceiling keeps "truncated" strictly
#: distinguishable from "legal and exactly at the ceiling".
_FFMPEG_FS_BLOCK = 4096

# --- Public API ---


def decode_base64_audio(value: str) -> bytes:
    """Decode a base64 audio payload, optionally wrapped in a ``data:`` URI.

    Single source of truth for the ``data:``-URI/base64 parse rules, shared by
    the convenience loaders, :func:`decode_audio`, and the conversion layer so
    every entry point accepts and rejects exactly the same forms.

    Rules:

    * A ``data:`` URI MUST carry the explicit ``;base64,`` marker; the bytes
      after it are base64-decoded. A ``data:`` URI without ``;base64,`` (e.g. a
      percent-encoded ``data:audio/wav,...``) is rejected rather than silently
      treated as base64.
    * Any other string is treated as a bare base64 payload.

    The ``data:`` scheme is detected case-insensitively (``data:``, ``DATA:``,
    ``Data:`` ...) so this matches the case-insensitive dispatch in
    :func:`load_audio` / :func:`decode_audio`; an upper/mixed-case ``DATA:`` URI
    decodes correctly instead of being mis-parsed as raw base64.

    Validation is strict (``validate=True``): non-base64 characters fail loudly
    rather than being silently dropped.

    Args:
        value: A ``data:...;base64,...`` URI (any case for the scheme) or a bare
            base64 string.

    Returns:
        The decoded bytes.

    Raises:
        AudioProcessingError: If a ``data:`` URI lacks the ``;base64,`` marker,
            or the payload is not valid base64.
    """
    return _decode_base64_payload(_base64_payload(value))


def _decode_base64_payload(payload: str) -> bytes:
    """Strictly decode an already-extracted base64 payload.

    The payload-level half of :func:`decode_base64_audio`, split out so the
    gate-and-decode helpers can extract a ``data:`` URI's payload ONCE and
    share it between the size estimate and the decode.

    Args:
        payload: The raw base64 payload (no ``data:`` wrapper).

    Returns:
        The decoded bytes.

    Raises:
        AudioProcessingError: If the payload is not valid base64.
    """
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise AudioProcessingError("Invalid base64 audio payload.") from exc


def _base64_payload(value: str) -> str:
    """Extract the raw base64 payload from a possible ``data:`` URI wrapper.

    Shared by :func:`decode_base64_audio` and the same-named gate-and-decode
    helper ``_decode_base64_bounded`` in both this module and the conversion
    layer (they bind different size contracts -- R9 decode cap vs R4 engine
    ``max_file_size`` -- but parse a ``data:`` URI identically) so every entry
    point applies exactly the same ``data:``-URI parse rules (single source of
    truth).

    Args:
        value: A ``data:...;base64,...`` URI (any case for the scheme) or a bare
            base64 string.

    Returns:
        The base64 payload portion of ``value``.

    Raises:
        AudioProcessingError: If a ``data:`` URI lacks the ``;base64,`` marker.
    """
    if value[:5].lower() == "data:":
        marker = ";base64,"
        if marker not in value:
            raise AudioProcessingError(
                "Malformed data: URI -- only base64-encoded data URIs are "
                "supported; the ';base64,' marker is required."
            )
        return value.split(marker, 1)[1]
    return value


def _estimate_payload_decoded_size(payload: str) -> int:
    """Estimate the decoded byte size of a base64 payload without decoding it.

    Computed from the payload length alone (every 4 base64 characters decode to
    3 bytes, minus trailing padding), so a size gate can run *before* the
    allocation that :func:`_decode_base64_payload` performs. The estimate is
    exact for a valid padded payload and a floor (underestimate) for a
    malformed one, so a gate using it never rejects a payload whose decoded
    form would actually fit -- the exact post-decode length check stays
    authoritative. Takes the already-extracted payload (no ``data:`` wrapper)
    so the gate-and-decode helpers extract a ``data:`` URI's payload ONCE and
    share it between the estimate and the decode (extraction is an O(n) scan
    plus, for a data: URI, a payload-sized transient slice copy).

    Args:
        payload: The raw base64 payload (no ``data:`` wrapper).

    Returns:
        The estimated decoded size in bytes (never negative).
    """
    padding = 2 if payload.endswith("==") else 1 if payload.endswith("=") else 0
    return max((len(payload) // 4) * 3 - padding, 0)


def _decode_base64_bounded(value: str, max_bytes: int | None) -> bytes:
    """Size-gate (pre-decode, spec R9) and decode a base64 payload in one step.

    The gate and the decode travel together so a base64-accepting loader path
    cannot take the decode without the gate: the decoded size is estimated
    from the payload length BEFORE the decode allocates it (the estimate never
    exceeds the true size, so an under-limit payload is never falsely
    rejected), and the exact decoded length is re-checked after. The
    ``data:``-URI payload is extracted once and shared by both steps.

    Args:
        value: A ``data:...;base64,...`` URI or a bare base64 string.
        max_bytes: The encoded-payload cap, or ``None`` for unbounded.

    Returns:
        The decoded bytes (within ``max_bytes``).

    Raises:
        AudioProcessingError: If the estimated or exact decoded size exceeds
            ``max_bytes``, the payload is not valid base64, or a ``data:`` URI
            lacks the ``;base64,`` marker.
    """
    payload = _base64_payload(value)
    _enforce_decode_size(_estimate_payload_decoded_size(payload), max_bytes)
    decoded = _decode_base64_payload(payload)
    _enforce_decode_size(len(decoded), max_bytes)
    return decoded


def _validate_local_source_path(path: str) -> str:
    """Validate and absolutize a local file path before handing it to ffmpeg.

    This is the file-input half of the bare-str-never-URL defense (design
    decision D1): ffmpeg must only ever open a real local file, never a network
    URL (``http://``, ``tcp://``), a protocol-chaining input (``concat:``,
    ``data:``) or an option-injection string (a leading ``-``).

    Args:
        path: The candidate local file path.

    Returns:
        The resolved absolute path as a string.

    Raises:
        AudioProcessingError: If the path is empty, looks like a CLI option, or
            does not resolve to an existing regular file.
    """
    if not path or path.startswith("-"):
        raise AudioProcessingError(
            "Refusing to decode an audio source that begins with '-' "
            "(possible option injection). Pass a real local file path."
        )
    resolved = pathlib.Path(path).expanduser()
    try:
        is_file = resolved.is_file()
    except OSError as exc:
        # A bare str is always a local path (never sniffed -- spec R1), so a
        # long ``data:``/base64 string reaches here and ``is_file()`` raises
        # ENAMETOOLONG (or another path OSError). Re-raise as the module's
        # contract error so it does not escape decode_audio()/execute_plan()'s
        # documented Raises and get mis-mapped to a server 500 (same
        # OSError->AudioProcessingError discipline as _enforce_path_decode_size).
        raise AudioProcessingError(
            f"Audio file not found or not a regular file: {path!r} ({exc}). "
            "Bare strings are always treated as local file paths; wrap URLs in "
            "AudioUrl and base64/data: URIs in AudioBase64 (or use "
            "decode_audio_from_data_uri)."
        ) from exc
    if not is_file:
        raise AudioProcessingError(
            f"Audio file not found or not a regular file: {path!r}. "
            "Bare strings are always treated as local file paths; wrap URLs in "
            "AudioUrl and base64 in AudioBase64."
        )
    return str(resolved.resolve())


def _enforce_path_decode_size(path: str, max_bytes: int | None) -> None:
    """Pre-check a validated file's ENCODED size, surviving the validate→stat race.

    The path was confirmed to exist by :func:`_validate_local_source_path`, but a
    concurrent delete can race between that check and this ``stat()`` (a TOCTOU
    window that is real under server concurrency). A bare ``OSError`` here would
    escape past every public decoder's documented ``AudioProcessingError``
    contract; re-raise it as that contract error so callers' error handling holds.

    Args:
        path: A validated local file path (already absolutized).
        max_bytes: The encoded-size cap, or ``None`` for unbounded.

    Raises:
        AudioProcessingError: If the file exceeds ``max_bytes`` (spec R9) or
            became unreadable between validation and ``stat``.
    """
    try:
        size = pathlib.Path(path).stat().st_size
    except OSError as exc:
        raise AudioProcessingError(
            f"Audio file became unreadable before decoding: {path!r} ({exc})."
        ) from exc
    _enforce_decode_size(size, max_bytes)


def _enforce_decode_size(num_bytes: int, max_bytes: int | None) -> None:
    """Raise if a buffered **encoded** payload exceeds the size cap (spec R9).

    Honesty note: this bounds only the size of the ENCODED input (a file's
    ``st_size`` or ``len(bytes)``) before it is buffered. It does NOT bound the
    decoded float32 array, whose size is ``duration x sample_rate x 4`` and can
    be many times larger for a compressed codec. The decoded output is bounded
    separately, per decode backend: the stdlib WAV path probes the
    header-declared frame count (:func:`_guard_wav_nframes`), the soundfile path
    checks the decoded array against the module ceiling
    (:func:`_enforce_decoded_output_ceiling`), and the FFmpeg path self-limits
    via ``-fs`` plus a post-capture check (:func:`_load_with_ffmpeg`).

    Args:
        num_bytes: Size of the encoded payload about to be buffered, in bytes.
        max_bytes: The cap in bytes, or ``None`` for **truly unbounded** (no
            check). Callers that want the default 2 GiB ceiling pass it
            explicitly; only ``None`` disables the cap entirely.

    Raises:
        AudioProcessingError: If ``num_bytes`` exceeds ``max_bytes``.
    """
    if max_bytes is None:
        return
    if num_bytes > max_bytes:
        raise AudioProcessingError(
            f"Audio payload is {num_bytes} bytes, exceeding the decode limit of "
            f"{max_bytes} bytes. Provide a smaller input or an engine without this "
            "limit."
        )


class _WavAllocationGuardError(AudioProcessingError):
    """A WAV header declared a frame count that would over-allocate (spec R9).

    A dedicated subclass so the stdlib WAV decode paths can re-raise it past
    their broad ``except AudioProcessingError`` fallback: a header-declared
    decompression bomb is a hard rejection, not a "try the next decoder" signal
    (every decoder would parse the same hostile header). It remains an
    :class:`AudioProcessingError`, so public callers handle it uniformly.
    """


class _DecodedOutputCeilingError(AudioProcessingError):
    """A decoded waveform exceeded the module's decoded-output ceiling (spec R9).

    A dedicated subclass so the soundfile decode paths can re-raise it past
    their broad ``except Exception`` fallback: a decoded output past the ceiling
    is a hard rejection, not a "try the next decoder" signal (the FFmpeg
    fallback would just re-decode the same expansive input). It remains an
    :class:`AudioProcessingError`, so public callers handle it uniformly.
    """


def _read_with_soundfile(source: str | io.BytesIO) -> tuple[NDArray[np.float32], int] | None:
    """Decode with ``soundfile``, or return ``None`` when fallback should run.

    The single ``soundfile`` attempt shared by every decode path, so the
    exception-ladder discipline lives in exactly one place:

    * a missing ``soundfile`` (ImportError) or an ordinary decode failure
      returns ``None`` -- the caller falls through to the FFmpeg layer;
    * a decoded output past the module ceiling re-raises
      (:class:`_DecodedOutputCeilingError`): the FFmpeg fallback would just
      re-decode the same expansive input, so a copied ladder that forgot this
      re-raise would route a decompression bomb into a SECOND full decode.

    Args:
        source: A local file path or a ``BytesIO`` over encoded bytes.

    Returns:
        The decoded ``(float32 waveform, native sample rate)`` pair, or
        ``None`` when the caller should fall back to FFmpeg.

    Raises:
        _DecodedOutputCeilingError: If the header-declared OR actually decoded
            output exceeds the decoded-output ceiling (hard rejection, never
            falls back).
    """
    try:
        import soundfile as sf  # pyright: ignore[reportMissingTypeStubs]

        # Bound the allocation BEFORE it happens (spec R9): soundfile.read()
        # pre-allocates np.empty(header_frames x channels) off the file *header*,
        # so a header that declares a very long duration would drive a multi-GB
        # allocation before the post-decode ceiling below could ever run. Probe
        # the header with sf.info() (no decode) and reject up front -- the
        # equivalent of the WAV path's _guard_wav_nframes and the FFmpeg path's
        # ``-fs`` self-limit. The post-decode check is kept as defense in depth
        # for formats whose header under-reports (info frames <= 0 / unknown).
        # The probe rewinds a BytesIO source before returning, so sf.read() below
        # sees the stream from the start.
        _enforce_soundfile_info_ceiling(sf, source)

        sf_read: Any = getattr(sf, "read")
        audio, orig_sr = cast(tuple[NDArray[np.float32], int], sf_read(source, dtype="float32"))
        _enforce_decoded_output_ceiling(audio)
    except ImportError:
        logger.debug("`soundfile` not installed; falling back to FFmpeg.")
        return None
    except _DecodedOutputCeilingError:
        # Hard rejection: every fallback would re-decode the same input.
        raise
    except Exception as e:
        logger.debug("soundfile decode failed, falling back. Error: %s", e)
        return None
    return audio, orig_sr


def _enforce_soundfile_info_ceiling(sf: Any, source: str | io.BytesIO) -> None:
    """Reject a soundfile input whose header declares an over-ceiling decode (R9).

    Reads only the container header via ``soundfile.info`` (no sample decode) and
    rejects before :func:`soundfile.read` pre-allocates its output array, so a
    header claiming an enormous duration cannot drive a multi-GB allocation. A
    header that does not report a usable positive frame count (``frames <= 0``,
    seen for some streaming/edge formats) is left to the post-decode ceiling,
    and any ``soundfile`` failure here is swallowed so a header-probe hiccup
    degrades to the normal decode+post-check path rather than masking a decode
    error -- this guard only ever *tightens*, never rejects a file the decode
    would have accepted.

    Args:
        sf: The imported ``soundfile`` module.
        source: A local file path or a ``BytesIO`` over encoded bytes. A
            ``BytesIO`` is rewound to 0 before returning so the caller can decode.

    Raises:
        _DecodedOutputCeilingError: If ``frames x channels x 4`` (float32 bytes)
            exceeds the decoded-output ceiling. A hard rejection that must not
            fall back to another decoder.
    """
    try:
        info: Any = sf.info(source)
        frames = int(info.frames)
        channels = int(info.channels)
    except Exception as e:  # noqa: BLE001 - probe is advisory; never masks decode
        logger.debug("soundfile.info probe failed; deferring to post-decode check: %s", e)
        return
    finally:
        if isinstance(source, io.BytesIO):
            source.seek(0)
    if frames <= 0 or channels <= 0:
        return
    declared = frames * channels * 4  # float32 output is 4 bytes/sample
    if declared > _DEFAULT_MAX_DECODE_BYTES:
        raise _DecodedOutputCeilingError(
            f"soundfile header declares {frames} frames x {channels} channels "
            f"({declared} bytes of float32 PCM), exceeding the "
            f"{_DEFAULT_MAX_DECODE_BYTES}-byte decoded-output ceiling. The input "
            "likely declares a very long duration; provide a shorter clip."
        )


def _enforce_decoded_output_ceiling(audio: NDArray[Any]) -> None:
    """Reject a decoded waveform larger than the decoded-output ceiling (R9).

    The encoded-size cap (``max_bytes``) cannot bound the decoded array -- a
    compressed codec can expand far past it. The FFmpeg path bounds its decoded
    output with ``-fs`` plus a post-capture check; this is the soundfile path's
    equivalent, applied to the decoded array. The ceiling is the module's
    dedicated decode bound, deliberately NOT the engine's ``max_file_size``
    (which is an encoded-size contract).

    Args:
        audio: The freshly decoded waveform.

    Raises:
        _DecodedOutputCeilingError: If the array exceeds the ceiling. This is a
            hard rejection that must not fall back to another decoder.
    """
    if audio.nbytes > _DEFAULT_MAX_DECODE_BYTES:
        raise _DecodedOutputCeilingError(
            f"Decoded audio is {audio.nbytes} bytes, exceeding the "
            f"{_DEFAULT_MAX_DECODE_BYTES}-byte decoded-output ceiling. The input "
            "likely declares a very long duration; provide a shorter clip."
        )


def _guard_wav_nframes(wf: wave.Wave_read, max_bytes: int | None) -> None:
    """Reject a WAV whose header-declared frame count would over-allocate (R9).

    ``wave.readframes(getnframes())`` derives its read size from the WAV
    *header's* frame count, which an attacker controls independently of the file
    that actually follows. Probe ``getnframes() x channels x sampwidth`` against
    the cap *before* the read so a header claiming an enormous ``nframes`` is
    rejected up front rather than driving a large allocation off untrusted
    metadata. When ``max_bytes`` is ``None`` the module's default ceiling is used
    as the sane derived limit (this guard is a sanity bound, not the caller's
    encoded-size cap, so it stays bounded even for an uncapped caller).

    Args:
        wf: An open :class:`wave.Wave_read` positioned before the data read.
        max_bytes: The caller's encoded-size cap, or ``None`` to fall back to the
            module's default ceiling as the sanity bound.

    Raises:
        _WavAllocationGuardError: If the header-declared PCM size exceeds the
            limit. This is a hard rejection that must not fall back to another
            decoder.
    """
    limit = max_bytes if max_bytes is not None else _DEFAULT_MAX_DECODE_BYTES
    declared = wf.getnframes() * wf.getnchannels() * wf.getsampwidth()
    if declared > limit:
        raise _WavAllocationGuardError(
            f"WAV header declares {wf.getnframes()} frames "
            f"({declared} bytes of PCM), exceeding the {limit}-byte limit. The "
            "header is likely corrupt or hostile; provide a smaller file or an "
            "engine without this limit."
        )


def _read_wav_stdlib(path: str, max_bytes: int | None) -> tuple[NDArray[np.float32], int] | None:
    """Decode an 8/16-bit PCM WAV via stdlib ``wave``, or ``None`` to fall back.

    The single stdlib-WAV decode attempt shared by every path decoder, so the
    bomb-guard and exception-ladder discipline live in exactly one place (a
    copied ladder that dropped the ``_WavAllocationGuardError`` re-raise would
    route a header-declared decompression bomb into a SECOND full decode). The
    returned waveform is **un-normalized** (native rate, native channels): each
    caller applies its own :func:`normalize_audio` target so this helper stays
    agnostic to resample-to-16k vs decode-at-native-rate.

    Args:
        path: A local file path. A non-``.wav`` path returns ``None`` immediately.
        max_bytes: The caller's encoded-size cap, threaded to
            :func:`_guard_wav_nframes` to bound the header-declared frame
            allocation (spec R9). ``None`` uses the module's default ceiling.

    Returns:
        The decoded ``(float32 waveform, native sample rate)`` pair, or ``None``
        when the caller should fall back to soundfile/FFmpeg (non-WAV suffix or
        an ordinary stdlib decode failure).

    Raises:
        _WavAllocationGuardError: If the WAV header declares an over-limit frame
            count (hard rejection, never falls back).
    """
    if not path.lower().endswith(".wav"):
        return None
    try:
        with wave.open(path, "rb") as wf:
            orig_sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            n_channels = wf.getnchannels()
            # Only handle 8-bit (unsigned) and 16-bit PCM via stdlib; others fall back.
            if sampwidth not in (1, 2):
                raise AudioProcessingError(
                    f"Unsupported WAV sample width via stdlib: {sampwidth * 8} bits"
                )
            # Spec R9: getnframes() is a header-declared count an attacker
            # controls; bound the read it drives before allocating from it.
            _guard_wav_nframes(wf, max_bytes)
            frames = wf.readframes(wf.getnframes())
            # 16-bit PCM uses the canonical wire codec (spec R4) so the WAV reader
            # and the streaming wire path share ONE float<->pcm16 definition (the
            # deliberate 32767/32768 round-trip asymmetry lives only in wire.py).
            # 8-bit unsigned PCM has no wire codec, so it is decoded inline; uint8
            # has no endianness, while 16-bit is little-endian per the WAV contract.
            if sampwidth == 2:
                audio = pcm16_decode(frames)
            else:
                # 8-bit unsigned PCM: center at 128, scale to [-1, 1].
                audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                audio = (audio - 128.0) / 128.0
            # Re-affirm dtype for the static checker after arithmetic.
            audio = np.asarray(audio, dtype=np.float32)
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)
            return audio, orig_sr
    except _WavAllocationGuardError:
        # A header-declared decompression bomb is a hard rejection: every decoder
        # would parse the same hostile header, so do NOT fall back.
        raise
    except (wave.Error, AudioProcessingError, OSError, ValueError) as e:
        logger.debug(
            "Could not load WAV with stdlib `wave` (unsupported format or "
            "corrupted file); falling back to soundfile/ffmpeg. Error: %s",
            e,
        )
        return None


def _read_stream_capped(stream: BinaryIO, max_bytes: int | None) -> bytes:
    """Read a binary stream into bytes without exceeding ``max_bytes`` in memory.

    Unlike a bare ``stream.read()`` (which buffers the WHOLE stream before any
    size is observed), this reads at most ``max_bytes + 1`` bytes so an untrusted
    stream cannot force an unbounded allocation: if the stream yields more than
    ``max_bytes`` the read is aborted and an error raised (spec R9). When
    ``max_bytes`` is ``None`` the cap is disabled and the whole stream is read.

    The capped read **loops** to EOF rather than issuing a single
    ``stream.read(n)``: per the IO contract a ``RawIOBase`` (and any duck-typed
    ``read``, both accepted by :func:`_is_binary_io`) MAY return fewer than ``n``
    bytes on one call without being at EOF -- a single read of a pipe, socket
    wrapper, or unbuffered file would otherwise return only what is immediately
    available and the remainder would be silently dropped, truncating the audio.
    The loop preserves the ``max_bytes + 1`` memory ceiling (it never requests
    more than one byte past the cap in total).

    Args:
        stream: A binary IO stream positioned at the data to read.
        max_bytes: Maximum bytes to admit, or ``None`` for unbounded.

    Returns:
        The stream contents (at most ``max_bytes`` bytes when a cap is set).

    Raises:
        AudioProcessingError: If the stream exceeds ``max_bytes``.
    """
    if max_bytes is None:
        return stream.read()
    # Read up to one byte past the cap so an over-limit stream is detectable
    # without ever holding more than max_bytes + 1 bytes in memory. Loop because
    # a single read() may legally short-read (RawIOBase / pipe / socket); stop
    # only at genuine EOF (empty read) or once the over-limit sentinel is seen.
    chunks: list[bytes] = []
    remaining = max_bytes + 1
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break  # genuine EOF
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) > max_bytes:
        raise AudioProcessingError(
            f"Audio stream exceeds the decode limit of {max_bytes} bytes. Provide "
            "a smaller input or an engine without this limit."
        )
    return data


# numpy datatype check
@overload
def ensure_datatype(audio: NDArray[Any]) -> NDArray[np.float32]: ...


@overload
def ensure_datatype(audio: NDArray[Any], data_type: Literal["float32"]) -> NDArray[np.float32]: ...


@overload
def ensure_datatype(
    audio: NDArray[Any], data_type: np.dtype[np.float32]
) -> NDArray[np.float32]: ...


@overload
def ensure_datatype(audio: NDArray[Any], data_type: type[np.float32]) -> NDArray[np.float32]: ...


@overload
def ensure_datatype(audio: NDArray[Any], data_type: DTypeLike) -> NDArray[Any]: ...


def ensure_datatype(audio: NDArray[Any], data_type: DTypeLike = np.float32) -> NDArray[Any]:
    """Convert a NumPy array to the specified dtype (default: ``float32``).

    Args:
        audio: Input NumPy array of any dtype.
        data_type: Target dtype (e.g., ``"float32"``, ``np.float32``). Default: ``np.float32``.

    Returns:
        NumPy array with the requested dtype. Returns a view (no copy) if already matching.

    Raises:
        TypeError: If ``data_type`` is not a valid NumPy dtype.

    Example:
        >>> audio = ensure_datatype(raw_audio)  # -> float32
        >>> audio = ensure_datatype(raw_audio, "float64")  # -> float64
    """
    # Compute the target dtype for runtime comparison; helps static checkers.
    # Use np.asarray (not astype(copy=False)) per spec DEP.2 / D4: copy=False is
    # banned because its no-copy guarantee differs subtly across numpy 1.x/2.x.
    target_dtype: np.dtype[np.generic] = np.dtype(data_type)
    if audio.dtype != target_dtype:
        audio = np.asarray(audio, dtype=target_dtype)
    return audio


def _to_normalized_float32(audio: NDArray[Any]) -> NDArray[np.float32]:
    """Convert any common PCM dtype to ``float32`` scaled to ``[-1.0, 1.0]``.

    Integer PCM stores samples as raw codes, not amplitudes in ``[-1, 1]``, so a
    bare ``astype(float32)`` followed by a ``[-1, 1]`` clip silently corrupts the
    waveform (e.g. ``int16`` ``32767`` would clip to ``1.0`` and ``1000`` would
    clip to ``1.0`` as well -- a silent-wrong-result). This rescales by the
    dtype's full-scale magnitude instead:

    * **Signed integers** (``int8``/``int16``/``int32``/``int64``) divide by
      ``2**(bits-1)`` (e.g. ``int16`` -> ``/32768``), so ``-min`` maps to exactly
      ``-1.0`` and ``+max`` to just under ``+1.0`` (the standard asymmetric PCM
      convention).
    * **Unsigned integers** (``uint8``/``uint16``/``uint32``/``uint64``) are
      midpoint-centered then scaled: ``(x - 2**(bits-1)) / 2**(bits-1)`` (e.g.
      ``uint8`` centers at ``128`` then divides by ``128``).
    * **Floating** dtypes are taken as already-normalized amplitudes and only
      cast to ``float32`` (the caller still clips for safety).

    The division is performed in ``float64`` before the final ``float32`` cast so
    a 32/64-bit integer full-scale value does not lose precision mid-scale.

    Args:
        audio: Input waveform of any signed/unsigned PCM integer or floating
            dtype.

    Returns:
        A ``float32`` array scaled to ``[-1.0, 1.0]`` (integers) or cast as-is
        (floats).
    """
    dtype = audio.dtype
    if np.issubdtype(dtype, np.floating):
        return ensure_datatype(audio, "float32")
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if info.min == 0:
            # Unsigned PCM: center on the midpoint, then scale by the half-range.
            half_range = (int(info.max) + 1) / 2.0
            scaled = (audio.astype(np.float64) - half_range) / half_range
        else:
            # Signed PCM: divide by the full-scale magnitude (``-min``), so the
            # most-negative code maps to exactly -1.0.
            scaled = audio.astype(np.float64) / float(-int(info.min))
        return np.asarray(scaled, dtype=np.float32)
    # Anything exotic (e.g. bool, complex) is handled by a plain cast; clipping by
    # the caller keeps the contract range. This mirrors the historical behavior
    # for non-PCM dtypes rather than failing the rare caller.
    return ensure_datatype(audio, "float32")


def normalize_audio(
    audio: NDArray[Any],
    original_sr: int,
    target_sample_rate: int = 16000,
    target_channels: int | None = 1,
) -> NDArray[np.float32]:
    """Normalize a raw waveform to the Standard ASR audio format.

    **Standard ASR Audio Format:**

    - **dtype:** ``np.float32``
    - **Sample rate:** ``16000`` Hz (configurable)
    - **Channels:** Mono ``(n_samples,)`` or multi-channel ``(n_samples, n_channels)``
    - **Value range:** ``[-1.0, 1.0]``

    Args:
        audio: Input waveform, 1D ``(n_samples,)`` or 2D ``(n_samples, n_channels)``.
            Signed/unsigned PCM **integer** dtypes are rescaled to ``[-1, 1]`` by
            their full-scale magnitude (e.g. ``int16`` is divided by ``32768``,
            ``uint8`` is centered at ``128`` then divided by ``128``); **floating**
            dtypes are taken as already-normalized amplitudes (and clipped for
            safety). All inputs become ``float32``.
        original_sr: Sample rate of the input audio (Hz). Must be > 0.
        target_sample_rate: Target sample rate. Default: ``16000`` Hz.
        target_channels: Target channel count. ``1`` = mono (default), ``2`` = stereo,
            ``None`` = preserve original.

    Returns:
        Normalized waveform: ``np.float32``, resampled to ``target_sample_rate``, with
        ``target_channels`` channels, values clipped to ``[-1.0, 1.0]``.

    Raises:
        AudioProcessingError: Invalid parameters or empty audio.

    Note:
        **Resampling:** Uses ``scipy.signal.resample_poly`` for high-quality
        conversion when ``scipy`` is installed (the ``[audio]`` extra), and
        degrades to the built-in numpy-only anti-aliasing Fourier resampler
        otherwise (a missing extra is never fatal, spec AI R8).

        **Channel conversion:**

        - Stereo → Mono: arithmetic mean of channels.
        - Mono → Stereo: channel replication.
        - Multi-channel down-mix: truncates to first N channels (for better quality,
          use FFmpeg path via ``load_audio``).

        **Invalid values:** NaN/Inf are sanitized (NaN→0.0, ±Inf→±1.0) with a warning.
    """
    # Scale integer PCM to [-1, 1] by its full-scale magnitude before any
    # processing; a bare float cast + clip would corrupt integer samples
    # (e.g. int16 32767 -> 1.0 AND 1000 -> 1.0). Floats pass through unscaled.
    processed_audio: NDArray[np.float32] = _to_normalized_float32(audio)
    # Basic parameter validation

    if original_sr <= 0:
        raise AudioProcessingError(f"original_sr must be > 0, got {original_sr}")
    if target_sample_rate <= 0:
        raise AudioProcessingError(f"target_sample_rate must be > 0, got {target_sample_rate}")
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(f"target_channels must be None or > 0, got {target_channels}")

    # Check for empty audio
    if processed_audio.size == 0:
        raise AudioProcessingError("Cannot process empty audio array")

    # 1. Resample if necessary. Delegate to the shared backend selector so the
    #    scipy-preferred / numpy-fallback ladder lives in exactly one place
    #    (resampling.resample_with_backend) -- it prefers scipy's polyphase
    #    resampler (the [audio] extra) and degrades to the core numpy-only
    #    anti-aliasing Fourier resampler, treating ANY scipy import-time failure
    #    (not just a plain ImportError -- e.g. a broken/partial build) as a
    #    non-fatal fall to the built-in (spec AI R8). Warn only when the fallback
    #    actually ran.
    if original_sr != target_sample_rate:
        from ..resampling import resample_with_backend as _resample_with_backend

        processed_audio, _backend = _resample_with_backend(
            processed_audio, original_sr, target_sample_rate
        )
        if _backend == "fallback":
            logger.warning(
                "scipy not available; using the built-in anti-aliasing fallback "
                "resampler. Install standard-asr[audio] for higher quality."
            )

    # Ensure at least 2D for uniform channel processing
    if processed_audio.ndim == 1:
        processed_audio = processed_audio[:, np.newaxis]

    # Help static type checker with tuple[int, ...] for shape
    current_channels = int(processed_audio.shape[1])

    # 2. Channel conversion
    if target_channels is not None and current_channels != target_channels:
        if target_channels == 1:
            # Downmix to mono using average across channels
            processed_audio = processed_audio.mean(axis=1, dtype=np.float32)[:, np.newaxis]
        else:
            # Note: For multi-to-multi down-mixing (e.g., 6->2), this implementation performs a
            # simple channel selection/truncation instead of a perceptually accurate mix.
            # For higher quality down-mix, ensure FFmpeg is installed and prefer the FFmpeg path.
            if target_channels > current_channels:  # Upscale (e.g., mono to stereo)
                reps = int(math.ceil(target_channels / current_channels))
                processed_audio = np.tile(processed_audio, (1, reps))[:, :target_channels]
            else:  # Down-mix by truncation
                logger.warning(
                    "Down-mixing from %d to %d channels by taking the first %d channels. "
                    "This may result in information loss. For high-quality down-mixing, "
                    "ensure your audio is processed via the FFmpeg backend.",
                    current_channels,
                    target_channels,
                    target_channels,
                )
                processed_audio = processed_audio[:, :target_channels]

    # Clean up any NaN/Inf values that might have been introduced during processing
    if not np.isfinite(processed_audio).all():
        bad_count = (~np.isfinite(processed_audio)).sum()
        logger.warning(
            "Detected %d invalid samples (NaN/Inf) in audio; replacing with safe values.",
            int(bad_count),
        )
        processed_audio = np.nan_to_num(processed_audio, nan=0.0, posinf=1.0, neginf=-1.0)

    # Clip to contract range, then cast. Clip BEFORE cast (DEP.2 defensive
    # ordering) and use np.asarray instead of astype(copy=False) (DEP.2 ban).
    processed_audio = np.asarray(np.clip(processed_audio, -1.0, 1.0), dtype=np.float32)
    # Respect contract: mono->1D, multi->2D even if n_samples==1
    if int(processed_audio.shape[1]) == 1:
        return processed_audio[:, 0]
    return processed_audio


def load_audio(
    source: str | bytes | bytearray | memoryview | pathlib.Path | BinaryIO,
    target_sample_rate: int = 16000,
    target_channels: int | None = 1,
    *,
    max_bytes: int | None = _DEFAULT_MAX_DECODE_BYTES,
) -> NDArray[np.float32]:
    """Load audio from given source and convert to Standard ASR format.

    **Accepts:** File path, ``bytes``, ``pathlib.Path``, base64 data URI, or file-like object.

    **Returns:** ``np.float32`` array, 16kHz mono by default, values in ``[-1.0, 1.0]``.

    Convenience loader for application code that wants a decoded array directly
    (e.g. to plot or pre-process audio). It handles format detection, decoding,
    resampling, and channel conversion automatically, and for a ``str`` it
    auto-detects a base64 ``data:`` URI.

    This is **not** the engine input boundary. When transcribing, pass an
    :data:`~standard_asr.audio_input.AudioInput` to ``transcribe`` and let the
    standard negotiation layer decode/convert per the engine's ``accepted_input``
    -- there a bare ``str`` is **always** a file path and is never sniffed (a
    security boundary against SSRF / data-URI confusion). This helper's
    convenience sniffing is intentional and local to it.

    Args:
        source: Audio input. Supported types:

            - ``str``: File path (``"audio.mp3"``) or base64 data URI
              (``"data:audio/wav;base64,..."``).
            - ``bytes`` / ``bytearray`` / ``memoryview``: Raw audio file bytes.
            - ``pathlib.Path``: File path object.
            - ``BinaryIO``: File-like object opened in binary mode.

        target_sample_rate: Output sample rate in Hz. Default: ``16000``.
        target_channels: Output channels. ``1`` = mono (default), ``2`` = stereo,
            ``None`` = preserve.
        max_bytes: Max ENCODED input size, threaded to the underlying loaders and
            defaulting to the 2 GiB ceiling (spec R9). It bounds the encoded
            payload only -- a file via ``stat``, ``bytes`` via length, and a
            ``BinaryIO`` stream via a capped read that aborts past the limit
            instead of buffering the whole stream. The decoded array is bounded
            separately by the module's fixed decoded-output ceiling, which
            ``max_bytes`` does not change. Pass ``None`` to disable the encoded
            cap entirely (truly unbounded); set it explicitly for untrusted
            input.

    Returns:
        Waveform as ``np.float32`` array:

        - Shape: ``(n_samples,)`` for mono, ``(n_samples, n_channels)`` for multi-channel.
        - Sample rate: ``target_sample_rate`` Hz.
        - Values: Clipped to ``[-1.0, 1.0]``.

    Raises:
        AudioProcessingError: Invalid parameters or decoding/processing failures,
            including missing or unreadable paths, or input exceeding ``max_bytes``.
        FFmpegNotFoundError: FFmpeg fallback needed but not installed.
        TypeError: Unsupported source type.

    Example:
        >>> audio = load_audio("speech.mp3")  # Load from file
        >>> audio = load_audio(audio_bytes)   # Load from bytes
        >>> audio = load_audio(Path("~/audio.wav"), target_sample_rate=8000)  # Custom sample rate

    Note:
        **Decoding priority:**
        - File paths: stdlib ``wave`` → ``soundfile`` → FFmpeg subprocess.
        - Bytes/data URIs/BinaryIO: ``soundfile`` → FFmpeg subprocess.

        **Base64:** Only data URIs (``data:...;base64,...``) are auto-detected.
        For raw base64 strings (eg. ``YmFT...Y0``), decode manually:
        ``load_audio(base64.b64decode(s))``.

        **BinaryIO:** Reads from the current stream position; does not seek to the beginning.

        **Formats:** WAV, MP3, FLAC, OGG, and any format supported by FFmpeg.
    """
    if target_sample_rate <= 0:
        raise AudioProcessingError(f"target_sample_rate must be > 0, got {target_sample_rate}")
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(f"target_channels must be None or > 0, got {target_channels}")
    if isinstance(source, str):
        # Improved base64 detection logic to avoid false positives
        s = source.strip()

        # First check: if it has explicit base64 data URI prefix, treat as base64
        if s.lower().startswith("data:") and ";base64," in s:
            # Gate-and-decode (R9): the decoded size is estimated from the
            # payload length and checked BEFORE the decode allocates it.
            source_bytes = _decode_base64_bounded(s, max_bytes)
            return load_audio_from_bytes(
                source_bytes, target_sample_rate, target_channels, max_bytes=max_bytes
            )

        # Second check: if it exists as a file path, prioritize as path
        try:
            path = pathlib.Path(s).expanduser()
            if path.exists():
                return load_audio_from_path(
                    str(path), target_sample_rate, target_channels, max_bytes=max_bytes
                )
        except (OSError, ValueError):
            # Path operations failed, continue to treat as file path anyway
            pass

        # Default: treat as file path (will raise FileNotFoundError if not found)
        return load_audio_from_path(s, target_sample_rate, target_channels, max_bytes=max_bytes)

    # Handle pathlib.Path by converting to string
    if isinstance(source, pathlib.Path):
        return load_audio_from_path(
            str(source), target_sample_rate, target_channels, max_bytes=max_bytes
        )

    # Bytes-like objects
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = source.tobytes() if isinstance(source, memoryview) else bytes(source)
        return load_audio_from_bytes(data, target_sample_rate, target_channels, max_bytes=max_bytes)

    # File-like object that returns bytes. Read it with a running cap so an
    # untrusted stream cannot blow up memory before max_bytes is observed (R9).
    if _is_binary_io(source):
        data = _read_stream_capped(source, max_bytes)
        return load_audio_from_bytes(data, target_sample_rate, target_channels, max_bytes=max_bytes)

    raise TypeError(f"Unsupported audio source type: {type(source)}")


# --- Public Loaders ---


def _is_binary_io(obj: Any) -> TypeGuard[BinaryIO]:
    """Check if an object is a binary IO stream (internal helper).

    Args:
        obj: Any object to check.

    Returns:
        ``True`` if ``obj`` is a binary IO (returns bytes on read), ``False`` otherwise.

    Raises:
        None.
    """
    try:
        if isinstance(obj, (io.BufferedIOBase, io.BytesIO, io.RawIOBase)):
            return True
        read_attr = getattr(obj, "read", None)
        if not callable(read_attr):
            return False
        # Probe a zero-length read without consuming data. We avoid peek/read1/seek
        # for broad compatibility and to not alter stream position/state. Some custom
        # streams may not support read(0); in such case we return False.
        sample = read_attr(0)  # pyright: ignore[reportCallIssue]
        return isinstance(sample, (bytes, bytearray, memoryview))
    except Exception:
        return False


def load_audio_from_path(
    path: str,
    target_sample_rate: int = 16000,
    target_channels: int | None = 1,
    *,
    max_bytes: int | None = _DEFAULT_MAX_DECODE_BYTES,
) -> NDArray[np.float32]:
    """Load audio from a file path and convert to Standard ASR format.

    **Accepts:** File path as string (e.g., ``"speech.mp3"``, ``"~/audio.wav"``).

    **Returns:** ``np.float32`` array, resampled and channel-converted.

    This is a convenience loader, **not** the engine input boundary. The
    ``max_bytes`` cap bounds only the on-disk ENCODED file size via ``stat``
    (spec R9); the decoded array is bounded separately by the module's fixed
    decoded-output ceiling on every decode backend. Callers handling untrusted
    input SHOULD set ``max_bytes`` (or ``None`` to disable the default 2 GiB
    encoded cap).

    Args:
        path: Path to audio file. Supports ``~`` expansion.
        target_sample_rate: Output sample rate (Hz). Default: ``16000``.
        target_channels: Output channels. ``1`` = mono, ``2`` = stereo, ``None`` = preserve.
        max_bytes: Max encoded file size; defaults to the 2 GiB ceiling. Pass
            ``None`` to disable the cap.

    Returns:
        Waveform as ``np.float32``, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.

    Raises:
        AudioProcessingError: Decoding failed, including missing or unreadable
            files, or the file exceeds ``max_bytes``.
        FFmpegNotFoundError: FFmpeg fallback needed but not installed.

    Note:
        **Decoding priority:** stdlib ``wave`` (WAV) → ``soundfile`` → FFmpeg.

        For broader format support, install ``soundfile`` or ensure FFmpeg is in PATH.
    """
    # Basic parameter validation
    if target_sample_rate <= 0:
        raise AudioProcessingError(f"target_sample_rate must be > 0, got {target_sample_rate}")
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(f"target_channels must be None or > 0, got {target_channels}")

    # Expand user (~) to avoid surprises across platforms
    from os import fspath

    path = fspath(pathlib.Path(path).expanduser())
    expanded = pathlib.Path(path)
    # Reject an EXISTING non-regular file (FIFO, device, directory) up front:
    # stdlib `wave`/`soundfile` would otherwise block forever on a FIFO with no
    # timeout (a hang/DoS vector), and only the FFmpeg layer's
    # _validate_local_source_path currently rejects it. A genuinely MISSING path
    # is deliberately left to fall through so the decoders surface their clear
    # "not found" error (preserving the convenience-loader semantics).
    #
    # The probe syscalls (exists/is_file/stat) are wrapped in
    # OSError -> AudioProcessingError, mirroring _validate_local_source_path: a bare
    # str is never sniffed, so a pathologically long path string (e.g. a real-sized
    # data:/base64 URI mistakenly passed as a path, > ~1024 chars) makes the first
    # probe raise OSError(ENAMETOOLONG), which must surface as the contracted
    # AudioProcessingError rather than leak through the documented contract.
    try:
        if expanded.exists() and not expanded.is_file():
            raise AudioProcessingError(
                f"Audio path is not a regular file: {path!r}. Pass a real local file "
                "(FIFOs, devices, and directories are not supported)."
            )
        # Precheck the encoded file size via stat() before any read/decode (spec R9),
        # only when the file exists so a missing path still surfaces "not found".
        if expanded.is_file():
            _enforce_decode_size(expanded.stat().st_size, max_bytes)
    except OSError as exc:
        raise AudioProcessingError(
            f"Audio file not found or not a regular file: {path!r} ({exc}). Bare "
            "strings are always treated as local file paths; wrap URLs in AudioUrl "
            "and base64/data: URIs in AudioBase64 (or use decode_audio_from_data_uri)."
        ) from exc

    # Layer 1: WAV files with Python standard library `wave` (shared helper owns
    # the bomb-guard + fall-back-vs-hard-reject exception discipline).
    wav = _read_wav_stdlib(path, max_bytes)
    if wav is not None:
        audio, orig_sr = wav
        return normalize_audio(audio, orig_sr, target_sample_rate, target_channels)

    # Layer 2: Use `soundfile` for formats like FLAC, OGG, etc. (the shared
    # attempt owns the fall-back-vs-hard-reject exception discipline).
    decoded = _read_with_soundfile(path)
    if decoded is not None:
        audio, orig_sr = decoded
        # normalize_audio never raises ImportError: a missing scipy degrades to
        # the built-in anti-aliasing fallback resampler internally (spec AI R8).
        return normalize_audio(audio, orig_sr, target_sample_rate, target_channels)

    # Layer 3: Final fallback to FFmpeg
    return _load_with_ffmpeg(path, target_sample_rate, target_channels)


def load_audio_from_bytes(
    data: bytes,
    target_sample_rate: int = 16000,
    target_channels: int | None = 1,
    *,
    max_bytes: int | None = _DEFAULT_MAX_DECODE_BYTES,
) -> NDArray[np.float32]:
    """Load audio from raw bytes and convert to Standard ASR format.

    **Accepts:** Audio file content as ``bytes`` (e.g., from ``file.read()``, HTTP response).

    **Returns:** ``np.float32`` array, resampled and channel-converted.

    This is a convenience loader, **not** the engine input boundary. The
    ``max_bytes`` cap bounds only the ENCODED ``data`` length (spec R9); the
    decoded array (which a compressed codec can expand far past it) is bounded
    separately by the module's fixed decoded-output ceiling on every decode
    backend. Callers handling untrusted input SHOULD set ``max_bytes`` (or
    ``None`` to disable the default 2 GiB encoded cap).

    Args:
        data: Raw bytes of an audio file (any format: WAV, MP3, FLAC, etc.).
        target_sample_rate: Output sample rate (Hz). Default: ``16000``.
        target_channels: Output channels. ``1`` = mono, ``2`` = stereo, ``None`` = preserve.
        max_bytes: Max encoded payload size; defaults to the 2 GiB ceiling.
            Pass ``None`` to disable the cap.

    Returns:
        Waveform as ``np.float32``, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.

    Raises:
        AudioProcessingError: Decoding failed, empty audio, or ``data`` exceeds
            ``max_bytes``.
        FFmpegNotFoundError: FFmpeg fallback needed but not installed.

    Note:
        **Decoding priority:** ``soundfile`` → FFmpeg. Install one for format support.
    """
    # Basic parameter validation
    if target_sample_rate <= 0:
        raise AudioProcessingError(f"target_sample_rate must be > 0, got {target_sample_rate}")
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(f"target_channels must be None or > 0, got {target_channels}")
    _enforce_decode_size(len(data), max_bytes)
    # Layer 2: `soundfile` is the best primary method for bytes (the shared
    # attempt owns the fall-back-vs-hard-reject exception discipline).
    decoded = _read_with_soundfile(io.BytesIO(data))
    if decoded is not None:
        audio, orig_sr = decoded
        # normalize_audio never raises ImportError: a missing scipy degrades to
        # the built-in anti-aliasing fallback resampler internally (spec AI R8).
        return normalize_audio(audio, orig_sr, target_sample_rate, target_channels)

    # Layer 3: Final fallback to FFmpeg
    return _load_with_ffmpeg(data, target_sample_rate, target_channels)


def decode_audio(
    source: str | bytes | bytearray | memoryview | pathlib.Path,
    *,
    target_channels: int | None = 1,
    max_bytes: int | None = None,
) -> tuple[NDArray[np.float32], int]:
    """Decode audio to a waveform at its **native** sample rate.

    Unlike :func:`load_audio`, this primitive does **not** resample: it returns
    the decoded waveform together with the source's original sample rate, so the
    caller can make the single authoritative resampling decision (spec R7). This
    is what the conversion layer needs to honour 8 kHz telephony and 24 kHz
    realtime engines without a spurious round-trip through 16 kHz (spec R7, the
    "MUST NOT upsample native-rate input" clause).

    This is the engine-input decode boundary. A bare ``str`` is **always** a
    local file path and is **never** content-sniffed (spec R1 / §3.1: a bare
    ``str`` coerces to ``AudioPath``; the discriminant is the explicit type tag,
    never string content). A string that happens to look like a ``data:`` URI is
    therefore opened as a file (and fails "not found"), **not** decoded as
    base64 -- decode an explicit base64/``data:`` payload with
    :func:`decode_audio_from_data_uri`, or let the conversion layer's
    ``AudioBase64`` path decode it to bytes first; never pass the string here.

    Args:
        source: File path, ``pathlib.Path``, or raw bytes. A bare ``str`` is
            always a local file path (never sniffed as a ``data:`` URI); use
            :func:`decode_audio_from_data_uri` for base64/``data:`` payloads.
        target_channels: Output channels. ``1`` = mono (default), ``None`` =
            preserve the source channel layout.
        max_bytes: Cap on the buffered ENCODED payload size (spec R9). ``None``
            (the default) means **truly unbounded** -- this primitive is driven
            by the conversion layer, which threads the engine's declared limit
            through verbatim (an engine with no limit -> ``None`` -> no cap).
            Callers handling untrusted input directly SHOULD pass a positive cap.

    Returns:
        A ``(array, native_sample_rate)`` pair. The array is ``float32`` in
        ``[-1, 1]`` at the source's original sample rate.

    Raises:
        AudioProcessingError: Decoding failed, the input is missing/oversize, or
            looks like an injected option.
        FFmpegNotFoundError: FFmpeg fallback needed but not installed.
        TypeError: Unsupported source type.
    """
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(f"target_channels must be None or > 0, got {target_channels}")

    if isinstance(source, str):
        # A bare str is ALWAYS a local file path -- never content-sniffed for a
        # data: URI (spec R1 / §3.1). Sniffing here would let a pathological
        # filename literally named "data:audio/...;base64,..." be decoded as
        # inline base64 instead of opened as a file, silently overriding the
        # explicit AudioPath type tag the conversion layer relies on. No
        # strip(): a path with surrounding whitespace is a legitimate (if
        # unusual) path and MUST NOT be rewritten.
        path = _validate_local_source_path(source)
        _enforce_path_decode_size(path, max_bytes)
        return _decode_path_native(path, target_channels, max_bytes)

    if isinstance(source, pathlib.Path):
        # Strict path-only entry: no sniff, no strip -- a bare path is ALWAYS a
        # local file (spec R1). A path shaped like a data: URI is rejected here.
        path = _validate_local_source_path(str(source))
        _enforce_path_decode_size(path, max_bytes)
        return _decode_path_native(path, target_channels, max_bytes)

    # Defensive: the annotation narrows to bytes-like here, but this is a public
    # boundary that must reject mistyped runtime input gracefully.
    if not isinstance(source, (bytes, bytearray, memoryview)):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"Unsupported audio source type: {type(source)}")
    data = source.tobytes() if isinstance(source, memoryview) else bytes(source)
    _enforce_decode_size(len(data), max_bytes)
    return _decode_bytes_native(data, target_channels)


def decode_audio_from_data_uri(
    value: str,
    *,
    target_channels: int | None = 1,
    max_bytes: int | None = None,
) -> tuple[NDArray[np.float32], int]:
    """Decode a base64 ``data:`` URI (or bare base64) to a native-rate waveform.

    The **explicit** base64/``data:`` decode entry point. :func:`decode_audio`
    deliberately never content-sniffs a bare ``str`` (spec R1 / §3.1: a bare
    ``str`` is always a local file path), so a caller that genuinely holds a
    base64 payload routes it here -- the decision to treat the string as base64
    is made by the call site's choice of function, not by inspecting the string's
    contents. This mirrors the conversion layer's ``AudioBase64`` handling.

    Args:
        value: A ``data:...;base64,...`` URI (any case for the scheme) or a bare
            base64 string. A ``data:`` URI MUST carry the ``;base64,`` marker.
        target_channels: Output channels. ``1`` = mono (default), ``None`` =
            preserve the source channel layout.
        max_bytes: Cap on the decoded ENCODED payload size (spec R9). ``None``
            (the default) means **truly unbounded**; callers handling untrusted
            input SHOULD pass a positive cap.

    Returns:
        A ``(array, native_sample_rate)`` pair. The array is ``float32`` in
        ``[-1, 1]`` at the source's original sample rate.

    Raises:
        AudioProcessingError: The payload is not valid base64, a ``data:`` URI
            lacks the ``;base64,`` marker, the decoded size exceeds ``max_bytes``,
            or decoding failed.
        FFmpegNotFoundError: FFmpeg fallback needed but not installed.
        TypeError: ``value`` is not a ``str``.
    """
    if not isinstance(value, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"decode_audio_from_data_uri requires a str, got {type(value)}")
    if target_channels is not None and target_channels <= 0:
        raise AudioProcessingError(f"target_channels must be None or > 0, got {target_channels}")
    # Gate-and-decode (R9): the decoded size is estimated from the payload length
    # and checked BEFORE the decode allocates it (the exact length is re-checked
    # inside _decode_base64_bounded).
    decoded = _decode_base64_bounded(value, max_bytes)
    return _decode_bytes_native(decoded, target_channels)


def _decode_path_native(
    path: str, target_channels: int | None, max_bytes: int | None = None
) -> tuple[NDArray[np.float32], int]:
    """Decode a (validated, existing) file path to ``(array, native_sr)``.

    Args:
        path: A validated absolute local file path.
        target_channels: Output channels, or ``None`` to preserve.
        max_bytes: The caller's encoded-size cap, threaded through to bound the
            stdlib WAV path's header-declared frame allocation (spec R9). ``None``
            falls back to the module's default ceiling as the sanity bound.

    Returns:
        The decoded ``float32`` waveform and its native sample rate.

    Raises:
        AudioProcessingError: Decoding failed.
        FFmpegNotFoundError: FFmpeg fallback needed but not installed.
    """
    # Layer 1: stdlib WAV via the shared helper (decode at the NATIVE rate, so
    # normalize_audio's target equals the source rate -- no resample here).
    wav = _read_wav_stdlib(path, max_bytes)
    if wav is not None:
        audio, orig_sr = wav
        return normalize_audio(audio, orig_sr, orig_sr, target_channels), orig_sr

    decoded = _read_with_soundfile(path)
    if decoded is not None:
        audio, orig_sr = decoded
        return normalize_audio(audio, orig_sr, orig_sr, target_channels), orig_sr

    return _decode_with_ffmpeg_native(path, target_channels)


def _decode_bytes_native(
    data: bytes, target_channels: int | None
) -> tuple[NDArray[np.float32], int]:
    """Decode raw bytes to ``(array, native_sr)`` without resampling.

    Args:
        data: Encoded audio bytes.
        target_channels: Output channels, or ``None`` to preserve.

    Returns:
        The decoded ``float32`` waveform and its native sample rate.

    Raises:
        AudioProcessingError: Decoding failed.
        FFmpegNotFoundError: FFmpeg fallback needed but not installed.
    """
    decoded = _read_with_soundfile(io.BytesIO(data))
    if decoded is not None:
        audio, orig_sr = decoded
        return normalize_audio(audio, orig_sr, orig_sr, target_channels), orig_sr

    return _decode_with_ffmpeg_native(data, target_channels)


def _decode_with_ffmpeg_native(
    source: str | bytes, target_channels: int | None
) -> tuple[NDArray[np.float32], int]:
    """Decode via FFmpeg preserving the native sample rate.

    Args:
        source: A validated local file path, or raw bytes.
        target_channels: Output channels, or ``None`` to preserve.

    Returns:
        The decoded ``float32`` waveform and its native sample rate.

    Raises:
        AudioProcessingError: If the native rate cannot be determined or decoding
            fails.
        FFmpegNotFoundError: FFmpeg not in PATH.
    """
    native_sr = _probe_sample_rate_with_ffprobe(source)
    if native_sr is None:
        raise AudioProcessingError(
            "Could not determine the native sample rate via ffprobe; install "
            "ffprobe or the [audio] extra (soundfile) for native-rate decoding."
        )
    array = _load_with_ffmpeg(source, native_sr, target_channels)
    return array, native_sr


def _load_with_ffmpeg(
    source: str | bytes,
    target_sample_rate: int,
    target_channels: int | None,
    timeout: float = 120.0,
    max_output_bytes: int | None = _DEFAULT_MAX_DECODE_BYTES,
) -> NDArray[np.float32]:
    """Decode audio via FFmpeg subprocess (internal fallback).

    Output-size bound (spec R9): the encoded-input cap does NOT bound the decoded
    PCM (output size is ``duration x sample_rate x channels x 4``), so a crafted
    long-duration input could otherwise force a multi-GB allocation. This bounds
    the decoded output two ways: ffmpeg is given ``-fs <max_output_bytes>`` so it
    stops writing once the limit is reached (the OS buffer for ``capture_output``
    therefore cannot grow past it), and the captured stdout is re-checked against
    the same ceiling afterwards and rejected if exceeded (defense in depth for an
    ffmpeg build that ignores ``-fs``). The guarantee is an output-byte ceiling,
    not a streaming decoder; ``timeout`` still bounds wall-clock time.

    Args:
        source: File path or raw bytes.
        target_sample_rate: Output sample rate (Hz).
        target_channels: Output channels, or ``None`` to auto-detect via ffprobe.
        timeout: Max seconds before aborting. Default: ``120.0``.
        max_output_bytes: Ceiling on the decoded PCM output, in bytes. ``None``
            disables it (unbounded). Defaults to the 2 GiB module ceiling.

    Returns:
        Waveform as ``np.float32``, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.

    Raises:
        FFmpegNotFoundError: FFmpeg not in PATH.
        AudioProcessingError: Decoding failed, timeout, empty output, or decoded
            output exceeding ``max_output_bytes``.
    """
    # Security (D1 / spec R5 rationale): a string source is a *local file* and
    # nothing else. Validate + absolutize it up front (before probing), and
    # constrain ffmpeg/ffprobe to the file/pipe protocols so they can never be
    # coerced into fetching http(s)://, tcp://, concat:, data:, etc. via a
    # crafted input string.
    #
    # Validate BEFORE the ffmpeg-presence check: a missing or
    # mistyped path must surface the decoders' clear "not found" error even when
    # ffmpeg is absent, not a misleading "install ffmpeg" that sends a beginner
    # off installing an unrelated dependency for a typo'd filename.
    if isinstance(source, bytes):
        input_arg = "pipe:0"
        probe_source: str | bytes = source
    else:
        input_arg = _validate_local_source_path(source)
        probe_source = input_arg

    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError(
            "FFmpeg not found in PATH. Install via: 'brew install ffmpeg' (macOS), "
            "'sudo apt-get install ffmpeg' (Debian/Ubuntu), 'winget install ffmpeg' or "
            "'choco install ffmpeg' (Windows)."
        )

    # If target_channels is None, attempt to preserve original channels via ffprobe.
    if target_channels is None:
        detected_channels = _probe_channels_with_ffprobe(probe_source)
        if detected_channels is None:
            logger.warning(
                "ffprobe not available or failed to detect channels. "
                "Defaulting to mono (1 channel)."
            )
            final_target_channels = 1
        else:
            final_target_channels = detected_channels
    else:
        final_target_channels = target_channels

    cmd = [
        "ffmpeg",
        "-nostdin",  # Prevent FFmpeg from waiting for stdin
        "-nostats",
        "-loglevel",
        "error",
        "-threads",
        "0",  # Use optimal number of threads
        "-protocol_whitelist",
        "file,pipe",  # Disallow network/chaining protocols (LFI/SSRF defense).
        "-i",
        input_arg,
        # Output options (must follow the input):
        "-vn",
        "-sn",
        "-dn",
        "-map",
        "a:0",  # Explicitly select the first audio stream
        "-f",
        "f32le",  # Output format: 32-bit floating-point, little-endian
        "-ac",
        str(final_target_channels),  # Set number of audio channels
        "-ar",
        str(target_sample_rate),  # Set audio sample rate
    ]
    if max_output_bytes is not None:
        # Make ffmpeg self-limit its output so capture_output cannot buffer past
        # the ceiling: ``-fs`` stops writing once the byte limit is reached
        # (spec R9). ffmpeg flushes whole 4096-byte blocks, so with ``-fs`` set
        # to the ceiling itself a stream truncated at a block-aligned ceiling
        # (the 2 GiB default is one) would land EXACTLY on the limit and be
        # indistinguishable from a legal maximal output -- a silently truncated
        # transcript. Grant one block of headroom instead: legal output
        # (<= ceiling) never reaches the limit, while a truncated stream
        # necessarily exceeds the ceiling and trips the post-capture check
        # below.
        cmd += ["-fs", str(max_output_bytes + _FFMPEG_FS_BLOCK)]
    cmd.append("-")  # Pipe output to stdout

    input_data = source if isinstance(source, bytes) else None

    try:
        proc = subprocess.run(
            cmd, capture_output=True, input=input_data, check=True, timeout=timeout
        )
        if not proc.stdout:
            raise AudioProcessingError("FFmpeg produced no audio data.")
        # Defense in depth: reject decoded output past the ceiling even if an
        # ffmpeg build ignored ``-fs`` (spec R9). Bounds the np.frombuffer copy.
        if max_output_bytes is not None and len(proc.stdout) > max_output_bytes:
            raise AudioProcessingError(
                f"FFmpeg decoded output exceeds the {max_output_bytes}-byte ceiling. "
                "The input likely declares a very long duration; provide a shorter "
                "clip or an engine without this limit."
            )
        audio: NDArray[np.float32] = np.frombuffer(proc.stdout, dtype=np.float32)

        # Contract guarantee: check for empty decoded audio
        if audio.size == 0:
            raise AudioProcessingError("FFmpeg decoded audio is empty (no audio samples).")

        # Reshape the flat array into (n_samples, n_channels) if multi-channel
        if final_target_channels > 1:
            n = (audio.size // final_target_channels) * final_target_channels
            if n != audio.size:
                logger.warning("Dropping %d trailing samples to align channels.", audio.size - n)
            audio = audio[:n].reshape(-1, final_target_channels)

            # Check for empty array after channel alignment
            if audio.size == 0:
                raise AudioProcessingError(
                    "FFmpeg produced too few samples to form a complete multi-channel frame."
                )

        # Contract guarantee: clean up any NaN/Inf values from FFmpeg
        if not np.isfinite(audio).all():
            bad_count = (~np.isfinite(audio)).sum()
            logger.warning(
                "Detected %d invalid samples (NaN/Inf) from FFmpeg; replacing with safe values.",
                int(bad_count),
            )
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        # Contract guarantee: ensure values are in [-1, 1] range
        audio = np.clip(audio, -1.0, 1.0)

        # Respect contract: mono->1D, multi->2D even if n_samples==1
        if final_target_channels == 1:
            # Ensure we return 1D array (n_samples,) not scalar for single sample
            return audio.reshape(-1)
        return audio

    except subprocess.TimeoutExpired as e:
        raise AudioProcessingError(
            f"FFmpeg timed out after {timeout} seconds while processing audio. "
            "This may indicate corrupted input or very large file."
        ) from e
    except subprocess.CalledProcessError as e:
        # Limit stderr output to prevent overwhelming error messages
        stderr_msg = e.stderr.decode(errors="ignore")[:2000]
        if len(stderr_msg) == 2000:
            stderr_msg += "... (truncated)"

        raise AudioProcessingError(
            f"FFmpeg failed to process audio: {stderr_msg} | "
            "Install via 'brew install ffmpeg', 'sudo apt-get install ffmpeg', "
            "'winget install ffmpeg' or 'choco install ffmpeg'."
        ) from e


def _probe_stream_entry(source: str | bytes, entry: str, timeout: float = 5.0) -> int | None:
    """Query a single integer ``stream=<entry>`` value via ffprobe (guarded).

    Like the ffmpeg decode path, this constrains ffprobe to the ``file,pipe``
    protocols, so a crafted path can never trigger a network fetch (D1 / spec R5
    rationale). String sources are forwarded verbatim; callers that accept
    untrusted paths MUST validate them first via :func:`_validate_local_source_path`.

    Args:
        source: A local file path, or raw bytes.
        entry: The ``stream=`` field to read (e.g. ``"channels"``,
            ``"sample_rate"``).
        timeout: Max seconds to wait. Default: ``5.0``.

    Returns:
        The positive integer value, or ``None`` if ffprobe is unavailable,
        detection failed, or the reported value is not a positive integer.

    Raises:
        None.
    """
    if shutil.which("ffprobe") is None:
        return None

    if isinstance(source, bytes):
        input_arg = "pipe:0"
        input_data: bytes | None = source
    else:
        input_arg = source
        input_data = None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-select_streams",
        "a:0",
        "-show_entries",
        f"stream={entry}",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_arg,
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, input=input_data, check=True, timeout=timeout
        )
        text = proc.stdout.decode().strip()
        if not text.isdigit():
            return None
        value = int(text)
        # ffprobe can report 0 for a hostile/broken stream; a non-positive count
        # or rate is "unknown", never forwarded (``-ar 0`` is a silent no-op on
        # some ffmpeg builds, which would skip resampling without any error).
        return value if value > 0 else None
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError:
        return None


def _probe_channels_with_ffprobe(source: str | bytes, timeout: float = 5.0) -> int | None:
    """Detect audio channel count via ffprobe (internal helper).

    Args:
        source: File path or raw bytes.
        timeout: Max seconds to wait. Default: ``5.0``.

    Returns:
        Number of channels, or ``None`` if ffprobe unavailable or detection failed.

    Raises:
        AudioProcessingError: If a string source is not a valid local file.
    """
    return _probe_stream_entry(source, "channels", timeout)


def _probe_sample_rate_with_ffprobe(source: str | bytes, timeout: float = 5.0) -> int | None:
    """Detect the native sample rate via ffprobe (internal helper).

    Args:
        source: File path or raw bytes.
        timeout: Max seconds to wait. Default: ``5.0``.

    Returns:
        Native sample rate in Hz, or ``None`` if ffprobe unavailable or detection
        failed.

    Raises:
        AudioProcessingError: If a string source is not a valid local file.
    """
    return _probe_stream_entry(source, "sample_rate", timeout)
