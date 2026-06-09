# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Execute audio conversion plans into engine-ready prepared audio.

The engine base layer negotiates a :class:`~standard_asr.audio_negotiation.ConversionPlan`
and then asks this module to *execute* it -- decoding, encoding, reading,
base64-decoding and resampling as required -- producing a :class:`PreparedAudio`
in exactly one of the engine's accepted shapes, plus a list of
:class:`~standard_asr.results.Diagnostic` describing any lossy or assumed steps
(spec, section "Audio Input & Sample Rate", rules R3/R4/R6/R7/R8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .audio_input import (
    AudioArray,
    AudioBase64,
    AudioBytes,
    AudioInput,
    AudioPath,
    AudioStorageUri,
    AudioUrl,
    InputKind,
)
from .audio_negotiation import ConversionOp, ConversionPlan, validate_fetchable_url
from .exceptions import AudioProcessingError
from .resampling import resample_with_backend
from .results import Diagnostic
from .utils.audio_loader import decode_audio, decode_base64_audio
from .utils.save_utils import encode_array_to_wav_bytes

#: Canonical fallback sample rate when a bare array omits its rate (spec R6).
ASSUMED_SAMPLE_RATE = 16000


def _empty_diagnostics() -> list[Diagnostic]:
    """Return an empty diagnostics list (typed factory for dataclass default).

    Returns:
        An empty list of diagnostics.
    """
    return []


@dataclass
class PreparedAudio:
    """Audio negotiated into exactly one engine-accepted shape.

    Exactly one payload slot is populated according to :attr:`kind`.

    Args:
        kind: The accepted shape this payload represents.
        array: Waveform (for ``ARRAY``).
        sample_rate: Sample rate of ``array`` in Hz (for ``ARRAY``).
        data: Encoded bytes (for ``ENCODED_BYTES``).
        container: Optional container hint for ``data``.
        path: File path (for ``ENCODED_FILE``).
        url: Remote URL (for ``FETCHABLE_URL``).
        storage_uri: Provider cloud-storage URI (for ``STORAGE_URI``).
    """

    kind: InputKind
    array: NDArray[np.float32] | None = None
    sample_rate: int | None = None
    data: bytes | None = None
    container: str | None = None
    path: str | None = None
    url: str | None = None
    storage_uri: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=_empty_diagnostics)


def _target_array_sample_rate(
    accepted: list[int] | str,
    native_sample_rate: int,
    required_input_sample_rate: int | None,
    source_sample_rate: int | None = None,
) -> int:
    """Choose a target sample rate for array delivery.

    Selection policy (first match wins):

    1. ``required_input_sample_rate`` if the engine accepts it (a hard wire
       requirement is authoritative).
    2. ``native_sample_rate`` if accepted (the model's own rate is ideal).
    3. Otherwise an **explicit nearest-reachable** rate relative to the source
       (RESA-3): the accepted rate closest in absolute distance to
       ``source_sample_rate``, and -- to honour R7's anti-upsampling spirit --
       preferring a rate that does **not** upsample (``<= source``) over one
       that does when both are equally near. This is deterministic and
       independent of the order in which ``accepted_sample_rates`` was declared
       (the old ``accepted[0]`` could silently upsample, e.g. ``[48000, 16000]``
       for 22050 Hz input picked 48000). When the source rate is unknown the
       smallest accepted rate is chosen (minimises gratuitous upsampling).

    Args:
        accepted: The engine's accepted sample rates, or ``"any"``.
        native_sample_rate: The model's native sample rate.
        required_input_sample_rate: A hard-required rate, if any.
        source_sample_rate: The input waveform's current rate, used to pick the
            nearest reachable target. ``None`` when unknown.

    Returns:
        A sample rate the engine accepts.
    """
    if not isinstance(accepted, list):
        return native_sample_rate
    if required_input_sample_rate is not None and required_input_sample_rate in accepted:
        return required_input_sample_rate
    if native_sample_rate in accepted:
        return native_sample_rate
    if source_sample_rate is None:
        # No source reference: pick the smallest accepted rate to minimise
        # gratuitous upsampling, deterministically (order-independent).
        return min(accepted)
    # Nearest reachable, preferring not to upsample: sort by (distance, would-
    # upsample) so an equally-near non-upsampling rate wins, and a non-upsampling
    # rate beats a farther upsampling one only when nearer.
    return min(
        accepted,
        key=lambda rate: (abs(rate - source_sample_rate), rate > source_sample_rate),
    )


def execute_plan(
    provided: AudioInput,
    plan: ConversionPlan,
    *,
    accepted_sample_rates: list[int] | str,
    native_sample_rate: int,
    required_input_sample_rate: int | None = None,
    max_file_size: int | None = None,
    max_audio_duration: float | None = None,
    strict: bool = True,
    allow_private_addresses: bool = False,
) -> PreparedAudio:
    """Execute a conversion plan, returning engine-ready prepared audio.

    Args:
        provided: The application-provided audio input.
        plan: The negotiated conversion plan.
        accepted_sample_rates: Engine accepted sample rates, or ``"any"``.
        native_sample_rate: The model's native sample rate.
        required_input_sample_rate: A hard-required rate, if any.
        max_file_size: Engine max payload size; prechecked on every encoded
            payload and used to bound the decode buffer (spec R4/R9).
        max_audio_duration: Engine max accepted duration in seconds, if any.
            Enforced on the decoded array (where duration is measurable);
            encoded passthrough relies on ``max_file_size`` instead.
        strict: Whether to raise (vs assume + diagnostic) on a missing rate.
        allow_private_addresses: Opt-in to relax the R5 SSRF check that rejects
            URLs resolving to private/loopback/link-local addresses. HTTPS is
            still required.

    Returns:
        The :class:`PreparedAudio` with any conversion diagnostics attached.

    Raises:
        AudioProcessingError: On a missing sample rate in strict mode, an
            oversize encode/payload, or a decode failure.
        UnsafeAudioUrlError: When a ``FETCHABLE_URL`` target fails the R5 SSRF
            policy (not HTTPS, or a private/reserved address).
    """
    diags: list[Diagnostic] = []
    ops = plan.operations
    target = plan.target_kind

    if target is InputKind.FETCHABLE_URL:
        assert isinstance(provided, AudioUrl)
        # R5.1: validate HTTPS + non-private address before forwarding the
        # literal URL to the engine. The standard never fetches it (v1).
        validate_fetchable_url(provided.value, allow_private_addresses=allow_private_addresses)
        return PreparedAudio(kind=target, url=provided.value, diagnostics=diags)

    if target is InputKind.STORAGE_URI:
        # The engine resolves the storage URI with its own cloud-SDK credentials;
        # the standard forwards the literal and runs no SSRF validator (the
        # scheme allowlist was already enforced at AudioStorageUri construction).
        assert isinstance(provided, AudioStorageUri)
        return PreparedAudio(kind=target, storage_uri=provided.value, diagnostics=diags)

    if target in (InputKind.ENCODED_FILE, InputKind.ENCODED_BYTES):
        prepared = _prepare_encoded(
            provided,
            plan,
            accepted_sample_rates=accepted_sample_rates,
            native_sample_rate=native_sample_rate,
            required_input_sample_rate=required_input_sample_rate,
            max_file_size=max_file_size,
            max_audio_duration=max_audio_duration,
            strict=strict,
            diags=diags,
        )
        prepared.diagnostics = diags
        return prepared

    # target is ARRAY
    array, sample_rate = _prepare_array(provided, ops, max_file_size, diags)
    array, sample_rate = _apply_sample_rate(
        array,
        sample_rate,
        accepted_sample_rates,
        native_sample_rate,
        required_input_sample_rate,
        strict,
        diags,
    )
    _check_duration(array, sample_rate, max_audio_duration)
    return PreparedAudio(
        kind=InputKind.ARRAY,
        array=array,
        sample_rate=sample_rate,
        diagnostics=diags,
    )


def _prepare_encoded(
    provided: AudioInput,
    plan: ConversionPlan,
    *,
    accepted_sample_rates: list[int] | str,
    native_sample_rate: int,
    required_input_sample_rate: int | None,
    max_file_size: int | None,
    max_audio_duration: float | None,
    strict: bool,
    diags: list[Diagnostic],
) -> PreparedAudio:
    """Prepare an encoded (file/bytes) payload from the provided input.

    Args:
        provided: The provided audio input.
        plan: The conversion plan (target is file/bytes).
        accepted_sample_rates: Engine accepted sample rates, or ``"any"``. The
            array-to-WAV (``ENCODE_WAV``) path resamples to an accepted rate
            before encoding, so an encoded-input engine never receives off-rate
            WAV content (spec R7).
        native_sample_rate: The model's native sample rate.
        required_input_sample_rate: A hard-required rate, if any.
        max_file_size: Engine max payload size for the WAV-encode precheck.
        max_audio_duration: Engine max accepted duration in seconds, enforced on
            the ``ENCODE_WAV`` array (where duration is measurable).
        strict: Whether to raise (vs assume + diagnostic) when a bare array has
            no sample rate before encoding it to WAV (spec R6).
        diags: Diagnostics accumulator.

    Returns:
        Prepared encoded audio.
    """
    ops = plan.operations
    if ConversionOp.PASSTHROUGH in ops:
        if isinstance(provided, AudioPath):
            # File path passthrough: prefer stat() over reading the file (R9/H9).
            _check_file_size(Path(provided.value), max_file_size)
            return PreparedAudio(kind=InputKind.ENCODED_FILE, path=str(provided.value))
        assert isinstance(provided, AudioBytes)
        _check_payload_size(len(provided.data), max_file_size)
        return PreparedAudio(
            kind=InputKind.ENCODED_BYTES,
            data=provided.data,
            container=provided.container,
        )
    if ConversionOp.READ_FILE in ops:
        assert isinstance(provided, AudioPath)
        path = Path(provided.value)
        # Precheck via stat() before reading the whole file into memory (R9/H9).
        _check_file_size(path, max_file_size)
        return PreparedAudio(
            kind=InputKind.ENCODED_BYTES,
            data=path.read_bytes(),
            container=path.suffix.lstrip(".") or None,
        )
    if ConversionOp.ENCODE_WAV in ops:
        assert isinstance(provided, AudioArray)
        # R6/R7: resolve a missing rate (strict raises, best_effort assumes +
        # diagnoses) and resample the array to an accepted rate BEFORE encoding,
        # so an encoded-input engine that declares a restricted
        # accepted_sample_rates never receives off-rate WAV content. The bare
        # array path enforces the identical policy via the same helper.
        samples, sr = _apply_sample_rate(
            provided.samples,
            provided.sample_rate,
            accepted_sample_rates,
            native_sample_rate,
            required_input_sample_rate,
            strict,
            diags,
        )
        _check_duration(samples, sr, max_audio_duration)
        result = encode_array_to_wav_bytes(samples, sr, max_file_size=max_file_size)
        diags.append(
            Diagnostic(
                level="warning",
                code="audio_conversion",
                message="Encoded array to WAV/16-bit PCM (lossy float->int16).",
                param="audio",
                provided="array",
                effective="encoded_bytes",
            )
        )
        if result.downmixed:
            diags.append(
                Diagnostic(
                    level="warning",
                    code="audio_conversion",
                    message="Downmixed multi-channel audio to mono for encoding.",
                    param="audio",
                )
            )
        return PreparedAudio(kind=InputKind.ENCODED_BYTES, data=result.data, container="wav")
    if ConversionOp.B64_DECODE in ops:  # base64 -> bytes
        assert isinstance(provided, AudioBase64)
        decoded = decode_base64_audio(provided.value)
        _check_payload_size(len(decoded), max_file_size)
        return PreparedAudio(kind=InputKind.ENCODED_BYTES, data=decoded)
    raise AudioProcessingError("Unsupported encoded conversion plan.")  # pragma: no cover


def _check_duration(
    array: NDArray[np.float32], sample_rate: int, max_audio_duration: float | None
) -> None:
    """Enforce an engine's ``max_audio_duration`` on a decoded array (spec R10).

    Enforced here, where the sample count and rate are both known, so a declared
    duration limit is an actual contract rather than advisory metadata. Encoded
    passthrough (where duration is not measurable without a full decode) relies
    on ``max_file_size`` instead.

    Args:
        array: The decoded waveform (``(n_samples,)`` or ``(n_samples, ch)``).
        sample_rate: The array's sample rate in Hz.
        max_audio_duration: The engine's limit in seconds, or ``None``.

    Raises:
        AudioProcessingError: If the duration exceeds ``max_audio_duration``.
    """
    if max_audio_duration is None:
        return
    duration = array.shape[0] / sample_rate
    if duration > max_audio_duration:
        raise AudioProcessingError(
            f"Audio duration is {duration:.3f}s, which exceeds the engine's "
            f"max_audio_duration of {max_audio_duration}s. Provide a shorter "
            "clip or use an engine without this limit."
        )


def _check_payload_size(num_bytes: int, max_file_size: int | None) -> None:
    """Enforce an engine's ``max_file_size`` on an encoded payload (spec R4/H9).

    Args:
        num_bytes: Size of the encoded payload in bytes.
        max_file_size: The engine's declared limit, or ``None`` for no limit.

    Raises:
        AudioProcessingError: If the payload exceeds ``max_file_size``.
    """
    if max_file_size is not None and num_bytes > max_file_size:
        raise AudioProcessingError(
            f"Encoded audio is {num_bytes} bytes, which exceeds the engine's "
            f"max_file_size of {max_file_size} bytes. Provide a shorter clip or "
            "use an engine without this limit."
        )


def _check_file_size(path: Path, max_file_size: int | None) -> None:
    """Enforce ``max_file_size`` against a file's size via ``stat`` (spec R9/H9).

    Args:
        path: The local file path.
        max_file_size: The engine's declared limit, or ``None`` for no limit.

    Raises:
        AudioProcessingError: If the file is missing or exceeds ``max_file_size``.
    """
    if max_file_size is None:
        return
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AudioProcessingError(f"Cannot stat audio file {str(path)!r}: {exc}.") from exc
    _check_payload_size(size, max_file_size)


def _prepare_array(
    provided: AudioInput,
    ops: tuple[ConversionOp, ...],
    max_file_size: int | None,
    diags: list[Diagnostic],
) -> tuple[NDArray[np.float32], int | None]:
    """Produce a waveform array from the provided input.

    The decode path returns the source's **native** sample rate -- it does NOT
    resample. The single authoritative R7 resampling decision is made later by
    :func:`_apply_sample_rate`, so 8 kHz telephony and 24 kHz realtime inputs are
    not silently forced through 16 kHz (spec R7).

    Args:
        provided: The provided audio input.
        ops: The plan operations.
        max_file_size: Engine payload limit, used to bound the decode buffer (R9).
        diags: Diagnostics accumulator.

    Returns:
        A ``(array, sample_rate)`` pair; ``sample_rate`` may be ``None`` for a
        bare array that omitted its rate.
    """
    if ConversionOp.PASSTHROUGH in ops:
        assert isinstance(provided, AudioArray)
        # np.asarray (not astype(copy=False)) per DEP.2.
        return np.asarray(provided.samples, dtype=np.float32), provided.sample_rate

    # Decode path: AudioPath / AudioBytes / AudioBase64 -> array.
    if isinstance(provided, AudioPath):
        source: str | bytes = str(provided.value)
    elif isinstance(provided, AudioBytes):
        source = provided.data
    elif isinstance(provided, AudioBase64):
        source = decode_base64_audio(provided.value)
    else:  # pragma: no cover - matrix guarantees the above
        raise AudioProcessingError("Cannot decode this input to an array.")

    # Decode at the NATIVE rate; the sample-rate stage owns any resampling (R7).
    array, native_sr = decode_audio(source, target_channels=1, max_bytes=max_file_size)
    diags.append(
        Diagnostic(
            level="info",
            code="audio_conversion",
            message=f"Decoded encoded audio to a waveform array at {native_sr} Hz.",
            param="audio",
            effective="array",
        )
    )
    return array, native_sr


def _apply_sample_rate(
    array: NDArray[np.float32],
    sample_rate: int | None,
    accepted: list[int] | str,
    native_sample_rate: int,
    required_input_sample_rate: int | None,
    strict: bool,
    diags: list[Diagnostic],
) -> tuple[NDArray[np.float32], int]:
    """Apply the sample-rate rules (R6--R8) to an array payload.

    Args:
        array: The waveform array.
        sample_rate: Its sample rate, or ``None`` if unknown.
        accepted: Engine accepted sample rates, or ``"any"``.
        native_sample_rate: The model's native sample rate.
        required_input_sample_rate: A hard-required rate, if any.
        strict: Whether to raise (vs assume) on a missing rate.
        diags: Diagnostics accumulator.

    Returns:
        A ``(array, sample_rate)`` pair at an accepted rate.

    Raises:
        AudioProcessingError: If the rate is missing in strict mode.
    """
    if sample_rate is None:
        if strict:
            raise AudioProcessingError(
                "Audio array has no sample rate. Pass "
                "AudioArray(samples, sample_rate) or enable best_effort."
            )
        sample_rate = ASSUMED_SAMPLE_RATE
        diags.append(_assumed_sample_rate_diag())

    if not isinstance(accepted, list) or sample_rate in accepted:
        return array, sample_rate

    target = _target_array_sample_rate(
        accepted, native_sample_rate, required_input_sample_rate, source_sample_rate=sample_rate
    )
    resampled, backend = resample_with_backend(array, sample_rate, target)
    label = "built-in fallback" if backend == "fallback" else "scipy resample_poly"
    diags.append(
        Diagnostic(
            level="info",
            code="resampled_with",
            message=f"Resampled {sample_rate} Hz -> {target} Hz ({label}).",
            param="audio",
            # The rate transition lives in ``provided`` and the structured
            # ``effective`` carries the *backend* identifier, so the spec R8
            # contract reads as ``resampled_with=<scipy|fallback>`` without any
            # English prose parsing -- a cross-language/REST client can detect the
            # low-quality numpy fallback from the structured field alone.
            provided=f"{sample_rate}->{target}",
            effective=backend,
        )
    )
    return resampled, target


def _assumed_sample_rate_diag() -> Diagnostic:
    """Build the diagnostic emitted when a sample rate is assumed.

    Returns:
        The ``assumed_sample_rate`` diagnostic.
    """
    return Diagnostic(
        level="warning",
        code="assumed_sample_rate",
        message=f"No sample rate provided; assumed {ASSUMED_SAMPLE_RATE} Hz.",
        param="audio",
        effective=ASSUMED_SAMPLE_RATE,
    )


__all__ = ["ASSUMED_SAMPLE_RATE", "PreparedAudio", "execute_plan"]
