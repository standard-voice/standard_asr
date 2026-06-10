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
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .asr_properties import (
    AcceptedSampleRates,
    nearest_accepted_sample_rate,
    sample_rate_accepted,
)
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
from .utils.audio_loader import (
    _base64_payload,  # pyright: ignore[reportPrivateUsage]
    _decode_base64_payload,  # pyright: ignore[reportPrivateUsage]
    _estimate_payload_decoded_size,  # pyright: ignore[reportPrivateUsage]
    decode_audio,
)
from .utils.save_utils import encode_array_to_wav_bytes

#: Canonical fallback sample rate when a bare array omits its rate (spec R6).
ASSUMED_SAMPLE_RATE = 16000


def _empty_diagnostics() -> list[Diagnostic]:
    """Return an empty diagnostics list (typed factory for dataclass default).

    Returns:
        An empty list of diagnostics.
    """
    return []


_T = TypeVar("_T", bound=AudioInput)


def _narrow(provided: AudioInput, expected: type[_T]) -> _T:
    """Assert -- at runtime, not via ``assert`` -- that ``provided`` matches a plan.

    ``execute_plan`` takes ``provided`` and ``plan`` as two independent public
    arguments; a plan is only valid for the variant :func:`negotiate` built it
    from. The internal handlers narrow ``provided`` to the variant their plan
    op implies. A bare ``assert`` would be stripped under ``python -O``, so a
    mismatched ``plan``/``provided`` pair (direct misuse, not the standard
    pipeline) would degrade to an ``AttributeError`` or a wrong-shape delivery
    instead of a structured, contracted error (spec: error paths explicit). This
    raises :class:`AudioProcessingError` unconditionally on a mismatch.

    Args:
        provided: The application-provided audio input.
        expected: The variant the negotiated plan requires.

    Returns:
        ``provided`` narrowed to ``expected``.

    Raises:
        AudioProcessingError: If ``provided`` is not an instance of ``expected``
            (the plan was built for a different variant).
    """
    if not isinstance(provided, expected):
        raise AudioProcessingError(
            f"Conversion plan/provided mismatch: the plan was built for "
            f"{expected.__name__}, but the provided input is "
            f"{type(provided).__name__}. Pass the plan returned by negotiate() "
            "for this exact input."
        )
    return provided


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
    accepted: AcceptedSampleRates,
    native_sample_rate: int,
    required_input_sample_rate: int | None,
    source_sample_rate: int | None = None,
) -> int:
    """Choose a target sample rate for array delivery (spec R7.2).

    Selection policy (first match wins), per the normative spec R7.2 order:

    1. ``required_input_sample_rate`` if the engine accepts it (a hard wire
       requirement is authoritative).
    2. ``native_sample_rate`` if accepted (the model's own rate is ideal).
    3. **Defensive fallback only** -- per spec R7.2 this branch is unreachable
       through the standard engine pipeline, where ``BaseProperties`` enforces
       that ``required_input_sample_rate`` and ``native_sample_rate`` are both in
       the engine's ``accepted_sample_rates`` (the two reachability invariants,
       which hold for a discrete list and a ``SampleRateRange`` alike), so step 1
       or 2 always matches. It is reachable only when a caller invokes
       ``execute_plan`` directly with declarations that violate the invariants.
       It then picks an **explicit nearest-reachable** rate relative to the
       source: for a discrete list, the accepted rate closest in absolute
       distance to ``source_sample_rate``, preferring -- to honour R7's
       anti-upsampling spirit -- a rate that does **not** upsample (``<= source``)
       over one that does when both are equally near (deterministic and
       order-independent; the old ``accepted[0]`` could silently upsample, e.g.
       ``[48000, 16000]`` for 22050 Hz input picked 48000); for a
       ``SampleRateRange``, the source clamped into ``[min, max]``. When the
       source rate is unknown the smallest accepted rate (list) / the range
       minimum is chosen (minimises gratuitous upsampling).

    Args:
        accepted: The engine's accepted sample rates (list, range, or ``"any"``).
        native_sample_rate: The model's native sample rate.
        required_input_sample_rate: A hard-required rate, if any.
        source_sample_rate: The input waveform's current rate, used to pick the
            nearest reachable target. ``None`` when unknown.

    Returns:
        A sample rate the engine accepts.
    """
    if accepted == "any":
        return native_sample_rate
    if required_input_sample_rate is not None and sample_rate_accepted(
        accepted, required_input_sample_rate
    ):
        return required_input_sample_rate
    if sample_rate_accepted(accepted, native_sample_rate):
        return native_sample_rate
    if source_sample_rate is None:
        # No source reference: pick the smallest reachable rate to minimise
        # gratuitous upsampling, deterministically. (For "any" we returned native
        # above; here ``accepted`` is a list or range.)
        return min(accepted) if isinstance(accepted, list) else accepted.min
    # Nearest reachable (list: nearest member preferring not to upsample; range:
    # source clamped into [min, max]). Both via the shared helper.
    return nearest_accepted_sample_rate(accepted, source_sample_rate)


def execute_plan(
    provided: AudioInput,
    plan: ConversionPlan,
    *,
    accepted_sample_rates: AcceptedSampleRates,
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
        accepted_sample_rates: Engine accepted sample rates (a list, a
            :class:`~standard_asr.asr_properties.SampleRateRange`, or ``"any"``).
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
        url = _narrow(provided, AudioUrl)
        # R5.1: validate HTTPS + non-private address before forwarding the
        # literal URL to the engine. The standard never fetches it (v1).
        validate_fetchable_url(url.value, allow_private_addresses=allow_private_addresses)
        return PreparedAudio(kind=target, url=url.value, diagnostics=diags)

    if target is InputKind.STORAGE_URI:
        # The engine resolves the storage URI with its own cloud-SDK credentials;
        # the standard forwards the literal and runs no SSRF validator (the
        # scheme allowlist was already enforced at AudioStorageUri construction).
        storage = _narrow(provided, AudioStorageUri)
        return PreparedAudio(kind=target, storage_uri=storage.value, diagnostics=diags)

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
    _diagnose_non_finite(array, diags)
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
    accepted_sample_rates: AcceptedSampleRates,
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
        accepted_sample_rates: Engine accepted sample rates (a list, a
            :class:`~standard_asr.asr_properties.SampleRateRange`, or ``"any"``).
            The array-to-WAV (``ENCODE_WAV``) path resamples to an accepted rate
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
        # Gate the file fast-path on the PLAN's target, not the provided variant:
        # target=ENCODED_FILE is reachable only from an AudioPath source, so the
        # legitimate path is unchanged, but an AudioPath paired with an
        # ENCODED_BYTES plan must fall through to the _narrow mismatch guard below
        # rather than silently deliver an ENCODED_FILE.
        if plan.target_kind is InputKind.ENCODED_FILE:
            file_src = _narrow(provided, AudioPath)
            # File path passthrough: prefer stat() over reading the file (spec R9).
            _check_file_size(Path(file_src.value), max_file_size)
            return PreparedAudio(kind=InputKind.ENCODED_FILE, path=str(file_src.value))
        provided_bytes = _narrow(provided, AudioBytes)
        _check_payload_size(len(provided_bytes.data), max_file_size)
        return PreparedAudio(
            kind=InputKind.ENCODED_BYTES,
            data=provided_bytes.data,
            container=provided_bytes.container,
        )
    if ConversionOp.READ_FILE in ops:
        path = Path(_narrow(provided, AudioPath).value)
        # Precheck via stat() before reading the whole file into memory (spec R9).
        _check_file_size(path, max_file_size)
        return PreparedAudio(
            kind=InputKind.ENCODED_BYTES,
            data=_read_file_bytes(path),
            container=path.suffix.lstrip(".") or None,
        )
    if ConversionOp.ENCODE_WAV in ops:
        array_in = _narrow(provided, AudioArray)
        # R6/R7: resolve a missing rate (strict raises, best_effort assumes +
        # diagnoses) and resample the array to an accepted rate BEFORE encoding,
        # so an encoded-input engine that declares a restricted
        # accepted_sample_rates never receives off-rate WAV content. The bare
        # array path enforces the identical policy via the same helper.
        samples, sr = _apply_sample_rate(
            array_in.samples,
            array_in.sample_rate,
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
        if result.sanitized_non_finite:
            # The WAV encoder MUST sanitize NaN/Inf (int16 cannot represent them),
            # but that mutation MUST be visible to the caller -- the array-delivery
            # path emits the same ``non_finite_audio`` diagnostic, so encode-path
            # engines are not silently denied a signal the array path surfaces
            # (spec R3 / explicit > implicit).
            diags.append(
                Diagnostic(
                    level="warning",
                    code="non_finite_audio",
                    message=(
                        f"Sanitized {result.sanitized_non_finite} non-finite "
                        "sample(s) (NaN/Inf) to 0/+-1 during WAV encoding."
                    ),
                    param="audio",
                )
            )
        return PreparedAudio(kind=InputKind.ENCODED_BYTES, data=result.data, container="wav")
    if ConversionOp.B64_DECODE in ops:  # base64 -> bytes
        decoded = _decode_base64_bounded(_narrow(provided, AudioBase64).value, max_file_size)
        return PreparedAudio(kind=InputKind.ENCODED_BYTES, data=decoded)
    raise AudioProcessingError("Unsupported encoded conversion plan.")  # pragma: no cover


def _decode_base64_bounded(value: str, max_file_size: int | None) -> bytes:
    """Size-gate (pre-decode, spec R9) and decode a base64 payload in one step.

    The conversion layer's single base64 entry point, shared by the
    encoded-delivery and decode-to-array paths: the gate and the decode travel
    together, so a future third base64-accepting path cannot take the decode
    without the gate (re-opening R9). The decoded size is estimated from the
    payload length alone and checked BEFORE the decode allocates it; the
    estimate never exceeds the true decoded size, so an under-limit payload is
    never falsely rejected, and the exact post-decode check stays
    authoritative. The ``data:``-URI payload is extracted ONCE and shared by
    the estimate and the decode (separate top-level calls each re-ran the
    extraction: a duplicate O(n) scan plus, for a data: URI, a second
    payload-sized transient slice copy).

    Args:
        value: A ``data:...;base64,...`` URI or a bare base64 string.
        max_file_size: The engine's declared limit, or ``None`` for no limit.

    Returns:
        The decoded bytes (within ``max_file_size``).

    Raises:
        AudioProcessingError: If the estimated or exact decoded size exceeds
            ``max_file_size``, or the payload is malformed.
    """
    payload = _base64_payload(value)
    _check_payload_size(_estimate_payload_decoded_size(payload), max_file_size)
    decoded = _decode_base64_payload(payload)
    _check_payload_size(len(decoded), max_file_size)
    return decoded


def _diagnose_non_finite(array: NDArray[np.float32], diags: list[Diagnostic]) -> None:
    """Diagnose -- never sanitize -- non-finite samples in an array delivery.

    NaN/Inf in application-provided float audio is forwarded unchanged: clipping
    or zeroing here would silently mutate audio the application may have shaped
    deliberately, and the decode paths already sanitize their own output. The
    structured warning makes the condition visible to the caller instead of
    letting it degrade transcription silently (explicit > implicit).

    Args:
        array: The waveform about to be delivered (passthrough or resampled).
        diags: Diagnostics accumulator.
    """
    finite = np.isfinite(array)
    if bool(finite.all()):
        return
    bad = int(np.count_nonzero(~finite))
    diags.append(
        Diagnostic(
            level="warning",
            code="non_finite_audio",
            message=(
                f"Array delivery contains {bad} non-finite sample(s) (NaN/Inf); "
                "forwarded unchanged."
            ),
            param="audio",
        )
    )


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
    """Enforce an engine's ``max_file_size`` on an encoded payload (spec R4).

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
    """Enforce ``max_file_size`` against a file's size via ``stat`` (spec R9).

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


def _read_file_bytes(path: Path) -> bytes:
    """Read a local file into bytes, raising the contracted error on failure.

    The READ_FILE op only ran ``stat()`` when the engine declared a
    ``max_file_size`` (the size precheck). With no declared limit a missing or
    unreadable file would otherwise surface as a bare ``FileNotFoundError`` /
    ``OSError`` from ``read_bytes()`` -- outside ``execute_plan``'s documented
    ``AudioProcessingError`` / ``UnsafeAudioUrlError`` contract, and varying with
    engine metadata (a limit-declaring engine got ``AudioProcessingError`` for
    the same missing path). Wrap it so the failure type is engine-independent and
    contracted, and so the server's catch-all does not leak a raw filesystem
    error string.

    Args:
        path: The local file path to read.

    Returns:
        The file's bytes.

    Raises:
        AudioProcessingError: If the file cannot be read (missing, a directory,
            permission denied, etc.).
    """
    try:
        return path.read_bytes()
    except OSError as exc:
        raise AudioProcessingError(f"Cannot read audio file {str(path)!r}: {exc}.") from exc


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
        array_src = _narrow(provided, AudioArray)
        # np.asarray (not astype(copy=False)) per DEP.2.
        return np.asarray(array_src.samples, dtype=np.float32), array_src.sample_rate

    # Decode path: AudioPath / AudioBytes / AudioBase64 -> array.
    if isinstance(provided, AudioPath):
        # Hand a Path (never a str) to decode_audio so it takes the path-only
        # branch: no data:-URI content sniffing and no leading/trailing strip()
        # of the path (spec R1 -- discrimination MUST NOT sniff string content,
        # and a bare path is ALWAYS a local file). Passing str() would route an
        # AudioPath("data:audio/wav;base64,...") into base64 decoding (a silent
        # wrong result that fails loudly on an encoded-bytes engine instead) and
        # would strip a real "/tmp/x.wav " path down to a different file. The
        # Path branch raises an actionable "wrap base64 in AudioBase64" error.
        source: str | bytes | Path = Path(provided.value)
    elif isinstance(provided, AudioBytes):
        source = provided.data
    elif isinstance(provided, AudioBase64):
        # Same gate-and-decode as the encoded path (R9): the decoded size is
        # estimated and checked BEFORE the decode allocates it.
        source = _decode_base64_bounded(provided.value, max_file_size)
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
    accepted: AcceptedSampleRates,
    native_sample_rate: int,
    required_input_sample_rate: int | None,
    strict: bool,
    diags: list[Diagnostic],
) -> tuple[NDArray[np.float32], int]:
    """Apply the sample-rate rules (R6--R8) to an array payload.

    Args:
        array: The waveform array.
        sample_rate: Its sample rate, or ``None`` if unknown.
        accepted: Engine accepted sample rates (list, range, or ``"any"``).
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

    if required_input_sample_rate is not None and sample_rate != required_input_sample_rate:
        # A hard-required rate is authoritative for the batch path (spec R7.1):
        # always resample to it, even when accepted_sample_rates is "any" or
        # already contains the source rate. An engine that hard-requires a wire
        # rate must receive exactly it.
        target = required_input_sample_rate
    elif sample_rate_accepted(accepted, sample_rate):
        # Already accepted (in the list, inside the range, or "any") -> passthrough.
        return array, sample_rate
    else:
        target = _target_array_sample_rate(
            accepted, native_sample_rate, required_input_sample_rate, source_sample_rate=sample_rate
        )
    try:
        resampled, backend = resample_with_backend(array, sample_rate, target)
    except ValueError as exc:
        # resample_with_backend raises a bare ValueError for an empty array (a
        # non-positive rate is already blocked at AudioArray construction). Wrap
        # it so execute_plan only ever raises its documented AudioProcessingError
        # / UnsafeAudioUrlError -- a direct caller handing in an empty array gets
        # the contracted error instead of a bare ValueError that escapes the
        # transcribe() Raises contract and maps to a 500 on the server path.
        raise AudioProcessingError(
            f"Cannot resample the audio to {target} Hz: {exc} Provide a non-empty waveform."
        ) from exc
    if backend == "fallback":
        # Design decision D3 / spec R8: when the low-quality built-in fallback
        # resampler runs (because the [audio] extra is absent), the quality
        # degradation MUST be visible at WARNING level with an install hint --
        # consumers that filter diagnostics by ``level >= warning`` (a reasonable
        # default) would otherwise miss it. The structured ``effective`` field
        # still carries the machine-readable backend id. The scipy path stays
        # informational (no degradation). This mirrors normalize_audio's
        # logger.warning in the loader path.
        diags.append(
            Diagnostic(
                level="warning",
                code="resampled_with",
                message=(
                    f"Resampled {sample_rate} Hz -> {target} Hz with the built-in "
                    "fallback resampler. Install standard-asr[audio] for a "
                    "higher-quality resampler."
                ),
                param="audio",
                provided=f"{sample_rate}->{target}",
                effective=backend,
            )
        )
    else:
        diags.append(
            Diagnostic(
                level="info",
                code="resampled_with",
                message=f"Resampled {sample_rate} Hz -> {target} Hz (scipy resample_poly).",
                param="audio",
                # The rate transition lives in ``provided`` and the structured
                # ``effective`` carries the *backend* identifier, so the spec R8
                # contract reads as ``resampled_with=<scipy|fallback>`` without any
                # English prose parsing -- a cross-language/REST client can detect
                # the low-quality numpy fallback from the structured field alone.
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
