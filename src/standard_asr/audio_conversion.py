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

import base64
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
    AudioUrl,
    InputKind,
)
from .audio_negotiation import ConversionOp, ConversionPlan
from .exceptions import AudioProcessingError
from .resampling import resample
from .results import Diagnostic
from .utils.audio_loader import load_audio
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
    """

    kind: InputKind
    array: NDArray[np.float32] | None = None
    sample_rate: int | None = None
    data: bytes | None = None
    container: str | None = None
    path: str | None = None
    url: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=_empty_diagnostics)


def _target_array_sample_rate(
    accepted: list[int] | str,
    native_sample_rate: int,
    required_input_sample_rate: int | None,
) -> int:
    """Choose a target sample rate for array delivery.

    Args:
        accepted: The engine's accepted sample rates, or ``"any"``.
        native_sample_rate: The model's native sample rate.
        required_input_sample_rate: A hard-required rate, if any.

    Returns:
        A sample rate the engine accepts.
    """
    if not isinstance(accepted, list):
        return native_sample_rate
    if required_input_sample_rate is not None and required_input_sample_rate in accepted:
        return required_input_sample_rate
    if native_sample_rate in accepted:
        return native_sample_rate
    return accepted[0]


def _decode_b64(value: str) -> bytes:
    """Decode a base64 string or ``data:`` URI into bytes.

    Args:
        value: A base64 payload, optionally a ``data:...;base64,...`` URI.

    Returns:
        The decoded bytes.

    Raises:
        AudioProcessingError: If the payload is not valid base64.
    """
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise AudioProcessingError("Invalid base64 audio payload.") from exc


def execute_plan(
    provided: AudioInput,
    plan: ConversionPlan,
    *,
    accepted_sample_rates: list[int] | str,
    native_sample_rate: int,
    required_input_sample_rate: int | None = None,
    max_file_size: int | None = None,
    strict: bool = True,
) -> PreparedAudio:
    """Execute a conversion plan, returning engine-ready prepared audio.

    Args:
        provided: The application-provided audio input.
        plan: The negotiated conversion plan.
        accepted_sample_rates: Engine accepted sample rates, or ``"any"``.
        native_sample_rate: The model's native sample rate.
        required_input_sample_rate: A hard-required rate, if any.
        max_file_size: Engine max payload size for the WAV-encode precheck.
        strict: Whether to raise (vs assume + diagnostic) on a missing rate.

    Returns:
        The :class:`PreparedAudio` with any conversion diagnostics attached.

    Raises:
        AudioProcessingError: On a missing sample rate in strict mode, an
            oversize encode, or a decode failure.
    """
    diags: list[Diagnostic] = []
    ops = plan.operations
    target = plan.target_kind

    if target is InputKind.FETCHABLE_URL:
        assert isinstance(provided, AudioUrl)
        return PreparedAudio(kind=target, url=provided.value, diagnostics=diags)

    if target in (InputKind.ENCODED_FILE, InputKind.ENCODED_BYTES):
        prepared = _prepare_encoded(provided, plan, max_file_size, diags)
        prepared.diagnostics = diags
        return prepared

    # target is ARRAY
    array, sample_rate = _prepare_array(provided, ops, diags)
    array, sample_rate = _apply_sample_rate(
        array,
        sample_rate,
        accepted_sample_rates,
        native_sample_rate,
        required_input_sample_rate,
        strict,
        diags,
    )
    return PreparedAudio(
        kind=InputKind.ARRAY,
        array=array,
        sample_rate=sample_rate,
        diagnostics=diags,
    )


def _prepare_encoded(
    provided: AudioInput,
    plan: ConversionPlan,
    max_file_size: int | None,
    diags: list[Diagnostic],
) -> PreparedAudio:
    """Prepare an encoded (file/bytes) payload from the provided input.

    Args:
        provided: The provided audio input.
        plan: The conversion plan (target is file/bytes).
        max_file_size: Engine max payload size for the WAV-encode precheck.
        diags: Diagnostics accumulator.

    Returns:
        Prepared encoded audio.
    """
    ops = plan.operations
    if ConversionOp.PASSTHROUGH in ops:
        if isinstance(provided, AudioPath):
            return PreparedAudio(kind=InputKind.ENCODED_FILE, path=str(provided.value))
        assert isinstance(provided, AudioBytes)
        return PreparedAudio(
            kind=InputKind.ENCODED_BYTES,
            data=provided.data,
            container=provided.container,
        )
    if ConversionOp.READ_FILE in ops:
        assert isinstance(provided, AudioPath)
        path = Path(provided.value)
        return PreparedAudio(
            kind=InputKind.ENCODED_BYTES,
            data=path.read_bytes(),
            container=path.suffix.lstrip(".") or None,
        )
    if ConversionOp.ENCODE_WAV in ops:
        assert isinstance(provided, AudioArray)
        sr = provided.sample_rate or ASSUMED_SAMPLE_RATE
        if provided.sample_rate is None:
            diags.append(_assumed_sample_rate_diag())
        result = encode_array_to_wav_bytes(
            provided.samples, sr, max_file_size=max_file_size
        )
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
        return PreparedAudio(
            kind=InputKind.ENCODED_BYTES, data=result.data, container="wav"
        )
    if ConversionOp.B64_DECODE in ops:  # base64 -> bytes
        assert isinstance(provided, AudioBase64)
        return PreparedAudio(
            kind=InputKind.ENCODED_BYTES, data=_decode_b64(provided.value)
        )
    raise AudioProcessingError("Unsupported encoded conversion plan.")  # pragma: no cover


def _prepare_array(
    provided: AudioInput,
    ops: tuple[ConversionOp, ...],
    diags: list[Diagnostic],
) -> tuple[NDArray[np.float32], int | None]:
    """Produce a waveform array from the provided input.

    Args:
        provided: The provided audio input.
        ops: The plan operations.
        diags: Diagnostics accumulator.

    Returns:
        A ``(array, sample_rate)`` pair; ``sample_rate`` may be ``None`` for a
        bare array that omitted its rate.
    """
    if ConversionOp.PASSTHROUGH in ops:
        assert isinstance(provided, AudioArray)
        return provided.samples.astype(np.float32, copy=False), provided.sample_rate

    # Decode path: AudioPath / AudioBytes / AudioBase64 -> array.
    if isinstance(provided, AudioPath):
        source: str | bytes = str(provided.value)
    elif isinstance(provided, AudioBytes):
        source = provided.data
    elif isinstance(provided, AudioBase64):
        source = _decode_b64(provided.value)
    else:  # pragma: no cover - matrix guarantees the above
        raise AudioProcessingError("Cannot decode this input to an array.")

    # Decode preserving the native rate; let the sample-rate stage resample.
    array = load_audio(source, target_sr=ASSUMED_SAMPLE_RATE, target_channels=1)
    diags.append(
        Diagnostic(
            level="info",
            code="audio_conversion",
            message="Decoded encoded audio to a waveform array.",
            param="audio",
            effective="array",
        )
    )
    # load_audio already produced 16 kHz mono; treat that as the known rate.
    return array, ASSUMED_SAMPLE_RATE


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
        accepted, native_sample_rate, required_input_sample_rate
    )
    resampled = resample(array, sample_rate, target)
    diags.append(
        Diagnostic(
            level="info",
            code="resampled_with",
            message=f"Resampled {sample_rate} Hz -> {target} Hz (fallback).",
            param="audio",
            provided=sample_rate,
            effective=target,
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
