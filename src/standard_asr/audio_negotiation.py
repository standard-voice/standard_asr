# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Audio input negotiation between application-provided and engine-accepted shapes.

The standard layer negotiates a deterministic, lowest-cost conversion path
between the :data:`~standard_asr.audio_input.AudioInput` variant an application
provides and the set of :class:`~standard_asr.audio_input.InputKind` shapes an
engine accepts. When the provided shape is already accepted, the path is a
zero-cost passthrough. When no path exists, negotiation reports a
:class:`NoViablePath`; the engine layer turns that into an
:class:`~standard_asr.exceptions.IncompatibleAudioInputError` at call time.

This module implements the normative conversion matrix (spec, section
"Audio Input & Sample Rate", rule R3). Sample-rate decisions (R6--R8) are
layered on top by the engine base class and are not part of *kind* negotiation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .audio_input import (
    AudioArray,
    AudioBase64,
    AudioBytes,
    AudioInput,
    AudioPath,
    AudioUrl,
    InputKind,
)
from .exceptions import IncompatibleAudioInputError

#: Engine-accepted shapes treated as the single "encoded" matrix column.
_ENCODED_KINDS = frozenset({InputKind.ENCODED_FILE, InputKind.ENCODED_BYTES})


class ConversionOp(str, Enum):
    """A single deterministic step in an audio conversion plan.

    Attributes:
        PASSTHROUGH: No transformation -- the provided shape is delivered as-is.
        READ_FILE: Read a local file into encoded bytes.
        DECODE: Decode encoded audio into a waveform array (needs ``[audio]``).
        ENCODE_WAV: Encode a waveform array into WAV/16-bit PCM bytes (lossy).
        B64_DECODE: Decode a base64 / data-URI payload into encoded bytes.
    """

    PASSTHROUGH = "passthrough"
    READ_FILE = "read_file"
    DECODE = "decode"
    ENCODE_WAV = "encode_wav"
    B64_DECODE = "b64_decode"


@dataclass(frozen=True)
class ConversionPlan:
    """A viable plan for delivering provided audio to an engine.

    Args:
        source_type: Name of the provided variant (e.g. ``"AudioPath"``).
        target_kind: The :class:`InputKind` the engine receives.
        operations: Ordered conversion steps to apply.
        lossy: Whether the conversion loses information (e.g. float32->int16).
        requires_audio_extra: Whether the conversion needs the ``[audio]`` extra
            (decoding compressed formats).
    """

    source_type: str
    target_kind: InputKind
    operations: tuple[ConversionOp, ...]
    lossy: bool
    requires_audio_extra: bool

    @property
    def is_passthrough(self) -> bool:
        """Return whether the plan applies no transformation.

        Returns:
            ``True`` if the provided shape is delivered without conversion.
        """
        return self.operations == (ConversionOp.PASSTHROUGH,)


@dataclass(frozen=True)
class NoViablePath:
    """The result of negotiation when no conversion path exists.

    Args:
        source_type: Name of the provided variant.
        accepted: The engine's accepted input kinds.
        hint: Actionable guidance for resolving the mismatch.
    """

    source_type: str
    accepted: frozenset[InputKind]
    hint: str


def negotiate(
    provided: AudioInput, accepted: set[InputKind] | frozenset[InputKind]
) -> ConversionPlan | NoViablePath:
    """Negotiate a conversion path from a provided shape to an accepted one.

    Implements the normative conversion matrix. Preference order within a
    variant is: zero-cost passthrough, then non-lossy conversion, then lossy
    conversion. Sample-rate handling is applied separately by the engine layer.

    Args:
        provided: The :data:`AudioInput` variant the application provided.
        accepted: The :class:`InputKind` shapes the engine accepts.

    Returns:
        A :class:`ConversionPlan` if a path exists, otherwise a
        :class:`NoViablePath`.
    """
    accepted = frozenset(accepted)
    source = type(provided).__name__

    if isinstance(provided, AudioArray):
        return _negotiate_array(source, accepted)
    if isinstance(provided, AudioPath):
        return _negotiate_path(source, accepted)
    if isinstance(provided, AudioBytes):
        return _negotiate_bytes(source, accepted)
    if isinstance(provided, AudioBase64):
        return _negotiate_base64(source, accepted)
    if isinstance(provided, AudioUrl):
        return _negotiate_url(source, accepted)
    # Unreachable for a well-formed AudioInput union.
    return NoViablePath(  # pragma: no cover
        source, accepted, "Unknown audio input variant."
    )


def can_accept(
    provided: AudioInput, accepted: set[InputKind] | frozenset[InputKind]
) -> bool:
    """Return whether the provided audio can be delivered to the engine.

    A pre-call determination helper: ``True`` iff :func:`negotiate` yields a
    :class:`ConversionPlan`.

    Args:
        provided: The provided :data:`AudioInput` variant.
        accepted: The engine's accepted input kinds.

    Returns:
        ``True`` if a viable conversion path exists.
    """
    return isinstance(negotiate(provided, accepted), ConversionPlan)


def negotiate_or_raise(
    provided: AudioInput, accepted: set[InputKind] | frozenset[InputKind]
) -> ConversionPlan:
    """Negotiate a conversion path or raise on failure.

    Args:
        provided: The provided :data:`AudioInput` variant.
        accepted: The engine's accepted input kinds.

    Returns:
        The viable :class:`ConversionPlan`.

    Raises:
        IncompatibleAudioInputError: If no viable path exists.
    """
    result = negotiate(provided, accepted)
    if isinstance(result, NoViablePath):
        raise IncompatibleAudioInputError(
            provided=result.source_type,
            accepted=sorted(k.value for k in result.accepted),
            hint=result.hint,
        )
    return result


def _negotiate_array(source: str, accepted: frozenset[InputKind]) -> ConversionPlan | NoViablePath:
    """Negotiate a path for an :class:`AudioArray` source.

    Args:
        source: Source variant name.
        accepted: Engine-accepted input kinds.

    Returns:
        A plan (passthrough to array, or lossy WAV encode), or no viable path.
    """
    if InputKind.ARRAY in accepted:
        return ConversionPlan(
            source, InputKind.ARRAY, (ConversionOp.PASSTHROUGH,), False, False
        )
    if accepted & _ENCODED_KINDS:
        # Encode to WAV/16-bit PCM bytes (float32 -> int16 is lossy).
        return ConversionPlan(
            source, InputKind.ENCODED_BYTES, (ConversionOp.ENCODE_WAV,), True, False
        )
    return NoViablePath(
        source,
        accepted,
        "Provide a local file via AudioPath, or use an engine that accepts arrays.",
    )


def _negotiate_path(source: str, accepted: frozenset[InputKind]) -> ConversionPlan | NoViablePath:
    """Negotiate a path for an :class:`AudioPath` source.

    Args:
        source: Source variant name.
        accepted: Engine-accepted input kinds.

    Returns:
        A plan (passthrough file, read-to-bytes, or decode), or no viable path.
    """
    if InputKind.ENCODED_FILE in accepted:
        return ConversionPlan(
            source, InputKind.ENCODED_FILE, (ConversionOp.PASSTHROUGH,), False, False
        )
    if InputKind.ENCODED_BYTES in accepted:
        return ConversionPlan(
            source, InputKind.ENCODED_BYTES, (ConversionOp.READ_FILE,), False, False
        )
    if InputKind.ARRAY in accepted:
        return ConversionPlan(
            source, InputKind.ARRAY, (ConversionOp.DECODE,), False, True
        )
    return NoViablePath(
        source, accepted, "The standard does not synthesize a fetchable URL in v1."
    )


def _negotiate_bytes(source: str, accepted: frozenset[InputKind]) -> ConversionPlan | NoViablePath:
    """Negotiate a path for an :class:`AudioBytes` source.

    Args:
        source: Source variant name.
        accepted: Engine-accepted input kinds.

    Returns:
        A plan (passthrough bytes or decode), or no viable path.
    """
    if accepted & _ENCODED_KINDS:
        return ConversionPlan(
            source, InputKind.ENCODED_BYTES, (ConversionOp.PASSTHROUGH,), False, False
        )
    if InputKind.ARRAY in accepted:
        return ConversionPlan(
            source, InputKind.ARRAY, (ConversionOp.DECODE,), False, True
        )
    return NoViablePath(
        source, accepted, "The standard does not synthesize a fetchable URL in v1."
    )


def _negotiate_base64(source: str, accepted: frozenset[InputKind]) -> ConversionPlan | NoViablePath:
    """Negotiate a path for an :class:`AudioBase64` source.

    Args:
        source: Source variant name.
        accepted: Engine-accepted input kinds.

    Returns:
        A plan (b64-decode to bytes, or b64-decode then decode), or no path.
    """
    if accepted & _ENCODED_KINDS:
        return ConversionPlan(
            source, InputKind.ENCODED_BYTES, (ConversionOp.B64_DECODE,), False, False
        )
    if InputKind.ARRAY in accepted:
        return ConversionPlan(
            source,
            InputKind.ARRAY,
            (ConversionOp.B64_DECODE, ConversionOp.DECODE),
            False,
            True,
        )
    return NoViablePath(
        source, accepted, "The standard does not synthesize a fetchable URL in v1."
    )


def _negotiate_url(source: str, accepted: frozenset[InputKind]) -> ConversionPlan | NoViablePath:
    """Negotiate a path for an :class:`AudioUrl` source.

    In v1 the standard never fetches URLs (SSRF risk); a URL is only viable if
    the engine fetches it server-side.

    Args:
        source: Source variant name.
        accepted: Engine-accepted input kinds.

    Returns:
        A passthrough plan if the engine accepts URLs, otherwise no viable path.
    """
    if InputKind.FETCHABLE_URL in accepted:
        return ConversionPlan(
            source, InputKind.FETCHABLE_URL, (ConversionOp.PASSTHROUGH,), False, False
        )
    return NoViablePath(
        source,
        accepted,
        "This engine does not fetch URLs; in v1 the standard does not fetch them "
        "either. Provide a local file via AudioPath.",
    )


__all__ = [
    "ConversionOp",
    "ConversionPlan",
    "NoViablePath",
    "can_accept",
    "negotiate",
    "negotiate_or_raise",
]
