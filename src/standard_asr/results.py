# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard ASR transcription result models (constant schema).

The result schema is **constant**: capabilities and parameters decide whether
optional fields are *populated*, never the return type's shape (spec, section
"Transcription Result"). The same :class:`Segment` / :class:`Word` submodels are
shared between batch results and streaming events.

Null rules (disambiguation):

* A field is ``None`` -> the data was **not requested / not applicable**.
* A field is ``[]`` -> it **was requested but is empty** (e.g. silence).
* Whether a feature is *supported* is answered by capabilities, never by a
  field being ``None``.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field


class Diagnostic(BaseModel):
    """A structured, non-fatal notification from the standard layer.

    Diagnostics report lossy conversions, assumed parameters, best_effort
    degradations, and similar non-ideal paths.

    Args:
        level: Severity, ``"info"`` or ``"warning"``.
        code: Stable machine-readable code (e.g. ``"audio_conversion"``).
        message: Human-readable explanation.
        param: The parameter the diagnostic concerns, if any.
        provided: The value the application provided, if relevant.
        effective: The value that took effect, if relevant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: Literal["info", "warning"] = Field(default="info")
    code: str = Field(..., description="Stable machine-readable diagnostic code.")
    message: str = Field(..., description="Human-readable explanation.")
    param: str | None = Field(default=None, description="Parameter concerned, if any.")
    provided: Any | None = Field(default=None, description="Value provided, if any.")
    effective: Any | None = Field(default=None, description="Value applied, if any.")


class Word(BaseModel):
    """Word-level detail, shared between batch results and streaming events.

    Note:
        ``start`` / ``end`` are NOT constrained to ``>= 0``. The origin is the
        first submitted sample (t=0), but negative offsets are intentionally
        permitted to support streaming pre-roll (audio buffered before the
        nominal session start). Forbidding negatives would make those legitimate
        cases unrepresentable; renderers clamp to zero for the SRT/VTT grammar.

    Args:
        start: Word start time in seconds (origin = first submitted sample;
            may be negative for pre-roll).
        end: Word end time in seconds.
        text: Word text.
        probability: Optional confidence in ``[0, 1]``.
        logprob: Optional log-probability (kept separate from ``probability``).
        speaker: Optional speaker label.
        channel: Optional channel index for provenance.
        extra: Engine-specific extra data.

    Returns:
        None.

    Raises:
        ValueError: If field validation fails.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: float = Field(..., description="Word start time in seconds.")
    end: float = Field(..., description="Word end time in seconds.")
    text: str = Field(..., description="Word text.")
    probability: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Confidence in [0, 1]."
    )
    logprob: float | None = Field(
        default=None, description="Log-probability (separate from probability)."
    )
    speaker: str | None = Field(default=None, description="Optional speaker label.")
    channel: int | None = Field(default=None, description="Optional channel index.")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific extra data."
    )


class Segment(BaseModel):
    """Segment-level detail, shared between batch results and streaming events.

    Note:
        ``start`` / ``end`` are NOT constrained to ``>= 0``; negative offsets
        are permitted for streaming pre-roll (see :class:`Word`).

    Args:
        start: Segment start time in seconds (origin = first submitted sample;
            may be negative for pre-roll).
        end: Segment end time in seconds.
        text: Segment transcript text.
        words: Optional word-level details for this segment.
        speaker: Optional speaker label (authoritative diarization shape).
        channel: Optional channel index for provenance.
        avg_logprob: Optional average log-probability.
        no_speech_prob: Optional no-speech probability.
        temperature: Optional decoding temperature.
        compression_ratio: Optional compression-ratio metric.
        extra: Engine-specific extra data.

    Returns:
        None.

    Raises:
        ValueError: If field validation fails.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: float = Field(..., description="Segment start time in seconds.")
    end: float = Field(..., description="Segment end time in seconds.")
    text: str = Field(..., description="Segment transcript text.")
    words: list[Word] | None = Field(
        default=None, description="Word-level details for this segment."
    )
    speaker: str | None = Field(default=None, description="Optional speaker label.")
    channel: int | None = Field(default=None, description="Optional channel index.")
    avg_logprob: float | None = Field(
        default=None, description="Optional average log-probability."
    )
    no_speech_prob: float | None = Field(
        default=None, description="Optional no-speech probability."
    )
    temperature: float | None = Field(
        default=None, description="Optional decoding temperature."
    )
    compression_ratio: float | None = Field(
        default=None, description="Optional compression-ratio metric."
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific extra data."
    )


class ChannelResult(BaseModel):
    """Per-channel transcription for multi-channel audio.

    Args:
        channel: Channel index.
        text: Full transcript for this channel.
        segments: Optional segment-level details for this channel.
        words: Optional flattened word-level details for this channel.

    Returns:
        None.

    Raises:
        ValueError: If field validation fails.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    channel: int = Field(..., description="Channel index.")
    text: str = Field(..., description="Full transcript for this channel.")
    segments: list[Segment] | None = Field(
        default=None, description="Segment-level details for this channel."
    )
    words: list[Word] | None = Field(
        default=None, description="Word-level details for this channel."
    )


class TranscriptionResult(BaseModel):
    """The constant-shape result returned by ``transcribe`` and stream reduction.

    The top-level ``text`` / ``segments`` / ``words`` are always the complete,
    channel- and speaker-agnostic transcription. For multi-channel audio they
    are the time-merge of all channels (never channel-0-only), so ignoring
    ``channels`` is always safe and lossless.

    Args:
        text: Full transcript (required).
        detected_language: Detected language (BCP-47) in ``auto`` mode.
        language_confidence: Detection confidence in ``[0, 1]``.
        duration: Audio duration in seconds, if known.
        segments: Time-ordered segments across all channels, if available.
        words: Flattened word-level details, if available.
        channels: Per-channel results when channel separation was performed.
        diagnostics: Conversion / best_effort / degradation diagnostics.
        metadata: Standardized engine-agnostic metadata.
        extra: Engine-specific / experimental data (incl. provider formats).

    Returns:
        None.

    Raises:
        ValueError: If field validation fails.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(..., description="Full transcript text.")
    detected_language: str | None = Field(
        default=None, description="Detected language (BCP-47) in auto mode."
    )
    language_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Detection confidence in [0, 1]."
    )
    duration: float | None = Field(
        default=None, description="Audio duration in seconds, if known."
    )
    segments: list[Segment] | None = Field(
        default=None, description="Time-ordered segments, if available."
    )
    words: list[Word] | None = Field(
        default=None, description="Flattened word-level details, if available."
    )
    channels: list[ChannelResult] | None = Field(
        default=None, description="Per-channel results, if channel-separated."
    )
    diagnostics: list[Diagnostic] = Field(
        default_factory=lambda: cast("list[Diagnostic]", []),
        description="Non-fatal diagnostics.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Standardized engine-agnostic metadata."
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific / experimental data."
    )


__all__ = [
    "ChannelResult",
    "Diagnostic",
    "Segment",
    "TranscriptionResult",
    "Word",
]
