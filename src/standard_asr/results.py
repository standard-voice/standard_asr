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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
        Time is measured in float seconds with the origin at the first submitted
        sample (audio time ``t=0``), the same origin as the streaming cursor
        (spec TR.2 / ST). ``start`` / ``end`` are therefore non-negative finite
        floats and ``end >= start`` (a zero-duration span is allowed). NaN / Inf
        are rejected (``allow_inf_nan=False``). Adapters convert ms /
        protobuf-duration / ticks into this frame; a negative or inverted span is
        an adapter bug, so the model refuses to represent one rather than let it
        surface as a silent wrong timestamp downstream.

    Args:
        start: Word start time in seconds (origin = first submitted sample;
            non-negative, finite).
        end: Word end time in seconds (non-negative, finite, ``>= start``).
        text: Word text.
        probability: Optional confidence in ``[0, 1]``.
        logprob: Optional log-probability (kept separate from ``probability``).
        speaker: Optional speaker label.
        channel: Optional channel index for provenance (``>= 0``).
        extra: Engine-specific extra data.

    Returns:
        None.

    Raises:
        ValueError: If field validation fails (incl. NaN/Inf, a negative time,
            or ``end < start``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    start: float = Field(..., ge=0.0, description="Word start time in seconds (>= 0).")
    end: float = Field(..., ge=0.0, description="Word end time in seconds (>= 0, >= start).")
    text: str = Field(..., description="Word text.")
    probability: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Confidence in [0, 1]."
    )
    logprob: float | None = Field(
        default=None, description="Log-probability (separate from probability)."
    )
    speaker: str | None = Field(default=None, description="Optional speaker label.")
    channel: int | None = Field(default=None, ge=0, description="Optional channel index (>= 0).")
    extra: dict[str, Any] = Field(default_factory=dict, description="Engine-specific extra data.")

    @model_validator(mode="after")
    def _check_span(self) -> Word:
        """Reject an inverted span (``end < start``) at construction.

        ``ge=0`` and ``allow_inf_nan=False`` already constrain each bound to a
        non-negative finite value; this enforces the remaining TR.2 invariant
        that a span never runs backwards. Equal bounds (zero duration) are
        allowed.

        Returns:
            The validated word.

        Raises:
            ValueError: If ``end`` is earlier than ``start``.
        """
        if self.end < self.start:
            raise ValueError(f"Word end ({self.end}) must be >= start ({self.start}).")
        return self


class Segment(BaseModel):
    """Segment-level detail, shared between batch results and streaming events.

    Note:
        ``start`` / ``end`` follow the same time frame as :class:`Word`:
        non-negative finite float seconds with origin at the first submitted
        sample (``t=0``), ``end >= start`` (zero-duration allowed), and NaN / Inf
        rejected (spec TR.2). Within one channel segments are time-ordered; the
        top-level :class:`TranscriptionResult.segments` are sorted by
        ``(start, channel)`` (cross-channel spans may overlap).

    Args:
        start: Segment start time in seconds (origin = first submitted sample;
            non-negative, finite).
        end: Segment end time in seconds (non-negative, finite, ``>= start``).
        text: Segment transcript text.
        words: Optional word-level details for this segment.
        speaker: Optional speaker label (authoritative diarization shape).
        channel: Optional channel index for provenance (``>= 0``).
        avg_logprob: Optional average log-probability.
        no_speech_prob: Optional no-speech probability.
        temperature: Optional decoding temperature.
        compression_ratio: Optional compression-ratio metric.
        extra: Engine-specific extra data.

    Returns:
        None.

    Raises:
        ValueError: If field validation fails (incl. NaN/Inf, a negative time,
            or ``end < start``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    start: float = Field(..., ge=0.0, description="Segment start time in seconds (>= 0).")
    end: float = Field(..., ge=0.0, description="Segment end time in seconds (>= 0, >= start).")
    text: str = Field(..., description="Segment transcript text.")
    words: list[Word] | None = Field(
        default=None, description="Word-level details for this segment."
    )
    speaker: str | None = Field(default=None, description="Optional speaker label.")
    channel: int | None = Field(default=None, ge=0, description="Optional channel index (>= 0).")
    avg_logprob: float | None = Field(default=None, description="Optional average log-probability.")
    no_speech_prob: float | None = Field(
        default=None, description="Optional no-speech probability."
    )
    temperature: float | None = Field(default=None, description="Optional decoding temperature.")
    compression_ratio: float | None = Field(
        default=None, description="Optional compression-ratio metric."
    )
    extra: dict[str, Any] = Field(default_factory=dict, description="Engine-specific extra data.")

    @model_validator(mode="after")
    def _check_span(self) -> Segment:
        """Reject an inverted span (``end < start``) at construction.

        ``ge=0`` and ``allow_inf_nan=False`` already constrain each bound to a
        non-negative finite value; this enforces the remaining TR.2 invariant
        that a span never runs backwards. Equal bounds (zero duration) are
        allowed.

        Returns:
            The validated segment.

        Raises:
            ValueError: If ``end`` is earlier than ``start``.
        """
        if self.end < self.start:
            raise ValueError(f"Segment end ({self.end}) must be >= start ({self.start}).")
        return self


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

    channel: int = Field(..., ge=0, description="Channel index (>= 0).")
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
        detected_language: Detected language as a well-formed BCP-47 tag in
            ``auto`` mode; ``None`` when not applicable.
        language_confidence: Detection confidence in ``[0, 1]``.
        duration: Audio duration in seconds, if known (non-negative, finite).
        segments: Time-ordered segments across all channels, if available.
        words: Flattened word-level details, if available.
        channels: Per-channel results when channel separation was performed.
        diagnostics: Conversion / best_effort / degradation diagnostics.
        metadata: Standardized engine-agnostic metadata.
        extra: Engine-specific / experimental data (incl. provider formats).

    Returns:
        None.

    Raises:
        ValueError: If field validation fails (incl. NaN/Inf, a negative
            ``duration``, or a malformed ``detected_language``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    text: str = Field(..., description="Full transcript text.")
    detected_language: str | None = Field(
        default=None, description="Detected language (well-formed BCP-47) in auto mode."
    )
    language_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Detection confidence in [0, 1]."
    )
    duration: float | None = Field(
        default=None, ge=0.0, description="Audio duration in seconds, if known (>= 0)."
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

    @field_validator("detected_language")
    @classmethod
    def _check_detected_language(cls, value: str | None) -> str | None:
        """Validate and canonicalize ``detected_language`` (spec TR.1).

        ``detected_language`` is the language the engine *resolved to*, so it must
        be a concrete, well-formed BCP-47 tag -- never the reserved ``"auto"``
        directive (which is a request to detect, not a detection result). A
        malformed value (e.g. the native name ``"English"``) is rejected loudly
        rather than echoed back, mirroring how :mod:`standard_asr.language`
        validates request tags. A valid tag is normalized to canonical casing so
        echoed values read consistently (``zh-Hans``). The import is deferred
        because :mod:`standard_asr.language` imports from this module.

        Args:
            value: The candidate detected-language tag, or ``None``.

        Returns:
            The canonicalized tag, or ``None`` when not applicable.

        Raises:
            ValueError: If ``value`` is the reserved ``"auto"`` token or is not a
                well-formed BCP-47 tag.
        """
        if value is None:
            return None
        from .language import AUTO, is_valid_bcp47, normalize_bcp47

        if not is_valid_bcp47(value):
            raise ValueError(
                f"detected_language is not a well-formed BCP-47 tag: {value!r} "
                "(e.g. 'en', 'en-US', 'zh-Hans')."
            )
        normalized = normalize_bcp47(value)
        if normalized == AUTO:
            raise ValueError(
                "detected_language MUST be a concrete detected tag, not the reserved 'auto'."
            )
        return normalized

    @model_validator(mode="after")
    def _check_top_level_derivable_from_channels(self) -> TranscriptionResult:
        """Reject a result whose top level is not derivable from ``channels``.

        Spec TR.4 promises that ignoring ``channels`` is always safe and
        lossless: when ``channels`` is present, the top-level fields are the
        time-merge of all channels. A result whose channel entries carry
        ``segments`` / ``words`` while the corresponding top-level field is
        ``None`` breaks that promise -- a channel-agnostic consumer (e.g. the
        SRT/VTT renderers, built over the constant top-level ``segments``)
        would silently lose all per-channel detail. That shape is an engine
        bug, so the model refuses to represent it.

        Returns:
            The validated result.

        Raises:
            ValueError: If any ``channels`` entry carries ``segments`` (or
                ``words``) while the top-level field is ``None``.
        """
        if self.channels is not None:
            for name in ("segments", "words"):
                if getattr(self, name) is None and any(
                    getattr(entry, name) is not None for entry in self.channels
                ):
                    raise ValueError(
                        f"channels entries carry {name} but the top-level {name} is None; "
                        f"spec TR.4 requires the top level to be derivable from channels "
                        f"(ignoring channels must be lossless). Populate the top-level "
                        f"{name} with the time-merged union of all channels' {name}."
                    )
        return self


__all__ = [
    "ChannelResult",
    "Diagnostic",
    "Segment",
    "TranscriptionResult",
    "Word",
]
