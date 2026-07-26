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

from typing import Any, Literal, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Diagnostic(BaseModel):
    """A structured, non-fatal notification from the standard layer.

    Diagnostics report lossy conversions, assumed parameters, best_effort
    degradations, and similar non-ideal paths.

    Attributes:
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


def validate_speaker_label(value: str | None) -> str | None:
    """Validate a speaker label at construction (shared by every ``speaker`` field).

    One rule for :class:`Segment`, :class:`Word`, and
    :class:`~standard_asr.runtime.streaming.TranscriptionEvent`: a label
    must be non-empty, not whitespace-only, and carry no leading/trailing
    whitespace. ``""`` would be an undefined third state between ``None``
    (no attribution) and a real label; edge whitespace (``"A "`` vs ``"A"``)
    silently breaks within-result label consistency -- two strings = two
    speakers -- so both are **rejected, never normalized** (normalizing would
    hide an adapter bug behind a silently rewritten value; the same stance as
    ``phrase_hints``). ``None`` (no attribution) passes through.

    Args:
        value: The candidate speaker label, or ``None``.

    Returns:
        The validated value unchanged.

    Raises:
        ValueError: If ``value`` is empty, whitespace-only, or has leading or
            trailing whitespace.
    """
    if value is None:
        return value
    # The raw value is NOT echoed in the message: a speaker label can carry a
    # personal name and this error surfaces verbatim through server 422 bodies
    # and logs (the same redaction stance as the language-tag validator).
    if not value.strip():
        raise ValueError(
            "speaker label must not be empty or whitespace-only (use None for "
            "'no speaker attribution')."
        )
    if value != value.strip():
        raise ValueError(
            "speaker label must not have leading or trailing whitespace (two "
            "labels differing only in edge whitespace would read as two "
            "different speakers)."
        )
    return value


class Word(BaseModel):
    """Word-level detail, shared between batch results and streaming events.

    Note:
        Time is measured in float seconds with the origin at the first submitted
        sample (audio time ``t=0``), the same origin as the streaming cursor.
        ``start`` / ``end`` are therefore non-negative finite
        floats and ``end >= start`` (a zero-duration span is allowed). NaN / Inf
        are rejected (``allow_inf_nan=False``). Adapters convert ms /
        protobuf-duration / ticks into this frame; a negative or inverted span is
        an adapter bug, so the model refuses to represent one rather than let it
        surface as a silent wrong timestamp downstream.

    Attributes:
        start: Word start time in seconds (origin = first submitted sample;
            non-negative, finite).
        end: Word end time in seconds (non-negative, finite, ``>= start``).
        text: Word text.
        probability: Optional confidence in ``[0, 1]``.
        logprob: Optional log-probability (kept separate from ``probability``).
        speaker: Optional speaker label.
        channel: Optional channel index for provenance (``>= 0``).
        extra: Engine-specific extra data.

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
    speaker: str | None = Field(
        default=None, description="Optional speaker label (non-empty, no edge whitespace)."
    )
    channel: int | None = Field(default=None, ge=0, description="Optional channel index (>= 0).")
    extra: dict[str, Any] = Field(default_factory=dict, description="Engine-specific extra data.")

    @field_validator("speaker")
    @classmethod
    def _check_speaker(cls, value: str | None) -> str | None:
        """Reject a malformed speaker label at construction (fail-fast).

        Delegates to :func:`validate_speaker_label` -- the shared rule for
        every ``speaker`` field, so batch and streaming accept exactly the
        same labels.

        Args:
            value: The candidate speaker label, or ``None``.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If the label is empty, whitespace-only, or padded.
        """
        return validate_speaker_label(value)

    @model_validator(mode="after")
    def _check_span(self) -> Word:
        """Reject an inverted span (``end < start``) at construction.

        ``ge=0`` and ``allow_inf_nan=False`` already constrain each bound to a
        non-negative finite value; this enforces the remaining invariant
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
        rejected. Within one channel segments are time-ordered; the
        top-level :class:`TranscriptionResult.segments` are sorted by
        ``(start, channel, speaker)`` (cross-channel spans may overlap).
        ``speaker`` is the final tie-break for equal-``(start, channel)``
        overlapping segments (the single-channel multi-speaker case); ``None``
        sorts before any real label.

    Attributes:
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
    speaker: str | None = Field(
        default=None,
        description="Optional speaker label (non-empty, no edge whitespace).",
    )
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

    @field_validator("speaker")
    @classmethod
    def _check_speaker(cls, value: str | None) -> str | None:
        """Reject a malformed speaker label at construction (fail-fast).

        Delegates to :func:`validate_speaker_label` -- the shared rule for
        every ``speaker`` field. ``Segment.speaker`` is the authoritative
        diarization shape, so a malformed label here is the most
        damaging: it would flow into renderers and reducers as a phantom
        speaker.

        Args:
            value: The candidate speaker label, or ``None``.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If the label is empty, whitespace-only, or padded.
        """
        return validate_speaker_label(value)

    @model_validator(mode="after")
    def _check_span(self) -> Segment:
        """Reject an inverted span (``end < start``) at construction.

        ``ge=0`` and ``allow_inf_nan=False`` already constrain each bound to a
        non-negative finite value; this enforces the remaining invariant
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


def synthesize_segment_speaker(words: Sequence[Word] | None) -> str | None:
    """Derive a segment-level speaker label from its words (the pinned synthesis rule).

    THE single synthesis rule of the standard layer, used when an engine
    populates ``Word.speaker`` but leaves the authoritative ``Segment.speaker``
    ``None``:

    * **Majority by word count** -- the label carried by the most words wins.
    * **Tie** -> the speaker of the earliest (lowest-index) word among the
      tied labels.
    * Words with ``speaker=None`` do **not** vote.
    * No speaker-bearing words (or ``words`` ``None``/empty) -> ``None``.

    This function is deliberately the ONLY implementation -- both the batch
    post-processing (``EngineBase.transcribe``) and the streaming reducer
    (:class:`~standard_asr.runtime.streaming.StreamReducer`) call it, never a private
    copy. Portability demands it: the same engine and audio MUST yield the same
    ``Segment.speaker`` whether the app took the batch or the streaming path;
    two drifting copies of the rule would silently break that promise.

    Args:
        words: The segment's word-level details, or ``None``.

    Returns:
        The synthesized speaker label, or ``None`` when no word carries one.
    """
    if not words:
        return None
    counts: dict[str, int] = {}
    first_index: dict[str, int] = {}
    for index, word in enumerate(words):
        if word.speaker is None:
            continue
        counts[word.speaker] = counts.get(word.speaker, 0) + 1
        first_index.setdefault(word.speaker, index)
    if not counts:
        return None
    # Highest count wins; on a count tie the LOWER first_index (earlier word)
    # wins, hence the negated index in the key.
    return max(counts, key=lambda label: (counts[label], -first_index[label]))


class ChannelResult(BaseModel):
    """Per-channel transcription for multi-channel audio.

    Attributes:
        channel: Channel index.
        text: Full transcript for this channel.
        segments: Optional segment-level details for this channel.
        words: Optional flattened word-level details for this channel.

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

    Attributes:
        text: Full transcript (required).
        detected_language: Detected language as a well-formed BCP-47 tag in
            ``auto`` mode; ``None`` when not applicable.
        language_confidence: Detection confidence in ``[0, 1]``.
        duration: Audio duration in seconds, if known (non-negative, finite).
        segments: Segments across all channels, if available. They
            SHOULD be sorted by ``(start, channel, speaker)`` (monotonic within
            a channel; ``speaker`` is the final tie-break, ``None`` sorting
            first); this ordering is an **engine obligation**, neither enforced
            at construction nor checked by the compliance suite (the streaming
            reducer legitimately keeps arrival order for timestamp-less engines).
            The SRT/VTT renderers' defensive re-sort is the only standard-layer
            safety net.
        words: Flattened word-level details, if available.
        channels: Per-channel results when channel separation was performed. Each
            ``channel`` index MUST be unique (one entry per channel),
            enforced at construction.
        diagnostics: Conversion / best_effort / degradation diagnostics.
        extra: Engine-specific / experimental data (incl. provider formats).

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
    # No `metadata` pocket: the spec removed blanket metadata from Properties
    # and Capabilities ("no known use case, invites unstructured data, breaks
    # machine readability"), and a result-side "standardized metadata" dict with
    # no standardized keys, no writer, and no reader was the same disease.
    # Standardized result data gets a real field (additive-minor); everything
    # engine-specific goes in `extra`.
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific / experimental data."
    )

    @field_validator("detected_language")
    @classmethod
    def _check_detected_language(cls, value: str | None) -> str | None:
        """Validate and canonicalize ``detected_language``.

        Delegates to the shared
        :func:`~standard_asr.contract.language.validate_detected_language` -- the same
        rule as ``TranscriptionEvent.detected_language``, because the event
        field feeds the next session's ``language`` (reconnect
        continuity) and the two sides MUST accept exactly the same tags. The
        import is deferred because :mod:`standard_asr.contract.language` imports from
        this module.

        Args:
            value: The candidate detected-language tag, or ``None``.

        Returns:
            The canonicalized tag, or ``None`` when not applicable.

        Raises:
            ValueError: If ``value`` is the reserved ``"auto"`` token or is not a
                well-formed BCP-47 tag.
        """
        from standard_asr.contract.language import validate_detected_language

        return validate_detected_language(value)

    @model_validator(mode="after")
    def _check_top_level_derivable_from_channels(self) -> TranscriptionResult:
        """Reject channel shapes the spec forbids: duplicates and lossy top level.

        Two construct-time invariants over ``channels`` (cheap to check in
        the single pass that already walks the list):

        1. **Unique channel index.** The standard defines ``channels`` as one
           ``ChannelResult`` *per channel*, so two entries with the same
           ``channel`` index is a semantically illegal shape (which channel's
           ``text`` wins? how does the "top level derivable from channels"
           invariant resolve the ambiguity?). A consumer keying a dict by
           ``channel`` index would silently lose one entry, so the model refuses
           to represent the duplicate.
        2. **Top level derivable from channels.** The standard promises that ignoring
           ``channels`` is always safe and lossless: when ``channels`` is
           present, the top-level fields are the time-merge of all channels. A
           result whose channel entries carry ``segments`` / ``words`` while the
           corresponding top-level field is ``None`` breaks that promise -- a
           channel-agnostic consumer (e.g. the SRT/VTT renderers, built over the
           constant top-level ``segments``) would silently lose all per-channel
           detail. That shape is an engine bug, so the model refuses it.

        The complementary ordering invariant (top-level ``segments`` sorted
        by ``(start, channel, speaker)``, monotonic within a channel) is
        intentionally *not* enforced here: the streaming reducer
        (:class:`~standard_asr.runtime.streaming.StreamReducer`) legitimately preserves
        arrival order for timestamp-less engines and sorts only by ``start``
        (no channel/speaker tie-break), so a strict ``(start, channel,
        speaker)`` construct-time check would reject valid reduced results. For
        the same reason the compliance suite does not check ordering either;
        ordering is an engine obligation, and the renderers' defensive re-sort
        at their boundary is the only standard-layer safety net.

        Returns:
            The validated result.

        Raises:
            ValueError: If two ``channels`` entries share a ``channel`` index, or
                if any ``channels`` entry carries ``segments`` (or ``words``)
                while the top-level field is ``None``.
        """
        if self.channels is not None:
            seen: set[int] = set()
            for entry in self.channels:
                if entry.channel in seen:
                    raise ValueError(
                        f"channels contains duplicate entries for channel index "
                        f"{entry.channel}; the standard defines channels as one ChannelResult "
                        f"per channel, so each channel index MUST be unique (a duplicate "
                        f"makes the top-level merge ambiguous and silently drops data for "
                        f"consumers keyed by channel)."
                    )
                seen.add(entry.channel)
            for name in ("segments", "words"):
                if getattr(self, name) is None and any(
                    getattr(entry, name) is not None for entry in self.channels
                ):
                    raise ValueError(
                        f"channels entries carry {name} but the top-level {name} is None; "
                        f"the standard requires the top level to be derivable from channels "
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
    "synthesize_segment_speaker",
    "validate_speaker_label",
]
