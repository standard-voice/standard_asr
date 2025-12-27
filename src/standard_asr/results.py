"""Standard ASR transcription result models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Word(BaseModel):
    """Word-level timestamp information.

    Args:
        start: Start time in seconds.
        end: End time in seconds.
        text: Word text.
        probability: Optional probability score.
        speaker: Optional speaker label.
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
        None, description="Optional probability score for the word."
    )
    speaker: str | None = Field(None, description="Optional speaker label.")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific extra data."
    )


class Segment(BaseModel):
    """Segment-level transcription metadata.

    Args:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        text: Segment transcript text.
        words: Optional word-level details for this segment.
        speaker: Optional speaker label.
        temperature: Optional decoding temperature.
        avg_logprob: Optional average log probability.
        compression_ratio: Optional compression ratio metric.
        no_speech_prob: Optional no-speech probability.
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
        None, description="Word-level details for this segment."
    )
    speaker: str | None = Field(None, description="Optional speaker label.")
    temperature: float | None = Field(
        None, description="Optional decoding temperature."
    )
    avg_logprob: float | None = Field(
        None, description="Optional average log probability."
    )
    compression_ratio: float | None = Field(
        None, description="Optional compression ratio metric."
    )
    no_speech_prob: float | None = Field(
        None, description="Optional no-speech probability."
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific extra data."
    )


class TranscriptionResult(BaseModel):
    """Standard transcription result returned by Standard ASR engines.

    Args:
        text: Full transcript text.
        language: Detected or forced language tag (BCP 47).
        duration: Audio duration in seconds (if known).
        segments: Optional list of segment metadata.
        words: Optional flattened word-level list.
        metadata: Engine-agnostic metadata.
        extra: Engine-specific extra data.

    Returns:
        None.

    Raises:
        ValueError: If field validation fails.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(..., description="Full transcript text.")
    language: str | None = Field(
        None, description="Detected or forced language tag (BCP 47)."
    )
    duration: float | None = Field(
        None, description="Audio duration in seconds, if available."
    )
    segments: list[Segment] | None = Field(
        None, description="Segment-level details, if available."
    )
    words: list[Word] | None = Field(
        None, description="Flattened word-level details, if available."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-agnostic metadata for this transcription.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific extra data."
    )

    def text_only(self) -> str:
        """Return the transcript text.

        Args:
            None.

        Returns:
            Transcript text.

        Raises:
            None.
        """

        return self.text


__all__ = ["Segment", "TranscriptionResult", "Word"]
