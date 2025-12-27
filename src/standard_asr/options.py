"""Inference option models for Standard ASR."""

from __future__ import annotations

from typing import Any, Mapping, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class BaseTranscribeOptions(BaseModel):
    """Base options model for ASR transcription requests.

    Args:
        language: Optional BCP 47 language tag.
        task: Transcription task, e.g. ``transcribe`` or ``translate``.
        word_timestamps: Whether word-level timestamps are requested.
        speaker_diarization: Whether speaker diarization is requested.
        extra: Engine-specific options not yet standardized.

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    language: str | None = Field(
        None, description="BCP 47 language tag to force, if supported."
    )
    task: str = Field(
        "transcribe",
        description="Task type (e.g., 'transcribe' or 'translate').",
    )
    word_timestamps: bool = Field(
        False, description="Request word-level timestamps when supported."
    )
    speaker_diarization: bool = Field(
        False, description="Request speaker diarization when supported."
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-specific options not yet standardized.",
    )


OptionsT = TypeVar("OptionsT", bound=BaseTranscribeOptions)


def coerce_options(
    options: OptionsT | BaseTranscribeOptions | Mapping[str, Any] | None,
    options_cls: type[OptionsT],
) -> OptionsT:
    """Coerce an arbitrary options object into a concrete options model.

    Args:
        options: Options instance, mapping, or ``None``.
        options_cls: Target Pydantic model class.

    Returns:
        Parsed options instance of ``options_cls``.

    Raises:
        ValueError: If validation fails while parsing ``options``.
    """
    if options is None:
        return options_cls()
    if isinstance(options, options_cls):
        return options
    if isinstance(options, BaseTranscribeOptions):
        return options_cls.model_validate(options.model_dump())
    return options_cls.model_validate(options)


__all__ = ["BaseTranscribeOptions", "coerce_options"]
