# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Wire audio format for incremental streaming input.

When an application opens a streaming session that feeds raw PCM frames
incrementally (``start_transcription(audio_format=...)``), it declares the wire
format once via :class:`AudioFormat`. The format is locked for the lifetime of
the session. This is distinct from the batch :data:`~standard_asr.audio_input.AudioInput`
union: streaming feeds bare PCM frames whose encoding/sample-rate/channels are
not self-describing, so they MUST be declared up front.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AudioFormat(BaseModel):
    """Declared wire format for raw PCM frames fed to a streaming session.

    Args:
        encoding: Wire encoding of the PCM frames (e.g. ``"pcm_s16le"``,
            ``"mulaw"``). MUST be one of the engine's ``wire_encodings``.
        sample_rate: Sample rate of the frames in Hz.
        channels: Number of interleaved channels. Defaults to ``1`` (mono).

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    encoding: str = Field(
        ...,
        description="Wire encoding of the PCM frames (e.g. 'pcm_s16le', 'mulaw').",
    )
    sample_rate: int = Field(..., gt=0, description="Sample rate in Hz.")
    channels: int = Field(default=1, gt=0, description="Number of channels.")

    @field_validator("encoding")
    @classmethod
    def _normalize_encoding(cls, value: str) -> str:
        """Strip and lowercase the wire encoding (case-insensitive identifier).

        ``BaseProperties.wire_encodings`` stores its allowlist stripped and
        lowercased, and
        :meth:`~standard_asr.asr_interface.EngineBase.ensure_stream_format_supported`
        checks the request encoding against that normalized list. Normalizing the
        request encoding the same way here keeps the match case-insensitive: an
        engine declaring ``"pcm_s16le"`` MUST accept a session opened with
        ``AudioFormat(encoding="PCM_S16LE")`` -- a pure case difference is the
        same encoding, never a fail-closed mismatch that bricks a valid session.

        Args:
            value: The declared wire encoding.

        Returns:
            The stripped, lowercased encoding.

        Raises:
            ValueError: If the encoding is blank after stripping.
        """
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("encoding must not be blank.")
        return cleaned


__all__ = ["AudioFormat"]
