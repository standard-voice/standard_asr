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

from pydantic import BaseModel, ConfigDict, Field


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


__all__ = ["AudioFormat"]
