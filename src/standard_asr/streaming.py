"""Streaming transcription protocol definitions."""

from __future__ import annotations

from typing import Any, AsyncIterable, AsyncIterator, Iterable, Iterator, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from .options import BaseTranscribeOptions


class StreamChunk(BaseModel):
    """Single streaming transcription chunk.

    Args:
        text: Partial transcript text.
        start: Optional chunk start time in seconds.
        end: Optional chunk end time in seconds.
        is_final: Whether this chunk is a final, stable result.
        extra: Engine-specific data.

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(..., description="Partial transcript text.")
    start: float | None = Field(None, description="Chunk start time in seconds.")
    end: float | None = Field(None, description="Chunk end time in seconds.")
    is_final: bool = Field(
        False, description="Whether this chunk represents a final result."
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific chunk metadata."
    )


class StreamingASR(Protocol):
    """Optional protocol for ASR engines that support streaming.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def transcribe_stream(
        self,
        audio_stream: Iterable[NDArray[np.float32]],
        options: BaseTranscribeOptions | None = None,
    ) -> Iterator[StreamChunk]:
        """Transcribe streaming audio input.

        Args:
            audio_stream: Iterable of audio chunks.
            options: Optional per-request options.

        Returns:
            Iterator of streaming chunks.

        Raises:
            TranscriptionError: If transcription fails.
        """
        raise NotImplementedError

    async def transcribe_stream_async(
        self,
        audio_stream: AsyncIterable[NDArray[np.float32]],
        options: BaseTranscribeOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Asynchronously transcribe streaming audio input.

        Args:
            audio_stream: Async iterable of audio chunks.
            options: Optional per-request options.

        Returns:
            Async iterator of streaming chunks.

        Raises:
            TranscriptionError: If transcription fails.
        """
        raise NotImplementedError


__all__ = ["StreamChunk", "StreamingASR"]
