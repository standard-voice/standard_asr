"""Tests covering the StandardASR protocol helper methods."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from standard_asr import BaseConfig, BaseProperties, StandardASR, TranscriptionResult


class _AsyncConfig(BaseConfig[Literal["dummy"]]):
    engine: Literal["dummy"] = "dummy"


class _AsyncProperties(BaseProperties):
    engine_id: str = "dummy"
    model_name: str = "async"
    protocol_version: str = "0.2.0"
    supported_languages: list[str] = ["en"]
    supported_devices: list[str] = ["cpu"]
    supported_sample_rates: list[int] = [16000]
    supported_channels: list[int] = [1]
    audio_dtype: str = "float32"


class _AsyncASR(StandardASR):
    properties: ClassVar[BaseProperties] = _AsyncProperties()

    def __init__(self) -> None:
        self.called = False
        self.config = _AsyncConfig(engine="dummy")

    def transcribe(
        self, audio: NDArray[np.float32], options: Any = None
    ) -> TranscriptionResult:
        self.called = True
        return TranscriptionResult(text="ok")


def test_transcribe_async_calls_sync() -> None:
    """Ensure StandardASR.transcribe_async delegates to transcribe."""
    asr = _AsyncASR()
    audio = np.zeros(4, dtype=np.float32)

    result = asyncio.run(asr.transcribe_async(audio))

    assert result.text == "ok"
    assert asr.called is True
