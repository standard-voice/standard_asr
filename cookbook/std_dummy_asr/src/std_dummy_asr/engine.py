"""Lightweight Standard ASR implementation for demo purposes."""

from __future__ import annotations

from typing import ClassVar, Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from standard_asr import (
    BaseConfig,
    BaseTranscribeOptions,
    StandardASR,
    TranscriptionResult,
)
from standard_asr.asr_properties import BaseProperties
from standard_asr.features import FeatureFlag
from standard_asr.options import coerce_options
from standard_asr.runtime import validate_audio_input


class DummyASRConfig(BaseConfig[Literal["dummy"]]):
    """Configuration model for the dummy ASR engine.

    Args:
        engine: Discriminator identifying this engine (always ``"dummy"``).
        message: Text prefix inserted into the transcript.

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    engine: Literal["dummy"] = "dummy"
    message: str = Field(
        "echo",
        description=(
            "Text prefix included in the emitted transcript for demo purposes."
        ),
    )


class DummyASRProperties(BaseProperties):
    """Static metadata describing the dummy ASR engine.

    Args:
        None.

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    engine_id: str = "dummy"
    model_name: str = "echo"
    protocol_version: str = "0.2.0"
    supported_languages: list[str] = ["en"]
    supported_devices: list[str] = ["cpu"]
    supported_sample_rates: list[int] = [16000]
    supported_channels: list[int] = [1, 2]
    audio_dtype: str = "float32"
    features: set[FeatureFlag] = set()
    description: str | None = "Dummy echo engine for testing and demos."


class DummyASR(StandardASR):
    """Trivial ASR implementation that reports the input shape.

    Args:
        message: Text prefix for the transcript.

    Returns:
        None.

    Raises:
        ValueError: If configuration validation fails.
    """

    config: BaseConfig[str]
    properties: ClassVar[BaseProperties] = DummyASRProperties()

    def __init__(self, message: str = "echo") -> None:
        self.config = DummyASRConfig(engine="dummy", message=message)

    def transcribe(
        self,
        audio: NDArray[np.float32],
        options: BaseTranscribeOptions | dict[str, object] | None = None,
    ) -> TranscriptionResult:
        """Return a short description of the provided audio buffer.

        Args:
            audio: Waveform array in ``float32`` format.
            options: Optional transcription options (model or dict).

        Returns:
            Standard ASR transcription result.

        Raises:
            ValueError: If the audio input is invalid.
        """
        validate_audio_input(audio, self.properties)
        resolved_options = coerce_options(options, BaseTranscribeOptions)

        array = np.asarray(audio)
        samples = int(array.size)
        config = cast(DummyASRConfig, self.config)
        text = f"{config.message}: {samples} samples"

        return TranscriptionResult(
            text=text,
            language=resolved_options.language,
            metadata={"samples": samples},
        )
