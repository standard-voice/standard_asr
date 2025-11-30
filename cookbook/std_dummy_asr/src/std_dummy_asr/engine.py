"""Lightweight Standard ASR implementation for demo purposes."""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from standard_asr import BaseConfig, StandardASR
from standard_asr.asr_properties import BaseProperties


class DummyASRConfig(BaseConfig[Literal["dummy"]]):
    """Configuration model for the dummy ASR engine.

    Args:
        engine: Discriminator identifying this engine (always ``"dummy"``).
        message: Text prefix inserted into the transcript.
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

    Attributes:
        model_name: Preset name advertised via entry points.
        protocol_version: Protocol version implemented by this engine.
        supported_language: Languages represented as IETF BCP 47 tags.
        supported_device: Allowed compute targets.
        audio_dtype: Numpy dtype string for accepted audio buffers.
        supported_channels: Supported channel counts.
    """

    model_name: str = "echo"
    protocol_version: str = "0.1.0"
    supported_language: list[str] = ["en"]
    supported_device: list[str] = ["cpu"]
    audio_dtype: str = "float32"
    supported_channels: list[int] = [1, 2]


class DummyASR(StandardASR):
    """Trivial ASR implementation that reports the input shape.

    Attributes:
        config: Instance configuration captured at construction time.
        properties: Class-level metadata describing capabilities.
    """

    config: DummyASRConfig
    properties: ClassVar[DummyASRProperties] = DummyASRProperties()

    def __init__(self, message: str = "echo") -> None:
        self.config = DummyASRConfig(engine="dummy", message=message)

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        """Return a short description of the provided audio buffer.

        Args:
            audio: Waveform array in ``float32`` format.

        Returns:
            A synthetic transcript containing the configured prefix and sample count.
        """

        array = np.asarray(audio)
        samples = int(array.size)
        return f"{self.config.message}: {samples} samples"
