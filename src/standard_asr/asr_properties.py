"""
Base properties of ASR engines.
"""

from pydantic import (
    BaseModel,
    Field,
)
import numpy as np
from numpy.typing import DTypeLike


class BaseProperties(BaseModel):
    """
    Base class for ASR engine properties.
    ASR Properties provide metadata about the ASR engine, including its capabilities and configuration.
    """

    model_name: str = Field(..., description="Name of the ASR model.")
    protocol_version: str = Field(
        ..., description="Version of the ASR protocol. Use semantic versioning"
    )
    supported_language: list[str] = Field(
        ..., description="List of supported languages in your ASR engine."
    )
    supported_device: list[str]
    audio_dtype: DTypeLike = Field(
        np.float32,
        description="Data type of the audio input. This is most likely float32 in audio field. Put None if unsure or unspecified.",
    )
