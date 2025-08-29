# Copyright 2025 The Standard ASR Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    This metadata needs to be available before initializing the ASR engine, so users of standard asr can make informed decisions about which engine to use.

    """

    model_name: str = Field(..., description="Name of the ASR model.")
    protocol_version: str = Field(
        ..., description="Version of the ASR protocol. Use semantic versioning"
    )
    # language field must be BCP 47 format.
    # If your ASR support other formats, implement conversion in your code.
    supported_language: list[str] = Field(
        ...,
        title="Supported Languages",
        description="List of supported languages of this ASR engine in IETF BCP 47 format. If your ASR support other formats (like ISO 639-1), implement conversion in your code.",
    )
    supported_device: list[str]
    audio_dtype: DTypeLike = Field(
        np.float32,
        description="Data type of the audio input. This is most likely float32 in audio field. Put None if unsure or unspecified.",
    )
