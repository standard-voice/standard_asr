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
Configuration models for ASR engines.

This module defines pydantic models used to configure supported Automatic Speech
Recognition (ASR) engines (local and remote). A discriminated union (field
'engine') enables ergonomic parsing of heterogeneous configuration payloads.

Public API:
- ModelType: Enum categorizing model locality.
- BaseConfig: Abstract base for engine configs.
- FasterWhisperConfig, AzureConfig: Concrete engine configs.
- ASRConfig: Discriminated union type for any supported config.
- parse_asr_config: Helper to parse arbitrary dicts into a concrete config.
"""

from __future__ import annotations

import logging

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


logger = logging.getLogger(__name__)

# ------------ Base Configuration Model ------------


class BaseConfig(BaseModel):
    """
    Base class for all ASR engine configuration models.
    ASR Config provides information about the initialization parameters needed to initialized an ASR engine.

    Attributes:
        engine (str): Discriminator identifying the target engine.
    """

    # configuring properties for BaseConfig pydantic model
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    engine: str = Field(
        ..., description="Unique name of the ASR engine (discriminator)."
    )

    # if you want to add language options, remember to use supported_language from
    # your asr properties to validate input

    # your custom properties
