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
Base Configuration models for ASR engines.
"""

from __future__ import annotations

import logging
from typing import Generic, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


logger = logging.getLogger(__name__)

# Generic type variable for engine name (discriminator)
EngineNameT = TypeVar("EngineNameT", bound=str, covariant=True)

# ------------ Base Configuration Model ------------


class BaseConfig(BaseModel, Generic[EngineNameT]):
    """
    Base class for all ASR engine configuration models.
    ASR Config provides information about the initialization parameters needed to initialized an ASR engine.

    Attributes:
        engine (str): Discriminator identifying the target engine.
    """

    # properties for BaseConfig pydantic model
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    engine: EngineNameT = Field(
        ..., description="Unique name of the ASR engine (discriminator or identifier)."
    )  #! 这玩意儿是不是应该放在 properties 里 而不是配置里？ 这玩意儿类似身份证名字啊，config 又不是身份证

    # if you want to add language options, remember to use supported_language from
    # your asr properties to validate input

    # your custom properties
