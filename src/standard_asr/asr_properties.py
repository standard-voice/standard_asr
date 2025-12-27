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

"""Base properties of ASR engines."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .features import FeatureFlag
from .language import is_valid_bcp47, normalize_bcp47


class BaseProperties(BaseModel):
    """Base class for ASR engine properties.

    Args:
        engine_id: Engine identifier (PEP 503 normalized).
        model_name: Model preset name within the engine.
        protocol_version: Standard ASR protocol version supported by the engine.
        supported_languages: Supported languages in BCP 47 format.
        supported_devices: Supported compute devices (e.g., cpu, cuda, mps).
        supported_sample_rates: Supported input sample rates in Hz.
        supported_channels: Supported channel counts.
        audio_dtype: Expected NumPy dtype name for audio input.
        features: Supported optional features.
        description: Optional human-readable description.
        extra: Engine-specific metadata.

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    engine_id: str = Field(..., description="Engine identifier (PEP 503 normalized).")
    model_name: str = Field(..., description="Model preset name within the engine.")
    protocol_version: str = Field(
        ..., description="Standard ASR protocol version supported by the engine."
    )
    supported_languages: list[str] = Field(
        ..., description="Supported languages in BCP 47 format."
    )
    supported_devices: list[str] = Field(
        ..., description="Supported compute devices (e.g., cpu, cuda, mps)."
    )
    supported_sample_rates: list[int] = Field(
        default_factory=lambda: [16000],
        description="Supported input sample rates in Hz.",
    )
    supported_channels: list[int] = Field(
        default_factory=lambda: [1],
        description="Supported channel counts.",
    )
    audio_dtype: str = Field(
        "float32",
        description="Expected NumPy dtype name for audio input (e.g., float32).",
    )
    features: set[FeatureFlag] = Field(
        default_factory=set,
        description="Supported optional feature flags.",
    )
    description: str | None = Field(
        None, description="Optional human-readable description of the engine/model."
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific metadata."
    )

    @field_validator("supported_languages")
    @classmethod
    def _validate_languages(cls, value: list[str]) -> list[str]:
        """Validate and normalize supported language tags.

        Args:
            value: List of language tags.

        Returns:
            Normalized list of tags.

        Raises:
            ValueError: If any tag is invalid.
        """
        normalized: list[str] = []
        for tag in value:
            if not is_valid_bcp47(tag):
                raise ValueError(f"Invalid BCP 47 language tag: {tag!r}")
            normalized.append(normalize_bcp47(tag))
        if not normalized:
            raise ValueError("supported_languages must not be empty.")
        return normalized

    @field_validator("supported_sample_rates")
    @classmethod
    def _validate_sample_rates(cls, value: list[int]) -> list[int]:
        """Validate supported sample rates.

        Args:
            value: List of sample rates.

        Returns:
            The validated list of sample rates.

        Raises:
            ValueError: If any sample rate is invalid.
        """
        if not value:
            raise ValueError("supported_sample_rates must not be empty.")
        for rate in value:
            if rate <= 0:
                raise ValueError("supported_sample_rates must be positive.")
        return value

    @field_validator("supported_channels")
    @classmethod
    def _validate_channels(cls, value: list[int]) -> list[int]:
        """Validate supported channel counts.

        Args:
            value: List of channel counts.

        Returns:
            The validated list of channel counts.

        Raises:
            ValueError: If any channel count is invalid.
        """
        if not value:
            raise ValueError("supported_channels must not be empty.")
        for channels in value:
            if channels <= 0:
                raise ValueError("supported_channels must be positive.")
        return value

    @property
    def numpy_dtype(self) -> np.dtype[np.generic]:
        """Return the NumPy dtype object for ``audio_dtype``.

        Args:
            None.

        Returns:
            NumPy dtype instance.

        Raises:
            TypeError: If ``audio_dtype`` is not a valid dtype.
        """
        return np.dtype(self.audio_dtype)

    @property
    def model_id(self) -> str:
        """Return the fully qualified model identifier (engine/model).

        Args:
            None.

        Returns:
            Model identifier in ``engine/model`` format.

        Raises:
            None.
        """
        return f"{self.engine_id}/{self.model_name}"


__all__ = ["BaseProperties"]
