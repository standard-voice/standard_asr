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

"""Protocol definitions for Standard ASR engines."""

import asyncio
from typing import Any, ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

from .asr_config import BaseConfig
from .asr_properties import BaseProperties
from .options import BaseTranscribeOptions
from .results import TranscriptionResult


class StandardASR(Protocol):
    """Protocol defining the interface for ASR implementations.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    config: BaseConfig[str]
    properties: ClassVar[BaseProperties]

    def transcribe(
        self,
        audio: NDArray[np.float32],
        options: BaseTranscribeOptions | dict[str, Any] | None = None,
    ) -> TranscriptionResult:
        """Transcribe a pre-processed audio waveform into structured text.

        Args:
            audio: Audio waveform as ``np.float32`` following the Standard ASR contract.
            options: Optional per-request inference options (model or dict).

        Returns:
            Structured transcription result.

        Raises:
            ValueError: If input audio does not match engine capabilities.
            TranscriptionError: If transcription fails.
        """
        raise NotImplementedError(  # pragma: no cover
            "transcribe method must be implemented by subclasses."
        )

    async def transcribe_async(
        self,
        audio: NDArray[np.float32],
        options: BaseTranscribeOptions | dict[str, Any] | None = None,
    ) -> TranscriptionResult:
        """Asynchronously transcribe an audio waveform.

        Args:
            audio: Audio waveform as ``np.float32`` following the Standard ASR contract.
            options: Optional per-request inference options (model or dict).

        Returns:
            Structured transcription result.

        Raises:
            ValueError: If input audio does not match engine capabilities.
            TranscriptionError: If transcription fails.
        """
        return await asyncio.to_thread(self.transcribe, audio, options)


__all__ = ["StandardASR"]
