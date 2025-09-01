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

import asyncio
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .config import BaseConfig
from .asr_properties import BaseProperties


class StandardASR(Protocol):
    """Protocol defining the interface for ASR (Automatic Speech Recognition) implementations.

    This protocol defines the expected methods that any ASR implementation should provide
    for transcribing audio data.
    """

    # Use config to type the configuration of the ASR engine,
    # which should also be used in the constructor to validate the input configuration.
    config: BaseConfig
    properties: BaseProperties

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        """
        Transcribes a pre-processed audio waveform into structured text.

        This is the core transcription method. It strictly adheres to the
        Standard ASR Audio Contract.

        Args:
            audio (NDArray[np.float32]): The audio waveform, which MUST conform to the
                following standard format:
                - dtype: np.float32
                - Sample Rate: 16,000 Hz
                - Value Range: [-1.0, 1.0]
                - Shape:
                    - Mono: (n_samples,)
                    - Multi-channel: (n_samples, n_channels)
                The number of channels MUST be one of the values listed in the
                ASR properties (`self.properties.supported_channels`).


        Returns:
            str: The transcribed text as a string.

        Raises:
            ValueError: If the input audio array's properties (e.g., shape)
                do not match the engine's capabilities.
            TranscriptionError: If the transcription process fails for any reason.
        """
        raise NotImplementedError(
            "transcribe method must be implemented by subclasses."
        )

    async def transcribe_async(self, audio: NDArray[np.float32]) -> str:
        """Asynchronously transcribes a pre-processed audio waveform into structured text.

        By default, this runs the synchronous `transcribe` in a separate thread.
        Implementations can override this method to provide a true async implementation.
        This is the core asynchronous transcription method. It strictly adheres to the
        Standard ASR Audio Contract.

        Args:
            audio (NDArray[np.float32]): The audio waveform, which MUST conform to the
                following standard format:
                - dtype: np.float32
                - Sample Rate: 16,000 Hz
                - Value Range: [-1.0, 1.0]
                - Shape:
                    - Mono: (n_samples,)
                    - Multi-channel: (n_samples, n_channels)
                The number of channels MUST be one of the values listed in the
                ASR properties (`self.properties.supported_channels`).

        Returns:
            The transcription result as a string.

        Raises:
            ValueError: If the input audio array's properties (e.g., shape)
                do not match the engine's capabilities.
            TranscriptionError: If the transcription process fails for any reason.
        """
        # Call the sync transcribe method in a separate thread
        return await asyncio.to_thread(self.transcribe, audio)
