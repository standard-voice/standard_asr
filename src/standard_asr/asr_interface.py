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

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe speech audio in numpy array format and return the transcription.

        Args:
            audio: The numpy array of the audio data to transcribe.

        Returns:
            The transcription result as a string.
        """
        ...

    async def transcribe_async(self, audio: np.ndarray) -> str:
        """Asynchronously transcribe speech audio in numpy array format.

        By default, this runs the synchronous transcribe in a coroutine.
        Implementations can override this method to provide true async implementation.

        Args:
            audio: The numpy array of the audio data to transcribe.

        Returns:
            The transcription result as a string.
        """
        return await asyncio.to_thread(self.transcribe, audio)


