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


