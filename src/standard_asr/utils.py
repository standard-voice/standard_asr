"""Useful utility functions for ASR engines. Use them to simplify common tasks."""

import numpy as np
from numpy.typing import DTypeLike


# numpy datatype check
def ensure_datatype(audio: np.ndarray, data_type: DTypeLike = np.float32) -> np.ndarray:
    """Ensure the audio numpy array is of the specified data type.

    Args:
        audio: The audio data as a NumPy array.
        data_type: The target NumPy data type. Defaults to np.float32. None if unsure or unspecified.

    Returns:
        The audio array, converted to the specified data type if necessary.
    """
    if audio.dtype != data_type:
        audio = audio.astype(data_type)
    return audio


def nparray_to_audio_file(
    audio: np.ndarray, file_path: str, sample_rate: int = 16000
) -> None:
    """Convert a numpy array of audio data to a .wav file.

    Args:
        audio: The numpy array of audio data.
        file_path: The path to save the .wav file.
        sample_rate: The sample rate of the audio data.

    Raises:
        OSError: If the file cannot be written to the specified path.
    """
    import logging
    import wave

    # Make sure the audio is in the range [-1, 1]
    audio = np.clip(audio, -1, 1)
    # Convert the audio to 16-bit PCM
    audio_integer = (audio * 32767).astype(np.int16)

    try:
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16 bits
            wf.setframerate(sample_rate)
            wf.writeframes(audio_integer.tobytes())
    except OSError as e:
        logging.error(f"❌ Error writing audio to file {file_path}: {e}")
        raise
