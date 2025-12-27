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

"""Utility helpers for saving audio and common engine tasks.

This module currently contains a small helper for writing a normalized NumPy
array to a WAV file using standard library facilities.
"""

import numpy as np
from numpy.typing import NDArray

import logging

logger = logging.getLogger(__name__)


def nparray_to_audio_file(
    audio: NDArray[np.float32], file_path: str, sample_rate: int = 16000
) -> None:
    """Write a float32 waveform to a WAV file as 16-bit PCM.

    The input is treated as a Standard ASR–normalized waveform (dtype
    ``np.float32``, values in roughly ``[-1.0, 1.0]``). Values are clipped to
    ``[-1, 1]`` and linearly mapped to signed 16-bit PCM for storage. Mono input
    uses 1 channel; 2D input is interpreted as ``(n_samples, n_channels)``.

    Args:
        audio (NDArray[np.float32]): Waveform array to save. Mono can be 1D.
        file_path (str): Destination path for the ``.wav`` file.
        sample_rate (int): Sample rate to write (Hz). Defaults to ``16000``.

    Raises:
        OSError: If writing to ``file_path`` fails (permissions, disk, etc.).
    """
    import wave

    # Make sure the audio is in the range [-1, 1]
    audio = np.clip(audio, -1, 1)
    # Convert the audio to 16-bit PCM
    audio_integer: NDArray[np.int16] = (audio * 32767).astype(np.int16)

    if audio_integer.ndim == 1:
        channels = 1
        frames = audio_integer.tobytes()
    else:
        channels = int(audio_integer.shape[1])
        frames = audio_integer.reshape(-1, channels).tobytes()

    try:
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 2 bytes = 16 bits
            wf.setframerate(sample_rate)
            wf.writeframes(frames)
    except OSError as e:
        logger.error("Error writing audio to file %s: %s", file_path, e)
        raise
