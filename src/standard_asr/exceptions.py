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


class StandardASRError(Exception):
    """Base exception class for all errors raised by the standard_asr library."""

    pass


class ConfigError(StandardASRError, ValueError):
    """Raised when there is an error in the user-provided configuration."""

    pass


class TranscriptionError(StandardASRError):
    """Raised when an error occurs during the transcription process."""

    pass


class AudioProcessingError(StandardASRError):
    """
    Raised when an error occurs during audio loading or processing.
    This is typically raised by functions in the audio_loader module.
    """

    pass


class FFmpegNotFoundError(AudioProcessingError, FileNotFoundError):
    """Raised when FFmpeg is required but not found in the system `PATH`."""

    pass


class FFprobeNotFoundError(AudioProcessingError, FileNotFoundError):
    """Raised when FFprobe is required but not found in the system `PATH`."""

    pass


class DiscoveryError(StandardASRError):
    """Base class for discovery and plugin-related errors."""

    pass


class EntrypointValidationError(DiscoveryError, ValueError):
    """Raised when an entry point name or metadata is invalid."""

    pass


class FactoryLoadError(DiscoveryError, ImportError):
    """Raised when an entry point target cannot be imported or is not callable."""

    pass
