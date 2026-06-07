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


class IncompatibleAudioInputError(AudioProcessingError):
    """Raised when no viable conversion path exists for the provided audio.

    This happens when the shape an application provides cannot be negotiated
    into any shape the engine accepts (e.g. a local array given to an engine
    that only accepts a server-fetchable URL).

    Args:
        provided: Human-readable description of the provided input shape.
        accepted: The engine's accepted input kinds.
        hint: Actionable guidance for resolving the mismatch.
    """

    def __init__(self, provided: str, accepted: object, hint: str) -> None:
        self.provided = provided
        self.accepted = accepted
        self.hint = hint
        super().__init__(
            f"Cannot deliver {provided} to an engine that accepts {accepted}. {hint}"
        )


class UnsupportedFeatureError(StandardASRError):
    """Raised in strict mode when a requested standard feature is unsupported.

    In best_effort mode the unsupported parameter is ignored and a structured
    diagnostic is returned instead of raising.
    """

    pass


class InvalidProviderParamError(StandardASRError, ValueError):
    """Raised when ``provider_params`` are invalid for the target engine.

    Unlike standard-set parameters, ``provider_params`` errors are always raised
    regardless of strict / best_effort -- they indicate a code-level bug (such
    as passing one engine's params model to another after a swap).
    """

    pass


class StreamClosedError(StandardASRError):
    """Raised when audio is fed to a streaming session after it was closed."""

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
