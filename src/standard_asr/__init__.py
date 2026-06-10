# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard ASR -- the open interface between applications and ASR engines.

This top-level namespace is the **application-developer surface**: discover an
engine, hand it audio, read a constant-shape result, and (optionally) stream.
That is the whole 80% path::

    from standard_asr import discover_models, RuntimeParams

    registry = discover_models()
    engine = registry.create("faster-whisper/large-v3")
    result = engine.transcribe("meeting.wav", RuntimeParams(language="en"))
    print(result.text)

The deeper surfaces live in dedicated, audience-signaling submodules so the names
you reach for are never buried under names you don't:

- :mod:`standard_asr.engine` -- everything an **engine author** implements and
  declares (``EngineBase``, the config/properties surface, the full capability
  vocabulary).
- :mod:`standard_asr.compliance` -- the compliance checks an engine author runs
  against their plugin.
- granular modules (:mod:`standard_asr.wire`, :mod:`standard_asr.audio_negotiation`,
  ``...``) expose the framework internals for advanced use.
"""

from .asr_interface import StandardASR
from .audio_format import AudioFormat
from .audio_input import (
    AudioArray,
    AudioBase64,
    AudioBytes,
    AudioInput,
    AudioInputLike,
    AudioPath,
    AudioStorageUri,
    AudioUrl,
)
from .audio_negotiation import UnsafeAudioUrlError
from .discovery import ModelRegistry, ModelSpec, discover_models
from .exceptions import (
    AudioProcessingError,
    ConfigError,
    DiscoveryError,
    EntrypointValidationError,
    FactoryLoadError,
    FFmpegNotFoundError,
    FFprobeNotFoundError,
    IncompatibleAudioInputError,
    InvalidProviderParamError,
    InvalidSessionUseError,
    StandardASRError,
    StreamClosedError,
    StructuredError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from .renderers import to_srt, to_vtt
from .results import ChannelResult, Diagnostic, Segment, TranscriptionResult, Word
from .runtime_params import RuntimeParams, WordTimestampGranularity
from .streaming import (
    StreamDeadlines,
    SyncSession,
    TranscriptionEvent,
    TranscriptionSession,
)

__all__ = [
    "AudioArray",
    "AudioBase64",
    "AudioBytes",
    "AudioFormat",
    "AudioInput",
    "AudioInputLike",
    "AudioPath",
    "AudioProcessingError",
    "AudioStorageUri",
    "AudioUrl",
    "ChannelResult",
    "ConfigError",
    "Diagnostic",
    "DiscoveryError",
    "EntrypointValidationError",
    "FFmpegNotFoundError",
    "FFprobeNotFoundError",
    "FactoryLoadError",
    "IncompatibleAudioInputError",
    "InvalidProviderParamError",
    "InvalidSessionUseError",
    "ModelRegistry",
    "ModelSpec",
    "RuntimeParams",
    "Segment",
    "StandardASR",
    "StandardASRError",
    "StreamClosedError",
    "StreamDeadlines",
    "StructuredError",
    "SyncSession",
    "TranscriptionError",
    "TranscriptionEvent",
    "TranscriptionResult",
    "TranscriptionSession",
    "UnsafeAudioUrlError",
    "UnsupportedFeatureError",
    "Word",
    "WordTimestampGranularity",
    "discover_models",
    "to_srt",
    "to_vtt",
]
