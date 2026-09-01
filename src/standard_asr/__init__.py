# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard ASR -- the open interface between applications and ASR engines.

This top-level namespace is the **application-developer surface**: discover an
engine, inspect its inference artifacts when needed, hand it audio, read a
constant-shape result, and optionally stream. That is the whole 80% path::

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
- granular modules (:mod:`standard_asr.audio.wire`, :mod:`standard_asr.audio.negotiation`,
  ``...``) expose the framework internals for advanced use.
"""

from standard_asr.audio.format import AudioFormat
from standard_asr.audio.input import (
    AudioArray,
    AudioBase64,
    AudioBytes,
    AudioInput,
    AudioInputLike,
    AudioPath,
    AudioStorageUri,
    AudioUrl,
)
from standard_asr.audio.negotiation import UnsafeAudioUrlError
from standard_asr.contract.artifacts import (
    ARTIFACT_ACTION_ACCEPT_TERMS,
    ARTIFACT_ACTION_AUTHENTICATE,
    ARTIFACT_ACTION_INSTALL_EXTERNAL,
    ARTIFACT_ACTION_OTHER,
    ARTIFACT_ACTION_PROVIDE_ARTIFACTS,
    ARTIFACT_ACTION_REQUEST_ACCESS,
    ARTIFACT_BLOCKER_ACTION_REQUIRED,
    ARTIFACT_BLOCKER_DOWNLOADS_DISABLED,
    ARTIFACT_BLOCKER_UNSUPPORTED,
    ARTIFACT_CORRUPT,
    ARTIFACT_INCOMPLETE,
    ARTIFACT_MISSING,
    ARTIFACT_PROGRESS_CONVERTING,
    ARTIFACT_PROGRESS_EXTRACTING,
    ARTIFACT_PROGRESS_FINALIZING,
    ARTIFACT_PROGRESS_RESOLVING,
    ARTIFACT_PROGRESS_TRANSFERRING,
    ARTIFACT_PROGRESS_UNIT_BYTES,
    ARTIFACT_PROGRESS_UNIT_FILES,
    ARTIFACT_PROGRESS_VERIFYING,
    ARTIFACT_READY,
    ARTIFACT_UNKNOWN,
    ARTIFACTS_NOT_APPLICABLE,
    ARTIFACTS_READY,
    ARTIFACTS_UNAVAILABLE,
    ARTIFACTS_UNKNOWN,
    ArtifactAcquisitionBlocker,
    ArtifactAction,
    ArtifactActionKind,
    ArtifactContext,
    ArtifactProgress,
    ArtifactProgressCallback,
    ArtifactProgressPhase,
    ArtifactProgressUnit,
    ArtifactReadiness,
    ArtifactReport,
    ArtifactRequirement,
    ArtifactState,
    HttpsUrl,
)
from standard_asr.contract.capabilities import ModeName
from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactProgressCallbackError,
    ArtifactStatusError,
    ArtifactUnavailableError,
    AudioProcessingError,
    ConfigError,
    ConfigurationRequiredError,
    DiscoveryError,
    EngineContractError,
    EntrypointValidationError,
    FactoryLoadError,
    FFmpegNotFoundError,
    FFprobeNotFoundError,
    IncompatibleAudioInputError,
    InvalidProviderParamError,
    InvalidSessionUseError,
    ProtocolCompatibilityError,
    StandardASRError,
    StreamClosedError,
    StructuredError,
    SubtitleRenderingError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from standard_asr.contract.params import (
    DIARIZE,
    DiarizationRequest,
    RuntimeParams,
    WordTimestampGranularity,
)
from standard_asr.contract.results import (
    DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE,
    ChannelResult,
    Diagnostic,
    Segment,
    TranscriptionResult,
    Word,
)
from standard_asr.plugins.discovery import ModelRegistry, ModelSpec, discover_models
from standard_asr.renderers import UnrenderablePolicy, to_srt, to_vtt
from standard_asr.runtime.interface import StandardASR, require_artifact_protocol
from standard_asr.runtime.streaming import (
    StreamDeadlines,
    SyncSession,
    TranscriptionEvent,
    TranscriptionSession,
)

__all__ = [
    "ARTIFACTS_NOT_APPLICABLE",
    "ARTIFACTS_READY",
    "ARTIFACTS_UNAVAILABLE",
    "ARTIFACTS_UNKNOWN",
    "ARTIFACT_ACTION_ACCEPT_TERMS",
    "ARTIFACT_ACTION_AUTHENTICATE",
    "ARTIFACT_ACTION_INSTALL_EXTERNAL",
    "ARTIFACT_ACTION_OTHER",
    "ARTIFACT_ACTION_PROVIDE_ARTIFACTS",
    "ARTIFACT_ACTION_REQUEST_ACCESS",
    "ARTIFACT_BLOCKER_ACTION_REQUIRED",
    "ARTIFACT_BLOCKER_DOWNLOADS_DISABLED",
    "ARTIFACT_BLOCKER_UNSUPPORTED",
    "ARTIFACT_CORRUPT",
    "ARTIFACT_INCOMPLETE",
    "ARTIFACT_MISSING",
    "ARTIFACT_PROGRESS_CONVERTING",
    "ARTIFACT_PROGRESS_EXTRACTING",
    "ARTIFACT_PROGRESS_FINALIZING",
    "ARTIFACT_PROGRESS_RESOLVING",
    "ARTIFACT_PROGRESS_TRANSFERRING",
    "ARTIFACT_PROGRESS_UNIT_BYTES",
    "ARTIFACT_PROGRESS_UNIT_FILES",
    "ARTIFACT_PROGRESS_VERIFYING",
    "ARTIFACT_READY",
    "ARTIFACT_UNKNOWN",
    "ArtifactAcquisitionBlocker",
    "ArtifactAcquisitionError",
    "ArtifactAction",
    "ArtifactActionKind",
    "ArtifactContext",
    "ArtifactProgress",
    "ArtifactProgressCallback",
    "ArtifactProgressCallbackError",
    "ArtifactProgressPhase",
    "ArtifactProgressUnit",
    "ArtifactReadiness",
    "ArtifactReport",
    "ArtifactRequirement",
    "ArtifactState",
    "ArtifactStatusError",
    "ArtifactUnavailableError",
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
    "ConfigurationRequiredError",
    "DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE",
    "DIARIZE",
    "Diagnostic",
    "DiarizationRequest",
    "DiscoveryError",
    "EngineContractError",
    "EntrypointValidationError",
    "FFmpegNotFoundError",
    "FFprobeNotFoundError",
    "FactoryLoadError",
    "IncompatibleAudioInputError",
    "InvalidProviderParamError",
    "InvalidSessionUseError",
    "HttpsUrl",
    "ModeName",
    "ProtocolCompatibilityError",
    "UnrenderablePolicy",
    "ModelRegistry",
    "ModelSpec",
    "RuntimeParams",
    "Segment",
    "StandardASR",
    "StandardASRError",
    "StreamClosedError",
    "StreamDeadlines",
    "StructuredError",
    "SubtitleRenderingError",
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
    "require_artifact_protocol",
    "to_srt",
    "to_vtt",
]
