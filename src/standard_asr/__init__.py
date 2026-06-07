# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard ASR package."""

from .asr_config import (
    BaseConfig,
    CredentialsConfigMixin,
    DeviceConfigMixin,
    DownloadConfigMixin,
    LanguageConfigMixin,
    env_var_name,
    secret_field,
)
from .asr_interface import EngineBase, StandardASR
from .asr_properties import BaseProperties
from .audio_conversion import PreparedAudio
from .audio_format import AudioFormat
from .audio_input import (
    AudioArray,
    AudioBase64,
    AudioBytes,
    AudioInput,
    AudioPath,
    AudioUrl,
    InputKind,
    coerce_audio_input,
)
from .audio_negotiation import (
    ConversionOp,
    ConversionPlan,
    NoViablePath,
    can_accept,
    negotiate,
    negotiate_or_raise,
)
from .capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    StreamingCapabilities,
)
from .compliance import ComplianceIssue, ComplianceReport, check_entrypoints
from .discovery import (
    ModelRegistry,
    ModelSpec,
    discover_models,
    parse_entrypoint_name,
    pep503_normalize,
)
from .language import AUTO
from .options import BaseTranscribeOptions
from .renderers import to_srt, to_vtt
from .results import (
    ChannelResult,
    Diagnostic,
    Segment,
    TranscriptionResult,
    Word,
)
from .runtime import allow_downloads, ensure_cache_dir, resolve_cache_dir
from .runtime_params import ProviderParams, RuntimeParams, WordTimestampGranularity
from .streaming import StreamChunk, StreamingASR
from .utils.audio_loader import (
    load_audio,
    load_audio_from_bytes,
    load_audio_from_path,
    normalize_audio,
)

__all__ = [
    "AUTO",
    "AudioArray",
    "AudioBase64",
    "AudioBytes",
    "AudioFormat",
    "AudioInput",
    "AudioPath",
    "AudioUrl",
    "BaseConfig",
    "BaseProperties",
    "BaseTranscribeOptions",
    "BatchCapabilities",
    "ChannelResult",
    "ComplianceIssue",
    "ComplianceReport",
    "ConversionOp",
    "ConversionPlan",
    "CredentialsConfigMixin",
    "DeclaredCapabilities",
    "DeviceConfigMixin",
    "Diagnostic",
    "DownloadConfigMixin",
    "EngineBase",
    "InputKind",
    "LanguageConfigMixin",
    "PreparedAudio",
    "ModelRegistry",
    "ModelSpec",
    "NoViablePath",
    "ProviderParams",
    "RuntimeParams",
    "Segment",
    "StandardASR",
    "StreamChunk",
    "StreamingASR",
    "StreamingCapabilities",
    "TranscriptionResult",
    "Word",
    "WordTimestampGranularity",
    "allow_downloads",
    "can_accept",
    "check_entrypoints",
    "coerce_audio_input",
    "discover_models",
    "ensure_cache_dir",
    "env_var_name",
    "load_audio",
    "load_audio_from_bytes",
    "load_audio_from_path",
    "negotiate",
    "negotiate_or_raise",
    "normalize_audio",
    "parse_entrypoint_name",
    "pep503_normalize",
    "resolve_cache_dir",
    "secret_field",
    "to_srt",
    "to_vtt",
]
