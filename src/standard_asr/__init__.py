# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard ASR package."""

from .asr_config import BaseConfig
from .asr_interface import StandardASR
from .asr_properties import BaseProperties
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
from .results import Segment, TranscriptionResult, Word
from .runtime import allow_downloads, ensure_cache_dir, resolve_cache_dir
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
    "ComplianceIssue",
    "ComplianceReport",
    "ConversionOp",
    "ConversionPlan",
    "DeclaredCapabilities",
    "InputKind",
    "ModelRegistry",
    "ModelSpec",
    "NoViablePath",
    "Segment",
    "StandardASR",
    "StreamChunk",
    "StreamingASR",
    "StreamingCapabilities",
    "TranscriptionResult",
    "Word",
    "allow_downloads",
    "can_accept",
    "check_entrypoints",
    "coerce_audio_input",
    "discover_models",
    "ensure_cache_dir",
    "load_audio",
    "load_audio_from_bytes",
    "load_audio_from_path",
    "negotiate",
    "negotiate_or_raise",
    "normalize_audio",
    "parse_entrypoint_name",
    "pep503_normalize",
    "resolve_cache_dir",
]
