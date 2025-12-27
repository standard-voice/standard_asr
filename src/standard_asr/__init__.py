"""Standard ASR package."""

from .asr_config import BaseConfig
from .asr_interface import StandardASR
from .asr_properties import BaseProperties
from .compliance import ComplianceIssue, ComplianceReport, check_entrypoints
from .discovery import (
    ModelRegistry,
    ModelSpec,
    discover_models,
    parse_entrypoint_name,
    pep503_normalize,
)
from .features import FeatureFlag
from .options import BaseTranscribeOptions
from .results import Segment, TranscriptionResult, Word
from .runtime import (
    allow_downloads,
    ensure_cache_dir,
    resolve_cache_dir,
    validate_audio_input,
)
from .streaming import StreamChunk, StreamingASR
from .utils.audio_loader import (
    load_audio,
    load_audio_from_path,
    load_audio_from_bytes,
    normalize_audio,
)

__all__ = [
    "StandardASR",
    "BaseConfig",
    "BaseProperties",
    "BaseTranscribeOptions",
    "TranscriptionResult",
    "Segment",
    "Word",
    "FeatureFlag",
    "StreamChunk",
    "StreamingASR",
    "check_entrypoints",
    "ComplianceIssue",
    "ComplianceReport",
    "discover_models",
    "ModelRegistry",
    "ModelSpec",
    "parse_entrypoint_name",
    "pep503_normalize",
    "allow_downloads",
    "ensure_cache_dir",
    "resolve_cache_dir",
    "validate_audio_input",
    "load_audio",
    "load_audio_from_path",
    "load_audio_from_bytes",
    "normalize_audio",
]
