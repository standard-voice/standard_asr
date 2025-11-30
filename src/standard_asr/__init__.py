"""
Standard ASR package.
"""

from .asr_interface import StandardASR
from .asr_config import BaseConfig
from .compliance import (
    ComplianceIssue,
    ComplianceReport,
    check_entrypoints,
)
from .discovery import (
    ModelRegistry,
    ModelSpec,
    discover_models,
    parse_entrypoint_name,
    pep503_normalize,
)
from .utils.audio_loader import (
    load_audio,
    load_audio_from_path,
    load_audio_from_bytes,
    normalize_audio,
)

__all__ = [
    "StandardASR",
    "BaseConfig",
    "check_entrypoints",
    "ComplianceIssue",
    "ComplianceReport",
    "discover_models",
    "ModelRegistry",
    "ModelSpec",
    "parse_entrypoint_name",
    "pep503_normalize",
    "load_audio",
    "load_audio_from_path",
    "load_audio_from_bytes",
    "normalize_audio",
]
