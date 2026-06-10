# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Engine-author surface: everything you need to build a compliant ASR plugin.

This module is the **single import path for engine authors**. Where the top-level
``standard_asr`` namespace is curated for *application* developers (discover an
engine, pass audio, read a result), ``standard_asr.engine`` aggregates the types
an *engine* author implements and declares against:

- the base class and protocol (:class:`EngineBase`, :class:`StandardASR`);
- the typed config surface (:class:`BaseConfig`, the applicability mixins,
  :func:`secret_field`);
- static metadata (:class:`BaseProperties`, :class:`SampleRateRange`,
  :class:`InputKind`);
- the full capability vocabulary (:class:`DeclaredCapabilities` and every
  ``*Cap`` / ``*Constraints`` node);
- language resolution and download-policy helpers (:func:`effective_language`,
  :data:`AUTO`, :func:`resolve_download_root`);
- the result and streaming types an engine constructs and emits.

Exceptions an engine raises live in :mod:`standard_asr.exceptions` (and are also
re-exported at the package top level). Compliance helpers for testing your plugin
live in :mod:`standard_asr.compliance`.

Example:
    >>> from standard_asr.engine import (
    ...     EngineBase,
    ...     BaseConfig,
    ...     BaseProperties,
    ...     DeclaredCapabilities,
    ...     BatchCapabilities,
    ...     LanguageCaps,
    ...     FlagCap,
    ... )
"""

from __future__ import annotations

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
from .asr_properties import BaseProperties, SampleRateRange
from .audio_conversion import PreparedAudio
from .audio_format import AudioFormat
from .audio_input import InputKind
from .capabilities import (
    BatchCapabilities,
    CandidateLanguagesCap,
    CandidateLanguagesConstraints,
    DeclaredCapabilities,
    DiarizationCap,
    DiarizationConstraints,
    FinalityCap,
    FlagCap,
    GuidanceCaps,
    LanguageCaps,
    PhraseHintsCap,
    PhraseHintsConstraints,
    PromptCap,
    PromptConstraints,
    ReconnectCap,
    StreamingCapabilities,
    StreamingGuidanceCaps,
    StreamTimestampsCap,
    WordTimestampGranularityName,
    WordTimestampsCap,
    granularity_offers_all,
)
from .language import AUTO, effective_candidate_languages, effective_language, normalize_bcp47
from .param_gating import Mode
from .results import ChannelResult, Diagnostic, Segment, TranscriptionResult, Word
from .runtime import allow_downloads, resolve_download_root
from .runtime_params import ProviderParams, RuntimeParams, WordTimestampGranularity
from .streaming import TranscriptionEvent, TranscriptionSession

__all__ = [
    "AUTO",
    "AudioFormat",
    "BaseConfig",
    "BaseProperties",
    "BatchCapabilities",
    "CandidateLanguagesCap",
    "CandidateLanguagesConstraints",
    "ChannelResult",
    "CredentialsConfigMixin",
    "DeclaredCapabilities",
    "DeviceConfigMixin",
    "Diagnostic",
    "DiarizationCap",
    "DiarizationConstraints",
    "DownloadConfigMixin",
    "EngineBase",
    "FinalityCap",
    "FlagCap",
    "GuidanceCaps",
    "InputKind",
    "LanguageCaps",
    "LanguageConfigMixin",
    "Mode",
    "PhraseHintsCap",
    "PhraseHintsConstraints",
    "PreparedAudio",
    "PromptCap",
    "PromptConstraints",
    "ProviderParams",
    "ReconnectCap",
    "RuntimeParams",
    "SampleRateRange",
    "Segment",
    "StandardASR",
    "StreamTimestampsCap",
    "StreamingCapabilities",
    "StreamingGuidanceCaps",
    "TranscriptionEvent",
    "TranscriptionResult",
    "TranscriptionSession",
    "Word",
    "WordTimestampGranularity",
    "WordTimestampGranularityName",
    "WordTimestampsCap",
    "allow_downloads",
    "effective_candidate_languages",
    "effective_language",
    "env_var_name",
    "granularity_offers_all",
    "normalize_bcp47",
    "resolve_download_root",
    "secret_field",
]
