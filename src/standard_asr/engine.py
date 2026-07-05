# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Engine-author surface: everything you need to build a compliant ASR plugin.

This module is the **single import path for engine authors**. Where the top-level
``standard_asr`` namespace is curated for *application* developers (discover an
engine, pass audio, read a result), ``standard_asr.engine`` aggregates the types
an *engine* author implements and declares against:

- the base class and protocol (:class:`EngineBase`, :class:`StandardASR`),
  plus the standard session-establishment wire-format rule
  (:func:`ensure_wire_format_supported`) for structural engines that implement
  their own establishment guard;
- the typed config surface (:class:`BaseConfig`, the applicability mixins,
  :func:`secret_field`);
- static metadata (:class:`BaseProperties`, :class:`SampleRateRange`,
  :class:`InputKind`), and the :func:`sample_rate_accepted` /
  :func:`nearest_accepted_sample_rate` helpers an engine reuses so its
  ``accepted_sample_rates`` membership and resample-target choices match the
  standard's;
- the full capability vocabulary (:class:`DeclaredCapabilities` and every
  ``*Cap`` / ``*Constraints`` node);
- language resolution and download-policy helpers (:func:`effective_language`,
  :data:`AUTO`, :func:`resolve_download_root`);
- the result and streaming types an engine constructs and emits, plus the
  wire projection helper (:func:`to_json_value`) for values headed into a
  wire-visible slot.

Exceptions an engine raises live in :mod:`standard_asr.contract.exceptions` (and are also
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

from standard_asr.audio.conversion import PreparedAudio
from standard_asr.audio.format import AudioFormat
from standard_asr.audio.input import InputKind
from standard_asr.contract.capabilities import (
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
from standard_asr.contract.language import (
    AUTO,
    effective_candidate_languages,
    effective_language,
    normalize_bcp47,
)
from standard_asr.contract.params import (
    DIARIZE,
    DiarizationRequest,
    ProviderParams,
    RuntimeParams,
    WordTimestampGranularity,
)
from standard_asr.contract.properties import (
    BaseProperties,
    SampleRateRange,
    nearest_accepted_sample_rate,
    sample_rate_accepted,
)
from standard_asr.contract.results import (
    ChannelResult,
    Diagnostic,
    Segment,
    TranscriptionResult,
    Word,
    to_json_value,
)
from standard_asr.runtime.config import (
    BaseConfig,
    CredentialsConfigMixin,
    DeviceConfigMixin,
    DownloadConfigMixin,
    LanguageConfigMixin,
    env_var_name,
    secret_field,
)
from standard_asr.runtime.downloads import allow_downloads, resolve_download_root
from standard_asr.runtime.gating import Mode
from standard_asr.runtime.interface import EngineBase, StandardASR, ensure_wire_format_supported
from standard_asr.runtime.streaming import TranscriptionEvent, TranscriptionSession

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
    "DIARIZE",
    "DeclaredCapabilities",
    "DeviceConfigMixin",
    "Diagnostic",
    "DiarizationCap",
    "DiarizationConstraints",
    "DiarizationRequest",
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
    "ensure_wire_format_supported",
    "env_var_name",
    "granularity_offers_all",
    "nearest_accepted_sample_rate",
    "normalize_bcp47",
    "resolve_download_root",
    "sample_rate_accepted",
    "secret_field",
    "to_json_value",
]
