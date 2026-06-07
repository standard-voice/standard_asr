# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the EngineBase transcribe pipeline."""

from __future__ import annotations

import asyncio
from typing import ClassVar, Literal

import numpy as np
import pytest

from standard_asr import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    PreparedAudio,
    RuntimeParams,
    StandardASR,
    TranscriptionResult,
)
from standard_asr.asr_config import LanguageConfigMixin
from standard_asr.audio_input import AudioArray, AudioPath, InputKind
from standard_asr.capabilities import (
    BatchCapabilities,
    CandidateLanguagesCap,
    CandidateLanguagesConstraints,
    DeclaredCapabilities,
    FlagCap,
    LanguageCaps,
)
from standard_asr.exceptions import (
    IncompatibleAudioInputError,
    InvalidProviderParamError,
    UnsupportedFeatureError,
)
from standard_asr.runtime_params import ProviderParams, WordTimestampGranularity


class _Config(LanguageConfigMixin, BaseConfig[Literal["arr"]]):
    engine: Literal["arr"] = "arr"
    default_language: str | None = "en"


class _ArrayProps(BaseProperties):
    engine_id: str = "arr"
    model_name: str = "echo"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | Literal["any"] = [16000]
    selectable_languages: list[str] = ["en", "auto"]
    detectable_languages: list[str] = ["en"]


_CAPS = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
    )
)


class _MyParams(ProviderParams):
    beam: int = 1


class _ArrayEngine(EngineBase):
    properties: ClassVar[BaseProperties] = _ArrayProps()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _CAPS
    provider_params_type: ClassVar[type[ProviderParams] | None] = _MyParams

    def __init__(self, *, strict: bool = True) -> None:
        self.config = _Config(strict=strict)

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        assert prepared.kind is InputKind.ARRAY
        assert prepared.array is not None
        return TranscriptionResult(
            text=f"n={prepared.array.size}", detected_language=params.language
        )


def _audio() -> AudioArray:
    return AudioArray(np.zeros(8, dtype=np.float32), 16000)


def test_engine_is_standard_asr() -> None:
    assert isinstance(_ArrayEngine(), StandardASR)


def test_transcribe_array_passthrough() -> None:
    result = _ArrayEngine().transcribe(_audio(), RuntimeParams(language="en"))
    assert result.text == "n=8"
    assert result.detected_language == "en"


def test_bare_tuple_coerced() -> None:
    result = _ArrayEngine().transcribe((np.zeros(4, dtype=np.float32), 16000))
    assert result.text == "n=4"


def test_incompatible_input_raises() -> None:
    from standard_asr.audio_input import AudioUrl

    with pytest.raises(IncompatibleAudioInputError):
        _ArrayEngine().transcribe(AudioUrl("https://x/a.wav"))


def test_unsupported_param_strict_raises() -> None:
    # word_timestamps is not declared supported.
    with pytest.raises(UnsupportedFeatureError):
        _ArrayEngine().transcribe(
            _audio(), RuntimeParams(word_timestamps=WordTimestampGranularity.WORD)
        )


def test_unsupported_param_best_effort_drops() -> None:
    result = _ArrayEngine(strict=False).transcribe(
        _audio(), RuntimeParams(word_timestamps=WordTimestampGranularity.WORD)
    )
    assert any(d.code == "unsupported_parameter_ignored" for d in result.diagnostics)


def test_provider_params_swap_raises() -> None:
    class _OtherParams(ProviderParams):
        x: int = 0

    with pytest.raises(InvalidProviderParamError):
        _ArrayEngine().transcribe(_audio(), RuntimeParams(provider_params=_OtherParams()))


def test_provider_params_correct_type_ok() -> None:
    result = _ArrayEngine().transcribe(_audio(), RuntimeParams(provider_params=_MyParams(beam=5)))
    assert result.text == "n=8"


def test_bare_array_strict_missing_rate_raises() -> None:
    with pytest.raises(Exception):
        _ArrayEngine().transcribe(AudioArray(np.zeros(8, dtype=np.float32)))


def test_bare_array_best_effort_assumes_rate() -> None:
    result = _ArrayEngine(strict=False).transcribe(AudioArray(np.zeros(8, dtype=np.float32)))
    assert any(d.code == "assumed_sample_rate" for d in result.diagnostics)


def test_streaming_unsupported_by_default() -> None:
    with pytest.raises(UnsupportedFeatureError):
        _ArrayEngine().start_transcription()


def test_transcribe_async() -> None:
    result = asyncio.run(_ArrayEngine().transcribe_async(_audio()))
    assert result.text == "n=8"


def test_supports_routes_to_effective() -> None:
    engine = _ArrayEngine()
    assert engine.supports("batch.language.runtime_override") is True
    assert engine.supports("batch.word_timestamps") is False


def test_candidate_max_returns_none_for_absent_mode_domain() -> None:
    # _CAPS declares only the batch domain; the streaming domain is None, so the
    # candidate-max lookup for that mode must short-circuit to None rather than
    # dereference a missing domain.
    engine = _ArrayEngine()
    assert engine._candidate_max("streaming") is None  # pyright: ignore[reportPrivateUsage]
    # The present batch domain has no candidate constraint -> also None.
    assert engine._candidate_max("batch") is None  # pyright: ignore[reportPrivateUsage]


def test_resample_diagnostic_for_off_rate_array() -> None:
    class _AnyRateProps(_ArrayProps):
        accepted_sample_rates: list[int] | Literal["any"] = [16000]

    class _Engine(_ArrayEngine):
        properties: ClassVar[BaseProperties] = _AnyRateProps()

    result = _Engine().transcribe(AudioArray(np.zeros(48000, dtype=np.float32), 48000))
    assert any(d.code == "resampled_with" for d in result.diagnostics)


def test_path_to_array_engine_needs_decode(tmp_path: object) -> None:
    # AudioPath to an array-only engine triggers a decode plan; with no real
    # file this raises an audio processing error (decode failure), proving the
    # negotiation routed to decode rather than failing closed.
    with pytest.raises(Exception):
        _ArrayEngine().transcribe(AudioPath("/nonexistent/file.wav"))


# --- H2: provider_params validated BEFORE audio decode (fail-fast) -----------


class _DecodeTracker(_ArrayEngine):
    """Engine that records whether audio decode was reached."""

    decoded: bool = False

    def _transcribe(
        self, prepared: PreparedAudio, params: RuntimeParams
    ) -> TranscriptionResult:  # pragma: no cover - should not run on bad params
        type(self).decoded = True
        return super()._transcribe(prepared, params)


def test_provider_params_rejected_before_audio_decode() -> None:
    from standard_asr.audio_input import AudioPath

    class _OtherParams(ProviderParams):
        x: int = 0

    _DecodeTracker.decoded = False
    # A nonexistent path would blow up in decode; the swapped provider_params
    # must be rejected first, so no decode is attempted.
    with pytest.raises(InvalidProviderParamError):
        _DecodeTracker().transcribe(
            AudioPath("/nonexistent/file.wav"),
            RuntimeParams(provider_params=_OtherParams()),
        )
    assert _DecodeTracker.decoded is False


def test_unsupported_param_strict_rejected_before_audio_decode() -> None:
    from standard_asr.audio_input import AudioPath

    _DecodeTracker.decoded = False
    with pytest.raises(UnsupportedFeatureError):
        _DecodeTracker().transcribe(
            AudioPath("/nonexistent/file.wav"),
            RuntimeParams(word_timestamps=WordTimestampGranularity.WORD),
        )
    assert _DecodeTracker.decoded is False


# --- default_language enforcement (IC.6 / LANG R1) ---------------------------


class _NoDefaultConfig(LanguageConfigMixin, BaseConfig[Literal["arr"]]):
    engine: Literal["arr"] = "arr"
    # default_language intentionally left None.


class _NoDefaultEngine(_ArrayEngine):
    def __init__(self) -> None:
        self.config = _NoDefaultConfig()


def test_language_axis_requires_default_language() -> None:
    from standard_asr.exceptions import ConfigError

    with pytest.raises(ConfigError, match="default_language"):
        _NoDefaultEngine().transcribe(_audio())


class _BadDefaultConfig(LanguageConfigMixin, BaseConfig[Literal["arr"]]):
    engine: Literal["arr"] = "arr"
    default_language: str | None = "fr"  # not in selectable_languages


class _BadDefaultEngine(_ArrayEngine):
    def __init__(self) -> None:
        self.config = _BadDefaultConfig()


def test_default_language_must_be_selectable() -> None:
    from standard_asr.exceptions import ConfigError

    with pytest.raises(ConfigError, match="selectable_languages"):
        _BadDefaultEngine().transcribe(_audio())


class _NoLangProps(_ArrayProps):
    selectable_languages: list[str] = []
    detectable_languages: list[str] = []


class _NoLangConfig(BaseConfig[Literal["arr"]]):
    engine: Literal["arr"] = "arr"


class _NoLangEngine(_ArrayEngine):
    properties: ClassVar[BaseProperties] = _NoLangProps()

    def __init__(self) -> None:
        self.config = _NoLangConfig()


def test_no_language_axis_skips_default_language_check() -> None:
    result = _NoLangEngine().transcribe(_audio())
    assert result.text == "n=8"


# --- H3: candidate-language validation runs in the standard layer ------------


_CAND_CAPS = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(
            runtime_override=FlagCap(supported=True),
            candidate_languages=CandidateLanguagesCap(
                supported=True,
                constraints=CandidateLanguagesConstraints(max=2),
            ),
        ),
    )
)


class _AutoProps(_ArrayProps):
    selectable_languages: list[str] = ["en", "auto"]
    detectable_languages: list[str] = ["en", "ja"]


class _AutoConfig(LanguageConfigMixin, BaseConfig[Literal["arr"]]):
    engine: Literal["arr"] = "arr"
    default_language: str | None = "auto"


class _AutoEngine(_ArrayEngine):
    properties: ClassVar[BaseProperties] = _AutoProps()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _CAND_CAPS

    def __init__(self, *, strict: bool = True) -> None:
        self.config = _AutoConfig(strict=strict)


def test_candidate_languages_strict_non_detectable_raises_in_base() -> None:
    with pytest.raises(ValueError, match="detectable"):
        _AutoEngine().transcribe(
            _audio(), RuntimeParams(language="auto", candidate_languages=["zz"])
        )


def test_candidate_languages_best_effort_emits_diagnostic_in_base() -> None:
    result = _AutoEngine(strict=False).transcribe(
        _audio(), RuntimeParams(language="auto", candidate_languages=["en", "zz"])
    )
    assert any(d.code == "candidate_language_dropped" for d in result.diagnostics)


# --- streaming mutual-exclusion guard ----------------------------------------


def test_start_transcription_rejects_both_inputs() -> None:
    from standard_asr.audio_format import AudioFormat

    fmt = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _ArrayEngine().start_transcription(audio_format=fmt, audio=_audio())


def test_start_transcription_guard_helper_is_static() -> None:
    from standard_asr.audio_format import AudioFormat

    fmt = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)
    # One input only is fine (no raise from the guard itself).
    EngineBase.ensure_stream_inputs_exclusive(fmt, None)
    EngineBase.ensure_stream_inputs_exclusive(None, _audio())
    EngineBase.ensure_stream_inputs_exclusive(None, None)
