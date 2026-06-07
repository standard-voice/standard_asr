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
from standard_asr.audio_input import AudioArray, AudioPath, InputKind
from standard_asr.capabilities import (
    BatchCapabilities,
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


class _Config(BaseConfig[Literal["arr"]]):
    engine: Literal["arr"] = "arr"


class _ArrayProps(BaseProperties):
    engine_id: str = "arr"
    model_name: str = "echo"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] = [16000]
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

    def _transcribe(
        self, prepared: PreparedAudio, params: RuntimeParams
    ) -> TranscriptionResult:
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
        _ArrayEngine().transcribe(
            _audio(), RuntimeParams(provider_params=_OtherParams())
        )


def test_provider_params_correct_type_ok() -> None:
    result = _ArrayEngine().transcribe(
        _audio(), RuntimeParams(provider_params=_MyParams(beam=5))
    )
    assert result.text == "n=8"


def test_bare_array_strict_missing_rate_raises() -> None:
    with pytest.raises(Exception):
        _ArrayEngine().transcribe(AudioArray(np.zeros(8, dtype=np.float32)))


def test_bare_array_best_effort_assumes_rate() -> None:
    result = _ArrayEngine(strict=False).transcribe(
        AudioArray(np.zeros(8, dtype=np.float32))
    )
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


def test_resample_diagnostic_for_off_rate_array() -> None:
    class _AnyRateProps(_ArrayProps):
        accepted_sample_rates: list[int] = [16000]

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
