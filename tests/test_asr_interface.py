# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the EngineBase transcribe pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
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
from standard_asr.audio_format import AudioFormat
from standard_asr.audio_input import AudioArray, AudioPath, AudioUrl, InputKind
from standard_asr.audio_negotiation import UnsafeAudioUrlError
from standard_asr.capabilities import (
    BatchCapabilities,
    CandidateLanguagesCap,
    CandidateLanguagesConstraints,
    DeclaredCapabilities,
    FlagCap,
    LanguageCaps,
    StreamingCapabilities,
)
from standard_asr.exceptions import (
    AudioProcessingError,
    IncompatibleAudioInputError,
    InvalidProviderParamError,
    UnsupportedFeatureError,
)
from standard_asr.runtime_params import ProviderParams, WordTimestampGranularity
from standard_asr.streaming import TranscriptionEvent, TranscriptionSession


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


class _CapturingArrayEngine(_ArrayEngine):
    """Array engine that records params handed to the engine hook."""

    received: ClassVar[RuntimeParams | None] = None

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        type(self).received = params
        return super()._transcribe(prepared, params)


def _audio() -> AudioArray:
    return AudioArray(np.zeros(8, dtype=np.float32), 16000)


def test_engine_is_standard_asr() -> None:
    assert isinstance(_ArrayEngine(), StandardASR)


def test_standard_asr_protocol_includes_full_surface() -> None:
    # The protocol now structurally describes the full spec surface: batch
    # (transcribe / transcribe_async) AND the streaming entry point
    # (start_transcription). A duck-typed object missing the streaming surface is
    # NOT a StandardASR, so structural typing covers the whole contract (INTE-5).
    class _Partial:
        config = _ArrayEngine().config
        properties = _ArrayProps()
        declared_capabilities = DeclaredCapabilities()

        def transcribe(self, audio: object, params: object = None) -> None:  # pragma: no cover
            return None

        async def transcribe_async(
            self, audio: object, params: object = None
        ) -> None:  # pragma: no cover
            return None

        def supports(self, dot_path: str) -> bool:  # pragma: no cover
            return False

    assert not isinstance(_Partial(), StandardASR)  # missing start_transcription


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


class _NoLangCapturingEngine(_CapturingArrayEngine):
    properties: ClassVar[BaseProperties] = _NoLangProps()

    def __init__(self) -> None:
        self.config = _NoLangConfig()


def test_no_language_axis_skips_default_language_check() -> None:
    result = _NoLangEngine().transcribe(_audio())
    assert result.text == "n=8"


def test_no_language_axis_leaves_runtime_params_unchanged() -> None:
    _NoLangCapturingEngine.received = None
    params = RuntimeParams(language="fr", candidate_languages=["zz"])

    _NoLangCapturingEngine().transcribe(_audio(), params)

    assert _NoLangCapturingEngine.received is params


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


class _AutoManyProps(_ArrayProps):
    selectable_languages: list[str] = ["auto"]
    detectable_languages: list[str] = ["en", "ja", "ko"]


class _CapturingAutoEngine(_CapturingArrayEngine):
    properties: ClassVar[BaseProperties] = _AutoManyProps()
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


def test_candidate_languages_best_effort_filtered_and_truncated_flow_to_batch_hook() -> None:
    _CapturingAutoEngine.received = None

    result = _CapturingAutoEngine(strict=False).transcribe(
        _audio(),
        RuntimeParams(language="auto", candidate_languages=["ja", "zz", "en", "ko"]),
    )

    assert _CapturingAutoEngine.received is not None
    assert _CapturingAutoEngine.received.language == "auto"
    assert _CapturingAutoEngine.received.candidate_languages == ["ja", "en"]
    assert any(d.code == "candidate_language_dropped" for d in result.diagnostics)
    assert any(d.code == "candidate_languages_truncated" for d in result.diagnostics)


class _NoCandConfig(LanguageConfigMixin, BaseConfig[Literal["arr"]]):
    engine: Literal["arr"] = "arr"
    default_language: str | None = "auto"


_NO_CAND_CAPS = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(
            runtime_override=FlagCap(supported=True),
            candidate_languages=CandidateLanguagesCap(supported=False),
        )
    )
)


class _NoCandEngine(_ArrayEngine):
    properties: ClassVar[BaseProperties] = _AutoProps()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _NO_CAND_CAPS

    def __init__(self, *, strict: bool = True) -> None:
        self.config = _NoCandConfig(strict=strict)


def test_unsupported_candidate_languages_strict_does_not_raise_single_diagnostic() -> None:
    # RUNT-1: §Language R3 step 2 -- an unsupported candidate_languages axis
    # resolves to None + exactly one diagnostic and never raises, even in strict.
    result = _NoCandEngine(strict=True).transcribe(
        _audio(), RuntimeParams(language="auto", candidate_languages=["en", "ja"])
    )
    # RUNT-2: exactly ONE diagnostic for this axis (no gate_params duplicate).
    cand_diags = [d for d in result.diagnostics if d.param == "candidate_languages"]
    assert len(cand_diags) == 1
    assert cand_diags[0].code == "candidate_languages_ignored"


class _EnglishOnlyProps(_ArrayProps):
    selectable_languages: list[str] = ["en"]
    detectable_languages: list[str] = []


class _EnglishOnlyEngine(_CapturingArrayEngine):
    properties: ClassVar[BaseProperties] = _EnglishOnlyProps()


def test_runtime_language_strict_rejects_non_selectable_language() -> None:
    _EnglishOnlyEngine.received = None

    with pytest.raises(UnsupportedFeatureError, match="not selectable") as exc_info:
        _EnglishOnlyEngine().transcribe(_audio(), RuntimeParams(language="fr"))

    assert exc_info.value.param == "language"
    assert exc_info.value.mode == "batch"
    assert _EnglishOnlyEngine.received is None


def test_runtime_language_best_effort_falls_back_to_default_for_engine_hook() -> None:
    _EnglishOnlyEngine.received = None

    result = _EnglishOnlyEngine(strict=False).transcribe(_audio(), RuntimeParams(language="fr"))

    assert _EnglishOnlyEngine.received is not None
    assert _EnglishOnlyEngine.received.language == "en"
    assert result.detected_language == "en"
    diag = next(d for d in result.diagnostics if d.code == "language_not_selectable")
    assert diag.param == "language"
    assert diag.provided == "fr"
    assert diag.effective == "en"


_NO_OVERRIDE_CAPS = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=False)),
    )
)


class _NoOverrideEngine(_CapturingArrayEngine):
    declared_capabilities: ClassVar[DeclaredCapabilities] = _NO_OVERRIDE_CAPS


def test_runtime_language_best_effort_unsupported_override_flows_default_to_hook() -> None:
    _NoOverrideEngine.received = None

    result = _NoOverrideEngine(strict=False).transcribe(_audio(), RuntimeParams(language="auto"))

    assert _NoOverrideEngine.received is not None
    assert _NoOverrideEngine.received.language == "en"
    assert result.detected_language == "en"
    assert any(d.code == "unsupported_parameter_ignored" for d in result.diagnostics)


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


class _WireProps(_ArrayProps):
    wire_encodings: list[str] | None = ["pcm_s16le", "mulaw"]


class _WireEngine(_ArrayEngine):
    properties: ClassVar[BaseProperties] = _WireProps()


def test_ensure_stream_format_supported_rejects_unknown_encoding() -> None:
    from standard_asr.audio_format import AudioFormat

    engine = _WireEngine()
    # A declared encoding at an accepted rate passes; an undeclared encoding is
    # fail-closed at session start (encoding is checked before the rate).
    engine.ensure_stream_format_supported(AudioFormat(encoding="mulaw", sample_rate=16000))
    with pytest.raises(UnsupportedFeatureError, match="wire encoding") as exc_info:
        engine.ensure_stream_format_supported(AudioFormat(encoding="opus", sample_rate=48000))
    assert exc_info.value.param == "audio_format.encoding"
    assert exc_info.value.mode == "streaming"
    assert exc_info.value.hint is not None


def test_ensure_stream_format_supported_skips_encoding_when_no_wire_encodings() -> None:
    # An engine that declares no wire_encodings cannot validate encoding; the
    # encoding check is a no-op. The sample-rate fail-closed still applies.
    from standard_asr.audio_format import AudioFormat

    _ArrayEngine().ensure_stream_format_supported(
        AudioFormat(encoding="anything", sample_rate=16000)
    )


def test_ensure_stream_format_supported_rejects_unreachable_sample_rate() -> None:
    # X-AU-5: v1 does NOT resample streaming wire frames, so a wire sample_rate
    # the engine does not accept must be rejected (fail-closed), not forwarded as
    # frames the engine never declared (silent mistranscription). _ArrayProps
    # accepts only [16000] and declares no required_input_sample_rate.
    from standard_asr.audio_format import AudioFormat

    engine = _ArrayEngine()
    with pytest.raises(UnsupportedFeatureError, match="wire sample_rate") as exc_info:
        engine.ensure_stream_format_supported(AudioFormat(encoding="pcm_s16le", sample_rate=44100))
    assert exc_info.value.param == "audio_format.sample_rate"
    assert exc_info.value.mode == "streaming"
    assert exc_info.value.hint is not None


def test_ensure_stream_format_supported_accepts_required_rate_not_in_list() -> None:
    # The wire sample_rate is accepted when it equals required_input_sample_rate
    # (the rate the engine's wire protocol hard-requires).
    from standard_asr.audio_format import AudioFormat

    class _ReqProps(_ArrayProps):
        native_sample_rate: int = 24000
        accepted_sample_rates: list[int] | Literal["any"] = [24000]
        required_input_sample_rate: int | None = 24000

    class _ReqEngine(_ArrayEngine):
        properties: ClassVar[BaseProperties] = _ReqProps()

    _ReqEngine().ensure_stream_format_supported(
        AudioFormat(encoding="pcm_s16le", sample_rate=24000)
    )


def test_ensure_stream_format_supported_allows_any_rate_for_any_engine() -> None:
    # When accepted_sample_rates is "any", no wire-rate constraint applies.
    from standard_asr.audio_format import AudioFormat

    class _AnyRateProps(_ArrayProps):
        accepted_sample_rates: list[int] | Literal["any"] = "any"

    class _AnyEngine(_ArrayEngine):
        properties: ClassVar[BaseProperties] = _AnyRateProps()

    _AnyEngine().ensure_stream_format_supported(
        AudioFormat(encoding="pcm_s16le", sample_rate=44100)
    )


def test_required_input_sample_rate_must_be_accepted() -> None:
    from pydantic import ValidationError

    # required rate present in accepted list: valid (native also in the list, so
    # the separate native-rate reachability invariant is satisfied).
    _ArrayProps(
        native_sample_rate=24000,
        accepted_sample_rates=[24000],
        required_input_sample_rate=24000,
    )
    # required rate absent from a concrete accepted list: contradictory engine.
    with pytest.raises(ValidationError, match="required_input_sample_rate"):
        _ArrayProps(
            accepted_sample_rates=[16000],
            required_input_sample_rate=24000,
        )
    # 'any' accepts every rate, so a required rate is always reachable.
    _ArrayProps(accepted_sample_rates="any", required_input_sample_rate=24000)


# --- RUNT-3: streaming runtime-param gating via the template seam -------------


class _StreamSession(TranscriptionSession):
    """Minimal session: ends immediately (the base appends ``done``)."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        return
        yield  # pragma: no cover - unreachable, marks this an async generator


class _StreamEngine(_ArrayEngine):
    """Streaming engine overriding the new ``_start_transcription`` hook."""

    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        ),
        streaming=StreamingCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        ),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
    )

    #: Captures the params the base handed to the hook (R5 freeze assertions).
    received: ClassVar[RuntimeParams | None] = None
    #: Captures the prepared whole-input audio handed to the hook.
    received_prepared_audio: ClassVar[PreparedAudio | None] = None
    #: Distinguishes "hook received None" from "hook was not reached".
    hook_called: ClassVar[bool] = False

    def _start_transcription(
        self,
        *,
        gated_params: RuntimeParams,
        audio_format: AudioFormat | None = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        type(self).received = gated_params
        type(self).received_prepared_audio = prepared_audio
        type(self).hook_called = True
        return _StreamSession()


class _NoStreamingInputEngine(_StreamEngine):
    """Streaming hook is present, but incremental input is not declared."""

    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        ),
        streaming_output=FlagCap(supported=True),
    )


class _NoStreamingOutputEngine(_StreamEngine):
    """Streaming hook is present, but whole-input streaming output is not declared."""

    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        ),
        streaming_input=FlagCap(supported=True),
    )


class _UrlStreamProps(_ArrayProps):
    accepted_input: set[InputKind] = {InputKind.FETCHABLE_URL}


class _UrlStreamEngine(_StreamEngine):
    properties: ClassVar[BaseProperties] = _UrlStreamProps()


class _EncodedStreamProps(_ArrayProps):
    accepted_input: set[InputKind] = {InputKind.ENCODED_BYTES}


class _EncodedStreamEngine(_StreamEngine):
    properties: ClassVar[BaseProperties] = _EncodedStreamProps()


def test_missing_streaming_input_capability_rejects_audio_format_session() -> None:
    _NoStreamingInputEngine.hook_called = False

    with pytest.raises(UnsupportedFeatureError) as exc_info:
        _NoStreamingInputEngine().start_transcription(
            audio_format=AudioFormat(encoding="pcm_s16le", sample_rate=16000),
        )

    error = exc_info.value
    message = str(error)
    assert error.param == "audio_format"
    assert error.mode == "streaming"
    assert "audio_format" in message
    assert "streaming mode" in message
    assert "streaming-input" in message
    assert "streaming_input" in message
    assert _NoStreamingInputEngine.hook_called is False


def test_missing_streaming_output_capability_rejects_whole_input_session() -> None:
    _NoStreamingOutputEngine.hook_called = False

    with pytest.raises(UnsupportedFeatureError) as exc_info:
        _NoStreamingOutputEngine().start_transcription(audio=_audio())

    error = exc_info.value
    message = str(error)
    assert error.param == "audio"
    assert error.mode == "streaming"
    assert "audio=" in message
    assert "whole-input streaming mode" in message
    assert "streaming-output" in message
    assert "streaming_output" in message
    assert _NoStreamingOutputEngine.hook_called is False


def test_declared_streaming_input_allows_audio_format_session() -> None:
    _NoStreamingOutputEngine.hook_called = False
    _NoStreamingOutputEngine.received_prepared_audio = None

    _NoStreamingOutputEngine().start_transcription(
        audio_format=AudioFormat(encoding="pcm_s16le", sample_rate=16000),
    )

    assert _NoStreamingOutputEngine.hook_called is True
    assert _NoStreamingOutputEngine.received_prepared_audio is None


def test_declared_streaming_output_allows_whole_input_session() -> None:
    _NoStreamingInputEngine.hook_called = False
    _NoStreamingInputEngine.received_prepared_audio = None

    _NoStreamingInputEngine().start_transcription(audio=_audio())

    assert _NoStreamingInputEngine.hook_called is True
    assert _NoStreamingInputEngine.received_prepared_audio is not None


def test_whole_input_streaming_audio_url_rejected_by_ssrf_before_hook() -> None:
    _UrlStreamEngine.hook_called = False

    with pytest.raises(UnsafeAudioUrlError):
        _UrlStreamEngine().start_transcription(audio=AudioUrl("https://127.0.0.1/a.wav"))

    assert _UrlStreamEngine.hook_called is False


def test_whole_input_streaming_bare_array_strict_requires_sample_rate() -> None:
    _StreamEngine.hook_called = False

    with pytest.raises(AudioProcessingError, match="no sample rate"):
        _StreamEngine(strict=True).start_transcription(
            audio=np.zeros(8, dtype=np.float32),
        )

    assert _StreamEngine.hook_called is False


def test_whole_input_streaming_bare_array_best_effort_assumes_rate_and_diagnoses() -> None:
    _StreamEngine.hook_called = False
    _StreamEngine.received_prepared_audio = None

    session = _StreamEngine(strict=False).start_transcription(
        audio=np.zeros(8, dtype=np.float32),
    )

    prepared = _StreamEngine.received_prepared_audio
    assert _StreamEngine.hook_called is True
    assert prepared is not None
    assert prepared.kind is InputKind.ARRAY
    assert prepared.sample_rate == 16000
    assert any(d.code == "assumed_sample_rate" for d in session.diagnostics())


def test_whole_input_streaming_array_to_encoded_hook_receives_prepared_audio() -> None:
    _EncodedStreamEngine.hook_called = False
    _EncodedStreamEngine.received_prepared_audio = None

    _EncodedStreamEngine().start_transcription(
        audio=AudioArray(np.zeros(8, dtype=np.float32), 16000),
    )

    prepared = _EncodedStreamEngine.received_prepared_audio
    assert _EncodedStreamEngine.hook_called is True
    assert isinstance(prepared, PreparedAudio)
    assert prepared.kind is InputKind.ENCODED_BYTES
    assert prepared.array is None
    assert prepared.data is not None
    assert prepared.data.startswith(b"RIFF")
    assert prepared.container == "wav"


def test_incremental_streaming_audio_format_hook_receives_no_prepared_audio() -> None:
    _StreamEngine.hook_called = False
    _StreamEngine.received_prepared_audio = None

    _StreamEngine().start_transcription(
        audio_format=AudioFormat(encoding="pcm_s16le", sample_rate=16000),
    )

    assert _StreamEngine.hook_called is True
    assert _StreamEngine.received_prepared_audio is None


def test_streaming_provider_params_swap_raises() -> None:
    # R3 swap-safety on the streaming path: a wrong provider_params type ALWAYS
    # raises, regardless of strict/best_effort, before any session is built.
    class _OtherParams(ProviderParams):
        x: int = 0

    _StreamEngine.received = None
    with pytest.raises(InvalidProviderParamError):
        _StreamEngine(strict=False).start_transcription(
            params=RuntimeParams(provider_params=_OtherParams())
        )
    assert _StreamEngine.received is None  # never reached the hook


def test_streaming_unsupported_param_strict_raises() -> None:
    # word_timestamps is not declared in streaming caps -> strict raises.
    with pytest.raises(UnsupportedFeatureError):
        _StreamEngine(strict=True).start_transcription(
            params=RuntimeParams(word_timestamps=WordTimestampGranularity.WORD)
        )


def test_streaming_unsupported_param_best_effort_drops_and_diagnoses() -> None:
    # best_effort drops the param and surfaces a diagnostic via the session.
    session = _StreamEngine(strict=False).start_transcription(
        params=RuntimeParams(word_timestamps=WordTimestampGranularity.WORD)
    )
    assert any(d.code == "unsupported_parameter_ignored" for d in session.diagnostics())
    # R5 freeze: the gated (frozen) params handed to the hook have the
    # unsupported param dropped.
    assert _StreamEngine.received is not None
    assert _StreamEngine.received.word_timestamps is None


def test_streaming_gated_params_flow_to_hook() -> None:
    # A supported param flows through unchanged to the hook (R5 freeze).
    _StreamEngine.received = None
    _StreamEngine().start_transcription(params=RuntimeParams(language="en"))
    assert _StreamEngine.received is not None
    assert _StreamEngine.received.language == "en"


def test_streaming_candidate_language_effective_params_flow_to_hook_and_diagnose() -> None:
    # Candidate-language resolution updates the hook params and surfaces
    # diagnostics through the session.
    class _StreamAutoEngine(_StreamEngine):
        properties: ClassVar[BaseProperties] = _AutoProps()
        declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
            streaming=StreamingCapabilities(
                language=LanguageCaps(
                    runtime_override=FlagCap(supported=True),
                    candidate_languages=CandidateLanguagesCap(
                        supported=True,
                        constraints=CandidateLanguagesConstraints(max=2),
                    ),
                ),
            ),
        )

        def __init__(self, *, strict: bool = True) -> None:
            self.config = _AutoConfig(strict=strict)

    _StreamAutoEngine.received = None
    session = _StreamAutoEngine(strict=False).start_transcription(
        params=RuntimeParams(language="auto", candidate_languages=["en", "zz"])
    )
    assert _StreamAutoEngine.received is not None
    assert _StreamAutoEngine.received.language == "auto"
    assert _StreamAutoEngine.received.candidate_languages == ["en"]
    assert any(d.code == "candidate_language_dropped" for d in session.diagnostics())


def test_streaming_validates_wire_format() -> None:
    # The base template validates the wire format (fail-closed) before building.
    from standard_asr.audio_format import AudioFormat

    with pytest.raises(UnsupportedFeatureError, match="wire sample_rate"):
        _StreamEngine().start_transcription(
            audio_format=AudioFormat(encoding="pcm_s16le", sample_rate=44100)
        )


def test_non_streaming_engine_raises_unsupported_without_param_error() -> None:
    # A batch-only engine (no _start_transcription override) reports "does not
    # support streaming" even when given an unsupported param -- the unsupported
    # streaming error wins over any param/wire error, and no gating runs.
    with pytest.raises(UnsupportedFeatureError, match="does not support streaming"):
        _ArrayEngine().start_transcription(
            params=RuntimeParams(word_timestamps=WordTimestampGranularity.WORD)
        )


def test_base_start_transcription_hook_raises_unsupported() -> None:
    # The base _start_transcription hook is a defensive raise: it is normally
    # unreachable (the template guards on _overrides_streaming first), but a
    # subclass that calls super()._start_transcription() must get the clear error.
    with pytest.raises(UnsupportedFeatureError, match="does not support streaming"):
        EngineBase._start_transcription(  # pyright: ignore[reportPrivateUsage]
            _ArrayEngine(),
            gated_params=RuntimeParams(),
            audio_format=None,
            prepared_audio=None,
        )


def test_non_streaming_engine_runs_exclusivity_guard_first() -> None:
    # Even for a non-streaming engine, the mutual-exclusion guard runs first
    # (preserving the prior behaviour): passing both inputs raises ValueError,
    # not the unsupported-streaming error.
    from standard_asr.audio_format import AudioFormat

    fmt = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _ArrayEngine().start_transcription(audio_format=fmt, audio=_audio())
