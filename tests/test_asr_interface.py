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
from standard_asr.engine import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    PreparedAudio,
    SampleRateRange,
)
from standard_asr.exceptions import (
    AudioProcessingError,
    IncompatibleAudioInputError,
    InvalidProviderParamError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from standard_asr.language import AUTO, DIAG_CANDIDATE_LANGUAGES_IGNORED
from standard_asr.runtime_params import ProviderParams, WordTimestampGranularity
from standard_asr.streaming import StreamDeadlines, TranscriptionEvent, TranscriptionSession


class _Config(LanguageConfigMixin, BaseConfig[Literal["arr"]]):
    engine: Literal["arr"] = "arr"
    default_language: str | None = "en"


class _ArrayProps(BaseProperties):
    engine_id: str = "arr"
    model_name: str = "echo"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
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
        # detected_language echoes the effective language so tests can observe
        # language resolution. The 'auto' directive is a request to detect, not
        # a detection result, so this stub (which performs no detection) reports
        # None for it -- mirroring what a real engine must do (TranscriptionResult
        # rejects 'auto' as a detected value).
        detected = None if params.language == AUTO else params.language
        return TranscriptionResult(text=f"n={prepared.array.size}", detected_language=detected)


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
    # NOT a StandardASR, so structural typing covers the whole contract.
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


class _UrlProps(_ArrayProps):
    accepted_input: set[InputKind] = {InputKind.FETCHABLE_URL}


class _UrlConfig(LanguageConfigMixin, BaseConfig[Literal["arr"]]):
    engine: Literal["arr"] = "arr"
    default_language: str | None = "en"


class _UrlEngine(_ArrayEngine):
    """URL-accepting batch engine that echoes the forwarded URL as text."""

    properties: ClassVar[BaseProperties] = _UrlProps()

    def __init__(self, *, strict: bool = True, allow_private_urls: bool = False) -> None:
        self.config = _UrlConfig(strict=strict, allow_private_urls=allow_private_urls)

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        assert prepared.kind is InputKind.FETCHABLE_URL
        assert prepared.url is not None
        return TranscriptionResult(text=prepared.url)


def test_private_audio_url_rejected_by_default() -> None:
    # The default config keeps the R5 SSRF rejection -- a private /
    # loopback URL target is refused through the public transcribe path.
    with pytest.raises(UnsafeAudioUrlError) as exc_info:
        _UrlEngine().transcribe(AudioUrl("https://127.0.0.1/a.wav"))
    # The error hint points at the REACHABLE config switch (not the internal
    # validate_fetchable_url parameter), so the guidance is actionable.
    assert "allow_private_urls=True" in (exc_info.value.hint or "")


def test_private_audio_url_allowed_by_config_opt_in() -> None:
    # The spec R5 opt-in is now reachable through the public engine
    # pipeline -- BaseConfig.allow_private_urls=True threads through
    # EngineBase._prepare_audio so a trusted internal HTTPS endpoint is forwarded.
    result = _UrlEngine(allow_private_urls=True).transcribe(AudioUrl("https://127.0.0.1/a.wav"))
    assert result.text == "https://127.0.0.1/a.wav"


def test_private_audio_url_opt_in_still_requires_https() -> None:
    # Allow_private_urls relaxes ONLY the private-address rejection;
    # the HTTPS requirement is not relaxable.
    with pytest.raises(UnsafeAudioUrlError):
        _UrlEngine(allow_private_urls=True).transcribe(AudioUrl("http://127.0.0.1/a.wav"))


def test_allow_private_urls_excluded_from_env_fallback() -> None:
    # The safety switch MUST NOT be sourced from the environment, so
    # the environment can never silently relax the SSRF policy (mirrors strict).
    # Asserting the behavioral outcome (env_overrides drops it) also proves the
    # _ENV_EXCLUDED_FIELDS membership without reaching into the protected member.
    env = {"STANDARD_ASR_ARR__ALLOW_PRIVATE_URLS": "true"}
    assert "allow_private_urls" not in _UrlConfig.env_overrides("arr", environ=env)
    # The other standard fields are still env-sourced, proving the exclusion is
    # specific to allow_private_urls and not an empty-env artifact.
    assert _UrlConfig.env_overrides(
        "arr", environ={"STANDARD_ASR_ARR__DEFAULT_LANGUAGE": "fr"}
    ) == {"default_language": "fr"}


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


def test_batch_engine_failure_wraps_as_transcription_error() -> None:
    # The batch error contract (spec Runtime R7). An engine's native
    # execution failure inside _transcribe MUST surface as a portable
    # TranscriptionError importable from the package top level, preserving the
    # original exception as __cause__, so applications can catch one type across
    # every engine. This mirrors the streaming engine_error event (spec ST 6.2).
    import standard_asr

    assert standard_asr.TranscriptionError is TranscriptionError

    class _FailingEngine(_ArrayEngine):
        def _transcribe(
            self, prepared: PreparedAudio, params: RuntimeParams
        ) -> TranscriptionResult:
            try:
                raise RuntimeError("native SDK blew up")
            except RuntimeError as exc:
                raise TranscriptionError("engine inference failed") from exc

    with pytest.raises(TranscriptionError) as exc_info:
        _FailingEngine().transcribe(_audio())
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    # It is a StandardASRError so a broad standard handler also catches it.
    assert isinstance(exc_info.value, standard_asr.StandardASRError)


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
        accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]

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


# --- provider_params validated BEFORE audio decode (fail-fast) ---------------


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


class _WhitespaceDefaultConfig(LanguageConfigMixin, BaseConfig[Literal["arr"]]):
    engine: Literal["arr"] = "arr"
    # Whitespace-only: set (not None) but malformed -- canonicalization rejects it.
    default_language: str | None = "   "


class _WhitespaceDefaultEngine(_ArrayEngine):
    def __init__(self) -> None:
        self.config = _WhitespaceDefaultConfig()


def test_malformed_default_language_raises_config_error() -> None:
    # An empty/whitespace default_language must surface as the
    # documented ConfigError naming the engine and the malformed value, not leak
    # the normalizer's bare ValueError.
    from standard_asr.exceptions import ConfigError

    with pytest.raises(ConfigError, match="malformed") as exc_info:
        _WhitespaceDefaultEngine().transcribe(_audio())
    assert "'   '" in str(exc_info.value)
    assert "'arr'" in str(exc_info.value)


class _WhitespaceSelectableEngine(_ArrayEngine):
    # A malformed (whitespace-only) DECLARED tag can no longer be built through
    # validation (BaseProperties validates class-level defaults too), so
    # model_construct simulates the remaining ways such an object can reach the
    # runtime: a validation-bypassing construction or a third-party
    # StandardASR implementation. The membership canonicalization must still
    # catch it as defense in depth.
    properties: ClassVar[BaseProperties] = _ArrayProps.model_construct(
        selectable_languages=["en", "   "]
    )


def test_malformed_declared_selectable_tag_raises_config_error() -> None:
    # A malformed declared selectable tag is an engine-author
    # bug; the membership canonicalization must report it as ConfigError naming
    # the offending declaration, not leak a bare ValueError.
    from standard_asr.exceptions import ConfigError

    with pytest.raises(ConfigError, match="selectable_languages") as exc_info:
        _WhitespaceSelectableEngine().transcribe(_audio())
    assert "malformed" in str(exc_info.value)
    assert "'arr'" in str(exc_info.value)


class _WhitespaceDetectableEngine(_ArrayEngine):
    # Same defense-in-depth contract as _WhitespaceSelectableEngine, for the
    # detectable_languages axis (see the comment there).
    properties: ClassVar[BaseProperties] = _ArrayProps.model_construct(
        detectable_languages=["en", "   "]
    )


def test_malformed_declared_detectable_tag_raises_config_error() -> None:
    # The declared-side detectable canonicalization carries the same
    # ConfigError contract as its default_language / selectable_languages
    # siblings. Previously a best_effort 'auto' request hit the per-request
    # canonicalization inside effective_candidate_languages and surfaced the
    # normalizer's bare ValueError (an uncontracted HTTP 500 via the server)
    # with the engine-misdeclaration cause invisible.
    from standard_asr.exceptions import ConfigError

    with pytest.raises(ConfigError, match="detectable_languages") as exc_info:
        _WhitespaceDetectableEngine(strict=False).transcribe(
            _audio(), RuntimeParams(language="auto")
        )
    assert "malformed" in str(exc_info.value)
    assert "'arr'" in str(exc_info.value)


class _NonCanonicalLangProps(_ArrayProps):
    selectable_languages: list[str] = ["en-US", "auto"]
    detectable_languages: list[str] = ["en-US"]


class _NonCanonicalLangConfig(LanguageConfigMixin, BaseConfig[Literal["arr"]]):
    engine: Literal["arr"] = "arr"
    default_language: str | None = "en-us"  # non-canonical casing on purpose


class _NonCanonicalLangEngine(_ArrayEngine):
    properties: ClassVar[BaseProperties] = _NonCanonicalLangProps()

    def __init__(self) -> None:
        self.config = _NonCanonicalLangConfig()


def test_non_canonical_default_language_is_canonicalized_not_rejected() -> None:
    # Regression: BCP-47 language matching is case-insensitive. A non-canonical
    # default_language ("en-us") declared against a canonical selectable set
    # (["en-US"]) -- both as class-level defaults, which pydantic does NOT run the
    # field validators on -- must be matched case-insensitively instead of
    # spuriously failing the LANG R1 totality check and blocking the engine.
    engine = _NonCanonicalLangEngine()
    # No ConfigError is raised, and the engine receives the CANONICAL effective
    # language (echoed as detected_language), not the raw "en-us".
    result = engine.transcribe(_audio())
    assert result.detected_language == "en-US"


def test_region_tagged_request_matches_selectable_primary_subtag() -> None:
    # spec LANG R4: a region/script refinement ("en-US") of a
    # selectable primary subtag ("en", in _ArrayProps.selectable_languages) is
    # accepted via RFC 4647 lookup, and the full tag is handed to the engine to
    # reduce -- so engines that declare only primary subtags need not enumerate
    # every region variant. An informational diagnostic announces the reduction.
    result = _ArrayEngine().transcribe(_audio(), RuntimeParams(language="en-US"))
    assert result.detected_language == "en-US"
    refinement = [d for d in result.diagnostics if d.code == "language_refinement_accepted"]
    assert len(refinement) == 1
    assert refinement[0].level == "info"
    assert refinement[0].provided == "en-US"
    assert refinement[0].effective == "en-US"


def test_exact_selectable_language_emits_no_refinement_diagnostic() -> None:
    # An exact selectable match ("en") is NOT a refinement, so the
    # informational diagnostic MUST NOT fire (it would be noise on the common path).
    result = _ArrayEngine().transcribe(_audio(), RuntimeParams(language="en"))
    assert result.detected_language == "en"
    assert not any(d.code == "language_refinement_accepted" for d in result.diagnostics)


def test_rfc4647_lookup_skips_singleton_subtag() -> None:
    # spec LANG R4: RFC 4647 §3.4 drops a singleton (single-char)
    # subtag together with the subtag before it, so "zh-x-foo" truncates straight
    # to "zh" (never the meaningless "zh-x"). An engine declaring "zh" accepts it.
    class _ZhProps(_ArrayProps):
        selectable_languages: list[str] = ["zh", "auto"]
        detectable_languages: list[str] = ["zh"]

    class _ZhConfig(LanguageConfigMixin, BaseConfig[Literal["arr"]]):
        engine: Literal["arr"] = "arr"
        default_language: str | None = "zh"

    class _ZhEngine(_ArrayEngine):
        properties: ClassVar[BaseProperties] = _ZhProps()

        def __init__(self, *, strict: bool = True) -> None:
            self.config = _ZhConfig(strict=strict)

    result = _ZhEngine().transcribe(_audio(), RuntimeParams(language="zh-x-foo"))
    assert result.detected_language == "zh-x-foo"  # full tag handed to the engine
    assert any(d.code == "language_refinement_accepted" for d in result.diagnostics)


def test_selectable_match_singleton_helper_reduces_to_primary() -> None:
    # Direct check of the RFC 4647 singleton rule in _selectable_match
    # -- "zh-x-foo" must match selectable {"zh"} (reducing past the "zh-x"
    # singleton step), while an unrelated primary ("fr") still does not match.
    from standard_asr.asr_interface import (
        _selectable_match,  # pyright: ignore[reportPrivateUsage]
    )

    assert _selectable_match("zh-x-foo", frozenset({"zh"})) == "zh"
    assert _selectable_match("en-us", frozenset({"en"})) == "en"
    assert _selectable_match("en", frozenset({"en"})) == "en"
    assert _selectable_match("fr", frozenset({"en"})) is None
    # discriminators: the singleton "x" is dropped together with its
    # preceding "zh" subtag, so "zh-x" is NEVER produced as a lookup key. A
    # singleton-UNAWARE regression would instead truncate to "zh-x" and (i) match a
    # selectable {"zh-x"} it must NOT, and (ii) return "zh-x" instead of "zh" when
    # both are selectable. The four asserts above pass on BOTH the singleton-aware
    # and naive loops; these two are the only ones that fail on the naive one.
    assert _selectable_match("zh-x-foo", frozenset({"zh-x"})) is None
    assert _selectable_match("zh-x-foo", frozenset({"zh-x", "zh"})) == "zh"


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


# --- candidate-language validation runs in the standard layer ----------------


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


class _NonCanonicalDetectableProps(_ArrayProps):
    selectable_languages: list[str] = ["auto"]
    # Non-canonical class-level defaults: pydantic does NOT run field validators
    # on defaults, so these reach the standard layer raw.
    detectable_languages: list[str] = ["zh-hans", "en", "pt-br"]


class _NonCanonicalDetectableEngine(_CapturingArrayEngine):
    properties: ClassVar[BaseProperties] = _NonCanonicalDetectableProps()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _CAND_CAPS

    def __init__(self, *, strict: bool = True) -> None:
        self.config = _AutoConfig(strict=strict)


def test_canonical_candidates_match_non_canonical_detectable_declaration() -> None:
    # Engine path: a strict request with canonical candidate
    # tags must match the engine's raw, non-canonical detectable declaration
    # instead of raising "not detectable" for languages the engine CAN detect.
    _NonCanonicalDetectableEngine.received = None

    result = _NonCanonicalDetectableEngine(strict=True).transcribe(
        _audio(), RuntimeParams(language="auto", candidate_languages=["zh-Hans", "pt-BR"])
    )

    assert _NonCanonicalDetectableEngine.received is not None
    assert _NonCanonicalDetectableEngine.received.candidate_languages == ["zh-Hans", "pt-BR"]
    assert not any(d.code == "candidate_language_dropped" for d in result.diagnostics)


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
    # §Language R3 step 3 -- when a candidate_languages list WAS provided but the
    # axis is unsupported, resolution yields None + exactly one diagnostic and
    # never raises, even in strict. The diagnostic carries the ignored list.
    result = _NoCandEngine(strict=True).transcribe(
        _audio(), RuntimeParams(language="auto", candidate_languages=["en", "ja"])
    )
    # Exactly ONE diagnostic for this axis (no gate_params duplicate).
    cand_diags = [d for d in result.diagnostics if d.param == "candidate_languages"]
    assert len(cand_diags) == 1
    assert cand_diags[0].code == DIAG_CANDIDATE_LANGUAGES_IGNORED
    assert cand_diags[0].provided == ["en", "ja"]
    assert cand_diags[0].effective is None


def test_unsupported_candidate_languages_no_list_no_false_diagnostic() -> None:
    # (end-to-end): an `auto` engine that does NOT support candidate
    # languages must NOT inject a false `candidate_languages_ignored` warning on
    # an ordinary request that supplies no candidate list (the common
    # Whisper-family shape: default_language="auto", no candidate support). The
    # diagnostic flows through _resolve_language_axis into result.diagnostics, so
    # the regression is asserted on the full transcribe pipeline, not just the
    # helper. Holds in both strict and best_effort.
    for strict in (True, False):
        result = _NoCandEngine(strict=strict).transcribe(_audio(), RuntimeParams(language="auto"))
        cand_diags = [d for d in result.diagnostics if d.param == "candidate_languages"]
        assert cand_diags == []


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


def test_runtime_language_dropped_override_reports_fallback_final_value() -> None:
    # When the engine lacks language.runtime_override, the gate drops
    # the requested language with effective=None -- but the engine actually
    # transcribes in default_language. The gate cannot see default_language, so a
    # follow-up language_fell_back diagnostic surfaces the TRUE final value the
    # spec's best_effort contract requires (which parameter, why, final value).
    _NoOverrideEngine.received = None

    result = _NoOverrideEngine(strict=False).transcribe(_audio(), RuntimeParams(language="fr"))

    # Engine ran in default_language despite the request for "fr".
    assert _NoOverrideEngine.received is not None
    assert _NoOverrideEngine.received.language == "en"
    fell_back = next(d for d in result.diagnostics if d.code == "language_fell_back")
    assert fell_back.param == "language"
    assert fell_back.provided == "fr"  # what the caller asked for
    assert fell_back.effective == "en"  # what actually took effect
    # The gate's own drop diagnostic (effective=None) is still present; together
    # they give "which param / why / final value".
    assert any(d.code == "unsupported_parameter_ignored" for d in result.diagnostics)


def test_no_language_request_emits_no_fallback_diagnostic() -> None:
    # The fallback diagnostic must fire ONLY when the caller actually requested a
    # language that got dropped. A request with no language (None) on the same
    # no-override engine must NOT emit language_fell_back (nothing was dropped).
    _NoOverrideEngine.received = None

    result = _NoOverrideEngine(strict=False).transcribe(_audio(), RuntimeParams())

    assert not any(d.code == "language_fell_back" for d in result.diagnostics)


def test_supported_override_does_not_emit_fallback_diagnostic() -> None:
    # When language.runtime_override IS supported, the request is honored (or
    # handled by the selectability path), so no language_fell_back is emitted --
    # the diagnostic is specific to the dropped-override case.
    result = _ArrayEngine(strict=False).transcribe(_audio(), RuntimeParams(language="en"))
    assert not any(d.code == "language_fell_back" for d in result.diagnostics)


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


def test_audio_format_encoding_is_normalized() -> None:
    # AudioFormat.encoding is a case-insensitive identifier: it is stripped and
    # lowercased on construction, mirroring BaseProperties.wire_encodings so the
    # two normalize to the same form. A blank encoding is rejected.
    from pydantic import ValidationError

    from standard_asr.audio_format import AudioFormat

    assert AudioFormat(encoding="PCM_S16LE", sample_rate=16000).encoding == "pcm_s16le"
    assert AudioFormat(encoding="  Mulaw  ", sample_rate=16000).encoding == "mulaw"
    with pytest.raises(ValidationError, match="must not be blank"):
        AudioFormat(encoding="   ", sample_rate=16000)


def test_ensure_stream_format_supported_matches_encoding_case_insensitively() -> None:
    # An engine declaring wire_encodings=['pcm_s16le'] (already lowercased) MUST
    # accept a session opened with a differently-cased AudioFormat encoding: the
    # request encoding is normalized to the same form, so it is not a mismatch.
    from standard_asr.audio_format import AudioFormat

    _WireEngine().ensure_stream_format_supported(
        AudioFormat(encoding="PCM_S16LE", sample_rate=16000)
    )


def test_ensure_stream_format_supported_rejects_multichannel_wire() -> None:
    # v1 streaming wire is mono-only: the standard layer does not process
    # incremental wire frames, so it cannot downmix multi-channel frames the way
    # the batch path does. A stereo wire format is rejected at session start.
    from standard_asr.audio_format import AudioFormat

    engine = _WireEngine()
    engine.ensure_stream_format_supported(
        AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)
    )
    with pytest.raises(UnsupportedFeatureError, match="mono-only") as exc_info:
        engine.ensure_stream_format_supported(
            AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=2)
        )
    assert exc_info.value.param == "audio_format.channels"
    assert exc_info.value.mode == "streaming"
    assert exc_info.value.hint is not None


def test_ensure_stream_format_supported_skips_encoding_when_no_wire_encodings() -> None:
    # An engine that declares no wire_encodings cannot validate encoding; the
    # encoding check is a no-op. The sample-rate fail-closed still applies.
    from standard_asr.audio_format import AudioFormat

    _ArrayEngine().ensure_stream_format_supported(
        AudioFormat(encoding="anything", sample_rate=16000)
    )


# --------------------------------------------------------------------------- #
# AW-2 -- recommended_wire_format (single source of truth)
# --------------------------------------------------------------------------- #
def test_recommended_wire_format_uses_first_encoding_and_native_rate() -> None:
    fmt = _WireEngine().recommended_wire_format()
    assert fmt == AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)


def test_recommended_wire_format_prefers_required_input_sample_rate() -> None:
    # A hard-required input rate wins over the native rate, and the recommended
    # format is self-consistent: the engine's own guard must accept it.
    class _RequiredRateProps(_WireProps):
        native_sample_rate: int = 16000
        required_input_sample_rate: int | None = 8000
        accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = "any"

    class _RequiredRateEngine(_ArrayEngine):
        properties: ClassVar[BaseProperties] = _RequiredRateProps()

    engine = _RequiredRateEngine()
    fmt = engine.recommended_wire_format()
    assert fmt is not None
    assert fmt.sample_rate == 8000  # required, not native 16000
    engine.ensure_stream_format_supported(fmt)  # self-consistent: must not raise


def test_recommended_wire_format_falls_back_to_canonical_encoding() -> None:
    # Unconstrained wire_encodings: recommend the canonical pcm_s16le rather than
    # nothing -- the engine accepts any encoding, so a session can still open.
    class _NoWireProps(_ArrayProps):
        wire_encodings: list[str] | None = None

    class _NoWireEngine(_ArrayEngine):
        properties: ClassVar[BaseProperties] = _NoWireProps()

    fmt = _NoWireEngine().recommended_wire_format()
    assert fmt is not None
    assert fmt.encoding == "pcm_s16le"


def test_recommended_wire_format_none_when_no_usable_sample_rate() -> None:
    class _NoRateEngine(_ArrayEngine):
        properties: ClassVar[BaseProperties] = _WireProps.model_construct(
            wire_encodings=["pcm_s16le"], native_sample_rate=0, required_input_sample_rate=None
        )

    assert _NoRateEngine().recommended_wire_format() is None


def test_ensure_stream_format_supported_rejects_unreachable_sample_rate() -> None:
    # v1 does NOT resample streaming wire frames, so a wire sample_rate
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


def test_ensure_stream_format_supported_range_admits_in_range_rate() -> None:
    # A streaming engine declaring accepted_sample_rates as a range
    # accepts any wire sample_rate inside [min, max] at session start, and rejects
    # one outside it (v1 does not resample streaming wire frames).
    from standard_asr.audio_format import AudioFormat

    class _RangeProps(_ArrayProps):
        native_sample_rate: int = 16000
        accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = SampleRateRange(
            min=8000, max=48000
        )

    class _RangeEngine(_ArrayEngine):
        properties: ClassVar[BaseProperties] = _RangeProps()

    engine = _RangeEngine()
    # In-range wire rate is accepted (no raise).
    engine.ensure_stream_format_supported(AudioFormat(encoding="pcm_s16le", sample_rate=22050))
    engine.ensure_stream_format_supported(AudioFormat(encoding="pcm_s16le", sample_rate=8000))
    engine.ensure_stream_format_supported(AudioFormat(encoding="pcm_s16le", sample_rate=48000))
    # Out-of-range wire rate is fail-closed.
    with pytest.raises(UnsupportedFeatureError, match="wire sample_rate") as exc_info:
        engine.ensure_stream_format_supported(AudioFormat(encoding="pcm_s16le", sample_rate=96000))
    assert exc_info.value.param == "audio_format.sample_rate"


def test_ensure_stream_format_supported_accepts_required_rate_not_in_list() -> None:
    # The wire sample_rate is accepted when it equals required_input_sample_rate
    # (the rate the engine's wire protocol hard-requires).
    from standard_asr.audio_format import AudioFormat

    class _ReqProps(_ArrayProps):
        native_sample_rate: int = 24000
        accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [24000]
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
        accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = "any"

    class _AnyEngine(_ArrayEngine):
        properties: ClassVar[BaseProperties] = _AnyRateProps()

    _AnyEngine().ensure_stream_format_supported(
        AudioFormat(encoding="pcm_s16le", sample_rate=44100)
    )


def test_ensure_stream_format_supported_enforces_required_rate_under_any() -> None:
    # required_input_sample_rate + accepted_sample_rates="any"
    # is constructible (the declaration-time reachability validator only checks
    # concrete lists), so the session guard must still fail-closed on a wire
    # rate that differs from the hard-required one -- v1 does not resample
    # streaming wire frames.
    from standard_asr.audio_format import AudioFormat

    class _ReqAnyProps(_ArrayProps):
        accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = "any"
        required_input_sample_rate: int | None = 24000

    class _ReqAnyEngine(_ArrayEngine):
        properties: ClassVar[BaseProperties] = _ReqAnyProps()

    engine = _ReqAnyEngine()
    engine.ensure_stream_format_supported(AudioFormat(encoding="pcm_s16le", sample_rate=24000))
    with pytest.raises(UnsupportedFeatureError, match="required_input_sample_rate") as exc_info:
        engine.ensure_stream_format_supported(AudioFormat(encoding="pcm_s16le", sample_rate=44100))
    assert exc_info.value.param == "audio_format.sample_rate"
    assert exc_info.value.mode == "streaming"
    assert exc_info.value.hint is not None


def test_ensure_stream_format_supported_required_rate_beats_accepted_list() -> None:
    # FV §7.1: required_input_sample_rate binds even when the wire rate IS in
    # the concrete accepted_sample_rates list. accepted_sample_rates describes
    # the batch path (which resamples to the required rate before the engine);
    # v1 does not resample streaming wire frames, so a 16 kHz wire against a
    # 24 kHz-required engine would be interpreted as 24 kHz frames -- a silent
    # mistranscription. The deliberate semantics: hard-reject at establishment.
    from standard_asr.audio_format import AudioFormat

    class _ReqListProps(_ArrayProps):
        native_sample_rate: int = 24000
        accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000, 24000]
        required_input_sample_rate: int | None = 24000

    class _ReqListEngine(_ArrayEngine):
        properties: ClassVar[BaseProperties] = _ReqListProps()

    engine = _ReqListEngine()
    engine.ensure_stream_format_supported(AudioFormat(encoding="pcm_s16le", sample_rate=24000))
    with pytest.raises(UnsupportedFeatureError, match="required_input_sample_rate") as exc_info:
        engine.ensure_stream_format_supported(AudioFormat(encoding="pcm_s16le", sample_rate=16000))
    assert exc_info.value.param == "audio_format.sample_rate"
    assert exc_info.value.mode == "streaming"


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


# --- Streaming runtime-param gating via the template seam ---------------------


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


def test_start_transcription_is_keyword_only() -> None:
    # Start_transcription is keyword-only (the spec §3.1 signature now
    # carries the leading `*`), so the three same-typed optional params cannot be
    # confused positionally. Passing audio_format positionally MUST raise TypeError.
    fmt = AudioFormat(encoding="pcm_s16le", sample_rate=16000)
    with pytest.raises(TypeError):
        _StreamEngine().start_transcription(fmt)  # type: ignore[misc]


def test_start_transcription_audio_accepts_coercion() -> None:
    # The `audio` parameter is AudioInputLike and accepts the same
    # bare-value coercion as transcribe -- here a bare (ndarray, sample_rate)
    # tuple is coerced to an AudioArray and reaches the hook as prepared audio.
    _NoStreamingInputEngine.hook_called = False
    _NoStreamingInputEngine.received_prepared_audio = None

    _NoStreamingInputEngine().start_transcription(audio=(np.zeros(8, dtype=np.float32), 16000))

    assert _NoStreamingInputEngine.hook_called is True
    assert _NoStreamingInputEngine.received_prepared_audio is not None


def test_bare_call_requires_streaming_input_capability() -> None:
    # A bare start_transcription opens an incremental (self-managed
    # wire) session == the streaming_input axis. An engine that implements the
    # hook but declares only streaming_output MUST fail-closed on the bare call
    # (fail-closed, R1) instead of handing back an unfeedable session -- the hook
    # MUST NOT be reached.
    _NoStreamingInputEngine.hook_called = False

    with pytest.raises(UnsupportedFeatureError) as exc_info:
        _NoStreamingInputEngine().start_transcription()

    error = exc_info.value
    assert error.param == "audio_format"
    assert error.mode == "streaming"
    assert "streaming_input" in str(error)
    assert error.hint is not None
    assert _NoStreamingInputEngine.hook_called is False


def test_bare_call_allowed_for_streaming_input_engine() -> None:
    # The complement of the gate: an engine that declares streaming_input
    # accepts the bare incremental session and reaches the hook with no audio.
    _StreamEngine.hook_called = False
    _StreamEngine.received_prepared_audio = None

    _StreamEngine().start_transcription()

    assert _StreamEngine.hook_called is True
    assert _StreamEngine.received_prepared_audio is None


def test_batch_only_engine_bare_call_reports_does_not_support_streaming() -> None:
    # The bare-path streaming_input gate is placed AFTER the
    # hook-override defense, so a batch-only engine (no hook) still gets the
    # clearer "does not support streaming" message, not the capability-specific one.
    with pytest.raises(UnsupportedFeatureError, match="does not support streaming"):
        _ArrayEngine().start_transcription()


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


def test_start_transcription_applies_app_deadline_overrides() -> None:
    # The base template applies the application's deadline overrides AFTER the
    # hook constructed the session (spec ST.6.1): explicitly-set fields win
    # over the adapter's construction-time choices, unset fields keep them,
    # and omitting `deadlines` leaves the session untouched.
    class _DeadlineChoosingEngine(_StreamEngine):
        def _start_transcription(
            self,
            *,
            gated_params: RuntimeParams,
            audio_format: AudioFormat | None = None,
            prepared_audio: PreparedAudio | None = None,
        ) -> TranscriptionSession:
            return _StreamSession(done_timeout=7.0, max_idle=9.0)

    overridden = _DeadlineChoosingEngine().start_transcription(
        deadlines=StreamDeadlines(max_idle=0.5)
    )
    assert overridden.done_timeout == 7.0  # adapter choice kept (field unset)
    assert overridden.max_idle == 0.5  # application's explicit field wins

    untouched = _DeadlineChoosingEngine().start_transcription()
    assert untouched.done_timeout == 7.0
    assert untouched.max_idle == 9.0


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
            # Declares streaming_input so the bare incremental session below is
            # accepted by the capability gate.
            streaming_input=FlagCap(supported=True),
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


def test_engine_base_default_prepare_is_noop() -> None:
    # spec IC.11: EngineBase provides a default no-op prepare() so an engine with
    # nothing to warm up inherits it; it accepts no arguments and returns None.
    engine = _ArrayEngine()
    assert engine.prepare() is None
