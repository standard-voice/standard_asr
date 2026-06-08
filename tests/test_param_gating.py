# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for runtime parameter capability gating."""

from __future__ import annotations

import pytest

from standard_asr.capabilities import (
    BatchCapabilities,
    CandidateLanguagesCap,
    CandidateLanguagesConstraints,
    DeclaredCapabilities,
    FlagCap,
    GuidanceCaps,
    LanguageCaps,
    PhraseHintsCap,
    PromptCap,
    WordTimestampsCap,
)
from standard_asr.exceptions import InvalidProviderParamError, UnsupportedFeatureError
from standard_asr.param_gating import (
    _gate_granularity,  # pyright: ignore[reportPrivateUsage]
    _try_degrade_to_prompt,  # pyright: ignore[reportPrivateUsage]
    gate_params,
)
from standard_asr.results import Diagnostic
from standard_asr.runtime_params import (
    ProviderParams,
    RuntimeParams,
    WordTimestampGranularity,
)


def _caps(
    *, prompt: bool = False, phrase_hints: bool = False, override: bool = False
) -> DeclaredCapabilities:
    return DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=override)),
            guidance=GuidanceCaps(
                prompt=PromptCap(supported=prompt),
                phrase_hints=PhraseHintsCap(supported=phrase_hints),
            ),
        )
    )


class _P(ProviderParams):
    x: int = 0


def test_supported_param_passes() -> None:
    gated, diags = gate_params(
        RuntimeParams(language="en"), _caps(override=True), "batch", strict=True
    )
    assert gated.language == "en"
    assert diags == []


def test_unsupported_strict_raises() -> None:
    with pytest.raises(UnsupportedFeatureError):
        gate_params(RuntimeParams(language="en"), _caps(), "batch", strict=True)


def test_unsupported_best_effort_drops() -> None:
    gated, diags = gate_params(RuntimeParams(language="en"), _caps(), "batch", strict=False)
    assert gated.language is None
    assert diags[0].code == "unsupported_parameter_ignored"


def test_provider_params_unexpected_raises() -> None:
    with pytest.raises(InvalidProviderParamError):
        gate_params(RuntimeParams(provider_params=_P()), _caps(), "batch", strict=True)


def test_provider_params_wrong_type_raises() -> None:
    class _Q(ProviderParams):
        y: int = 0

    with pytest.raises(InvalidProviderParamError):
        gate_params(
            RuntimeParams(provider_params=_P()),
            _caps(),
            "batch",
            strict=True,
            expected_provider_type=_Q,
        )


def test_provider_params_correct_type_ok() -> None:
    gated, _ = gate_params(
        RuntimeParams(provider_params=_P(x=1)),
        _caps(),
        "batch",
        strict=True,
        expected_provider_type=_P,
    )
    assert isinstance(gated.provider_params, _P)


def test_degrade_phrase_hints_to_prompt() -> None:
    params = RuntimeParams(phrase_hints=["Anthropic", "Claude"], on_unsupported="degrade_to_prompt")
    gated, diags = gate_params(params, _caps(prompt=True), "batch", strict=True)
    assert gated.phrase_hints is None
    assert gated.prompt is not None
    assert "Anthropic" in gated.prompt
    assert any(d.code == "guidance_degraded_to_prompt" for d in diags)


def test_degrade_not_requested_strict_raises() -> None:
    with pytest.raises(UnsupportedFeatureError):
        gate_params(RuntimeParams(phrase_hints=["x"]), _caps(prompt=True), "batch", strict=True)


def test_degrade_but_prompt_unsupported_strict_raises() -> None:
    params = RuntimeParams(phrase_hints=["x"], on_unsupported="degrade_to_prompt")
    with pytest.raises(UnsupportedFeatureError):
        gate_params(params, _caps(prompt=False), "batch", strict=True)


# --------------------------------------------------------------------------- #
# H4: word_timestamps granularity validated against the offered granularities.
# --------------------------------------------------------------------------- #
def _wt_caps(*granularities: str) -> DeclaredCapabilities:
    return DeclaredCapabilities(
        batch=BatchCapabilities(
            word_timestamps=WordTimestampsCap(
                supported=True,
                granularities=list(granularities),  # type: ignore[arg-type]
            )
        )
    )


def test_granularity_offered_passes() -> None:
    params = RuntimeParams(word_timestamps=WordTimestampGranularity.WORD)
    gated, diags = gate_params(params, _wt_caps("word", "segment"), "batch", strict=True)
    assert gated.word_timestamps is WordTimestampGranularity.WORD
    assert diags == []


def test_granularity_not_offered_strict_raises() -> None:
    params = RuntimeParams(word_timestamps=WordTimestampGranularity.CHAR)
    with pytest.raises(UnsupportedFeatureError):
        gate_params(params, _wt_caps("word", "segment"), "batch", strict=True)


def test_granularity_not_offered_best_effort_drops() -> None:
    params = RuntimeParams(word_timestamps=WordTimestampGranularity.CHAR)
    gated, diags = gate_params(params, _wt_caps("word", "segment"), "batch", strict=False)
    assert gated.word_timestamps is None
    assert any(d.code == "unsupported_granularity_ignored" for d in diags)


def test_granularity_empty_list_defers_to_feature_flag() -> None:
    # Engine supports word_timestamps but did not enumerate granularities ->
    # back-compat: requested granularity is accepted.
    params = RuntimeParams(word_timestamps=WordTimestampGranularity.CHAR)
    gated, diags = gate_params(params, _wt_caps(), "batch", strict=True)
    assert gated.word_timestamps is WordTimestampGranularity.CHAR
    assert diags == []


def test_word_timestamps_feature_unsupported_strict_raises() -> None:
    params = RuntimeParams(word_timestamps=WordTimestampGranularity.WORD)
    caps = DeclaredCapabilities(batch=BatchCapabilities())  # supported=False default
    with pytest.raises(UnsupportedFeatureError):
        gate_params(params, caps, "batch", strict=True)


# --------------------------------------------------------------------------- #
# H5: empty list ([]) is the "requested-but-empty" sentinel -> not a request.
# --------------------------------------------------------------------------- #
def test_empty_phrase_hints_not_gated_when_unsupported() -> None:
    # phrase_hints unsupported, but [] carries nothing to honor -> no raise,
    # no diagnostic, value preserved as [].
    params = RuntimeParams(phrase_hints=[])
    gated, diags = gate_params(params, _caps(phrase_hints=False), "batch", strict=True)
    assert gated.phrase_hints == []
    assert diags == []


def test_empty_phrase_hints_no_garbage_degrade() -> None:
    # degrade_to_prompt on phrase_hints=[] must NOT produce "Relevant terms: .".
    params = RuntimeParams(phrase_hints=[], on_unsupported="degrade_to_prompt")
    gated, diags = gate_params(params, _caps(prompt=True), "batch", strict=True)
    assert gated.prompt is None
    assert gated.phrase_hints == []
    assert diags == []


def test_candidate_languages_not_gated_here_when_unsupported() -> None:
    # RUNT-1/RUNT-2: candidate_languages is owned solely by language.py (spec
    # §Language R3), so gate_params must NOT touch it -- even an unsupported,
    # non-empty request in strict mode passes through untouched (no raise, no
    # diagnostic). language.effective_candidate_languages resolves the axis to
    # None + a single diagnostic; that is asserted in test_language.py.
    caps = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(candidate_languages=CandidateLanguagesCap(supported=False))
        )
    )
    for value in ([], ["en", "ja"]):
        params = RuntimeParams(candidate_languages=value)
        gated, diags = gate_params(params, caps, "batch", strict=True)
        assert gated.candidate_languages == value
        assert diags == []


# --------------------------------------------------------------------------- #
# Direct exercise of the defensive guards in the granularity / degrade helpers.
# These guards protect the contract regardless of call site; gate_params skips
# them (it filters None / [] earlier), so they are unit-tested directly.
# --------------------------------------------------------------------------- #
def test_gate_granularity_no_request_returns_false() -> None:
    # No word_timestamps requested -> the helper is a no-op and reports False.
    updates: dict[str, object] = {}
    diags: list[Diagnostic] = []
    changed = _gate_granularity(
        RuntimeParams(), _wt_caps("word"), "batch", updates, diags, strict=True
    )
    assert changed is False
    assert updates == {}
    assert diags == []


def test_gate_granularity_non_word_timestamps_node_returns_false() -> None:
    # word_timestamps does not resolve to a WordTimestampsCap leaf (here: the
    # whole mode domain is absent, so node_at returns None) -> the helper does not
    # over-constrain an unknown/missing shape and reports no change.
    caps = DeclaredCapabilities()  # no batch domain at all
    updates: dict[str, object] = {}
    diags: list[Diagnostic] = []
    changed = _gate_granularity(
        RuntimeParams(word_timestamps=WordTimestampGranularity.WORD),
        caps,
        "batch",
        updates,
        diags,
        strict=True,
    )
    assert changed is False
    assert updates == {}
    assert diags == []


def test_try_degrade_empty_hints_returns_false() -> None:
    # degrade_to_prompt with no hints must never frame an empty "Relevant terms: ."
    params = RuntimeParams(phrase_hints=[], on_unsupported="degrade_to_prompt")
    updates: dict[str, object] = {}
    diags: list[Diagnostic] = []
    applied = _try_degrade_to_prompt(params, _caps(prompt=True), "batch", updates, diags)
    assert applied is False
    assert updates == {}
    assert diags == []


def test_candidate_languages_supported_passes_through_untouched() -> None:
    # Even when supported, gate_params leaves candidate_languages alone (it is
    # not in _GATED_PARAMS); resolution/validation happens in language.py.
    params = RuntimeParams(candidate_languages=["en", "ja"])
    caps = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(
                candidate_languages=CandidateLanguagesCap(
                    supported=True,
                    constraints=CandidateLanguagesConstraints(max=3),
                )
            )
        )
    )
    gated, diags = gate_params(params, caps, "batch", strict=True)
    assert gated.candidate_languages == ["en", "ja"]
    assert diags == []
