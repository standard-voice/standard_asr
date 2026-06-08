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
    PhraseHintsConstraints,
    PromptCap,
    PromptConstraints,
    StreamingCapabilities,
    WordTimestampsCap,
)
from standard_asr.exceptions import InvalidProviderParamError, UnsupportedFeatureError
from standard_asr.param_gating import (
    _enforce_phrase_hints_limits,  # pyright: ignore[reportPrivateUsage]
    _enforce_prompt_limit,  # pyright: ignore[reportPrivateUsage]
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


def test_supported_word_timestamps_cannot_have_empty_granularities() -> None:
    # RUNT-6: a supported WordTimestampsCap MUST enumerate granularities, so the
    # ambiguous "empty => honor anything" state is unrepresentable. Constructing
    # _wt_caps() (supported=True, no granularities) raises at validation time.
    with pytest.raises(ValueError, match="non-empty"):
        _wt_caps()


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
    applied = _try_degrade_to_prompt(
        params, _caps(prompt=True), "batch", updates, diags, strict=False
    )
    assert applied is False
    assert updates == {}
    assert diags == []


# --------------------------------------------------------------------------- #
# RUNT-4: declared guidance limits are enforced (truncate/raise), and degrade
# respects max_tokens instead of silently exceeding it.
# --------------------------------------------------------------------------- #
def _guidance_caps(
    *,
    prompt_max_tokens: int | None = None,
    hints_max_terms: int | None = None,
    hints_max_chars: int | None = None,
    hints_max_words: int | None = None,
) -> DeclaredCapabilities:
    return DeclaredCapabilities(
        batch=BatchCapabilities(
            guidance=GuidanceCaps(
                prompt=PromptCap(
                    supported=True,
                    constraints=PromptConstraints(max_tokens=prompt_max_tokens),
                ),
                phrase_hints=PhraseHintsCap(
                    supported=True,
                    constraints=PhraseHintsConstraints(
                        max_terms=hints_max_terms,
                        max_chars_per_term=hints_max_chars,
                        max_words_per_term=hints_max_words,
                    ),
                ),
            )
        )
    )


def test_prompt_within_max_tokens_passes() -> None:
    params = RuntimeParams(prompt="one two three")
    gated, diags = gate_params(params, _guidance_caps(prompt_max_tokens=5), "batch", strict=True)
    assert gated.prompt == "one two three"
    assert diags == []


def test_prompt_over_max_tokens_strict_raises() -> None:
    params = RuntimeParams(prompt="one two three four")
    with pytest.raises(UnsupportedFeatureError, match="tokens"):
        gate_params(params, _guidance_caps(prompt_max_tokens=2), "batch", strict=True)


def test_prompt_over_max_tokens_best_effort_truncates() -> None:
    params = RuntimeParams(prompt="one two three four")
    gated, diags = gate_params(params, _guidance_caps(prompt_max_tokens=2), "batch", strict=False)
    assert gated.prompt == "one two"
    diag = next(d for d in diags if d.code == "prompt_truncated")
    assert diag.effective == "one two"
    assert diag.provided == 4


def test_prompt_unbounded_limit_noop() -> None:
    params = RuntimeParams(prompt="a b c d e f")
    gated, diags = gate_params(params, _guidance_caps(prompt_max_tokens=None), "batch", strict=True)
    assert gated.prompt == "a b c d e f"
    assert diags == []


def test_phrase_hints_too_many_terms_strict_raises() -> None:
    params = RuntimeParams(phrase_hints=["a", "b", "c"])
    with pytest.raises(UnsupportedFeatureError, match="limits"):
        gate_params(params, _guidance_caps(hints_max_terms=2), "batch", strict=True)


def test_phrase_hints_too_many_terms_best_effort_truncates() -> None:
    params = RuntimeParams(phrase_hints=["a", "b", "c"])
    gated, diags = gate_params(params, _guidance_caps(hints_max_terms=2), "batch", strict=False)
    assert gated.phrase_hints == ["a", "b"]
    diag = next(d for d in diags if d.code == "phrase_hints_truncated")
    assert diag.effective == ["a", "b"]


def test_phrase_hints_over_long_term_best_effort_shortens() -> None:
    params = RuntimeParams(phrase_hints=["alpha beta gamma", "short"])
    gated, diags = gate_params(
        params,
        _guidance_caps(hints_max_chars=5, hints_max_words=2),
        "batch",
        strict=False,
    )
    # "alpha beta gamma" -> first 2 words "alpha beta" -> first 5 chars "alpha".
    assert gated.phrase_hints == ["alpha", "short"]
    assert any(d.code == "phrase_hints_truncated" for d in diags)


def test_phrase_hints_within_limits_passes() -> None:
    params = RuntimeParams(phrase_hints=["a", "b"])
    gated, diags = gate_params(
        params, _guidance_caps(hints_max_terms=3, hints_max_chars=10), "batch", strict=True
    )
    assert gated.phrase_hints == ["a", "b"]
    assert diags == []


def test_degrade_respects_max_tokens_best_effort_truncates() -> None:
    # phrase_hints unsupported -> degrade; but the framed prompt exceeds the
    # prompt channel's max_tokens, so it is truncated (not silently emitted).
    caps = DeclaredCapabilities(
        batch=BatchCapabilities(
            guidance=GuidanceCaps(
                prompt=PromptCap(supported=True, constraints=PromptConstraints(max_tokens=3)),
                phrase_hints=PhraseHintsCap(supported=False),
            )
        )
    )
    params = RuntimeParams(
        phrase_hints=["Anthropic", "Claude", "Opus"], on_unsupported="degrade_to_prompt"
    )
    gated, diags = gate_params(params, caps, "batch", strict=False)
    assert gated.phrase_hints is None
    assert gated.prompt is not None
    assert len(gated.prompt.split()) == 3
    assert any(d.code == "guidance_degraded_to_prompt" for d in diags)
    assert any(d.code == "prompt_truncated" for d in diags)


def test_degrade_over_max_tokens_strict_raises_not_silent() -> None:
    caps = DeclaredCapabilities(
        batch=BatchCapabilities(
            guidance=GuidanceCaps(
                prompt=PromptCap(supported=True, constraints=PromptConstraints(max_tokens=3)),
                phrase_hints=PhraseHintsCap(supported=False),
            )
        )
    )
    params = RuntimeParams(
        phrase_hints=["Anthropic", "Claude", "Opus"], on_unsupported="degrade_to_prompt"
    )
    with pytest.raises(UnsupportedFeatureError):
        gate_params(params, caps, "batch", strict=True)


def test_guidance_limits_enforced_in_streaming_mode() -> None:
    # Mode-correctness: the same guidance-limit enforcement applies under
    # mode="streaming" (RUNT-3 follow-up wires gate_params(mode="streaming")).
    caps = DeclaredCapabilities(
        streaming=StreamingCapabilities(
            guidance=GuidanceCaps(
                prompt=PromptCap(supported=True, constraints=PromptConstraints(max_tokens=2)),
            )
        )
    )
    params = RuntimeParams(prompt="one two three")
    gated, diags = gate_params(params, caps, "streaming", strict=False)
    assert gated.prompt == "one two"
    assert any(d.code == "prompt_truncated" for d in diags)


def test_enforce_prompt_limit_guards() -> None:
    # Defensive guards (exercised directly; gate_params filters these earlier):
    # no prompt -> no-op; a non-PromptCap node (here: missing domain) -> no-op.
    updates: dict[str, object] = {}
    diags: list[Diagnostic] = []
    _enforce_prompt_limit(
        RuntimeParams(), _guidance_caps(prompt_max_tokens=1), "batch", updates, diags, strict=True
    )
    assert updates == {} and diags == []
    _enforce_prompt_limit(
        RuntimeParams(prompt="a b c"),
        DeclaredCapabilities(),  # no batch domain -> node_at returns None
        "batch",
        updates,
        diags,
        strict=True,
    )
    assert updates == {} and diags == []


def test_enforce_phrase_hints_limits_guards() -> None:
    updates: dict[str, object] = {}
    diags: list[Diagnostic] = []
    _enforce_phrase_hints_limits(
        RuntimeParams(), _guidance_caps(hints_max_terms=1), "batch", updates, diags, strict=True
    )
    assert updates == {} and diags == []
    _enforce_phrase_hints_limits(
        RuntimeParams(phrase_hints=["a", "b"]),
        DeclaredCapabilities(),  # no batch domain -> node_at returns None
        "batch",
        updates,
        diags,
        strict=True,
    )
    assert updates == {} and diags == []


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
