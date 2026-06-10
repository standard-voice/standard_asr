# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RuntimeParams and the provider-params escape hatch."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from standard_asr.runtime_params import (
    ProviderParams,
    RuntimeParams,
    WireRuntimeParams,
    WordTimestampGranularity,
)


class _OpenAIParams(ProviderParams):
    temperature: float = 0.0


def test_defaults_are_none() -> None:
    params = RuntimeParams()
    assert params.language is None
    assert params.candidate_languages is None
    assert params.word_timestamps is None
    assert params.prompt is None
    assert params.phrase_hints is None
    assert params.on_unsupported == "fail"
    assert params.provider_params is None


def test_closed_type_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        RuntimeParams(unknown_field=1)  # type: ignore[call-arg]


def test_word_timestamps_enum() -> None:
    params = RuntimeParams(word_timestamps=WordTimestampGranularity.WORD)
    assert params.word_timestamps is WordTimestampGranularity.WORD


def test_granularity_vocabulary_single_source_of_truth() -> None:
    # The request-side enum (WordTimestampGranularity) and the
    # declaration-side capability Literal (WordTimestampGranularityName) define
    # the same granularity vocabulary. They MUST stay identical -- an additive
    # change to one without the other would silently desync gating from
    # declaration. This drift test fails the moment the two sets diverge.
    from typing import get_args

    from standard_asr.capabilities import WordTimestampGranularityName

    enum_values = {g.value for g in WordTimestampGranularity}
    literal_values = set(get_args(WordTimestampGranularityName))
    assert enum_values == literal_values


def test_provider_params_typed() -> None:
    params = RuntimeParams(provider_params=_OpenAIParams(temperature=0.2))
    assert isinstance(params.provider_params, _OpenAIParams)


def test_provider_params_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        _OpenAIParams(nope=1)  # type: ignore[call-arg]


def test_provider_params_rejects_bare_base_instance() -> None:
    # The bare ProviderParams base carries no fields and is never a
    # valid concrete params model. Passing a bare instance is refused at
    # construction (so it never reaches the gate with a misleading
    # "swapped engine?" message about a wrong-engine model).
    with pytest.raises(ValidationError, match="concrete ProviderParams subclass"):
        RuntimeParams(provider_params=ProviderParams())


def test_provider_params_rejects_mapping_coerced_to_bare_base() -> None:
    # `provider_params={}` would otherwise be coerced by pydantic into
    # a bare ProviderParams() instance (the field is typed ProviderParams | None),
    # then trip the gate's swap-safety check with a misleading message. Reject the
    # coercion at construction with a message that names the real fix.
    with pytest.raises(ValidationError, match="concrete ProviderParams subclass"):
        RuntimeParams(provider_params={})  # type: ignore[arg-type]


def test_provider_params_concrete_subclass_still_accepted() -> None:
    # The fix must not reject the legitimate case: a concrete engine params
    # subclass instance passes construction unchanged.
    params = RuntimeParams(provider_params=_OpenAIParams(temperature=0.5))
    assert isinstance(params.provider_params, _OpenAIParams)
    assert params.provider_params.temperature == 0.5


def test_on_unsupported_choices() -> None:
    assert RuntimeParams(on_unsupported="degrade_to_prompt").on_unsupported == ("degrade_to_prompt")
    with pytest.raises(ValidationError):
        RuntimeParams(on_unsupported="bogus")  # type: ignore[arg-type]


# Candidate_languages gets the same construct-time fail-fast contract
# as the scalar `language` field, on BOTH the in-process and the wire models.
@pytest.mark.parametrize("model", [RuntimeParams, WireRuntimeParams])
@pytest.mark.parametrize("value", [None, [], ["en", "ja"], ["zh-Hans", "pt-BR"], ["zz"]])
def test_candidate_languages_accepts_wellformed_and_sentinels(
    model: type[RuntimeParams | WireRuntimeParams], value: list[str] | None
) -> None:
    # None and the [] "requested-but-empty" sentinel carry nothing to validate;
    # well-formed tags pass even if not detectable (membership/limit stay owned by
    # language.py, which needs engine capabilities unavailable at construction).
    assert model(candidate_languages=value).candidate_languages == value


@pytest.mark.parametrize("model", [RuntimeParams, WireRuntimeParams])
@pytest.mark.parametrize("value", [["english"], ["en", ""], ["en", "   "], ["en", "123-"]])
def test_candidate_languages_rejects_malformed_items(
    model: type[RuntimeParams | WireRuntimeParams], value: list[str]
) -> None:
    # A malformed candidate is an invalid value (a code bug), rejected at
    # construction regardless of strict/best_effort -- not left dormant until a
    # later auto-mode request on a supporting engine (Language R3 step 2 would
    # otherwise short-circuit it away before step 4's per-item check).
    with pytest.raises(ValidationError, match="well-formed BCP-47"):
        model(candidate_languages=value)


@pytest.mark.parametrize("model", [RuntimeParams, WireRuntimeParams])
@pytest.mark.parametrize("value", [["auto"], ["en", "AUTO"], ["Auto", "en"]])
def test_candidate_languages_rejects_auto(
    model: type[RuntimeParams | WireRuntimeParams], value: list[str]
) -> None:
    # 'auto' is a directive, never a candidate; its presence is a caller bug and
    # always raises (mirrors language.effective_candidate_languages, case-folded).
    with pytest.raises(ValidationError, match="MUST NOT contain 'auto'"):
        model(candidate_languages=value)


@pytest.mark.parametrize("model", [RuntimeParams, WireRuntimeParams])
def test_candidate_languages_error_never_echoes_raw_value(
    model: type[RuntimeParams | WireRuntimeParams],
) -> None:
    # Like the scalar `language` message, the candidate message must not embed the
    # submitted value (it is echoed verbatim by the server's unauthenticated 422
    # body / logs), so a mis-pasted secret sent as a candidate is not reflected.
    sentinel = "another secret value"
    with pytest.raises(ValidationError) as exc_info:
        model(candidate_languages=[sentinel])
    assert all(sentinel not in err["msg"] for err in exc_info.value.errors())


@pytest.mark.parametrize("model", [RuntimeParams, WireRuntimeParams])
@pytest.mark.parametrize("value", [None, [], ["Anthropic"], ["Anthropic", "Claude"]])
def test_phrase_hints_accepts_real_terms_and_sentinels(
    model: type[RuntimeParams | WireRuntimeParams], value: list[str] | None
) -> None:
    # None and the [] sentinel carry no term; real terms pass through.
    assert model(phrase_hints=value).phrase_hints == value


@pytest.mark.parametrize("model", [RuntimeParams, WireRuntimeParams])
@pytest.mark.parametrize("value", [[""], ["Anthropic", ""], ["   "], ["\t\n"], ["ok", "  "]])
def test_phrase_hints_rejects_blank_terms(
    model: type[RuntimeParams | WireRuntimeParams], value: list[str]
) -> None:
    # A blank / whitespace-only term carries no boost signal, would
    # fool the degrade survival check ("" in x is always true), and could be
    # handed to the engine as an empty string. Reject it at construction.
    with pytest.raises(ValidationError, match="empty or whitespace-only"):
        model(phrase_hints=value)


@pytest.mark.parametrize("model", [RuntimeParams, WireRuntimeParams])
def test_phrase_hints_blank_error_never_echoes_raw_value(
    model: type[RuntimeParams | WireRuntimeParams],
) -> None:
    # A phrase hint may carry sensitive text; the blank-term message names the
    # rule, never the offending entry (echoed verbatim by the server's 422 body).
    sentinel = "secret phrase hint text"
    with pytest.raises(ValidationError) as exc_info:
        model(phrase_hints=[sentinel, ""])
    assert all(sentinel not in err["msg"] for err in exc_info.value.errors())


def test_on_unsupported_is_part_of_wire_portable_set() -> None:
    # On_unsupported is a top-level RuntimeParams field carried over
    # the wire (it is in the portable set), so a cross-language client can express
    # the degrade opt-in. The D5 drift guard already binds the field sets; this
    # pins the specific field a wire client must be able to send.
    assert "on_unsupported" in WireRuntimeParams.model_fields
    wire = WireRuntimeParams(on_unsupported="degrade_to_prompt")
    assert wire.to_runtime_params().on_unsupported == "degrade_to_prompt"


def test_frozen() -> None:
    params = RuntimeParams(language="en")
    with pytest.raises(ValidationError):
        params.language = "fr"  # type: ignore[misc]


@pytest.mark.parametrize("tag", ["en", "en-US", "zh-Hans", "auto", None])
def test_language_accepts_wellformed_tags(tag: str | None) -> None:
    assert RuntimeParams(language=tag).language == tag


@pytest.mark.parametrize("tag", ["english", "e", "en-", "123"])
def test_language_rejects_malformed_tags(tag: str) -> None:
    # A malformed language tag is an invalid value, rejected at construction
    # regardless of strict/best_effort (like provider_params errors).
    with pytest.raises(ValidationError, match="well-formed BCP-47"):
        RuntimeParams(language=tag)


@pytest.mark.parametrize("model", [RuntimeParams, WireRuntimeParams])
def test_language_error_never_echoes_raw_value(
    model: type[RuntimeParams | WireRuntimeParams],
) -> None:
    # The malformed-tag message must not embed the submitted value: it is
    # surfaced verbatim by the server's unauthenticated 422 body (spec server.md
    # "validation errors never echo the request input"), CLI output, and logs,
    # so a mis-pasted secret sent as `language` would otherwise be reflected.
    sentinel = "my secret passphrase here"
    with pytest.raises(ValidationError) as exc_info:
        model(language=sentinel)
    # The server's sanitizer forwards `msg` verbatim for non-credential fields,
    # so `msg` itself must already be value-free.
    assert all(sentinel not in err["msg"] for err in exc_info.value.errors())


# --- WireRuntimeParams (portable-only wire view, D5) --------------------------


def test_wire_params_field_set_matches_portable_runtime_params() -> None:
    # D5 drift guard: the wire view is exactly RuntimeParams minus the
    # discover-only provider_params escape hatch. An additive change to the
    # portable set must update both, or this fails (mirrors the import-time
    # assertion in runtime_params).
    assert set(WireRuntimeParams.model_fields) == (
        set(RuntimeParams.model_fields) - {"provider_params"}
    )


def test_wire_params_forbids_provider_params() -> None:
    # provider_params cannot be sent over the wire; supplying it is rejected.
    with pytest.raises(ValidationError) as exc_info:
        WireRuntimeParams.model_validate({"provider_params": {"beam": 5}})
    assert any(err["loc"] == ("provider_params",) for err in exc_info.value.errors())


def test_wire_params_validates_language() -> None:
    # The wire view applies the same language validation as RuntimeParams.
    assert WireRuntimeParams(language="en").language == "en"
    with pytest.raises(ValidationError, match="well-formed BCP-47"):
        WireRuntimeParams(language="english")


def test_wire_params_to_runtime_params_round_trips_portable_fields() -> None:
    wire = WireRuntimeParams(
        language="en",
        candidate_languages=["en", "fr"],
        word_timestamps=WordTimestampGranularity.WORD,
        prompt="hi",
        phrase_hints=["foo"],
        on_unsupported="degrade_to_prompt",
    )
    params = wire.to_runtime_params()
    assert isinstance(params, RuntimeParams)
    assert params.language == "en"
    assert params.candidate_languages == ["en", "fr"]
    assert params.word_timestamps is WordTimestampGranularity.WORD
    assert params.prompt == "hi"
    assert params.phrase_hints == ["foo"]
    assert params.on_unsupported == "degrade_to_prompt"
    # provider_params is necessarily None (it cannot be sent).
    assert params.provider_params is None


def test_wire_params_is_frozen_and_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        WireRuntimeParams(unknown=1)  # type: ignore[call-arg]
    wire = WireRuntimeParams(language="en")
    with pytest.raises(ValidationError):
        wire.language = "fr"  # type: ignore[misc]
