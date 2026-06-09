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
    # X-EL-3: the request-side enum (WordTimestampGranularity) and the
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


def test_on_unsupported_choices() -> None:
    assert RuntimeParams(on_unsupported="degrade_to_prompt").on_unsupported == ("degrade_to_prompt")
    with pytest.raises(ValidationError):
        RuntimeParams(on_unsupported="bogus")  # type: ignore[arg-type]


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
