# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RuntimeParams and the provider-params escape hatch."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from standard_asr.runtime_params import (
    ProviderParams,
    RuntimeParams,
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
