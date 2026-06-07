# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for runtime parameter capability gating."""

from __future__ import annotations

import pytest

from standard_asr.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    GuidanceCaps,
    LanguageCaps,
    PhraseHintsCap,
    PromptCap,
    FlagCap,
)
from standard_asr.exceptions import InvalidProviderParamError, UnsupportedFeatureError
from standard_asr.param_gating import gate_params
from standard_asr.runtime_params import ProviderParams, RuntimeParams


def _caps(*, prompt: bool = False, phrase_hints: bool = False, override: bool = False) -> DeclaredCapabilities:
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
    gated, diags = gate_params(
        RuntimeParams(language="en"), _caps(), "batch", strict=False
    )
    assert gated.language is None
    assert diags[0].code == "unsupported_parameter_ignored"


def test_provider_params_unexpected_raises() -> None:
    with pytest.raises(InvalidProviderParamError):
        gate_params(
            RuntimeParams(provider_params=_P()), _caps(), "batch", strict=True
        )


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
    params = RuntimeParams(
        phrase_hints=["Anthropic", "Claude"], on_unsupported="degrade_to_prompt"
    )
    gated, diags = gate_params(params, _caps(prompt=True), "batch", strict=True)
    assert gated.phrase_hints is None
    assert gated.prompt is not None
    assert "Anthropic" in gated.prompt
    assert any(d.code == "guidance_degraded_to_prompt" for d in diags)


def test_degrade_not_requested_strict_raises() -> None:
    with pytest.raises(UnsupportedFeatureError):
        gate_params(
            RuntimeParams(phrase_hints=["x"]), _caps(prompt=True), "batch", strict=True
        )


def test_degrade_but_prompt_unsupported_strict_raises() -> None:
    params = RuntimeParams(phrase_hints=["x"], on_unsupported="degrade_to_prompt")
    with pytest.raises(UnsupportedFeatureError):
        gate_params(params, _caps(prompt=False), "batch", strict=True)
