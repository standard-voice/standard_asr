# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for BaseProperties validation."""

from __future__ import annotations

from typing import Any

import pytest

from standard_asr.asr_properties import BaseProperties
from standard_asr.audio_input import InputKind


def _base_kwargs() -> dict[str, Any]:
    return {
        "engine_id": "engine",
        "model_name": "model",
        "protocol_version": "0.2.0",
        "accepted_input": {InputKind.ARRAY},
        "native_sample_rate": 16000,
        "accepted_sample_rates": [16000],
        "selectable_languages": ["en-US"],
    }


def test_selectable_language_normalization() -> None:
    data = _base_kwargs()
    data["selectable_languages"] = ["EN_us", "auto"]
    data["detectable_languages"] = ["en"]
    props = BaseProperties(**data)
    assert props.selectable_languages == ["en-us", "auto"]
    assert props.supports_auto is True
    assert props.has_language_axis is True


def test_invalid_selectable_language_raises() -> None:
    data = _base_kwargs()
    data["selectable_languages"] = ["bad@@tag"]
    with pytest.raises(ValueError):
        BaseProperties(**data)


def test_native_language_name_rejected_in_selectable() -> None:
    # A native language name (not BCP-47) must fail loud, not be silently
    # accepted as a 7-letter primary subtag.
    data = _base_kwargs()
    data["selectable_languages"] = ["Chinese"]
    with pytest.raises(ValueError, match="BCP 47"):
        BaseProperties(**data)


def test_no_language_axis_is_allowed() -> None:
    data = _base_kwargs()
    data["selectable_languages"] = []
    props = BaseProperties(**data)
    assert props.has_language_axis is False


def test_auto_requires_detectable() -> None:
    data = _base_kwargs()
    data["selectable_languages"] = ["auto"]
    data["detectable_languages"] = []
    with pytest.raises(ValueError, match="detectable_languages"):
        BaseProperties(**data)


def test_detectable_rejects_auto_token() -> None:
    data = _base_kwargs()
    data["detectable_languages"] = ["auto"]
    with pytest.raises(ValueError):
        BaseProperties(**data)


def test_accepted_input_must_be_nonempty() -> None:
    data = _base_kwargs()
    data["accepted_input"] = set()
    with pytest.raises(ValueError, match="accepted_input"):
        BaseProperties(**data)


def test_accepted_sample_rates_any() -> None:
    data = _base_kwargs()
    data["accepted_sample_rates"] = "any"
    props = BaseProperties(**data)
    assert props.self_describes_sample_rate is True


def test_accepted_sample_rates_validation() -> None:
    data = _base_kwargs()
    data["accepted_sample_rates"] = []
    with pytest.raises(ValueError):
        BaseProperties(**data)
    data = _base_kwargs()
    data["accepted_sample_rates"] = [0]
    with pytest.raises(ValueError):
        BaseProperties(**data)


def test_native_sample_rate_positive() -> None:
    data = _base_kwargs()
    data["native_sample_rate"] = 0
    with pytest.raises(ValueError):
        BaseProperties(**data)


def test_engine_id_validation_errors() -> None:
    data = _base_kwargs()
    data["engine_id"] = "Bad/Engine"
    with pytest.raises(ValueError):
        BaseProperties(**data)


def test_model_name_validation_errors() -> None:
    data = _base_kwargs()
    data["model_name"] = "bad/name"
    with pytest.raises(ValueError):
        BaseProperties(**data)


def test_model_id() -> None:
    props = BaseProperties(**_base_kwargs())
    assert props.model_id == "engine/model"
