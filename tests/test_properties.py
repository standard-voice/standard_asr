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


def test_native_rate_must_be_in_accepted_rates() -> None:
    # A telephony 8 kHz native rate excluded from accepted_sample_rates would be
    # silently upsampled -- reject at declaration time.
    data = _base_kwargs()
    data["native_sample_rate"] = 8000
    data["accepted_sample_rates"] = [16000]
    with pytest.raises(ValueError, match="native_sample_rate"):
        BaseProperties(**data)


def test_native_rate_in_accepted_rates_ok() -> None:
    data = _base_kwargs()
    data["native_sample_rate"] = 8000
    data["accepted_sample_rates"] = [8000, 16000]
    props = BaseProperties(**data)
    assert props.native_sample_rate == 8000


def test_native_rate_reachability_skipped_for_any() -> None:
    data = _base_kwargs()
    data["native_sample_rate"] = 8000
    data["accepted_sample_rates"] = "any"
    props = BaseProperties(**data)
    assert props.native_sample_rate == 8000


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


def test_wire_encodings_none_is_unconstrained() -> None:
    # Explicitly None runs the validator's None branch and means "unconstrained".
    data = _base_kwargs()
    data["wire_encodings"] = None
    props = BaseProperties(**data)
    assert props.wire_encodings is None


def test_wire_encodings_normalized_to_lowercase() -> None:
    data = _base_kwargs()
    data["wire_encodings"] = ["PCM_S16LE", "MuLaw"]
    props = BaseProperties(**data)
    assert props.wire_encodings == ["pcm_s16le", "mulaw"]


def test_wire_encodings_empty_list_rejected() -> None:
    data = _base_kwargs()
    data["wire_encodings"] = []
    with pytest.raises(ValueError, match="empty list"):
        BaseProperties(**data)


def test_wire_encodings_blank_entry_rejected() -> None:
    data = _base_kwargs()
    data["wire_encodings"] = ["pcm_s16le", "   "]
    with pytest.raises(ValueError, match="blank"):
        BaseProperties(**data)


def test_wire_encodings_duplicate_entry_rejected() -> None:
    # Case-insensitive: "PCM_S16LE" and "pcm_s16le" are the same encoding.
    data = _base_kwargs()
    data["wire_encodings"] = ["pcm_s16le", "PCM_S16LE"]
    with pytest.raises(ValueError, match="duplicate"):
        BaseProperties(**data)


def test_detectable_rejects_invalid_bcp47() -> None:
    # A non-'auto' but malformed tag in detectable_languages must fail loud.
    data = _base_kwargs()
    data["detectable_languages"] = ["not@@valid"]
    with pytest.raises(ValueError, match="BCP 47"):
        BaseProperties(**data)
