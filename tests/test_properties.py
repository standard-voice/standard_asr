# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for BaseProperties validation."""

from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import ValidationError

from standard_asr.asr_properties import (
    BaseProperties,
    SampleRateRange,
    nearest_accepted_sample_rate,
    sample_rate_accepted,
)
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
    # Canonical BCP-47 casing (region UPPER), reserved 'auto' preserved.
    assert props.selectable_languages == ["en-US", "auto"]
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


@pytest.mark.parametrize("spelling", ["AUTO", "Auto", "aUtO"])
def test_detectable_rejects_case_variant_auto(spelling: str) -> None:
    # The reserved token must be recognised AFTER normalization. An
    # upper/mixed-case "AUTO" validates as a 4-letter BCP-47 primary subtag and
    # normalizes to "auto", so a pre-normalization literal `== "auto"` guard let
    # it slip into detectable_languages despite the reserved-token ban.
    data = _base_kwargs()
    data["detectable_languages"] = [spelling]
    with pytest.raises(ValueError, match="auto"):
        BaseProperties(**data)


@pytest.mark.parametrize("spelling", ["AUTO", "Auto"])
def test_selectable_case_variant_auto_normalized_and_allowed(spelling: str) -> None:
    # (mirror): on the selectable side the case variant is normalized to
    # the reserved token and ACCEPTED (auto is selectable), not treated as a
    # BCP-47 tag. detectable is required because auto becomes selectable.
    data = _base_kwargs()
    data["selectable_languages"] = [spelling, "en"]
    data["detectable_languages"] = ["en"]
    props = BaseProperties(**data)
    assert props.selectable_languages == ["auto", "en"]
    assert props.supports_auto is True


def test_selectable_rejects_case_only_duplicate() -> None:
    # "en-US" and "EN-US" collapse to the same canonical tag; a
    # post-normalization duplicate is a declaration error (mirrors wire_encodings).
    data = _base_kwargs()
    data["selectable_languages"] = ["en-US", "EN-US"]
    with pytest.raises(ValueError, match="duplicate"):
        BaseProperties(**data)


def test_detectable_rejects_duplicate() -> None:
    # Detectable_languages is a set; a (canonical) duplicate is rejected.
    data = _base_kwargs()
    data["detectable_languages"] = ["en", "EN"]
    with pytest.raises(ValueError, match="duplicate"):
        BaseProperties(**data)


def test_accepted_sample_rates_rejects_duplicate() -> None:
    # A repeated rate is a declaration error, rejected like the other
    # declaration-list duplicate guards.
    data = _base_kwargs()
    data["accepted_sample_rates"] = [16000, 16000, 8000]
    with pytest.raises(ValueError, match="duplicate"):
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
    # Renamed from self_describes_sample_rate (misleading name).
    assert props.accepts_any_sample_rate is True
    # A concrete list is not "any".
    data["accepted_sample_rates"] = [16000]
    assert BaseProperties(**data).accepts_any_sample_rate is False


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


def test_description_is_optional_display_field() -> None:
    # description is a defined, optional, display-only field (spec §AI 3.2).
    assert BaseProperties(**_base_kwargs()).description is None
    props = BaseProperties(**_base_kwargs(), description="A demo engine")
    assert props.description == "A demo engine"


def test_no_free_form_extra_metadata_pocket() -> None:
    # The free-form `extra: dict[str, Any]` metadata pocket was removed
    # (it duplicated the blanket metadata §C deliberately dropped and formed an
    # unschema'd parallel declaration channel). The model is extra="forbid", so an
    # unknown key -- whether spelled `extra` or anything else -- is rejected, not
    # silently absorbed. (Use model_validate with a dict so the now-removed key is
    # a runtime-rejected extra, not a static type error.)
    assert "extra" not in BaseProperties.model_fields
    with pytest.raises(ValidationError):
        BaseProperties.model_validate({**_base_kwargs(), "extra": {"license": "MIT"}})


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


# --- Class-level declaration defaults ---
#
# Plugins declare properties as a subclass with class-level defaults (the
# documented pattern in adapting_engine.md). pydantic
# only runs field validators on defaults with validate_default=True; these
# tests pin that the declaration path cannot bypass validation.


def test_class_level_defaults_run_field_validators() -> None:
    class _BadDeclaration(BaseProperties):
        engine_id: str = "My/Engine!!"  # illegal identifier
        model_name: str = "model"
        protocol_version: str = "0.2.0"
        accepted_input: set[InputKind] = {InputKind.ARRAY}
        native_sample_rate: int = 16000
        accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [
            16000,
            -5,
        ]  # negative rate
        selectable_languages: list[str] = ["Chinese"]  # not BCP-47

    with pytest.raises(ValidationError):
        _BadDeclaration()


def test_class_level_defaults_are_normalized() -> None:
    class _UnnormalizedDeclaration(BaseProperties):
        engine_id: str = "engine"
        model_name: str = "model"
        protocol_version: str = "0.2.0"
        accepted_input: set[InputKind] = {InputKind.ARRAY}
        native_sample_rate: int = 16000
        accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
        selectable_languages: list[str] = ["EN_us"]
        wire_encodings: list[str] | None = ["PCM_S16LE"]

    props = _UnnormalizedDeclaration()
    assert props.selectable_languages == ["en-US"]
    assert props.wire_encodings == ["pcm_s16le"]


# --------------------------------------------------------------------------- #
# Accepted_sample_rates as a continuous range (AWS 8000-48000).
# --------------------------------------------------------------------------- #
def test_sample_rate_range_validates_bounds() -> None:
    r = SampleRateRange(min=8000, max=48000)
    assert (r.min, r.max) == (8000, 48000)
    # min == max (a single-point range) is allowed.
    assert SampleRateRange(min=16000, max=16000).max == 16000
    # min > max is a declaration error.
    with pytest.raises(ValidationError, match="must be <= max"):
        SampleRateRange(min=48000, max=8000)
    # non-positive bounds rejected (gt=0).
    with pytest.raises(ValidationError):
        SampleRateRange(min=0, max=48000)
    with pytest.raises(ValidationError):
        SampleRateRange(min=8000, max=0)


def test_sample_rate_range_contains_is_inclusive() -> None:
    r = SampleRateRange(min=8000, max=48000)
    assert r.contains(8000) is True  # lower bound inclusive
    assert r.contains(48000) is True  # upper bound inclusive
    assert r.contains(16000) is True
    assert r.contains(7999) is False
    assert r.contains(48001) is False


def test_sample_rate_accepted_across_variants() -> None:
    # The shared membership predicate must agree for all three variants.
    assert sample_rate_accepted("any", 12345) is True
    assert sample_rate_accepted([8000, 16000], 16000) is True
    assert sample_rate_accepted([8000, 16000], 22050) is False
    rng = SampleRateRange(min=8000, max=48000)
    assert sample_rate_accepted(rng, 22050) is True
    assert sample_rate_accepted(rng, 6000) is False


def test_nearest_accepted_sample_rate_list_and_range() -> None:
    # List: nearest member, preferring not to upsample on ties.
    assert nearest_accepted_sample_rate([16000, 48000], 22050) == 16000
    assert nearest_accepted_sample_rate([8000, 44100], 40000) == 44100
    # Range: clamp into [min, max] (the closest reachable in-range rate).
    rng = SampleRateRange(min=8000, max=48000)
    assert nearest_accepted_sample_rate(rng, 6000) == 8000  # below -> lower bound
    assert nearest_accepted_sample_rate(rng, 60000) == 48000  # above -> upper bound
    assert nearest_accepted_sample_rate(rng, 22050) == 22050  # inside -> itself
    # "any" has no finite target to choose: an "any" engine accepts the source.
    with pytest.raises(ValueError, match="undefined for 'any'"):
        nearest_accepted_sample_rate("any", 16000)


def test_properties_accepts_sample_rate_range() -> None:
    data = _base_kwargs()
    data["accepted_sample_rates"] = SampleRateRange(min=8000, max=48000)
    data["native_sample_rate"] = 16000
    props = BaseProperties(**data)
    assert isinstance(props.accepted_sample_rates, SampleRateRange)
    # Not "any" -> accepts_any_sample_rate is False.
    assert props.accepts_any_sample_rate is False
    # Serializes to the {min,max} wire shape for cross-language clients.
    assert props.model_dump(mode="json")["accepted_sample_rates"] == {"min": 8000, "max": 48000}


def test_range_reachability_native_must_be_in_range() -> None:
    data = _base_kwargs()
    data["accepted_sample_rates"] = SampleRateRange(min=8000, max=48000)
    # Native inside the range is fine.
    data["native_sample_rate"] = 16000
    assert BaseProperties(**data).native_sample_rate == 16000
    # Native below the range is self-contradictory (its own input would resample).
    data["native_sample_rate"] = 4000
    with pytest.raises(ValidationError, match="native_sample_rate"):
        BaseProperties(**data)


def test_range_reachability_required_must_be_in_range() -> None:
    data = _base_kwargs()
    data["accepted_sample_rates"] = SampleRateRange(min=8000, max=48000)
    data["native_sample_rate"] = 16000
    # Required inside the range is fine.
    data["required_input_sample_rate"] = 24000
    assert BaseProperties(**data).required_input_sample_rate == 24000
    # Required outside the range is an unreachable resample target.
    data["required_input_sample_rate"] = 96000
    with pytest.raises(ValidationError, match="required_input_sample_rate"):
        BaseProperties(**data)
