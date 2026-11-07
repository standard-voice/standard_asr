# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the StructuredError contract (EC-2: param / hint / details / retriable).

These pin the new exported exception surface so a regression -- for example,
defaulting ``retriable`` to ``False``, dropping a structured field, or rejecting the new
keywords -- fails loudly (the constructor lines reach 100% line coverage via
default-message construction elsewhere, but the keyword paths and resulting
attribute values were otherwise unasserted).
"""

from __future__ import annotations

import itertools

import pytest

from standard_asr.contract.exceptions import (
    ArtifactReportContradictionError,
    ConfigError,
    EngineContractError,
    InvalidProviderParamError,
    StandardASRError,
    StructuredError,
    TranscriptionError,
    UnsupportedFeatureError,
)


def test_transcription_error_retriable_defaults_to_none() -> None:
    # None means "unknown" -- the safe reading is "do not assume it is retryable".
    assert TranscriptionError("boom").retriable is None
    assert TranscriptionError("boom", retriable=True).retriable is True
    assert TranscriptionError("boom", retriable=False).retriable is False


def test_config_error_carries_structured_fields() -> None:
    err = ConfigError(
        "bad config",
        param="base_url",
        hint="use https",
        details=[{"type": "value_error", "loc": ["base_url"], "msg": "[redacted]"}],
    )
    assert err.param == "base_url"
    assert err.hint == "use https"
    assert err.details == [{"type": "value_error", "loc": ["base_url"], "msg": "[redacted]"}]
    # ConfigError MUST remain a ValueError (the server maps it to HTTP 422) and a
    # StandardASRError, while gaining the structured base.
    assert isinstance(err, StructuredError)
    assert isinstance(err, ValueError)
    assert isinstance(err, StandardASRError)


def test_structured_fields_default_to_none_and_message_is_preserved() -> None:
    err = ConfigError("just a message")
    assert err.param is None
    assert err.hint is None
    assert err.details is None
    assert str(err) == "just a message"


def test_unsupported_feature_error_keeps_mode_and_structured_fields() -> None:
    err = UnsupportedFeatureError(
        "word timestamps unsupported", param="word_timestamps", mode="batch", hint="declare it"
    )
    assert err.param == "word_timestamps"
    assert err.mode == "batch"
    assert err.hint == "declare it"
    assert err.details is None
    assert isinstance(err, StructuredError)


def test_invalid_provider_param_error_is_structured_value_error() -> None:
    err = InvalidProviderParamError("wrong engine's params", param="beam_size")
    assert err.param == "beam_size"
    assert isinstance(err, StructuredError)
    assert isinstance(err, ValueError)


def test_structured_error_exposes_exactly_the_documented_attributes() -> None:
    """The app-developer guide's attribute names must be real.

    ``docs/content/app-developers/errors.md`` advertised ``.detail`` / ``.engine_id``,
    which never existed -- copying the published snippet raised
    ``AttributeError``, the very "docs teach a wrong API" failure this
    branch exists to remove. Pin the real surface so the guide cannot drift
    from it again.
    """
    err = ConfigError("bad", param="base_url", hint="use https", details=[{"loc": ["x"]}])
    assert err.param == "base_url"
    assert err.hint == "use https"
    assert err.details == [{"loc": ["x"]}]
    for absent in ("detail", "engine_id"):
        assert not hasattr(err, absent), absent

    unsupported = UnsupportedFeatureError(
        "no word timestamps", param="word_timestamps", mode="batch", hint="drop it"
    )
    assert (unsupported.param, unsupported.mode, unsupported.hint) == (
        "word_timestamps",
        "batch",
        "drop it",
    )


#: The order in which the ``contradictions`` attribute keeps the fields.
_FIXED_ORDER = (
    "applicable",
    "supports_explicit_acquisition",
    "may_acquire_during_inference",
)


@pytest.mark.parametrize(
    "contradictions",
    [
        (),
        ("explicit",),
        ("applicable", "explicit"),
    ],
)
def test_artifact_report_contradiction_error_rejects_a_malformed_list(
    contradictions: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError) as caught:
        ArtifactReportContradictionError("x", contradictions=contradictions)  # pyright: ignore[reportArgumentType]
    assert str(caught.value) == (
        "contradictions must name at least one field, and only these: "
        "applicable, supports_explicit_acquisition, may_acquire_during_inference."
    )


@pytest.mark.parametrize(
    "contradictions",
    [
        *(order for size in (1, 2, 3) for order in itertools.permutations(_FIXED_ORDER, size)),
        ("applicable", "applicable"),
        ("may_acquire_during_inference", "applicable", "may_acquire_during_inference"),
        (*_FIXED_ORDER, *reversed(_FIXED_ORDER)),
    ],
)
def test_artifact_report_contradiction_error_accepts_any_order_and_repeats(
    contradictions: tuple[str, ...],
) -> None:
    # Readers of the attribute rely on seeing each field once, in one fixed
    # order; neither the order nor a repeat in the input carries meaning.
    error = ArtifactReportContradictionError("x", contradictions=contradictions)  # pyright: ignore[reportArgumentType]
    assert error.contradictions == tuple(sorted(set(contradictions), key=_FIXED_ORDER.index))


def test_artifact_report_contradiction_error_keeps_its_fields() -> None:
    error = ArtifactReportContradictionError(
        "message", contradictions=("applicable", "may_acquire_during_inference")
    )
    assert isinstance(error, EngineContractError)
    assert error.contradictions == ("applicable", "may_acquire_during_inference")
    assert str(error) == "message"


def test_artifact_report_contradiction_error_reads_its_input_once() -> None:
    # The constructor checks the fields and then sorts them, so it copies the
    # input first; an iterator would otherwise be empty by the second pass.
    fields = iter(("may_acquire_during_inference", "applicable"))
    error = ArtifactReportContradictionError("x", contradictions=fields)  # pyright: ignore[reportArgumentType]
    assert error.contradictions == ("applicable", "may_acquire_during_inference")
