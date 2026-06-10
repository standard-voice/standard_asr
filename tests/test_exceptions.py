# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the StructuredError contract (EC-2: param / hint / details / retriable).

These pin the new exported exception surface so a regression -- e.g. defaulting
``retriable`` to ``False``, dropping a structured field, or rejecting the new
keywords -- fails loudly (the constructor lines reach 100% line coverage via
default-message construction elsewhere, but the keyword paths and resulting
attribute values were otherwise unasserted).
"""

from __future__ import annotations

from standard_asr.exceptions import (
    ConfigError,
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
