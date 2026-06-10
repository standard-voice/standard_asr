# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared credential-safe validation-error rendering.

These cover the single owner that the server (HTTP/WS) and the CLI ``--options``
parser both delegate to, so the "validation errors never echo the request input"
rule (spec server.md §1) cannot drift between transports.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from standard_asr.error_redaction import (
    config_error_from_validation,
    loc_is_credential,
    loc_to_list,
    sanitize_validation_errors,
    sanitized_validation_message,
)


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str | None = None


def test_loc_to_list_wraps_scalar_and_passes_sequence() -> None:
    assert loc_to_list("api_key") == ["api_key"]
    assert loc_to_list(("a", 0)) == ["a", 0]
    assert loc_to_list(["b", 1]) == ["b", 1]


def test_loc_is_credential_matches_credential_tokens() -> None:
    assert loc_is_credential(["api_key"]) is True
    assert loc_is_credential(["Authorization"]) is True  # case-insensitive
    assert loc_is_credential(["language"]) is False
    assert loc_is_credential([0, 1]) is False  # non-string components ignored


def test_sanitize_drops_input_echo() -> None:
    # The extra-forbidden value must not survive into the sanitized entries.
    secret = "sk-SECRET"  # noqa: S105 - test fixture
    try:
        _Closed.model_validate({"api_key": secret})
    except ValidationError as exc:
        sanitized = sanitize_validation_errors(exc.errors())
    else:  # pragma: no cover - the model must reject the extra key
        pytest.fail("expected a ValidationError")

    flat = repr(sanitized)
    assert secret not in flat
    # Each entry keeps only the safe structured fields.
    for entry in sanitized:
        assert set(entry) == {"type", "loc", "msg"}
    # A credential-named field has its message redacted too.
    assert any(entry["msg"] == "[redacted]" for entry in sanitized)


def test_sanitized_message_never_echoes_value_and_names_field() -> None:
    secret = "sk-LEAK"  # noqa: S105 - test fixture
    try:
        _Closed.model_validate({"api_key": secret})
    except ValidationError as exc:
        message = sanitized_validation_message(exc)
    else:  # pragma: no cover
        pytest.fail("expected a ValidationError")

    assert secret not in message
    assert message.startswith("Invalid options:")
    assert "[redacted]" in message


def test_sanitized_message_custom_prefix() -> None:
    try:
        _Closed.model_validate({"unknown": 1})
    except ValidationError as exc:
        message = sanitized_validation_message(exc, prefix="Invalid configuration")
    else:  # pragma: no cover
        pytest.fail("expected a ValidationError")

    assert message.startswith("Invalid configuration:")
    # A non-credential field name is preserved (only the value is dropped).
    assert "unknown" in message


class _EchoingConfig(BaseModel):
    """A config whose validators embed the offending value in their message.

    Models the realistic leak vector: a third-party engine config the standard
    layer does not control, with a per-field validator and a cross-field model
    validator that each echo the value -- the discipline the in-tree validators
    follow by hand, but an external author may not (so the redactor must be
    value-aware, not just credential-name-aware).
    """

    base_url: str = "https://example"
    region: str = "us"

    @field_validator("base_url")
    @classmethod
    def _https_only(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError(f"base_url must be https, got {value!r}")
        return value

    @field_validator("region")
    @classmethod
    def _no_spaces(cls, value: str) -> str:
        if " " in value:
            raise ValueError("region must not contain spaces")  # value-free message
        return value

    @model_validator(mode="after")
    def _known_region(self) -> _EchoingConfig:
        if self.region != "us":
            raise ValueError(f"unsupported region {self.region!r}")
        return self


def test_field_validator_value_in_message_is_redacted() -> None:
    # A NON-credential-named field whose validator echoes the value must not leak
    # a mis-placed secret through msg / details (EC-1 value-aware redaction). This
    # FAILS under loc-only redaction.
    secret = "sk-live-FIELD-LEAK"  # noqa: S105 - test fixture
    try:
        _EchoingConfig(base_url=secret)
    except ValidationError as exc:
        sanitized = sanitize_validation_errors(exc.errors())
        message = sanitized_validation_message(exc)
        wrapped = config_error_from_validation(exc)
    else:  # pragma: no cover
        pytest.fail("expected a ValidationError")

    assert secret not in repr(sanitized)
    assert secret not in message
    assert secret not in str(wrapped)
    assert secret not in repr(wrapped.details)
    assert "base_url" in message  # the field is still named


def test_cross_field_model_validator_value_is_redacted() -> None:
    # A model-level validator's error has loc == () (the credential-name check can
    # never fire), yet an echoed field value must still be redacted by content.
    secret = "sk-live-MODEL-LEAK"  # noqa: S105 - test fixture
    try:
        _EchoingConfig(region=secret)
    except ValidationError as exc:
        sanitized = sanitize_validation_errors(exc.errors())
        message = sanitized_validation_message(exc)
        wrapped = config_error_from_validation(exc)
    else:  # pragma: no cover
        pytest.fail("expected a ValidationError")

    assert secret not in repr(sanitized)
    assert secret not in message
    assert secret not in str(wrapped)
    assert secret not in repr(wrapped.details)


def test_value_free_validator_message_is_preserved() -> None:
    # The redactor only redacts when the value actually appears: a value-free
    # validator message stays helpful (no over-redaction).
    try:
        _EchoingConfig(region="has space")
    except ValidationError as exc:
        message = sanitized_validation_message(exc)
    else:  # pragma: no cover
        pytest.fail("expected a ValidationError")

    assert "must not contain spaces" in message
    assert "has space" not in message


def test_sanitize_handles_entry_without_message() -> None:
    # Defensive: an error entry lacking a "msg" must not crash the redactor and
    # is treated as non-echoing (covers the msg guard).
    sanitized = sanitize_validation_errors([{"type": "x", "loc": ["field"], "input": "v"}])
    assert sanitized == [{"type": "x", "loc": ["field"], "msg": None}]
