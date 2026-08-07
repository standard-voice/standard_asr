# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the accident-model scrubber (runtime/redaction.py).

The module's contract is three cheap rules (see AGENTS.md's trust model):
validation-error detail never echoes the input, exception text destined for
an operator surface goes through the chain summary, and a full traceback is
logged only when the chain carries no ``ValidationError``.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from standard_asr.contract.exceptions import ConfigError
from standard_asr.runtime.redaction import (
    chain_has_validation_error,
    config_error_from_validation,
    loc_is_credential,
    loc_to_list,
    log_exception_safely,
    safe_exception_summary,
    safe_str,
    sanitize_validation_errors,
    sanitized_validation_message,
)

_SECRET = "sk-REDACTION-SENTINEL"  # noqa: S105 - test fixture, not a real credential

logger = logging.getLogger("standard_asr.tests.redaction")


class _Params(BaseModel):
    """A strict model whose failures echo the offending input."""

    model_config = ConfigDict(extra="forbid")

    beam: int


class _CredentialModel(BaseModel):
    """A model with a credential-named field (message redacted wholesale)."""

    api_key: int  # int so a str input fails validation


class _EchoingModel(BaseModel):
    """A model whose validator message embeds the input value."""

    hint: str

    @field_validator("hint")
    @classmethod
    def _reject(cls, value: str) -> str:
        raise ValueError(f"bad value {value}")


class _CrossFieldEcho(BaseModel):
    """A cross-field validator (empty loc) echoing a field value."""

    a: str
    b: str

    @model_validator(mode="after")
    def _reject(self) -> "_CrossFieldEcho":
        raise ValueError(f"mismatch: {self.b}")


def _validation_error(model: type[BaseModel], payload: dict[str, object]) -> ValidationError:
    try:
        model.model_validate(payload)
    except ValidationError as exc:
        return exc
    raise AssertionError("validation must fail")


# --------------------------------------------------------------------------- #
# loc helpers
# --------------------------------------------------------------------------- #
def test_loc_to_list_normalizes_tuples_and_scalars() -> None:
    assert loc_to_list(("a", 0, "b")) == ["a", 0, "b"]
    assert loc_to_list(["x"]) == ["x"]
    assert loc_to_list("bare") == ["bare"]


def test_loc_is_credential_matches_tokens_case_insensitively() -> None:
    assert loc_is_credential(["options", "api_key"]) is True
    assert loc_is_credential(["Authorization"]) is True
    assert loc_is_credential(["options", 3, "beam"]) is False


# --------------------------------------------------------------------------- #
# sanitize_validation_errors
# --------------------------------------------------------------------------- #
def test_input_echo_ctx_and_url_are_dropped() -> None:
    exc = _validation_error(_Params, {"beam": _SECRET})
    raw = exc.errors()
    assert any(_SECRET in repr(entry) for entry in raw)  # pydantic does echo

    sanitized = sanitize_validation_errors(raw)
    assert _SECRET not in repr(sanitized)
    entry = sanitized[0]
    assert set(entry) == {"type", "loc", "msg"}  # input/ctx/url rebuilt away
    assert entry["loc"] == ["beam"]
    assert entry["msg"]  # the value-free builtin message survives


def test_credential_named_field_message_is_redacted() -> None:
    exc = _validation_error(_CredentialModel, {"api_key": _SECRET})
    sanitized = sanitize_validation_errors(exc.errors())
    assert sanitized[0]["msg"] == "[redacted]"
    assert _SECRET not in repr(sanitized)


def test_validator_message_echoing_the_input_is_redacted() -> None:
    exc = _validation_error(_EchoingModel, {"hint": _SECRET})
    sanitized = sanitize_validation_errors(exc.errors())
    assert sanitized[0]["msg"] == "[redacted]"
    assert _SECRET not in repr(sanitized)


def test_cross_field_echo_with_empty_loc_is_redacted() -> None:
    # A model-level validator has loc == (): the credential-name rule cannot
    # fire, so the input-mapping content scan must catch the echo.
    exc = _validation_error(_CrossFieldEcho, {"a": "x", "b": _SECRET})
    sanitized = sanitize_validation_errors(exc.errors())
    assert all(_SECRET not in repr(entry) for entry in sanitized)


def test_loc_prefix_anchors_and_participates_in_credential_detection() -> None:
    exc = _validation_error(_Params, {"beam": "no"})
    sanitized = sanitize_validation_errors(exc.errors(), loc_prefix=["options"])
    assert sanitized[0]["loc"] == ["options", "beam"]

    # A credential-named PREFIX redacts the message too.
    entries = [{"type": "int_parsing", "loc": ("value",), "msg": "not an int", "input": "x"}]
    redacted = sanitize_validation_errors(entries, loc_prefix=["api_key"])
    assert redacted[0]["msg"] == "[redacted]"


def test_mapping_key_shaped_like_key_material_is_masked_in_loc() -> None:
    # A dict-typed field puts the caller's own MAPPING KEY into loc; a
    # mis-pasted secret used as a key must not be echoed by the path.
    class _WithDict(BaseModel):
        settings: dict[str, int]

    pasted = "sk-LONG-PASTED-SECRET-VALUE-0123456789-abcdef"  # noqa: S105
    exc = _validation_error(_WithDict, {"settings": {pasted: "bad"}})
    sanitized = sanitize_validation_errors(exc.errors())
    assert pasted not in repr(sanitized)
    assert sanitized[0]["loc"] == ["settings", "[redacted-key]"]

    # Field-name-shaped components, integers, and pydantic's structural
    # markers pass verbatim (masking real field names costs DX; an
    # identifier-shaped sensitive value is an accepted limit -- see the
    # module docstring).
    entries = [{"type": "t", "loc": ("settings", 3, "[key]", "threads"), "msg": "m"}]
    kept = sanitize_validation_errors(entries)
    assert kept[0]["loc"] == ["settings", 3, "[key]", "threads"]

    # The standard-authored prefix is never masked, whatever its shape.
    prefixed = sanitize_validation_errors(
        [{"type": "t", "loc": ("f",), "msg": "m"}], loc_prefix=["config (from --set)"]
    )
    assert prefixed[0]["loc"] == ["config (from --set)", "f"]


def test_unprintable_input_cannot_crash_the_echo_scan() -> None:
    class _Unprintable:
        def __str__(self) -> str:
            raise RuntimeError("no str for you")

    entries = [
        {"type": "t", "loc": ("f",), "msg": "plain message", "input": _Unprintable()},
        {"type": "t", "loc": (), "msg": "model message", "input": {"k": _Unprintable()}},
        {"type": "t", "loc": ("g",), "msg": None, "input": "x"},
    ]
    sanitized = sanitize_validation_errors(entries)
    assert sanitized[0]["msg"] == "plain message"
    assert sanitized[1]["msg"] == "model message"
    assert sanitized[2]["msg"] is None


# --------------------------------------------------------------------------- #
# sanitized_validation_message / config_error_from_validation
# --------------------------------------------------------------------------- #
def test_sanitized_message_names_fields_never_values() -> None:
    exc = _validation_error(_Params, {"beam": _SECRET, "extra": 1})
    message = sanitized_validation_message(exc, prefix="Invalid configuration")
    assert message.startswith("Invalid configuration: ")
    assert "beam" in message
    assert _SECRET not in message


def test_sanitized_message_handles_absent_msg_and_empty_loc() -> None:
    # Crafted defensive shapes: an entry with no msg must not render "None"
    # as content, and an empty loc renders "(root)".
    class _FakeError:
        def errors(self) -> list[dict[str, object]]:
            return [{"type": "t", "loc": (), "msg": None}]

    message = sanitized_validation_message(_FakeError())  # type: ignore[arg-type]
    assert message == "Invalid options: (root): (no message)"


def test_config_error_wrap_is_catchable_and_secret_free() -> None:
    exc = _validation_error(_CredentialModel, {"api_key": _SECRET})
    wrapped = config_error_from_validation(exc, prefix="Invalid configuration for 'x'")
    assert isinstance(wrapped, ConfigError)
    assert _SECRET not in str(wrapped)
    details = wrapped.details
    assert details is not None
    assert _SECRET not in repr(details)
    assert details[0]["loc"] == ["api_key"]


# --------------------------------------------------------------------------- #
# safe_str / chain helpers
# --------------------------------------------------------------------------- #
def test_safe_str_degrades_to_none_on_raising_str() -> None:
    class _Hostile:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    assert safe_str("plain") == "plain"
    assert safe_str(_Hostile()) is None


def test_chain_has_validation_error_follows_cause_and_suppressed_context() -> None:
    ve = _validation_error(_Params, {"beam": _SECRET})

    plain = RuntimeError("no pydantic anywhere")
    assert chain_has_validation_error(plain) is False

    try:
        raise RuntimeError("wrapper") from ve
    except RuntimeError as exc:
        assert chain_has_validation_error(exc) is True

    # `raise ... from None` hides the context from DISPLAY, not from the
    # chain: a wrapper that copied the error's text still carries the link.
    try:
        try:
            raise ve
        except ValidationError:
            raise RuntimeError(f"copied: {ve}") from None  # noqa: B904
    except RuntimeError as exc:
        assert chain_has_validation_error(exc) is True


def test_chain_walk_is_cycle_safe_and_bounded() -> None:
    # A self-referential cause terminates...
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert chain_has_validation_error(a) is False
    assert "caused by" in safe_exception_summary(a)

    # ...and an organically long context chain stays bounded.
    exc: BaseException = RuntimeError("link 0")
    for index in range(1, 40):
        nxt = RuntimeError(f"link {index}")
        nxt.__context__ = exc
        exc = nxt
    summary = safe_exception_summary(exc)
    assert summary.count("caused by") < 10


# --------------------------------------------------------------------------- #
# safe_exception_summary
# --------------------------------------------------------------------------- #
def test_summary_sanitizes_validation_error_links() -> None:
    ve = _validation_error(_Params, {"beam": _SECRET})
    try:
        raise RuntimeError("engine failed") from ve
    except RuntimeError as exc:
        summary = safe_exception_summary(exc)
    assert _SECRET not in summary
    assert "RuntimeError: engine failed" in summary
    assert "ValidationError" in summary
    assert "beam" in summary  # the field is named; the value is not


def test_summary_is_one_bounded_line() -> None:
    forged = "line one\nWARNING forged line and separators\x85everywhere"
    summary = safe_exception_summary(RuntimeError(forged))
    assert len(summary.splitlines()) == 1
    assert "WARNING forged line" in summary  # content kept, boundaries collapsed

    long_message = "x" * 5000
    bounded = safe_exception_summary(RuntimeError(long_message))
    assert len(bounded) < 600
    assert "...[truncated]" in bounded


def test_summary_degrades_on_raising_or_empty_str() -> None:
    class _HostileStr(Exception):
        def __str__(self) -> str:
            raise RuntimeError("boom")

    assert "HostileStr: <exception str() failed>" in safe_exception_summary(_HostileStr())
    # An empty message renders the bare type name, no dangling colon.
    assert safe_exception_summary(RuntimeError()) == "RuntimeError"


# --------------------------------------------------------------------------- #
# log_exception_safely
# --------------------------------------------------------------------------- #
def test_log_full_traceback_when_chain_is_pydantic_free(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            raise RuntimeError("engine exploded")
        except RuntimeError:
            log_exception_safely(logger, "transcription failed for %r", "m1")
    record = caplog.records[0]
    assert record.exc_info is not None  # full traceback kept for the operator
    assert "transcription failed for 'm1'" in record.getMessage()


def test_log_scrubbed_summary_when_chain_carries_validation_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ve = _validation_error(_Params, {"beam": _SECRET})
    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            raise RuntimeError("engine exploded") from ve
        except RuntimeError:
            log_exception_safely(logger, "transcription failed for %r", "m1")
    record = caplog.records[0]
    assert record.exc_info is None  # the traceback would re-render the echo
    text = record.getMessage()
    assert _SECRET not in text
    assert "transcription failed for 'm1'" in text
    assert "RuntimeError: engine exploded" in text


def test_log_scrubbed_path_survives_a_malformed_caller_format(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The scrubbed record is never lost to the caller's %-format.

    Appending " | %s" to the caller's format string changed its %-contract
    on exactly the scrubbed path: a literal % with no args started
    formatting, getMessage raised, and the handler DROPPED the record --
    the scrubbed diagnostic lost. The message is rendered first now, and a
    genuinely malformed caller format degrades to message + args repr.
    """
    ve = _validation_error(_Params, {"beam": _SECRET})

    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            raise RuntimeError("wrapper") from ve
        except RuntimeError:
            log_exception_safely(logger, "progress at 100% done")  # literal %
    text = caplog.records[0].getMessage()  # must not raise
    assert "progress at 100% done" in text
    assert _SECRET not in text

    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            raise RuntimeError("wrapper") from ve
        except RuntimeError:
            # Mismatched placeholders: degrade, never lose the record.
            log_exception_safely(logger, "model %s rate %d", "m1")
    text = caplog.records[0].getMessage()
    assert "model %s rate %d" in text and "'m1'" in text
    assert _SECRET not in text


def test_log_without_active_exception_logs_the_message_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger=logger.name):
        log_exception_safely(logger, "no exception context for %r", "m1")
    record = caplog.records[0]
    assert record.exc_info is None
    assert record.getMessage() == "no exception context for 'm1'"
