# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Credential-safe rendering of pydantic validation errors (shared by transports).

A pydantic :class:`~pydantic.ValidationError` echoes the offending ``input``
value verbatim in both its structured ``errors()`` entries and its ``str()``
form. When a caller mis-places a secret (an ``api_key`` dropped into a request
body, the server's ``options`` field, or the CLI's ``--options`` JSON), that
value would be reflected back into whatever surface renders the error -- a
client response, an intermediary proxy, CI logs, or a copy-pasted bug report.
That is a credential leak, and the project forbids it on **every** transport
(spec ``server.md`` §1: "validation errors never echo the request input";
``runtime_params.py`` keeps the raw tag out of the language-validator message
for the same reason).

This module is the single owner of that scrubbing rule so the server (HTTP/WS)
and the CLI cannot drift on it. It depends only on ``pydantic`` (no FastAPI), so
the near-zero-dependency CLI path can reuse it without pulling in the optional
``[server]`` stack.
"""

from __future__ import annotations

from typing import Any, Sequence, cast

from pydantic import ValidationError

from .exceptions import ConfigError

#: Substring tokens that mark a field name as credential-like. A field whose
#: name contains any of these (case-insensitive) has its value redacted from
#: validation-error detail, so a mis-placed secret (e.g. an ``api_key`` put in a
#: request body / ``options`` / ``--options``) is never reflected back to the
#: client / proxy / bug-report logs.
_CREDENTIAL_FIELD_TOKENS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "authorization",
    "auth",
    "credential",
    "private_key",
    "access_key",
    "session_key",
    "bearer",
)

#: Placeholder substituted for a redacted credential value in error detail.
_REDACTED: str = "[redacted]"


def loc_to_list(loc: object) -> list[object]:
    """Normalize a pydantic error ``loc`` into a plain list.

    Args:
        loc: The ``loc`` value from a single pydantic error entry (a tuple/list
            of path components, or a scalar).

    Returns:
        The components as a list (a scalar is wrapped in a single-element list).
    """
    if isinstance(loc, (list, tuple)):
        return list(loc)  # pyright: ignore[reportUnknownArgumentType]
    return [loc]


def loc_is_credential(loc: list[object]) -> bool:
    """Return whether a pydantic error ``loc`` names a credential-like field.

    Args:
        loc: The normalized ``loc`` path components.

    Returns:
        ``True`` if any string component of ``loc`` contains a credential token.
    """
    for part in loc:
        if isinstance(part, str):
            lowered = part.lower()
            if any(token in lowered for token in _CREDENTIAL_FIELD_TOKENS):
                return True
    return False


def _msg_echoes_input(error: dict[str, Any]) -> bool:
    """Return whether an entry's ``msg`` echoes the offending input value.

    pydantic embeds the offending value into the message for validator-authored
    errors -- ``value_error`` / ``assertion_error`` from a custom
    ``field_validator`` / ``model_validator`` raising ``ValueError(f"...{v}...")``
    (a cross-field model validator may echo any field's value, and its ``loc`` is
    empty so the credential-name check cannot catch it). Such a message would
    reflect a mis-placed secret back to the (unauthenticated) caller. We detect it
    by content: if the stringified input -- or, for a model-level error, any of
    the input mapping's values -- appears in the message, the message is echoing
    it. Value-free built-in messages (``missing``, ``int_parsing``,
    ``extra_forbidden``, ...) do not match and are preserved.

    Args:
        error: A single ``ValidationError.errors()`` entry.

    Returns:
        ``True`` if the message contains the offending input value verbatim.
    """
    msg = error.get("msg")
    if not isinstance(msg, str) or not msg:
        return False
    raw_input = error.get("input")
    if isinstance(raw_input, dict):
        # A model-level (cross-field) validator error carries the input mapping
        # (loc == ()); scan its field values so an echoed value is caught even
        # though the credential-name check cannot fire on an empty loc. (pydantic
        # gives a scalar for a field error and this dict for a model error.)
        candidates = [str(value) for value in cast("dict[Any, Any]", raw_input).values()]
    else:
        candidates = [str(raw_input)]
    return any(text and text in msg for text in candidates)


def sanitize_validation_errors(
    errors: Sequence[Any], *, loc_prefix: Sequence[object] = ()
) -> list[dict[str, Any]]:
    """Strip the echoed ``input`` (and ``url``) from pydantic error entries.

    FastAPI / pydantic's default error detail echoes the offending ``input``
    value verbatim (and may repeat it under ``ctx``). When a caller mis-places a
    secret (e.g. an ``api_key`` in the JSON body or ``options``), that value is
    reflected back into the client / any intermediary proxy / a copied bug report
    -- a credential leak. This rebuilds each entry from only the safe structured
    fields (``type``, ``loc``, ``msg``), thereby dropping the ``input`` echo, the
    ``ctx``, and the ``url`` entirely, and additionally redacts the ``msg`` of any
    entry whose ``loc`` names a credential-like field (whose validator message
    could itself contain the value).

    Args:
        errors: The raw ``ValidationError.errors()`` / ``RequestValidationError``
            error list.
        loc_prefix: Path components prepended to each entry's ``loc`` so a
            standalone ``ValidationError`` (e.g. from an ``options`` build or
            engine construction, whose ``loc`` is relative to its own model) is
            anchored under the request field it came from (e.g. ``["options"]`` /
            ``["config"]``). The prefix participates in credential-field
            detection so a prefixed credential path is still redacted.

    Returns:
        A new list of sanitized error entries safe to return to a client.
    """
    prefix = list(loc_prefix)
    sanitized: list[dict[str, Any]] = []
    for raw in errors:
        error: dict[str, Any] = dict(raw)
        loc = prefix + loc_to_list(error.get("loc", ()))
        # Redact the message when (a) the field is credential-named, or (b) the
        # message echoes the offending input value. pydantic embeds the value in
        # the message for validator-authored errors (value_error / assertion_error,
        # including cross-field model_validators whose loc is empty), which would
        # otherwise leak a mis-placed secret into the unauthenticated error
        # surface; value-free built-in messages (missing, int_parsing,
        # extra_forbidden, ...) are kept for usefulness.
        redact_msg = loc_is_credential(loc) or _msg_echoes_input(error)
        entry: dict[str, Any] = {
            "type": error.get("type"),
            "loc": loc,
            "msg": _REDACTED if redact_msg else error.get("msg"),
        }
        sanitized.append(entry)
    return sanitized


def sanitized_validation_message(exc: ValidationError, *, prefix: str = "Invalid options") -> str:
    """Build a safe, input-free summary string from a pydantic error.

    Used where a single ``detail`` string is expected (the ``options`` build
    path on the server and the CLI ``--options`` parser, engine-construction
    failures). Mirrors :func:`sanitize_validation_errors`: it names the offending
    field(s) and the validator message but never echoes the submitted value, and
    redacts credential-like fields entirely.

    Args:
        exc: The pydantic validation error.
        prefix: Leading label naming what failed validation (e.g.
            ``"Invalid configuration"`` for engine-construction errors).

    Returns:
        A human-readable, secret-free error string.
    """
    parts: list[str] = []
    for entry in sanitize_validation_errors(exc.errors()):
        loc = ".".join(str(p) for p in entry["loc"]) or "(root)"
        parts.append(f"{loc}: {entry['msg']}")
    joined = "; ".join(parts) or "invalid value"
    return f"{prefix}: {joined}"


def config_error_from_validation(
    exc: ValidationError, *, prefix: str = "Invalid configuration"
) -> ConfigError:
    """Wrap a construction-time ``ValidationError`` as a secret-safe ``ConfigError``.

    Init-config validation raises pydantic's ``ValidationError``, which an
    application cannot catch as the standard layer's :class:`ConfigError` and
    which echoes the offending input verbatim. This rebuilds it as a
    ``ConfigError`` whose message is the input-free summary
    (:func:`sanitized_validation_message`) and whose ``details`` carries the
    sanitized structured entries -- so callers can ``except ConfigError``
    uniformly and a mis-placed secret is never reflected back (EC-1; spec:
    explicit error contract, never echo a credential).

    Args:
        exc: The pydantic validation error raised at construction.
        prefix: Leading label naming what failed (e.g. the engine / model).

    Returns:
        A :class:`ConfigError` carrying the scrubbed message and ``details``.
    """
    return ConfigError(
        sanitized_validation_message(exc, prefix=prefix),
        details=sanitize_validation_errors(exc.errors()),
    )


__all__ = [
    "config_error_from_validation",
    "sanitize_validation_errors",
    "sanitized_validation_message",
]
