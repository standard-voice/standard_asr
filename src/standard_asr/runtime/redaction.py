# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Credential-safe rendering of validation errors and exceptions (accident model).

A pydantic :class:`~pydantic.ValidationError` echoes the offending ``input``
value verbatim in both its structured ``errors()`` entries and its ``str()``
form. When a caller mis-places a secret (an ``api_key`` dropped into a request
body, the server's ``options`` field, or the CLI's ``--options`` JSON), that
value would be reflected back into whatever surface renders the error -- a
client response, an intermediary proxy, CI logs, or a copy-pasted bug report.
That is a credential leak, and the project forbids it on **every** transport.

This module is the single owner of that scrubbing rule so the server (HTTP/WS),
the CLI, the compliance suite, and the streaming layer cannot drift on it. It
depends only on ``pydantic`` (no FastAPI), so the near-zero-dependency CLI path
can reuse it without pulling in the optional ``[server]`` stack.

**Trust model (see AGENTS.md).** The defense here targets ACCIDENTS, not
adversaries: an installed engine plugin already runs arbitrary in-process code,
so no log-path machinery can contain a malicious one, and none is attempted.
The rules are deliberately cheap and total:

* validation-error detail is rebuilt from ``type``/``loc``/``msg`` only (the
  ``input`` echo, ``ctx``, and ``url`` are dropped), credential-named fields
  and input-echoing messages are redacted, and a ``loc`` component that does
  not look like a field name (pasted key material is long or punctuated) is
  masked;
* exception text destined for an operator surface goes through
  :func:`safe_exception_summary`, which renders the ``cause``/``context``
  chain one line per link and substitutes the sanitized message for any
  ``ValidationError`` link;
* :func:`log_exception_safely` logs a full traceback only when the active
  chain carries no ``ValidationError`` (a traceback re-renders every link's
  raw message, echo included).

**Accepted limits, by design.** An author who interpolates an error's text
into their own message *and then* discards the chain, or a sensitive value
that happens to be shaped exactly like a field name, can still reach a log.
Closing those residuals required proving properties of third-party code and
introspecting pydantic internals -- machinery whose own complexity produced
more defects than the residuals it closed (the AGENTS.md hard budget exists
because of that history). If one of these residuals bites in practice, the
answer is a targeted rule here, not a prover.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any, Sequence, cast

from pydantic import ValidationError

from standard_asr.contract.exceptions import ConfigError

#: Substring tokens that mark a field name as credential-like. A field whose
#: name contains any of these (case-insensitive) has its value redacted from
#: validation-error detail, so a mis-placed secret (for example, an ``api_key`` put in a
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

#: Placeholder substituted for a ``loc`` component that does not look like a
#: schema field name (see :func:`_sanitize_loc_component`).
_REDACTED_KEY: str = "[redacted-key]"

#: The shape of a plausible schema field name: identifier-ish and short. A
#: pasted secret or key-material blob is long or carries punctuation
#: (``sk-...``, base64, a URL), so it fails this and is masked. Chosen loose
#: on purpose -- masking a real field name costs DX, while an
#: identifier-shaped sensitive value slipping through is an accepted limit
#: (see the module docstring).
_FIELD_NAME_SHAPED_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]{0,31}")

#: pydantic's own structural ``loc`` markers (for example, ``[key]`` for a mapping-key
#: error): schema machinery text, kept verbatim.
_PYDANTIC_LOC_MARKER = re.compile(r"\[[a-z_]+\]")

#: Length cap for one rendered chain link (see :func:`safe_exception_summary`).
_MAX_SUMMARY_CHARS = 400

#: Chain-link cap: a retry loop wrapping its failure each iteration builds an
#: arbitrarily long ``context`` chain; rendering stays bounded.
_MAX_CHAIN_LINKS = 8


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


def _sanitize_loc_component(part: object) -> object:
    """Mask a ``loc`` component that does not look like schema text.

    ``loc`` is not purely schema-derived: a ``dict[str, T]`` field puts the
    caller's own MAPPING KEY into the path, and a mis-pasted secret used as a
    key would otherwise be echoed by every surface that prints the path. Kept
    verbatim: non-strings (list indices, integer keys), field-name-shaped
    strings, and pydantic's structural markers. Everything else -- the long
    or punctuated shape of pasted key material -- becomes
    :data:`_REDACTED_KEY`.

    Args:
        part: One ``loc`` path component.

    Returns:
        The component, or :data:`_REDACTED_KEY`.
    """
    if not isinstance(part, str):
        return part
    if _FIELD_NAME_SHAPED_KEY.fullmatch(part) or _PYDANTIC_LOC_MARKER.fullmatch(part):
        return part
    return _REDACTED_KEY


def _msg_echoes_input(error: dict[str, Any]) -> bool:
    """Return whether an entry's ``msg`` echoes the offending input value.

    pydantic embeds the offending value into the message for validator-authored
    errors -- ``value_error`` / ``assertion_error`` from a custom
    ``field_validator`` / ``model_validator`` raising ``ValueError(f"...{v}...")``
    (a cross-field model validator may echo any field's value, and its ``loc`` is
    empty so the credential-name check cannot catch it). Such a message would
    reflect a mis-placed secret back to the (unauthenticated) caller. The check detects it
    by content: if the stringified input -- or, for a model-level error, any of
    the input mapping's values -- appears in the message, the message is echoing
    it. Value-free built-in messages (``missing``, ``int_parsing``,
    ``extra_forbidden``, and so on) do not match and are preserved.

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
        # safe_str, not str: the input is arbitrary caller data whose __str__
        # can raise, and this check runs inside error paths that must not
        # cascade. An unprintable value cannot have been interpolated into a
        # validator's f-string message either, so skipping it loses nothing.
        candidates = [safe_str(value) for value in cast("dict[Any, Any]", raw_input).values()]
    else:
        candidates = [safe_str(raw_input)]
    return any(text and text in msg for text in candidates)


def sanitize_validation_errors(
    errors: Sequence[Any], *, loc_prefix: Sequence[object] = ()
) -> list[dict[str, Any]]:
    """Strip the echoed ``input`` (and ``url``) from pydantic error entries.

    FastAPI / pydantic's default error detail echoes the offending ``input``
    value verbatim (and may repeat it under ``ctx``). When a caller mis-places a
    secret (for example, an ``api_key`` in the JSON body or ``options``), that value is
    reflected back into the client / any intermediary proxy / a copied bug report
    -- a credential leak. This rebuilds each entry from only the safe structured
    fields (``type``, ``loc``, ``msg``), thereby dropping the ``input`` echo, the
    ``ctx``, and the ``url`` entirely; redacts the ``msg`` of any entry whose
    ``loc`` names a credential-like field or whose message echoes the input
    value; and masks any ``loc`` component that does not look like schema text
    (a caller mapping key can be pasted key material -- see
    :func:`_sanitize_loc_component`).

    Args:
        errors: The raw ``ValidationError.errors()`` / ``RequestValidationError``
            error list.
        loc_prefix: Path components prepended to each entry's ``loc`` so a
            standalone ``ValidationError`` (for example, from an ``options`` build or
            engine construction, whose ``loc`` is relative to its own model) is
            anchored under the request field it came from (for example, ``["options"]`` /
            ``["config"]``). The prefix participates in credential-field
            detection so a prefixed credential path is still redacted; its own
            components are the standard layer's text and are kept verbatim.

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
        # extra_forbidden, and so on) are kept for usefulness.
        redact_msg = loc_is_credential(loc) or _msg_echoes_input(error)
        masked_loc = loc[: len(prefix)] + [
            _sanitize_loc_component(part) for part in loc[len(prefix) :]
        ]
        entry: dict[str, Any] = {
            "type": error.get("type"),
            "loc": masked_loc,
            "msg": _REDACTED if redact_msg else error.get("msg"),
        }
        sanitized.append(entry)
    return sanitized


def sanitized_validation_message(exc: ValidationError, *, prefix: str = "Invalid options") -> str:
    """Build a safe, input-free summary string from a pydantic error.

    Used where a single ``detail`` string is expected (the ``options`` build
    path on the server and the CLI ``--options`` parser, engine-construction
    failures). Mirrors :func:`sanitize_validation_errors`: it names the offending
    fields and the validator message but never echoes the submitted value, and
    redacts credential-like fields entirely.

    Args:
        exc: The pydantic validation error.
        prefix: Leading label naming what failed validation (for example,
            ``"Invalid configuration"`` for engine-construction errors).

    Returns:
        A human-readable, secret-free error string.
    """
    parts: list[str] = []
    for entry in sanitize_validation_errors(exc.errors()):
        loc = ".".join(str(p) for p in entry["loc"]) or "(root)"
        # An entry can legitimately carry no message (defensive shapes); the
        # literal string "None" would read as message CONTENT.
        msg = entry["msg"] if entry["msg"] is not None else "(no message)"
        parts.append(f"{loc}: {msg}")
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
        prefix: Leading label naming what failed (for example, the engine / model).

    Returns:
        A :class:`ConfigError` carrying the scrubbed message and ``details``.
    """
    return ConfigError(
        sanitized_validation_message(exc, prefix=prefix),
        details=sanitize_validation_errors(exc.errors()),
    )


def safe_str(value: object) -> str | None:
    """Stringify a value without raising (a total ``str``).

    Every renderer in this module runs inside an error path -- often the last
    containment layer before a crash escapes -- and an arbitrary object's
    ``__str__`` can itself raise. The error path must degrade, never cascade.

    Args:
        value: The value to stringify.

    Returns:
        ``str(value)``, or ``None`` when stringification raised.
    """
    try:
        return str(value)
    except Exception:
        return None


def _one_line(text: str) -> str:
    """Collapse a text to one bounded line.

    ``str.split()`` with no argument splits on every Unicode whitespace run --
    newlines, NEL, U+2028/U+2029 included -- so a multi-line exception message
    cannot forge additional records in a line-oriented log.

    Args:
        text: The raw text.

    Returns:
        A single-line, length-bounded string.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) > _MAX_SUMMARY_CHARS:
        return collapsed[:_MAX_SUMMARY_CHARS] + "...[truncated]"
    return collapsed


def _chain_links(exc: BaseException) -> list[BaseException]:
    """Return the exception's ``cause``/``context`` chain, outermost first.

    Follows ``__cause__`` when set, else ``__context__`` -- so
    ``raise ... from None`` still exposes the suppressed context (suppression
    is a display choice; a wrapper that copied a ``ValidationError``'s text
    into its own message still has that error in ``__context__``). Bounded
    and cycle-safe.

    Args:
        exc: The outermost exception.

    Returns:
        Up to :data:`_MAX_CHAIN_LINKS` chain links, outermost first.
    """
    links: list[BaseException] = []
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen and len(links) < _MAX_CHAIN_LINKS:
        seen.add(id(node))
        links.append(node)
        node = node.__cause__ if node.__cause__ is not None else node.__context__
    return links


def chain_has_validation_error(exc: BaseException) -> bool:
    """Return whether the ``cause``/``context`` chain carries a ``ValidationError``.

    The one cheap question every rich-rendering decision keys on: a raw
    traceback (or ``str()``) of a chain with a ``ValidationError`` link
    re-renders that link's message, ``input_value`` echo included.

    Args:
        exc: The outermost exception.

    Returns:
        ``True`` if any chain link is a pydantic ``ValidationError``.
    """
    return any(isinstance(link, ValidationError) for link in _chain_links(exc))


def safe_exception_summary(exc: BaseException) -> str:
    """Summarize an exception chain in one line, never echoing pydantic input.

    THE primitive for embedding third-party exception text into an
    operator-visible string -- compliance issue messages, streaming error
    events' ``extra["detail"]``, CLI error reports, server logs. Each
    ``cause``/``context`` link renders as ``TypeName: text`` where ``text``
    is the sanitized message for a ``ValidationError`` link
    (:func:`sanitized_validation_message` -- fields named, values never) and
    the link's own single-lined ``str()`` otherwise.

    Args:
        exc: The exception to summarize.

    Returns:
        A single-line, bounded, input-echo-free summary of the chain.
    """
    links = _chain_links(exc)
    # A wrapper that interpolated the chained error into its own message
    # (``raise RuntimeError(f"engine failed: {exc}")`` -- the most common
    # honest-author echo accident) copied ``str(ve)`` BYTE-FOR-BYTE, so a
    # plain substring test against the chained errors' own lines catches it
    # with no machinery. A paraphrased or re-encoded echo escapes this --
    # the documented accepted limit.
    echo_fragments = [
        line
        for link in links
        if isinstance(link, ValidationError)
        for line in str(link).splitlines()
        if len(line.strip()) >= 16
    ]
    parts: list[str] = []
    for link in links:
        name = type(link).__name__
        if isinstance(link, ValidationError):
            parts.append(sanitized_validation_message(link, prefix=name))
            continue
        rendered = safe_str(link)
        if rendered is None:
            parts.append(f"{name}: <exception str() failed>")
            continue
        if any(fragment in rendered for fragment in echo_fragments):
            parts.append(
                f"{name}: [message withheld: it interpolates the chained "
                "ValidationError, whose text echoes the offending input]"
            )
            continue
        text = _one_line(rendered)
        parts.append(f"{name}: {text}" if text else name)
    return " | caused by ".join(parts)


def log_exception_safely(log: logging.Logger, msg: str, *args: object) -> None:
    """Log the active exception without ever echoing pydantic input values.

    Drop-in replacement for ``Logger.exception`` inside an ``except`` block.
    A chain with no ``ValidationError`` link delegates to ``log.exception``
    unchanged (full traceback -- the operator keeps every frame); a chain
    carrying one logs the scrubbed one-line summary instead, because the
    native traceback formatter re-renders every link's raw message,
    ``input_value`` echo included.

    Args:
        log: The logger to write to.
        msg: The log message (may contain ``%``-style placeholders).
        *args: Arguments for the message's placeholders.
    """
    exc = sys.exc_info()[1]
    if exc is None:
        log.error(msg, *args)
        return
    if chain_has_validation_error(exc):
        # Render the caller's message FIRST, then pass both halves as plain
        # %s arguments. Appending " | %s" to the caller's format string
        # changed its own %-contract on exactly the scrubbed path: a literal
        # % with no args started formatting (getMessage raised, the handler
        # dropped the record -- the scrubbed diagnostic lost), and a single
        # mapping argument stopped being a mapping. A malformed caller
        # format degrades to the unformatted message plus the args' repr --
        # never a lost record.
        try:
            rendered = msg % args if args else msg
        except Exception:  # noqa: BLE001 - a bad caller format must not lose the record
            rendered = f"{msg} {args!r}"
        log.error("%s | %s", rendered, safe_exception_summary(exc))
        return
    log.exception(msg, *args)


__all__ = [
    "chain_has_validation_error",
    "config_error_from_validation",
    "loc_is_credential",
    "loc_to_list",
    "log_exception_safely",
    "safe_exception_summary",
    "safe_str",
    "sanitize_validation_errors",
    "sanitized_validation_message",
]
