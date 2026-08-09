# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The runtime sync-call boundary of the StandardASR protocol surface.

Every synchronous protocol member (``transcribe`` / ``start_transcription`` /
``supports`` / ``recommended_wire_format`` / ``prepare``) is implemented by
arbitrary third-party plugin code, so its return value is a trust boundary:
an ``async def`` implementation (or a sync wrapper delegating to one) hands
back an awaitable that is simultaneously the wrong type for every consumer
AND a leak (a never-awaited coroutine), and a wrong-typed value would
otherwise surface only as a confusing secondary ``AttributeError`` deep in
some other subsystem -- or worse, be silently misread (a truthy non-bool from
``supports()`` reads as "supported" everywhere).

This module is the single owner of the detection rule, shared by every
consumer of the protocol surface:

* the compliance suite adapts :func:`sync_result_defect` verdicts into stable
  issue codes (``protocol_member_not_synchronous`` /
  ``protocol_member_wrong_return_type`` /
  ``protocol_member_unclassifiable_result`` / ``sync_bridge_invalid_session``);
* the CLI, the reference server (REST and WebSocket), and any other runtime
  consumer raise through :func:`require_sync_result`, mapping the resulting
  :class:`~standard_asr.contract.exceptions.EngineContractError` onto their
  engine-fault surface (CLI exit 1, scrubbed HTTP 500, ``internal_error``
  frame) -- never onto a caller-fixable surface.

One owner means a new consumer can never re-invent half of the boundary (the
exact drift that once left real CLI/server call sites unguarded while the
compliance probes were). Messages carry type NAMES only -- never
``repr(value)`` -- so an engine-fabricated object can never smuggle payload
or credential text into an error surface.

Depends only on the stdlib and the contract layer, so every consumer (the
near-zero-dependency CLI included) can use it without optional extras.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Coroutine, Literal, cast

from standard_asr.contract.exceptions import EngineContractError

__all__ = [
    "SyncDefectKind",
    "SyncResultDefect",
    "require_sync_result",
    "safe_class_name",
    "safe_type_name",
    "sync_result_defect",
]

#: Fixed placeholder for a type whose metadata name cannot be safely rendered.
#: A name that does not conform to the identifier grammar below carries no
#: diagnostic value anyway (a real type's name identifies it), so degrading to
#: this loses nothing and denies the smuggling channel entirely.
_UNNAMEABLE_TYPE = "<unnameable type>"

#: Upper bound on a rendered metadata name's length. A pathological
#: ``type("A"*100000, (), {})`` must not amplify into an unbounded error
#: message; past this the name is refused (placeholder), never truncated into
#: a still-huge fragment.
_MAX_TYPE_NAME_CHARS = 200

#: The interpreter's OWN getset descriptors for a type's metadata names.
#: Reading a name THROUGH these (``descriptor.__get__(cls)``) reads the stored
#: C-level slot directly, bypassing a metaclass ``__getattribute__`` /
#: ``property`` that would otherwise hand back attacker-chosen text (or run
#: author code) on every ``cls.__name__`` read.
_TYPE_NAME_DESCRIPTOR = type.__dict__["__name__"]
_TYPE_QUALNAME_DESCRIPTOR = type.__dict__["__qualname__"]
_TYPE_MODULE_DESCRIPTOR = type.__dict__["__module__"]


def _conforms_to_name_grammar(value: str, *, dotted: bool) -> bool:
    """Return whether a metadata name is a safe identifier / qualified name.

    A legitimate type name is a Python identifier; a ``__qualname__`` is
    dot-separated identifiers with optional ``<locals>`` segments (a class
    defined inside a function); a ``__module__`` is dot-separated identifiers.
    Requiring this grammar rejects, in one test, every smuggling vector the
    bare reads admitted: newlines and other ``str.splitlines`` boundaries
    (which forge a second log/report record), C0/C1 control characters, and
    arbitrary payload text (spaces, credential punctuation) -- none of which
    can appear in an identifier -- while accepting every genuine name verbatim.

    Args:
        value: The already-length-checked, exact-``str`` name.
        dotted: Whether ``.`` separators are allowed (``__qualname__`` /
            ``__module__``); ``False`` for a bare ``__name__``.

    Returns:
        ``True`` if every segment is an identifier (or the ``<locals>``
        marker), ``False`` otherwise.
    """
    segments = value.split(".") if dotted else [value]
    return all(segment == "<locals>" or segment.isidentifier() for segment in segments)


def _canonical_metadata_name(descriptor: Any, cls: type, *, dotted: bool) -> str | None:
    """Read one type-metadata name safely, or ``None`` if it is unrenderable.

    Reads through the interpreter's own getset ``descriptor`` (bypassing a
    metaclass dispatch), then accepts the result only when it is an EXACT
    ``str`` (a ``str`` subclass could carry a hostile ``__str__``), within
    :data:`_MAX_TYPE_NAME_CHARS`, and conforming to the identifier grammar
    (:func:`_conforms_to_name_grammar`). Any failure yields ``None`` -- the
    caller renders its own fixed placeholder -- so no engine-fabricated name
    can carry a line boundary, control character, or payload text into a
    diagnostic surface.

    Args:
        descriptor: The interpreter getset descriptor to read through.
        cls: The type whose name is read (``type(value)``, unspoofable).
        dotted: Whether the name may contain ``.`` separators.

    Returns:
        The validated name, or ``None`` when it cannot be safely rendered.
    """
    try:
        raw = descriptor.__get__(cls)
    except BaseException:  # noqa: BLE001 - the boundary must stay total  # pragma: no cover
        # Reading the interpreter's own getset slot on a real type does not
        # raise; kept as containment so the namer is total even if some
        # exotic type made even this read fail.
        return None
    if type(raw) is not str or not raw or len(raw) > _MAX_TYPE_NAME_CHARS:
        return None
    if not _conforms_to_name_grammar(raw, dotted=dotted):
        return None
    return raw


#: Appended when containing a stray coroutine ran the plugin's own cleanup
#: code and that code raised. Fixed text, never the cleanup exception's own
#: message: the primary diagnosis is what the caller must act on, and a
#: cleanup message is plugin-authored text this module does not vet.
_CLEANUP_FAILED = " (its cleanup also raised)"

#: The classification a verdict records, so a consumer picks its surface
#: (compliance issue code, CLI report) from the VERDICT instead of
#: re-analyzing the value (re-runs of untrusted introspection can raise,
#: disagree with the first classification, or crash the error path).
#:
#: * ``"awaitable"`` -- the value is an awaitable (modality defect).
#: * ``"wrong_type"`` -- the value is not an instance of the pinned type(s).
#: * ``"unclassifiable"`` -- the classification itself could not be carried
#:   out (hostile or broken type metadata made the introspection raise). A
#:   value no consumer can safely classify cannot satisfy the contract, so
#:   this is a defect, not a pass (fail-closed), reported with fixed text.
SyncDefectKind = Literal["awaitable", "wrong_type", "unclassifiable"]


@dataclasses.dataclass(frozen=True)
class SyncResultDefect:
    """The verdict of the sync-call boundary for one return value.

    The verdict is the ONLY thing a consumer may build its report from: the
    boundary already contained every read of the value's (untrusted) type
    metadata, and re-reading that metadata on a consumer's error path can
    raise where the boundary's contained classification succeeded. The
    clause is fixed vocabulary plus type NAMES only -- never ``repr(value)``
    -- so it embeds into any error surface unchanged; ``str()`` of the
    verdict IS the clause, keeping f-string callsites explicit without
    re-deriving text from the value.

    Attributes:
        kind: The classification (see :data:`SyncDefectKind`).
        clause: The human-readable defect clause (e.g. ``"returned an
            awaitable (async def?)"``), for embedding directly.
    """

    kind: SyncDefectKind
    clause: str

    def __str__(self) -> str:
        """Render the defect as its clause (the embedding contract).

        Returns:
            :attr:`clause`, unchanged.
        """
        return self.clause


def safe_type_name(value: object) -> str:
    r"""Name a value's type for an error message, never raising.

    THE shared namer for every consumer that reports a protocol violation
    (this module's clauses, the compliance suite's issue messages), so a
    second call site cannot re-derive type metadata with weaker rules than
    the boundary that already classified the value.

    ``type(value)`` cannot be spoofed, but ``__name__`` / ``__qualname__`` /
    ``__module__`` are attribute reads like any other, and this helper runs
    on the error path of a boundary whose whole job is to produce a STABLE
    verdict. A read that raises here would replace the contract violation
    being reported with an unrelated exception, so the fallback is a fixed
    literal.

    A type's ``__name__`` is NOT a trusted string: ``type("A\nB", (), {})``
    stores a newline in it with no metaclass at all, and a metaclass can hand
    back arbitrary text on every read. Unescaped, that text embedded straight
    into an :class:`~standard_asr.contract.exceptions.EngineContractError`
    message -- a surface that reaches CLI stderr, server logs, and compliance
    reports with no later escaping pass -- so an engine-fabricated object
    could forge a second log record or smuggle payload text through a channel
    documented as "type NAMES only". The read now goes through the
    interpreter's own getset descriptor (past metaclass dispatch) and the
    result must be an exact-``str`` identifier within a length bound
    (:func:`_canonical_metadata_name`); anything else becomes the fixed
    placeholder.

    Args:
        value: The value whose type is being named.

    Returns:
        The type's ``__name__`` when it is a safe identifier, else
        :data:`_UNNAMEABLE_TYPE`.
    """
    return safe_class_name(type(value))


def safe_class_name(cls: type) -> str:
    """Name a CLASS for an error message, never raising.

    The class-level companion to :func:`safe_type_name` (which names a
    value's type): consumers that already hold a type object -- the compliance
    suite naming a declared ``config_type`` / ``provider_params_type`` in an
    issue message -- name it through the SAME hardened reader instead of a
    bare ``cls.__name__``, so a plugin-declared class cannot smuggle a line
    boundary or payload text into a report either.

    Args:
        cls: The class to name.

    Returns:
        The class's ``__name__`` when it is a safe identifier, else
        :data:`_UNNAMEABLE_TYPE`.
    """
    return _canonical_metadata_name(_TYPE_NAME_DESCRIPTOR, cls, dotted=False) or _UNNAMEABLE_TYPE


def _qualified_type_name(value: object) -> str | None:
    """Name a value's type with its module, never raising.

    Reads BOTH the module and the qualified name through the interpreter's
    getset descriptors (past metaclass dispatch), validating each against the
    identifier grammar (:func:`_canonical_metadata_name`), so this
    disambiguation surface cannot itself become a smuggling channel -- a
    forged ``__module__`` / ``__qualname__`` carrying a line boundary or
    payload never reaches the message.

    Args:
        value: The value whose type is being named.

    Returns:
        ``module.qualname`` when both are safe names, or ``None`` when either
        is unrenderable -- the caller renders its own fallback, because
        falling back to the bare ``__name__`` HERE could re-emit the exact
        name collision the qualification exists to disambiguate ("returned
        bool, not bool").
    """
    cls = type(value)
    module = _canonical_metadata_name(_TYPE_MODULE_DESCRIPTOR, cls, dotted=True)
    qualname = _canonical_metadata_name(_TYPE_QUALNAME_DESCRIPTOR, cls, dotted=True)
    if module is None or qualname is None:
        return None
    return f"{module}.{qualname}"


def _close_coroutine(value: object) -> bool:
    """Close a stray coroutine, containing whatever its cleanup runs.

    Closing throws ``GeneratorExit`` into a SUSPENDED coroutine, which runs
    its ``finally`` blocks -- arbitrary plugin code, on this thread. That
    code can raise, and it did: an escaping cleanup error replaced the
    ``EngineContractError`` this boundary exists to produce, handing fault
    ownership back to whatever caught it next (and carrying the plugin's own
    text into surfaces the boundary deliberately keeps free of it).

    Containment, not avoidance: closing is still right (an unclosed
    coroutine surfaces as a never-awaited ``RuntimeWarning`` that pollutes
    every run under warnings-as-errors). The cleanup outcome is REPORTED
    rather than swallowed, so an engine author sees that their teardown
    failed too.

    Args:
        value: The coroutine to close.

    Returns:
        ``True`` when cleanup completed, ``False`` when it raised.
    """
    try:
        cast("Coroutine[Any, Any, Any]", value).close()
    except BaseException:  # noqa: BLE001 - a plugin's finally must not win
        return False
    return True


def _pinned_names(expected_type: type | tuple[type, ...]) -> str:
    """Render the pinned return type(s) for an error message.

    Args:
        expected_type: The pinned type, or a tuple of acceptable types.

    Returns:
        A ``" / "``-joined display string; ``NoneType`` renders as ``"None"``
        so a require-``None`` member (``prepare()``) reads naturally.
    """
    types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
    return " / ".join("None" if t is type(None) else t.__name__ for t in types)


def sync_result_defect(
    value: object,
    *,
    expected_type: type | tuple[type, ...] | None = None,
) -> SyncResultDefect | None:
    """Classify a synchronous call's return value; contain a stray coroutine.

    THE single detection half of the sync-call boundary (see the module
    docstring for the consumer map). Defect shapes classified:

    * **Awaitable**: the member was (or delegated to) an ``async def``. A bare
      coroutine is CLOSED before returning so nothing leaks as a
      never-awaited ``RuntimeWarning``.
    * **Wrong type** (when ``expected_type`` is given): strict ``isinstance``
      against the protocol-pinned return type(s) -- quacking is not
      compliance (a ``numpy.bool_`` is NOT a ``bool``).
    * **Unclassifiable**: the classification's own introspection raised. A
      value's type can fight inspection through its metaclass or a
      ``__class__`` property (a proxy with a broken one is an ORDINARY bug,
      not an adversary); the contained verdict then reports a fixed
      unclassifiable defect instead of letting the metadata read displace
      the boundary's verdict with a raw plugin exception.

    TOTAL: every read of the value's type metadata -- the awaitable probe,
    the coroutine probe, the ``isinstance`` check, the type name -- runs
    inside containment, and a failed read yields a verdict (fail-closed),
    never an escape. The classification is formed FIRST and containment
    (coroutine close) runs after it, so nothing the plugin does on the way
    out can displace the verdict either.

    Args:
        value: The call's return value.
        expected_type: The pinned return type(s), or ``None`` to check only
            synchronicity. ``type(None)`` is accepted (and rendered as
            ``"None"``) for members pinned to return nothing.

    Returns:
        A :class:`SyncResultDefect` (any bare coroutine CLOSED), or ``None``
        for a conforming value. The clause never embeds ``repr(value)``.
    """
    try:
        awaitable = inspect.isawaitable(value)
    except BaseException:  # noqa: BLE001 - hostile metadata; the verdict must survive
        return SyncResultDefect(
            "unclassifiable",
            "returned a value that could not be safely classified for synchronicity "
            "(inspection raised)",
        )
    if awaitable:
        # The diagnosis is decided before any cleanup runs.
        clause = "returned an awaitable (async def?)"
        try:
            closable = inspect.iscoroutine(value)
        except BaseException:  # noqa: BLE001 - unclosable metadata  # pragma: no cover
            # A REAL coroutine cannot fail this check (its type exposes no
            # ``__class__`` hook), so this arm needs an object that passes
            # the awaitable probe and breaks the coroutine one in between --
            # kept as containment rather than reached in tests.
            closable = False  # pragma: no cover
        if closable and not _close_coroutine(value):
            return SyncResultDefect("awaitable", clause + _CLEANUP_FAILED)
        return SyncResultDefect("awaitable", clause)
    if expected_type is not None:
        try:
            type_matches = isinstance(value, expected_type)
        except BaseException:  # noqa: BLE001 - hostile metadata; fail closed, never raise
            return SyncResultDefect(
                "unclassifiable",
                "returned a value whose type could not be safely checked (inspection raised)",
            )
        if not type_matches:
            pinned = _pinned_names(expected_type)
            actual = safe_type_name(value)
            if actual in pinned.split(" / "):
                # A bare-name collision (e.g. numpy 2.x's ``bool_`` displays
                # as "bool") would render the self-contradictory "returned
                # bool, not bool"; qualify the actual type so the clause
                # stays explicit. The qualification reads can fail too --
                # never fall back to the colliding bare name.
                actual = _qualified_type_name(value) or f"{actual} (module/qualname unreadable)"
            return SyncResultDefect("wrong_type", f"returned {actual}, not {pinned}")
    return None


def require_sync_result(
    value: object,
    member: str,
    *,
    expected_type: type | tuple[type, ...] | None = None,
) -> None:
    """Enforce the sync-call boundary at a real consumer call site.

    The raising wrapper over :func:`sync_result_defect` for consumers outside
    the compliance suite (the CLI's transcribe/prepare paths, the server's
    REST and WebSocket paths). A defect raises
    :class:`~standard_asr.contract.exceptions.EngineContractError` -- an
    engine/plugin fault by definition, deliberately NOT a ``ValueError``, so
    transports and the CLI route it onto their engine-fault surface instead
    of a caller-fixable one.

    Args:
        value: The member's return value (any bare coroutine is closed).
        member: Display name of the member (e.g. ``"transcribe()"``).
        expected_type: The member's pinned return type(s), or ``None`` to
            check only synchronicity.

    Raises:
        EngineContractError: If the value is an awaitable or (when
            ``expected_type`` is given) not an instance of the pinned type.
            The message names the member and type names only -- never the
            value itself.
    """
    defect = sync_result_defect(value, expected_type=expected_type)
    if defect is None:
        return
    expectation = f" returning {_pinned_names(expected_type)}" if expected_type is not None else ""
    raise EngineContractError(
        f"{member} {defect.clause}. The StandardASR protocol pins {member} as a "
        f"SYNCHRONOUS member{expectation} (async behavior lives in "
        "transcribe_async and inside the returned session). This is an "
        "engine/plugin bug, not a caller error -- report it to the engine's "
        "author."
    )
