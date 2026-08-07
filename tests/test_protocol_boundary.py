# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared runtime sync-call boundary.

``standard_asr.runtime.protocol_boundary`` is the single owner of the
detection rule for a synchronous protocol member returning an awaitable or a
wrong-typed value (the compliance suite adapts its verdicts into issue codes;
the CLI/server raise through it). These tests pin the detection semantics,
the coroutine containment, the raising adapter's fault classification, and
the no-value-repr message guarantee.
"""

from __future__ import annotations

import builtins
from typing import Any, Generator

import pytest

from standard_asr.contract.exceptions import EngineContractError, StandardASRError
from standard_asr.runtime.protocol_boundary import require_sync_result, sync_result_defect


async def _async_result() -> str:
    """Return a sentinel from an ``async def`` (never actually awaited).

    Returns:
        A sentinel string (unreachable in these tests).
    """
    return "never-awaited"  # pragma: no cover - the coroutine is never driven


class _NonCoroutineAwaitable:
    """An awaitable that is not a coroutine (e.g. a custom future type)."""

    def __await__(self) -> Generator[Any, None, str]:
        """Make the object awaitable.

        Yields:
            Nothing (the generator is never driven in these tests).

        Returns:
            A sentinel string (unreachable).
        """
        yield None  # pragma: no cover - never driven
        return "value"  # pragma: no cover - never driven


class _SecretRepr:
    """A value whose ``repr`` embeds payload text that must never leak."""

    def __repr__(self) -> str:
        """Return a payload-bearing representation.

        Returns:
            A string containing the sentinel secret.
        """
        return "SecretRepr(sk-SENTINEL-SECRET)"


def test_conforming_values_have_no_defect() -> None:
    """A plain value passes; with a pinned type, a real instance passes."""
    assert sync_result_defect("anything") is None
    assert sync_result_defect(True, expected_type=bool) is None
    assert sync_result_defect(None, expected_type=type(None)) is None
    assert sync_result_defect(1.5, expected_type=(int, float)) is None


def test_bare_coroutine_is_classified_and_closed() -> None:
    """A coroutine return is a defect, and the coroutine is CLOSED (no leak).

    Under warnings-as-errors a never-awaited coroutine would fail the suite at
    GC time; the boundary must close it before reporting. A closed coroutine's
    ``cr_frame`` is ``None``.
    """
    coro = _async_result()
    defect = sync_result_defect(coro)
    assert defect is not None
    assert defect.kind == "awaitable"
    assert defect.clause == "returned an awaitable (async def?)"
    assert str(defect) == defect.clause  # f-string embedding IS the clause
    assert coro.cr_frame is None  # closed: nothing leaks


def test_non_coroutine_awaitable_is_classified_without_close() -> None:
    """A non-coroutine awaitable is still the awaitable defect (no crash).

    Only bare coroutines can be closed; other awaitables are classified the
    same way without attempting containment.
    """
    defect = sync_result_defect(_NonCoroutineAwaitable())
    assert defect is not None and defect.clause == "returned an awaitable (async def?)"


def test_awaitable_wins_over_type_check_and_is_closed() -> None:
    """With a pinned type, an awaitable is the modality defect, not a type miss."""
    coro = _async_result()
    defect = sync_result_defect(coro, expected_type=bool)
    assert defect is not None and defect.clause == "returned an awaitable (async def?)"
    assert coro.cr_frame is None


def test_wrong_type_is_strict_isinstance() -> None:
    """The type pin is strict ``isinstance`` and names types, not values."""
    defect = sync_result_defect("false", expected_type=bool)
    assert defect is not None and defect.clause == "returned str, not bool"
    assert defect.kind == "wrong_type"
    # A tuple pin renders as a joined list; NoneType renders as "None".
    defect = sync_result_defect(object(), expected_type=(bool, type(None)))
    assert defect is not None and defect.clause == "returned object, not bool / None"
    # prepare()'s pin: any non-None return is a defect.
    defect = sync_result_defect("done", expected_type=type(None))
    assert defect is not None and defect.clause == "returned str, not None"


def test_bool_pin_rejects_non_bool_truthy_lookalikes() -> None:
    """Strictness pins: an ``int`` 1 and a ``numpy.bool_`` are NOT ``bool``.

    Both coerce truthy everywhere, which is exactly why quacking is not
    compliance -- the protocol pins the type. numpy 2.x displays ``bool_`` as
    bare "bool", so the clause module-qualifies it rather than emitting the
    self-contradictory "returned bool, not bool".
    """
    import numpy as np

    defect = sync_result_defect(1, expected_type=bool)
    assert defect is not None and defect.clause == "returned int, not bool"
    # numpy 1.x names the scalar type "bool_" (no collision); numpy 2.x names
    # it "bool", which the boundary must qualify -- either way the clause is a
    # defect that never reads as the self-contradictory "returned bool, not bool".
    defect = sync_result_defect(np.bool_(True), expected_type=bool)
    assert defect is not None
    assert defect.clause in ("returned bool_, not bool", "returned numpy.bool, not bool")


def test_bare_name_collision_is_module_qualified() -> None:
    """A foreign type whose bare name collides with the pin gets qualified.

    Deterministic (numpy-version-independent) pin of the collision branch: a
    class literally named ``bool`` must not produce "returned bool, not bool".
    """
    fake_bool = type("bool", (), {})
    defect = sync_result_defect(fake_bool(), expected_type=bool)
    assert defect is not None
    assert defect.clause == f"returned {fake_bool.__module__}.bool, not bool"


def test_defect_message_never_embeds_the_value_repr() -> None:
    """The defect clause carries type names only -- never ``repr(value)``."""
    defect = sync_result_defect(_SecretRepr(), expected_type=bool)
    assert defect is not None
    assert "SENTINEL-SECRET" not in defect.clause


def test_require_sync_result_passes_conforming_values() -> None:
    """The raising adapter is a no-op for a conforming value."""
    require_sync_result(True, "supports()", expected_type=bool)
    require_sync_result("anything", "transcribe()")


def test_require_sync_result_raises_engine_contract_error() -> None:
    """A defect raises ``EngineContractError`` naming member and pinned type."""
    with pytest.raises(EngineContractError) as excinfo:
        require_sync_result("done", "prepare()", expected_type=type(None))
    message = str(excinfo.value)
    assert "prepare()" in message
    assert "returned str, not None" in message
    assert "returning None" in message
    assert "engine/plugin bug" in message


def test_require_sync_result_closes_the_stray_coroutine() -> None:
    """The raising path contains the coroutine before raising."""
    coro = _async_result()
    with pytest.raises(EngineContractError) as excinfo:
        require_sync_result(coro, "transcribe()")
    assert coro.cr_frame is None
    # Without a pinned type the message still names the member and modality.
    assert "transcribe() returned an awaitable" in str(excinfo.value)


def test_require_sync_result_message_never_embeds_the_value_repr() -> None:
    """The raised message carries type names only -- never the value."""
    with pytest.raises(EngineContractError) as excinfo:
        require_sync_result(_SecretRepr(), "supports()", expected_type=bool)
    assert "SENTINEL-SECRET" not in str(excinfo.value)


def test_engine_contract_error_is_engine_fault_not_value_error() -> None:
    """``EngineContractError`` maps to engine-fault surfaces, never 422/exit 2.

    The ``ValueError`` family is routed to caller-fixable surfaces (HTTP 422,
    CLI usage exit 2); an engine breaking the protocol contract must never
    land there, so the exception deliberately does NOT mix in ``ValueError``.
    """
    assert issubclass(EngineContractError, StandardASRError)
    assert not issubclass(EngineContractError, ValueError)


async def _inner_await() -> None:
    """Suspend once so the caller can start and park a coroutine."""
    await _Suspend()


class _Suspend:
    """An awaitable that yields exactly once (parks the coroutine)."""

    def __await__(self) -> Generator[Any, None, None]:
        """Suspend the running coroutine.

        Yields:
            ``None``, once.
        """
        yield None


async def _cleanup_raises() -> None:
    """Suspend, then raise from ``finally`` when closed.

    Raises:
        RuntimeError: From the ``finally`` block, on close.
    """
    try:
        await _inner_await()
    finally:
        raise RuntimeError("cleanup failed: sk-CLEANUP-SENTINEL")


async def _cleanup_exits() -> None:
    """Suspend, then raise a ``BaseException`` from ``finally`` when closed.

    Raises:
        SystemExit: From the ``finally`` block, on close.
    """
    try:
        await _inner_await()
    finally:
        raise SystemExit("cleanup exit")


def _started(factory: Any) -> Any:
    """Start a coroutine and park it at its first suspension point.

    Args:
        factory: The coroutine function to start.

    Returns:
        The started, suspended coroutine.
    """
    coro = factory()
    coro.send(None)
    return coro


@pytest.mark.parametrize("factory", [_cleanup_raises, _cleanup_exits])
def test_plugin_cleanup_cannot_displace_the_boundary_verdict(factory: Any) -> None:
    """A suspended coroutine's ``finally`` runs on close -- and must not win.

    Closing throws ``GeneratorExit`` into a SUSPENDED coroutine, which runs
    the author's ``finally`` blocks on this thread. An escaping cleanup error
    replaced the contract verdict entirely: the caller saw the plugin's
    RuntimeError (or SystemExit) instead of the engine-fault classification
    this boundary exists to produce, and the plugin's own message rode along
    into surfaces the boundary keeps free of it.

    Args:
        factory: The coroutine function whose cleanup raises.
    """
    defect = sync_result_defect(_started(factory))
    assert defect is not None
    assert defect.clause.startswith("returned an awaitable")
    # The failure is REPORTED, not swallowed -- but with fixed text, never
    # the cleanup exception's own (unvetted, plugin-authored) message.
    assert "cleanup also raised" in defect.clause
    assert "sk-CLEANUP-SENTINEL" not in defect.clause

    with pytest.raises(EngineContractError) as excinfo:
        require_sync_result(_started(factory), "transcribe()")
    assert "cleanup also raised" in str(excinfo.value)
    assert "sk-CLEANUP-SENTINEL" not in str(excinfo.value)


def test_clean_cleanup_is_not_reported_as_a_failure() -> None:
    """A coroutine that closes cleanly yields the bare defect clause."""
    defect = sync_result_defect(_async_result())
    assert defect is not None and defect.clause == "returned an awaitable (async def?)"


def test_type_naming_bypasses_a_metaclass_name_hijack() -> None:
    """A metaclass cannot substitute the rendered name via attribute dispatch.

    ``type(value)`` cannot be spoofed, but a metaclass ``__getattribute__`` /
    ``property`` on ``__name__`` could hand back attacker-chosen text (or run
    author code) on every ``cls.__name__`` read. The name is read through the
    interpreter's own getset descriptor, which reads the stored C-level slot
    directly, so the hijack is BYPASSED and the real name is recovered.
    """

    class _HostileMeta(type):
        @property
        def __name__(cls) -> str:  # type: ignore[override]
            """Hand back forged text on every ``__name__`` read.

            Returns:
                Attacker-chosen text (which the boundary must not use).
            """
            return "sk-FORGED\nSECOND-LINE"

    class _Real(metaclass=_HostileMeta):
        pass

    # The forged, newline-carrying text never appears; the real name does.
    defect = sync_result_defect(_Real(), expected_type=bool)
    assert defect is not None and defect.clause == "returned _Real, not bool"

    with pytest.raises(EngineContractError):
        require_sync_result(_Real(), "supports()", expected_type=bool)


def test_type_name_with_a_newline_becomes_the_placeholder() -> None:
    """A stored name that is not an identifier renders as the placeholder.

    ``type("A\\nB", (), {})`` stores a newline in ``__name__`` with no
    metaclass at all -- and that text embedded straight into an
    ``EngineContractError`` message, a surface with no later escaping pass,
    forging a second log/report record. A name that fails the identifier
    grammar is refused (fixed placeholder), so no line boundary or payload
    can reach the message.
    """
    forged = type("sk-STANDARD-ASR-REVIEW-F4\nforged-line", (), {})
    defect = sync_result_defect(forged(), expected_type=bool)
    assert defect is not None
    assert defect.clause == "returned <unnameable type>, not bool"
    assert "\n" not in defect.clause
    assert "sk-STANDARD-ASR-REVIEW-F4" not in defect.clause

    with pytest.raises(EngineContractError) as excinfo:
        require_sync_result(forged(), "transcribe()", expected_type=bool)
    message = str(excinfo.value)
    assert "sk-STANDARD-ASR-REVIEW-F4" not in message
    # No forged record: the whole message is one splitlines segment plus the
    # deliberate paragraph the boundary itself writes -- never one the name
    # smuggled in. (The value-name portion carries no newline at all.)
    assert "\nforged-line" not in message


def test_overlong_type_name_is_refused() -> None:
    """A pathologically long name is refused, never truncated into a fragment."""
    huge = type("A" * 5000, (), {})
    defect = sync_result_defect(huge(), expected_type=bool)
    assert defect is not None
    assert defect.clause == "returned <unnameable type>, not bool"


def test_str_subclass_name_is_refused() -> None:
    """An exact ``str`` is required: a ``str`` subclass could carry a hostile display.

    Python accepts a ``str`` subclass as ``__name__`` -- and such a subclass
    could override ``__str__`` to smuggle text when the name is later
    formatted. The canonicalizer requires ``type(raw) is str`` exactly, so a
    subclass name renders as the fixed placeholder.
    """
    from standard_asr.runtime.protocol_boundary import safe_class_name

    class _EvilStr(str):
        def __str__(self) -> str:  # pragma: no cover - must never be reached
            """Return forged text if this subclass is ever coerced.

            Returns:
                Attacker-chosen text.
            """
            return "sk-EVIL"

    forged = type("legit", (), {})
    forged.__name__ = _EvilStr("legit")  # type: ignore[assignment]
    assert safe_class_name(forged) == "<unnameable type>"


def test_bare_name_collision_qualification_bypasses_a_module_hijack() -> None:
    """A colliding name is qualified with its REAL module, past a metaclass hijack.

    A metaclass ``__getattribute__`` that forges ``__module__`` / ``__qualname__``
    cannot make the qualification emit attacker text: both are read through the
    interpreter's getset descriptors, so the real module qualifies the colliding
    name and the clause never becomes the self-contradictory "returned bool, not bool".
    """

    class _PartialMeta(type):
        def __getattribute__(cls, name: str) -> object:
            """Forge the reads used to qualify a colliding name.

            Args:
                name: The attribute being read.

            Returns:
                Forged text for ``__module__``/``__qualname__``, else the
                real attribute.
            """
            if name in ("__module__", "__qualname__"):
                return "sk-FORGED\nSECOND"
            return super().__getattribute__(name)

    colliding = _PartialMeta("bool", (), {})
    defect = sync_result_defect(colliding(), expected_type=builtins.bool)
    assert defect is not None
    # Qualified with the REAL module (this test module), not the forged text.
    assert defect.clause == f"returned {__name__}.bool, not bool"
    assert "sk-FORGED" not in defect.clause
    assert "returned bool, not bool" != defect.clause


def test_bare_name_collision_falls_back_when_module_is_unnameable() -> None:
    """When the module/qualname is genuinely malformed, the fixed note is used.

    A class whose stored ``__module__`` is not a dotted identifier (a newline,
    a space) cannot be qualified, so the clause carries the fixed
    "module/qualname unreadable" note rather than re-emitting the colliding
    bare name ("returned bool, not bool").
    """
    colliding = type("bool", (), {"__module__": "bad module\nforged"})
    defect = sync_result_defect(colliding(), expected_type=builtins.bool)
    assert defect is not None
    assert defect.clause == "returned bool (module/qualname unreadable), not bool"
    assert "\n" not in defect.clause
    assert "returned bool, not bool" != defect.clause


def test_hostile_type_metadata_cannot_displace_the_verdict() -> None:
    """Classification is TOTAL: introspection raising yields a verdict.

    The round-11 counterexample: a metaclass whose ``__mro__`` read raises
    made ``inspect.isawaitable`` itself raise OUTSIDE the boundary's
    containment, so the raw plugin exception escaped EVERY consumer (CLI,
    REST/WS, compliance, sync bridge) in place of the stable contract
    verdict -- and carried plugin-authored text across the boundary.
    """

    class _HostileMroMeta(type):
        @property
        def __mro__(cls) -> tuple[type, ...]:
            """Fail the MRO read ABC introspection performs.

            Returns:
                Never returns.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("hostile mro read")

    class _Hostile(metaclass=_HostileMroMeta):
        pass

    value = _Hostile()
    defect = sync_result_defect(value)
    assert defect is not None
    assert defect.kind == "unclassifiable"
    assert "could not be safely classified" in defect.clause

    defect_typed = sync_result_defect(value, expected_type=bool)
    assert defect_typed is not None and defect_typed.kind == "unclassifiable"

    # The raising adapter classifies it as an engine fault, never lets the
    # plugin's own RuntimeError escape.
    with pytest.raises(EngineContractError) as excinfo:
        require_sync_result(value, "transcribe()")
    assert "could not be safely classified" in str(excinfo.value)


def test_hostile_dunder_class_property_cannot_displace_the_verdict() -> None:
    """A broken ``__class__`` property is an ORDINARY bug, not an adversary.

    A proxy class whose ``__class__`` property raises is a plausible
    implementation mistake; ABC-based introspection (``isawaitable``
    included) reads it, so the boundary must contain the read.
    """

    class _BrokenProxy:
        @property
        def __class__(self) -> type:  # pyright: ignore[reportIncompatibleMethodOverride]
            """Raise instead of answering.

            Returns:
                Never returns.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("broken proxy")

    defect = sync_result_defect(_BrokenProxy())
    assert defect is not None
    assert defect.kind == "unclassifiable"
    assert "could not be safely classified" in defect.clause


def test_hostile_pinned_type_cannot_displace_the_verdict() -> None:
    """The isinstance half is contained too -- even against a hostile ABC.

    Callers pin ordinary protocol types, but the classification rule must
    not RETREAT to raising for exotic pins: ``isinstance`` against a class
    whose metaclass's ``__instancecheck__`` raises must yield the
    unclassifiable verdict, not an escape.
    """

    class _HostileInstancecheckMeta(type):
        def __instancecheck__(cls, instance: object) -> bool:
            """Raise instead of answering.

            Args:
                instance: The object being checked.

            Returns:
                Never returns.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("hostile instancecheck")

    class _HostilePin(metaclass=_HostileInstancecheckMeta):
        pass

    defect = sync_result_defect(object(), expected_type=_HostilePin)
    assert defect is not None
    assert defect.kind == "unclassifiable"
    assert "type could not be safely checked" in defect.clause

    with pytest.raises(EngineContractError) as excinfo:
        require_sync_result(object(), "supports()", expected_type=_HostilePin)
    assert "type could not be safely checked" in str(excinfo.value)
