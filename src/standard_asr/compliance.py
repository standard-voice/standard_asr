# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compliance helpers for Standard ASR plugin authors."""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

from pydantic import BaseModel

from .asr_config import BaseConfig
from .asr_interface import EngineBase
from .asr_properties import BaseProperties
from .capabilities import DeclaredCapabilities
from .discovery import FactoryLoadError, ModelRegistry, discover_models
from .exceptions import UnsupportedFeatureError
from .runtime_params import ProviderParams, RuntimeParams, WordTimestampGranularity
from .streaming import (
    SyncSession,
    TranscriptionEvent,
    TranscriptionSession,
    _LifecycleGuard,  # pyright: ignore[reportPrivateUsage]
)

__all__ = [
    "ComplianceIssue",
    "ComplianceReport",
    "check_entrypoints",
    "check_event_sequence",
    "check_streaming_param_gating",
    "check_sync_bridge",
]

#: Candidate (param-field, params-builder, capability-suffix) probes for an
#: unsupported standard streaming parameter. The check picks the first whose
#: capability the engine does NOT support, so it always exercises the gating
#: drop/raise path. The builder returns a fully-typed :class:`RuntimeParams`.
_GATING_PROBES: tuple[tuple[str, Callable[[], RuntimeParams], str], ...] = (
    (
        "word_timestamps",
        lambda: RuntimeParams(word_timestamps=WordTimestampGranularity.WORD),
        "streaming.word_timestamps",
    ),
    (
        "prompt",
        lambda: RuntimeParams(prompt="the quick brown fox"),
        "streaming.guidance.prompt",
    ),
)


def _is_closed_model(model: type[BaseModel]) -> bool:
    """Return ``True`` if *model* forbids extra fields (``extra="forbid"``).

    Args:
        model: A pydantic model type.

    Returns:
        ``True`` when the model is a closed type.
    """
    return model.model_config.get("extra") == "forbid"


@dataclass(frozen=True, slots=True)
class ComplianceIssue:
    """Single compliance issue detected during validation.

    Args:
        level: Issue severity (error or warning).
        message: Human-readable message.
        model: Optional model identifier.

    Returns:
        None.

    Raises:
        None.
    """

    level: Literal["error", "warning"]
    message: str
    model: str | None = None


@dataclass(frozen=True, slots=True)
class ComplianceReport:
    """Aggregate result returned by :func:`check_entrypoints`.

    Args:
        registry: Model registry used for the check.
        issues: Collected compliance issues.

    Returns:
        None.

    Raises:
        None.
    """

    registry: ModelRegistry
    issues: list[ComplianceIssue]

    @property
    def passed(self) -> bool:
        """Return ``True`` when no errors were encountered.

        Args:
            None.

        Returns:
            ``True`` when no error-level issues exist.

        Raises:
            None.
        """

        return not any(issue.level == "error" for issue in self.issues)

    def iter_level(self, level: Literal["error", "warning"]) -> Iterable[ComplianceIssue]:
        """Yield issues matching *level*.

        Args:
            level: Severity level to filter.

        Returns:
            Iterable of matching issues.

        Raises:
            None.
        """

        for issue in self.issues:
            if issue.level == level:
                yield issue


def _can_call_without_args(factory: object) -> bool:
    """Return ``True`` if *factory* can be invoked without arguments.

    Args:
        factory: Entry point callable.

    Returns:
        ``True`` when the callable has no required parameters.

    Raises:
        None.
    """

    try:
        signature = inspect.signature(factory)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if (
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and parameter.default is inspect.Signature.empty
        ):
            return False
    return True


def check_entrypoints(
    registry: ModelRegistry | None = None,
    *,
    strict_discovery: bool = False,
    instantiate: bool = True,
) -> ComplianceReport:
    """Validate that discovered entry points conform to expectations.

    Args:
        registry: Optional pre-discovered registry.
        strict_discovery: Fail on invalid entry points when discovering.
        instantiate: If ``True``, instantiate zero-arg factories and verify metadata.

    Returns:
        Compliance report summarizing findings.

    Raises:
        None.
    """

    if registry is None:
        registry = discover_models(strict=strict_discovery)

    issues: list[ComplianceIssue] = []

    if len(registry) == 0:
        issues.append(
            ComplianceIssue(
                level="error",
                message="No standard_asr.models entry points were discovered.",
                model=None,
            )
        )
        return ComplianceReport(registry=registry, issues=issues)

    # §R.4 R1: RuntimeParams MUST be a closed type. This is a global invariant of
    # the standard, not per-engine, but surface it once so authors see it.
    if not _is_closed_model(RuntimeParams):
        issues.append(
            ComplianceIssue(
                level="error",
                message="RuntimeParams is not a closed type (extra='forbid').",
                model=None,
            )
        )

    for name in registry.names():
        spec = registry.spec(name)
        try:
            factory = spec.load_factory()
        except FactoryLoadError as exc:
            issues.append(ComplianceIssue(level="error", message=str(exc), model=name))
            continue

        # §3.1 / §C: declared metadata MUST be readable from the class without
        # instantiation. Resolve the class and read its ClassVars directly.
        _check_class_level_metadata(spec, name, issues)

        if not instantiate:
            continue

        if not _can_call_without_args(factory):
            issues.append(
                ComplianceIssue(
                    level="warning",
                    message=(
                        "Factory cannot be invoked without arguments; skipped instantiation check."
                    ),
                    model=name,
                )
            )
            continue

        try:
            instance = factory()
        except Exception as exc:  # noqa: BLE001
            issues.append(
                ComplianceIssue(
                    level="error",
                    message=f"Factory invocation failed with {exc!r}.",
                    model=name,
                )
            )
            continue

        if not hasattr(instance, "transcribe") or not callable(getattr(instance, "transcribe")):
            issues.append(
                ComplianceIssue(
                    level="error",
                    message=(
                        "Factory did not return an object with a callable 'transcribe' attribute."
                    ),
                    model=name,
                )
            )
            continue

        properties = getattr(instance, "properties", None)
        if not isinstance(properties, BaseProperties):
            issues.append(
                ComplianceIssue(
                    level="error",
                    message=(
                        "Instance is missing a BaseProperties-compatible 'properties' attribute."
                    ),
                    model=name,
                )
            )
        else:
            if properties.model_id != spec.key:
                issues.append(
                    ComplianceIssue(
                        level="error",
                        message=(
                            "Instance properties.model_id does not match the entry point key "
                            f"({properties.model_id!r} != {spec.key!r})."
                        ),
                        model=name,
                    )
                )

        config = getattr(instance, "config", None)
        if not isinstance(config, BaseConfig):
            issues.append(
                ComplianceIssue(
                    level="error",
                    message="Instance is missing a BaseConfig-compatible 'config' attribute.",
                    model=name,
                )
            )

        declared = getattr(instance, "declared_capabilities", None)
        if not isinstance(declared, DeclaredCapabilities):
            issues.append(
                ComplianceIssue(
                    level="error",
                    message=(
                        "Instance is missing a DeclaredCapabilities "
                        "'declared_capabilities' attribute."
                    ),
                    model=name,
                )
            )
        else:
            try:
                effective = getattr(instance, "effective_capabilities", None)
            except Exception as exc:  # noqa: BLE001
                # A buggy ``effective_capabilities`` property MUST NOT crash the
                # whole compliance run (this function promises ``Raises: None``);
                # report the offender and keep checking the other engines.
                issues.append(
                    ComplianceIssue(
                        level="error",
                        message=(
                            f"Reading effective_capabilities raised {exc!r}; the "
                            "property MUST return a DeclaredCapabilities (or None) "
                            "without raising."
                        ),
                        model=name,
                    )
                )
            else:
                if isinstance(effective, DeclaredCapabilities) and not declared.covers(effective):
                    issues.append(
                        ComplianceIssue(
                            level="error",
                            message=(
                                "effective_capabilities is not a subset of "
                                "declared_capabilities (effective MUST only narrow)."
                            ),
                            model=name,
                        )
                    )

    return ComplianceReport(registry=registry, issues=issues)


def _check_class_level_metadata(spec: object, name: str, issues: list[ComplianceIssue]) -> None:
    """Verify class-level metadata is readable without instantiation (§3.1/§C).

    Reads ``declared_capabilities`` and ``provider_params_type`` from the engine
    *class* (never the instance) and validates that, when present, the
    provider-params type is a closed :class:`ProviderParams` subclass (§R.4 R1).

    Args:
        spec: The :class:`~standard_asr.discovery.ModelSpec`.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    engine_class_getter = getattr(spec, "engine_class", None)
    if not callable(engine_class_getter):  # pragma: no cover - defensive
        return
    try:
        engine_class = engine_class_getter()
    except FactoryLoadError as exc:
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    "declared_capabilities/properties are not readable without "
                    f"instantiation: {exc}"
                ),
                model=name,
            )
        )
        return

    declared = inspect.getattr_static(engine_class, "declared_capabilities", None)
    if not isinstance(declared, DeclaredCapabilities):
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    "Engine class does not expose a class-level "
                    "'declared_capabilities' (ClassVar) readable without "
                    "instantiation."
                ),
                model=name,
            )
        )

    properties = inspect.getattr_static(engine_class, "properties", None)
    if not isinstance(properties, BaseProperties):
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    "Engine class does not expose a class-level 'properties' "
                    "(ClassVar) readable without instantiation."
                ),
                model=name,
            )
        )

    params_type = inspect.getattr_static(engine_class, "provider_params_type", None)
    if params_type is None:
        return
    if not (isinstance(params_type, type) and issubclass(params_type, ProviderParams)):
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    "provider_params_type is set but is not a ProviderParams "
                    f"subclass (got {params_type!r})."
                ),
                model=name,
            )
        )
    elif not _is_closed_model(params_type):
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    "provider_params_type must be a closed type (extra='forbid'); "
                    f"{params_type.__name__} is not."
                ),
                model=name,
            )
        )


def check_event_sequence(events: Iterable[TranscriptionEvent]) -> ComplianceReport:
    """Validate a *recorded* streaming event sequence against the invariants.

    Behavioral check for streaming adapters that is **pure**: it replays an
    already-captured event stream through the standard lifecycle/frontier guard
    and reports every invariant it violates, without ever instantiating or
    calling an engine. (Behavioral checks that would require *running* a model --
    strict sample-rate, the input-conversion matrix, language membership -- are
    deliberately left to unit tests, because invoking a cloud engine from a
    compliance run would be a billable side effect; this one only inspects data
    the author already produced.)

    Detected violations (each an error): an illegal lifecycle transition
    (``partial``/``final`` after a segment is finalized/superseded; a non-closed
    ``final`` after ``final``; superseding a ``closed`` segment), a non-monotonic
    ``stable_until`` or ``audio_processed_until``, a rewritten frozen prefix, the
    full ``supersede`` invariants of spec ST.5.2 -- frozen-prefix preservation
    across a replacement (the concatenated frozen text of ``old_ids`` MUST be
    preserved by ``new_ids``), ``old_ids`` that were never announced, a
    ``new_id`` that reintroduces an already-known segment, and an empty
    ``new_ids`` (pure deletion) that would destroy frozen text -- and an event
    stream that never reaches a terminal (``done`` / non-recoverable ``error``)
    event.

    These checks are obtained by replaying the events through the same
    :class:`~standard_asr.streaming._LifecycleGuard` the runtime uses, so the
    compliance verdict cannot drift from the runtime's enforcement.

    Args:
        events: The recorded events to validate, in emission order.

    Returns:
        A :class:`ComplianceReport`; ``passed`` is ``True`` when the sequence
        honours every streaming invariant.
    """
    guard = _LifecycleGuard(strict=False)
    issues: list[ComplianceIssue] = []
    saw_any = False
    saw_terminal = False
    for event in events:
        saw_any = True
        guard.admit(event)
        if event.is_terminal:
            saw_terminal = True
    for diagnostic in guard.diagnostics:
        issues.append(
            ComplianceIssue(
                level="error",
                message=(f"streaming invariant violated ({diagnostic.code}): {diagnostic.message}"),
                model=None,
            )
        )
    if saw_any and not saw_terminal:
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    "event stream ended without a terminal (done / non-recoverable "
                    "error) event (spec ST.6.1: the stream MUST terminate)."
                ),
                model=None,
            )
        )
    return ComplianceReport(registry=ModelRegistry({}), issues=issues)


def check_streaming_param_gating(engine: EngineBase) -> ComplianceReport:
    """Assert a streaming engine gates an unsupported standard parameter.

    Closes the RUNT-3 gap as a *compliance* failure rather than a silent one:
    the base :meth:`~standard_asr.asr_interface.EngineBase.start_transcription`
    template runs ``gate_params(mode="streaming")`` for every engine, so a
    "forgot to gate" engine (one that bypassed the template) must show up here.

    The check opens a streaming session with the first standard parameter the
    engine does **not** support in ``streaming`` mode and asserts the standard
    contract:

    * **strict** policy -- the call MUST raise
      :class:`~standard_asr.exceptions.UnsupportedFeatureError`;
    * **best_effort** policy -- the call MUST succeed, drop the parameter, and
      surface an ``unsupported_parameter_ignored`` diagnostic via
      ``session.diagnostics()``.

    An engine that declared streaming support (``streaming_input`` or
    ``streaming_output``) yet accepts the unsupported parameter -- the "forgot
    to gate" engine that bypassed the base template -- is a compliance
    **failure** here, so the gap is loud rather than silent.

    Engines that declare no streaming support, or that support every probed
    parameter, yield a clean (no-op) pass -- there is nothing to gate.

    Args:
        engine: The engine instance to exercise. Its ``config.strict`` selects
            which branch (strict raise / best_effort drop) is asserted.

    Returns:
        A :class:`ComplianceReport`; ``passed`` is ``True`` when the engine gated
        the unsupported parameter per its policy (or had nothing to gate).

    Raises:
        None.
    """
    issues: list[ComplianceIssue] = []
    model = engine.properties.engine_id

    if not (engine.supports("streaming_input") or engine.supports("streaming_output")):
        # The engine does not declare streaming support; there is no streaming
        # gating contract to exercise.
        return ComplianceReport(registry=ModelRegistry({}), issues=issues)

    probe = next(
        (p for p in _GATING_PROBES if not engine.supports(p[2])),
        None,
    )
    if probe is None:
        # The engine supports every probed standard parameter, so there is no
        # unsupported-parameter path to exercise here.
        return ComplianceReport(registry=ModelRegistry({}), issues=issues)

    field_name, build_params, _cap = probe
    params = build_params()
    strict = bool(getattr(engine.config, "strict", True))

    try:
        session = engine.start_transcription(params=params)
    except UnsupportedFeatureError:
        if not strict:
            issues.append(
                ComplianceIssue(
                    level="error",
                    message=(
                        f"best_effort engine raised UnsupportedFeatureError for an "
                        f"unsupported streaming parameter {field_name!r}; it MUST drop "
                        "it and emit a diagnostic instead (spec Runtime R2)."
                    ),
                    model=model,
                )
            )
        return ComplianceReport(registry=ModelRegistry({}), issues=issues)

    # The session was created: only valid under best_effort, and only if the
    # parameter was dropped + diagnosed.
    if strict:
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    f"strict engine accepted an unsupported streaming parameter "
                    f"{field_name!r} without raising; it MUST raise "
                    "UnsupportedFeatureError (spec Runtime R2 / RUNT-3 gating gap)."
                ),
                model=model,
            )
        )
    elif not any(d.code == "unsupported_parameter_ignored" for d in session.diagnostics()):
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    f"best_effort engine silently swallowed unsupported streaming "
                    f"parameter {field_name!r}: no 'unsupported_parameter_ignored' "
                    "diagnostic surfaced via session.diagnostics() (spec Runtime R2)."
                ),
                model=model,
            )
        )
    return ComplianceReport(registry=ModelRegistry({}), issues=issues)


def check_sync_bridge(
    session_factory: Callable[[], TranscriptionSession],
    *,
    timeout: float = 5.0,
) -> ComplianceReport:
    """Drive an async adapter's :class:`SyncSession` from an external thread.

    Implements the spec ST.6.5 mandate: a sync-bridge no-deadlock / no-leak
    test. A fresh session is created and driven synchronously from a *different*
    thread than the one that built it, feeding no audio and immediately ending
    input. The test asserts the session terminates (emits a terminal event and
    tears down) within ``timeout`` -- a deadlock or a leaked background loop/
    thread shows up as a timeout.

    Args:
        session_factory: A zero-argument callable returning a fresh async
            :class:`TranscriptionSession` (e.g. ``engine.start_transcription``
            bound with its arguments).
        timeout: Seconds to allow the bridged session to drain and close.

    Returns:
        A :class:`ComplianceReport`. ``passed`` is ``True`` when the bridge
        terminated cleanly with no surviving worker thread.

    Raises:
        None.
    """
    issues: list[ComplianceIssue] = []
    outcome: dict[str, object] = {}
    before = {t.ident for t in threading.enumerate()}

    def _drive() -> None:
        try:
            with SyncSession(session_factory()) as sync:
                sync.end_audio()
                events = list(sync)
            outcome["terminal"] = any(getattr(ev, "is_terminal", False) for ev in events)
        except Exception as exc:  # noqa: BLE001 - reported as a compliance error
            outcome["error"] = repr(exc)

    worker = threading.Thread(target=_drive, name="compliance-sync-bridge")
    worker.start()
    worker.join(timeout=timeout)

    if worker.is_alive():
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    f"SyncSession did not terminate within {timeout}s (deadlock). "
                    "Check the §6.5 adapter contract: bind loop resources in "
                    "__aenter__, never touch the ambient event loop."
                ),
                model=None,
            )
        )
        return ComplianceReport(registry=ModelRegistry({}), issues=issues)

    if "error" in outcome:
        issues.append(
            ComplianceIssue(
                level="error",
                message=f"SyncSession raised while bridging: {outcome['error']}.",
                model=None,
            )
        )
    elif not outcome.get("terminal"):
        # A well-formed session always lands a terminal event (the base producer
        # force-appends ``done``); reaching here means a non-compliant adapter
        # bypassed the base class and closed without one. This is exactly what the
        # compliance check exists to catch.
        issues.append(
            ComplianceIssue(
                level="error",
                message="SyncSession ended without emitting a terminal event.",
                model=None,
            )
        )

    # Leak check: the bridge owns a background loop thread torn down on __exit__.
    leaked = {
        t.name
        for t in threading.enumerate()
        if t.ident not in before and t.is_alive() and t.name != "compliance-sync-bridge"
    }
    if leaked:
        issues.append(
            ComplianceIssue(
                level="error",
                message=f"SyncSession leaked background thread(s): {sorted(leaked)}.",
                model=None,
            )
        )

    return ComplianceReport(registry=ModelRegistry({}), issues=issues)
