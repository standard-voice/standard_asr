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
from .capabilities import DeclaredCapabilities, PromptCap, WordTimestampsCap
from .discovery import FactoryLoadError, ModelRegistry, ModelSpec, discover_models
from .exceptions import ConfigError, UnsupportedFeatureError
from .param_gating import (
    DIAG_PROMPT_TRUNCATED,
    DIAG_UNSUPPORTED_GRANULARITY_IGNORED,
    DIAG_UNSUPPORTED_PARAMETER_IGNORED,
    _count_tokens,  # pyright: ignore[reportPrivateUsage]
)
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
#: drop/raise path; when every probe is supported it falls back to a
#: sub-constraint probe (:func:`_pick_sub_constraint_probe`). The builder
#: returns a fully-typed :class:`RuntimeParams`.
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

#: Upper bound (in approximate tokens) on the synthesized over-budget prompt
#: probe. The gating contract is violated by ANY prompt over the declared
#: budget, so a compliant probe never needs to be longer than the smallest
#: violating prompt -- and a legal-but-extreme ``max_tokens`` declaration
#: (``PromptConstraints.max_tokens`` has no upper bound; an LLM-backed engine
#: may advertise a 10^9-token context) MUST NOT make the compliance suite
#: allocate gigabytes and OOM the run it exists to keep alive. Budgets at or
#: above the cap skip the prompt probe (the granularity probe may still apply).
_SUB_CONSTRAINT_PROBE_MAX_TOKENS = 4096


def _pick_sub_constraint_probe(engine: EngineBase) -> tuple[str, RuntimeParams, str] | None:
    """Build a probe violating a declared sub-constraint of a supported feature.

    Used when the engine supports every probe in :data:`_GATING_PROBES` at the
    feature level: gating MUST also enforce a supported feature's declared
    *sub-constraints* (spec Runtime R2 -- a prompt over the declared
    ``max_tokens`` budget, a word-timestamp granularity not in the declared
    ``granularities``), so the check falls back to violating one of those. The
    best_effort contract differs per constraint (an over-budget prompt is
    truncated with ``prompt_truncated``; an unoffered granularity is dropped
    with ``unsupported_granularity_ignored``), so each probe carries the
    diagnostic code (imported from the gating layer, the single source of
    truth) it must surface.

    The prompt probe is bounded by
    :data:`_SUB_CONSTRAINT_PROBE_MAX_TOKENS`: a declared budget at or above
    the cap falls through to the granularity probe instead of materializing an
    arbitrarily large string.

    Args:
        engine: The engine under test.

    Returns:
        A ``(field_name, params, expected_diagnostic_code)`` triple, or ``None``
        when the engine declares no violable sub-constraint.
    """
    capabilities = engine.effective_capabilities
    prompt = capabilities.node_at("streaming.guidance.prompt")
    if isinstance(prompt, PromptCap) and prompt.is_supported:
        max_tokens = prompt.constraints.max_tokens
        if max_tokens is not None and max_tokens < _SUB_CONSTRAINT_PROBE_MAX_TOKENS:
            # Gating's own _count_tokens is the reference: one whitespace word
            # costs one token, so max_tokens + 1 words is over budget by
            # construction. The explicit check binds the probe to the helper
            # rather than to this comment, so a future counting refinement
            # cannot silently turn the probe into an in-budget prompt that
            # exercises nothing.
            over_budget = " ".join(["token"] * (max_tokens + 1))
            if _count_tokens(over_budget) > max_tokens:  # pragma: no branch
                return "prompt", RuntimeParams(prompt=over_budget), DIAG_PROMPT_TRUNCATED
    timestamps = capabilities.node_at("streaming.word_timestamps")
    if isinstance(timestamps, WordTimestampsCap) and timestamps.is_supported:
        offered = set(timestamps.granularities)
        missing = next((g for g in WordTimestampGranularity if g.value not in offered), None)
        if missing is not None:
            return (
                "word_timestamps",
                RuntimeParams(word_timestamps=missing),
                DIAG_UNSUPPORTED_GRANULARITY_IGNORED,
            )
    return None


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

    # §R.4 R1: RuntimeParams MUST be a closed type. This is a global invariant of
    # the standard, not per-engine, so it MUST be verified even in a bare
    # environment with no plugins installed -- it runs *before* the empty-registry
    # early return so the global invariant is never silently unchecked.
    if not _is_closed_model(RuntimeParams):
        issues.append(
            ComplianceIssue(
                level="error",
                message="RuntimeParams is not a closed type (extra='forbid').",
                model=None,
            )
        )

    if len(registry) == 0:
        issues.append(
            ComplianceIssue(
                level="error",
                message="No standard_asr.models entry points were discovered.",
                model=None,
            )
        )
        return ComplianceReport(registry=registry, issues=issues)

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

        _check_required_surface(instance, spec, name, issues)

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
        else:
            declared_config_type = inspect.getattr_static(type(instance), "config_type", None)
            if (
                isinstance(declared_config_type, type)
                and issubclass(declared_config_type, BaseConfig)
                and not isinstance(config, declared_config_type)
            ):
                issues.append(
                    ComplianceIssue(
                        level="error",
                        message=(
                            "Instance config is not an instance of the declared "
                            f"config_type ({config.__class__.__name__!r} is not a "
                            f"{declared_config_type.__name__!r}); the schema published "
                            "for UIs would not match the config actually consumed."
                        ),
                        model=name,
                    )
                )
            _check_language_axis_config(instance, name, issues)

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
                if isinstance(effective, DeclaredCapabilities):
                    if not declared.covers(effective):
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
                elif effective is not None:
                    # A non-None, wrong-typed ``effective`` is itself a violation:
                    # the effective ⊆ declared invariant MUST NOT be evadable by
                    # returning the wrong type (which would silently skip the
                    # subset check). ``None`` (engine declares no narrowing) stays
                    # a legitimate no-op.
                    issues.append(
                        ComplianceIssue(
                            level="error",
                            message=(
                                "effective_capabilities is not a DeclaredCapabilities "
                                f"(got {type(effective).__name__!r}); it MUST be a "
                                "DeclaredCapabilities (or None) so the effective ⊆ "
                                "declared invariant can be verified."
                            ),
                            model=name,
                        )
                    )

    return ComplianceReport(registry=registry, issues=issues)


def _check_language_axis_config(
    instance: object,
    name: str,
    issues: list[ComplianceIssue],
) -> None:
    """Verify a language-axis engine is constructed with a usable default language.

    An engine whose properties expose a language axis but whose config lacks a
    valid ``default_language`` passes construction (IC.9 keeps ``__init__``
    pure) and then raises ``ConfigError`` on the **user's first transcribe** --
    the worst place for an engine-author bug to surface. Catch it at compliance
    time instead. For :class:`EngineBase` engines this reuses the exact runtime
    validation (presence, selectable-membership, canonicalization), so the
    compliance verdict cannot drift from runtime behavior; for structural
    engines it falls back to the presence check (spec IC.6 / LANG R1).

    Args:
        instance: The instantiated engine to inspect.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    if isinstance(instance, EngineBase):
        try:
            instance._validate_language_config()  # pyright: ignore[reportPrivateUsage]
        except (ConfigError, ValueError) as exc:
            issues.append(
                ComplianceIssue(
                    level="error",
                    message=f"Language config is invalid; every transcribe will fail: {exc}",
                    model=name,
                )
            )
        return

    properties = getattr(instance, "properties", None)
    config = getattr(instance, "config", None)
    if (
        isinstance(properties, BaseProperties)
        and properties.has_language_axis
        and getattr(config, "default_language", None) is None
    ):
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    "Engine exposes a language axis (selectable_languages is non-empty) "
                    "but its config does not set default_language; every transcribe "
                    "will raise ConfigError (spec IC.6 / LANG R1)."
                ),
                model=name,
            )
        )


#: Public callables every compliant engine MUST expose unconditionally
#: (StandardASR protocol, spec §3.1). ``start_transcription`` is required only
#: when the engine declares a streaming axis -- handled separately below.
_ALWAYS_REQUIRED_METHODS: tuple[str, ...] = ("transcribe", "transcribe_async", "supports")


def _check_required_surface(
    instance: object,
    spec: ModelSpec,
    name: str,
    issues: list[ComplianceIssue],
) -> None:
    """Verify the engine exposes the full required public surface (D9).

    Every engine MUST expose the unconditional batch/query surface
    (:meth:`transcribe`, :meth:`transcribe_async`, :meth:`supports`); a missing
    member is a compliance **error**, not a silent accept. ``start_transcription``
    is required **only** when the engine declares a streaming axis
    (``streaming_input`` or ``streaming_output``) -- a batch-only engine
    legitimately omits it (spec §3.2). The ``properties``/``declared_capabilities``
    attributes are verified by the caller's type checks; this helper covers the
    callable methods and the conditional streaming entry point.

    Args:
        instance: The instantiated engine to inspect.
        spec: The engine's discovery :class:`~standard_asr.discovery.ModelSpec`.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    for method in _ALWAYS_REQUIRED_METHODS:
        if not callable(getattr(instance, method, None)):
            issues.append(
                ComplianceIssue(
                    level="error",
                    message=(
                        f"Instance is missing a callable {method!r} method "
                        "(required by the StandardASR protocol)."
                    ),
                    model=name,
                )
            )

    # ``start_transcription`` is required iff the engine declares streaming. Read
    # the declared axes defensively: a malformed ``declared_capabilities`` (its
    # own error is raised elsewhere) simply means we cannot assert a streaming
    # requirement here, so we do not over-report.
    declared = getattr(instance, "declared_capabilities", None)
    declares_streaming = isinstance(declared, DeclaredCapabilities) and (
        declared.supports("streaming_input") or declared.supports("streaming_output")
    )
    if declares_streaming and not callable(getattr(instance, "start_transcription", None)):
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    "Instance declares a streaming axis (streaming_input / "
                    "streaming_output) but is missing a callable "
                    "'start_transcription' method (spec §3.1)."
                ),
                model=name,
            )
        )


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

    config_type = inspect.getattr_static(engine_class, "config_type", None)
    if config_type is None:
        # DX nudge, not an error: without a class-level ``config_type`` a
        # settings UI cannot discover the engine's config schema (G.3.1) --
        # constructing a credentialed engine to read ``type(engine.config)``
        # requires the very values the UI is meant to collect.
        issues.append(
            ComplianceIssue(
                level="warning",
                message=(
                    "Engine class does not declare a class-level 'config_type'; "
                    "its init-config JSON Schema is not discoverable without "
                    "instantiation (registry.config_schema / GET /v1/config-schema)."
                ),
                model=name,
            )
        )
    elif not (isinstance(config_type, type) and issubclass(config_type, BaseConfig)):
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    f"config_type is set but is not a BaseConfig subclass (got {config_type!r})."
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


def check_event_sequence(
    events: Iterable[TranscriptionEvent],
    *,
    allow_empty: bool = False,
) -> ComplianceReport:
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
    ``new_ids`` (pure deletion) that would destroy frozen text -- an event stream
    that never reaches a terminal (``done`` / non-recoverable ``error``) event,
    **an empty sequence** (unless ``allow_empty=True``), and **any event emitted
    after the session-terminal** event (a terminal MUST be the last event).

    The per-segment lifecycle / frozen-prefix / supersede checks are obtained by
    replaying the events through the same
    :class:`~standard_asr.streaming._LifecycleGuard` the runtime uses, so the
    compliance verdict cannot drift from the runtime's enforcement. Events after
    the session-terminal are flagged and **not** replayed (they do not exist in a
    well-formed stream, so they MUST NOT mutate segment state).

    Args:
        events: The recorded events to validate, in emission order.
        allow_empty: When ``True``, an empty sequence is accepted (the rare
            intentional case). Default ``False`` -- an empty sequence is a
            violation, because a real session always emits at least a terminal
            event.

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
        if saw_terminal:
            # A session-terminal (done / non-recoverable error) MUST be the last
            # event. Flag the stray event and do NOT admit it to the guard: it is
            # invalid by position, so it must not pollute segment lifecycle state.
            issues.append(
                ComplianceIssue(
                    level="error",
                    message=(
                        f"event {event.type!r} emitted after the session-terminal event "
                        "(spec ST.6.1: a terminal done / non-recoverable error MUST be "
                        "the last event)."
                    ),
                    model=None,
                )
            )
            continue
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
    # Sweep for supersede frozen-prefix obligations the replacement never fully
    # re-froze before the sequence ended. This is the permitted (conservative)
    # direction of spec ST.5.2, so it is a soft WARNING -- it does NOT fail the
    # report -- consistent with how the runtime surfaces it via diagnostics().
    # Harvested AFTER the error loop above so it is not mis-promoted to error.
    for obligation in guard.finalize():
        issues.append(
            ComplianceIssue(
                level="warning",
                message=(f"streaming soft diagnostic ({obligation.code}): {obligation.message}"),
                model=None,
            )
        )
    if not saw_any:
        if not allow_empty:
            issues.append(
                ComplianceIssue(
                    level="error",
                    message=(
                        "empty event sequence: a streaming session MUST emit at least a "
                        "terminal (done / non-recoverable error) event (pass "
                        "allow_empty=True only for the rare intentional case)."
                    ),
                    model=None,
                )
            )
    elif not saw_terminal:
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
    * **best_effort** policy -- the call MUST succeed, drop (or degrade) the
      parameter, and surface the probe's expected diagnostic (e.g.
      ``unsupported_parameter_ignored``) via ``session.diagnostics()``.

    When the engine supports every probed parameter at the feature level, the
    check falls back to violating a declared **sub-constraint** of a supported
    feature (a prompt over its ``max_tokens`` budget, or a word-timestamp
    granularity outside the declared ``granularities``; see
    :func:`_pick_sub_constraint_probe`) and asserts the same strict-raise /
    best_effort-diagnose contract.

    An engine that declared streaming support (``streaming_input`` or
    ``streaming_output``) yet accepts the violating parameter -- the "forgot
    to gate" engine that bypassed the base template -- is a compliance
    **failure** here, so the gap is loud rather than silent. An engine that
    raises anything *other* than ``UnsupportedFeatureError`` from the probe is
    likewise recorded as a compliance error (never re-raised), so one crashing
    engine cannot abort the run.

    Engines that declare no streaming support, or that support every probed
    parameter and declare no violable sub-constraint, yield a clean (no-op)
    pass -- there is nothing to gate.

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

    try:
        if not (engine.supports("streaming_input") or engine.supports("streaming_output")):
            # The engine does not declare streaming support; there is no
            # streaming gating contract to exercise.
            return ComplianceReport(registry=ModelRegistry({}), issues=issues)

        probe = next(
            (
                (p[0], p[1](), DIAG_UNSUPPORTED_PARAMETER_IGNORED)
                for p in _GATING_PROBES
                if not engine.supports(p[2])
            ),
            None,
        )
        if probe is None:
            # Every probed parameter is supported at the feature level; fall
            # back to violating a declared sub-constraint of a supported
            # feature so the finer-grained half of the gating contract is
            # exercised too.
            probe = _pick_sub_constraint_probe(engine)
    except Exception as exc:  # noqa: BLE001
        # Probe selection reads engine-author surface (supports() /
        # effective_capabilities); contain a crash there exactly like the
        # start_transcription containment below -- this function promises
        # ``Raises: None`` and one broken engine must not abort the run.
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    f"selecting a streaming gating probe raised {exc!r}; "
                    "supports()/effective_capabilities must not raise while the "
                    "compliance suite probes the engine's declarations."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=ModelRegistry({}), issues=issues)
    if probe is None:
        # The engine supports every probed standard parameter and declares no
        # violable sub-constraint, so there is no gating path to exercise here.
        return ComplianceReport(registry=ModelRegistry({}), issues=issues)

    field_name, params, expected_code = probe
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
    except Exception as exc:  # noqa: BLE001
        # Any other exception is the engine crashing on the probe, not a gating
        # verdict. Mirror the broad-except guards used elsewhere in this module:
        # record an error and keep the compliance run alive (this function
        # promises ``Raises: None``).
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    f"start_transcription raised {exc!r} while probing streaming "
                    f"parameter {field_name!r}; the only contractual exception for a "
                    "gated parameter is UnsupportedFeatureError (spec Runtime R2)."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=ModelRegistry({}), issues=issues)

    # The session was created: only valid under best_effort, and only if the
    # probe was dropped/degraded + diagnosed.
    if strict:
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    f"strict engine accepted streaming parameter {field_name!r} "
                    "violating its declared capabilities without raising; it MUST "
                    "raise UnsupportedFeatureError (spec Runtime R2 / RUNT-3 "
                    "gating gap)."
                ),
                model=model,
            )
        )
    elif not any(d.code == expected_code for d in session.diagnostics()):
        issues.append(
            ComplianceIssue(
                level="error",
                message=(
                    f"best_effort engine silently swallowed streaming parameter "
                    f"{field_name!r}: no {expected_code!r} diagnostic surfaced via "
                    "session.diagnostics() (spec Runtime R2)."
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
