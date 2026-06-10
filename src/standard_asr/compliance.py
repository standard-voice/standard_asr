# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compliance helpers for Standard ASR plugin authors."""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from .asr_config import BaseConfig
from .asr_interface import EngineBase
from .asr_properties import BaseProperties
from .audio_format import AudioFormat
from .capabilities import DeclaredCapabilities, PromptCap, StreamingCapabilities, WordTimestampsCap
from .discovery import FactoryLoadError, ModelRegistry, ModelSpec, discover_models
from .exceptions import (
    ConfigError,
    EntrypointValidationError,
    InvalidProviderParamError,
    UnsupportedFeatureError,
)
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
    "assert_prefix_invariant",
    "check_entrypoints",
    "check_event_sequence",
    "check_provider_params_swap_safety",
    "check_recommended_wire_format",
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

    Mirrors the runtime :class:`~standard_asr.results.Diagnostic` shape: every
    issue carries a stable, machine-readable :attr:`code` so a CI pipeline can
    assert against (or whitelist) a specific category without string-matching the
    human-readable :attr:`message` -- the message is for humans and MAY be
    reworded, the code is the programmatic contract (the same reasoning that
    gives ``Diagnostic`` a ``code``).

    Attributes:
        level: Issue severity (``"error"`` or ``"warning"``).
        code: Stable machine-readable category identifier (e.g.
            ``"entrypoint_factory_failed"``, ``"streaming_invariant"``). Safe to
            match in CI; never reworded within a major version.
        message: Human-readable description (for display; MAY be reworded).
        model: The model key the issue is attributed to, or ``None`` for
            registry-/environment-level issues.
    """

    level: Literal["error", "warning"]
    code: str
    message: str
    model: str | None = None


@dataclass(frozen=True, slots=True)
class ComplianceReport:
    """Aggregate result returned by the compliance check functions.

    Attributes:
        registry: Model registry the entry-point check ran against. The
            behavioral checks (:func:`check_event_sequence`,
            :func:`check_streaming_param_gating`, :func:`check_recommended_wire_format`,
            :func:`check_sync_bridge`) do not operate on a registry and pass ``None``.
        issues: Collected compliance issues.
    """

    registry: ModelRegistry | None
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

    Environment-level invariants are reported as issues alongside per-engine
    ones -- never as raised exceptions -- so a single command yields one report
    even when discovery itself found problems:

    * **IC.2 engine-identity collisions** (an ``engine_id`` provided by more than
      one distribution, making ``config.engine`` routing ambiguous) are reported
      as **errors** here. The discovery layer only *marks* them
      (``registry.shadowed_engine_ids``); the compliance suite is the fail-loud
      layer spec IC.2 mandates, so a collision is a compliance failure even on a
      default (non-strict) run rather than a log line.
    * **Invalid entry-point names** surface as an error too. When
      ``strict_discovery`` is ``True``, ``discover_models`` would normally *raise*
      on them; this function catches that and converts it into an error issue (so
      its ``Raises: None`` contract holds and a report is always returned), then
      re-discovers leniently so the valid engines are still checked.

    Args:
        registry: Optional pre-discovered registry. When provided, discovery is
            skipped and ``strict_discovery`` is ignored (collisions are still
            reported from ``registry.shadowed_engine_ids``).
        strict_discovery: Treat invalid entry-point names as a hard discovery
            error (reported as an error issue here, never raised). Default
            ``False``. IC.2 collisions are reported as errors regardless.
        instantiate: If ``True``, instantiate zero-arg factories and verify metadata.

    Returns:
        Compliance report summarizing findings.

    Raises:
        None.
    """

    issues: list[ComplianceIssue] = []

    if registry is None:
        try:
            registry = discover_models(strict=strict_discovery)
        except EntrypointValidationError as exc:
            # strict discovery raises on invalid entry-point names (and IC.2
            # collisions). A compliance check MUST always return a report, so
            # convert the failure to an error issue and re-discover leniently to
            # still check the valid engines. IC.2 collisions are additionally
            # reported per-engine_id below from ``shadowed_engine_ids``.
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="entrypoint_invalid",
                    message=(f"Strict discovery rejected one or more entry points: {exc}"),
                    model=None,
                )
            )
            registry = discover_models(strict=False)

    # §R.4 R1: RuntimeParams MUST be a closed type. This is a global invariant of
    # the standard, not per-engine, so it MUST be verified even in a bare
    # environment with no plugins installed -- it runs *before* the empty-registry
    # early return so the global invariant is never silently unchecked.
    if not _is_closed_model(RuntimeParams):
        issues.append(
            ComplianceIssue(
                level="error",
                code="runtime_params_not_closed",
                message="RuntimeParams is not a closed type (extra='forbid').",
                model=None,
            )
        )

    # IC.2: an engine_id contributed by more than one distribution makes
    # config.engine routing depend on install order. The discovery layer only
    # marks these (consumers may surface or reject); the compliance suite is the
    # fail-loud layer, so each collision is an error -- reported even on a default
    # non-strict run, which is exactly where the discovery warning is easy to miss.
    for engine_id in sorted(registry.shadowed_engine_ids):
        issues.append(
            ComplianceIssue(
                level="error",
                code="engine_id_collision",
                message=(
                    f"engine_id {engine_id!r} is provided by more than one "
                    "distribution (IC.2 identity collision); config.engine routing "
                    "is ambiguous. Install only one provider for this engine_id, or "
                    "have the authors choose distinct engine_ids."
                ),
                model=engine_id,
            )
        )

    if len(registry) == 0:
        issues.append(
            ComplianceIssue(
                level="error",
                code="no_entrypoints",
                message="No standard_asr.models entry points were discovered.",
                model=None,
            )
        )
        return ComplianceReport(registry=registry, issues=issues)

    for name in registry.names():
        _check_engine(registry, name, instantiate=instantiate, issues=issues)

    return ComplianceReport(registry=registry, issues=issues)


def _check_engine(
    registry: ModelRegistry,
    name: str,
    *,
    instantiate: bool,
    issues: list[ComplianceIssue],
) -> None:
    """Run every per-engine check for one model key, with crash containment.

    A single engine whose author wrote a property that raises (a malformed
    ``@property properties`` / ``config``, not just the ``effective_capabilities``
    already guarded) MUST NOT abort the whole compliance run --
    :func:`check_entrypoints` promises ``Raises: None`` so a multi-plugin
    environment still gets a verdict on the other engines. Any unexpected
    exception from this engine's surface is therefore caught and reported as an
    error issue against that engine.

    Args:
        registry: The discovered registry (for the spec lookup).
        name: The model key to check.
        instantiate: Whether to instantiate the factory and verify the instance.
        issues: The mutable list of issues to append to.
    """
    try:
        _check_engine_unguarded(registry, name, instantiate=instantiate, issues=issues)
    except Exception as exc:  # noqa: BLE001
        issues.append(
            ComplianceIssue(
                level="error",
                code="engine_check_crashed",
                message=(
                    f"Checking engine {name!r} raised {exc!r}; an engine's public "
                    "surface (properties / config / capabilities) MUST be readable "
                    "without raising during a compliance check."
                ),
                model=name,
            )
        )


def _check_engine_unguarded(
    registry: ModelRegistry,
    name: str,
    *,
    instantiate: bool,
    issues: list[ComplianceIssue],
) -> None:
    """Per-engine checks (the body :func:`_check_engine` wraps for containment).

    Args:
        registry: The discovered registry (for the spec lookup).
        name: The model key to check.
        instantiate: Whether to instantiate the factory and verify the instance.
        issues: The mutable list of issues to append to.
    """
    spec = registry.spec(name)
    try:
        factory = spec.load_factory()
    except FactoryLoadError as exc:
        issues.append(
            ComplianceIssue(
                level="error", code="entrypoint_factory_unloadable", message=str(exc), model=name
            )
        )
        return

    # §3.1 / §C: declared metadata MUST be readable from the class without
    # instantiation. Resolve the class and read its ClassVars directly.
    _check_class_level_metadata(spec, name, issues)

    if not instantiate:
        return

    if not _can_call_without_args(factory):
        issues.append(
            ComplianceIssue(
                level="warning",
                code="factory_requires_args",
                message=(
                    "Factory cannot be invoked without arguments; skipped instantiation check."
                ),
                model=name,
            )
        )
        return

    try:
        instance = factory()
    except (ConfigError, ValidationError) as exc:
        # IC.4: a credentialed engine's zero-arg factory raises when the required
        # credential is absent (explicit config > env > raise). On a clean CI with
        # no env vars set this is the *correct* behavior, so it MUST NOT be a
        # compliance error -- otherwise the verdict would depend on the runtime's
        # credential state rather than the plugin. Report it as a warning skip and
        # point at the env var; pass --no-instantiate or set the credential to run
        # the full instance-level checks.
        issues.append(
            ComplianceIssue(
                level="warning",
                code="factory_requires_config",
                message=(
                    "Skipped instantiation: the factory requires configuration not "
                    f"present in this environment ({exc!r}). Set the engine's "
                    "STANDARD_ASR_<ENGINE>_<FIELD> environment variable (e.g. an API "
                    "key) or pass an explicit config to run the full instance checks."
                ),
                model=name,
            )
        )
        return
    except Exception as exc:  # noqa: BLE001
        issues.append(
            ComplianceIssue(
                level="error",
                code="entrypoint_factory_failed",
                message=f"Factory invocation failed with {exc!r}.",
                model=name,
            )
        )
        return

    _check_required_surface(instance, name, issues)
    _check_prepare_hook(instance, name, issues)
    _check_instance_properties(instance, spec, name, issues)
    _check_instance_config(instance, name, issues)
    _check_instance_capabilities(instance, name, issues)


def _check_instance_properties(
    instance: object,
    spec: ModelSpec,
    name: str,
    issues: list[ComplianceIssue],
) -> None:
    """Verify the instance's ``properties`` (presence, identity match, re-validation).

    Args:
        instance: The instantiated engine.
        spec: The engine's discovery spec.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    properties = getattr(instance, "properties", None)
    if not isinstance(properties, BaseProperties):
        issues.append(
            ComplianceIssue(
                level="error",
                code="missing_properties",
                message="Instance is missing a BaseProperties-compatible 'properties' attribute.",
                model=name,
            )
        )
        return
    if properties.model_id != spec.model_id:
        issues.append(
            ComplianceIssue(
                level="error",
                code="properties_key_mismatch",
                message=(
                    "Instance properties.model_id does not match the entry point key "
                    f"({properties.model_id!r} != {spec.model_id!r})."
                ),
                model=name,
            )
        )
    # Defense in depth: re-validate the declared properties through the full
    # pydantic pipeline. Declaration-time validation covers the documented
    # subclass-with-defaults pattern (validate_default), but an engine could still
    # hand back an instance built through a validation-bypassing path
    # (model_construct, mutated copies); a round-trip catches those before they are
    # certified compliant.
    try:
        type(properties).model_validate(properties.model_dump())
    except ValidationError as exc:
        issues.append(
            ComplianceIssue(
                level="error",
                code="properties_revalidation_failed",
                message=f"Instance properties fail re-validation: {exc}",
                model=name,
            )
        )


def _check_instance_config(
    instance: object,
    name: str,
    issues: list[ComplianceIssue],
) -> None:
    """Verify the instance's ``config`` (presence, declared-type match, language axis).

    Args:
        instance: The instantiated engine.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    config = getattr(instance, "config", None)
    if not isinstance(config, BaseConfig):
        issues.append(
            ComplianceIssue(
                level="error",
                code="missing_config",
                message="Instance is missing a BaseConfig-compatible 'config' attribute.",
                model=name,
            )
        )
        return
    declared_config_type = inspect.getattr_static(type(instance), "config_type", None)
    if (
        isinstance(declared_config_type, type)
        and issubclass(declared_config_type, BaseConfig)
        and not isinstance(config, declared_config_type)
    ):
        issues.append(
            ComplianceIssue(
                level="error",
                code="config_type_mismatch",
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


def _check_instance_capabilities(
    instance: object,
    name: str,
    issues: list[ComplianceIssue],
) -> None:
    """Verify ``declared_capabilities`` and the ``effective ⊆ declared`` invariant.

    Args:
        instance: The instantiated engine.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    declared = getattr(instance, "declared_capabilities", None)
    if not isinstance(declared, DeclaredCapabilities):
        issues.append(
            ComplianceIssue(
                level="error",
                code="missing_declared_capabilities",
                message=(
                    "Instance is missing a DeclaredCapabilities 'declared_capabilities' attribute."
                ),
                model=name,
            )
        )
        return
    try:
        effective = getattr(instance, "effective_capabilities", None)
    except Exception as exc:  # noqa: BLE001
        # A buggy ``effective_capabilities`` property MUST NOT crash the whole
        # compliance run (this function promises ``Raises: None``); report the
        # offender and keep checking the other engines.
        issues.append(
            ComplianceIssue(
                level="error",
                code="effective_capabilities_raised",
                message=(
                    f"Reading effective_capabilities raised {exc!r}; the "
                    "property MUST return a DeclaredCapabilities (or None) "
                    "without raising."
                ),
                model=name,
            )
        )
        return
    if isinstance(effective, DeclaredCapabilities):
        if not declared.covers(effective):
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="effective_widens_declared",
                    message=(
                        "effective_capabilities is not a subset of "
                        "declared_capabilities (effective MUST only narrow)."
                    ),
                    model=name,
                )
            )
    elif effective is not None:
        # A non-None, wrong-typed ``effective`` is itself a violation: the
        # effective ⊆ declared invariant MUST NOT be evadable by returning the
        # wrong type (which would silently skip the subset check). ``None`` (engine
        # declares no narrowing) stays a legitimate no-op.
        issues.append(
            ComplianceIssue(
                level="error",
                code="effective_capabilities_wrong_type",
                message=(
                    "effective_capabilities is not a DeclaredCapabilities "
                    f"(got {type(effective).__name__!r}); it MUST be a "
                    "DeclaredCapabilities (or None) so the effective ⊆ "
                    "declared invariant can be verified."
                ),
                model=name,
            )
        )


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
                    code="language_config_invalid",
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
                code="language_axis_without_default",
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

    For an :class:`EngineBase` engine the streaming requirement uses the same
    :meth:`~standard_asr.asr_interface.EngineBase._overrides_streaming` predicate
    the runtime template uses to decide whether streaming is *actually*
    implemented (G.2.1 "compliance shares the runtime's validation logic"): the
    base class always supplies a ``start_transcription`` template, so a mere
    ``callable(...)`` check would certify an engine that declares streaming yet
    never overrides the hook -- a capability lie the runtime rejects with
    ``UnsupportedFeatureError`` at the user's first ``start_transcription`` call.

    Args:
        instance: The instantiated engine to inspect.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    for method in _ALWAYS_REQUIRED_METHODS:
        if not callable(getattr(instance, method, None)):
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="missing_required_method",
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
    if not declares_streaming:
        return
    if isinstance(instance, EngineBase):
        # The base template always provides start_transcription, so presence is
        # not enough: the engine must override the _start_transcription hook, or
        # the runtime raises UnsupportedFeatureError at session establishment.
        if not instance._overrides_streaming():  # pyright: ignore[reportPrivateUsage]
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="streaming_declared_not_implemented",
                    message=(
                        "Instance declares a streaming axis (streaming_input / "
                        "streaming_output) but does not implement the streaming hook "
                        "(_start_transcription); start_transcription would raise "
                        "UnsupportedFeatureError at runtime (spec §3.1 / C R1 "
                        "fail-closed: a declared capability is a promise)."
                    ),
                    model=name,
                )
            )
        return
    if not callable(getattr(instance, "start_transcription", None)):
        issues.append(
            ComplianceIssue(
                level="error",
                code="missing_start_transcription",
                message=(
                    "Instance declares a streaming axis (streaming_input / "
                    "streaming_output) but is missing a callable "
                    "'start_transcription' method (spec §3.1)."
                ),
                model=name,
            )
        )


def prepare_requires_arguments(prepare: Callable[..., object]) -> bool:
    """Return whether a ``prepare()`` warm-up hook needs caller-supplied arguments.

    A spec IC.11 warm-up hook MUST be invocable with no arguments. A parameter
    makes the hook non-conforming only when it is *required*: it has no default
    and is positional-or-keyword, positional-only, or keyword-only. ``*args`` and
    ``**kwargs`` impose no required argument, and a bound method's ``self`` is
    already supplied, so neither counts.

    This is the single definition of the zero-argument half of the contract,
    shared by :func:`_check_prepare_hook` and the ``standard-asr models prepare``
    CLI command so the compliance verdict and the runtime behaviour cannot drift
    (goal G.2.1).

    Args:
        prepare: An engine's ``prepare`` attribute, already confirmed callable.

    Returns:
        ``True`` when calling ``prepare()`` with no arguments would fail because a
        required parameter is unfilled; ``False`` for a valid zero-argument hook
        (or one whose signature cannot be introspected).
    """
    try:
        signature = inspect.signature(prepare)
    except (TypeError, ValueError):
        # A callable whose signature cannot be introspected (e.g. some C builtins)
        # cannot be proven to require arguments; treat it as zero-arg and let an
        # actual call surface any real arity error.
        return False
    return any(
        parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        for parameter in signature.parameters.values()
    )


def _check_prepare_hook(instance: object, name: str, issues: list[ComplianceIssue]) -> None:
    """Verify the optional ``prepare()`` warm-up hook honours its contract (IC.11).

    ``prepare()`` is optional, but when present (overridden past the
    :class:`~standard_asr.asr_interface.EngineBase` no-op) it MUST be a
    **synchronous, zero-argument** method (spec IC.11). A coroutine ``prepare``
    is the dangerous case: ``standard-asr models prepare`` would call it, get an
    un-awaited coroutine, and report a false "prepare complete" without ever
    warming up -- a silent success the suite must catch. A ``prepare`` that
    requires arguments can never be driven by the toolchain. Both are recorded as
    compliance errors.

    Args:
        instance: The instantiated engine to inspect.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    prepare = getattr(instance, "prepare", None)
    if prepare is None or not callable(prepare):
        # No hook (structural engine) or a non-callable attribute. A non-callable
        # 'prepare' is rejected at call time by the CLI; here, absence of a
        # callable simply means there is no warm-up contract to verify.
        return
    if inspect.iscoroutinefunction(prepare):
        issues.append(
            ComplianceIssue(
                level="error",
                code="prepare_hook_is_coroutine",
                message=(
                    "prepare() is a coroutine function; the warm-up hook MUST be a "
                    "synchronous zero-argument method (spec IC.11) -- an async "
                    "prepare() would be reported complete without ever warming up."
                ),
                model=name,
            )
        )
        return
    if prepare_requires_arguments(prepare):
        issues.append(
            ComplianceIssue(
                level="error",
                code="prepare_hook_requires_args",
                message=(
                    "prepare() requires arguments; the warm-up hook MUST be callable "
                    "with no arguments (spec IC.11)."
                ),
                model=name,
            )
        )


def _check_streaming_wire_encodings_declared(
    declared: object,
    properties: object,
    name: str,
    issues: list[ComplianceIssue],
) -> None:
    """Nudge a ``streaming_input`` engine to declare ``wire_encodings``.

    ``wire_encodings`` is the fail-closed allowlist that
    :meth:`~standard_asr.asr_interface.EngineBase.ensure_stream_format_supported`
    matches a declared ``audio_format`` against; when it is ``None`` the encoding
    check is skipped (``None`` means "unconstrained", spec §AI), so an engine that
    actually frames PCM but forgets to declare it would read a ``mulaw``
    ``audio_format`` session's frames as PCM -- a silent mistranscription (the
    cardinal sin). An engine that declares ``streaming_input`` can be opened with
    an explicit ``audio_format``, so the omission is reported here as a **warning**
    (a DX nudge, like the missing ``config_type`` one) rather than an error: a
    bare-call engine that self-manages its wire format (spec ST §3.1) legitimately
    leaves it unconstrained, and only ever opens ``audio_format``-less sessions, so
    a hard error would be wrong. This is the compensating compliance signal the
    fail-open ``None`` default lacks at runtime.

    Args:
        declared: The engine's class-level ``declared_capabilities`` (any object;
            only acted on when it is a :class:`DeclaredCapabilities`).
        properties: The engine's class-level ``properties`` (any object; only
            acted on when it is a :class:`BaseProperties`).
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    if not (isinstance(declared, DeclaredCapabilities) and isinstance(properties, BaseProperties)):
        return
    if declared.supports("streaming_input") and properties.wire_encodings is None:
        issues.append(
            ComplianceIssue(
                level="warning",
                code="streaming_input_without_wire_encodings",
                message=(
                    "Engine declares 'streaming_input' but does not declare "
                    "'wire_encodings'; an audio_format session's wire encoding then "
                    "cannot be validated and a non-PCM (e.g. mulaw) frame would be "
                    "read as PCM -- a silent mistranscription. Declare wire_encodings "
                    "(e.g. ['pcm_s16le']) unless the adapter self-manages its wire "
                    "format via bare start_transcription() (spec §AI wire_encodings)."
                ),
                model=name,
            )
        )


def _check_streaming_axis_declared(
    declared: object,
    name: str,
    issues: list[ComplianceIssue],
) -> None:
    """Flag a populated ``streaming`` domain with neither transport axis supported.

    A present ``streaming`` capabilities domain means streaming is supported
    (``supports("streaming")`` is True), but the *usable* transport is the
    ``streaming_input`` / ``streaming_output`` flags -- and ``start_transcription``
    fails closed when neither is supported (it raises on the input path AND the
    whole-input output path). So a tree that populates the ``streaming`` domain yet
    leaves both flags unsupported declares a streaming engine that EVERY
    ``start_transcription`` call rejects: shipped, discoverable as "streaming", and
    uncallable. The inverse mistake (a flag without the domain) is already a
    construction-time ``ValueError``
    (:meth:`~standard_asr.capabilities.DeclaredCapabilities.\
_require_streaming_domain_for_streaming_flags`); this closes the asymmetry on the
    silent side (CC-1).

    Unlike the ``wire_encodings`` nudge -- which a self-managing-wire adapter may
    legitimately trip, so it is a *warning* -- there is NO legitimate engine with a
    streaming domain and neither axis (it cannot be opened at all), so this is an
    **error**: the compliance run MUST fail rather than soft-nudge a definitely
    broken engine.

    Args:
        declared: The engine's class-level ``declared_capabilities`` (any object;
            only acted on when it is a :class:`DeclaredCapabilities`).
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    if not isinstance(declared, DeclaredCapabilities):
        return
    if declared.streaming is not None and not (
        declared.supports("streaming_input") or declared.supports("streaming_output")
    ):
        issues.append(
            ComplianceIssue(
                level="error",
                code="streaming_domain_without_axis",
                message=(
                    "Engine declares a 'streaming' capabilities domain but neither "
                    "'streaming_input' nor 'streaming_output' is supported; every "
                    "start_transcription call then fails closed "
                    "(UnsupportedFeatureError) -- a streaming engine nobody can call. "
                    "Declare streaming_input=FlagCap(supported=True) and/or "
                    "streaming_output=FlagCap(supported=True), or drop the streaming "
                    "domain."
                ),
                model=name,
            )
        )


def _check_class_level_metadata(spec: ModelSpec, name: str, issues: list[ComplianceIssue]) -> None:
    """Verify class-level metadata is readable without instantiation (§3.1/§C).

    Reads ``declared_capabilities`` and ``provider_params_type`` from the engine
    *class* (never the instance) and validates that, when present, the
    provider-params type is a closed :class:`ProviderParams` subclass (§R.4 R1).

    Args:
        spec: The :class:`~standard_asr.discovery.ModelSpec`.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    try:
        engine_class = spec.engine_class()
    except FactoryLoadError as exc:
        issues.append(
            ComplianceIssue(
                level="error",
                code="class_metadata_unreadable",
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
                code="missing_class_declared_capabilities",
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
                code="missing_class_properties",
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
                code="missing_config_type",
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
                code="config_type_not_baseconfig",
                message=(
                    f"config_type is set but is not a BaseConfig subclass (got {config_type!r})."
                ),
                model=name,
            )
        )

    _check_streaming_wire_encodings_declared(declared, properties, name, issues)
    _check_streaming_axis_declared(declared, name, issues)

    params_type = inspect.getattr_static(engine_class, "provider_params_type", None)
    if params_type is None:
        return
    if not (isinstance(params_type, type) and issubclass(params_type, ProviderParams)):
        issues.append(
            ComplianceIssue(
                level="error",
                code="provider_params_type_not_subclass",
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
                code="provider_params_type_not_closed",
                message=(
                    "provider_params_type must be a closed type (extra='forbid'); "
                    f"{params_type.__name__} is not."
                ),
                model=name,
            )
        )
    elif params_type is ProviderParams:
        issues.append(
            ComplianceIssue(
                level="error",
                code="provider_params_type_is_bare_base",
                message=(
                    "provider_params_type is the bare ProviderParams base, which has "
                    "no fields and admits any params -- this zeroes swap-safety (spec "
                    "§3.2). Publish a distinct terminal ProviderParams subclass as the "
                    "engine's provider_params type."
                ),
                model=name,
            )
        )


def _cross_check_event_capabilities(
    event: TranscriptionEvent,
    streaming: StreamingCapabilities,
    issues: list[ComplianceIssue],
) -> None:
    """Cross-check one event's fields against the declared streaming capabilities.

    The "no-timestamp streaming" profile couples a declared streaming capability
    with the event field it gates; an engine that declares the capability
    unsupported yet emits the field anyway is a capability⇄stream desync the
    structural invariants cannot see (SF-4). The stream MUST NOT *exceed* what the
    capabilities promise:

    * ``word_stability`` unsupported ⇒ no event may carry a meaningful
      ``stable_until`` (> 0): the field asserts a frozen prefix the engine declared
      it does not provide.
    * streaming ``timestamps`` mode ``none`` ⇒ no event may carry
      ``audio_processed_until``: that cursor is a streaming timestamp the engine
      declared it does not emit.
    * ``word_timestamps`` unsupported ⇒ no event may carry ``words``: per-word
      timings are word timestamps the engine declared it does not produce.

    The reverse -- declaring a capability a given recorded stream simply never
    exercises -- is not a violation, so each check is one-directional.

    Args:
        event: The event to check.
        streaming: The engine's declared streaming capabilities.
        issues: The mutable list of issues to append to.
    """
    if (
        event.stable_until is not None
        and event.stable_until > 0
        and not streaming.word_stability.is_supported
    ):
        issues.append(
            ComplianceIssue(
                level="error",
                code="stream_exceeds_word_stability",
                message=(
                    f"event emits stable_until={event.stable_until} (a frozen prefix) but "
                    "the engine declares streaming.word_stability unsupported -- the "
                    "declared capabilities and the emitted stream disagree. Declare "
                    "word_stability supported, or do not emit a non-zero stable_until."
                ),
                model=None,
            )
        )
    if event.audio_processed_until is not None and not streaming.timestamps.is_supported:
        issues.append(
            ComplianceIssue(
                level="error",
                code="stream_exceeds_timestamps",
                message=(
                    f"event emits audio_processed_until={event.audio_processed_until} but the "
                    "engine declares streaming.timestamps mode 'none' (no streaming "
                    "timestamps) -- the declared capabilities and the emitted stream "
                    "disagree. Declare a timestamps mode, or do not emit "
                    "audio_processed_until."
                ),
                model=None,
            )
        )
    if event.words and not streaming.word_timestamps.is_supported:
        issues.append(
            ComplianceIssue(
                level="error",
                code="stream_exceeds_word_timestamps",
                message=(
                    "event emits per-word timings (words) but the engine declares "
                    "streaming.word_timestamps unsupported -- the declared capabilities "
                    "and the emitted stream disagree. Declare word_timestamps supported, "
                    "or do not emit words."
                ),
                model=None,
            )
        )


def check_event_sequence(
    events: Iterable[TranscriptionEvent],
    *,
    allow_empty: bool = False,
    capabilities: DeclaredCapabilities | None = None,
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
        capabilities: When provided, additionally cross-check each event against
            the engine's declared streaming capabilities (SF-4): a stream MUST NOT
            *exceed* what it declares -- e.g. emit a non-zero ``stable_until`` while
            ``word_stability`` is unsupported, an ``audio_processed_until`` cursor
            while ``timestamps`` mode is ``none``, or ``words`` while
            ``word_timestamps`` is unsupported. Pass
            ``engine.declared_capabilities`` to catch a declaration that disagrees
            with the engine's actual output. ``None`` skips the cross-check.

    Returns:
        A :class:`ComplianceReport`; ``passed`` is ``True`` when the sequence
        honours every streaming invariant.
    """
    guard = _LifecycleGuard(strict=False)
    issues: list[ComplianceIssue] = []
    saw_any = False
    saw_terminal = False
    # The streaming sub-domain to cross-check events against (SF-4); ``None`` when
    # no capabilities were supplied or the tree declares no streaming domain.
    streaming_caps = capabilities.streaming if capabilities is not None else None
    for event in events:
        saw_any = True
        if saw_terminal:
            # A session-terminal (done / non-recoverable error) MUST be the last
            # event. Flag the stray event and do NOT admit it to the guard: it is
            # invalid by position, so it must not pollute segment lifecycle state.
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="event_after_terminal",
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
        if streaming_caps is not None:
            _cross_check_event_capabilities(event, streaming_caps, issues)
        if event.is_terminal:
            saw_terminal = True
    for diagnostic in guard.diagnostics:
        # Pass the guard's stable diagnostic code through structurally (namespaced
        # so it cannot collide with this module's own codes) instead of only
        # interpolating it into the message: a CI pipeline can match
        # ``streaming_invariant:<guard_code>`` without parsing free text. The
        # message keeps the human-readable form (and the code, for terminals).
        issues.append(
            ComplianceIssue(
                level="error",
                code=f"streaming_invariant:{diagnostic.code}",
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
                code=f"streaming_soft:{obligation.code}",
                message=(f"streaming soft diagnostic ({obligation.code}): {obligation.message}"),
                model=None,
            )
        )
    if not saw_any:
        if not allow_empty:
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="empty_event_sequence",
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
                code="missing_terminal_event",
                message=(
                    "event stream ended without a terminal (done / non-recoverable "
                    "error) event (spec ST.6.1: the stream MUST terminate)."
                ),
                model=None,
            )
        )
    return ComplianceReport(registry=None, issues=issues)


#: Guard diagnostic codes that signal a violated frozen-prefix / stability
#: invariant (as opposed to a lifecycle or audio-cursor one), scoping
#: :func:`assert_prefix_invariant` to exactly the prefix invariant.
_PREFIX_INVARIANT_CODES: frozenset[str] = frozenset(
    {"frozen_prefix_rewritten", "frozen_prefix_rewritten_supersede", "stable_until_clamped"}
)


def assert_prefix_invariant(events: Iterable[TranscriptionEvent]) -> None:
    """Assert a recorded stream's partials honour the frozen-prefix invariant.

    Test helper for engine authors. Partials are **lossy under backpressure**: the
    base coalesces pending partials when the consumer is slow (spec ST.6.4), so the
    partial *count* is non-deterministic -- the same engine may surface five
    partials or none purely by consumer timing. Asserting a count is therefore
    flaky; assert the **invariant** instead. This checks only the prefix invariant
    -- a segment's frozen prefix (``text[:stable_until]``) is never rewritten and
    ``stable_until`` never regresses -- across however many partials survived
    coalescing, and (unlike :func:`check_event_sequence`) does NOT require a
    terminal event, so it also applies to a mid-stream slice. It replays events
    through the same runtime :class:`~standard_asr.streaming._LifecycleGuard` the
    runtime uses, so the assertion cannot drift from enforcement.

    Args:
        events: The recorded events, in emission order.

    Raises:
        AssertionError: If any segment's frozen prefix was rewritten or its
            ``stable_until`` regressed.
    """
    guard = _LifecycleGuard(strict=False)
    for event in events:
        guard.admit(event)
    violations = [d for d in guard.diagnostics if d.code in _PREFIX_INVARIANT_CODES]
    if violations:
        detail = "; ".join(f"{d.code}: {d.message}" for d in violations)
        raise AssertionError(
            "stream violates the frozen-prefix invariant (partials must form "
            "monotonic, never-rewritten prefixes; assert this, not partial counts): "
            f"{detail}"
        )


def _safe_engine_id(engine: object) -> str | None:
    """Read ``engine.properties.engine_id`` without ever raising (issue attribution).

    The behavioral checks promise ``Raises: None`` and use the engine id only to
    attribute issues. An engine author may have written a ``properties`` (or
    ``engine_id``) that raises a non-``AttributeError`` -- ``getattr`` only
    swallows ``AttributeError`` -- so the read is fully contained here: a broken
    declaration yields ``None`` attribution rather than aborting the check that
    exists to diagnose such breakage.

    Args:
        engine: The engine under test.

    Returns:
        The engine id, or ``None`` when it cannot be read.
    """
    try:
        return getattr(getattr(engine, "properties", None), "engine_id", None)
    except Exception:  # noqa: BLE001 - attribution must never raise
        return None


def _synthesize_probe_audio_format(engine: EngineBase) -> AudioFormat:
    """Build a *legal* wire :class:`AudioFormat` for a ``streaming_input`` probe.

    The streaming gating probe must hand the engine's
    :meth:`~standard_asr.asr_interface.EngineBase._start_transcription` hook a
    valid session context: an engine that does not self-manage its wire format
    (an incremental ElevenLabs-style adapter) legitimately fail-louds when opened
    with ``audio_format=None`` (spec §AI R6 -- bare-PCM streaming locks the
    sample rate at session establishment). Probing it with no ``audio_format``
    would make that *correct* fail-loud read as a compliance error. So the probe
    uses the engine's own
    :meth:`~standard_asr.asr_interface.EngineBase.recommended_wire_format` -- the
    single source of truth (AW-2) -- which yields a format the engine's own
    :meth:`~standard_asr.asr_interface.EngineBase.ensure_stream_format_supported`
    accepts.

    Args:
        engine: The engine under test (must declare ``streaming_input``).

    Returns:
        A wire format that the engine's session-establishment guard accepts.

    Raises:
        ValueError: When the engine recommends no usable wire format (declares no
            usable sample rate), so no legal probe context can be built. The
            caller maps this to a ``gating_probe_context_unbuildable`` issue.
    """
    fmt = engine.recommended_wire_format()
    if fmt is None:
        raise ValueError("engine declares no usable wire sample rate to synthesize a probe format")
    return fmt


def check_streaming_param_gating(engine: EngineBase) -> ComplianceReport:
    """Assert a streaming engine gates an unsupported standard parameter.

    Closes the streaming-gating bypass gap as a *compliance* failure rather
    than a silent one: the base
    :meth:`~standard_asr.asr_interface.EngineBase.start_transcription`
    template runs ``gate_params(mode="streaming")`` for every engine, so a
    "forgot to gate" engine (one that bypassed the template) must show up here.

    The check establishes a streaming session (via ``start_transcription``,
    which constructs the session but does **not** enter its context, so no wire
    connection is opened) for the first standard parameter the engine does
    **not** support in ``streaming`` mode and asserts the standard contract:

    * **strict** policy -- the call MUST raise
      :class:`~standard_asr.exceptions.UnsupportedFeatureError` whose ``param``
      identifies the gated field;
    * **best_effort** policy -- the call MUST succeed, drop (or degrade) the
      parameter, and surface the probe's expected diagnostic (e.g.
      ``unsupported_parameter_ignored``) via ``session.diagnostics()``.

    When the engine supports every probed parameter at the feature level, the
    check falls back to violating a declared **sub-constraint** of a supported
    feature (a prompt over its ``max_tokens`` budget, or a word-timestamp
    granularity outside the declared ``granularities``; see
    :func:`_pick_sub_constraint_probe`) and asserts the same strict-raise /
    best_effort-diagnose contract.

    **Legal session context.** A ``streaming_input`` engine is probed with a
    synthesized, *valid* wire :class:`AudioFormat` (see
    :func:`_synthesize_probe_audio_format`), so an engine that legitimately
    fail-louds on a missing ``audio_format`` (spec §AI R6) is not misjudged as
    non-compliant for obeying the standard. A ``streaming_output``-only engine is
    probed with a one-sample silent ``audio`` input, but **only under the strict
    policy**: strict gating raises *before* the audio is decoded or the model is
    touched (gate order: params first, then audio), so the probe is free of the
    billable side effect a best_effort probe would incur by reaching the engine.
    A best_effort ``streaming_output``-only engine is therefore reported as a
    ``warning`` skip (inconclusive) rather than driven into real inference.

    **Distinguishing a gating raise from "streaming unsupported".** The strict
    contract is satisfied only by an ``UnsupportedFeatureError`` whose ``param``
    equals the probed field. An engine that *declares* a streaming axis but never
    implements the hook raises an ``UnsupportedFeatureError`` with no (or a
    different) ``param``; that is a capability lie, not a gating success, and is
    recorded as a distinct error instead of being mistaken for a clean pass.

    An engine that declared streaming support yet accepts the violating
    parameter -- the "forgot to gate" engine that bypassed the base template --
    is a compliance **failure** here, so the gap is loud rather than silent. An
    engine that raises anything *other* than ``UnsupportedFeatureError`` from the
    probe is likewise recorded as a compliance error (never re-raised), so one
    crashing engine cannot abort the run.

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
    model = _safe_engine_id(engine)

    try:
        supports_input = engine.supports("streaming_input")
        supports_output = engine.supports("streaming_output")
        if not (supports_input or supports_output):
            # The engine does not declare streaming support; there is no
            # streaming gating contract to exercise.
            return ComplianceReport(registry=None, issues=issues)

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
                code="gating_probe_selection_raised",
                message=(
                    f"selecting a streaming gating probe raised {exc!r}; "
                    "supports()/effective_capabilities must not raise while the "
                    "compliance suite probes the engine's declarations."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)
    if probe is None:
        # The engine supports every probed standard parameter and declares no
        # violable sub-constraint, so there is no gating path to exercise here.
        return ComplianceReport(registry=None, issues=issues)

    field_name, params, expected_code = probe
    strict = bool(getattr(engine.config, "strict", True))

    # Build a legal session context so the probe exercises *gating*, not a missing
    # audio_format / audio fail-loud. Prefer the incremental (streaming_input)
    # path; fall back to whole-input only for streaming_output-only engines, and
    # only under strict (a best_effort probe there would run real inference).
    open_kwargs: dict[str, object] = {"params": params}
    if supports_input:
        try:
            open_kwargs["audio_format"] = _synthesize_probe_audio_format(engine)
        except Exception as exc:  # noqa: BLE001
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="gating_probe_context_unbuildable",
                    message=(
                        f"could not synthesize a legal wire audio_format from the "
                        f"engine's Properties to probe gating ({exc!r}); declare a "
                        "reachable native_sample_rate / wire_encodings."
                    ),
                    model=model,
                )
            )
            return ComplianceReport(registry=None, issues=issues)
    elif not strict:
        # streaming_output-only + best_effort: reaching gating requires an
        # ``audio`` input, which best_effort would decode and feed to the model
        # (a billable side effect for a cloud engine). Skip with an honest,
        # actionable note rather than driving real inference.
        issues.append(
            ComplianceIssue(
                level="warning",
                code="gating_probe_skipped_billable",
                message=(
                    "skipped streaming gating probe: a streaming_output-only engine "
                    "needs a whole-input 'audio' to reach gating, and a best_effort "
                    "probe would decode it and invoke the model (a billable side "
                    "effect). Run the engine in strict mode to exercise gating "
                    "without inference, or assert gating in a unit test."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)
    else:
        # streaming_output-only + strict: gate_params raises before the audio is
        # decoded or the model touched, so a one-sample silent input is safe.
        open_kwargs["audio"] = np.zeros(1, dtype=np.float32)

    try:
        session = engine.start_transcription(**open_kwargs)  # type: ignore[arg-type]
    except UnsupportedFeatureError as exc:
        if not strict:
            # A best_effort engine MUST drop the unsupported parameter and emit a
            # diagnostic, never raise -- regardless of the exception's ``param``.
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="gating_best_effort_raised",
                    message=(
                        f"best_effort engine raised UnsupportedFeatureError for an "
                        f"unsupported streaming parameter {field_name!r}; it MUST drop "
                        "it and emit a diagnostic instead (spec Runtime R2)."
                    ),
                    model=model,
                )
            )
            return ComplianceReport(registry=None, issues=issues)
        if exc.param != field_name:
            # strict mode: a genuine gating rejection carries param==field_name. A
            # raise with a different (or absent) ``param`` is NOT a gating success
            # -- most often a "declares a streaming axis but never implements the
            # hook" capability lie (the base template raises with param=None), or a
            # wire-format rejection. Either way it is a distinct compliance error,
            # not the clean pass the old code mistook it for.
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="gating_probe_unexpected_unsupported",
                    message=(
                        f"start_transcription raised UnsupportedFeatureError with "
                        f"param={exc.param!r} while probing streaming parameter "
                        f"{field_name!r}; a gating rejection MUST carry "
                        f"param={field_name!r}. An engine that declares streaming but "
                        "does not implement the hook (param=None) is a capability lie, "
                        "not a gating success."
                    ),
                    model=model,
                )
            )
            return ComplianceReport(registry=None, issues=issues)
        # strict + param matches the probed field: correct gating rejection.
        return ComplianceReport(registry=None, issues=issues)
    except Exception as exc:  # noqa: BLE001
        # Any other exception is the engine crashing on the probe, not a gating
        # verdict. Mirror the broad-except guards used elsewhere in this module:
        # record an error and keep the compliance run alive (this function
        # promises ``Raises: None``).
        issues.append(
            ComplianceIssue(
                level="error",
                code="gating_probe_crashed",
                message=(
                    f"start_transcription raised {exc!r} while probing streaming "
                    f"parameter {field_name!r}; the only contractual exception for a "
                    "gated parameter is UnsupportedFeatureError (spec Runtime R2)."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)

    # The session was created but NOT opened: the base start_transcription
    # template constructs the session without entering its context (no
    # __aenter__/_open), and the best_effort verdict below needs only
    # session.diagnostics() -- a pure read of construction-time diagnostics.
    # Entering the session to "close" it would instead OPEN a billable wire
    # handshake the probe never incurred (spec §6.5; cf. the
    # gating_probe_skipped_billable sibling). Tearing down a genuinely opened
    # session is check_sync_bridge's job, not this probe's, so there is no teardown.
    if strict:
        issues.append(
            ComplianceIssue(
                level="error",
                code="gating_strict_accepted",
                message=(
                    f"strict engine accepted streaming parameter {field_name!r} "
                    "violating its declared capabilities without raising; it MUST "
                    "raise UnsupportedFeatureError (spec Runtime R2 gating gap)."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)

    try:
        diagnostics = session.diagnostics()
    except Exception as exc:  # noqa: BLE001
        # session.diagnostics() reads engine-author surface; contain a crash here
        # exactly like the start_transcription / probe-selection containment above
        # -- this function promises ``Raises: None`` and one broken engine must not
        # abort the whole compliance run.
        issues.append(
            ComplianceIssue(
                level="error",
                code="gating_diagnostics_raised",
                message=(
                    f"session.diagnostics() raised {exc!r} while checking for the "
                    f"expected {expected_code!r} diagnostic on best_effort streaming "
                    f"parameter {field_name!r}; diagnostics() must not raise (spec "
                    "Runtime R2)."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)

    if not any(d.code == expected_code for d in diagnostics):
        issues.append(
            ComplianceIssue(
                level="error",
                code="gating_best_effort_swallowed",
                message=(
                    f"best_effort engine silently swallowed streaming parameter "
                    f"{field_name!r}: no {expected_code!r} diagnostic surfaced via "
                    "session.diagnostics() (spec Runtime R2)."
                ),
                model=model,
            )
        )
    return ComplianceReport(registry=None, issues=issues)


class _ForeignProviderParams(ProviderParams):
    """A closed ``provider_params`` type no real engine declares (swap-safety probe).

    Used by :func:`check_provider_params_swap_safety` as the "wrong engine's
    params" a swapped-engine bug would pass. It is closed (``extra="forbid"``) so
    it is itself a valid ``ProviderParams``, and -- being private to this module
    -- it can never coincide with an engine's declared ``provider_params_type``,
    so a compliant engine MUST reject it.
    """

    model_config = ConfigDict(extra="forbid")


def check_provider_params_swap_safety(engine: EngineBase) -> ComplianceReport:
    """Assert an engine always rejects another engine's ``provider_params`` (R3).

    Spec Runtime R3 makes ``provider_params`` swap-safety an unconditional MUST:
    a wrong-typed ``provider_params`` (the classic "switched engines, forgot to
    change the params model" bug) MUST raise
    :class:`~standard_asr.exceptions.InvalidProviderParamError` **independent of
    strict / best_effort** -- it is a code bug, not a capability negotiation, so
    it is never silently dropped. The :class:`EngineBase` template enforces this
    in ``gate_params`` *before* any audio is decoded or the model is touched, so
    an engine that bypassed the template and forgot the check is the gap this
    probe closes -- the same "bypassed the template must show up here" reasoning
    behind :func:`check_streaming_param_gating`.

    The probe calls the engine's public :meth:`~standard_asr.asr_interface.EngineBase.transcribe`
    with a foreign :class:`ProviderParams` subclass private to this module (so it
    can never be the engine's own declared type) and a one-sample silent input.
    Because provider-params validation precedes audio decoding and inference, the
    probe incurs no billable side effect under either policy. The contract is the
    same for a strict and a best_effort engine: it MUST raise
    ``InvalidProviderParamError``.

    Args:
        engine: The engine instance to exercise (any policy).

    Returns:
        A :class:`ComplianceReport`; ``passed`` is ``True`` when the engine raised
        ``InvalidProviderParamError`` for the foreign provider params.

    Raises:
        None.
    """
    issues: list[ComplianceIssue] = []
    model = _safe_engine_id(engine)
    params = RuntimeParams(provider_params=_ForeignProviderParams())
    silence = np.zeros(1, dtype=np.float32)

    try:
        engine.transcribe(silence, params)
    except InvalidProviderParamError:
        # Correct: swapped provider_params rejected before any model work.
        return ComplianceReport(registry=None, issues=issues)
    except ConfigError as exc:
        # The engine raised BEFORE the provider-params gate could run: the base
        # template validates the language config (_validate_language_config) ahead
        # of gate_params, and that method promises ConfigError (it even wraps a
        # malformed-tag ValueError into ConfigError), so a broken language axis
        # surfaces here as ConfigError. R3 swap-safety was therefore never
        # exercised -- this is unverifiable, not a swap miss; attribute it to the
        # real defect rather than mislabel a language_config_invalid engine as
        # swap-unsafe. (A bare ValueError is NOT caught here: a swap rejection
        # using the wrong exception type must still fall through to the broad
        # handler below and be reported as provider_params_swap_not_enforced.)
        issues.append(
            ComplianceIssue(
                level="error",
                code="provider_params_swap_unverifiable",
                message=(
                    f"transcribe raised {exc!r} before the provider_params gate, so "
                    "Runtime R3 swap-safety could not be exercised; resolve the "
                    "engine's language_config_invalid defect first."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)
    except Exception as exc:  # noqa: BLE001
        # Any other exception means the engine did NOT enforce R3 swap-safety on
        # the provider-params-first path (it failed later, for a different reason,
        # or crashed). Report it; never re-raise (this function promises
        # ``Raises: None``).
        issues.append(
            ComplianceIssue(
                level="error",
                code="provider_params_swap_not_enforced",
                message=(
                    f"transcribe raised {exc!r} for a foreign provider_params type "
                    "instead of InvalidProviderParamError; spec Runtime R3 requires "
                    "provider_params swap-safety to raise InvalidProviderParamError "
                    "ALWAYS (independent of strict/best_effort), validated before "
                    "audio decoding."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)

    # No exception at all: the engine silently accepted another engine's params --
    # exactly the swap bug R3 exists to make loud.
    issues.append(
        ComplianceIssue(
            level="error",
            code="provider_params_swap_accepted",
            message=(
                "transcribe accepted a foreign provider_params type without raising; "
                "spec Runtime R3 requires it to raise InvalidProviderParamError "
                "(swap-safety), independent of strict/best_effort."
            ),
            model=model,
        )
    )
    return ComplianceReport(registry=None, issues=issues)


def check_recommended_wire_format(engine: EngineBase) -> ComplianceReport:
    """Assert an engine's recommended wire format is one it would itself accept.

    :meth:`~standard_asr.asr_interface.EngineBase.recommended_wire_format` is the
    single source of truth for the minimal wire :class:`AudioFormat` the standard
    layer opens a ``streaming_input`` session with when the application chose none
    -- the CLI sync-bridge runner and the streaming gating probe both rely on it
    (AW-2). A self-inconsistent engine, whose recommended format its own
    :meth:`~standard_asr.asr_interface.EngineBase.ensure_stream_format_supported`
    rejects, would make those paths fail-loud on a format the standard layer
    chose rather than the application -- a silent-looking compliance trap. This
    closes that loop: when a format is recommended it MUST pass the engine's own
    session-establishment guard.

    Args:
        engine: The engine under test (declares ``streaming_input``).

    Returns:
        A :class:`ComplianceReport`. ``passed`` is ``True`` when no format is
        recommended, or the recommended format is accepted by the engine.

    Raises:
        None.
    """
    issues: list[ComplianceIssue] = []
    try:
        fmt = engine.recommended_wire_format()
    except Exception as exc:  # noqa: BLE001 - reported as a compliance error
        issues.append(
            ComplianceIssue(
                level="error",
                code="recommended_wire_format_raised",
                message=f"EngineBase.recommended_wire_format() raised: {exc!r}.",
                model=None,
            )
        )
        return ComplianceReport(registry=None, issues=issues)
    if fmt is not None:
        try:
            engine.ensure_stream_format_supported(fmt)
        except Exception as exc:  # noqa: BLE001 - reported as a compliance error
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="recommended_wire_format_self_inconsistent",
                    message=(
                        f"recommended_wire_format() returned {fmt!r}, but the engine's "
                        f"own ensure_stream_format_supported rejects it: {exc!r}. The "
                        "recommended format must be one the engine accepts."
                    ),
                    model=None,
                )
            )
    return ComplianceReport(registry=None, issues=issues)


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
        timeout: Seconds to allow the bridged session to drain and close. This
            MUST exceed the adapter's real ``_open`` + ``_close`` cost: a slow but
            compliant adapter (a cloud session doing a real network handshake) is
            *not* a deadlock, so when a run reports a timeout, re-run with a larger
            value to tell "slow" from "stuck". The driver thread is a daemon, so a
            false positive (or a real deadlock) never blocks interpreter exit --
            the process is not held hostage by the fault this check diagnoses.

    Returns:
        A :class:`ComplianceReport`. ``passed`` is ``True`` when the bridge
        terminated cleanly with no leaked background loop thread.

    Raises:
        None.
    """
    issues: list[ComplianceIssue] = []
    outcome: dict[str, object] = {}
    worker_name = "compliance-sync-bridge"

    def _drive() -> None:
        sync: SyncSession | None = None
        try:
            sync = SyncSession(session_factory())
            with sync:
                sync.end_audio()
                events = list(sync)
            outcome["terminal"] = any(getattr(ev, "is_terminal", False) for ev in events)
        except Exception as exc:  # noqa: BLE001 - reported as a compliance error
            outcome["error"] = repr(exc)
        finally:
            # Record the bridge's OWN loop-thread liveness so the leak check below
            # asserts on this thread specifically. A compliant adapter may pull in a
            # dependency that spawns a benign daemon thread (e.g. tqdm's monitor, a
            # thread-pool worker) during the session; a process-wide thread diff
            # would mis-report that as a sync_bridge_thread_leak (CC-2).
            outcome["loop_alive"] = sync.is_loop_alive() if sync is not None else False

    # daemon=True: this thread only *observes* the bridge; the leak check below is
    # responsible for catching a surviving loop thread. If the bridged session
    # genuinely deadlocks, a non-daemon worker would block interpreter shutdown for
    # the full done_timeout (up to 300s), so the process that just reported the
    # deadlock would itself hang on it -- the daemon flag prevents that.
    worker = threading.Thread(target=_drive, name=worker_name, daemon=True)
    worker.start()
    worker.join(timeout=timeout)

    if worker.is_alive():
        issues.append(
            ComplianceIssue(
                level="error",
                code="sync_bridge_did_not_terminate",
                message=(
                    f"SyncSession did not terminate within {timeout}s -- this may be a "
                    "deadlock OR an adapter whose _open/_close legitimately takes "
                    f"longer than {timeout}s. Re-run with a larger timeout to "
                    "disambiguate. If it is a deadlock, check the §6.5 adapter "
                    "contract: bind loop resources in __aenter__, never touch the "
                    "ambient event loop."
                ),
                model=None,
            )
        )
        return ComplianceReport(registry=None, issues=issues)

    if "error" in outcome:
        issues.append(
            ComplianceIssue(
                level="error",
                code="sync_bridge_raised",
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
                code="sync_bridge_no_terminal",
                message="SyncSession ended without emitting a terminal event.",
                model=None,
            )
        )

    # Leak check: the bridge owns a background loop thread that __exit__ (and a
    # failed __enter__) MUST tear down. Assert on the bridge's OWN thread (recorded
    # in _drive via is_loop_alive), not a process-wide thread diff -- a compliant
    # adapter may spawn a benign daemon thread during the session, which a diff
    # would mis-flag as a sync_bridge_thread_leak (CC-2).
    if outcome.get("loop_alive"):
        issues.append(
            ComplianceIssue(
                level="error",
                code="sync_bridge_thread_leak",
                message="SyncSession did not tear down its owned background loop thread on close.",
                model=None,
            )
        )

    return ComplianceReport(registry=None, issues=issues)
