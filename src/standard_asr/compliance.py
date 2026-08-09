# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compliance helpers for Standard ASR plugin authors."""

from __future__ import annotations

import inspect
import math
import threading
from dataclasses import dataclass
from typing import Callable, ClassVar, Iterable, Literal, Protocol, Sequence, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from standard_asr.audio.format import AudioFormat
from standard_asr.contract.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    PromptCap,
    StreamingCapabilities,
    WordTimestampsCap,
)
from standard_asr.contract.exceptions import (
    ConfigError,
    ConfigurationRequiredError,
    EngineContractError,
    EntrypointValidationError,
    InvalidProviderParamError,
    UnsupportedFeatureError,
)
from standard_asr.contract.params import (
    DIARIZE,
    ProviderParams,
    RuntimeParams,
    WordTimestampGranularity,
)
from standard_asr.contract.properties import BaseProperties
from standard_asr.contract.results import Segment, TranscriptionResult, Word
from standard_asr.plugins.discovery import (
    FactoryLoadError,
    ModelRegistry,
    ModelSpec,
    discover_models,
)
from standard_asr.runtime.config import BaseConfig
from standard_asr.runtime.gating import (
    DIAG_PROMPT_TRUNCATED,
    DIAG_UNSUPPORTED_GRANULARITY_IGNORED,
    DIAG_UNSUPPORTED_PARAMETER_IGNORED,
    _count_tokens,  # pyright: ignore[reportPrivateUsage]
)
from standard_asr.runtime.interface import (
    EngineBase,
    StandardASR,
    ensure_wire_format_supported,
)
from standard_asr.runtime.protocol_boundary import (
    safe_class_name,
    safe_type_name,
    sync_result_defect,
)
from standard_asr.runtime.redaction import safe_exception_summary, sanitized_validation_message
from standard_asr.runtime.streaming import (
    SyncSession,
    TranscriptionEvent,
    TranscriptionSession,
    _LifecycleGuard,  # pyright: ignore[reportPrivateUsage]
)

__all__ = [
    "ComplianceIssue",
    "ComplianceReport",
    "DEFAULT_SYNC_BRIDGE_TIMEOUT",
    "SupportsCapabilities",
    "SupportsWireRecommendation",
    "assert_prefix_invariant",
    "check_entrypoints",
    "check_event_sequence",
    "check_provider_params_swap_safety",
    "check_recommended_wire_format",
    "check_streaming_param_gating",
    "check_sync_bridge",
    "check_transcription_result",
    "prepare_requires_arguments",
    "validate_bridge_timeout",
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
    (
        "diarization",
        lambda: RuntimeParams(diarization=DIARIZE),
        "streaming.diarization",
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


def _pick_sub_constraint_probe(engine: StandardASR) -> tuple[str, RuntimeParams, str] | None:
    """Build a probe violating a declared sub-constraint of a supported feature.

    Used when the engine supports every probe in :data:`_GATING_PROBES` at the
    feature level: gating MUST also enforce a supported feature's declared
    *sub-constraints* (a prompt over the declared ``max_tokens`` budget, a
    word-timestamp granularity not in the declared ``granularities``), so the
    check falls back to violating one of those. The
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
        when the engine declares no violable sub-constraint (or exposes no
        readable capability tree to derive one from).
    """
    # ``effective_capabilities`` is an EngineBase convenience, NOT a StandardASR
    # protocol member: a fully-compliant structural engine may omit it, and
    # reading it bare turned that omission into an AttributeError the caller
    # reported as gating_probe_selection_raised -- a false FAILURE of a
    # compliant engine. Fall back to the protocol's ``declared_capabilities``
    # (EngineBase's effective_capabilities defaults to exactly that); with
    # neither readable there is no sub-constraint to derive, which is a no-op
    # pass, not an engine fault. A PRESENT-but-raising attribute still
    # propagates to the caller's containment (a broken surface stays loud).
    capabilities = getattr(engine, "effective_capabilities", None)
    if not isinstance(capabilities, DeclaredCapabilities):
        capabilities = getattr(engine, "declared_capabilities", None)
    if not isinstance(capabilities, DeclaredCapabilities):
        return None
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

    Mirrors the runtime :class:`~standard_asr.contract.results.Diagnostic` shape: every
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

        Returns:
            ``True`` when no error-level issues exist.
        """
        return not any(issue.level == "error" for issue in self.issues)

    def iter_level(self, level: Literal["error", "warning"]) -> Iterable[ComplianceIssue]:
        """Yield issues matching *level*.

        Args:
            level: Severity level to filter.

        Returns:
            Iterable of matching issues.
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
    names: Iterable[str] | None = None,
) -> ComplianceReport:
    """Validate that discovered entry points conform to expectations.

    Environment-level invariants are reported as issues alongside per-engine
    ones -- never as raised exceptions -- so a single command yields one report
    even when discovery itself found problems:

    * **Engine-identity collisions** (an ``engine_id`` provided by more than
      one distribution, making ``config.engine`` routing ambiguous) are reported
      as **errors** here. The discovery layer only *marks* them
      (``registry.shadowed_engine_ids``); the compliance suite is the fail-loud
      layer the standard mandates for them, so a collision is a compliance
      failure even on a default (non-strict) run rather than a log line.
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
            ``False``. Engine-identity collisions are reported as errors regardless.
        instantiate: If ``True``, instantiate zero-arg factories and verify
            the instance surface -- including one BEHAVIORAL probe: on an
            engine declaring no streaming axis, ``start_transcription()`` is
            called once with no arguments and MUST raise
            ``UnsupportedFeatureError`` (a compliant engine refuses at the
            capability gate before constructing anything; a returned session
            is never entered, but a non-compliant implementation may still
            run arbitrary author code in the method body). ``False`` skips
            instantiation and the probe with it.
        names: Restrict the PER-ENGINE checks to these model keys; ``None``
            checks every discovered engine. The registry-global invariants
            (RuntimeParams closedness, engine-identity collisions,
            no-entry-points) always evaluate the whole environment -- they
            are environment facts, not per-engine verdicts. Pass the user's
            named subset here rather than filtering the report afterwards:
            the instance checks EXECUTE engine code (construction, a
            ``supports()`` sweep, the ``start_transcription()`` refusal
            probe -- a model load, for a cloud engine potentially a billable
            call), a side effect that must not be paid on a co-installed
            plugin the caller never named, for a verdict they are never
            shown.

    Returns:
        Compliance report summarizing findings.
    """
    issues: list[ComplianceIssue] = []

    if registry is None:
        try:
            registry = discover_models(strict=strict_discovery)
        except EntrypointValidationError as exc:
            # strict discovery raises on invalid entry-point names (and
            # engine-identity collisions). A compliance check MUST always return
            # a report, so convert the failure to an error issue and re-discover
            # leniently to still check the valid engines. Identity collisions are
            # additionally reported per-engine_id below from ``shadowed_engine_ids``.
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="entrypoint_invalid",
                    message=(
                        "Strict discovery rejected one or more entry points: "
                        f"{safe_exception_summary(exc)}"
                    ),
                    model=None,
                )
            )
            registry = discover_models(strict=False)

    # RuntimeParams MUST be a closed type. This is a global invariant of
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

    # An engine_id contributed by more than one distribution makes
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
                    "distribution (an identity collision); config.engine routing "
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

    # The per-engine loop honors the caller's named scope; an unknown name is
    # the caller's own lookup error to surface (the CLI resolves its names
    # against this same registry), not silently "checked".
    named = None if names is None else set(names)
    selected = [n for n in registry.names() if named is None or n in named]
    for name in selected:
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
                    f"Checking engine {name!r} raised "
                    f"{safe_exception_summary(exc)}; an engine's public "
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

    # Declared metadata MUST be readable from the class without
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
    except ConfigurationRequiredError as exc:
        # A credentialed engine's zero-arg factory raises this when the required
        # credential is absent (explicit config > env > raise; from_env raises
        # the narrow subtype automatically). On a clean CI with
        # no env vars set this is the *correct* behavior, so it MUST NOT be a
        # compliance error -- otherwise the verdict would depend on the runtime's
        # credential state rather than the plugin. Report it as a warning skip and
        # point at the env var; pass --no-instantiate or set the credential to run
        # the full instance-level checks.
        # `exc` is typically from_env's sanitized ConfigurationRequiredError,
        # but an engine building config another way raises its own -- embed
        # through the total safe renderer, never repr().
        issues.append(
            ComplianceIssue(
                level="warning",
                code="factory_requires_config",
                message=(
                    "Skipped instantiation: the factory requires configuration not "
                    f"present in this environment ({safe_exception_summary(exc)}). "
                    "Set the engine's "
                    "STANDARD_ASR_<ENGINE>__<FIELD> environment variable (double "
                    "underscore between engine and field, per env_var_name; e.g. an "
                    "API key) or pass an explicit config to run the full instance "
                    "checks."
                ),
                model=name,
            )
        )
        return
    except (ConfigError, ValidationError) as exc:
        # Any OTHER config/validation failure is a defect, not a missing
        # credential: an invalid supplied value, an internally inconsistent
        # declaration, or a factory building a broken internal model. Waiving
        # these as "requires config" let a broken plugin read as
        # green-with-warning -- the skip is reserved for the narrow
        # ConfigurationRequiredError (absence), which from_env raises
        # automatically; an engine building config another way must raise it
        # itself for the missing-credential state.
        #
        # ONE total boundary for the embedded text: a raw ValidationError
        # reaches here un-wrapped (the factory is called directly, not through
        # ModelRegistry.create's sanitizing wrap) and its repr echoes the
        # offending input_value; an engine-authored ConfigError may have
        # interpolated a chained ValidationError's echo into its own message
        # (raise ConfigError(f"bad: {ve}") from ve), which repr() re-leaks
        # and a hostile __repr__ turns into a second crash site.
        # safe_exception_summary handles all of it: sanitized loc/msg for the
        # ValidationError, marked sanitized wrappers kept, everything else
        # withheld, total under hostile __str__/__repr__.
        defect = (
            sanitized_validation_message(exc, prefix="ValidationError")
            if isinstance(exc, ValidationError)
            else safe_exception_summary(exc)
        )
        issues.append(
            ComplianceIssue(
                level="error",
                code="factory_config_invalid",
                message=(
                    f"Factory invocation failed with a configuration/validation "
                    f"defect ({defect}). If this state is actually 'required "
                    "configuration absent from the environment' (e.g. a missing "
                    "credential), raise ConfigurationRequiredError instead "
                    "(BaseConfig.from_env does so automatically); compliance "
                    "skips that state rather than failing it."
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
                message=f"Factory invocation failed with {safe_exception_summary(exc)}.",
                model=name,
            )
        )
        return

    _check_required_surface(instance, name, issues)
    _check_prepare_hook(instance, name, issues)
    _check_instance_properties(instance, spec, name, issues)
    _check_instance_config(instance, name, issues)
    _check_instance_capabilities(instance, name, issues)
    _check_supports_contract(instance, name, issues)
    _check_instance_wire_format(instance, name, issues)


def _check_instance_wire_format(
    instance: object,
    name: str,
    issues: list[ComplianceIssue],
) -> None:
    """Round-trip ``recommended_wire_format()`` for EVERY constructed engine.

    The protocol member is unconditionally required (spec §3.1: the
    recommendation is Properties-pure and capability-blind), so its
    self-consistency round-trip holds for every engine — batch-only included.
    Gating it on a streaming axis (as the CLI once did) let a batch-only
    engine ship a raising, wrong-typed, or self-inconsistent implementation
    that every consumer of the member would then trip over. Runs here, at the
    entrypoint layer, so one ``compliance run`` exercises it exactly once per
    engine.

    Args:
        instance: The instantiated engine.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    member = getattr(instance, "recommended_wire_format", None)
    if not callable(member) or inspect.iscoroutinefunction(member):
        # Absence and the `async def` modality are already reported by the
        # surface checks; calling here would crash redundantly or manufacture
        # the very coroutine the modality check exists to prevent.
        return
    if not isinstance(getattr(instance, "properties", None), BaseProperties):
        # The round-trip validates the format against the engine's Properties;
        # a missing/invalid Properties is already reported by the properties
        # checks, and running the round-trip against it would only add noise.
        return
    # Runtime-verified above (callable member + typed properties); the cast
    # only names what was just checked.
    engine = cast(SupportsWireRecommendation, instance)
    issues.extend(_wire_format_round_trip_issues(engine, model=name))


def _check_supports_contract(
    instance: object,
    name: str,
    issues: list[ComplianceIssue],
) -> None:
    """Verify ``supports()`` semantics: shape, fail-closed unknowns, tree agreement.

    Every capability negotiation in the ecosystem consumes ``supports()``,
    so both a wrong return SHAPE (a truthy non-bool reads as "supported"
    everywhere; an awaitable is truthy AND a leaked coroutine) and a wrong
    ANSWER (a hand-written ``supports()`` diverging from the capability tree
    it is defined to query -- spec R5) are silent wrong capability verdicts.
    Three layers, cheapest first, each pure metadata (no session is opened):

    1. **Shape probe** (the canonical ``streaming_input`` path): synchronous,
       real ``bool``. A broken shape stops here -- sweeping a malformed
       ``supports()`` over the whole tree would flood one defect into dozens
       of issues.
    2. **Unknown path fail-closed** (spec R5: a missing path returns
       ``False``, without raising): a sentinel path guaranteed to name no
       real capability MUST answer literal ``False``
       (``supports_not_fail_closed``).
    3. **Equivalence sweep**: for every queryable node path
       (:meth:`~standard_asr.contract.capabilities.DeclaredCapabilities.iter_queryable_paths`
       -- supported nodes, unsupported nodes, containers, constraint
       submodels, ``x_*`` subtrees) the answer MUST be identical to the
       engine's own capability tree's answer. The baseline is
       ``effective_capabilities`` when it is a valid tree (what
       ``EngineBase.supports`` itself queries and what R5's "current
       usability" means), else ``declared_capabilities``, else the sweep is
       skipped (an invalid tree is already reported by the capabilities
       checks). Mismatches aggregate into ONE
       ``supports_disagrees_with_capabilities`` issue naming the first few
       paths and the totals -- a systematically wrong implementation must not
       flood the report.

    Args:
        instance: The instantiated engine to probe.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    supports = getattr(instance, "supports", None)
    if not callable(supports) or inspect.iscoroutinefunction(supports):
        # Absence and the `async def` modality are already reported by the
        # surface checks; calling here would crash redundantly or manufacture
        # the very coroutine the modality check exists to prevent.
        return
    try:
        value = supports("streaming_input")
    except Exception as exc:  # noqa: BLE001 - contained per-engine, run continues
        issues.append(
            ComplianceIssue(
                level="error",
                code="supports_raised",
                message=(
                    f"supports('streaming_input') raised "
                    f"{safe_exception_summary(exc)}; the capability "
                    "surface must answer a dot-path query without raising "
                    "(consumers fail closed on it, so a raising supports() reads "
                    "as 'nothing supported')."
                ),
                model=name,
            )
        )
        return
    if _sync_member_violation(value, "supports()", name, issues, expected_type=bool):
        return
    _check_supports_unknown_path(supports, name, issues)
    _check_supports_tree_agreement(instance, supports, name, issues)


#: A dot-path guaranteed to name no real capability: its first segment is not a
#: capability-tree field and (not being ``x_*``-prefixed) can never resolve into
#: the extension namespace either, so the fail-closed contract (spec R5) pins
#: the answer to a literal ``False`` for every compliant engine.
_SUPPORTS_UNKNOWN_PROBE = "standard_asr_compliance.nonexistent_capability_probe"

#: Cap on the mismatch paths named inline by ``supports_disagrees_with_capabilities``
#: (the totals always report the full extent).
_SUPPORTS_MISMATCH_DISPLAY_LIMIT = 10


def _check_supports_unknown_path(
    supports: Callable[[str], object],
    name: str,
    issues: list[ComplianceIssue],
) -> None:
    """Probe an unknown capability path: the answer MUST be literal ``False``.

    The reported description never embeds an arbitrary return value's
    ``repr`` (the sync-call boundary's own rule -- an engine-fabricated
    object could smuggle payload text into the report): a ``bool`` shows its
    value, anything else shows its type only.

    Args:
        supports: The engine's (shape-verified) ``supports`` callable.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    try:
        answer = supports(_SUPPORTS_UNKNOWN_PROBE)
    except Exception as exc:  # noqa: BLE001 - contained per-engine, run continues
        described = f"raised {safe_type_name(exc)}"
    else:
        if answer is False:
            return
        defect = sync_result_defect(answer)
        if defect is not None:
            # A stray coroutine has been closed by the boundary; report the
            # modality honestly rather than repr-ing a dead coroutine object.
            described = (
                "answered an awaitable"
                if defect.kind == "awaitable"
                else f"returned a result the sync boundary could not classify ({defect.clause})"
            )
        elif answer is True:
            described = "answered True"
        else:
            described = f"answered a {safe_type_name(answer)} (value withheld)"
    issues.append(
        ComplianceIssue(
            level="error",
            code="supports_not_fail_closed",
            message=(
                f"supports() {described} for an unknown capability path; "
                "the capability model is fail-closed (spec R5): a path that "
                "does not exist in the tree MUST answer literal False, without "
                "raising. Anything else makes applications negotiate features "
                "the engine never declared."
            ),
            model=name,
        )
    )


def _check_supports_tree_agreement(
    instance: object,
    supports: Callable[[str], object],
    name: str,
    issues: list[ComplianceIssue],
) -> None:
    """Sweep every queryable path: ``supports()`` MUST agree with the tree.

    Args:
        instance: The instantiated engine (for the baseline trees).
        supports: The engine's (shape-verified) ``supports`` callable.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    baseline = _supports_baseline_tree(instance)
    if baseline is None:
        # An absent/invalid capability tree is already reported by the
        # capabilities checks; sweeping against it would only manufacture a
        # cascade of noise on top of the real defect.
        return
    mismatches: list[str] = []
    total = 0
    for path in baseline.iter_queryable_paths():
        total += 1
        expected = baseline.supports(path)
        try:
            actual = supports(path)
        except Exception as exc:  # noqa: BLE001 - contained per-engine, run continues
            mismatches.append(f"{path} (raised {safe_type_name(exc)})")
            continue
        defect = sync_result_defect(actual, expected_type=bool)
        if defect is not None:
            mismatches.append(f"{path} ({defect})")
            continue
        if actual is not expected:
            mismatches.append(f"{path} (answered {actual!r}, tree says {expected!r})")
    if not mismatches:
        return
    shown = mismatches[:_SUPPORTS_MISMATCH_DISPLAY_LIMIT]
    overflow = len(mismatches) - len(shown)
    suffix = f"; ... and {overflow} more" if overflow else ""
    issues.append(
        ComplianceIssue(
            level="error",
            code="supports_disagrees_with_capabilities",
            message=(
                f"supports() disagrees with the engine's capability tree on "
                f"{len(mismatches)} of {total} queryable paths: "
                f"{'; '.join(shown)}{suffix}. supports() is defined as a direct "
                "query of the effective capability tree (spec R5; "
                "EngineBase.supports IS effective_capabilities.supports) -- "
                "every capability negotiation reads these answers, so a "
                "divergence is a silent wrong verdict on every consumer."
            ),
            model=name,
        )
    )


def _supports_baseline_tree(instance: object) -> DeclaredCapabilities | None:
    """Pick the tree ``supports()`` is expected to answer from, defensively.

    ``effective_capabilities`` (an ``EngineBase`` convenience, not a protocol
    member) wins when present and valid -- it is what ``EngineBase.supports``
    queries and what spec R5's "current usability" means; a structural engine
    without it falls back to the protocol's ``declared_capabilities`` (what
    ``EngineBase`` defaults effective to). Anything invalid yields ``None``
    (the capability checks own reporting that).

    Args:
        instance: The instantiated engine.

    Returns:
        The baseline tree, or ``None`` when no valid tree is reachable.
    """
    for attr in ("effective_capabilities", "declared_capabilities"):
        try:
            tree = getattr(instance, attr, None)
        except Exception:  # noqa: BLE001 - a raising convenience property
            continue
        if isinstance(tree, DeclaredCapabilities):
            return tree
    return None


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
        # str(exc) echoes the offending input_value; the message lands in
        # terminals/CI logs, so render the sanitized loc/msg summary instead.
        issues.append(
            ComplianceIssue(
                level="error",
                code="properties_revalidation_failed",
                message=sanitized_validation_message(
                    exc,
                    prefix="Instance properties fail re-validation",
                ),
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
                    f"config_type ({safe_type_name(cast('object', config))!r} is not a "
                    f"{safe_class_name(cast('type', declared_config_type))!r}); the schema "
                    "published for UIs would not match the config actually consumed."
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
                    f"Reading effective_capabilities raised {safe_exception_summary(exc)}; the "
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
                    f"(got {safe_type_name(effective)!r}); it MUST be a "
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
    valid ``default_language`` passes construction (the standard keeps ``__init__``
    pure) and then raises ``ConfigError`` on the **user's first transcribe** --
    the worst place for an engine-author bug to surface. Catch it at compliance
    time instead. For :class:`EngineBase` engines this reuses the exact runtime
    validation (presence, selectable-membership, canonicalization), so the
    compliance verdict cannot drift from runtime behavior; for structural
    engines it falls back to the presence check.

    Args:
        instance: The instantiated engine to inspect.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    if isinstance(instance, EngineBase):
        try:
            instance._validate_language_config()  # pyright: ignore[reportPrivateUsage]
        except (ConfigError, EngineContractError, ValueError) as exc:
            # EngineContractError covers the DECLARATION side of the same
            # runtime validation (a missing IC.6 default, a malformed
            # declared tag); ConfigError/ValueError the value side. A raw
            # ValidationError IS a ValueError and its str() echoes the
            # offending input; scrub it before the message reaches CI logs.
            detail = (
                sanitized_validation_message(exc, prefix="ValidationError")
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="language_config_invalid",
                    message=f"Language config is invalid; every transcribe will fail: {detail}",
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
                    "will raise ConfigError."
                ),
                model=name,
            )
        )


#: Public callables every compliant engine MUST expose unconditionally
#: (StandardASR protocol -- the COMPLETE public surface, batch-only included).
_ALWAYS_REQUIRED_METHODS: tuple[str, ...] = (
    "transcribe",
    "transcribe_async",
    # ALWAYS present per the protocol ("start_transcription is always
    # present; a batch-only engine raises UnsupportedFeatureError from it"):
    # the protocol's whole point is that callers type an engine as StandardASR
    # and call the streaming entry point without a cast or hasattr probe. A
    # batch-only engine that OMITS the method hands those callers an
    # AttributeError instead of the standardized fail-closed rejection --
    # certifying that shape would let compliance pass an object that does not
    # satisfy the very protocol it certifies.
    "start_transcription",
    "supports",
    # Part of the StandardASR protocol: the documented first step of the
    # streaming journey, derivable from Properties (EngineBase provides it for
    # free; a structural engine must implement it).
    "recommended_wire_format",
)


def _check_required_surface(
    instance: object,
    name: str,
    issues: list[ComplianceIssue],
) -> None:
    """Verify the engine exposes the full required public surface.

    Every engine MUST expose the unconditional surface pinned by
    :data:`_ALWAYS_REQUIRED_METHODS` -- :meth:`transcribe`,
    :meth:`transcribe_async`, :meth:`start_transcription` (ALWAYS present per
    the ``StandardASR`` protocol; a batch-only engine raises
    ``UnsupportedFeatureError`` from it rather than omitting it, so protocol-
    typed callers never hit an ``AttributeError``), :meth:`supports`, and
    :meth:`recommended_wire_format` (derivable from Properties even for
    batch-only engines); a missing member is a compliance **error**, not a
    silent accept. The ``properties``/``declared_capabilities`` attributes are
    verified by the caller's type checks; this helper covers the callable
    methods plus the streaming-declaration consistency check below.

    For an :class:`EngineBase` engine the streaming requirement uses the same
    :meth:`~standard_asr.runtime.interface.EngineBase._overrides_streaming` predicate
    the runtime template uses to decide whether streaming is *actually*
    implemented (so compliance shares the runtime's validation logic): the
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
        attr = getattr(instance, method, None)
        if not callable(attr):
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
        elif method != "transcribe_async" and inspect.iscoroutinefunction(attr):
            # Modality is part of the surface: every member except
            # transcribe_async is SYNCHRONOUS (async behavior lives in
            # transcribe_async and inside the returned session). An
            # `async def` implementation hands protocol-typed callers a
            # coroutine where a result is pinned -- and every behavioral
            # probe would otherwise manufacture never-awaited coroutines
            # (RuntimeWarnings under warnings-as-errors) exercising it.
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="protocol_member_not_synchronous",
                    message=(
                        f"{method!r} is an `async def`; the StandardASR "
                        "protocol pins it as a SYNCHRONOUS member (async "
                        "behavior lives in transcribe_async and inside the "
                        "returned session)."
                    ),
                    model=name,
                )
            )

    # Presence is unconditional (checked above for every engine); what remains
    # is the streaming-declaration CONSISTENCY check. Read the declared axes
    # defensively: a malformed ``declared_capabilities`` (its own error is
    # raised elsewhere) simply means we cannot assert anything here, so we do
    # not over-report.
    declared = getattr(instance, "declared_capabilities", None)
    declares_streaming = isinstance(declared, DeclaredCapabilities) and (
        declared.supports("streaming_input") or declared.supports("streaming_output")
    )
    if not declares_streaming:
        if isinstance(declared, DeclaredCapabilities):
            # Presence alone does not verify the protocol's batch-only
            # promise; the refusal probe below does. Skipped when
            # declared_capabilities is unreadable (we cannot know the engine
            # is batch-only, and its own error is reported elsewhere).
            _check_batch_only_streaming_refusal(instance, name, issues)
        return
    if isinstance(instance, EngineBase) and not instance._overrides_streaming():  # pyright: ignore[reportPrivateUsage]
        # The base template always provides start_transcription, so presence is
        # not enough for a streaming-DECLARING EngineBase engine: it must
        # override the _start_transcription hook, or the runtime raises
        # UnsupportedFeatureError at session establishment.
        issues.append(
            ComplianceIssue(
                level="error",
                code="streaming_declared_not_implemented",
                message=(
                    "Instance declares a streaming axis (streaming_input / "
                    "streaming_output) but does not implement the streaming hook "
                    "(_start_transcription); start_transcription would raise "
                    "UnsupportedFeatureError at runtime (fail-closed: a declared "
                    "capability is a promise)."
                ),
                model=name,
            )
        )


def _sync_member_violation(
    value: object,
    member: str,
    model: str | None,
    issues: list[ComplianceIssue],
    *,
    expected_type: type | tuple[type, ...] | None = None,
) -> bool:
    """Contain and report a sync protocol member's wrong-shaped return value.

    THE single guard every behavioral probe applies after calling a
    SYNCHRONOUS ``StandardASR`` member (``transcribe`` /
    ``start_transcription`` / ``supports`` / ``recommended_wire_format`` --
    async behavior lives in ``transcribe_async`` and inside the returned
    session). Two defect shapes are contained:

    * **Awaitable**: an ``async def`` implementation (or a sync wrapper
      delegating to one) hands back an awaitable that (a) is the wrong
      result type for every consumer and (b) becomes a never-awaited
      coroutine polluting the run with a ``RuntimeWarning`` under
      warnings-as-errors. A bare coroutine is CLOSED so nothing leaks;
      reported as ``protocol_member_not_synchronous``.
    * **Wrong type** (when ``expected_type`` is given): a value outside the
      member's pinned return type -- the canonical case is ``supports()``
      returning a truthy non-bool (``"false"``, an object), which every
      truthiness-based consumer would silently misread as "supported", a
      wrong capability negotiation. Reported as
      ``protocol_member_wrong_return_type``. The check is strict
      ``isinstance`` (a ``numpy.bool_`` is NOT a ``bool``): the protocol
      pins the type, so quacking is not compliance -- return a real ``bool``.
    * **Unclassifiable**: the boundary's own classification introspection
      raised against the value's type metadata (a hostile metaclass, a
      broken ``__class__`` property). No consumer can safely classify such
      a result, so that IS the defect -- reported as
      ``protocol_member_unclassifiable_result`` with the boundary's honest
      clause, never by re-inspecting the value here.

    Any defect gets one stable code instead of masquerading as whatever
    verdict the probe would have drawn from the malformed value.

    Args:
        value: The member's return value.
        member: Display name of the member (for the message).
        model: The model key to attribute the issue to, or ``None``.
        issues: The mutable issue list to append to.
        expected_type: The member's pinned return type(s), or ``None`` to
            check only synchronicity.

    Returns:
        ``True`` when the value violated the contract (reported and, for a
        coroutine, closed); ``False`` for a normal conforming result.
    """
    defect = sync_result_defect(value, expected_type=expected_type)
    if defect is None:
        return False
    # The VERDICT selects the issue code; the value is never re-inspected
    # here (its metadata already trained containment once).
    if defect.kind == "awaitable":
        issues.append(
            ComplianceIssue(
                level="error",
                code="protocol_member_not_synchronous",
                message=(
                    f"{member} returned an awaitable; the StandardASR protocol pins "
                    "it as a SYNCHRONOUS member (async behavior lives in "
                    "transcribe_async and inside the returned session), so "
                    "protocol-typed callers can never await its result."
                ),
                model=model,
            )
        )
        return True
    if defect.kind == "unclassifiable":
        issues.append(
            ComplianceIssue(
                level="error",
                code="protocol_member_unclassifiable_result",
                message=(
                    f"{member} {defect}; the sync-return boundary itself could not "
                    "reach a verdict because the value's own type metadata raised "
                    "under inspection. A result no consumer can safely classify "
                    "violates the contract outright (fail-closed), independent of "
                    "what the value might have been."
                ),
                model=model,
            )
        )
        return True
    issues.append(
        ComplianceIssue(
            level="error",
            code="protocol_member_wrong_return_type",
            message=(
                f"{member} returned {safe_type_name(value)!r}, not the "
                "protocol-pinned return type; consumers negotiating "
                "capabilities on truthiness would silently misread it "
                "(e.g. a non-empty string reads as 'supported'). Return "
                "a real value of the pinned type (for supports(): a bool)."
            ),
            model=model,
        )
    )
    return True


def _check_batch_only_streaming_refusal(
    instance: object,
    name: str,
    issues: list[ComplianceIssue],
) -> None:
    """Verify a batch-only engine REFUSES ``start_transcription()`` correctly.

    The protocol's batch-only promise is behavioral, not just structural:
    ``start_transcription`` is always present AND a batch-only engine raises
    ``UnsupportedFeatureError`` from it (fail-closed: an undeclared capability
    is not supported, spec Capabilities R1). Checking presence alone would
    certify an engine that silently ACCEPTS a streaming session it never
    declared -- the inverse capability lie -- or one that hands protocol-typed
    callers a non-standard exception where the contract pins the type.

    Side-effect envelope: same as the streaming gating check -- the no-arg
    call may CONSTRUCT a session on a non-compliant engine but never enters
    it, so the standard layer opens no wire connection. A compliant engine
    raises at the capability gate before any construction.

    Args:
        instance: The instantiated engine, whose ``declared_capabilities``
            declare no streaming axis.
        name: The model key (for issue attribution).
        issues: The mutable list of issues to append to.
    """
    method = getattr(instance, "start_transcription", None)
    if not callable(method):
        # Absence is already reported by the unconditional surface check;
        # probing a non-callable would just crash with a redundant TypeError.
        return
    if inspect.iscoroutinefunction(method):
        # Already reported by the surface modality check
        # (protocol_member_not_synchronous); calling an `async def` here
        # would only manufacture a never-awaited coroutine on top of it.
        return
    try:
        session = method()
    except UnsupportedFeatureError:
        return
    except Exception as exc:  # noqa: BLE001 - reported, never re-raised
        issues.append(
            ComplianceIssue(
                level="error",
                code="batch_only_streaming_refusal_wrong_error",
                message=(
                    f"start_transcription() on a batch-only engine raised "
                    f"{safe_exception_summary(exc)}; "
                    "the StandardASR protocol pins the refusal type: a batch-only "
                    "engine MUST raise UnsupportedFeatureError so protocol-typed "
                    "callers can rely on one standardized fail-closed rejection."
                ),
                model=name,
            )
        )
        return
    if _sync_member_violation(session, "start_transcription()", name, issues):
        # A SYNC wrapper returning an awaitable (e.g. delegating to an
        # internal `async def`) slips the iscoroutinefunction pre-checks; the
        # shared guard closed the stray coroutine and reported the modality
        # defect -- it must not additionally read as "returned a session".
        return
    issues.append(
        ComplianceIssue(
            level="error",
            code="batch_only_streaming_not_refused",
            message=(
                "start_transcription() on a batch-only engine returned "
                f"{safe_type_name(session)!r} instead of raising "
                "UnsupportedFeatureError. Accepting a streaming session while "
                "declaring no streaming axis is a capability lie in reverse "
                "(undeclared-but-implemented); declare the axis or refuse the "
                "call fail-closed."
            ),
            model=name,
        )
    )


def prepare_requires_arguments(prepare: Callable[..., object]) -> bool:
    """Return whether a ``prepare()`` warm-up hook needs caller-supplied arguments.

    A warm-up hook MUST be invocable with no arguments. A parameter
    makes the hook non-conforming only when it is *required*: it has no default
    and is positional-or-keyword, positional-only, or keyword-only. ``*args`` and
    ``**kwargs`` impose no required argument, and a bound method's ``self`` is
    already supplied, so neither counts.

    This is the single definition of the zero-argument half of the contract,
    shared by :func:`_check_prepare_hook` and the ``standard-asr prepare``
    CLI command so the compliance verdict and the runtime behavior cannot drift.

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
    """Verify the optional ``prepare()`` warm-up hook honors its contract.

    ``prepare()`` is optional, but when present (overridden past the
    :class:`~standard_asr.runtime.interface.EngineBase` no-op) it MUST be a
    **synchronous, zero-argument** method. A coroutine ``prepare``
    is the dangerous case: ``standard-asr prepare`` would call it, get an
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
                    "synchronous zero-argument method -- an async prepare() would be "
                    "reported complete without ever warming up."
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
                    "with no arguments."
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
    :meth:`~standard_asr.runtime.interface.EngineBase.ensure_stream_format_supported`
    matches a declared ``audio_format`` against; when it is ``None`` the encoding
    check is skipped (``None`` means "unconstrained"), so an engine that
    actually frames PCM but forgets to declare it would read a ``mulaw``
    ``audio_format`` session's frames as PCM -- a silent mistranscription (the
    cardinal sin). An engine that declares ``streaming_input`` can be opened with
    an explicit ``audio_format``, so the omission is reported here as a **warning**
    (a DX nudge, like the missing ``config_type`` one) rather than an error: a
    bare-call engine that self-manages its wire format legitimately
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
                    "(e.g. ['pcm_s16le']) unless the engine self-manages its wire "
                    "format via bare start_transcription()."
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
    (:meth:`~standard_asr.contract.capabilities.DeclaredCapabilities.\
_require_streaming_domain_for_streaming_flags`); this closes the asymmetry on the
    silent side.

    Unlike the ``wire_encodings`` nudge -- which a self-managing-wire engine may
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
    """Verify class-level metadata is readable without instantiation.

    Reads ``declared_capabilities`` and ``provider_params_type`` from the engine
    *class* (never the instance) and validates that, when present, the
    provider-params type is a closed :class:`ProviderParams` subclass.

    Args:
        spec: The :class:`~standard_asr.plugins.discovery.ModelSpec`.
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
                    f"instantiation: {safe_exception_summary(exc)}"
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
        # settings UI cannot discover the engine's config schema --
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
                    f"{safe_class_name(params_type)} is not."
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
                    "no fields and admits any params -- this zeroes swap-safety. "
                    "Publish a distinct terminal ProviderParams subclass as the "
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
    structural invariants cannot see. The stream MUST NOT *exceed* what the
    capabilities promise:

    * ``word_stability`` unsupported ⇒ no event may carry a meaningful
      ``stable_until`` (> 0): the field asserts a frozen prefix the engine declared
      it does not provide.
    * streaming ``timestamps`` mode ``none`` ⇒ no event may carry
      ``audio_processed_until``: that cursor is a streaming timestamp the engine
      declared it does not emit.
    * ``word_timestamps`` unsupported ⇒ no event may carry ``words``: per-word
      timings are word timestamps the engine declared it does not produce.
    * ``diarization`` unsupported ⇒ no event may carry a speaker label (its own
      ``speaker`` or any ``words[i].speaker``): attribution is diarization
      output the engine declared it does not produce. This negative check is
      the only diarization coverage the suite can offer -- *positive*
      diarization behavior (correct labels) is unverifiable without
      multi-speaker fixtures, and the standard probes feed silence.

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
    if (
        event.speaker is not None or any(word.speaker is not None for word in event.words or [])
    ) and not streaming.diarization.is_supported:
        issues.append(
            ComplianceIssue(
                level="error",
                code="stream_exceeds_diarization",
                message=(
                    "event emits a speaker label but the engine declares "
                    "streaming.diarization unsupported -- the declared capabilities and "
                    "the emitted stream disagree. Declare diarization supported "
                    "(always_on if architecturally non-disableable), or do not emit "
                    "speaker labels."
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

    Behavioral check for streaming engines that is **pure**: it replays an
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
    full ``supersede`` invariants -- frozen-prefix preservation
    across a replacement (the concatenated frozen text of ``old_ids`` MUST be
    preserved by ``new_ids``), ``old_ids`` that were never announced, a
    ``new_id`` that reintroduces an already-known segment, and an empty
    ``new_ids`` (pure deletion) that would destroy frozen text -- an event stream
    that never reaches a terminal (``done`` / non-recoverable ``error``) event,
    **an empty sequence** (unless ``allow_empty=True``), and **any event emitted
    after the session-terminal** event (a terminal MUST be the last event).

    The per-segment lifecycle / frozen-prefix / supersede checks are obtained by
    replaying the events through the same
    :class:`~standard_asr.runtime.streaming._LifecycleGuard` the runtime uses, so the
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
            the engine's declared streaming capabilities: a stream MUST NOT
            *exceed* what it declares -- e.g. emit a non-zero ``stable_until`` while
            ``word_stability`` is unsupported, an ``audio_processed_until`` cursor
            while ``timestamps`` mode is ``none``, ``words`` while
            ``word_timestamps`` is unsupported, or a speaker label (event- or
            word-level) while ``diarization`` is unsupported. Pass
            ``engine.declared_capabilities`` to catch a declaration that disagrees
            with the engine's actual output. ``None`` skips the cross-check.

    Returns:
        A :class:`ComplianceReport`; ``passed`` is ``True`` when the sequence
        honors every streaming invariant.
    """
    guard = _LifecycleGuard(strict=False)
    issues: list[ComplianceIssue] = []
    saw_any = False
    saw_terminal = False
    # The streaming sub-domain to cross-check events against; ``None`` when
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
                        "(a terminal done / non-recoverable error MUST be the last "
                        "event)."
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
    # direction of the supersede rule, so it is a soft WARNING -- it does NOT fail the
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
                    "error) event (the stream MUST terminate)."
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
    """Assert a recorded stream's partials honor the frozen-prefix invariant.

    Test helper for engine authors. Partials are **lossy under backpressure**: the
    base coalesces pending partials when the consumer is slow, so the
    partial *count* is non-deterministic -- the same engine may surface five
    partials or none purely by consumer timing. Asserting a count is therefore
    flaky; assert the **invariant** instead. This checks only the prefix invariant
    -- a segment's frozen prefix (``text[:stable_until]``) is never rewritten and
    ``stable_until`` never regresses -- across however many partials survived
    coalescing, and (unlike :func:`check_event_sequence`) does NOT require a
    terminal event, so it also applies to a mid-stream slice. It replays events
    through the same runtime :class:`~standard_asr.runtime.streaming._LifecycleGuard` the
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


def _carries_speaker_labels(
    segments: Sequence[Segment] | None, words: Sequence[Word] | None
) -> bool:
    """Return whether any segment or word in one result view carries a speaker.

    One "view" is a ``(segments, words)`` pair -- either the top level of a
    :class:`~standard_asr.contract.results.TranscriptionResult` or a single
    ``channels[i]`` entry; :func:`_cross_check_result_capabilities` walks every
    view because within-result consistency spans them all (a label
    hidden under ``channels`` alone is still diarization output).

    Args:
        segments: The view's segments, or ``None``.
        words: The view's flattened words, or ``None``.

    Returns:
        ``True`` when any ``Segment.speaker``, ``Segment.words[i].speaker``, or
        flattened ``Word.speaker`` is non-``None``.
    """
    for segment in segments or []:
        if segment.speaker is not None:
            return True
        if any(word.speaker is not None for word in segment.words or []):
            return True
    return any(word.speaker is not None for word in words or [])


def _cross_check_result_capabilities(
    result: TranscriptionResult,
    batch: BatchCapabilities | None,
    issues: list[ComplianceIssue],
) -> None:
    """Cross-check a batch result's speaker labels against the declared capabilities.

    The batch twin of the streaming ``stream_exceeds_diarization`` check in
    :func:`_cross_check_event_capabilities`: a result MUST NOT carry speaker
    labels (anywhere -- top-level ``segments[]`` / ``words[]`` or any
    ``channels[i]`` view) when ``batch.diarization`` is declared unsupported. A
    missing ``batch`` domain is the same verdict (fail-closed: no declaration
    is no support). One-directional like its streaming sibling: a supporting
    engine whose recorded result happens to carry no labels is not a violation.

    Args:
        result: The recorded result to check.
        batch: The engine's declared batch capabilities, or ``None``.
        issues: The mutable list of issues to append to.
    """
    if batch is not None and batch.diarization.is_supported:
        return
    if _carries_speaker_labels(result.segments, result.words) or any(
        _carries_speaker_labels(channel.segments, channel.words)
        for channel in result.channels or []
    ):
        issues.append(
            ComplianceIssue(
                level="error",
                code="result_exceeds_diarization",
                message=(
                    "result carries speaker labels but the engine declares "
                    "batch.diarization unsupported -- the declared capabilities and "
                    "the produced result disagree. Declare diarization supported "
                    "(always_on if architecturally non-disableable), or do not emit "
                    "speaker labels."
                ),
                model=None,
            )
        )


def check_transcription_result(
    result: TranscriptionResult,
    *,
    capabilities: DeclaredCapabilities,
) -> ComplianceReport:
    """Cross-check a *recorded* batch result against the declared capabilities.

    Behavioral check for batch engines that is **pure**, mirroring
    :func:`check_event_sequence`: it inspects a result the author already
    produced -- it never instantiates or calls an engine (invoking a cloud
    engine from a compliance run would be a billable side effect). Currently it
    verifies the diarization couple: a result carrying speaker labels anywhere
    (top-level ``segments[]`` / ``words[]``, a segment's ``words``, or any
    ``channels[i]`` view) while ``batch.diarization`` is unsupported (or the
    ``batch`` domain is absent -- fail-closed) is a capability⇄result desync,
    reported as an error with code ``result_exceeds_diarization``.

    **Honest scoping note:** this is the only diarization coverage the suite
    can offer. *Positive* diarization behavior -- labels present when
    requested, correctly attributed, consistent across views -- is unverifiable
    without multi-speaker audio fixtures, and the standard probes feed silence.
    The check is therefore an opportunistic *negative* cross-check an author
    runs over their own recorded results; an ``always_on`` engine necessarily
    declares ``supported=True`` (the model validator forbids the contradictory
    pair) and is never flagged here.

    Args:
        result: The recorded result to validate.
        capabilities: The engine's declared capabilities (pass
            ``engine.declared_capabilities``).

    Returns:
        A :class:`ComplianceReport`; ``passed`` is ``True`` when the result does
        not exceed the declared capabilities.
    """
    issues: list[ComplianceIssue] = []
    _cross_check_result_capabilities(result, capabilities.batch, issues)
    return ComplianceReport(registry=None, issues=issues)


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


def check_streaming_param_gating(engine: StandardASR) -> ComplianceReport:
    """Assert a streaming engine gates an unsupported standard parameter.

    Closes the streaming-gating bypass gap as a *compliance* failure rather
    than a silent one: the base
    :meth:`~standard_asr.runtime.interface.EngineBase.start_transcription`
    template runs ``gate_params(mode="streaming")`` for every engine, so a
    "forgot to gate" engine (one that bypassed the template) must show up here.

    The check establishes a streaming session (via ``start_transcription``,
    which constructs the session but does **not** enter its context, so no wire
    connection is opened) for the first standard parameter the engine does
    **not** support in ``streaming`` mode and asserts the standard contract:

    * **strict** policy -- the call MUST raise
      :class:`~standard_asr.contract.exceptions.UnsupportedFeatureError` whose ``param``
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
    valid wire :class:`AudioFormat` taken from the engine's own
    :meth:`~standard_asr.runtime.interface.StandardASR.recommended_wire_format`
    (guarded like every other sync-member call), so an engine that legitimately
    fail-louds on a missing ``audio_format`` is not misjudged as
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
    """
    issues: list[ComplianceIssue] = []
    model = _safe_engine_id(engine)

    try:
        # EVERY supports() result goes through the guard with
        # expected_type=bool BEFORE its truthiness is consulted: an
        # `async def` (or a conditional wrapper answering some paths with a
        # coroutine) hands back a TRUTHY awaitable, and a truthy non-bool
        # ("false", an object) reads as "supported" -- either way the probe
        # would negotiate capabilities on a lie while leaking never-awaited
        # coroutines per call.
        supports_input = engine.supports("streaming_input")
        if _sync_member_violation(supports_input, "supports()", model, issues, expected_type=bool):
            return ComplianceReport(registry=None, issues=issues)
        supports_output = engine.supports("streaming_output")
        if _sync_member_violation(supports_output, "supports()", model, issues, expected_type=bool):
            return ComplianceReport(registry=None, issues=issues)
        if not (supports_input or supports_output):
            # The engine does not declare streaming support; there is no
            # streaming gating contract to exercise.
            return ComplianceReport(registry=None, issues=issues)

        probe: tuple[str, RuntimeParams, str] | None = None
        for p in _GATING_PROBES:
            supported = engine.supports(p[2])
            if _sync_member_violation(supported, "supports()", model, issues, expected_type=bool):
                return ComplianceReport(registry=None, issues=issues)
            if not supported:
                probe = (p[0], p[1](), DIAG_UNSUPPORTED_PARAMETER_IGNORED)
                break
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
                    f"selecting a streaming gating probe raised "
                    f"{safe_exception_summary(exc)}; "
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
        # The probe must hand the engine's session hook a VALID wire format:
        # an engine that does not self-manage its wire format legitimately
        # fail-louds when opened with audio_format=None, and probing it bare
        # would make that correct rejection read as a compliance error. The
        # engine's own recommended_wire_format() is the single source (the
        # compliance suite separately asserts its self-consistency).
        try:
            fmt = engine.recommended_wire_format()
        except Exception as exc:  # noqa: BLE001
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="gating_probe_context_unbuildable",
                    message=(
                        f"could not synthesize a legal wire audio_format from the "
                        f"engine's Properties to probe gating "
                        f"({safe_exception_summary(exc)}); declare a "
                        "reachable native_sample_rate / wire_encodings."
                    ),
                    model=model,
                )
            )
            return ComplianceReport(registry=None, issues=issues)
        if _sync_member_violation(
            fmt,
            "recommended_wire_format()",
            model,
            issues,
            expected_type=(AudioFormat, type(None)),
        ):
            # A coroutine is not None and not an AudioFormat: without this
            # guard an `async def` recommendation leaked unawaited into the
            # session open below and drew a context/crash verdict for a
            # modality defect. The type pin also stops a duck-typed non-format
            # object from being fed into session establishment.
            return ComplianceReport(registry=None, issues=issues)
        if fmt is None:
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="gating_probe_context_unbuildable",
                    message=(
                        "could not synthesize a legal wire audio_format from the "
                        "engine's Properties to probe gating (no usable positive "
                        "sample rate is declared); declare a "
                        "reachable native_sample_rate / wire_encodings."
                    ),
                    model=model,
                )
            )
            return ComplianceReport(registry=None, issues=issues)
        open_kwargs["audio_format"] = fmt
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
                        "it and emit a diagnostic instead."
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
                    f"start_transcription raised "
                    f"{safe_exception_summary(exc)} while probing streaming "
                    f"parameter {field_name!r}; the only contractual exception for a "
                    "gated parameter is UnsupportedFeatureError."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)

    if _sync_member_violation(
        session, "start_transcription()", model, issues, expected_type=TranscriptionSession
    ):
        # A coroutine is TRUTHY: without this guard an `async def`
        # start_transcription read as "strict engine accepted the parameter"
        # (a wrong verdict for a different defect) while leaking a
        # never-awaited coroutine into the run. The type pin mirrors the
        # reference server's establishment boundary (require_sync_result
        # pins TranscriptionSession): a duck-typed object exposing only
        # diagnostics() satisfied the best_effort read below and PASSED the
        # default compliance run, then failed every /v1/stream WebSocket
        # with internal_error -- a defect the suite exists to catch before
        # the plugin ships, reachable in the default run only here
        # (check_sync_bridge pins it too but is opt-in/billable).
        return ComplianceReport(registry=None, issues=issues)

    # The session was created but NOT opened: the base start_transcription
    # template constructs the session without entering its context (no
    # __aenter__/_open), and the best_effort verdict below needs only
    # session.diagnostics() -- a pure read of construction-time diagnostics.
    # Entering the session to "close" it would instead OPEN a billable wire
    # handshake the probe never incurred (cf. the
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
                    "raise UnsupportedFeatureError (this is the streaming gating gap)."
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
                    f"session.diagnostics() raised "
                    f"{safe_exception_summary(exc)} while checking for the "
                    f"expected {expected_code!r} diagnostic on best_effort streaming "
                    f"parameter {field_name!r}; diagnostics() must not raise."
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
                    "session.diagnostics()."
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


def check_provider_params_swap_safety(engine: StandardASR) -> ComplianceReport:
    """Assert an engine always rejects another engine's ``provider_params``.

    The standard makes ``provider_params`` swap-safety an unconditional MUST:
    a wrong-typed ``provider_params`` (the classic "switched engines, forgot to
    change the params model" bug) MUST raise
    :class:`~standard_asr.contract.exceptions.InvalidProviderParamError` **independent of
    strict / best_effort** -- it is a code bug, not a capability negotiation, so
    it is never silently dropped. The :class:`EngineBase` template enforces this
    in ``gate_params`` *before* any audio is decoded or the model is touched, so
    an engine that bypassed the template and forgot the check is the gap this
    probe closes -- the same "bypassed the template must show up here" reasoning
    behind :func:`check_streaming_param_gating`.

    The probe calls the engine's public
    :meth:`~standard_asr.runtime.interface.EngineBase.transcribe`
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
    """
    issues: list[ComplianceIssue] = []
    model = _safe_engine_id(engine)
    params = RuntimeParams(provider_params=_ForeignProviderParams())
    silence = np.zeros(1, dtype=np.float32)

    try:
        result = engine.transcribe(silence, params)
    except InvalidProviderParamError:
        # Correct: swapped provider_params rejected before any model work.
        return ComplianceReport(registry=None, issues=issues)
    except (ConfigError, EngineContractError) as exc:
        # The engine raised BEFORE the provider-params gate could run: the base
        # template validates the language config (_validate_language_config)
        # ahead of gate_params, and that method promises ConfigError for a
        # bad configuration VALUE and EngineContractError for a DECLARATION
        # defect (a malformed declared tag, a missing IC.6 default), so a
        # broken language axis surfaces here as one of the two. Swap-safety
        # was therefore never
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
                    f"transcribe raised "
                    f"{safe_exception_summary(exc)} before the provider_params gate, so "
                    "swap-safety could not be exercised; resolve the "
                    "engine's language_config_invalid defect first."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)
    except Exception as exc:  # noqa: BLE001
        # Any other exception means the engine did NOT enforce swap-safety on
        # the provider-params-first path (it failed later, for a different reason,
        # or crashed). Report it; never re-raise (this function promises
        # ``Raises: None``).
        issues.append(
            ComplianceIssue(
                level="error",
                code="provider_params_swap_not_enforced",
                message=(
                    f"transcribe raised "
                    f"{safe_exception_summary(exc)} for a foreign provider_params type "
                    "instead of InvalidProviderParamError; the standard requires "
                    "provider_params swap-safety to raise InvalidProviderParamError "
                    "ALWAYS (independent of strict/best_effort), validated before "
                    "audio decoding."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)

    if _sync_member_violation(result, "transcribe()", model, issues):
        # An `async def` transcribe returns a coroutine WITHOUT raising:
        # without this guard the probe read that as "silently accepted the
        # foreign params" -- a wrong verdict for a different defect -- while
        # leaking a never-awaited coroutine into the run.
        return ComplianceReport(registry=None, issues=issues)

    # No exception at all: the engine silently accepted another engine's params --
    # exactly the swap bug this check exists to make loud.
    issues.append(
        ComplianceIssue(
            level="error",
            code="provider_params_swap_accepted",
            message=(
                "transcribe accepted a foreign provider_params type without raising; "
                "the standard requires it to raise InvalidProviderParamError "
                "(swap-safety), independent of strict/best_effort."
            ),
            model=model,
        )
    )
    return ComplianceReport(registry=None, issues=issues)


class SupportsWireRecommendation(Protocol):
    """The two-member surface :func:`check_recommended_wire_format` needs.

    A deliberately minimal protocol instead of ``EngineBase``: the check's
    subjects include structural (non-``EngineBase``) engines -- the standard's
    own promise -- and the previous ``EngineBase``-typed signature invited
    calling ``EngineBase``-only members on them (the check once called
    ``ensure_stream_format_supported``, not a ``StandardASR`` member, so a
    fully-compliant structural engine failed with a false
    ``recommended_wire_format_self_inconsistent`` verdict on an
    ``AttributeError``).
    """

    properties: ClassVar[BaseProperties]

    def recommended_wire_format(self) -> AudioFormat | None:
        """Return the engine's recommended minimal wire format.

        Returns:
            The recommended format, or ``None`` when none is derivable.
        """
        ...


def check_recommended_wire_format(
    engine: SupportsWireRecommendation, *, model: str | None = None
) -> ComplianceReport:
    """Assert an engine's recommended wire format is one it would itself accept.

    :meth:`~standard_asr.runtime.interface.EngineBase.recommended_wire_format` is the
    single source of truth for the minimal wire :class:`AudioFormat` the standard
    layer opens a ``streaming_input`` session with when the application chose none
    -- the CLI sync-bridge runner and the streaming gating probe both rely on it.
    A self-inconsistent engine, whose recommended format the standard
    session-establishment rule rejects for its own declared Properties, would
    make those paths fail-loud on a format the standard layer chose rather
    than the application -- a silent-looking compliance trap. This closes that
    loop: when a format is recommended it MUST pass
    :func:`~standard_asr.runtime.interface.ensure_wire_format_supported` -- the
    pure ``(Properties, AudioFormat)`` rule that
    :meth:`~standard_asr.runtime.interface.EngineBase.ensure_stream_format_supported`
    itself implements. Validating via the pure rule (never the ``EngineBase``
    method) keeps the verdict correct for structural engines, which have no
    such method.

    Args:
        engine: The engine under test. Deliberately NOT required to declare
            ``streaming_input`` (or any capability): the recommendation is
            Properties-pure and capability-blind (see
            :meth:`~standard_asr.runtime.interface.EngineBase.recommended_wire_format`),
            so the self-consistency round-trip holds for every engine — the
            protocol member is unconditionally required (spec §3.1) and the
            entrypoint-layer instance checks run this round-trip for EVERY
            successfully constructed engine (batch-only included; an
            output-only engine passes trivially).
        model: The model key (``engine/model``) to attribute issues to, or
            ``None`` for a single-engine run. In a multi-model run an
            unattributed issue renders as ``<registry>`` and the user cannot
            tell which engine failed.

    Returns:
        A :class:`ComplianceReport`. ``passed`` is ``True`` when no format is
        recommended, or the recommended format is accepted by the engine.
    """
    return ComplianceReport(
        registry=None, issues=_wire_format_round_trip_issues(engine, model=model)
    )


def _wire_format_round_trip_issues(
    engine: SupportsWireRecommendation, *, model: str | None
) -> list[ComplianceIssue]:
    """Run the wire-format self-consistency round-trip, returning its issues.

    The single body behind both entry points: the public
    :func:`check_recommended_wire_format` (library API, wraps the issues in a
    report) and the entrypoint-layer :func:`_check_instance_wire_format`
    (appends them to the per-engine instance-check list). One body means the
    two surfaces can never drift on what "self-consistent" means.

    Args:
        engine: The engine under test.
        model: The model key to attribute issues to, or ``None``.

    Returns:
        The issues found (empty for a compliant engine).
    """
    issues: list[ComplianceIssue] = []
    try:
        fmt = engine.recommended_wire_format()
    except Exception as exc:  # noqa: BLE001 - reported as a compliance error
        issues.append(
            ComplianceIssue(
                level="error",
                code="recommended_wire_format_raised",
                message=(
                    f"EngineBase.recommended_wire_format() raised: {safe_exception_summary(exc)}."
                ),
                model=model,
            )
        )
        return issues
    if _sync_member_violation(
        fmt,
        "recommended_wire_format()",
        model,
        issues,
        expected_type=(AudioFormat, type(None)),
    ):
        # A coroutine is not None: without this guard an `async def`
        # implementation fell into the round-trip below and was misreported
        # as self-inconsistent while leaking a never-awaited coroutine. The
        # type pin closes the same misreporting for any non-AudioFormat
        # return: a duck-typed object with plausible attributes used to pass
        # the round-trip silently, and one without them drew a
        # "self-inconsistent" verdict for what is a wrong-return-type defect.
        return issues
    if fmt is not None:
        try:
            ensure_wire_format_supported(engine.properties, fmt)
        except Exception as exc:  # noqa: BLE001 - reported as a compliance error
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="recommended_wire_format_self_inconsistent",
                    message=(
                        f"recommended_wire_format() returned {fmt!r}, but the standard "
                        "session-establishment rule rejects it for the engine's own "
                        f"declared Properties: "
                        f"{safe_exception_summary(exc)}. The recommended format must be "
                        "one the engine accepts."
                    ),
                    model=model,
                )
            )
    return issues


class SupportsCapabilities(Protocol):
    """The one-method surface :func:`check_sync_bridge` needs from an engine.

    A deliberately minimal protocol instead of ``StandardASR``: the check
    consults nothing but ``supports()``, and demanding the full surface would
    force every caller with a partial test double (or a wrapper) to fake
    members the check never touches. (``StandardASR`` itself is now
    strict-assignable from real plugins -- ``config`` is a read-only protocol
    property -- so this narrowing is about least-surface, not a typing
    workaround.)
    """

    def supports(self, dot_path: str) -> bool:
        """Return whether the capability at ``dot_path`` is supported.

        Args:
            dot_path: A capability dot-path.

        Returns:
            ``True`` if supported.
        """
        ...


#: Default per-phase timeout (seconds) for the sync-bridge check: session
#: establishment and the bridged drive each receive this budget. THE single
#: source of the value: the CLI's --bridge-timeout help and effective default
#: both read it, so a change here can never leave the CLI silently applying
#: (or --help advertising) a stale number.
DEFAULT_SYNC_BRIDGE_TIMEOUT = 5.0


def validate_bridge_timeout(timeout: float) -> float:
    """Validate a sync-bridge timeout: MUST be finite and strictly positive.

    The single owner of the rule, shared by :func:`check_sync_bridge` and the
    CLI's ``--bridge-timeout`` parser (which wraps the ``ValueError`` into an
    argparse usage error) so the two layers can never drift: ``<= 0`` yields
    an instant false "did not terminate" verdict against a compliant engine,
    ``inf``/``nan`` hangs the check on the very deadlock it diagnoses, and a
    finite value above ``threading.TIMEOUT_MAX`` would blow up as an
    ``OverflowError`` out of ``Thread.join`` / ``Future.result`` mid-check --
    a validated timeout MUST be one the bridge's waits can actually take. No
    clamping: silently shortening a caller's timeout would be an implicit
    rewrite of an explicit value.

    Args:
        timeout: The candidate timeout in seconds.

    Returns:
        ``timeout`` unchanged.

    Raises:
        ValueError: If ``timeout`` is not finite, not strictly positive, or
            exceeds this platform's ``threading.TIMEOUT_MAX``.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            f"sync-bridge timeout must be a finite number of seconds > 0, "
            f"got {timeout!r} (<= 0 yields an instant false 'did not terminate' "
            "verdict; inf/nan hangs the check on the deadlock it diagnoses)."
        )
    if timeout > threading.TIMEOUT_MAX:
        raise ValueError(
            f"sync-bridge timeout must be <= threading.TIMEOUT_MAX "
            f"({threading.TIMEOUT_MAX!r} s on this platform), got {timeout!r}: "
            "the bridge waits with Thread.join / Future.result, which raise "
            "OverflowError beyond the platform's lock-wait cap. Pass a "
            "smaller timeout (no clamping -- an over-cap budget is a caller "
            "mistake, not a value to silently rewrite)."
        )
    return timeout


def check_sync_bridge(
    session_factory: Callable[[], TranscriptionSession],
    *,
    timeout: float = DEFAULT_SYNC_BRIDGE_TIMEOUT,
    model: str | None = None,
    engine: SupportsCapabilities | None = None,
) -> ComplianceReport:
    """Drive an async engine's :class:`SyncSession` from an external thread.

    Implements the standard's sync-bridge mandate: a no-deadlock / no-leak
    test. A fresh session is created and driven synchronously from a *different*
    thread than the one that built it, feeding no audio and immediately ending
    input. The test asserts the session terminates (emits a terminal event and
    tears down) within ``timeout`` -- a deadlock or a leaked background loop/
    thread shows up as a timeout.

    Args:
        session_factory: A zero-argument callable returning a fresh async
            :class:`TranscriptionSession` (e.g. ``engine.start_transcription``
            bound with its arguments). The return crosses the same sync-call
            boundary as every protocol member: a factory handing back an
            awaitable (an ``async def`` ``start_transcription`` behind the
            CLI's canonical factory) or any non-``TranscriptionSession``
            object is reported as ``sync_bridge_invalid_session`` -- with a
            stray coroutine closed -- instead of being driven into
            :class:`SyncSession` and misreported as a bridge lifecycle fault.
        timeout: Seconds granted to EACH phase of the check independently:
            session establishment (``session_factory()`` plus, on an
            unsupported refusal, the ``supports()`` classification probe) and
            the bridged drive (open, end-of-audio, drain, close combined --
            also each bridged lifecycle call's ``submit_timeout``). Both
            phases run under bounded daemon workers, so a hanging
            ``start_transcription`` or ``supports()`` is reported instead of
            hanging the check; worst case the check takes about twice this
            value. Per-phase (not a shared total) so a slow-but-successful
            establishment can never starve the drive join into a false
            "did not terminate" verdict. MUST be finite and
            strictly positive: ``<= 0`` would make the wait return immediately
            (a false "did not terminate" verdict against a compliant engine)
            and ``inf``/``nan`` would hang the check on the very deadlock it
            exists to diagnose, so both are rejected loudly (the same rule the
            CLI's ``--bridge-timeout`` enforces at parse time). It also caps
            each bridged lifecycle call (forwarded as the ``SyncSession``
            ``submit_timeout``), so granting a larger budget genuinely extends
            slow-but-compliant ``_open``/``_close`` phases. This
            MUST exceed the engine's real ``_open`` + ``_close`` cost: a slow but
            compliant engine (a cloud session doing a real network handshake) is
            *not* a deadlock, so when a run reports a timeout, re-run with a larger
            value to tell "slow" from "stuck". The driver thread is a daemon, so a
            false positive (or a real deadlock) never blocks interpreter exit --
            the process is not held hostage by the fault this check diagnoses.
        model: The model key (``engine/model``) to attribute issues to, or
            ``None`` for a single-engine run (a multi-model run needs the
            attribution to name the failing engine).
        engine: The engine the factory drives, if available (anything with a
            ``supports()`` method -- see :class:`SupportsCapabilities`; the
            full ``StandardASR`` protocol is deliberately not required, so a
            real plugin passes without casts). Used for exactly
            one thing: classifying an ``UnsupportedFeatureError`` raised by
            ``session_factory()`` itself (session establishment). Only an
            engine that does NOT declare ``streaming_input`` earns the passing
            ``sync_bridge_not_applicable`` verdict -- the bridge feeds bare
            frames, which such an engine genuinely cannot accept. An engine
            that DECLARES ``streaming_input`` yet refuses establishment is a
            capability lie (a declared-but-unimplemented hook, or a
            recommended wire format its own guard rejects) and FAILS. Without
            ``engine`` the classification is fail-closed: an establishment
            refusal is reported as a failure, with a hint to pass ``engine=``
            when the engine is genuinely output-only.

    Returns:
        A :class:`ComplianceReport`. ``passed`` is ``True`` when the bridge
        terminated cleanly with no leaked background loop thread, or when the
        check is not applicable (session establishment refused as unsupported
        by an engine KNOWN not to declare ``streaming_input``; reported as a
        ``sync_bridge_not_applicable`` warning, never as an engine failure).
        An ``UnsupportedFeatureError`` from anywhere PAST establishment (the
        engine's ``_open``, ``end_audio``, event drain, close) is always a
        failing ``sync_bridge_raised`` -- the not-applicable carve-out is
        scoped to the factory call alone.

    Raises:
        ValueError: If ``timeout`` is not finite or not strictly positive (a
            caller code bug, rejected independent of any policy).
    """
    validate_bridge_timeout(timeout)
    issues: list[ComplianceIssue] = []

    # Establish the session BEFORE any bridging, in its own bounded daemon
    # worker (NOT on the calling thread): a hanging start_transcription -- or
    # a hanging engine.supports() during the classification probe below,
    # plugin code is arbitrary -- is exactly the fault class this no-deadlock
    # check exists to diagnose, so the check must never itself hang on either.
    # Scoping establishment outside the DRIVE worker keeps the not-applicable
    # carve-out surgical: an UnsupportedFeatureError from the engine's own
    # lifecycle (_open, end_audio, drain, close) can then NEVER be mistaken
    # for "the check does not apply" -- it stays a failing sync_bridge_raised
    # like any other mid-bridge exception. Each phase (establishment; bridged
    # drive) is granted the FULL ``timeout``: carving one budget across both
    # let a slow-but-successful establishment starve the drive join into an
    # instant false "did not terminate" verdict.
    established: dict[str, object] = {}
    established_lock = threading.Lock()
    abandoned = threading.Event()

    def _teardown_late_session(late_session: TranscriptionSession) -> None:
        """Best-effort close of a session that arrived after the check gave up.

        Args:
            late_session: The session ``start_transcription`` eventually built.
        """
        try:
            # Close-only drive: __exit__ without __enter__ is tolerated by the
            # base session (a never-entered session just awaits _close), so
            # the teardown NEVER opens the session -- driving _open here would
            # initiate a fresh (for cloud engines: billable) connection
            # purely to destroy it, the very cost that makes the bridge
            # opt-in.
            SyncSession(late_session, submit_timeout=timeout).__exit__(None, None, None)
        except BaseException:  # noqa: BLE001, S110 - best-effort; the check already
            # reported (this runs after the timeout verdict); BaseException so
            # the establish worker's late cleanup dies as quietly as intended
            # even under plugin SystemExit -- the same containment rule as the
            # two verdict-bearing workers.
            pass

    def _establish() -> None:
        try:
            session_local = session_factory()
        except UnsupportedFeatureError as exc:
            with established_lock:
                established["exc"] = exc
            # Classification probe runs HERE, inside the bounded worker:
            # supports() is plugin code and may block; the caller's thread
            # must stay hang-proof. ``classified`` is set only AFTER the probe
            # completes: an exc without it means the probe is still hanging,
            # and the main thread must report did-not-terminate rather than
            # classify on incomplete state (a wrong "Pass engine=" hint for a
            # caller who DID pass the engine).
            if engine is not None:
                try:
                    # cast to object, not the declared bool: the whole point
                    # of this guard is engines whose supports() violates its
                    # static type at runtime.
                    raw_declared = cast("object", engine.supports("streaming_input"))
                except BaseException:  # noqa: BLE001 - a broken supports() cannot earn a pass
                    # BaseException, like the two sibling workers: this probe
                    # runs INSIDE the `except UnsupportedFeatureError` block,
                    # so a BaseException raised here is NOT caught by that
                    # try's own BaseException arm (Python never routes an
                    # exception raised in an except block to a sibling
                    # clause). The worker would die with `classified` unset
                    # and the main thread would report
                    # sync_bridge_did_not_terminate -- a timeout verdict for
                    # what is really a broken supports().
                    with established_lock:
                        established["supports_raised"] = True
                else:
                    # The shared sync-call boundary: an awaitable (a TRUTHY
                    # coroutine bool() would coerce to a declared-streaming
                    # verdict, then leak unawaited) or a truthy non-bool
                    # ("false", an object) is a broken capability surface --
                    # classify fail-closed, never fabricate a verdict.
                    supports_defect = sync_result_defect(raw_declared, expected_type=bool)
                    with established_lock:
                        if supports_defect is not None:
                            established["supports_invalid"] = supports_defect
                        else:
                            established["declared_streaming_input"] = raw_declared
            with established_lock:
                established["classified"] = True
        except BaseException as exc:  # noqa: BLE001 - classified below
            # BaseException DELIBERATELY: this runs on a daemon worker thread,
            # where an uncaught SystemExit/KeyboardInterrupt (or any
            # BaseException an engine raises) would die silently and the main
            # thread would misread the empty state dict as an establishment
            # HANG -- a wrong verdict against the engine. Every escape is
            # classified into the state dict instead; the main thread decides.
            with established_lock:
                established["exc"] = exc
                established["classified"] = True
        else:
            # The factory's return value crosses the SAME sync-call boundary
            # as every protocol member: the CLI's canonical factory wraps
            # start_transcription, so an `async def` opener (or a sync
            # wrapper delegating to one) hands back an awaitable here --
            # storing it as the session would misreport a modality defect as
            # a bridge lifecycle fault deep inside SyncSession while the
            # coroutine leaked unawaited, and an arbitrary non-session object
            # would surface only as a confusing secondary AttributeError.
            factory_defect = sync_result_defect(session_local, expected_type=TranscriptionSession)
            if factory_defect is not None:
                with established_lock:
                    established["factory_invalid"] = factory_defect
                return
            with established_lock:
                if abandoned.is_set():
                    late = session_local
                else:
                    established["session"] = session_local
                    late = None
            if late is not None:
                # The check already reported an establishment timeout; do not
                # leak the late session's resources (connections, state).
                _teardown_late_session(late)

    establish_worker = threading.Thread(
        target=_establish, name="compliance-sync-bridge-establish", daemon=True
    )
    establish_worker.start()
    establish_worker.join(timeout=timeout)
    with established_lock:
        # Success requires either a stored session or a FULLY classified
        # exception (exc + classified): an exc whose supports() probe is still
        # hanging must read as a timeout, not be classified on partial state.
        # is_alive() is deliberately not consulted -- a worker momentarily
        # alive while exiting after a completed store must not read as hung.
        timed_out = (
            "session" not in established
            and "factory_invalid" not in established
            and not ("exc" in established and "classified" in established)
        )
        if timed_out:
            abandoned.set()
    if timed_out:
        issues.append(
            ComplianceIssue(
                level="error",
                code="sync_bridge_did_not_terminate",
                message=(
                    f"Session establishment did not complete within {timeout}s -- "
                    "start_transcription (or the supports() classification probe) "
                    "hung, or legitimately needs longer. Re-run with a larger "
                    "timeout to disambiguate (library: check_sync_bridge(..., "
                    "timeout=...); CLI: standard-asr compliance run "
                    "--include-bridge --bridge-timeout SECONDS). A session that "
                    "finishes establishing after this report is closed "
                    "best-effort, not leaked."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)
    factory_invalid = cast("str | None", established.get("factory_invalid"))
    if factory_invalid is not None:
        # The factory returned, but not a session: an awaitable (the CLI's
        # canonical factory wraps start_transcription, so this is an
        # `async def` opener -- the entry-point checks report the member as
        # protocol_member_not_synchronous) or some other non-session object.
        # Driving it into SyncSession would misreport the defect as a bridge
        # lifecycle fault; report it at the boundary it violated instead.
        issues.append(
            ComplianceIssue(
                level="error",
                code="sync_bridge_invalid_session",
                message=(
                    f"session_factory {factory_invalid}; check_sync_bridge "
                    "requires a factory that SYNCHRONOUSLY returns a "
                    "TranscriptionSession (start_transcription is a "
                    "synchronous protocol member -- async behavior lives "
                    "inside the returned session)."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)
    exc_or_none = established.get("exc")

    if isinstance(exc_or_none, UnsupportedFeatureError):
        exc = exc_or_none
        declared_streaming_input = cast("bool | None", established.get("declared_streaming_input"))
        supports_raised = bool(established.get("supports_raised"))
        supports_invalid = cast("str | None", established.get("supports_invalid"))
        if supports_invalid is not None:
            # supports() answered, but with the wrong SHAPE (an awaitable or
            # a non-bool): the declaration is unverifiable through a broken
            # capability surface, and fabricating a verdict from truthiness
            # would be a capability decision built on a type error.
            issues.append(
                ComplianceIssue(
                    level="error",
                    code="sync_bridge_raised",
                    message=(
                        "Session establishment raised UnsupportedFeatureError "
                        f"({safe_exception_summary(exc)}). "
                        f"The engine's own supports() {supports_invalid} "
                        "while verifying streaming_input -- a broken capability "
                        "surface cannot earn a not-applicable pass; supports() "
                        "must synchronously return a bool (the entry-point "
                        "checks flag this too)."
                    ),
                    model=model,
                )
            )
            return ComplianceReport(registry=None, issues=issues)
        if declared_streaming_input is False:
            # The one honest not-applicable shape: the engine itself says it
            # cannot accept bare-frame input, so the bridge has nothing to test.
            issues.append(
                ComplianceIssue(
                    level="warning",
                    code="sync_bridge_not_applicable",
                    message=(
                        "Sync-bridge check not applicable: the engine does not "
                        "declare streaming_input and refused session "
                        f"establishment as unsupported ({safe_exception_summary(exc)}). "
                        "The bridge feeds "
                        "bare PCM frames; this is a property of the check, not "
                        "an engine failure."
                    ),
                    model=model,
                )
            )
            return ComplianceReport(registry=None, issues=issues)
        # Fail-closed: the engine declares streaming_input (a refusal is then a
        # capability lie), or no engine was provided so the claim cannot be
        # verified -- an unverifiable establishment refusal MUST NOT pass.
        issues.append(
            ComplianceIssue(
                level="error",
                code="sync_bridge_raised",
                message=(
                    "Session establishment raised UnsupportedFeatureError "
                    f"({safe_exception_summary(exc)}). "
                    + (
                        "The engine DECLARES streaming_input, so refusing a "
                        "bare-frame session is a capability lie (a declared-but-"
                        "unimplemented streaming hook, or a recommended wire "
                        "format the engine's own guard rejects)."
                        if declared_streaming_input
                        else (
                            "The engine's own supports() raised while verifying "
                            "streaming_input -- a broken capability surface "
                            "cannot earn a not-applicable pass; fix supports() "
                            "first (the entry-point checks flag it too)."
                            if supports_raised
                            else "Pass engine=... so the check can verify "
                            "whether the engine declares streaming_input (a "
                            "genuinely output-only engine is then reported "
                            "not-applicable instead of failing)."
                        )
                    )
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)
    if isinstance(exc_or_none, BaseException):
        issues.append(
            ComplianceIssue(
                level="error",
                code="sync_bridge_raised",
                message=(f"Session establishment raised: {safe_exception_summary(exc_or_none)}."),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)
    # _establish stored either "exc" (handled above) or the constructed
    # session; the dict is object-typed only because it crosses the thread.
    session = cast(TranscriptionSession, established["session"])

    outcome: dict[str, object] = {}
    worker_name = "compliance-sync-bridge"

    def _drive() -> None:
        sync: SyncSession | None = None
        try:
            # The user's timeout budget applies to the bridged lifecycle calls
            # too (submit_timeout), not only the outer join: otherwise a
            # --bridge-timeout above SyncSession's internal 30 s default was
            # silently inert for open/close and a slow-but-compliant engine
            # failed as "raised" no matter how much time the user granted.
            sync = SyncSession(session, submit_timeout=timeout)
            with sync:
                sync.end_audio()
                events = list(sync)
            outcome["terminal"] = any(getattr(ev, "is_terminal", False) for ev in events)
        except BaseException as exc:  # noqa: BLE001 - reported as a compliance error
            # BaseException, matching _establish's containment: a SystemExit
            # (or CancelledError) out of plugin code on this daemon worker
            # would otherwise kill the thread WITHOUT writing "error", and
            # the main thread mis-reads the silent death as
            # sync_bridge_no_terminal -- a false verdict about the wrong
            # defect. Store the exception OBJECT; the main thread renders it
            # through the total safe renderer (freezing repr(exc) in-thread
            # had the same silent-death failure mode under a hostile
            # __repr__).
            outcome["error"] = exc
        finally:
            # Record the bridge's OWN loop-thread liveness so the leak check below
            # asserts on this thread specifically. A compliant engine may pull in a
            # dependency that spawns a benign daemon thread (e.g. tqdm's monitor, a
            # thread-pool worker) during the session; a process-wide thread diff
            # would mis-report that as a sync_bridge_thread_leak.
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
                    "deadlock OR an engine whose _open/_close legitimately takes "
                    f"longer than {timeout}s. Re-run with a larger timeout to "
                    "disambiguate (library: check_sync_bridge(..., timeout=...); "
                    "CLI: standard-asr compliance run --include-bridge "
                    "--bridge-timeout SECONDS). If it is a deadlock, check the "
                    "sync-bridge contract: bind loop resources in "
                    "__aenter__, never touch the ambient event loop."
                ),
                model=model,
            )
        )
        return ComplianceReport(registry=None, issues=issues)

    if "error" in outcome:
        issues.append(
            ComplianceIssue(
                level="error",
                code="sync_bridge_raised",
                message=(
                    "SyncSession raised while bridging: "
                    f"{safe_exception_summary(cast('BaseException', outcome['error']))}."
                ),
                model=model,
            )
        )
    elif not outcome.get("terminal"):
        # A well-formed session always lands a terminal event (the base producer
        # force-appends ``done``); reaching here means a non-compliant engine
        # bypassed the base class and closed without one. This is exactly what the
        # compliance check exists to catch.
        issues.append(
            ComplianceIssue(
                level="error",
                code="sync_bridge_no_terminal",
                message="SyncSession ended without emitting a terminal event.",
                model=model,
            )
        )

    # Leak check: the bridge owns a background loop thread that __exit__ (and a
    # failed __enter__) MUST tear down. Assert on the bridge's OWN thread (recorded
    # in _drive via is_loop_alive), not a process-wide thread diff -- a compliant
    # engine may spawn a benign daemon thread during the session, which a diff
    # would mis-flag as a sync_bridge_thread_leak.
    if outcome.get("loop_alive"):
        issues.append(
            ComplianceIssue(
                level="error",
                code="sync_bridge_thread_leak",
                message="SyncSession did not tear down its owned background loop thread on close.",
                model=model,
            )
        )

    return ComplianceReport(registry=None, issues=issues)
