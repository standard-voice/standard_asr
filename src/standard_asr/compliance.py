"""Compliance helpers for Standard ASR plugin authors."""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

from pydantic import BaseModel

from .asr_config import BaseConfig
from .asr_properties import BaseProperties
from .capabilities import DeclaredCapabilities
from .discovery import FactoryLoadError, ModelRegistry, discover_models
from .runtime_params import ProviderParams, RuntimeParams
from .streaming import SyncSession, TranscriptionSession

__all__ = [
    "ComplianceIssue",
    "ComplianceReport",
    "check_entrypoints",
    "check_sync_bridge",
]


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
            effective = getattr(instance, "effective_capabilities", None)
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
    elif not outcome.get("terminal"):  # pragma: no cover
        # Defensive: a well-formed session ALWAYS lands a terminal event (the base
        # producer force-appends ``done``), so a clean, error-free run that emits
        # no terminal event is only reachable from a non-compliant out-of-tree
        # adapter that bypasses the base class. Kept as a guard for that case.
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
