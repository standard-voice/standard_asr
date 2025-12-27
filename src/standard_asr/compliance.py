"""Compliance helpers for Standard ASR plugin authors."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Iterable, Literal

from .asr_config import BaseConfig
from .asr_properties import BaseProperties
from .discovery import ModelRegistry, discover_models
from .exceptions import FactoryLoadError


__all__ = [
    "ComplianceIssue",
    "ComplianceReport",
    "check_entrypoints",
]


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

    def iter_level(
        self, level: Literal["error", "warning"]
    ) -> Iterable[ComplianceIssue]:
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

    for name in registry.names():
        spec = registry.spec(name)
        try:
            factory = spec.load_factory()
        except FactoryLoadError as exc:
            issues.append(ComplianceIssue(level="error", message=str(exc), model=name))
            continue

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

        if not hasattr(instance, "transcribe") or not callable(
            getattr(instance, "transcribe")
        ):
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
                    message="Instance is missing a BaseProperties-compatible 'properties' attribute.",
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

    return ComplianceReport(registry=registry, issues=issues)
