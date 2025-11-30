"""Compliance helpers for Standard ASR plugin authors.

This module offers lightweight checks to ensure entry points exposed by an ASR
plugin can be discovered and instantiated predictably.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Iterable, Literal

from .discovery import ModelRegistry, discover_models
from .exceptions import FactoryLoadError


__all__ = [
    "ComplianceIssue",
    "ComplianceReport",
    "check_entrypoints",
]


@dataclass(frozen=True, slots=True)
class ComplianceIssue:
    """Single compliance issue detected during validation."""

    level: Literal["error", "warning"]
    message: str
    model: str | None = None


@dataclass(frozen=True, slots=True)
class ComplianceReport:
    """Aggregate result returned by :func:`check_entrypoints`."""

    registry: ModelRegistry
    issues: list[ComplianceIssue]

    @property
    def passed(self) -> bool:
        """Return ``True`` when no errors were encountered."""

        return not any(issue.level == "error" for issue in self.issues)

    def iter_level(
        self, level: Literal["error", "warning"]
    ) -> Iterable[ComplianceIssue]:
        """Yield issues matching *level*."""

        for issue in self.issues:
            if issue.level == level:
                yield issue


def _can_call_without_args(factory: object) -> bool:
    """Return ``True`` if *factory* can be invoked without arguments."""

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
        registry: Optional pre-discovered registry. When omitted a new registry is
            collected via :func:`discover_models`.
        strict_discovery: Forwarded to :func:`discover_models` when *registry* is ``None``.
        instantiate: When ``True`` the checker attempts to instantiate models that
            require no mandatory arguments and verifies a ``transcribe`` attribute
            exists on the resulting object.

    Returns:
        :class:`ComplianceReport` summarising findings.
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
                    message="Factory cannot be invoked without arguments; skipped instantiation check.",
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
                    message="Factory did not return an object with a callable 'transcribe' attribute.",
                    model=name,
                )
            )

    return ComplianceReport(registry=registry, issues=issues)
