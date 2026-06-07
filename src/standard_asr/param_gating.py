# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Capability gating for runtime parameters (spec, section "Runtime Parameters").

Given a request's :class:`~standard_asr.runtime_params.RuntimeParams` and the
engine's effective capabilities, this module enforces the standard's gating
rules:

* ``provider_params`` are validated first and errors **always raise**
  :class:`~standard_asr.exceptions.InvalidProviderParamError`, independent of the
  strict / best_effort policy (R3).
* Each portable standard-set parameter is checked against its capability path;
  unsupported parameters raise
  :class:`~standard_asr.exceptions.UnsupportedFeatureError` in strict mode, or
  are dropped with a diagnostic in best_effort mode (R2).
* The ``guidance`` family supports opt-in one-way degradation of
  ``phrase_hints`` to ``prompt`` (R4).
"""

from __future__ import annotations

from typing import Literal

from .capabilities import DeclaredCapabilities
from .exceptions import InvalidProviderParamError, UnsupportedFeatureError
from .results import Diagnostic
from .runtime_params import ProviderParams, RuntimeParams

Mode = Literal["batch", "streaming"]

#: Portable standard-set fields and their capability dot-path suffixes.
_GATED_PARAMS: tuple[tuple[str, str], ...] = (
    ("language", "language.runtime_override"),
    ("candidate_languages", "language.candidate_languages"),
    ("word_timestamps", "word_timestamps"),
    ("prompt", "guidance.prompt"),
    ("phrase_hints", "guidance.phrase_hints"),
)


def gate_params(
    params: RuntimeParams,
    capabilities: DeclaredCapabilities,
    mode: Mode,
    *,
    strict: bool,
    expected_provider_type: type[ProviderParams] | None = None,
) -> tuple[RuntimeParams, list[Diagnostic]]:
    """Validate and gate runtime parameters against engine capabilities.

    Args:
        params: The request parameters.
        capabilities: The engine's effective capabilities.
        mode: ``"batch"`` or ``"streaming"``.
        strict: Whether unsupported parameters raise (vs drop + diagnostic).
        expected_provider_type: The engine's expected ``provider_params`` type,
            or ``None`` if the engine accepts no provider params.

    Returns:
        A ``(gated_params, diagnostics)`` pair.

    Raises:
        InvalidProviderParamError: If provider params are the wrong type.
        UnsupportedFeatureError: In strict mode, if a parameter is unsupported.
    """
    _check_provider_params(params.provider_params, expected_provider_type)

    diagnostics: list[Diagnostic] = []
    updates: dict[str, object] = {}

    for field_name, cap_suffix in _GATED_PARAMS:
        value = getattr(params, field_name)
        if value is None:
            continue
        if capabilities.supports(f"{mode}.{cap_suffix}"):
            continue
        # Unsupported.
        if field_name == "phrase_hints" and _try_degrade_to_prompt(
            params, capabilities, mode, updates, diagnostics
        ):
            continue
        if strict:
            raise UnsupportedFeatureError(
                f"Parameter {field_name!r} is not supported in {mode} mode."
            )
        updates[field_name] = None
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="unsupported_parameter_ignored",
                message=f"Ignored unsupported parameter {field_name!r} in {mode} mode.",
                param=field_name,
                provided=value,
                effective=None,
            )
        )

    gated = params.model_copy(update=updates) if updates else params
    return gated, diagnostics


def _check_provider_params(
    provided: ProviderParams | None,
    expected: type[ProviderParams] | None,
) -> None:
    """Validate provider params type (swap-safe), always raising on mismatch.

    Args:
        provided: The request's provider params, if any.
        expected: The engine's expected provider-params type, if any.

    Raises:
        InvalidProviderParamError: If provided params are unexpected or the
            wrong type.
    """
    if provided is None:
        return
    if expected is None:
        raise InvalidProviderParamError(
            f"This engine accepts no provider_params, got {type(provided).__name__}."
        )
    if not isinstance(provided, expected):
        raise InvalidProviderParamError(
            f"provider_params must be {expected.__name__}, "
            f"got {type(provided).__name__} (swapped engine?)."
        )


def _try_degrade_to_prompt(
    params: RuntimeParams,
    capabilities: DeclaredCapabilities,
    mode: Mode,
    updates: dict[str, object],
    diagnostics: list[Diagnostic],
) -> bool:
    """Attempt the opt-in one-way phrase_hints -> prompt degradation.

    Args:
        params: The request parameters.
        capabilities: The engine's effective capabilities.
        mode: ``"batch"`` or ``"streaming"``.
        updates: Field-update accumulator (mutated on success).
        diagnostics: Diagnostics accumulator (mutated on success).

    Returns:
        ``True`` if the degradation was applied.
    """
    if params.on_unsupported != "degrade_to_prompt":
        return False
    if not capabilities.supports(f"{mode}.guidance.prompt"):
        return False
    hints = params.phrase_hints or []
    framed = "Relevant terms: " + ", ".join(hints) + "."
    existing = params.prompt
    updates["prompt"] = f"{existing}\n{framed}" if existing else framed
    updates["phrase_hints"] = None
    diagnostics.append(
        Diagnostic(
            level="warning",
            code="guidance_degraded_to_prompt",
            message="phrase_hints unsupported; degraded into the prompt channel.",
            param="phrase_hints",
            provided=hints,
            effective="prompt",
        )
    )
    return True


__all__ = ["Mode", "gate_params"]
