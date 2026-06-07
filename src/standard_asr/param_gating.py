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

from .capabilities import DeclaredCapabilities, WordTimestampsCap
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

#: List-typed channels whose empty-list value (``[]``) is the spec §R.3.3
#: "requested-but-empty" sentinel: an explicit "nothing to honor". It is NOT a
#: real request, so it is never gated, degraded, or reported as unsupported.
_EMPTY_IS_NOOP_FIELDS = frozenset({"candidate_languages", "phrase_hints"})


def _is_unset(field_name: str, value: object) -> bool:
    """Return whether a parameter value should be treated as "not requested".

    ``None`` is always "not requested". For the list channels in
    :data:`_EMPTY_IS_NOOP_FIELDS`, an empty list ``[]`` is the spec's explicit
    "requested-but-empty" sentinel (§R.3.3 null-semantics) -- there is nothing
    to honor, so gating skips it exactly like ``None`` (no gate, no degrade, no
    unsupported diagnostic).

    Args:
        field_name: The parameter field name.
        value: The parameter value.

    Returns:
        ``True`` if the value carries nothing actionable to gate.
    """
    if value is None:
        return True
    if field_name in _EMPTY_IS_NOOP_FIELDS and value == []:
        return True
    return False


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
        if _is_unset(field_name, value):
            continue
        if capabilities.supports(f"{mode}.{cap_suffix}"):
            # Supported at the feature level; some features carry finer-grained
            # sub-constraints that MUST also be satisfied (e.g. the requested
            # word-timestamp granularity must be one the engine offers). The
            # sub-check handles its own drop/raise; nothing else to do here.
            if field_name == "word_timestamps":
                _gate_granularity(params, capabilities, mode, updates, diagnostics, strict=strict)
            continue
        # Unsupported at the feature level.
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


def _gate_granularity(
    params: RuntimeParams,
    capabilities: DeclaredCapabilities,
    mode: Mode,
    updates: dict[str, object],
    diagnostics: list[Diagnostic],
    *,
    strict: bool,
) -> bool:
    """Validate the requested ``word_timestamps`` granularity against the engine.

    Feature-level support (``<mode>.word_timestamps.supported``) is necessary
    but not sufficient: the requested :class:`WordTimestampGranularity` MUST be
    one the engine actually offers in its ``granularities`` list. Honoring a
    granularity the engine does not provide would silently return the wrong
    granularity (the cardinal sin). In strict mode an unsupported granularity
    raises; in best_effort mode the parameter is dropped with a diagnostic.

    Args:
        params: The request parameters.
        capabilities: The engine's effective capabilities.
        mode: ``"batch"`` or ``"streaming"``.
        updates: Field-update accumulator (mutated on drop).
        diagnostics: Diagnostics accumulator (mutated on drop).
        strict: Whether an unsupported granularity raises (vs drop + diagnostic).

    Returns:
        ``True`` if the value was handled here (dropped). ``False`` if the
        requested granularity is offered and nothing was changed.

    Raises:
        UnsupportedFeatureError: In strict mode, if the granularity is not
            offered by the engine.
    """
    requested = params.word_timestamps
    if requested is None:
        return False
    node = capabilities.node_at(f"{mode}.word_timestamps")
    # A non-WordTimestampsCap node (e.g. an ``x_*`` override declared as a bare
    # flag) carries no granularity list; treat it as offering only the value it
    # was queried as supporting -- i.e. do not over-constrain unknown shapes.
    if not isinstance(node, WordTimestampsCap):
        return False
    offered = set(node.granularities)
    if not offered or requested.value in offered:
        # Empty granularities = engine did not enumerate; defer to feature flag
        # (back-compat: an engine supporting word_timestamps without listing
        # granularities is treated as offering whatever was requested).
        return False
    if strict:
        raise UnsupportedFeatureError(
            f"word_timestamps granularity {requested.value!r} is not supported "
            f"in {mode} mode (offered: {sorted(offered)})."
        )
    updates["word_timestamps"] = None
    diagnostics.append(
        Diagnostic(
            level="warning",
            code="unsupported_granularity_ignored",
            message=(
                f"Ignored unsupported word_timestamps granularity "
                f"{requested.value!r} in {mode} mode."
            ),
            param="word_timestamps",
            provided=requested.value,
            effective=None,
        )
    )
    return True


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
    hints = params.phrase_hints or []
    if not hints:
        # Empty / unset hints carry nothing to honor; never frame an empty
        # "Relevant terms: ." prompt. (The caller already skips ``[]`` via
        # _is_unset; this guards the contract regardless of call site.)
        return False
    if not capabilities.supports(f"{mode}.guidance.prompt"):
        return False
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
