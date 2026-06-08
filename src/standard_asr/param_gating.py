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

from .capabilities import (
    DeclaredCapabilities,
    PhraseHintsCap,
    PromptCap,
    WordTimestampsCap,
)
from .exceptions import InvalidProviderParamError, UnsupportedFeatureError
from .results import Diagnostic
from .runtime_params import ProviderParams, RuntimeParams

Mode = Literal["batch", "streaming"]

#: Portable standard-set fields and their capability dot-path suffixes.
#:
#: ``candidate_languages`` is deliberately absent: the language axis is owned
#: solely by :func:`standard_asr.language.effective_candidate_languages` per
#: spec §Language R3. R3 step 2 requires that an unsupported ``candidate_languages``
#: resolve to ``None`` + a single diagnostic and **never raise** (even in strict),
#: which contradicts this table's strict-raises gating. Gating it here too would
#: also double-diagnose in best_effort. So it has exactly one owner (language.py).
_GATED_PARAMS: tuple[tuple[str, str], ...] = (
    ("language", "language.runtime_override"),
    ("word_timestamps", "word_timestamps"),
    ("prompt", "guidance.prompt"),
    ("phrase_hints", "guidance.phrase_hints"),
)

#: List-typed channels whose empty-list value (``[]``) is the spec §R.3.3
#: "requested-but-empty" sentinel: an explicit "nothing to honor". It is NOT a
#: real request, so it is never gated, degraded, or reported as unsupported.
#: ``candidate_languages`` is no longer gated here (owned by language.py), but its
#: ``[]`` sentinel is still recognized so a stray direct caller stays consistent.
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
            # word-timestamp granularity must be one the engine offers, or the
            # declared guidance limits). The sub-check handles its own drop/raise
            # / truncation; nothing else to do here.
            if field_name == "word_timestamps":
                _gate_granularity(params, capabilities, mode, updates, diagnostics, strict=strict)
            elif field_name == "prompt":
                _enforce_prompt_limit(
                    params, capabilities, mode, updates, diagnostics, strict=strict
                )
            elif field_name == "phrase_hints":
                _enforce_phrase_hints_limits(
                    params, capabilities, mode, updates, diagnostics, strict=strict
                )
            continue
        # Unsupported at the feature level.
        if field_name == "phrase_hints" and _try_degrade_to_prompt(
            params, capabilities, mode, updates, diagnostics, strict=strict
        ):
            continue
        if strict:
            raise UnsupportedFeatureError(
                f"Parameter {field_name!r} is not supported in {mode} mode.",
                param=field_name,
                mode=mode,
                hint=(
                    "Use best_effort to drop it with a diagnostic, or choose an "
                    "engine that supports it."
                ),
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
    # RUNT-6: a supported WordTimestampsCap always enumerates its granularities
    # (enforced by WordTimestampsCap's validator), so there is no "empty =>
    # honor anything" ambiguity here -- the requested value MUST be offered.
    if requested.value in set(node.granularities):
        return False
    if strict:
        raise UnsupportedFeatureError(
            f"word_timestamps granularity {requested.value!r} is not supported "
            f"in {mode} mode (offered: {sorted(node.granularities)}).",
            param="word_timestamps",
            mode=mode,
            hint=f"Request one of the offered granularities: {sorted(node.granularities)}.",
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


def _count_tokens(text: str) -> int:
    """Approximate a prompt's token count by whitespace-delimited words.

    The standard layer has no engine tokenizer, so ``max_tokens`` is enforced
    against a whitespace word count -- a deterministic, engine-agnostic
    approximation of "prompt length". This is intentionally conservative for the
    Latin-script case and documented as such on the constraint
    (:attr:`PromptConstraints.max_tokens`).

    Args:
        text: The prompt text.

    Returns:
        The number of whitespace-delimited tokens.
    """
    return len(text.split())


def _enforce_prompt_limit(
    params: RuntimeParams,
    capabilities: DeclaredCapabilities,
    mode: Mode,
    updates: dict[str, object],
    diagnostics: list[Diagnostic],
    *,
    strict: bool,
) -> None:
    """Enforce the declared ``prompt.max_tokens`` limit on a supported prompt.

    A supported ``prompt`` MUST still respect the engine's declared token budget
    (spec §Runtime 3.3 / R4 -- guidance is best-effort but MUST NOT silently
    exceed a declared bound). In best_effort mode an over-budget prompt is
    truncated to ``max_tokens`` whitespace tokens with a diagnostic; in strict
    mode it raises. An absent (``None``) limit is unbounded -- nothing to do.

    Args:
        params: The request parameters.
        capabilities: The engine's effective capabilities.
        mode: ``"batch"`` or ``"streaming"``.
        updates: Field-update accumulator (mutated on truncation).
        diagnostics: Diagnostics accumulator (mutated on truncation).
        strict: Whether an over-limit prompt raises (vs truncate + diagnostic).

    Raises:
        UnsupportedFeatureError: In strict mode, if the prompt exceeds the limit.
    """
    prompt = params.prompt
    if prompt is None:
        return
    node = capabilities.node_at(f"{mode}.guidance.prompt")
    if not isinstance(node, PromptCap):
        return
    max_tokens = node.constraints.max_tokens
    if max_tokens is None:
        return
    tokens = prompt.split()
    if len(tokens) <= max_tokens:
        return
    if strict:
        raise UnsupportedFeatureError(
            f"prompt has {len(tokens)} tokens; max is {max_tokens} in {mode} mode.",
            param="prompt",
            mode=mode,
            hint=f"Shorten the prompt to at most {max_tokens} whitespace tokens.",
        )
    truncated = " ".join(tokens[:max_tokens])
    updates["prompt"] = truncated
    diagnostics.append(
        Diagnostic(
            level="warning",
            code="prompt_truncated",
            message=(f"Truncated prompt from {len(tokens)} to {max_tokens} tokens in {mode} mode."),
            param="prompt",
            provided=len(tokens),
            effective=truncated,
        )
    )


def _truncate_term(term: str, *, max_chars: int | None, max_words: int | None) -> str:
    """Truncate one phrase-hint term to the declared per-term limits.

    Args:
        term: The phrase-hint term.
        max_chars: Maximum characters per term, or ``None`` for unbounded.
        max_words: Maximum words per term, or ``None`` for unbounded.

    Returns:
        The term truncated to satisfy both limits (words first, then chars).
    """
    out = term
    if max_words is not None:
        out = " ".join(out.split()[:max_words])
    if max_chars is not None:
        out = out[:max_chars]
    return out


def _enforce_phrase_hints_limits(
    params: RuntimeParams,
    capabilities: DeclaredCapabilities,
    mode: Mode,
    updates: dict[str, object],
    diagnostics: list[Diagnostic],
    *,
    strict: bool,
) -> None:
    """Enforce declared ``phrase_hints`` limits on a supported hints list.

    A supported ``phrase_hints`` MUST respect the engine's declared limits
    (``max_terms`` / ``max_chars_per_term`` / ``max_words_per_term``; spec
    §Runtime 3.3 / R4). In best_effort mode an over-limit list is truncated (too
    many terms are dropped from the tail; over-long terms are shortened) with a
    single diagnostic; in strict mode any violation raises. Absent (``None``)
    limits are unbounded.

    Args:
        params: The request parameters.
        capabilities: The engine's effective capabilities.
        mode: ``"batch"`` or ``"streaming"``.
        updates: Field-update accumulator (mutated on truncation).
        diagnostics: Diagnostics accumulator (mutated on truncation).
        strict: Whether an over-limit list raises (vs truncate + diagnostic).

    Raises:
        UnsupportedFeatureError: In strict mode, if a declared limit is exceeded.
    """
    hints = params.phrase_hints
    if not hints:
        return
    node = capabilities.node_at(f"{mode}.guidance.phrase_hints")
    if not isinstance(node, PhraseHintsCap):
        return
    c = node.constraints
    violation = (c.max_terms is not None and len(hints) > c.max_terms) or any(
        (c.max_chars_per_term is not None and len(t) > c.max_chars_per_term)
        or (c.max_words_per_term is not None and len(t.split()) > c.max_words_per_term)
        for t in hints
    )
    if not violation:
        return
    if strict:
        raise UnsupportedFeatureError(
            f"phrase_hints violate declared limits in {mode} mode "
            f"(max_terms={c.max_terms}, max_chars_per_term={c.max_chars_per_term}, "
            f"max_words_per_term={c.max_words_per_term}).",
            param="phrase_hints",
            mode=mode,
            hint=(
                f"Keep within max_terms={c.max_terms}, "
                f"max_chars_per_term={c.max_chars_per_term}, "
                f"max_words_per_term={c.max_words_per_term}."
            ),
        )
    kept = hints if c.max_terms is None else hints[: c.max_terms]
    truncated = [
        _truncate_term(t, max_chars=c.max_chars_per_term, max_words=c.max_words_per_term)
        for t in kept
    ]
    updates["phrase_hints"] = truncated
    diagnostics.append(
        Diagnostic(
            level="warning",
            code="phrase_hints_truncated",
            message=f"Truncated phrase_hints to declared limits in {mode} mode.",
            param="phrase_hints",
            provided=hints,
            effective=truncated,
        )
    )


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
    *,
    strict: bool,
) -> bool:
    """Attempt the opt-in one-way phrase_hints -> prompt degradation.

    The synthesized prompt MUST itself respect the prompt channel's declared
    ``max_tokens`` budget -- degradation must never silently emit a prompt the
    engine cannot accept (spec §Runtime R4: never silently degrade). When the
    combined prompt would exceed the budget: in best_effort it is truncated to
    ``max_tokens`` tokens with a ``prompt_truncated`` diagnostic (in addition to
    the degrade diagnostic); in strict mode it raises, so the caller falls
    through to the standard unsupported-phrase_hints handling rather than
    applying a lossy degrade silently.

    Args:
        params: The request parameters.
        capabilities: The engine's effective capabilities.
        mode: ``"batch"`` or ``"streaming"``.
        updates: Field-update accumulator (mutated on success).
        diagnostics: Diagnostics accumulator (mutated on success).
        strict: Whether an over-budget synthesized prompt raises (vs truncate).

    Returns:
        ``True`` if the degradation was applied.

    Raises:
        UnsupportedFeatureError: In strict mode, if the synthesized prompt would
            exceed the declared ``prompt.max_tokens`` budget.
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
    combined = f"{existing}\n{framed}" if existing else framed

    node = capabilities.node_at(f"{mode}.guidance.prompt")
    max_tokens = node.constraints.max_tokens if isinstance(node, PromptCap) else None
    if max_tokens is not None and _count_tokens(combined) > max_tokens:
        if strict:
            raise UnsupportedFeatureError(
                f"Degraded prompt would have {_count_tokens(combined)} tokens; "
                f"max is {max_tokens} in {mode} mode.",
                param="prompt",
                mode=mode,
                hint=(
                    f"Provide fewer phrase_hints so the synthesized prompt fits "
                    f"within {max_tokens} tokens."
                ),
            )
        truncated = " ".join(combined.split()[:max_tokens])
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="prompt_truncated",
                message=(f"Truncated degraded prompt to {max_tokens} tokens in {mode} mode."),
                param="prompt",
                provided=_count_tokens(combined),
                effective=truncated,
            )
        )
        combined = truncated

    updates["prompt"] = combined
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
