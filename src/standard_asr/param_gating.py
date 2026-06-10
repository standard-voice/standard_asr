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

import unicodedata
from typing import Literal, cast

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

#: Diagnostic codes the gating layer emits (spec Runtime R2). These strings are
#: part of the standard's contract -- applications match on them and the
#: compliance suite (:mod:`standard_asr.compliance`) asserts engines surface
#: them -- so they live here, in the module that emits them, as the single
#: source of truth. A consumer hard-coding the literal instead of importing it
#: would silently test (or log) the wrong contract after a rename.
DIAG_UNSUPPORTED_PARAMETER_IGNORED = "unsupported_parameter_ignored"
DIAG_UNSUPPORTED_GRANULARITY_IGNORED = "unsupported_granularity_ignored"
DIAG_PROMPT_TRUNCATED = "prompt_truncated"

#: Portable standard-set fields and their capability dot-path suffixes.
#:
#: ``candidate_languages`` is deliberately absent: the language axis is owned
#: solely by :func:`standard_asr.language.effective_candidate_languages` per
#: spec §Language R3. R3 step 2 requires that an unsupported ``candidate_languages``
#: resolve to ``None`` + a single diagnostic and **never raise** (even in strict),
#: which contradicts this table's strict-raises gating. Gating it here too would
#: also double-diagnose in best_effort. So it has exactly one owner (language.py).
#:
#: ``prompt`` is deliberately LAST -- after ``phrase_hints`` -- so the opt-in
#: phrase_hints -> prompt degradation composes the final prompt BEFORE the
#: prompt budget is enforced. The budget is then enforced exactly once, on the
#: final composed value, and exactly one ``prompt_truncated`` diagnostic can
#: ever be produced per request (no retroactive deletion of an earlier one).
_GATED_PARAMS: tuple[tuple[str, str], ...] = (
    ("language", "language.runtime_override"),
    ("word_timestamps", "word_timestamps"),
    ("phrase_hints", "guidance.phrase_hints"),
    ("prompt", "guidance.prompt"),
)

#: Portable fields that are deliberately NOT in :data:`_GATED_PARAMS`, each with
#: the reason it is exempt from capability gating. Maintained alongside the gated
#: table so the drift guard below can prove the two together cover *every*
#: ``RuntimeParams`` field -- a future field that is neither gated nor explicitly
#: exempted would otherwise reach the engine ungated, and an unsupported engine
#: would silently ignore it (the cardinal sin), undetected by tests or the
#: compliance suite.
_UNGATED_PORTABLE_FIELDS: frozenset[str] = frozenset(
    {
        # Owned solely by language.effective_candidate_languages (spec Language
        # R3): its unsupported path resolves to None + one diagnostic and never
        # raises, which is incompatible with this table's strict-raises gating.
        "candidate_languages",
        # A policy directive (spec §3.1), not a capability-gated channel: it
        # controls WHETHER an unsupported guidance channel degrades, it is not
        # itself negotiated against a capability.
        "on_unsupported",
        # The typed escape hatch (spec §3.2): validated for swap-safety by
        # _check_provider_params (always-raise), never capability-gated.
        "provider_params",
    }
)

#: Drift guard (import-time, mirrored by a drift test): every portable
#: ``RuntimeParams`` field MUST be either capability-gated (in _GATED_PARAMS) or
#: explicitly exempt (in _UNGATED_PORTABLE_FIELDS) -- never silently neither. When
#: the standard set grows by an additive-minor field, this fails loudly until the
#: author classifies it, closing the "new param silently bypasses gating" hole.
assert {field for field, _ in _GATED_PARAMS} | _UNGATED_PORTABLE_FIELDS == set(
    RuntimeParams.model_fields
), (
    "RuntimeParams field set drifted from param gating: every portable field must "
    "be in _GATED_PARAMS or _UNGATED_PORTABLE_FIELDS (see the drift guard). "
    f"Gated={ {f for f, _ in _GATED_PARAMS} }, exempt={set(_UNGATED_PORTABLE_FIELDS)}, "
    f"fields={set(RuntimeParams.model_fields)}."
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
        # The full capability dot-path of the failing feature, so a caller can
        # mechanically relate the rejection back to a capabilities.supports(path)
        # query (the spec requires the best_effort diagnostic to say WHICH
        # capability is unsupported -- and the field->path map is module-private,
        # not something the caller can reconstruct: e.g. ``language`` gates on
        # ``language.runtime_override``, not on the field name).
        cap_path = f"{mode}.{cap_suffix}"
        if strict:
            raise UnsupportedFeatureError(
                f"Parameter {field_name!r} is not supported in {mode} mode "
                f"(capability {cap_path!r} not supported).",
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
                code=DIAG_UNSUPPORTED_PARAMETER_IGNORED,
                message=(
                    f"Ignored unsupported parameter {field_name!r} in {mode} mode "
                    f"(capability {cap_path!r} not supported)."
                ),
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
    # A supported WordTimestampsCap always enumerates its granularities
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
            code=DIAG_UNSUPPORTED_GRANULARITY_IGNORED,
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


#: Inclusive Unicode codepoint ranges for the scripts that are written
#: **without spaces between words**, where a whitespace word count grossly
#: under-estimates the token count (the spec's Qwen3 CJK prompt example collapses
#: to ~1 whitespace token).
#:
#: The spec §3.3 ``prompt.max_tokens`` guarantee is normative ("never under-counts
#: relative to whitespace + no-space-script tokenization", naming CJK / kana /
#: Hangul / Thai). That lower bound is defined per **Unicode Script property**, so
#: this table MUST cover *every* block whose script is one of those -- a missing
#: block is a real under-count of a real writing form (e.g. half-width katakana
#: from legacy Japanese input, or NFD Korean decomposed into conjoining Jamo by
#: some filesystems), not a long-tail edge. The set is therefore exhaustive for
#: the named scripts, not just their "main" blocks:
#:
#: * Japanese kana: Hiragana/Katakana (U+3040-30FF), Katakana Phonetic Extensions
#:   (U+31F0-31FF), half-width katakana (U+FF66-FF9F), Kana Extended-B / "Minnan"
#:   (U+1AFF0-1AFFF), Kana Supplement / Kana Extended-A / Small Kana Extension
#:   (U+1B000-1B16F), and SQUARE HIRAGANA HOKA (U+1F200).
#: * Korean Hangul: syllables (U+AC00-D7AF), conjoining Jamo (U+1100-11FF; the NFD
#:   form), Hangul Compatibility Jamo (U+3130-318F), Jamo Extended-A (U+A960-A97F),
#:   Jamo Extended-B (U+D7B0-D7FF), and half-width Hangul (U+FFA0-FFDC).
#: * CJK ideographs & radicals: Ext-A (U+3400-4DBF), base (U+4E00-9FFF), CJK
#:   Radicals Supplement (U+2E80-2EFF), Kangxi Radicals (U+2F00-2FDF), compatibility
#:   (U+F900-FAFF), Ideographic Symbols (U+16FE2-16FF1), Ext-B..F + compat supplement
#:   (U+20000-2FA1F; also covers Ext-I at U+2EBF0-2EE5F), and Ext-G/H (U+30000-323AF).
#: * Shared CJK symbol/enclosed blocks (Han iteration marks, circled/parenthesized
#:   Hangul, circled/squared kana): CJK Symbols & Punctuation (U+3000-303F; U+3000
#:   itself is whitespace, excluded at the call site) and Enclosed CJK Letters &
#:   Months + CJK Compatibility (U+3200-33FF).
#: * Other space-less scripts: Yi (U+A000-A4CF) and the major space-less
#:   South-East-Asian scripts -- Thai (U+0E00-0E7F), Lao (U+0E80-0EFF), Khmer
#:   (U+1780-17FF) + Khmer Symbols (U+19E0-19FF), Myanmar (U+1000-109F) + Myanmar
#:   Extended-A (U+AA60-AA7F) + Myanmar Extended-B (U+A9E0-A9FF).
#:
#: This table is hand-maintained against the Unicode Script DB: the stdlib exposes
#: no Script property and the near-zero-dep core (numpy + pydantic only) cannot pull
#: a Script database, so the table cannot be auto-verified against it. A per-block
#: regression test pins coverage; extend the table AND that test together on a future
#: UCD. Coarse block bounds may over-count unassigned/Common codepoints, which the
#: spec's lower-bound guarantee explicitly permits (it forbids under-counting, not
#: over-counting). Whitespace codepoints (e.g. the ideographic space U+3000) are
#: excluded at the call site.
_NO_SPACE_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0E00, 0x0E7F),  # Thai
    (0x0E80, 0x0EFF),  # Lao
    (0x1000, 0x109F),  # Myanmar
    (0x1100, 0x11FF),  # Hangul Jamo (conjoining; NFD-decomposed Korean)
    (0x1780, 0x17FF),  # Khmer
    (0x19E0, 0x19FF),  # Khmer Symbols
    (0x2E80, 0x2EFF),  # CJK Radicals Supplement
    (0x2F00, 0x2FDF),  # Kangxi Radicals
    (0x3000, 0x303F),  # CJK Symbols & Punctuation (U+3000 excluded as whitespace)
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0x3200, 0x33FF),  # Enclosed CJK Letters & Months + CJK Compatibility
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xA000, 0xA4CF),  # Yi
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xA9E0, 0xA9FF),  # Myanmar Extended-B
    (0xAA60, 0xAA7F),  # Myanmar Extended-A
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF66, 0xFF9F),  # Half-width katakana
    (0xFFA0, 0xFFDC),  # Half-width Hangul
    (0x16FE2, 0x16FF1),  # CJK Ideographic Symbols
    (0x1AFF0, 0x1AFFF),  # Kana Extended-B (Minnan)
    (0x1B000, 0x1B16F),  # Kana Supplement + Kana Extended-A + Small Kana Extension
    (0x1F200, 0x1F200),  # SQUARE HIRAGANA HOKA
    (0x20000, 0x2FA1F),  # CJK Unified Ideographs Extension B-F + compat supplement
    (0x30000, 0x323AF),  # CJK Unified Ideographs Extension G + H
)


def _is_no_space_codepoint(ch: str) -> bool:
    """Return whether *ch* belongs to a space-less script (CJK/kana/Hangul/SEA).

    Args:
        ch: A single character.

    Returns:
        ``True`` when ``ch`` is a non-whitespace codepoint in a script written
        without inter-word spaces, where each codepoint is at least one token.
    """
    if ch.isspace():
        return False
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _NO_SPACE_SCRIPT_RANGES)


def _count_tokens(text: str) -> int:
    """Approximate a prompt's token count (script-aware, engine-agnostic).

    The standard layer has no engine tokenizer, so ``max_tokens`` is enforced
    against an engine-agnostic, deterministic approximation of "prompt length":
    the number of whitespace-delimited words **plus one unit for every space-less
    (CJK / kana / Hangul / Thai / ...) codepoint**. The CJK term is essential --
    a whitespace word count alone collapses a long no-space prompt (the spec's
    Qwen3 ``context`` -> ``prompt`` example) to ~1 token and would let it slip
    past a ``max_tokens`` limit it actually blows.

    Scope of the guarantee: this count never under-counts relative to
    whitespace + no-space-script tokenization, but it MAY under-count an
    engine's subword (BPE) tokenization of long Latin / number / URL tokens --
    a long URL or a 19-digit number is one word here but many BPE tokens.
    Heuristic long-token splitting is deliberately NOT applied (it would risk
    over-truncating valid prompts), so engine authors should declare
    ``max_tokens`` with headroom below the provider's hard limit; see
    :attr:`~standard_asr.capabilities.PromptConstraints.max_tokens`.

    Args:
        text: The prompt text.

    Returns:
        The approximate token count: whitespace words + space-less codepoints.
    """
    return len(text.split()) + sum(1 for ch in text if _is_no_space_codepoint(ch))


def _is_combining_mark(ch: str) -> bool:
    """Return whether *ch* is a Unicode combining mark (category ``M*``).

    Used to keep prompt truncation from ending in a half-formed grapheme. The
    Unicode **general category** (``Mn`` / ``Mc`` / ``Me``) is the right test
    here, NOT ``unicodedata.combining`` (the canonical *combining class*): the
    latter is about normalization ordering and is ``0`` for many real combining
    marks -- e.g. the Thai vowel sign U+0E31 (category ``Mn``) has combining
    class ``0`` -- so a combining-class test would miss exactly the marks this
    guard exists to protect.

    Args:
        ch: A single character.

    Returns:
        ``True`` when ``ch`` is a nonspacing / spacing / enclosing combining mark.
    """
    return unicodedata.category(ch) in ("Mn", "Mc", "Me")


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Truncate *text* to a PREFIX whose :func:`_count_tokens` is ``<= max_tokens``.

    Truncation is "drop the tail", never "rewrite what survives": the result is a
    genuine prefix of ``text`` with its **original whitespace preserved** (a
    multi-line prompt keeps its newlines; the degrade path's ``"\\n"`` separator
    between the request prompt and the framed hints is not flattened). The longest
    fitting prefix length is found by binary search -- ``_count_tokens`` is
    monotonic non-decreasing in the prefix length, so the predicate "fits" is a
    step function the search bisects.

    The cut point is then pulled back so it never lands **inside a combining
    character sequence** (a base codepoint plus its following marks, e.g. a Thai
    base consonant U+0E01 and its vowel sign U+0E31): if the first dropped
    codepoint is a combining mark, the whole partial last cluster is removed so
    the result never ends in a base whose marks were sliced off. This only ever
    shortens the result, so the budget guarantee is preserved.

    Args:
        text: The prompt text to truncate.
        max_tokens: The maximum approximate token count to keep.

    Returns:
        The truncated prompt (a prefix of ``text``; ``_count_tokens`` is
        ``<= max_tokens``), trailing whitespace stripped.
    """
    if _count_tokens(text) <= max_tokens:
        return text
    # Binary-search the longest prefix that fits. ``_count_tokens(text[:mid])`` is
    # monotonic non-decreasing in ``mid``, so the search is well-defined.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _count_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    # Combining-sequence guard: if the cut falls right after a base whose marks
    # are being dropped (the first dropped codepoint is a combining mark), pull
    # the cut back past the entire partial cluster (its trailing marks already in
    # the prefix, then the base itself) so the result never ends in a half-formed
    # grapheme. Purely shortens, so the budget still holds.
    if lo < len(text) and _is_combining_mark(text[lo]):
        while lo > 0 and _is_combining_mark(text[lo - 1]):
            lo -= 1
        if lo > 0:
            lo -= 1
    return text[:lo].rstrip()


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
    exceed a declared bound). The budget is measured with the script-aware
    :func:`_count_tokens` (an approximation, not the engine's exact tokenizer;
    see its docstring for the guarantee scope) so a long no-space / CJK prompt
    cannot slip past a limit it actually blows. In best_effort mode an
    over-budget prompt is truncated to fit
    with a diagnostic; in strict mode it raises. An absent (``None``) limit is
    unbounded -- nothing to do.

    Enforcement reads the RUNNING value (``updates`` first, then the request):
    ``prompt`` is gated last (see :data:`_GATED_PARAMS`), so a phrase-hints
    degrade composed earlier in the request is what gets budgeted -- and a
    degrade-truncated composition (already within budget by construction)
    passes through untouched, keeping exactly one ``prompt_truncated``
    diagnostic per request without any retroactive filtering.

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
    prompt = cast("str | None", updates.get("prompt", params.prompt))
    if prompt is None:
        return
    node = capabilities.node_at(f"{mode}.guidance.prompt")
    if not isinstance(node, PromptCap):
        return
    max_tokens = node.constraints.max_tokens
    if max_tokens is None:
        return
    count = _count_tokens(prompt)
    if count <= max_tokens:
        return
    if strict:
        raise UnsupportedFeatureError(
            f"prompt has ~{count} tokens; max is {max_tokens} in {mode} mode.",
            param="prompt",
            mode=mode,
            hint=f"Shorten the prompt to at most {max_tokens} tokens (standard-layer estimate).",
        )
    truncated = _truncate_to_token_budget(prompt, max_tokens)
    updates["prompt"] = truncated
    diagnostics.append(
        Diagnostic(
            level="warning",
            code=DIAG_PROMPT_TRUNCATED,
            message=(f"Truncated prompt from ~{count} to {max_tokens} tokens in {mode} mode."),
            param="prompt",
            provided=count,
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

    The type match is **exact** (``type(provided) is expected``), not
    ``isinstance``: swap safety (spec §3.2 / §5.4) is an unconditional promise,
    and ``isinstance`` would silently honor a *subclass* of the expected type.
    That is a real hole, not a hypothetical -- a vendor's engine family
    naturally models its params with inheritance (``EngineBParams(EngineAParams)``),
    so engine A would accept a ``EngineBParams`` instance and silently ignore B's
    extra knobs (the cardinal sin: a parameter the caller set has no effect, with
    no error). Exact matching means every engine MUST publish a distinct terminal
    params type; inheritance is not a channel for declaring cross-engine
    compatibility. A bare :class:`ProviderParams` base instance can never match a
    concrete subclass here either (it is also rejected at construction, see
    :class:`~standard_asr.runtime_params.RuntimeParams`).

    Args:
        provided: The request's provider params, if any.
        expected: The engine's expected provider-params type, if any.

    Raises:
        InvalidProviderParamError: If provided params are unexpected or not the
            exact expected type.
    """
    if provided is None:
        return
    if expected is None:
        raise InvalidProviderParamError(
            f"This engine accepts no provider_params, got {type(provided).__name__}."
        )
    if type(provided) is not expected:
        raise InvalidProviderParamError(
            f"provider_params must be exactly {expected.__name__}, "
            f"got {type(provided).__name__} (swapped engine, or a subclass whose "
            f"extra knobs would be silently ignored?)."
        )


#: Marker prefixing the synthesized phrase-hints block in a degraded prompt.
#: Shared by the frame builder and the post-truncation survival check so the two
#: never drift, and so survival is scanned only within the framed region.
_PHRASE_HINTS_FRAME_PREFIX = "Relevant terms: "


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

    The hints are folded onto the request prompt and the composition is
    budgeted HERE, exactly once: ``phrase_hints`` is gated before ``prompt``
    (see :data:`_GATED_PARAMS`), so no prompt-only truncation has run yet, and
    the later prompt gate reads the running (already-budgeted) composition and
    emits nothing. The synthesized prompt MUST itself respect the prompt
    channel's declared ``max_tokens`` budget -- degradation must never
    silently emit a prompt the engine cannot accept (spec §Runtime R4: never
    silently degrade). When the combined prompt would exceed the budget: in
    best_effort it is truncated to ``max_tokens`` tokens with the request's
    single ``prompt_truncated`` diagnostic; in strict mode it raises, so the
    caller never applies a lossy degrade silently.

    The framed phrase-hints block sits at the tail of the synthesized prompt, so
    forward truncation cuts it first. If the budget is too small to keep ANY
    phrase-hint term, the degrade is reported with a distinct
    ``guidance_degrade_phrase_hints_dropped`` diagnostic (not the generic
    ``guidance_degraded_to_prompt``), because no hint content actually reached the
    prompt -- claiming a successful degrade there would mislead the caller into
    thinking the hints were folded in when they were silently cut. The normal
    path (where hint content survives) is unchanged.

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
    framed = _PHRASE_HINTS_FRAME_PREFIX + ", ".join(hints) + "."
    # Compose on the RUNNING value (``updates`` first, then the request) so the
    # composition is correct under any gating order. In the actual order
    # (phrase_hints before prompt, _GATED_PARAMS) no prompt truncation has run
    # yet: the budget below is the FIRST and ONLY enforcement of the composed
    # prompt -- the later prompt gate reads this already-budgeted value back
    # out of ``updates`` and emits nothing.
    existing = cast("str | None", updates.get("prompt", params.prompt))
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
        truncated = _truncate_to_token_budget(combined, max_tokens)
        # This truncation reflects the FINAL composition (request prompt +
        # framed hints) and is the request's ONLY prompt-budget enforcement:
        # phrase_hints is gated before prompt, so nothing was emitted earlier,
        # and the later prompt gate sees the within-budget result and emits
        # nothing -- exactly one prompt_truncated per request by construction.
        diagnostics.append(
            Diagnostic(
                level="warning",
                code=DIAG_PROMPT_TRUNCATED,
                message=(f"Truncated degraded prompt to {max_tokens} tokens in {mode} mode."),
                param="prompt",
                provided=_count_tokens(combined),
                effective=truncated,
            )
        )
        combined = truncated
        # Did any hint content actually survive? Determine this by POSITION, not by
        # scanning for the marker: ``_truncate_to_token_budget`` returns a genuine
        # prefix of the pre-truncation ``combined``, so the hint content occupies a
        # KNOWN character range -- it starts right after the framed marker, which
        # itself sits at ``frame_start`` (after the request prompt and its ``"\n"``
        # separator, if any). Scanning ``combined[content_start:]`` looks ONLY at
        # the surviving hint content and is immune to two false positives the old
        # ``rfind`` scan suffered: (a) a user prompt that literally contains
        # ``"Relevant terms: "`` (rfind could land on the user's text), and (b) a
        # hint term that is a substring of the marker text or of an original-prompt
        # word (e.g. hint ``"cat"`` inside ``"categorize"``). A partial mid-term
        # cut also reads as "not survived" because the full term no longer appears.
        # If the budget is so small that NO hint term survived, the degrade folded
        # no hint content into the prompt at all -- reporting a plain
        # ``guidance_degraded_to_prompt`` here would mislead the caller into
        # thinking the hints were honored when they were silently cut. Emit a
        # distinct, explicit signal instead (the loss is honest, not a generic
        # prompt_truncated; spec §Runtime R4: never silently degrade).
        frame_start = len(existing) + 1 if existing else 0
        content_start = frame_start + len(_PHRASE_HINTS_FRAME_PREFIX)
        surviving_hint_content = combined[content_start:]
        # ``if hint.strip()`` guards the survival logic regardless of call site:
        # a blank term (rejected at RuntimeParams construction) would otherwise
        # make ``"" in x`` always true and falsely report a successful degrade.
        if not any(hint in surviving_hint_content for hint in hints if hint.strip()):
            updates["prompt"] = combined
            updates["phrase_hints"] = None
            diagnostics.append(
                Diagnostic(
                    level="warning",
                    code="guidance_degrade_phrase_hints_dropped",
                    message=(
                        "phrase_hints unsupported; the synthesized prompt was truncated to "
                        f"the {max_tokens}-token budget before any phrase-hint term fit, so "
                        f"NO phrase-hint content reached the prompt in {mode} mode."
                    ),
                    param="phrase_hints",
                    provided=hints,
                    effective=None,
                )
            )
            return True

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


__all__ = [
    "DIAG_PROMPT_TRUNCATED",
    "DIAG_UNSUPPORTED_GRANULARITY_IGNORED",
    "DIAG_UNSUPPORTED_PARAMETER_IGNORED",
    "Mode",
    "gate_params",
]
