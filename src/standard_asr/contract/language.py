# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for validating, normalizing, and resolving BCP 47 language tags."""

from __future__ import annotations

import re
from collections.abc import Collection

from standard_asr.contract.capabilities import ModeName
from standard_asr.contract.exceptions import UnsupportedFeatureError
from standard_asr.contract.results import Diagnostic, to_json_value

#: Reserved token meaning "let the engine auto-detect the language".
#: This is NOT a BCP-47 tag; it is a Standard ASR reserved word.
AUTO = "auto"

#: Stable machine-readable diagnostic codes emitted by language resolution.
#: A diagnostic ``code`` is a wire-visible contract: downstream consumers (the
#: compliance suite, the server, applications) match on it, so it MUST have a
#: single source of truth exported as a module constant -- a consumer
#: hard-coding the literal would silently match (or log) the wrong contract
#: after a rename. This mirrors the ``DIAG_*`` convention ``runtime.gating``
#: established for its codes; keep the two modules consistent.
DIAG_CANDIDATE_LANGUAGES_IGNORED = "candidate_languages_ignored"
DIAG_CANDIDATE_LANGUAGE_DROPPED = "candidate_language_dropped"
DIAG_CANDIDATE_LANGUAGES_TRUNCATED = "candidate_languages_truncated"
#: The remaining language-family codes are emitted by the engine base's
#: language-axis resolution (``EngineBase._resolve_language_axis``), which has
#: the engine context (``default_language``, selectable set) this module never
#: sees -- but their single source of truth still lives HERE, with the rest of
#: the language-family codes, so a consumer imports every language diagnostic
#: code from one contract module.
DIAG_LANGUAGE_FELL_BACK = "language_fell_back"
DIAG_LANGUAGE_NOT_SELECTABLE = "language_not_selectable"
DIAG_LANGUAGE_REFINEMENT_ACCEPTED = "language_refinement_accepted"

_BCP47_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
_PRIVATE_USE_RE = re.compile(r"^(?:x|i)(?:-[A-Za-z0-9]{1,8})+$", re.IGNORECASE)

#: A BCP-47 *primary* language subtag is 2-3 alpha (ISO 639-1/-2/-3, for example, ``en``,
#: ``yue``) or 4 alpha (reserved) -- never a free-form word. Registered 5-8 long
#: primary subtags exist but are vanishingly rare in ASR; rejecting bare long
#: words (``"Chinese"``, ``"English"``) catches the common native-name
#: misconfiguration loudly while subtagged forms stay permissive (see
#: ``is_valid_bcp47``).
_PRIMARY_SUBTAG_RE = re.compile(r"^[A-Za-z]{2,4}$")


def _canonical_case_subtag(index: int, subtag: str) -> str:
    """Apply BCP-47 canonical casing to a single subtag at ``index``.

    Casing rules (RFC 5646 §2.1.1): the primary language subtag (index 0) is
    lowercase; a 4-alpha script subtag is Title-case; a 2-alpha or 3-digit
    region subtag is UPPERCASE; everything else (variants, 3-8 alpha extended
    subtags) is lowercase. These rules apply only to subtags BEFORE the first
    singleton (a 1-char subtag such as the ``u`` extension or ``x`` private-use
    marker); :func:`normalize_bcp47` lowercases everything after a singleton and
    never routes those subtags here. Casing is purely cosmetic -- membership
    comparisons remain exact because every tag is canonicalized identically.

    Args:
        index: The subtag's position (0 = primary language).
        subtag: The subtag text (no separators, not preceded by a singleton).

    Returns:
        The subtag in canonical casing.
    """
    lower = subtag.lower()
    if index == 0:
        return lower
    if len(subtag) == 4 and subtag.isalpha():
        return lower.capitalize()  # script subtag -> Titlecase (for example, ``Hans``)
    if subtag.isalpha() and len(subtag) == 2:
        return subtag.upper()  # alpha-2 region -> UPPERCASE (for example, ``US``)
    if subtag.isdigit() and len(subtag) == 3:
        return subtag  # numeric-3 region (for example, ``001``) -> unchanged
    return lower


def normalize_bcp47(tag: str) -> str:
    """Normalize a BCP 47 language tag into a consistent, canonical form.

    Trims/replaces separators, then applies BCP-47 canonical casing
    (language lowercase, script Titlecase, region UPPERCASE) so values echoed
    back to applications -- for example, ``detected_language`` and diagnostic
    ``provided``/``effective`` fields -- read canonically (``zh-Hans``, not
    ``zh-hans``). Per RFC 5646 §2.1.1 the script/region conventions apply only
    before the first singleton (a 1-char subtag); every subtag after it --
    extension and private-use subtags -- stays lowercase
    (``zh-Hans-u-co-pinyin``, never ``u-CO``). Membership comparisons are
    unaffected: both the declared set and the request are canonicalized through
    this same function, so matching stays case-insensitive in effect.

    Args:
        tag: Input language tag.

    Returns:
        The canonicalized tag (separators normalized to ``-``, canonical casing).

    Raises:
        ValueError: If ``tag`` is empty or only whitespace.
    """
    normalized = tag.strip().replace("_", "-")
    if not normalized:
        raise ValueError("Language tag must not be empty.")
    canonical: list[str] = []
    seen_singleton = False
    for index, part in enumerate(normalized.split("-")):
        canonical.append(part.lower() if seen_singleton else _canonical_case_subtag(index, part))
        if len(part) == 1:
            seen_singleton = True
    return "-".join(canonical)


def is_valid_bcp47(tag: str) -> bool:
    """Return ``True`` if *tag* appears to be a valid BCP 47 language tag.

    This validation is intentionally permissive and focuses on rejecting obvious
    errors (empty segments, invalid characters). It supports private-use tags
    like ``x-foo`` and the special ``und`` tag for undetermined language.

    A *single-subtag* tag MUST be a plausible primary language subtag (2-4
    alpha, per ISO 639); this rejects free-form native language names such as
    ``"Chinese"`` or ``"English"`` loudly rather than silently accepting a
    misconfiguration (adapters are responsible for mapping native names to
    BCP-47; see the language design note). Multi-subtag forms (``"zh-Hans"``)
    stay permissive because the extra structure already rules out a stray word.

    Args:
        tag: Candidate language tag.

    Returns:
        ``True`` if the tag passes basic validation, otherwise ``False``.
    """
    try:
        normalized = normalize_bcp47(tag)
    except ValueError:
        return False

    if normalized == "und":
        return True
    if _PRIVATE_USE_RE.match(normalized):
        return True
    if _BCP47_RE.match(normalized) is None:
        return False
    # A lone subtag must look like a primary language subtag, not a word.
    if "-" not in normalized:
        return _PRIMARY_SUBTAG_RE.match(normalized) is not None
    return True


def validate_detected_language(value: str | None) -> str | None:
    """Validate and canonicalize a ``detected_language`` value.

    ``detected_language`` is the language an engine *resolved to*, so it must
    be a concrete, well-formed BCP-47 tag -- never the reserved ``auto``
    directive (a request to detect, not a detection result) and never a native
    name like ``"English"``. The single shared implementation behind the
    ``TranscriptionResult.detected_language`` and
    ``TranscriptionEvent.detected_language`` field validators: the event field
    is the reconnect-continuity mechanism (it feeds the next
    session's ``language``), so the two sides MUST accept exactly the same
    tags or continuity breaks.

    Args:
        value: The candidate detected-language tag, or ``None``.

    Returns:
        The canonicalized tag, or ``None`` when not applicable.

    Raises:
        ValueError: If ``value`` is the reserved ``auto`` token or is not a
            well-formed BCP-47 tag.
    """
    if value is None:
        return None
    if not is_valid_bcp47(value):
        raise ValueError(
            f"detected_language is not a well-formed BCP-47 tag: {value!r} "
            "(for example, 'en', 'en-US', 'zh-Hans')."
        )
    normalized = normalize_bcp47(value)
    if normalized == AUTO:
        raise ValueError(
            "detected_language MUST be a concrete detected tag, not the reserved 'auto'."
        )
    return normalized


def effective_language(
    request_language: str | None,
    default_language: str | None,
    *,
    has_language_axis: bool,
    runtime_override_supported: bool,
) -> str | None:
    """Resolve the language in effect for a request.

    Args:
        request_language: The per-request language, if any.
        default_language: The engine's default language.
        has_language_axis: Whether the engine exposes a language axis.
        runtime_override_supported: Whether per-request override is supported.

    Returns:
        The effective language tag / ``"auto"``, or ``None`` if the engine has
        no language axis.
    """
    if runtime_override_supported and request_language is not None:
        return request_language
    if has_language_axis:
        return default_language
    return None


def effective_candidate_languages(
    effective_lang: str | None,
    request_candidates: list[str] | None,
    default_candidates: list[str] | None,
    *,
    candidate_supported: bool,
    detectable_languages: Collection[str],
    max_count: int | None,
    strict: bool,
    mode: ModeName | None = None,
) -> tuple[list[str] | None, list[Diagnostic]]:
    """Resolve the candidate languages in effect for a request.

    Args:
        effective_lang: The resolved effective language.
        request_candidates: Per-request candidate languages, if any.
        default_candidates: The engine's default candidate languages.
        candidate_supported: Whether candidate languages are supported.
        detectable_languages: Languages detectable in ``auto`` mode. Tags are
            canonicalized here (idempotent on already-canonical input; the
            engine base passes its pre-canonicalized, ConfigError-checked set).
        max_count: Maximum candidate count, if constrained.
        strict: Whether to raise (vs truncate/drop + diagnostic) on violations.
        mode: The mode being resolved (``"batch"`` / ``"streaming"``; the
            signature enforces the closed set), or ``None`` when unknown
            (direct callers outside the engine pipeline); carried on the
            strict-mode
            :class:`UnsupportedFeatureError` so the rejection reads like every
            other strict gate rejection.

    Returns:
        A ``(candidates, diagnostics)`` pair; ``candidates`` is ``None`` when not
        applicable. A ``candidate_languages_ignored`` diagnostic is emitted only
        when a candidate list was actually provided (per request or per the
        engine default) but the engine/mode does not support candidate
        languages -- never when no list was provided at all (there is nothing to
        ignore), so an ``auto`` engine without candidate support stays
        diagnostic-free on ordinary requests.

    Raises:
        ValueError: Independent of ``strict``, if a candidate is a malformed
            BCP-47 tag or the reserved ``"auto"`` token -- once per-item
            validation is reached: per spec §LANG R3 the unsupported-capability
            short-circuit (step 3) runs FIRST, so when the engine/mode does not
            support candidate languages the provided list is ignored with a
            diagnostic and its items are never validated here. Direct callers
            relying on unconditional malformed-item rejection get it from
            :class:`~standard_asr.contract.params.RuntimeParams` construction, which
            validates every candidate before any resolution runs (the engine
            pipeline is always covered by that). Also raised if a
            ``detectable_languages`` entry is empty/whitespace (engine paths
            pre-validate this into a ``ConfigError``; the bare error is the
            direct-call contract). These are caller code bugs, never policy.
        UnsupportedFeatureError: In strict mode, on a valid-but-unreachable
            candidate list -- a candidate not in ``detectable_languages`` or a
            list over ``max_count``. This is the standard strict-gate rejection
            type (spec, Runtime Parameters R2), so every transport maps it to
            the same client-error verdict as any other strict rejection (the
            server's 422) instead of an internal-error 500.
    """
    diagnostics: list[Diagnostic] = []
    if effective_lang != AUTO:
        return None, diagnostics

    # Select the effective list (request overrides default) BEFORE the
    # supported check. The "candidate languages ignored" diagnostic
    # asserts that something *was* ignored, so it MUST NOT fire when nothing was
    # provided: with no list there is nothing to ignore, and emitting a warning
    # anyway injects a false diagnostic into the most common path (an
    # ``auto`` engine that does not support candidate languages -- for example, most
    # local Whisper-family engines -- gets this warning on every ordinary
    # request). The spec carve-out still holds: a list that *was* provided but is
    # unsupported is reported (never raised), independent of strict/best_effort.
    chosen = request_candidates if request_candidates is not None else default_candidates
    if not chosen:
        return None, diagnostics

    if not candidate_supported:
        diagnostics.append(
            Diagnostic(
                level="warning",
                code=DIAG_CANDIDATE_LANGUAGES_IGNORED,
                message=(
                    "Candidate languages ignored: this engine/mode does not "
                    "support candidate languages."
                ),
                param="candidate_languages",
                provided=to_json_value(list(chosen)),
                effective=None,
            )
        )
        return None, diagnostics

    # Ordering: dedup-preserving-order FIRST, then validate each
    # surviving entry. Deduplicating before membership ensures a repeated
    # non-detectable candidate is reported (or dropped) exactly once.
    deduped: list[str] = []
    deduped_seen: set[str] = set()
    for tag in chosen:
        # A malformed tag is an invalid *value*, not an unsupported feature, so
        # -- like the scalar `language` validator (params.py) and the
        # 'auto' guard below -- it ALWAYS raises, independent of strict/
        # best_effort. Validating here keeps a common mistake ('english' instead
        # of 'en') from being silently dropped or misreported as "not detectable".
        # This runs before the 'auto' check so an empty/whitespace tag (which is
        # not 'auto') is attributed as malformed rather than tripping the
        # normalizer's empty-tag ValueError; 'auto' is well-formed by this guard
        # (it is a real lowercase word), so it survives to the reserved-word check.
        if not is_valid_bcp47(tag):
            raise ValueError(
                f"candidate_languages contains a malformed BCP-47 tag {tag!r} "
                "(for example, 'en', 'en-US', 'zh-Hans')."
            )
        norm = normalize_bcp47(tag)
        # 'auto' is a directive, not a candidate; its presence is a caller bug,
        # so it ALWAYS raises -- independent of strict/best_effort -- mirroring
        # the provider_params "always-raise on a code bug" policy
        # (candidate_languages cannot contain 'auto').
        # Compare AFTER normalization so 'AUTO'/'Auto'/'auto' are all rejected
        # with this explicit reason, not misreported as "not detectable".
        if norm == AUTO:
            raise ValueError("candidate_languages cannot contain 'auto'.")
        if norm in deduped_seen:
            continue
        deduped_seen.add(norm)
        deduped.append(norm)

    # Canonicalize the DECLARED side exactly like the candidates above: direct
    # callers may pass detectable_languages as raw non-canonical declarations,
    # and a raw 'zh-hans' must not misreport a canonical 'zh-Hans' candidate as
    # non-detectable. normalize_bcp47 is idempotent on already-canonical tags,
    # so the engine base's pre-canonicalized set passes through unchanged (and,
    # having been ConfigError-checked there, can never trip the empty-tag
    # ValueError mid-request).
    detectable = {normalize_bcp47(t) for t in detectable_languages}
    result: list[str] = []
    for norm in deduped:
        if norm not in detectable:
            if strict:
                # Valid-but-unreachable: the standard strict-gate rejection
                # (spec §RT R2), NOT a bare ValueError -- a bare ValueError is
                # reserved for caller code bugs (malformed / 'auto' above) and
                # would surface as an internal-error 500 through the server.
                raise UnsupportedFeatureError(
                    f"Candidate language {norm!r} is not detectable by this engine.",
                    param="candidate_languages",
                    mode=mode,
                    hint=(
                        "Request only detectable_languages members, or use "
                        "best_effort to drop non-detectable candidates with a "
                        "diagnostic."
                    ),
                )
            diagnostics.append(
                Diagnostic(
                    level="warning",
                    code=DIAG_CANDIDATE_LANGUAGE_DROPPED,
                    message=f"Dropped non-detectable candidate {norm!r}.",
                    param="candidate_languages",
                    provided=norm,
                    effective=None,
                )
            )
            continue
        result.append(norm)

    if max_count is not None and len(result) > max_count:
        if strict:
            raise UnsupportedFeatureError(
                f"candidate_languages has {len(result)} entries; max is {max_count}.",
                param="candidate_languages",
                mode=mode,
                hint=(
                    f"Pass at most {max_count} candidates, or use best_effort "
                    "to truncate with a diagnostic."
                ),
            )
        kept = result[:max_count]
        dropped = result[max_count:]
        diagnostics.append(
            Diagnostic(
                level="warning",
                code=DIAG_CANDIDATE_LANGUAGES_TRUNCATED,
                message=(
                    f"Truncated candidate languages to {max_count}: kept {kept}, dropped {dropped}."
                ),
                param="candidate_languages",
                provided=to_json_value(result),
                effective=to_json_value(kept),
            )
        )
        result = kept

    return (result or None), diagnostics


__all__ = [
    "AUTO",
    "DIAG_CANDIDATE_LANGUAGES_IGNORED",
    "DIAG_CANDIDATE_LANGUAGES_TRUNCATED",
    "DIAG_CANDIDATE_LANGUAGE_DROPPED",
    "DIAG_LANGUAGE_FELL_BACK",
    "DIAG_LANGUAGE_NOT_SELECTABLE",
    "DIAG_LANGUAGE_REFINEMENT_ACCEPTED",
    "effective_candidate_languages",
    "effective_language",
    "is_valid_bcp47",
    "normalize_bcp47",
    "validate_detected_language",
]
