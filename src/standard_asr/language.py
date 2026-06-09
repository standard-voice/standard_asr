# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for validating, normalizing, and resolving BCP 47 language tags."""

from __future__ import annotations

import re

from .results import Diagnostic

#: Reserved token meaning "let the engine auto-detect the language".
#: This is NOT a BCP-47 tag; it is a Standard ASR reserved word.
AUTO = "auto"

_BCP47_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
_PRIVATE_USE_RE = re.compile(r"^(?:x|i)(?:-[A-Za-z0-9]{1,8})+$", re.IGNORECASE)

#: A BCP-47 *primary* language subtag is 2-3 alpha (ISO 639-1/-2/-3, e.g. ``en``,
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
    subtags, extensions) is lowercase. Casing is purely cosmetic -- membership
    comparisons remain exact because every tag is canonicalized identically.

    Args:
        index: The subtag's position (0 = primary language).
        subtag: The subtag text (no separators).

    Returns:
        The subtag in canonical casing.
    """
    lower = subtag.lower()
    if index == 0:
        return lower
    if len(subtag) == 4 and subtag.isalpha():
        return lower.capitalize()  # script subtag -> Titlecase (e.g. "Hans")
    if subtag.isalpha() and len(subtag) == 2:
        return subtag.upper()  # alpha-2 region -> UPPERCASE (e.g. "US")
    if subtag.isdigit() and len(subtag) == 3:
        return subtag  # numeric-3 region (e.g. "001") -> unchanged
    return lower


def normalize_bcp47(tag: str) -> str:
    """Normalize a BCP 47 language tag into a consistent, canonical form.

    Trims/replaces separators, then applies BCP-47 canonical casing
    (language lowercase, script Titlecase, region UPPERCASE) so values echoed
    back to applications -- e.g. ``detected_language`` and diagnostic
    ``provided``/``effective`` fields -- read canonically (``zh-Hans``, not
    ``zh-hans``). Membership comparisons are unaffected: both the declared set
    and the request are canonicalized through this same function, so matching
    stays case-insensitive in effect.

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
    parts = normalized.split("-")
    return "-".join(_canonical_case_subtag(i, part) for i, part in enumerate(parts))


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


def effective_language(
    request_language: str | None,
    default_language: str | None,
    *,
    has_language_axis: bool,
    runtime_override_supported: bool,
) -> str | None:
    """Resolve the language in effect for a request (spec LANG R2).

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
    detectable_languages: list[str],
    max_count: int | None,
    strict: bool,
) -> tuple[list[str] | None, list[Diagnostic]]:
    """Resolve the candidate languages in effect for a request (spec LANG R3).

    Args:
        effective_lang: The resolved effective language.
        request_candidates: Per-request candidate languages, if any.
        default_candidates: The engine's default candidate languages.
        candidate_supported: Whether candidate languages are supported.
        detectable_languages: Languages detectable in ``auto`` mode.
        max_count: Maximum candidate count, if constrained.
        strict: Whether to raise (vs truncate/drop + diagnostic) on violations.

    Returns:
        A ``(candidates, diagnostics)`` pair; ``candidates`` is ``None`` when not
        applicable.

    Raises:
        ValueError: Unconditionally (independent of ``strict``) if a candidate is
            a malformed BCP-47 tag or the reserved ``"auto"`` token; or, in strict
            mode, on a non-detectable or over-limit candidate list.
    """
    diagnostics: list[Diagnostic] = []
    if effective_lang != AUTO:
        return None, diagnostics
    if not candidate_supported:
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="candidate_languages_ignored",
                message="Candidate languages ignored: unsupported here.",
                param="candidate_languages",
            )
        )
        return None, diagnostics

    chosen = request_candidates if request_candidates is not None else default_candidates
    if not chosen:
        return None, diagnostics

    # R3 step 4 ordering: dedup-preserving-order FIRST, then validate each
    # surviving entry. Deduplicating before membership ensures a repeated
    # non-detectable candidate is reported (or dropped) exactly once.
    deduped: list[str] = []
    deduped_seen: set[str] = set()
    for tag in chosen:
        # 'auto' is a directive, not a candidate; its presence is a caller bug,
        # so it ALWAYS raises -- independent of strict/best_effort -- mirroring
        # the provider_params "always-raise on a code bug" policy (R3 step 4 /
        # language design note: candidate_languages MUST NOT contain 'auto').
        if tag == AUTO:
            raise ValueError("candidate_languages MUST NOT contain 'auto'.")
        # A malformed tag is an invalid *value*, not an unsupported feature, so
        # -- like the scalar `language` validator (runtime_params.py) and the
        # 'auto' guard above -- it ALWAYS raises, independent of strict/
        # best_effort. Validating here keeps a common mistake ('english' instead
        # of 'en') from being silently dropped or misreported as "not detectable".
        if not is_valid_bcp47(tag):
            raise ValueError(
                f"candidate_languages contains a malformed BCP-47 tag {tag!r} "
                "(e.g. 'en', 'en-US', 'zh-Hans')."
            )
        norm = normalize_bcp47(tag)
        if norm in deduped_seen:
            continue
        deduped_seen.add(norm)
        deduped.append(norm)

    detectable = set(detectable_languages)
    result: list[str] = []
    for norm in deduped:
        if norm not in detectable:
            if strict:
                raise ValueError(f"Candidate language {norm!r} is not detectable.")
            diagnostics.append(
                Diagnostic(
                    level="warning",
                    code="candidate_language_dropped",
                    message=f"Dropped non-detectable candidate {norm!r}.",
                    param="candidate_languages",
                    provided=norm,
                )
            )
            continue
        result.append(norm)

    if max_count is not None and len(result) > max_count:
        if strict:
            raise ValueError(f"candidate_languages has {len(result)} entries; max is {max_count}.")
        kept = result[:max_count]
        dropped = result[max_count:]
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="candidate_languages_truncated",
                message=(
                    f"Truncated candidate languages to {max_count}: kept {kept}, dropped {dropped}."
                ),
                param="candidate_languages",
                provided=result,
                effective=kept,
            )
        )
        result = kept

    return (result or None), diagnostics


__all__ = [
    "AUTO",
    "effective_candidate_languages",
    "effective_language",
    "is_valid_bcp47",
    "normalize_bcp47",
]
