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


def normalize_bcp47(tag: str) -> str:
    """Normalize a BCP 47 language tag into a consistent, lowercase form.

    Args:
        tag: Input language tag.

    Returns:
        Normalized tag in lowercase with underscores replaced by hyphens.

    Raises:
        ValueError: If ``tag`` is empty or only whitespace.
    """
    normalized = tag.strip().replace("_", "-")
    if not normalized:
        raise ValueError("Language tag must not be empty.")
    return normalized.lower()


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
        ValueError: In strict mode, on an invalid or over-limit candidate list,
            or if the reserved ``"auto"`` token appears in the list.
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
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="candidate_languages_truncated",
                message=f"Truncated candidate languages to {max_count}.",
                param="candidate_languages",
                provided=len(result),
                effective=max_count,
            )
        )
        result = result[:max_count]

    return (result or None), diagnostics


__all__ = [
    "AUTO",
    "effective_candidate_languages",
    "effective_language",
    "is_valid_bcp47",
    "normalize_bcp47",
]
