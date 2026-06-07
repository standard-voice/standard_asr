"""Helpers for validating and normalizing BCP 47 language tags."""

from __future__ import annotations

import re

#: Reserved token meaning "let the engine auto-detect the language".
#: This is NOT a BCP-47 tag; it is a Standard ASR reserved word.
AUTO = "auto"

_BCP47_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
_PRIVATE_USE_RE = re.compile(r"^(?:x|i)(?:-[A-Za-z0-9]{1,8})+$", re.IGNORECASE)


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
    return _BCP47_RE.match(normalized) is not None


__all__ = ["AUTO", "is_valid_bcp47", "normalize_bcp47"]
