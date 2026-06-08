# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for BCP 47 language helpers and resolution."""

import pytest

from standard_asr.language import (
    AUTO,
    effective_candidate_languages,
    effective_language,
    is_valid_bcp47,
    normalize_bcp47,
)


def test_normalize_bcp47_canonical_casing() -> None:
    # LANG-2: canonical BCP-47 casing -- language lower, script Title, region UPPER.
    assert normalize_bcp47("EN-US") == "en-US"
    assert normalize_bcp47(" zh_cn ") == "zh-CN"
    assert normalize_bcp47("ZH-HANS") == "zh-Hans"
    assert normalize_bcp47("zh-hans-cn") == "zh-Hans-CN"
    assert normalize_bcp47("es-419") == "es-419"  # numeric region unchanged
    assert normalize_bcp47("en") == "en"


def test_normalize_bcp47_membership_is_case_insensitive_in_effect() -> None:
    # Two differently-cased spellings canonicalize to the same value, so
    # membership comparisons remain exact regardless of input casing.
    assert normalize_bcp47("zh-Hans") == normalize_bcp47("ZH-HANS") == "zh-Hans"


def test_is_valid_bcp47() -> None:
    assert is_valid_bcp47("en") is True
    assert is_valid_bcp47("en-US") is True
    assert is_valid_bcp47("und") is True
    assert is_valid_bcp47("x-private") is True
    assert is_valid_bcp47("") is False
    assert is_valid_bcp47("en--US") is False
    assert is_valid_bcp47("en@US") is False


def test_is_valid_bcp47_rejects_native_names() -> None:
    # Free-form native language names are NOT BCP-47 -> fail loud (adapters map).
    assert is_valid_bcp47("Chinese") is False
    assert is_valid_bcp47("English") is False
    assert is_valid_bcp47("Mandarin") is False


def test_is_valid_bcp47_accepts_real_codes() -> None:
    assert is_valid_bcp47("yue") is True  # 3-letter ISO 639-3
    assert is_valid_bcp47("zh-Hans") is True  # subtagged form stays permissive
    assert is_valid_bcp47("zh-Hant-HK") is True


def test_effective_language_runtime_override() -> None:
    assert (
        effective_language("fr", "en", has_language_axis=True, runtime_override_supported=True)
        == "fr"
    )


def test_effective_language_falls_back_to_default() -> None:
    assert (
        effective_language("fr", "en", has_language_axis=True, runtime_override_supported=False)
        == "en"
    )


def test_effective_language_no_axis() -> None:
    assert (
        effective_language(None, None, has_language_axis=False, runtime_override_supported=False)
        is None
    )


def test_effective_candidates_not_auto() -> None:
    result, diags = effective_candidate_languages(
        "en",
        ["ja"],
        None,
        candidate_supported=True,
        detectable_languages=["ja"],
        max_count=3,
        strict=True,
    )
    assert result is None
    assert diags == []


def test_effective_candidates_unsupported_diagnostic() -> None:
    result, diags = effective_candidate_languages(
        AUTO,
        ["ja"],
        None,
        candidate_supported=False,
        detectable_languages=["ja"],
        max_count=3,
        strict=True,
    )
    assert result is None
    assert diags[0].code == "candidate_languages_ignored"


def test_effective_candidates_none_when_no_chosen_list() -> None:
    # auto + supported but neither a request nor a default candidate list: there
    # is nothing to constrain detection to, so the result is None (no diagnostic).
    result, diags = effective_candidate_languages(
        AUTO,
        None,
        None,
        candidate_supported=True,
        detectable_languages=["en", "ja"],
        max_count=3,
        strict=True,
    )
    assert result is None
    assert diags == []


def test_effective_candidates_dedup_and_order() -> None:
    result, _ = effective_candidate_languages(
        AUTO,
        ["ja", "en", "ja"],
        None,
        candidate_supported=True,
        detectable_languages=["en", "ja", "ko"],
        max_count=3,
        strict=True,
    )
    assert result == ["ja", "en"]


def test_effective_candidates_rejects_auto() -> None:
    with pytest.raises(ValueError):
        effective_candidate_languages(
            AUTO,
            ["auto"],
            None,
            candidate_supported=True,
            detectable_languages=["en"],
            max_count=3,
            strict=True,
        )


def test_effective_candidates_strict_non_detectable_raises() -> None:
    with pytest.raises(ValueError):
        effective_candidate_languages(
            AUTO,
            ["zz"],
            None,
            candidate_supported=True,
            detectable_languages=["en"],
            max_count=3,
            strict=True,
        )


def test_effective_candidates_best_effort_drops_non_detectable() -> None:
    result, diags = effective_candidate_languages(
        AUTO,
        ["en", "zz"],
        None,
        candidate_supported=True,
        detectable_languages=["en"],
        max_count=3,
        strict=False,
    )
    assert result == ["en"]
    assert any(d.code == "candidate_language_dropped" for d in diags)


def test_effective_candidates_strict_over_max_raises() -> None:
    with pytest.raises(ValueError):
        effective_candidate_languages(
            AUTO,
            ["en", "ja", "ko"],
            None,
            candidate_supported=True,
            detectable_languages=["en", "ja", "ko"],
            max_count=2,
            strict=True,
        )


def test_effective_candidates_best_effort_truncates() -> None:
    result, diags = effective_candidate_languages(
        AUTO,
        ["en", "ja", "ko"],
        None,
        candidate_supported=True,
        detectable_languages=["en", "ja", "ko"],
        max_count=2,
        strict=False,
    )
    assert result == ["en", "ja"]
    # LANG-3: the diagnostic carries the final effective list and the dropped
    # tags, not just a count.
    diag = next(d for d in diags if d.code == "candidate_languages_truncated")
    assert diag.provided == ["en", "ja", "ko"]
    assert diag.effective == ["en", "ja"]
    assert "ko" in diag.message and "['en', 'ja']" in diag.message


def test_dedup_before_membership_single_drop_diagnostic() -> None:
    # A repeated NON-detectable candidate must be deduped first, so it is
    # reported / dropped exactly once (not twice).
    result, diags = effective_candidate_languages(
        AUTO,
        ["zz", "en", "zz"],
        None,
        candidate_supported=True,
        detectable_languages=["en"],
        max_count=3,
        strict=False,
    )
    assert result == ["en"]
    dropped = [d for d in diags if d.code == "candidate_language_dropped"]
    assert len(dropped) == 1


def test_auto_in_candidates_always_raises_even_best_effort() -> None:
    # 'auto' in a candidate list is a caller bug -> always raises, independent
    # of strict / best_effort.
    with pytest.raises(ValueError, match="auto"):
        effective_candidate_languages(
            AUTO,
            ["en", "auto"],
            None,
            candidate_supported=True,
            detectable_languages=["en"],
            max_count=3,
            strict=False,
        )


def test_effective_candidates_defaults_when_no_request() -> None:
    result, _ = effective_candidate_languages(
        AUTO,
        None,
        ["en"],
        candidate_supported=True,
        detectable_languages=["en"],
        max_count=3,
        strict=True,
    )
    assert result == ["en"]
