# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for BCP 47 language helpers and resolution."""

import pytest

from standard_asr.language import (
    AUTO,
    DIAG_CANDIDATE_LANGUAGE_DROPPED,
    DIAG_CANDIDATE_LANGUAGES_IGNORED,
    DIAG_CANDIDATE_LANGUAGES_TRUNCATED,
    effective_candidate_languages,
    effective_language,
    is_valid_bcp47,
    normalize_bcp47,
)


def test_normalize_bcp47_canonical_casing() -> None:
    # Canonical BCP-47 casing -- language lower, script Title, region UPPER.
    assert normalize_bcp47("EN-US") == "en-US"
    assert normalize_bcp47(" zh_cn ") == "zh-CN"
    assert normalize_bcp47("ZH-HANS") == "zh-Hans"
    assert normalize_bcp47("zh-hans-cn") == "zh-Hans-CN"
    assert normalize_bcp47("es-419") == "es-419"  # numeric region unchanged
    assert normalize_bcp47("en") == "en"


def test_normalize_bcp47_lowercases_after_singleton() -> None:
    # RFC 5646 §2.1.1: the script/region casing conventions
    # apply only BEFORE the first singleton; extension subtags (after 'u') and
    # private-use subtags (after 'x') stay lowercase. 'co' is an extension key
    # here, not a region -- never 'u-CO'.
    assert normalize_bcp47("zh-Hans-u-co-pinyin") == "zh-Hans-u-co-pinyin"
    assert normalize_bcp47("ZH-HANS-U-CO-PINYIN") == "zh-Hans-u-co-pinyin"
    assert normalize_bcp47("en-x-private-AB") == "en-x-private-ab"
    # Ordinary casing before any singleton is unaffected.
    assert normalize_bcp47("en-us") == "en-US"
    assert normalize_bcp47("zh-hans") == "zh-Hans"


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
    # A candidate list WAS provided but the engine/mode does not support
    # candidate languages: the ignored diagnostic is legitimate here and MUST
    # carry the ignored list as `provided` and `effective=None` (spec LANG R3
    # step 3 / §3: diagnostic includes which param, the reason, and the
    # effective value). Independent of strict -- this carve-out never raises.
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
    assert len(diags) == 1
    assert diags[0].code == DIAG_CANDIDATE_LANGUAGES_IGNORED
    assert diags[0].param == "candidate_languages"
    assert diags[0].provided == ["ja"]
    assert diags[0].effective is None


def test_effective_candidates_unsupported_uses_default_list_in_provided() -> None:
    # When no per-request list is given but a default candidate list is, the
    # ignored diagnostic reports the DEFAULT list as `provided` (it is the list
    # that was ignored). minor: provided must be populated.
    result, diags = effective_candidate_languages(
        AUTO,
        None,
        ["ja", "en"],
        candidate_supported=False,
        detectable_languages=["ja"],
        max_count=3,
        strict=False,
    )
    assert result is None
    assert len(diags) == 1
    assert diags[0].code == DIAG_CANDIDATE_LANGUAGES_IGNORED
    assert diags[0].provided == ["ja", "en"]


@pytest.mark.parametrize("strict", [True, False])
def test_effective_candidates_unsupported_no_list_emits_no_diagnostic(strict: bool) -> None:
    # When candidate languages are unsupported AND no list was
    # provided (neither per-request nor default), there is nothing to ignore,
    # so NO diagnostic is emitted. The previous implementation injected a false
    # `candidate_languages_ignored` warning on every ordinary auto request of a
    # non-candidate engine (most local Whisper-family engines), polluting the
    # most common path. Holds in both strict and best_effort.
    result, diags = effective_candidate_languages(
        AUTO,
        None,
        None,
        candidate_supported=False,
        detectable_languages=["en", "ja"],
        max_count=3,
        strict=strict,
    )
    assert result is None
    assert diags == []


def test_effective_candidates_unsupported_empty_list_emits_no_diagnostic() -> None:
    # An explicitly empty request list ([]) is "nothing to constrain", same as
    # no list: it must not trigger the ignored diagnostic either.
    result, diags = effective_candidate_languages(
        AUTO,
        [],
        None,
        candidate_supported=False,
        detectable_languages=["en"],
        max_count=3,
        strict=True,
    )
    assert result is None
    assert diags == []


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
    dropped = [d for d in diags if d.code == DIAG_CANDIDATE_LANGUAGE_DROPPED]
    assert len(dropped) == 1
    # minor: the dropped diagnostic carries the dropped tag as
    # `provided` and `effective=None` (it took no effect).
    assert dropped[0].provided == "zz"
    assert dropped[0].effective is None


def test_detectable_membership_canonicalizes_declared_side() -> None:
    # detectable_languages may reach here as a non-canonical
    # class-level default (pydantic does not run field validators on defaults).
    # A canonical candidate ('zh-Hans') must match the raw declaration
    # ('zh-hans') instead of raising in strict mode (or being dropped as
    # "non-detectable" in best_effort) for a language the engine CAN detect.
    result, diags = effective_candidate_languages(
        AUTO,
        ["zh-Hans", "pt-BR"],
        None,
        candidate_supported=True,
        detectable_languages=["zh-hans", "en", "pt-br"],
        max_count=3,
        strict=True,
    )
    assert result == ["zh-Hans", "pt-BR"]
    assert diags == []


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
    # The diagnostic carries the final effective list and the dropped
    # tags, not just a count.
    diag = next(d for d in diags if d.code == DIAG_CANDIDATE_LANGUAGES_TRUNCATED)
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
    dropped = [d for d in diags if d.code == DIAG_CANDIDATE_LANGUAGE_DROPPED]
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


@pytest.mark.parametrize("strict", [True, False])
def test_malformed_candidate_always_raises(strict: bool) -> None:
    # A malformed BCP-47 candidate ('english' instead of 'en') is a
    # caller bug -> always raises with a clear malformed-tag message naming the
    # offending tag, independent of strict / best_effort. It must NOT be silently
    # dropped (best_effort) or misreported as "not detectable" (strict).
    with pytest.raises(ValueError, match=r"malformed.*'english'"):
        effective_candidate_languages(
            AUTO,
            ["english"],
            None,
            candidate_supported=True,
            detectable_languages=["en"],
            max_count=3,
            strict=strict,
        )


@pytest.mark.parametrize("candidates", [["AUTO"], ["Auto", "en"], ["auto"]])
def test_reserved_auto_token_matched_case_insensitively(candidates: list[str]) -> None:
    # The reserved 'auto' token is matched case-insensitively (after
    # normalization), so 'AUTO' / 'Auto' / 'auto' all hit the explicit
    # reserved-word error rather than being misreported as "not detectable".
    with pytest.raises(ValueError, match="MUST NOT contain 'auto'"):
        effective_candidate_languages(
            AUTO,
            candidates,
            None,
            candidate_supported=True,
            detectable_languages=["en"],
            max_count=3,
            strict=True,
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
