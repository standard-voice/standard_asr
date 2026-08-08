# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for BCP 47 language helpers and resolution."""

import pytest

from standard_asr.contract.exceptions import UnsupportedFeatureError
from standard_asr.contract.language import (
    AUTO,
    DIAG_CANDIDATE_LANGUAGE_DROPPED,
    DIAG_CANDIDATE_LANGUAGES_IGNORED,
    DIAG_CANDIDATE_LANGUAGES_TRUNCATED,
    DIAG_LANGUAGE_FELL_BACK,
    DIAG_LANGUAGE_NOT_SELECTABLE,
    DIAG_LANGUAGE_REFINEMENT_ACCEPTED,
    effective_candidate_languages,
    effective_language,
    is_valid_bcp47,
    normalize_bcp47,
)


def test_language_diagnostic_code_constants_match_their_wire_literals() -> None:
    """Pin each language-family DIAG_* constant to its exact wire literal."""
    # The diagnostic ``code`` is a wire-visible contract consumers match on, so
    # the constant and its literal value are pinned together exactly once: a
    # silent rename of either side breaks here instead of in a consumer.
    assert DIAG_CANDIDATE_LANGUAGES_IGNORED == "candidate_languages_ignored"
    assert DIAG_CANDIDATE_LANGUAGE_DROPPED == "candidate_language_dropped"
    assert DIAG_CANDIDATE_LANGUAGES_TRUNCATED == "candidate_languages_truncated"
    # Emitted by EngineBase._resolve_language_axis, but their single source of
    # truth is this contract module -- so a consumer imports every
    # language-family code from one place.
    assert DIAG_LANGUAGE_FELL_BACK == "language_fell_back"
    assert DIAG_LANGUAGE_NOT_SELECTABLE == "language_not_selectable"
    assert DIAG_LANGUAGE_REFINEMENT_ACCEPTED == "language_refinement_accepted"


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
    # carry the ignored list as `provided` and `effective=None` (the diagnostic
    # names which param, the reason, and the effective value). Independent of
    # strict -- this carve-out never raises.
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
    """Strict mode rejects a non-detectable candidate as UnsupportedFeatureError."""
    # A well-formed candidate the engine simply cannot detect is
    # valid-but-unreachable POLICY, not a caller code bug: it MUST be the
    # standard strict-gate rejection (UnsupportedFeatureError, spec RT R2) so
    # every transport maps it to the same client-error verdict (the server's
    # 422) as any other strict rejection -- never a bare ValueError, which
    # transports treat as an internal 500.
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        effective_candidate_languages(
            AUTO,
            ["zz"],
            None,
            candidate_supported=True,
            detectable_languages=["en"],
            max_count=3,
            strict=True,
        )

    exc = excinfo.value
    assert exc.param == "candidate_languages"
    # No mode was supplied by this direct call, so none is claimed.
    assert exc.mode is None
    assert "detectable" in str(exc)
    assert exc.hint is not None and "best_effort" in exc.hint
    # Explicitly NOT a ValueError: that type is reserved for caller code bugs
    # (malformed tag / 'auto'), and conflating the two is the regression this
    # pins.
    assert not isinstance(exc, ValueError)


def test_effective_candidates_strict_non_detectable_carries_the_mode() -> None:
    """The strict rejection carries the mode the engine pipeline passes."""
    # When the caller knows the mode (the engine pipeline passes
    # mode="batch"/"streaming"), the rejection carries it, so the error reads
    # like every other strict gate rejection.
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        effective_candidate_languages(
            AUTO,
            ["zz"],
            None,
            candidate_supported=True,
            detectable_languages=["en"],
            max_count=3,
            strict=True,
            mode="streaming",
        )

    assert excinfo.value.mode == "streaming"
    assert excinfo.value.param == "candidate_languages"


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
    """Strict mode rejects an over-``max`` candidate list as UnsupportedFeatureError."""
    # Over-``max`` is the same class of rejection as non-detectable: a list the
    # engine cannot honour, not a malformed value -> UnsupportedFeatureError.
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        effective_candidate_languages(
            AUTO,
            ["en", "ja", "ko"],
            None,
            candidate_supported=True,
            detectable_languages=["en", "ja", "ko"],
            max_count=2,
            strict=True,
            mode="batch",
        )

    exc = excinfo.value
    assert exc.param == "candidate_languages"
    assert exc.mode == "batch"
    assert "3 entries" in str(exc) and "max is 2" in str(exc)
    assert exc.hint is not None and "at most 2" in exc.hint
    assert not isinstance(exc, ValueError)


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


@pytest.mark.parametrize("strict", [True, False])
def test_auto_in_candidates_always_raises_even_best_effort(strict: bool) -> None:
    """``'auto'`` in a candidate list raises a bare ValueError under either policy.

    Args:
        strict: The gating policy under test (parametrized).
    """
    # 'auto' in a candidate list is a caller CODE bug -> always a bare
    # ValueError, independent of strict / best_effort, and explicitly NOT the
    # UnsupportedFeatureError the policy rejections use (that type would claim
    # the engine merely lacks a feature the caller could ask differently for).
    with pytest.raises(ValueError, match="auto") as excinfo:
        effective_candidate_languages(
            AUTO,
            ["en", "auto"],
            None,
            candidate_supported=True,
            detectable_languages=["en"],
            max_count=3,
            strict=strict,
        )

    assert not isinstance(excinfo.value, UnsupportedFeatureError)


@pytest.mark.parametrize("strict", [True, False])
def test_malformed_candidate_always_raises(strict: bool) -> None:
    """A malformed candidate tag raises a bare ValueError under either policy."""
    # A malformed BCP-47 candidate ('english' instead of 'en') is a
    # caller bug -> always raises with a clear malformed-tag message naming the
    # offending tag, independent of strict / best_effort. It must NOT be silently
    # dropped (best_effort) or misreported as "not detectable" (strict).
    with pytest.raises(ValueError, match=r"malformed.*'english'") as excinfo:
        effective_candidate_languages(
            AUTO,
            ["english"],
            None,
            candidate_supported=True,
            detectable_languages=["en"],
            max_count=3,
            strict=strict,
        )

    # A code bug stays a bare ValueError even in strict mode, where the
    # valid-but-unreachable rejections raise UnsupportedFeatureError.
    assert not isinstance(excinfo.value, UnsupportedFeatureError)


@pytest.mark.parametrize("candidates", [["AUTO"], ["Auto", "en"], ["auto"]])
def test_reserved_auto_token_matched_case_insensitively(candidates: list[str]) -> None:
    # The reserved 'auto' token is matched case-insensitively (after
    # normalization), so 'AUTO' / 'Auto' / 'auto' all hit the explicit
    # reserved-word error rather than being misreported as "not detectable".
    with pytest.raises(ValueError, match="cannot contain 'auto'"):
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
