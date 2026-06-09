# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the package top-level public surface (``standard_asr.__all__``)."""

from __future__ import annotations

import standard_asr


def test_all_names_are_importable_and_unique() -> None:
    # Every name advertised in __all__ MUST resolve as a real attribute, and the
    # list MUST carry no duplicates (the public surface is a promise).
    assert len(standard_asr.__all__) == len(set(standard_asr.__all__))
    missing = [name for name in standard_asr.__all__ if not hasattr(standard_asr, name)]
    assert missing == []


def test_capability_vocabulary_is_re_exported_from_top_level() -> None:
    # X-EL-2: engine authors MUST be able to import the capability vocabulary
    # (node archetypes / cap classes / constraint models / the granularity
    # vocabulary / the unbounded helper) from the package top level without
    # reaching into the `capabilities` submodule.
    from standard_asr import (
        CandidateLanguagesCap,
        CandidateLanguagesConstraints,
        DiarizationCap,
        DiarizationConstraints,
        FinalityCap,
        FlagCap,
        GuidanceCaps,
        LanguageCaps,
        PhraseHintsCap,
        PhraseHintsConstraints,
        PromptCap,
        PromptConstraints,
        ReconnectCap,
        StreamTimestampsCap,
        WordTimestampGranularityName,
        WordTimestampsCap,
        granularity_offers_all,
    )
    from standard_asr import capabilities as caps_module

    # Dynamic X-EL-2 drift guard (mirrors the X-EL-3 granularity drift test):
    # EVERY name the capabilities submodule advertises in its `__all__` MUST be
    # re-exported from the package top level. A hardcoded list would silently miss
    # a future capability type added to capabilities.__all__ but forgotten in the
    # top-level re-export -- this set-subset assertion fails the moment one is.
    assert set(caps_module.__all__) <= set(standard_asr.__all__)

    # Each re-exported name is the very object defined in the submodule (a true
    # re-export, not a shadowing redefinition) -- checked over the full __all__,
    # not a hardcoded subset.
    for name in caps_module.__all__:
        assert getattr(standard_asr, name) is getattr(caps_module, name), name

    # The cap classes are usable straight from the top-level import.
    assert FlagCap(supported=True).is_supported is True
    assert WordTimestampsCap(supported=True, granularities=["word"]).is_supported is True
    assert (
        CandidateLanguagesCap(
            supported=True, constraints=CandidateLanguagesConstraints(max=3)
        ).is_supported
        is True
    )
    assert ReconnectCap(mode="lossy").is_supported is True
    assert FinalityCap is caps_module.FinalityCap
    assert StreamTimestampsCap is caps_module.StreamTimestampsCap
    assert LanguageCaps is caps_module.LanguageCaps
    assert GuidanceCaps is caps_module.GuidanceCaps
    assert PromptCap is caps_module.PromptCap
    assert PromptConstraints is caps_module.PromptConstraints
    assert PhraseHintsCap is caps_module.PhraseHintsCap
    assert PhraseHintsConstraints is caps_module.PhraseHintsConstraints
    assert DiarizationCap is caps_module.DiarizationCap
    assert DiarizationConstraints is caps_module.DiarizationConstraints
    # The granularity vocabulary (declaration side) is exported and non-empty.
    from typing import get_args

    assert get_args(WordTimestampGranularityName)
    assert granularity_offers_all([]) is True
