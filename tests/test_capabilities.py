# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the hierarchical capability system."""

from __future__ import annotations

from standard_asr.capabilities import (
    BatchCapabilities,
    CandidateLanguagesCap,
    CandidateLanguagesConstraints,
    DeclaredCapabilities,
    FlagCap,
    GuidanceCaps,
    LanguageCaps,
    PhraseHintsCap,
    PromptCap,
    ReconnectCap,
    StreamingCapabilities,
    WordTimestampsCap,
)


def _rich() -> DeclaredCapabilities:
    return DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(
                runtime_override=FlagCap(supported=True),
                candidate_languages=CandidateLanguagesCap(
                    supported=True,
                    constraints=CandidateLanguagesConstraints(max=3),
                ),
            ),
            word_timestamps=WordTimestampsCap(
                supported=True, granularities=["word", "segment"]
            ),
            guidance=GuidanceCaps(prompt=PromptCap(supported=True)),
        ),
        streaming=StreamingCapabilities(
            emits_partials=FlagCap(supported=True),
            reconnect=ReconnectCap(mode="lossy"),
        ),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
    )


def test_supports_leaf_flags() -> None:
    caps = _rich()
    assert caps.supports("batch.language.runtime_override") is True
    assert caps.supports("batch.word_timestamps") is True
    assert caps.supports("streaming.emits_partials") is True


def test_supports_top_level_orthogonal() -> None:
    caps = _rich()
    assert caps.supports("streaming_input") is True
    assert caps.supports("streaming_output") is True


def test_supports_fail_closed_missing_key() -> None:
    caps = _rich()
    # Not declared under streaming guidance -> False.
    assert caps.supports("streaming.guidance.phrase_hints") is False
    # batch guidance phrase_hints default supported=False.
    assert caps.supports("batch.guidance.phrase_hints") is False
    assert caps.supports("batch.totally.unknown.path") is False


def test_supports_mode_node() -> None:
    caps = _rich()
    assert caps.supports("streaming.reconnect") is True  # lossy != unsupported
    caps2 = DeclaredCapabilities(
        streaming=StreamingCapabilities(reconnect=ReconnectCap(mode="unsupported"))
    )
    assert caps2.supports("streaming.reconnect") is False


def test_omitted_streaming_domain_unsupported() -> None:
    caps = DeclaredCapabilities(batch=BatchCapabilities())
    assert caps.supports("streaming") is False
    assert caps.supports("streaming.emits_partials") is False
    assert caps.supports("batch") is True


def test_default_is_fail_closed() -> None:
    caps = DeclaredCapabilities()
    assert caps.supports("batch") is False
    assert caps.supports("streaming_input") is False


def test_covers_subset_invariant() -> None:
    declared = _rich()
    # Effective narrows: drop word_timestamps support.
    effective = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        ),
        streaming_input=FlagCap(supported=True),
    )
    assert declared.covers(effective) is True


def test_covers_rejects_widening() -> None:
    declared = DeclaredCapabilities(batch=BatchCapabilities())
    # Effective claims more than declared -> not covered.
    effective = DeclaredCapabilities(
        batch=BatchCapabilities(word_timestamps=WordTimestampsCap(supported=True))
    )
    assert declared.covers(effective) is False


def test_unknown_x_namespace_key_tolerated() -> None:
    # Forward-compat: extra keys parse without error and are queryable.
    caps = DeclaredCapabilities.model_validate(
        {
            "batch": {"x_acme_beamsearch": {"supported": True}},
            "streaming_input": {"supported": False},
        }
    )
    assert caps.supports("batch.x_acme_beamsearch") is True
    assert caps.supports("batch.x_acme_unknown") is False
