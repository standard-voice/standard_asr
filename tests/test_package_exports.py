# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the package public surface.

The surface is deliberately layered: the curated top-level ``standard_asr``
namespace is the **application-developer** surface, while ``standard_asr.engine``
is the **engine-author** facade. These tests pin both surfaces and guard against
the curation silently re-flattening.
"""

from __future__ import annotations

import importlib
from typing import get_args

import pytest

import standard_asr
from standard_asr import engine as engine_facade


def test_top_level_all_names_are_importable_and_unique() -> None:
    # Every name advertised in the top-level __all__ MUST resolve as a real
    # attribute, with no duplicates (the public surface is a promise).
    assert len(standard_asr.__all__) == len(set(standard_asr.__all__))
    missing = [name for name in standard_asr.__all__ if not hasattr(standard_asr, name)]
    assert missing == []


def test_engine_facade_all_names_are_importable_and_unique() -> None:
    # The engine-author facade obeys the same promise.
    assert len(engine_facade.__all__) == len(set(engine_facade.__all__))
    missing = [name for name in engine_facade.__all__ if not hasattr(engine_facade, name)]
    assert missing == []


def test_exceptions_are_fully_re_exported_at_top_level() -> None:
    # The WHOLE error contract is application-facing, so every name the
    # exceptions submodule advertises MUST be re-exported at the package top
    # level -- and be the very object the submodule defines (a true re-export,
    # not a shadowing redefinition). This set-subset assertion fails the moment a
    # future exception is added to the submodule but forgotten at the top level.
    exceptions = importlib.import_module("standard_asr.exceptions")
    assert set(exceptions.__all__) <= set(standard_asr.__all__)
    for name in exceptions.__all__:
        assert getattr(standard_asr, name) is getattr(exceptions, name), name


def test_capability_vocabulary_is_on_the_engine_facade() -> None:
    # The capability vocabulary is the ENGINE-AUTHOR surface, so it lives on
    # ``standard_asr.engine`` -- not the application-facing top level. EVERY name
    # the capabilities submodule advertises MUST be re-exported from the facade
    # (a true re-export). A future capability type added to capabilities.__all__
    # but forgotten on the facade fails this immediately.
    caps_module = importlib.import_module("standard_asr.capabilities")
    assert set(caps_module.__all__) <= set(engine_facade.__all__)
    for name in caps_module.__all__:
        assert getattr(engine_facade, name) is getattr(caps_module, name), name

    # The cap classes are usable straight from the facade import.
    from standard_asr.engine import (
        CandidateLanguagesCap,
        CandidateLanguagesConstraints,
        FlagCap,
        ReconnectCap,
        WordTimestampGranularityName,
        WordTimestampsCap,
        granularity_offers_all,
    )

    assert FlagCap(supported=True).is_supported is True
    assert WordTimestampsCap(supported=True, granularities=["word"]).is_supported is True
    assert (
        CandidateLanguagesCap(
            supported=True, constraints=CandidateLanguagesConstraints(max=3)
        ).is_supported
        is True
    )
    assert ReconnectCap(mode="lossy").is_supported is True
    # The granularity vocabulary (declaration side) is exported and non-empty.
    assert get_args(WordTimestampGranularityName)
    assert granularity_offers_all([]) is True


#: Engine-author / framework-internal names the curation deliberately moved OFF
#: the application-facing top level (to ``standard_asr.engine`` /
#: ``standard_asr.compliance`` / their own modules). This regression guard fails
#: the moment one leaks back into ``standard_asr.__all__`` and re-flattens the
#: surface the curation exists to keep sharp.
_CURATED_OFF_TOP_LEVEL: tuple[str, ...] = (
    # engine-author surface -> standard_asr.engine
    "EngineBase",
    "BaseConfig",
    "BaseProperties",
    "DeclaredCapabilities",
    "FlagCap",
    "PreparedAudio",
    "InputKind",
    "Mode",
    "secret_field",
    # compliance suite -> standard_asr.compliance
    "check_entrypoints",
    "ComplianceReport",
    # framework internals -> their own modules
    "negotiate",
    "pcm16_encode",
    "gate_params",
    "reduce_event",
    "diagnose",
)


@pytest.mark.parametrize("name", _CURATED_OFF_TOP_LEVEL)
def test_curated_names_are_not_on_the_application_top_level(name: str) -> None:
    assert name not in standard_asr.__all__


#: Types deliberately on BOTH surfaces: an application *consumes* them (reads a
#: result, drives a session) and an engine author *produces* them. Every other
#: name must live on exactly one tier -- this is what keeps the app-dev surface
#: small without hiding anything an author needs.
_DELIBERATE_DUAL_EXPORTS: frozenset[str] = frozenset(
    {
        "AudioFormat",
        "ChannelResult",
        "Diagnostic",
        "RuntimeParams",
        "Segment",
        "StandardASR",
        "TranscriptionEvent",
        "TranscriptionResult",
        "TranscriptionSession",
        "Word",
        "WordTimestampGranularity",
    }
)


def test_top_level_and_engine_facade_overlap_is_only_the_deliberate_duals() -> None:
    # The application surface and the engine-author facade are disjoint EXCEPT for
    # the produced-and-consumed types above. Any other name appearing on both
    # means a curated symbol leaked back (or an app-only/author-only type was
    # mis-placed) and the two surfaces are silently re-flattening.
    overlap = set(standard_asr.__all__) & set(engine_facade.__all__)
    assert overlap == set(_DELIBERATE_DUAL_EXPORTS)
