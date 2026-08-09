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
    exceptions = importlib.import_module("standard_asr.contract.exceptions")
    assert set(exceptions.__all__) <= set(standard_asr.__all__)
    for name in exceptions.__all__:
        assert getattr(standard_asr, name) is getattr(exceptions, name), name


#: Capability-vocabulary names the engine facade re-exports under a DIFFERENT
#: (established, layer-local) name mapped to the SAME object. Reachability from
#: the facade is what the guard below protects; a second spelling of one object
#: on the same surface would be exactly the noise the curation exists to
#: prevent. Each entry is verified by identity, so an alias that stops pointing
#: at the contract object fails just as loudly as a forgotten export.
_FACADE_ALIASES: dict[str, str] = {
    # ``ModeName`` is homed in contract.capabilities (the module that defines the
    # mode domains); ``runtime.gating`` re-exports it as ``Mode``, the name the
    # engine-author surface has always used (``gate_params(..., mode=...)``).
    "ModeName": "Mode",
}


def test_capability_vocabulary_is_on_the_engine_facade() -> None:
    """Every capabilities export is reachable from the engine-author facade.

    The capability vocabulary is the ENGINE-AUTHOR surface, so it lives on
    ``standard_asr.engine`` -- not the application-facing top level. EVERY name
    the capabilities submodule advertises MUST be reachable from the facade as
    a true re-export (under its own name, or the aliased name above); a future
    capability type added to ``capabilities.__all__`` but forgotten on the
    facade fails this immediately, and every alias must stand in for a name
    genuinely absent from the facade.
    """
    caps_module = importlib.import_module("standard_asr.contract.capabilities")
    caps_exports: list[str] = list(caps_module.__all__)
    expected = {_FACADE_ALIASES.get(name, name) for name in caps_exports}
    assert expected <= set(engine_facade.__all__)
    for name in caps_exports:
        facade_name = _FACADE_ALIASES.get(name, name)
        assert getattr(engine_facade, facade_name) is getattr(caps_module, name), name

    # Every alias must be a real alias -- that is, the contract name it stands in for
    # is genuinely absent from the facade, so this table can never be used to
    # excuse a name the facade actually forgot.
    for name in _FACADE_ALIASES:
        assert name not in engine_facade.__all__, name


def test_mode_is_one_literal_with_one_home() -> None:
    """The mode domain is ONE Literal with one defining home.

    ``runtime.gating.Mode`` is an ALIAS of the contract-layer ``ModeName``, not
    a second Literal: two independently declared mode domains would let the
    gating layer and the capability tree drift apart on what a mode even is.
    """
    caps_module = importlib.import_module("standard_asr.contract.capabilities")
    gating = importlib.import_module("standard_asr.runtime.gating")

    assert gating.Mode is caps_module.ModeName
    assert engine_facade.Mode is caps_module.ModeName
    assert get_args(caps_module.ModeName) == ("batch", "streaming")
    # It is advertised by the module that DEFINES the domains.
    assert "ModeName" in caps_module.__all__

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


def test_sample_rate_helpers_are_on_the_engine_facade() -> None:
    # The spec names sample_rate_accepted as THE membership implementation an
    # engine author reuses (with nearest_accepted_sample_rate for the resample
    # target), so both are re-exported from the engine facade -- and are the very
    # objects the properties module defines (a true re-export, not a copy).
    from standard_asr.engine import nearest_accepted_sample_rate, sample_rate_accepted

    properties = importlib.import_module("standard_asr.contract.properties")
    assert sample_rate_accepted is properties.sample_rate_accepted
    assert nearest_accepted_sample_rate is properties.nearest_accepted_sample_rate
    # They are the engine-author surface, not the application top level.
    assert "sample_rate_accepted" not in standard_asr.__all__
    assert "nearest_accepted_sample_rate" not in standard_asr.__all__


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
        # DIARIZE / DiarizationRequest: an application constructs the request
        # marker; an engine author receives it (and tests gating against it) --
        # the WordTimestampGranularity precedent.
        "DIARIZE",
        "Diagnostic",
        "DiarizationRequest",
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


def test_diarize_constant_identity_across_surfaces() -> None:
    # DIARIZE is one shared, stateless marker instance: every surface MUST
    # re-export the same object (a per-surface copy would still compare equal,
    # but identity is the cheap pin that these are true re-exports).
    runtime_params = importlib.import_module("standard_asr.contract.params")
    assert standard_asr.DIARIZE is engine_facade.DIARIZE
    assert standard_asr.DIARIZE is runtime_params.DIARIZE
    assert standard_asr.DiarizationRequest is runtime_params.DiarizationRequest
    assert engine_facade.DiarizationRequest is runtime_params.DiarizationRequest
