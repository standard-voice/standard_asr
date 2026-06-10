# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compliance helpers (entrypoint checks + sync-bridge driver)."""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator
from importlib.metadata import EntryPoint
from typing import Any, ClassVar, Literal

import numpy as np
import pytest

from standard_asr import TranscriptionResult
from standard_asr import compliance as compliance_module
from standard_asr.audio_format import AudioFormat
from standard_asr.audio_input import InputKind
from standard_asr.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    FlagCap,
    GuidanceCaps,
    LanguageCaps,
    PromptCap,
    PromptConstraints,
    StreamingCapabilities,
    StreamTimestampsCap,
    WordTimestampsCap,
)
from standard_asr.compliance import (
    ComplianceIssue,
    assert_prefix_invariant,
    check_entrypoints,
    check_event_sequence,
    check_provider_params_swap_safety,
    check_recommended_wire_format,
    check_streaming_param_gating,
    check_sync_bridge,
)
from standard_asr.discovery import ModelRegistry, discover_models
from standard_asr.engine import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    PreparedAudio,
    SampleRateRange,
)
from standard_asr.exceptions import (
    ConfigError,
    InvalidProviderParamError,
    UnsupportedFeatureError,
)
from standard_asr.results import Diagnostic, Word
from standard_asr.runtime_params import (
    ProviderParams,
    RuntimeParams,
)
from standard_asr.streaming import SyncSession, TranscriptionEvent, TranscriptionSession


# --------------------------------------------------------------------------- #
# Engine fixtures (declared as classes so they are loadable via entry points).
# --------------------------------------------------------------------------- #
class _Config(BaseConfig[Literal["dummy"]]):
    engine: Literal["dummy"] = "dummy"
    # The fixture properties expose a language axis (selectable_languages is
    # non-empty), so a compliant config MUST provide default_language (IC.6).
    default_language: str | None = "en"


class _ConfigNoLang(BaseConfig[Literal["dummy"]]):
    """Config WITHOUT default_language, for the language-axis violation tests."""

    engine: Literal["dummy"] = "dummy"


class _Props(BaseProperties):
    engine_id: str = "dummy"
    model_name: str = "demo"
    protocol_version: str = "0.2.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    selectable_languages: list[str] = ["en"]


_CAPS = DeclaredCapabilities(
    batch=BatchCapabilities(language=LanguageCaps(runtime_override=FlagCap(supported=True)))
)


class _GoodParams(ProviderParams):
    beam: int = 1


class _GoodASR:
    properties: ClassVar[_Props] = _Props()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _CAPS
    effective_capabilities: ClassVar[DeclaredCapabilities] = _CAPS
    provider_params_type: ClassVar[type[ProviderParams] | None] = _GoodParams

    def __init__(self) -> None:
        self.config = _Config(engine="dummy")

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="ok")

    async def transcribe_async(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="ok")

    def supports(self, dot_path: str) -> bool:
        return self.effective_capabilities.supports(dot_path)


def good_factory() -> _GoodASR:  # pyright: ignore[reportUnusedFunction]
    return _GoodASR()


class _BypassedPropsASR(_GoodASR):
    # Properties built through a validation-bypassing path (model_construct):
    # declaration-time validation never saw them, so the compliance round-trip
    # re-validation must be the layer that catches the malformed declaration.
    properties: ClassVar[_Props] = _Props.model_construct(selectable_languages=["en", "   "])


def bypassed_props_factory() -> _BypassedPropsASR:  # pyright: ignore[reportUnusedFunction]
    return _BypassedPropsASR()


class _WidenedASR(_GoodASR):
    # effective declares MORE than declared (word_timestamps) -> not a subset.
    effective_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
            word_timestamps=WordTimestampsCap(supported=True, granularities=["word"]),
        )
    )


def widened_factory() -> _WidenedASR:  # pyright: ignore[reportUnusedFunction]
    return _WidenedASR()


class _OpenParams(ProviderParams):
    model_config = {"extra": "allow"}  # not a closed type (violates R1)


class _OpenParamsASR(_GoodASR):
    provider_params_type: ClassVar[type[ProviderParams] | None] = _OpenParams


def open_params_factory() -> _OpenParamsASR:  # pyright: ignore[reportUnusedFunction]
    return _OpenParamsASR()


class _BareBaseParamsASR(_GoodASR):
    # provider_params_type is the bare ProviderParams base: no fields, admits any
    # params, so swap-safety is zeroed. The compliance suite must flag it (spec 3.2).
    provider_params_type: ClassVar[type[ProviderParams] | None] = ProviderParams


def bare_base_params_factory() -> _BareBaseParamsASR:  # pyright: ignore[reportUnusedFunction]
    return _BareBaseParamsASR()


class _NotProviderParams:
    """A provider_params_type that is not a ProviderParams subclass at all."""


class _BadParamsTypeASR(_GoodASR):
    provider_params_type: ClassVar[Any] = _NotProviderParams


def bad_params_type_factory() -> _BadParamsTypeASR:  # pyright: ignore[reportUnusedFunction]
    return _BadParamsTypeASR()


class _RaisingEffectiveASR(_GoodASR):
    """effective_capabilities is a property that raises (a buggy engine)."""

    properties: ClassVar[_Props] = _Props(engine_id="dummy2")

    @property
    def effective_capabilities(self) -> DeclaredCapabilities:  # type: ignore[override]
        raise RuntimeError("effective boom")


def raising_effective_factory() -> _RaisingEffectiveASR:  # pyright: ignore[reportUnusedFunction]
    return _RaisingEffectiveASR()


class _WrongTypeEffectiveASR(_GoodASR):
    """effective_capabilities is a non-None value of the wrong type."""

    effective_capabilities: ClassVar[Any] = "not-a-capabilities-tree"


def wrong_type_effective_factory() -> _WrongTypeEffectiveASR:  # pyright: ignore[reportUnusedFunction]
    return _WrongTypeEffectiveASR()


class _NoneEffectiveASR(_GoodASR):
    """effective_capabilities is None (engine declares no narrowing)."""

    effective_capabilities: ClassVar[Any] = None


def none_effective_factory() -> _NoneEffectiveASR:  # pyright: ignore[reportUnusedFunction]
    return _NoneEffectiveASR()


class _GoodConfigTypeASR(_GoodASR):
    """Engine declaring its config_type (the schema-discoverable good citizen)."""

    config_type: ClassVar[type[BaseConfig[str]] | None] = _Config


def good_config_type_factory() -> _GoodConfigTypeASR:  # pyright: ignore[reportUnusedFunction]
    return _GoodConfigTypeASR()


class _BadConfigTypeASR(_GoodASR):
    """config_type set to something that is not a BaseConfig subclass."""

    config_type: ClassVar[Any] = _NotProviderParams


def bad_config_type_factory() -> _BadConfigTypeASR:  # pyright: ignore[reportUnusedFunction]
    return _BadConfigTypeASR()


class _OtherConfig(BaseConfig[Literal["dummy"]]):
    engine: Literal["dummy"] = "dummy"
    default_language: str | None = "en"


class _MismatchedConfigTypeASR(_GoodASR):
    """Declares config_type=_OtherConfig but constructs a _Config instance."""

    config_type: ClassVar[type[BaseConfig[str]] | None] = _OtherConfig


def mismatched_config_type_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _MismatchedConfigTypeASR
):
    return _MismatchedConfigTypeASR()


class _AxisNoDefaultEngine(EngineBase):
    """EngineBase engine with a language axis but no default_language (IC.6 bug)."""

    properties: ClassVar[BaseProperties] = _Props()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _CAPS

    def __init__(self) -> None:
        self.config = _ConfigNoLang(engine="dummy")

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        return TranscriptionResult(text="never reached")


def axis_no_default_factory() -> _AxisNoDefaultEngine:  # pyright: ignore[reportUnusedFunction]
    return _AxisNoDefaultEngine()


class _StructuralAxisNoDefaultASR(_GoodASR):
    """Structural (non-EngineBase) engine with the same IC.6 violation."""

    def __init__(self) -> None:
        self.config = _ConfigNoLang(engine="dummy")


def structural_axis_no_default_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _StructuralAxisNoDefaultASR
):
    return _StructuralAxisNoDefaultASR()


def _registry(factory: str, key: str = "dummy/demo") -> ModelRegistry:
    eps = [
        EntryPoint(
            name=key,
            value=f"tests.test_compliance:{factory}",
            group="standard_asr.models",
        )
    ]
    return discover_models(eps=eps, strict=True)


def _registry_many(*pairs: tuple[str, str]) -> ModelRegistry:
    eps = [
        EntryPoint(
            name=key,
            value=f"tests.test_compliance:{factory}",
            group="standard_asr.models",
        )
        for factory, key in pairs
    ]
    return discover_models(eps=eps, strict=True)


# --------------------------------------------------------------------------- #
# check_entrypoints
# --------------------------------------------------------------------------- #
def test_check_entrypoints_empty_registry_errors() -> None:
    report = check_entrypoints(registry=ModelRegistry({}))
    assert report.passed is False
    assert any("No standard_asr.models" in i.message for i in report.issues)


def test_check_entrypoints_good_engine_passes() -> None:
    report = check_entrypoints(registry=_registry("good_factory"))
    assert report.passed is True, [i.message for i in report.issues]


def test_check_entrypoints_bypassed_properties_fail_revalidation() -> None:
    # Ultra review defense in depth: properties that dodged
    # declaration-time validation (model_construct) must be caught by the
    # compliance round-trip instead of being certified compliant.
    report = check_entrypoints(registry=_registry("bypassed_props_factory"))
    assert report.passed is False
    assert any("fail re-validation" in i.message for i in report.issues)


def test_check_entrypoints_runtime_params_not_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the global RuntimeParams-closed invariant to read as violated.
    def _not_closed(model: type) -> bool:
        return False

    monkeypatch.setattr(compliance_module, "_is_closed_model", _not_closed)
    report = check_entrypoints(registry=_registry("good_factory"))
    assert any("RuntimeParams is not a closed type" in i.message for i in report.issues)


def test_check_entrypoints_runtime_params_closedness_runs_with_no_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The RuntimeParams-closedness invariant is plugin-independent and MUST be
    # verified even in a bare environment (empty registry), not skipped by the
    # "no plugins" early return.
    def _not_closed(model: type) -> bool:
        return False

    monkeypatch.setattr(compliance_module, "_is_closed_model", _not_closed)
    report = check_entrypoints(registry=ModelRegistry({}))
    assert report.passed is False
    assert any("RuntimeParams is not a closed type" in i.message for i in report.issues)
    # The empty-registry diagnostic is still reported too.
    assert any("No standard_asr.models" in i.message for i in report.issues)


def test_check_entrypoints_effective_widens_declared() -> None:
    report = check_entrypoints(registry=_registry("widened_factory"))
    assert report.passed is False
    assert any("not a subset" in i.message for i in report.issues)


def test_check_entrypoints_raising_effective_caps_is_reported_not_crash() -> None:
    # A buggy effective_capabilities property that raises must surface as a
    # ComplianceIssue (not an uncaught exception), and other engines in the same
    # run must still be checked.
    report = check_entrypoints(
        registry=_registry_many(
            ("raising_effective_factory", "dummy2/demo"),
            ("widened_factory", "dummy/demo"),
        )
    )
    assert report.passed is False
    assert any("Reading effective_capabilities raised" in i.message for i in report.issues)
    # The widened engine was still reached and checked despite the earlier raiser.
    assert any("not a subset" in i.message for i in report.issues)


def test_check_entrypoints_wrong_typed_effective_caps_fails() -> None:
    # A non-None effective_capabilities of the wrong type MUST NOT silently skip
    # the effective ⊆ declared check -- it is itself a compliance error, so the
    # invariant cannot be evaded by returning the wrong type.
    report = check_entrypoints(registry=_registry("wrong_type_effective_factory"))
    assert report.passed is False
    assert any("is not a DeclaredCapabilities" in i.message for i in report.issues)


def test_check_entrypoints_none_effective_caps_passes() -> None:
    # effective_capabilities = None (engine declares no narrowing) is a legitimate
    # no-op and MUST pass.
    report = check_entrypoints(registry=_registry("none_effective_factory"))
    assert report.passed is True, [i.message for i in report.issues]


def test_check_entrypoints_open_provider_params_errors() -> None:
    report = check_entrypoints(registry=_registry("open_params_factory"))
    assert any("closed type" in i.message for i in report.issues)


def test_check_entrypoints_provider_params_not_subclass_errors() -> None:
    report = check_entrypoints(registry=_registry("bad_params_type_factory"))
    assert any("not a ProviderParams subclass" in i.message for i in report.issues)


def test_check_entrypoints_bare_base_provider_params_errors() -> None:
    # RR-011 / spec 3.2: declaring the bare ProviderParams base (no fields, admits
    # any params) zeroes swap-safety. The compliance suite must flag it as an error.
    report = check_entrypoints(registry=_registry("bare_base_params_factory"))
    assert report.passed is False
    assert any(i.code == "provider_params_type_is_bare_base" for i in report.issues)
    # Discriminator: the SAME _GoodASR base with a DISTINCT terminal params type
    # does NOT trip the bare-base check -- the bare base is the sole failing variable.
    good = check_entrypoints(registry=_registry("good_factory"))
    assert not any(i.code == "provider_params_type_is_bare_base" for i in good.issues)


def test_check_entrypoints_missing_config_type_warns_but_passes() -> None:
    # No class-level config_type: a DX warning (settings UIs cannot discover the
    # config schema without instantiation), but NOT a compliance failure.
    report = check_entrypoints(registry=_registry("good_factory"))
    assert report.passed is True, [i.message for i in report.issues]
    warnings = [i for i in report.issues if i.level == "warning"]
    assert any("config_type" in i.message for i in warnings)


def test_check_entrypoints_declared_config_type_no_warning() -> None:
    report = check_entrypoints(registry=_registry("good_config_type_factory"))
    assert report.passed is True, [i.message for i in report.issues]
    assert not any("config_type" in i.message for i in report.issues)


def test_check_entrypoints_config_type_not_baseconfig_errors() -> None:
    report = check_entrypoints(registry=_registry("bad_config_type_factory"))
    assert report.passed is False
    assert any("not a BaseConfig subclass" in i.message for i in report.issues)


def test_check_entrypoints_config_not_instance_of_config_type_errors() -> None:
    # Declaring config_type=X while constructing a Y config means the schema
    # published for UIs does not match the config actually consumed.
    report = check_entrypoints(registry=_registry("mismatched_config_type_factory"))
    assert report.passed is False
    assert any("not an instance of the declared" in i.message for i in report.issues)


def test_check_entrypoints_language_axis_without_default_enginebase_errors() -> None:
    # An EngineBase engine with a language axis but no default_language would
    # raise ConfigError on the user's FIRST transcribe; compliance must catch it
    # at author time, reusing the exact runtime validation.
    report = check_entrypoints(registry=_registry("axis_no_default_factory"))
    assert report.passed is False
    assert any("every transcribe will fail" in i.message for i in report.issues)


def test_check_entrypoints_language_axis_without_default_structural_errors() -> None:
    # Structural (non-EngineBase) engines get the IC.6 presence check.
    report = check_entrypoints(registry=_registry("structural_axis_no_default_factory"))
    assert report.passed is False
    assert any("default_language" in i.message for i in report.issues)


def unannotated_factory():  # type: ignore[no-untyped-def]  # pyright: ignore[reportUnusedFunction]
    # Loadable as a factory, but engine_class() cannot resolve the class without
    # instantiation (no concrete return annotation) -> FactoryLoadError.
    return _GoodASR()


def test_check_entrypoints_class_metadata_unreadable() -> None:
    # The factory loads, but the engine class is unresolvable without
    # instantiation; the class-level metadata check surfaces that as an error.
    report = check_entrypoints(registry=_registry("unannotated_factory"), instantiate=False)
    assert any("not readable without instantiation" in i.message for i in report.issues)


def test_check_entrypoints_no_instantiate_skips_invocation() -> None:
    # instantiate=False must still validate class metadata but never call the
    # factory; the good engine passes its class-level checks.
    report = check_entrypoints(registry=_registry("good_factory"), instantiate=False)
    assert report.passed is True, [i.message for i in report.issues]


# --------------------------------------------------------------------------- #
# Required-surface checks (D9): the full StandardASR method surface, conditional
# on the declared streaming axis, plus the identity match.
# --------------------------------------------------------------------------- #
class _NoAsyncASR(_GoodASR):
    """Batch engine missing the required ``transcribe_async`` method."""

    transcribe_async: ClassVar[None] = None  # type: ignore[assignment]


def no_async_factory() -> _NoAsyncASR:  # pyright: ignore[reportUnusedFunction]
    return _NoAsyncASR()


class _NoSupportsASR(_GoodASR):
    """Batch engine missing the required ``supports`` method."""

    supports: ClassVar[None] = None  # type: ignore[assignment]


def no_supports_factory() -> _NoSupportsASR:  # pyright: ignore[reportUnusedFunction]
    return _NoSupportsASR()


class _NoPropertiesProbeASR:
    """Engine exposing the methods but no ``properties`` attribute at all."""

    declared_capabilities: ClassVar[DeclaredCapabilities] = _CAPS
    effective_capabilities: ClassVar[DeclaredCapabilities] = _CAPS
    provider_params_type: ClassVar[type[ProviderParams] | None] = _GoodParams

    def __init__(self) -> None:
        self.config = _Config(engine="dummy")

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="ok")

    async def transcribe_async(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="ok")

    def supports(self, dot_path: str) -> bool:
        return self.effective_capabilities.supports(dot_path)


def no_properties_probe_factory() -> _NoPropertiesProbeASR:  # pyright: ignore[reportUnusedFunction]
    return _NoPropertiesProbeASR()


_STREAMING_CAPS = DeclaredCapabilities(
    batch=BatchCapabilities(),
    streaming=StreamingCapabilities(),
    streaming_output=FlagCap(supported=True),
)


class _StreamingNoStartASR(_GoodASR):
    """Declares a streaming axis but omits ``start_transcription`` (non-compliant)."""

    declared_capabilities: ClassVar[DeclaredCapabilities] = _STREAMING_CAPS
    effective_capabilities: ClassVar[DeclaredCapabilities] = _STREAMING_CAPS


def streaming_no_start_factory() -> _StreamingNoStartASR:  # pyright: ignore[reportUnusedFunction]
    return _StreamingNoStartASR()


# A streaming_input engine that omits wire_encodings opens a silent
# wire-mistranscription window on an audio_format session, so the compliance
# suite nudges it with a WARNING (not an error -- a self-managed-wire-format
# adapter legitimately leaves it unset). Declaring wire_encodings clears it.
_STREAMING_INPUT_CAPS = DeclaredCapabilities(
    batch=BatchCapabilities(),
    streaming=StreamingCapabilities(),
    streaming_input=FlagCap(supported=True),
)


class _WireProps(_Props):
    wire_encodings: list[str] | None = ["pcm_s16le"]


class _StreamingInputNoWireASR(_GoodASR):
    """Declares streaming_input but leaves wire_encodings unset (warning)."""

    declared_capabilities: ClassVar[DeclaredCapabilities] = _STREAMING_INPUT_CAPS
    effective_capabilities: ClassVar[DeclaredCapabilities] = _STREAMING_INPUT_CAPS

    def start_transcription(self, **kwargs: Any) -> TranscriptionSession:
        raise NotImplementedError  # pragma: no cover - presence is all the check needs


def streaming_input_no_wire_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _StreamingInputNoWireASR
):
    return _StreamingInputNoWireASR()


class _StreamingInputWithWireASR(_StreamingInputNoWireASR):
    """Declares streaming_input AND wire_encodings (no warning)."""

    properties: ClassVar[_Props] = _WireProps()


def streaming_input_with_wire_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _StreamingInputWithWireASR
):
    return _StreamingInputWithWireASR()


_STREAMING_NO_AXIS_CAPS = DeclaredCapabilities(streaming=StreamingCapabilities())


class _StreamingNoAxisASR(_GoodASR):
    """Populates the streaming domain but neither axis flag (uncallable; CC-1)."""

    declared_capabilities: ClassVar[DeclaredCapabilities] = _STREAMING_NO_AXIS_CAPS
    effective_capabilities: ClassVar[DeclaredCapabilities] = _STREAMING_NO_AXIS_CAPS

    def start_transcription(self, **kwargs: Any) -> TranscriptionSession:
        raise NotImplementedError  # pragma: no cover - presence is all the check needs


def streaming_no_axis_factory() -> _StreamingNoAxisASR:  # pyright: ignore[reportUnusedFunction]
    return _StreamingNoAxisASR()


def test_check_entrypoints_missing_transcribe_async_fails() -> None:
    report = check_entrypoints(registry=_registry("no_async_factory"))
    assert report.passed is False
    assert any("'transcribe_async'" in i.message and i.level == "error" for i in report.issues), [
        i.message for i in report.issues
    ]


def test_check_entrypoints_missing_supports_fails() -> None:
    report = check_entrypoints(registry=_registry("no_supports_factory"))
    assert report.passed is False
    assert any("'supports'" in i.message and i.level == "error" for i in report.issues), [
        i.message for i in report.issues
    ]


def test_check_entrypoints_missing_properties_fails() -> None:
    report = check_entrypoints(registry=_registry("no_properties_probe_factory"))
    assert report.passed is False
    assert any("'properties'" in i.message for i in report.issues), [
        i.message for i in report.issues
    ]


def test_check_entrypoints_streaming_engine_missing_start_transcription_fails() -> None:
    report = check_entrypoints(registry=_registry("streaming_no_start_factory"))
    assert report.passed is False
    assert any("'start_transcription'" in i.message for i in report.issues), [
        i.message for i in report.issues
    ]


def test_check_entrypoints_batch_only_without_start_transcription_passes() -> None:
    # A batch-only engine (no streaming axis declared) legitimately omits
    # start_transcription -- it MUST still pass. _GoodASR has no
    # start_transcription attribute, confirming the conditional requirement.
    assert not hasattr(_GoodASR, "start_transcription")
    report = check_entrypoints(registry=_registry("good_factory"))
    assert report.passed is True, [i.message for i in report.issues]


def test_check_entrypoints_streaming_input_without_wire_encodings_warns_but_passes() -> None:
    # A streaming_input engine that omits wire_encodings cannot have
    # an audio_format session's encoding validated -- a silent-mistranscription
    # window. The compliance suite flags it as a WARNING (DX nudge), not an
    # error: a self-managed-wire-format adapter may legitimately leave it unset.
    report = check_entrypoints(registry=_registry("streaming_input_no_wire_factory"))
    warnings = [i for i in report.issues if i.level == "warning"]
    assert any("wire_encodings" in i.message for i in warnings), [i.message for i in report.issues]
    # It is only a warning -- no wire_encodings error is raised for this.
    assert not any("wire_encodings" in i.message and i.level == "error" for i in report.issues), [
        i.message for i in report.issues
    ]


def test_check_entrypoints_streaming_input_with_wire_encodings_no_warning() -> None:
    # Declaring wire_encodings closes the silent-mistranscription
    # window, so the nudge does not fire.
    report = check_entrypoints(registry=_registry("streaming_input_with_wire_factory"))
    assert not any("wire_encodings" in i.message for i in report.issues), [
        i.message for i in report.issues
    ]


def test_check_entrypoints_batch_only_no_wire_encodings_no_warning() -> None:
    # The nudge is specific to streaming_input engines. A batch-only
    # engine without wire_encodings (the common case) MUST NOT be warned.
    report = check_entrypoints(registry=_registry("good_factory"))
    assert not any("wire_encodings" in i.message for i in report.issues), [
        i.message for i in report.issues
    ]


def test_check_entrypoints_streaming_domain_without_axis_is_error() -> None:
    # CC-1: a streaming capabilities domain with neither streaming_input nor
    # streaming_output is an uncallable engine -- every start_transcription fails
    # closed -- so it is a compliance ERROR, not a soft nudge (unlike wire_encodings,
    # there is no legitimate engine in this state).
    report = check_entrypoints(registry=_registry("streaming_no_axis_factory"))
    assert report.passed is False
    assert any(
        i.code == "streaming_domain_without_axis" and i.level == "error" for i in report.issues
    ), [i.message for i in report.issues]


def test_check_entrypoints_streaming_with_axis_no_cc1_error() -> None:
    # A streaming engine that declares an axis (input here) MUST NOT trip CC-1.
    report = check_entrypoints(registry=_registry("streaming_input_with_wire_factory"))
    assert not any(i.code == "streaming_domain_without_axis" for i in report.issues), [
        i.message for i in report.issues
    ]


def test_check_entrypoints_batch_only_no_cc1_error() -> None:
    # A batch-only engine (no streaming domain) MUST NOT trip CC-1.
    report = check_entrypoints(registry=_registry("good_factory"))
    assert not any(i.code == "streaming_domain_without_axis" for i in report.issues)


def test_check_entrypoints_properties_key_mismatch_fails() -> None:
    # The engine's declared identity (properties.model_id) MUST match its
    # entry-point key; a mismatch is a compliance error, not a silent accept.
    report = check_entrypoints(registry=_registry("good_factory", key="dummy/other"))
    assert report.passed is False
    assert any("does not match the entry point key" in i.message for i in report.issues), [
        i.message for i in report.issues
    ]


# --------------------------------------------------------------------------- #
# check_sync_bridge
# --------------------------------------------------------------------------- #
class _CleanSession(TranscriptionSession):
    """Ends immediately with a terminal ``done`` (clean bridge)."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        return
        yield  # pragma: no cover - makes this an async generator


def test_sync_bridge_clean_session_passes() -> None:
    report = check_sync_bridge(_CleanSession, timeout=5.0)
    assert report.passed is True, [i.message for i in report.issues]


class _RaisingSession(TranscriptionSession):
    async def _open(self) -> None:
        raise RuntimeError("open boom")

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        yield TranscriptionEvent.done()  # pragma: no cover - never reached


def test_sync_bridge_raising_session_reports_error() -> None:
    report = check_sync_bridge(_RaisingSession, timeout=5.0)
    assert report.passed is False
    assert any("raised while bridging" in i.message for i in report.issues)
    # The adapter's _open raised, so __enter__ raises -- but an exception-safe
    # __enter__ tears down its own loop thread before propagating. The failure must
    # be attributed to the raise alone, NOT also mis-reported as a thread leak.
    assert not any(i.code == "sync_bridge_thread_leak" for i in report.issues), [
        i.message for i in report.issues
    ]


def test_sync_bridge_factory_raising_reports_error() -> None:
    # A factory that raises (no session ever constructed) is reported as a bridge
    # error, and -- since nothing was started -- never as a thread leak.
    def _bad_factory() -> TranscriptionSession:
        raise RuntimeError("factory boom")

    report = check_sync_bridge(_bad_factory, timeout=5.0)
    assert report.passed is False
    assert any("raised while bridging" in i.message for i in report.issues)
    assert not any(i.code == "sync_bridge_thread_leak" for i in report.issues)


def test_sync_bridge_ignores_benign_daemon_thread() -> None:
    # CC-2 regression: a compliant adapter may pull in a dependency that spawns a
    # benign background daemon thread (e.g. tqdm's monitor, a thread-pool worker)
    # that is still alive when the bridge closes. The leak check MUST assert on the
    # bridge's OWN loop thread, not a process-wide thread diff -- otherwise such a
    # benign thread is mis-reported as a sync_bridge_thread_leak, failing a
    # perfectly compliant engine.
    release = threading.Event()

    class _SpawnsDaemonSession(TranscriptionSession):
        async def _open(self) -> None:
            threading.Thread(
                target=release.wait, name="benign-dependency-daemon", daemon=True
            ).start()

        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            return
            yield  # pragma: no cover - makes this an async generator

    try:
        report = check_sync_bridge(_SpawnsDaemonSession, timeout=5.0)
        assert report.passed is True, [i.message for i in report.issues]
    finally:
        release.set()  # let the benign daemon exit promptly


def test_sync_bridge_flags_genuine_loop_thread_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    # The leak check still fires for a REAL leak: force is_loop_alive() to report
    # the owned thread as surviving close (the actual thread is still joined
    # cleanly) to exercise the leak-detection branch deterministically.
    def _force_alive(_self: SyncSession) -> bool:
        return True

    monkeypatch.setattr(SyncSession, "is_loop_alive", _force_alive)
    report = check_sync_bridge(_CleanSession, timeout=5.0)
    assert report.passed is False
    assert any(i.code == "sync_bridge_thread_leak" for i in report.issues)


class _NoTerminalSession(TranscriptionSession):
    """Non-compliant adapter: closes the stream WITHOUT a terminal event.

    Overrides ``_run_producer`` to bypass the base class's force-appended
    ``done``, emitting a single non-terminal event and closing. This is the
    out-of-tree non-compliance the sync-bridge check must flag.
    """

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        return
        yield  # pragma: no cover - makes this an async generator

    async def _run_producer(self) -> None:
        self._buffer.put_forced(TranscriptionEvent.partial(segment_id="s0", text="hi"))
        self._buffer.close()


def test_sync_bridge_no_terminal_event_reports_error() -> None:
    report = check_sync_bridge(_NoTerminalSession, timeout=5.0)
    assert report.passed is False
    assert any("without emitting a terminal event" in i.message for i in report.issues)


def test_sync_bridge_deadlock_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # A factory whose driver thread never returns within the timeout surfaces as
    # a deadlock error. We simulate by making the driver block on a factory that
    # spins past the (tiny) timeout.
    class _HangSession(TranscriptionSession):
        async def _open(self) -> None:
            # Block the loop thread far longer than the bridge timeout so the
            # worker is still alive when join() returns.
            time.sleep(1.0)

        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.done()  # pragma: no cover - never reached

    report = check_sync_bridge(_HangSession, timeout=0.05)
    assert report.passed is False
    assert any("did not terminate" in i.message for i in report.issues)


# --------------------------------------------------------------------------- #
# check_recommended_wire_format (AW-2 self-consistency)
# --------------------------------------------------------------------------- #
def test_recommended_wire_format_self_consistent_passes() -> None:
    # A well-formed streaming engine's recommended format is accepted by its own
    # session-establishment guard.
    report = check_recommended_wire_format(_GatingStreamEngine())
    assert report.passed is True, [i.message for i in report.issues]


def test_recommended_wire_format_none_is_not_a_violation() -> None:
    # An engine that recommends no format (no usable sample rate) is not flagged --
    # there is simply nothing to assert consistency against.
    class _NoRateEngine(_GatingStreamEngine):
        properties: ClassVar[BaseProperties] = _StreamProps.model_construct(
            native_sample_rate=0, required_input_sample_rate=None
        )

    report = check_recommended_wire_format(_NoRateEngine())
    assert report.passed is True, [i.message for i in report.issues]


def test_recommended_wire_format_self_inconsistent_is_flagged() -> None:
    # An engine whose recommended rate is not among its own accepted rates (an
    # R7-violating declaration) is caught: the recommended format must be one the
    # engine itself accepts.
    class _InconsistentEngine(_GatingStreamEngine):
        properties: ClassVar[BaseProperties] = _StreamProps.model_construct(
            native_sample_rate=16000, accepted_sample_rates=[8000], required_input_sample_rate=None
        )

    report = check_recommended_wire_format(_InconsistentEngine())
    assert report.passed is False
    assert any(i.code == "recommended_wire_format_self_inconsistent" for i in report.issues)


def test_recommended_wire_format_raising_is_reported() -> None:
    # If recommended_wire_format() itself raises, that is surfaced as an error
    # rather than crashing the compliance run.
    class _RaisingEngine(_GatingStreamEngine):
        def recommended_wire_format(self) -> Any:
            raise RuntimeError("boom")

    report = check_recommended_wire_format(_RaisingEngine())
    assert report.passed is False
    assert any(i.code == "recommended_wire_format_raised" for i in report.issues)


# --------------------------------------------------------------------------- #
# check_event_sequence (pure streaming-invariant validator)
# --------------------------------------------------------------------------- #
def test_check_event_sequence_accepts_a_clean_stream() -> None:
    events = [
        TranscriptionEvent.partial("s0", "hel"),
        TranscriptionEvent.final("s0", "hello"),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is True
    assert report.issues == []


def test_check_event_sequence_flags_illegal_transition() -> None:
    # partial after final for the same segment is an illegal lifecycle transition.
    events = [
        TranscriptionEvent.final("s0", "hello"),
        TranscriptionEvent.partial("s0", "hello again"),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is False
    assert any("invariant violated" in i.message for i in report.issues)


def test_check_event_sequence_flags_double_retirement() -> None:
    # Ultra review superseded is terminal -- a stream retiring the
    # same id twice must fail compliance (the guard previously admitted it,
    # certifying non-compliant engines).
    events = [
        TranscriptionEvent.partial("a", "hello"),
        TranscriptionEvent.supersede(["a"], ["b"]),
        TranscriptionEvent.supersede(["a"], ["c"]),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is False
    assert any("invariant violated" in i.message for i in report.issues)


def test_check_event_sequence_flags_missing_terminal() -> None:
    report = check_event_sequence([TranscriptionEvent.final("s0", "hello")])
    assert report.passed is False
    assert any("without a terminal" in i.message for i in report.issues)


def test_check_event_sequence_flags_decreasing_audio_cursor() -> None:
    events = [
        TranscriptionEvent.progress(audio_processed_until=2.0),
        TranscriptionEvent.progress(audio_processed_until=1.0),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is False
    assert any("audio_cursor_decreased" in i.message for i in report.issues)


def test_check_event_sequence_empty_fails_by_default() -> None:
    # An empty stream is a violation by default: a real session always emits at
    # least a terminal event.
    report = check_event_sequence([])
    assert report.passed is False
    assert any("empty event sequence" in i.message for i in report.issues)


def test_check_event_sequence_empty_allowed_with_flag() -> None:
    report = check_event_sequence([], allow_empty=True)
    assert report.passed is True
    assert report.issues == []


def test_check_event_sequence_flags_event_after_done() -> None:
    # A terminal MUST be the last event; a partial after done is a violation.
    events = [
        TranscriptionEvent.final("s0", "hello"),
        TranscriptionEvent.done(),
        TranscriptionEvent.partial("s1", "late"),
    ]
    report = check_event_sequence(events)
    assert report.passed is False
    assert any("after the session-terminal" in i.message for i in report.issues)


def test_check_event_sequence_flags_event_after_nonrecoverable_error() -> None:
    # A non-recoverable error is terminal; any later event is a violation.
    events = [
        TranscriptionEvent.make_error("session_timeout", recoverable=False),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is False
    assert any("after the session-terminal" in i.message for i in report.issues)


def test_check_event_sequence_recoverable_error_is_not_terminal() -> None:
    # A recoverable error does NOT end the session, so events may legitimately
    # follow it; the stream still needs a real terminal to pass.
    events = [
        TranscriptionEvent.make_error("content_lost", recoverable=True),
        TranscriptionEvent.partial("s0", "resumed"),
        TranscriptionEvent.final("s0", "resumed"),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is True, [i.message for i in report.issues]


def test_check_event_sequence_accepts_closed_rewrite_frozen_prefix() -> None:
    events = [
        TranscriptionEvent.final("s0", "hello", stable_until=5),
        TranscriptionEvent.closed("s0", "Hello.", stable_until=6),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is True
    assert not any("frozen_prefix_rewritten" in i.message for i in report.issues)


def test_check_event_sequence_flags_non_closed_frozen_prefix_rewrite() -> None:
    events = [
        TranscriptionEvent.partial("s0", "hello", stable_until=5),
        TranscriptionEvent.final("s0", "Hello.", stable_until=6),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is False
    assert any("frozen_prefix_rewritten" in i.message for i in report.issues)


def test_check_event_sequence_flags_supersede_frozen_prefix_rewrite() -> None:
    # A supersede that rewrites the retired segment's frozen prefix (spec ST.5.2)
    # MUST be reported -- the cardinal sin.
    events = [
        TranscriptionEvent.final("a", "你好世界", stable_until=4),
        TranscriptionEvent.supersede(["a"], ["b"]),
        TranscriptionEvent.final("b", "再见", stable_until=2),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is False
    assert any("frozen_prefix_rewritten_supersede" in i.message for i in report.issues)


def test_check_event_sequence_does_not_cascade_after_supersede_rewrite() -> None:
    events = [
        TranscriptionEvent.final("a", "hello", stable_until=5),
        TranscriptionEvent.supersede(["a"], ["b"]),
        TranscriptionEvent.final("b", "bye", stable_until=3),
        TranscriptionEvent.final("b", "hello there", stable_until=5),
        TranscriptionEvent.done(),
    ]

    report = check_event_sequence(events)

    assert report.passed is False
    assert len(report.issues) == 1
    assert "frozen_prefix_rewritten_supersede" in report.issues[0].message


def test_check_event_sequence_accepts_supersede_merge_preserving_frozen() -> None:
    events = [
        TranscriptionEvent.final("a", "你好", stable_until=2),
        TranscriptionEvent.final("b", "世界", stable_until=2),
        TranscriptionEvent.supersede(["a", "b"], ["c"]),
        TranscriptionEvent.final("c", "你好世界！", stable_until=4),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is True


def test_check_event_sequence_warns_unfulfilled_supersede_obligation() -> None:
    # A8: the replacement re-froze "你好" but the retired frozen prefix was
    # "你好世界" -- the permitted conservative direction. The replay reports it as
    # a soft WARNING (it does NOT fail the report; the supersede is not rejected).
    events = [
        TranscriptionEvent.final("a", "你好世界", stable_until=4),
        TranscriptionEvent.supersede(["a"], ["b"]),
        TranscriptionEvent.final("b", "你好", stable_until=2),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is True, [i.message for i in report.issues]
    obligation = [i for i in report.issues if "supersede_obligation_unfulfilled" in i.message]
    assert len(obligation) == 1
    assert obligation[0].level == "warning"


def test_check_event_sequence_reconciled_supersede_has_no_obligation_warning() -> None:
    events = [
        TranscriptionEvent.final("a", "你好世界", stable_until=4),
        TranscriptionEvent.supersede(["a"], ["b"]),
        TranscriptionEvent.final("b", "你好世界", stable_until=4),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is True
    assert not any("supersede_obligation_unfulfilled" in i.message for i in report.issues)


def test_check_event_sequence_flags_unannounced_old_id() -> None:
    events = [
        TranscriptionEvent.supersede(["never-seen"], ["b"]),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is False
    assert any("supersede_unknown_old_id" in i.message for i in report.issues)


def test_check_event_sequence_flags_reintroduced_new_id() -> None:
    events = [
        TranscriptionEvent.partial("a", "x"),
        TranscriptionEvent.partial("b", "y"),
        TranscriptionEvent.supersede(["a"], ["b"]),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is False
    assert any("supersede_reintroduces_segment" in i.message for i in report.issues)


def test_check_event_sequence_flags_final_after_final() -> None:
    events = [
        TranscriptionEvent.final("a", "hello"),
        TranscriptionEvent.final("a", "rewritten"),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is False
    assert any("lifecycle_final_after_final" in i.message for i in report.issues)


def test_check_event_sequence_flags_empty_new_ids_deleting_frozen() -> None:
    events = [
        TranscriptionEvent.final("a", "你好", stable_until=2),
        TranscriptionEvent.supersede(["a"], []),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is False
    assert any("supersede_deletes_frozen_text" in i.message for i in report.issues)


# --------------------------------------------------------------------------- #
# check_event_sequence capability cross-check (SF-4)
# --------------------------------------------------------------------------- #
# Streaming caps with the timestamp/stability sub-caps left at their (unsupported)
# defaults -- the "no-timestamp streaming" profile.
_NO_TS_STREAMING_CAPS = DeclaredCapabilities(
    streaming=StreamingCapabilities(),
    streaming_input=FlagCap(supported=True),
)


def test_event_sequence_cross_check_skipped_without_capabilities() -> None:
    # No capabilities -> the cross-check does not run (a non-zero stable_until is
    # not, on its own, a structural violation).
    events = [
        TranscriptionEvent.partial("s0", "hello", stable_until=3),
        TranscriptionEvent.final("s0", "hello"),
        TranscriptionEvent.done(),
    ]
    assert check_event_sequence(events).passed is True


def test_event_sequence_cross_check_skipped_without_streaming_domain() -> None:
    # Capabilities without a streaming domain -> nothing to cross-check against.
    events = [
        TranscriptionEvent.partial("s0", "hello", stable_until=3),
        TranscriptionEvent.final("s0", "hello"),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events, capabilities=DeclaredCapabilities())
    assert not any(i.code.startswith("stream_exceeds_") for i in report.issues)


def test_event_sequence_flags_stable_until_without_word_stability() -> None:
    events = [
        TranscriptionEvent.partial("s0", "hello", stable_until=3),
        TranscriptionEvent.final("s0", "hello"),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events, capabilities=_NO_TS_STREAMING_CAPS)
    assert report.passed is False
    assert any(i.code == "stream_exceeds_word_stability" for i in report.issues)


def test_event_sequence_flags_audio_cursor_without_timestamps() -> None:
    events = [
        TranscriptionEvent.partial("s0", "hi", audio_processed_until=1.0),
        TranscriptionEvent.final("s0", "hi"),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events, capabilities=_NO_TS_STREAMING_CAPS)
    assert report.passed is False
    assert any(i.code == "stream_exceeds_timestamps" for i in report.issues)


def test_event_sequence_flags_words_without_word_timestamps() -> None:
    events = [
        TranscriptionEvent.final("s0", "hi", words=[Word(start=0.0, end=0.5, text="hi")]),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events, capabilities=_NO_TS_STREAMING_CAPS)
    assert report.passed is False
    assert any(i.code == "stream_exceeds_word_timestamps" for i in report.issues)


def test_event_sequence_consistent_stream_passes_cross_check() -> None:
    # When the caps DECLARE these fields supported, the same stream is consistent --
    # no cross-check error fires (the check is one-directional: stream must not
    # exceed declared caps, but may use less).
    caps = DeclaredCapabilities(
        streaming=StreamingCapabilities(
            word_stability=FlagCap(supported=True),
            timestamps=StreamTimestampsCap(mode="native_frame_aligned"),
            word_timestamps=WordTimestampsCap(supported=True, granularities=["word"]),
        ),
        streaming_input=FlagCap(supported=True),
    )
    events = [
        TranscriptionEvent.partial("s0", "hi", stable_until=1, audio_processed_until=1.0),
        TranscriptionEvent.final("s0", "hi", words=[Word(start=0.0, end=0.5, text="hi")]),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events, capabilities=caps)
    assert not any(i.code.startswith("stream_exceeds_") for i in report.issues), [
        i.message for i in report.issues
    ]


# --------------------------------------------------------------------------- #
# assert_prefix_invariant (SF-3 -- assert the invariant, not partial counts)
# --------------------------------------------------------------------------- #
def test_assert_prefix_invariant_accepts_consistent_partials() -> None:
    # Monotonic, never-rewritten prefixes pass -- regardless of how many partials
    # survived coalescing.
    events = [
        TranscriptionEvent.partial("s0", "hel", stable_until=2),
        TranscriptionEvent.partial("s0", "hello", stable_until=3),
        TranscriptionEvent.final("s0", "hello"),
        TranscriptionEvent.done(),
    ]
    assert_prefix_invariant(events)  # no raise


def test_assert_prefix_invariant_tolerates_non_terminated_slice() -> None:
    # Unlike check_event_sequence, the prefix helper does NOT require a terminal:
    # it applies to a mid-stream slice (the common shape when asserting partials).
    events = [
        TranscriptionEvent.partial("s0", "he", stable_until=1),
        TranscriptionEvent.partial("s0", "hel", stable_until=2),
    ]
    assert_prefix_invariant(events)  # no raise despite no terminal


def test_assert_prefix_invariant_flags_frozen_prefix_rewrite() -> None:
    # A rewritten frozen prefix (text[:stable_until] changed) is the invariant
    # violation the helper exists to catch -- raised as AssertionError for tests.
    events = [
        TranscriptionEvent.partial("s0", "hello", stable_until=5),
        TranscriptionEvent.final("s0", "Hello.", stable_until=6),
        TranscriptionEvent.done(),
    ]
    with pytest.raises(AssertionError, match="frozen-prefix invariant"):
        assert_prefix_invariant(events)


# --------------------------------------------------------------------------- #
# check_streaming_param_gating
# --------------------------------------------------------------------------- #
class _StreamProps(BaseProperties):
    engine_id: str = "streamer"
    model_name: str = "demo"
    protocol_version: str = "0.2.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    selectable_languages: list[str] = []  # no language axis -> no default needed


_STREAM_CAPS = DeclaredCapabilities(
    streaming=StreamingCapabilities(),
    streaming_input=FlagCap(supported=True),
    streaming_output=FlagCap(supported=True),
)


class _GatingSession(TranscriptionSession):
    """Ends immediately (the base appends ``done``)."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        return
        yield  # pragma: no cover - makes this an async generator


class _GatingStreamEngine(EngineBase):
    """Streaming engine that relies on the base template's gating (compliant)."""

    properties: ClassVar[BaseProperties] = _StreamProps()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _STREAM_CAPS

    def __init__(self, *, strict: bool = True) -> None:
        self.config = _Config(engine="dummy", strict=strict)

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        return TranscriptionResult(text="")

    def _start_transcription(
        self,
        *,
        gated_params: RuntimeParams,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        return _GatingSession()


class _UngatedStreamEngine(_GatingStreamEngine):
    """Non-compliant: overrides the PUBLIC start_transcription, bypassing gating."""

    def start_transcription(
        self,
        *,
        audio_format: Any = None,
        params: Any = None,
        audio: Any = None,
        deadlines: Any = None,
    ) -> TranscriptionSession:
        # Forgot to gate: returns a session for ANY params, no gate_params call.
        return _GatingSession()


class _BatchOnlyEngine(EngineBase):
    """No streaming support declared; start_transcription raises unsupported."""

    properties: ClassVar[BaseProperties] = _StreamProps()
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities()

    def __init__(self) -> None:
        self.config = _Config(engine="dummy")

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        return TranscriptionResult(text="")


class _AllSupportedStreamEngine(_GatingStreamEngine):
    """Supports every probed param with no violable sub-constraint -> nothing to gate.

    Every granularity is offered and the prompt budget is unbounded, so neither
    the feature-level probes nor the sub-constraint fallback can build a
    violating request.
    """

    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(
            word_timestamps=WordTimestampsCap(
                supported=True, granularities=["word", "segment", "char"]
            ),
            guidance=GuidanceCaps(prompt=PromptCap(supported=True)),
        ),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
    )


class _PromptConstrainedStreamEngine(_GatingStreamEngine):
    """Supports every probed feature; prompt carries a small ``max_tokens`` budget."""

    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(
            word_timestamps=WordTimestampsCap(
                supported=True, granularities=["word", "segment", "char"]
            ),
            guidance=GuidanceCaps(
                prompt=PromptCap(supported=True, constraints=PromptConstraints(max_tokens=3))
            ),
        ),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
    )


class _GranularityLimitedStreamEngine(_GatingStreamEngine):
    """Supports every probed feature; word timestamps offer only ``word``."""

    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(
            word_timestamps=WordTimestampsCap(supported=True, granularities=["word"]),
            guidance=GuidanceCaps(prompt=PromptCap(supported=True)),
        ),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
    )


def test_streaming_gating_strict_engine_passes() -> None:
    report = check_streaming_param_gating(_GatingStreamEngine(strict=True))
    assert report.passed is True, [i.message for i in report.issues]


def test_streaming_gating_best_effort_engine_passes() -> None:
    report = check_streaming_param_gating(_GatingStreamEngine(strict=False))
    assert report.passed is True, [i.message for i in report.issues]


def test_streaming_gating_ungated_engine_fails() -> None:
    # strict engine that bypassed the template accepts the unsupported param.
    report = check_streaming_param_gating(_UngatedStreamEngine(strict=True))
    assert report.passed is False
    assert any("without raising" in i.message for i in report.issues)


def test_streaming_gating_ungated_best_effort_engine_fails() -> None:
    # best_effort engine that bypassed the template never emits the diagnostic.
    report = check_streaming_param_gating(_UngatedStreamEngine(strict=False))
    assert report.passed is False
    assert any("silently swallowed" in i.message for i in report.issues)


def test_streaming_gating_non_streaming_engine_is_noop_pass() -> None:
    report = check_streaming_param_gating(_BatchOnlyEngine())
    assert report.passed is True
    assert report.issues == []


def test_streaming_gating_all_supported_engine_is_noop_pass() -> None:
    report = check_streaming_param_gating(_AllSupportedStreamEngine())
    assert report.passed is True
    assert report.issues == []


def test_streaming_gating_best_effort_engine_raising_fails() -> None:
    # A best_effort engine that wrongly RAISES for the unsupported param fails.
    class _RaisingBestEffortEngine(_GatingStreamEngine):
        def start_transcription(
            self,
            *,
            audio_format: Any = None,
            params: Any = None,
            audio: Any = None,
            deadlines: Any = None,
        ) -> TranscriptionSession:
            raise UnsupportedFeatureError("I refuse the param even in best_effort.")

    report = check_streaming_param_gating(_RaisingBestEffortEngine(strict=False))
    assert report.passed is False
    assert any("MUST drop it" in i.message for i in report.issues)


def test_streaming_gating_engine_crash_is_reported_not_raised() -> None:
    # A non-UnsupportedFeatureError exception (an engine bug)
    # MUST surface as a compliance error, never crash the whole compliance run.
    class _CrashingEngine(_GatingStreamEngine):
        def start_transcription(
            self,
            *,
            audio_format: Any = None,
            params: Any = None,
            audio: Any = None,
            deadlines: Any = None,
        ) -> TranscriptionSession:
            raise RuntimeError("engine exploded")

    report = check_streaming_param_gating(_CrashingEngine(strict=True))
    assert report.passed is False
    assert any(
        "RuntimeError" in i.message and "UnsupportedFeatureError" in i.message
        for i in report.issues
    )


def test_streaming_gating_diagnostics_raise_is_reported_not_raised() -> None:
    # A best_effort engine whose session.diagnostics() itself raises MUST surface
    # as a compliance error, never crash the whole run (check promises Raises: None).
    class _DiagRaisingSession(_GatingSession):
        def diagnostics(self) -> list[Diagnostic]:
            raise RuntimeError("diagnostics exploded")

    class _DiagRaisingEngine(_GatingStreamEngine):
        def _start_transcription(
            self,
            *,
            gated_params: RuntimeParams,
            audio_format: Any = None,
            prepared_audio: PreparedAudio | None = None,
        ) -> TranscriptionSession:
            return _DiagRaisingSession()

    report = check_streaming_param_gating(_DiagRaisingEngine(strict=False))
    assert report.passed is False
    assert any(
        "diagnostics() raised" in i.message and "RuntimeError" in i.message for i in report.issues
    )


# --------------------------------------------------------------------------- #
# Sub-constraint gating fallback: every probed feature is
# supported, so the check must violate a declared sub-constraint instead.
# --------------------------------------------------------------------------- #
def test_streaming_gating_sub_constraint_prompt_strict_passes() -> None:
    # Strict engine on the base template raises for the over-budget prompt.
    report = check_streaming_param_gating(_PromptConstrainedStreamEngine(strict=True))
    assert report.passed is True, [i.message for i in report.issues]


def test_streaming_gating_sub_constraint_prompt_best_effort_passes() -> None:
    # best_effort engine truncates and surfaces the prompt_truncated diagnostic.
    report = check_streaming_param_gating(_PromptConstrainedStreamEngine(strict=False))
    assert report.passed is True, [i.message for i in report.issues]


class _UngatedPromptConstrainedEngine(_PromptConstrainedStreamEngine):
    """Bypasses the template: accepts the over-budget prompt without gating."""

    def start_transcription(
        self,
        *,
        audio_format: Any = None,
        params: Any = None,
        audio: Any = None,
        deadlines: Any = None,
    ) -> TranscriptionSession:
        return _GatingSession()


def test_streaming_gating_sub_constraint_ungated_strict_fails() -> None:
    report = check_streaming_param_gating(_UngatedPromptConstrainedEngine(strict=True))
    assert report.passed is False
    assert any("'prompt'" in i.message and "without raising" in i.message for i in report.issues)


def test_streaming_gating_sub_constraint_ungated_best_effort_fails() -> None:
    report = check_streaming_param_gating(_UngatedPromptConstrainedEngine(strict=False))
    assert report.passed is False
    assert any("'prompt_truncated'" in i.message for i in report.issues)


def test_streaming_gating_sub_constraint_granularity_strict_passes() -> None:
    # The prompt is unconstrained, so the fallback probes an unoffered
    # word-timestamp granularity instead; the template engine gates it.
    report = check_streaming_param_gating(_GranularityLimitedStreamEngine(strict=True))
    assert report.passed is True, [i.message for i in report.issues]


def test_streaming_gating_sub_constraint_granularity_best_effort_passes() -> None:
    report = check_streaming_param_gating(_GranularityLimitedStreamEngine(strict=False))
    assert report.passed is True, [i.message for i in report.issues]


def test_pick_sub_constraint_probe_granularity_carries_its_code() -> None:
    # The granularity probe must request a granularity OUTSIDE the declared set
    # and carry the drop diagnostic code the runtime emits for it.
    probe = compliance_module._pick_sub_constraint_probe(  # pyright: ignore[reportPrivateUsage]
        _GranularityLimitedStreamEngine()
    )
    assert probe is not None
    field_name, params, expected_code = probe
    assert field_name == "word_timestamps"
    assert params.word_timestamps is not None
    assert params.word_timestamps.value != "word"
    assert expected_code == "unsupported_granularity_ignored"


def test_pick_sub_constraint_probe_none_without_streaming_domain() -> None:
    # No streaming domain -> no constrainable nodes resolve -> no probe. (The
    # public check never reaches the helper in this state; it stays fail-safe.)
    probe = compliance_module._pick_sub_constraint_probe(  # pyright: ignore[reportPrivateUsage]
        _BatchOnlyEngine()
    )
    assert probe is None


# --------------------------------------------------------------------------- #
# The sub-constraint probe is bounded against extreme declarations
# --------------------------------------------------------------------------- #
class _ExtremeBudgetEngine(_GatingStreamEngine):
    """Legal-but-extreme ``max_tokens`` (no upper bound exists on the field)."""

    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(
            word_timestamps=WordTimestampsCap(supported=True, granularities=["word"]),
            guidance=GuidanceCaps(
                prompt=PromptCap(supported=True, constraints=PromptConstraints(max_tokens=10**9))
            ),
        ),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
    )


def test_streaming_gating_extreme_max_tokens_completes() -> None:
    # A 10^9-token budget must not make the probe materialize a
    # multi-gigabyte prompt (it was allocated OUTSIDE the crash containment and
    # would OOM the run). Past the cap the prompt probe is skipped and the
    # granularity probe exercises the sub-constraint contract instead.
    probe = compliance_module._pick_sub_constraint_probe(  # pyright: ignore[reportPrivateUsage]
        _ExtremeBudgetEngine()
    )
    assert probe is not None
    assert probe[0] == "word_timestamps"
    report = check_streaming_param_gating(_ExtremeBudgetEngine(strict=True))
    assert report.passed is True, [i.message for i in report.issues]


def test_streaming_gating_extreme_max_tokens_without_other_probe_is_clean() -> None:
    # Past the cap with every granularity offered there is no violable
    # sub-constraint left: the check completes as a clean no-op pass.
    class _ExtremeBudgetOnlyEngine(_GatingStreamEngine):
        declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
            streaming=StreamingCapabilities(
                word_timestamps=WordTimestampsCap(
                    supported=True, granularities=["word", "segment", "char"]
                ),
                guidance=GuidanceCaps(
                    prompt=PromptCap(
                        supported=True, constraints=PromptConstraints(max_tokens=10**9)
                    )
                ),
            ),
            streaming_input=FlagCap(supported=True),
            streaming_output=FlagCap(supported=True),
        )

    report = check_streaming_param_gating(_ExtremeBudgetOnlyEngine(strict=True))
    assert report.passed is True
    assert report.issues == []


def test_streaming_gating_probe_selection_crash_contained() -> None:
    # Probe selection reads engine-author surface (supports() delegates to
    # effective_capabilities); a crash there must surface as a compliance
    # error, never escape a function promising ``Raises: None``.
    class _BrokenCapsEngine(_GatingStreamEngine):
        @property
        def effective_capabilities(self) -> DeclaredCapabilities:
            raise RuntimeError("capabilities exploded")

    report = check_streaming_param_gating(_BrokenCapsEngine(strict=True))
    assert report.passed is False
    assert any("selecting a streaming gating probe" in i.message for i in report.issues)


# --------------------------------------------------------------------------- #
# Every ComplianceIssue carries a machine-readable, stable code.
# --------------------------------------------------------------------------- #
def test_compliance_issue_has_code_field() -> None:
    # The structured code is the programmatic contract (mirrors Diagnostic.code);
    # CI matches the code, not the rewordable message.
    issue = ComplianceIssue(level="error", code="some_code", message="m", model=None)
    assert issue.code == "some_code"


def test_all_issue_codes_are_nonempty_strings() -> None:
    # Across a report that exercises many issue kinds, every issue MUST carry a
    # non-empty code so no construction site forgot it.
    report = check_entrypoints(
        registry=_registry_many(
            ("widened_factory", "dummy/demo"),
            ("good_factory", "dummy2/demo"),
        )
    )
    assert report.issues  # the widened engine produced at least one issue
    for issue in report.issues:
        assert isinstance(issue.code, str) and issue.code


def test_event_sequence_passes_through_guard_code() -> None:
    # The guard's stable diagnostic code is surfaced STRUCTURALLY
    # (namespaced) instead of only being interpolated into the message.
    events = [
        TranscriptionEvent.final("a", "hello"),
        TranscriptionEvent.final("a", "rewritten"),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is False
    assert any(i.code == "streaming_invariant:lifecycle_final_after_final" for i in report.issues)


def test_event_sequence_soft_obligation_code_is_namespaced() -> None:
    events = [
        TranscriptionEvent.final("a", "你好世界", stable_until=4),
        TranscriptionEvent.supersede(["a"], ["b"]),
        TranscriptionEvent.final("b", "你好", stable_until=2),
        TranscriptionEvent.done(),
    ]
    report = check_event_sequence(events)
    assert report.passed is True
    assert any(
        i.code == "streaming_soft:supersede_obligation_unfulfilled" and i.level == "warning"
        for i in report.issues
    )


# --------------------------------------------------------------------------- #
# IC.2 engine-identity collisions (shadowed engine_id) and invalid
# entry points are REPORTED as issues, never silently passed (default run) and
# never as a raised exception (strict run).
# --------------------------------------------------------------------------- #
def test_check_entrypoints_reports_shadowed_engine_id() -> None:
    # Two distributions (dist-less, distinct targets) claim engine_id 'dummy':
    # config.engine routing is ambiguous (IC.2). A default compliance run MUST
    # fail on it, not pass with a mere discovery log line.
    eps = [
        EntryPoint(
            name="dummy/a", value="tests.test_compliance:good_factory", group="standard_asr.models"
        ),
        EntryPoint(
            name="dummy/b",
            value="tests.test_compliance:widened_factory",
            group="standard_asr.models",
        ),
    ]
    registry = discover_models(eps=eps, strict=False)
    assert registry.shadowed_engine_ids == {"dummy"}

    report = check_entrypoints(registry=registry)
    assert report.passed is False
    collision = [i for i in report.issues if i.code == "engine_id_collision"]
    assert len(collision) == 1
    assert "more than one distribution" in collision[0].message
    assert collision[0].model == "dummy"


def test_check_entrypoints_strict_invalid_name_reported_not_raised() -> None:
    # An invalid entry-point name under strict discovery would normally RAISE
    # EntrypointValidationError; check_entrypoints (Raises: None) MUST convert it
    # to an error issue and still return a report.
    eps = [
        EntryPoint(
            name="dummy/bad name",  # space -> invalid model name
            value="tests.test_compliance:good_factory",
            group="standard_asr.models",
        ),
    ]
    # passing registry=None forces internal discovery; inject eps via the helper.
    report = _check_entrypoints_discovering(eps, strict_discovery=True)
    assert any(i.code == "entrypoint_invalid" for i in report.issues)
    # Reported as an error, never raised.
    assert report.passed is False


def _check_entrypoints_discovering(
    eps: list[EntryPoint], *, strict_discovery: bool
) -> compliance_module.ComplianceReport:
    """Run check_entrypoints with internal discovery over injected entry points."""
    import standard_asr.compliance as _cm

    real_discover = _cm.discover_models

    def _fake_discover(*args: Any, strict: bool = False, **kwargs: Any) -> ModelRegistry:
        return real_discover(eps, strict=strict)

    original = _cm.discover_models
    _cm.discover_models = _fake_discover  # type: ignore[assignment]
    try:
        return _cm.check_entrypoints(strict_discovery=strict_discovery)
    finally:
        _cm.discover_models = original  # type: ignore[assignment]


def test_check_entrypoints_strict_invalid_name_still_checks_valid_engines() -> None:
    # After capturing the strict failure, discovery is re-run leniently so the
    # valid engines in the same environment are still checked.
    eps = [
        EntryPoint(
            name="dummy/bad name",
            value="tests.test_compliance:good_factory",
            group="standard_asr.models",
        ),
        EntryPoint(
            name="good/demo",
            value="tests.test_compliance:good_factory",
            group="standard_asr.models",
        ),
    ]
    report = _check_entrypoints_discovering(eps, strict_discovery=True)
    # The invalid name was reported...
    assert any(i.code == "entrypoint_invalid" for i in report.issues)
    # ...and the valid engine's class-level checks still ran (its config_type
    # warning is present), proving lenient re-discovery happened.
    assert any(i.code == "missing_config_type" for i in report.issues)


# --------------------------------------------------------------------------- #
# Credential-requiring factory -> warning skip (not an error), and
# per-engine crash containment (a broken property does not abort the run).
# --------------------------------------------------------------------------- #
class _CredentialedASR(_GoodASR):
    """Zero-arg factory that raises ConfigError when a credential is absent (IC.4)."""


def credentialed_factory() -> _CredentialedASR:  # pyright: ignore[reportUnusedFunction]
    raise ConfigError("STANDARD_ASR_DUMMY_API_KEY is required but not set.")


def test_check_entrypoints_missing_credential_is_warning_not_error() -> None:
    # IC.4: a credentialed engine's factory MUST raise when the credential is
    # absent (explicit > env > raise). On a clean CI that is the CORRECT behavior,
    # so it MUST be a warning skip, not a compliance error -- otherwise the verdict
    # depends on the runtime's credential state, not the plugin.
    report = check_entrypoints(registry=_registry("credentialed_factory"))
    assert report.passed is True, [i.message for i in report.issues]
    skips = [i for i in report.issues if i.code == "factory_requires_config"]
    assert len(skips) == 1
    assert skips[0].level == "warning"
    assert "STANDARD_ASR" in skips[0].message


def credentialed_validation_factory() -> _CredentialedASR:  # pyright: ignore[reportUnusedFunction]
    # A pydantic ValidationError (missing required field) is the other shape of
    # "needs configuration" and must also be a warning skip, not an error.
    _Config(engine="dummy", default_language=object())  # type: ignore[arg-type]
    return _CredentialedASR()  # pragma: no cover - never reached


def test_check_entrypoints_validation_error_factory_is_warning() -> None:
    report = check_entrypoints(registry=_registry("credentialed_validation_factory"))
    assert report.passed is True, [i.message for i in report.issues]
    assert any(i.code == "factory_requires_config" for i in report.issues)


class _CrashingPropsASR(_GoodASR):
    """A buggy @property properties that raises a NON-AttributeError."""

    @property
    def properties(self) -> _Props:  # type: ignore[override]
        raise RuntimeError("properties exploded")


def crashing_props_factory() -> _CrashingPropsASR:  # pyright: ignore[reportUnusedFunction]
    return _CrashingPropsASR()


def test_check_entrypoints_crashing_property_is_contained() -> None:
    # A property that raises a non-AttributeError (getattr default only swallows
    # AttributeError) must surface as an error issue against that engine and MUST
    # NOT abort the run (check_entrypoints promises Raises: None); the other engine
    # is still checked.
    report = check_entrypoints(
        registry=_registry_many(
            ("crashing_props_factory", "dummy/demo"),
            ("widened_factory", "dummy2/demo"),
        )
    )
    assert report.passed is False
    assert any(i.code == "engine_check_crashed" and i.model == "dummy/demo" for i in report.issues)
    # The second engine was still reached and checked.
    assert any(i.code == "effective_widens_declared" for i in report.issues)


def test_check_entrypoints_factory_failure_non_config_is_still_error() -> None:
    # A factory crash that is NOT a config/validation problem stays an error.
    class _BoomASR(_GoodASR):
        pass

    def _boom_factory() -> _BoomASR:
        raise RuntimeError("unexpected boom")

    import standard_asr.compliance as _cm

    # Register the local factory under a module attribute the entry point resolves.
    setattr(_cm, "_boom_factory_for_test", _boom_factory)
    try:
        eps = [
            EntryPoint(
                name="dummy/demo",
                value="standard_asr.compliance:_boom_factory_for_test",
                group="standard_asr.models",
            )
        ]
        registry = discover_models(eps=eps, strict=True)
        report = check_entrypoints(registry=registry)
        assert report.passed is False
        assert any(i.code == "entrypoint_factory_failed" for i in report.issues)
    finally:
        delattr(_cm, "_boom_factory_for_test")


# --------------------------------------------------------------------------- #
# An EngineBase engine that DECLARES streaming but never implements
# the hook is a false PASS no longer -- caught both at the surface check and in
# the gating probe (strict).
# --------------------------------------------------------------------------- #
class _StreamingDeclaredNoHookEngine(EngineBase):
    """Declares streaming_input/output but never overrides _start_transcription."""

    properties: ClassVar[BaseProperties] = _StreamProps()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _STREAM_CAPS

    def __init__(self, *, strict: bool = True) -> None:
        self.config = _Config(engine="dummy", strict=strict)

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        return TranscriptionResult(text="")


def streaming_declared_no_hook_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _StreamingDeclaredNoHookEngine
):
    return _StreamingDeclaredNoHookEngine()


def test_required_surface_flags_streaming_declared_but_hook_not_implemented() -> None:
    # _check_required_surface uses the runtime _overrides_streaming() predicate, so
    # the base template's always-present start_transcription cannot certify an
    # engine that declares streaming yet never overrides the hook (capability lie).
    report = check_entrypoints(registry=_registry("streaming_declared_no_hook_factory"))
    assert report.passed is False
    assert any(i.code == "streaming_declared_not_implemented" for i in report.issues)


class _CompliantStreamingEngine(_GatingStreamEngine):
    """A streaming EngineBase that DOES implement the hook (passes the surface check)."""

    properties: ClassVar[BaseProperties] = _StreamProps(engine_id="streamer-ok")


def compliant_streaming_factory() -> _CompliantStreamingEngine:  # pyright: ignore[reportUnusedFunction]
    return _CompliantStreamingEngine()


def test_required_surface_streaming_enginebase_with_hook_passes() -> None:
    # The other side of the _overrides_streaming() branch: a streaming EngineBase
    # that implements the hook passes the required-surface check. The entry-point
    # key must match the engine's declared model_id ("streamer-ok/demo").
    report = check_entrypoints(
        registry=_registry("compliant_streaming_factory", "streamer-ok/demo")
    )
    assert not any(i.code == "streaming_declared_not_implemented" for i in report.issues), [
        i.message for i in report.issues
    ]


def test_gating_strict_flags_streaming_declared_but_hook_not_implemented() -> None:
    # The gating probe's strict branch distinguishes a real gating raise
    # (param==field) from the base template's "does not support streaming" raise
    # (param=None) -- the latter is a capability lie, not a clean pass.
    report = check_streaming_param_gating(_StreamingDeclaredNoHookEngine(strict=True))
    assert report.passed is False
    assert any(i.code == "gating_probe_unexpected_unsupported" for i in report.issues)


# --------------------------------------------------------------------------- #
# A streaming_input engine that legitimately FAIL-LOUDS on a missing
# audio_format (spec AI R6) is NOT misjudged -- the probe synthesizes a legal
# wire format from the engine's Properties.
# --------------------------------------------------------------------------- #
class _FailLoudOnMissingFormatEngine(_GatingStreamEngine):
    """Compliant engine that REQUIRES an audio_format (does not self-manage wire)."""

    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(),
        streaming_input=FlagCap(supported=True),
    )

    def _start_transcription(
        self,
        *,
        gated_params: RuntimeParams,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        if audio_format is None:
            # spec AI R6: bare-PCM streaming locks the sample rate at session
            # establishment, so a non-self-managing engine fail-louds here.
            raise ValueError("audio_format is required for this engine (spec AI R6).")
        return _GatingSession()


def test_gating_does_not_misjudge_fail_loud_engine_best_effort() -> None:
    # The headline direction: a best_effort engine that correctly
    # fail-louds on a missing audio_format MUST NOT be reported as non-compliant.
    # The probe now passes a synthesized legal audio_format, so gating runs.
    report = check_streaming_param_gating(_FailLoudOnMissingFormatEngine(strict=False))
    assert report.passed is True, [i.message for i in report.issues]


def test_gating_does_not_misjudge_fail_loud_engine_strict() -> None:
    report = check_streaming_param_gating(_FailLoudOnMissingFormatEngine(strict=True))
    assert report.passed is True, [i.message for i in report.issues]


def test_gating_synthesizes_audio_format_respecting_required_rate() -> None:
    # When the engine hard-requires a wire sample rate, the synthesized format
    # MUST use it (else ensure_stream_format_supported would reject the probe).
    class _RequiredRateProps(_StreamProps):
        required_input_sample_rate: int | None = 8000
        accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = "any"
        wire_encodings: list[str] | None = ["pcm_s16le"]

    class _RequiredRateEngine(_FailLoudOnMissingFormatEngine):
        properties: ClassVar[BaseProperties] = _RequiredRateProps()

    fmt = compliance_module._synthesize_probe_audio_format(  # pyright: ignore[reportPrivateUsage]
        _RequiredRateEngine(strict=True)
    )
    assert isinstance(fmt, AudioFormat)
    assert fmt.sample_rate == 8000
    assert fmt.encoding == "pcm_s16le"
    assert fmt.channels == 1
    # And the full check passes (the synthesized format is accepted).
    report = check_streaming_param_gating(_RequiredRateEngine(strict=True))
    assert report.passed is True, [i.message for i in report.issues]


# --------------------------------------------------------------------------- #
# Streaming_output-only engines -- strict probes safely (gating
# raises before inference); best_effort is skipped (would be a billable probe).
# --------------------------------------------------------------------------- #
class _OutputOnlyStreamEngine(EngineBase):
    """streaming_output only (no streaming_input); overrides the hook."""

    properties: ClassVar[BaseProperties] = _StreamProps()
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(),
        streaming_output=FlagCap(supported=True),
    )

    def __init__(self, *, strict: bool = True) -> None:
        self.config = _Config(engine="dummy", strict=strict)

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        return TranscriptionResult(text="")

    def _start_transcription(
        self,
        *,
        gated_params: RuntimeParams,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:  # pragma: no cover - strict gating raises first
        return _GatingSession()


def test_gating_output_only_strict_passes_without_inference() -> None:
    # strict: gate_params raises before the silent audio is decoded or the model
    # is touched, so the probe exercises gating with no inference side effect.
    report = check_streaming_param_gating(_OutputOnlyStreamEngine(strict=True))
    assert report.passed is True, [i.message for i in report.issues]


def test_gating_output_only_best_effort_is_skipped_billable() -> None:
    # best_effort: reaching gating needs an ``audio`` input that would be decoded
    # and fed to the model (billable). Skip with an honest warning instead.
    report = check_streaming_param_gating(_OutputOnlyStreamEngine(strict=False))
    assert report.passed is True
    skips = [i for i in report.issues if i.code == "gating_probe_skipped_billable"]
    assert len(skips) == 1
    assert skips[0].level == "warning"


def test_gating_best_effort_streaming_input_probe_is_side_effect_free() -> None:
    # The best_effort gating verdict reads only session.diagnostics -- a
    # pure read of construction-time diagnostics on a session the base template
    # constructs but does NOT enter. The probe MUST NOT open the session; for a
    # real cloud engine that open is a billable wire handshake. Pre-fix a
    # try/finally tore the unopened session down via SyncSession, which OPENED it
    # (open + produce + close); the fix drops that teardown entirely.
    calls = {"open": 0, "close": 0, "produce": 0}

    class _CountingSession(_GatingSession):
        async def _open(self) -> None:
            calls["open"] += 1
            await super()._open()

        async def _close(self) -> None:
            calls["close"] += 1
            await super()._close()

        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            calls["produce"] += 1
            return
            yield  # pragma: no cover - makes this an async generator

    class _CountingEngine(_GatingStreamEngine):
        def _start_transcription(
            self,
            *,
            gated_params: RuntimeParams,
            audio_format: Any = None,
            prepared_audio: PreparedAudio | None = None,
        ) -> TranscriptionSession:
            return _CountingSession()

    report = check_streaming_param_gating(_CountingEngine(strict=False))
    assert report.passed is True, [i.message for i in report.issues]
    assert calls == {"open": 0, "close": 0, "produce": 0}


def test_gating_probe_context_unbuildable_is_reported() -> None:
    # If a legal wire audio_format cannot be synthesized from Properties (a broken
    # native_sample_rate read), report it rather than crash.
    class _NoFormatEngine(_FailLoudOnMissingFormatEngine):
        @property
        def properties(self) -> _StreamProps:  # type: ignore[override]
            class _Bad:
                required_input_sample_rate = None
                engine_id = "broken"

                @property
                def native_sample_rate(self) -> int:
                    raise RuntimeError("no native rate")

            return _Bad()  # type: ignore[return-value]

    report = check_streaming_param_gating(_NoFormatEngine(strict=True))
    assert report.passed is False
    assert any(i.code == "gating_probe_context_unbuildable" for i in report.issues)


def test_gating_probe_context_unbuildable_when_no_sample_rate() -> None:
    # recommended_wire_format() returns None (no usable sample rate, not a raise):
    # the gating probe still cannot build a legal context, and reports it.
    class _NoRateEngine(_GatingStreamEngine):
        properties: ClassVar[BaseProperties] = _StreamProps.model_construct(
            native_sample_rate=0, required_input_sample_rate=None
        )

    report = check_streaming_param_gating(_NoRateEngine(strict=True))
    assert report.passed is False
    assert any(i.code == "gating_probe_context_unbuildable" for i in report.issues)


def test_safe_engine_id_contains_raising_properties() -> None:
    # B: a ``properties`` that raises a NON-AttributeError (which
    # getattr does not swallow) must not escape the behavioral checks. Attribution
    # falls back to None and the run continues.
    class _RaisingPropsEngine(_GatingStreamEngine):
        @property
        def properties(self) -> _StreamProps:  # type: ignore[override]
            raise RuntimeError("properties read exploded")

    engine = _RaisingPropsEngine(strict=True)
    assert (
        compliance_module._safe_engine_id(engine)  # pyright: ignore[reportPrivateUsage]
        is None
    )
    # The gating check still returns a report (Raises: None) despite the broken
    # properties: synthesis reads properties, so it surfaces a contained error
    # rather than crashing the run. The key invariant is that it does NOT raise.
    report = check_streaming_param_gating(engine)
    assert report.passed is False
    assert any(i.code == "gating_probe_context_unbuildable" for i in report.issues)


def test_safe_engine_id_handles_missing_properties() -> None:
    # A structural object with no properties at all -> None (no crash).
    class _NoProps:
        pass

    assert (
        compliance_module._safe_engine_id(_NoProps())  # pyright: ignore[reportPrivateUsage]
        is None
    )


# --------------------------------------------------------------------------- #
# Provider_params swap-safety probe (always-raise, both policies).
# --------------------------------------------------------------------------- #
class _SwapSafeEngine(EngineBase):
    """Relies on the base template, which enforces R3 provider_params swap-safety."""

    properties: ClassVar[BaseProperties] = _StreamProps()
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities()

    def __init__(self, *, strict: bool = True) -> None:
        self.config = _Config(engine="dummy", strict=strict)

    def _transcribe(
        self, prepared: PreparedAudio, params: RuntimeParams
    ) -> TranscriptionResult:  # pragma: no cover - provider_params raises before this
        return TranscriptionResult(text="")


def test_provider_params_swap_safety_strict_passes() -> None:
    report = check_provider_params_swap_safety(_SwapSafeEngine(strict=True))
    assert report.passed is True, [i.message for i in report.issues]


def test_provider_params_swap_safety_best_effort_passes() -> None:
    # R3: the rejection is ALWAYS raised, independent of strict/best_effort.
    report = check_provider_params_swap_safety(_SwapSafeEngine(strict=False))
    assert report.passed is True, [i.message for i in report.issues]


class _SwapUnsafeEngine(_SwapSafeEngine):
    """Bypasses the template's transcribe and forgets the provider_params check."""

    def transcribe(self, audio: Any, params: RuntimeParams | None = None) -> TranscriptionResult:
        # Silently accepts ANY provider_params -- the swap bug R3 makes loud.
        return TranscriptionResult(text="ok")


def test_provider_params_swap_safety_accepted_fails() -> None:
    report = check_provider_params_swap_safety(_SwapUnsafeEngine(strict=True))
    assert report.passed is False
    assert any(i.code == "provider_params_swap_accepted" for i in report.issues)


class _SwapWrongErrorEngine(_SwapSafeEngine):
    """Bypasses the template and raises the WRONG exception type for swap."""

    def transcribe(self, audio: Any, params: RuntimeParams | None = None) -> TranscriptionResult:
        raise RuntimeError("not the contractual InvalidProviderParamError")


def test_provider_params_swap_safety_wrong_error_fails() -> None:
    report = check_provider_params_swap_safety(_SwapWrongErrorEngine(strict=True))
    assert report.passed is False
    assert any(i.code == "provider_params_swap_not_enforced" for i in report.issues)


def test_provider_params_swap_probe_raises_invalid_for_engine_without_params() -> None:
    # Sanity: an engine declaring NO provider_params_type still rejects a foreign
    # one (gate_params: "this engine accepts no provider_params").
    engine = _SwapSafeEngine(strict=True)
    foreign = compliance_module._ForeignProviderParams()  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(InvalidProviderParamError):
        engine.transcribe(np.zeros(1, dtype=np.float32), RuntimeParams(provider_params=foreign))


def test_provider_params_swap_safety_unverifiable_when_language_config_invalid() -> None:
    # Validate_language_config runs BEFORE the provider_params gate, so an
    # engine with a language axis but no default_language raises ConfigError before
    # R3 can be exercised. That must be reported as unverifiable -- NOT mislabeled
    # as a swap miss (provider_params_swap_not_enforced).
    report = check_provider_params_swap_safety(_AxisNoDefaultEngine())
    assert report.passed is False
    assert not any(i.code == "provider_params_swap_not_enforced" for i in report.issues)
    assert any(i.code == "provider_params_swap_unverifiable" for i in report.issues)
    assert any("language_config_invalid" in i.message for i in report.issues)


# --------------------------------------------------------------------------- #
# Behavioral reports carry registry=None (the field no
# longer lies about an empty model registry).
# --------------------------------------------------------------------------- #
def test_behavioral_reports_have_none_registry() -> None:
    assert check_event_sequence([], allow_empty=True).registry is None
    assert check_streaming_param_gating(_BatchOnlyEngine()).registry is None
    assert check_sync_bridge(_CleanSession, timeout=5.0).registry is None
    assert check_provider_params_swap_safety(_SwapSafeEngine(strict=True)).registry is None


# --------------------------------------------------------------------------- #
# minor: sync-bridge worker is a daemon (does not block exit).
# --------------------------------------------------------------------------- #
def test_sync_bridge_worker_is_daemon() -> None:
    # compliance.threading and streaming.threading are the SAME module object, so
    # monkeypatching Thread here also captures SyncSession's always-daemon loop
    # thread. Key the recorder by thread name and assert specifically on the worker
    # -- otherwise a worker regression to daemon=False is masked by the loop thread.
    created: list[tuple[str, bool]] = []
    real_thread = compliance_module.threading.Thread

    def _record(*args: Any, **kwargs: Any) -> Any:
        t = real_thread(*args, **kwargs)
        created.append((t.name, t.daemon))
        return t

    import standard_asr.compliance as _cm

    original = _cm.threading.Thread
    _cm.threading.Thread = _record  # type: ignore[assignment, misc]
    try:
        check_sync_bridge(_CleanSession, timeout=5.0)
    finally:
        _cm.threading.Thread = original  # type: ignore[misc]
    worker_daemons = [daemon for name, daemon in created if name == "compliance-sync-bridge"]
    assert worker_daemons == [True]


def test_sync_bridge_timeout_message_disambiguates_slow_vs_deadlock() -> None:
    class _HangSession(TranscriptionSession):
        async def _open(self) -> None:
            time.sleep(1.0)

        async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
            yield TranscriptionEvent.done()  # pragma: no cover - never reached

    report = check_sync_bridge(_HangSession, timeout=0.05)
    assert report.passed is False
    msg = next(i.message for i in report.issues if i.code == "sync_bridge_did_not_terminate")
    assert "deadlock OR" in msg
    assert "larger timeout" in msg


# _check_prepare_hook: the optional prepare warm-up hook MUST be
# a synchronous, zero-argument method when present (spec IC.11).
# --------------------------------------------------------------------------- #
class _AsyncPrepareASR(_GoodASR):
    """Declares an async prepare -- a silent-false-success risk."""

    async def prepare(self) -> None:  # noqa: D401 - test double
        return None


def async_prepare_factory() -> _AsyncPrepareASR:  # pyright: ignore[reportUnusedFunction]
    return _AsyncPrepareASR()


class _ArgsPrepareASR(_GoodASR):
    """Declares a prepare() that requires an argument (cannot be driven)."""

    def prepare(self, required: object) -> None:  # noqa: D401 - test double
        return None


def args_prepare_factory() -> _ArgsPrepareASR:  # pyright: ignore[reportUnusedFunction]
    return _ArgsPrepareASR()


class _GoodPrepareASR(_GoodASR):
    """Declares a compliant synchronous zero-argument prepare()."""

    def prepare(self) -> None:  # noqa: D401 - test double
        return None


def good_prepare_factory() -> _GoodPrepareASR:  # pyright: ignore[reportUnusedFunction]
    return _GoodPrepareASR()


def test_check_entrypoints_async_prepare_is_error() -> None:
    # An `async def prepare` would be called but never awaited and
    # silently reported "complete"; the suite must catch it as an error.
    report = check_entrypoints(registry=_registry("async_prepare_factory"))
    assert report.passed is False
    assert any("prepare()" in i.message and "coroutine" in i.message for i in report.issues), [
        i.message for i in report.issues
    ]


def test_check_entrypoints_prepare_requiring_args_is_error() -> None:
    # A prepare with required arguments cannot be driven by the
    # toolchain; recorded as an error.
    report = check_entrypoints(registry=_registry("args_prepare_factory"))
    assert report.passed is False
    assert any("prepare()" in i.message and "no arguments" in i.message for i in report.issues), [
        i.message for i in report.issues
    ]


def test_check_entrypoints_good_prepare_passes() -> None:
    # A compliant synchronous zero-argument prepare() raises no prepare issue.
    report = check_entrypoints(registry=_registry("good_prepare_factory"))
    assert not any("prepare()" in i.message for i in report.issues), [
        i.message for i in report.issues
    ]


def test_check_entrypoints_no_prepare_hook_is_fine() -> None:
    # The common case: a structural engine that declares no prepare() hook is
    # not flagged (the hook is optional, spec IC.11).
    report = check_entrypoints(registry=_registry("good_factory"))
    assert not any("prepare()" in i.message for i in report.issues), [
        i.message for i in report.issues
    ]


class _UnIntrospectablePrepareASR(_GoodASR):
    """prepare() is a callable whose signature cannot be introspected."""

    # `type` is callable, not a coroutine function, but inspect.signature(type)
    # raises ValueError -- exercising the defensive signature-read guard.
    prepare = type


def unintrospectable_prepare_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _UnIntrospectablePrepareASR
):
    return _UnIntrospectablePrepareASR()


def test_check_entrypoints_prepare_uninspectable_signature_is_tolerated() -> None:
    # A prepare() whose signature cannot be read raises no prepare error: the
    # guard returns without over-reporting (it is not the dangerous async case).
    report = check_entrypoints(registry=_registry("unintrospectable_prepare_factory"))
    assert not any("prepare()" in i.message for i in report.issues), [
        i.message for i in report.issues
    ]
