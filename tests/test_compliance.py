# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compliance helpers (entrypoint checks + sync-bridge driver)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from importlib.metadata import EntryPoint
from typing import Any, ClassVar, Literal

import pytest

from standard_asr import BaseConfig, BaseProperties, EngineBase, PreparedAudio, TranscriptionResult
from standard_asr import compliance as compliance_module
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
    WordTimestampsCap,
)
from standard_asr.compliance import (
    check_entrypoints,
    check_event_sequence,
    check_streaming_param_gating,
    check_sync_bridge,
)
from standard_asr.discovery import ModelRegistry, discover_models
from standard_asr.exceptions import UnsupportedFeatureError
from standard_asr.runtime_params import (
    ProviderParams,
    RuntimeParams,
)
from standard_asr.streaming import TranscriptionEvent, TranscriptionSession


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
    accepted_sample_rates: list[int] | Literal["any"] = [16000]
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
# check_streaming_param_gating (RUNT-3)
# --------------------------------------------------------------------------- #
class _StreamProps(BaseProperties):
    engine_id: str = "streamer"
    model_name: str = "demo"
    protocol_version: str = "0.2.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | Literal["any"] = [16000]
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
        self, *, audio_format: Any = None, params: Any = None, audio: Any = None
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
            self, *, audio_format: Any = None, params: Any = None, audio: Any = None
        ) -> TranscriptionSession:
            raise UnsupportedFeatureError("I refuse the param even in best_effort.")

    report = check_streaming_param_gating(_RaisingBestEffortEngine(strict=False))
    assert report.passed is False
    assert any("MUST drop it" in i.message for i in report.issues)


def test_streaming_gating_engine_crash_is_reported_not_raised() -> None:
    # R3-COMPLIANCE-02: a non-UnsupportedFeatureError exception (an engine bug)
    # MUST surface as a compliance error, never crash the whole compliance run.
    class _CrashingEngine(_GatingStreamEngine):
        def start_transcription(
            self, *, audio_format: Any = None, params: Any = None, audio: Any = None
        ) -> TranscriptionSession:
            raise RuntimeError("engine exploded")

    report = check_streaming_param_gating(_CrashingEngine(strict=True))
    assert report.passed is False
    assert any(
        "RuntimeError" in i.message and "UnsupportedFeatureError" in i.message
        for i in report.issues
    )


# --------------------------------------------------------------------------- #
# Sub-constraint gating fallback (R3-COMPLIANCE-01): every probed feature is
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
        self, *, audio_format: Any = None, params: Any = None, audio: Any = None
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
# FV-4 -- the sub-constraint probe is bounded against extreme declarations
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
    # FV-4: a 10^9-token budget must not make the probe materialize a
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
