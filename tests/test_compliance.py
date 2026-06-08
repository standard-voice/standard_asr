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


def test_check_event_sequence_empty_is_vacuously_ok() -> None:
    report = check_event_sequence([])
    assert report.passed is True


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
        self, *, gated_params: RuntimeParams, audio_format: Any = None, audio: Any = None
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
    """Supports every probed standard param in streaming -> nothing to gate."""

    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(
            word_timestamps=WordTimestampsCap(supported=True, granularities=["word", "segment"]),
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
