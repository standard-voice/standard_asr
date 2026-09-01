# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""CLI coverage for Standard ASR entrypoint tooling."""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections.abc import AsyncIterator, Callable
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import pytest

from standard_asr import (
    RuntimeParams,
    TranscriptionResult,
)
from standard_asr.audio.format import AudioFormat
from standard_asr.compliance import (
    DEFAULT_SYNC_BRIDGE_TIMEOUT,
    ComplianceIssue,
    ComplianceReport,
)
from standard_asr.contract.artifacts import (
    ARTIFACT_ACTION_ACCEPT_TERMS,
    ARTIFACT_BLOCKER_ACTION_REQUIRED,
    ARTIFACT_BLOCKER_DOWNLOADS_DISABLED,
    ARTIFACT_BLOCKER_UNSUPPORTED,
    ARTIFACT_MISSING,
    ARTIFACT_PROGRESS_TRANSFERRING,
    ARTIFACT_PROGRESS_UNIT_BYTES,
    ARTIFACT_READY,
    ARTIFACT_UNKNOWN,
    ArtifactAction,
    ArtifactContext,
    ArtifactProgress,
    ArtifactProgressCallback,
    ArtifactReport,
    ArtifactRequirement,
)
from standard_asr.contract.capabilities import (
    DeclaredCapabilities,
    FlagCap,
    StreamingCapabilities,
)
from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactProgressCallbackError,
    ArtifactStatusError,
    ArtifactUnavailableError,
    AudioProcessingError,
    ConfigError,
    ConfigurationRequiredError,
    EntrypointValidationError,
    FactoryLoadError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from standard_asr.contract.metadata import ArtifactDeclaration, DeclaredEngineMetadata
from standard_asr.engine import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    InputKind,
    PreparedAudio,
    SampleRateRange,
)
from standard_asr.plugins.discovery import ModelRegistry, discover_models
from standard_asr.runtime.interface import StandardASR
from standard_asr.runtime.streaming import TranscriptionEvent, TranscriptionSession
from standard_asr.toolchain import cli


def _demo_registry() -> ModelRegistry:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    return discover_models(eps=eps, strict=True)


def _patch_discover(monkeypatch: pytest.MonkeyPatch, registry: ModelRegistry) -> None:
    """Patch ``cli.discover_models`` to return a fixed registry (typed helper)."""

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)


def _patch_check_entrypoints(monkeypatch: pytest.MonkeyPatch, report: ComplianceReport) -> None:
    """Patch ``cli.check_entrypoints`` to return a fixed report (typed helper)."""

    def _check_entrypoints(**_: object) -> ComplianceReport:
        return report

    monkeypatch.setattr(cli, "check_entrypoints", _check_entrypoints)


class _ArtifactCliProperties(BaseProperties):
    """Protocol 1.1 properties for artifact CLI test doubles."""

    engine_id: str = "artifact-cli"
    model_name: str = "demo"
    protocol_version: str = "1.1.0"
    accepted_input: set[InputKind] = {InputKind.ENCODED_FILE}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    selectable_languages: list[str] = []


class _ArtifactCliLegacyProperties(_ArtifactCliProperties):
    """Protocol 1.0 properties used to verify compatibility errors."""

    protocol_version: str = "1.0.0"


_ARTIFACT_CLI_METADATA = DeclaredEngineMetadata(
    artifacts=ArtifactDeclaration(
        applicable=True,
        supports_explicit_acquisition=True,
        may_acquire_during_inference=True,
    )
)


class _ArtifactCliEngine:
    """Controllable structural engine for status, pull, and preflight tests."""

    properties: ClassVar[BaseProperties] = _ArtifactCliProperties()
    declared_metadata: ClassVar[DeclaredEngineMetadata] = _ARTIFACT_CLI_METADATA

    def __init__(
        self,
        report: ArtifactReport,
        *,
        status_error: BaseException | None = None,
        acquisition_error: BaseException | None = None,
    ) -> None:
        self.report = report
        self.status_error = status_error
        self.acquisition_error = acquisition_error
        self.status_contexts: list[ArtifactContext | None] = []
        self.refresh_values: list[bool] = []
        self.transcribe_calls = 0

    def artifact_status(self, context: ArtifactContext | None = None) -> ArtifactReport:
        self.status_contexts.append(context)
        if self.status_error is not None:
            raise self.status_error
        return self.report

    def acquire_artifacts(
        self,
        context: ArtifactContext | None = None,
        *,
        refresh: bool = False,
        progress: ArtifactProgressCallback | None = None,
    ) -> ArtifactReport:
        del context
        self.refresh_values.append(refresh)
        if self.acquisition_error is not None:
            raise self.acquisition_error
        if progress is not None:
            progress(
                ArtifactProgress(
                    phase=ARTIFACT_PROGRESS_TRANSFERRING,
                    artifact_id="weights",
                    completed_units=5,
                    total_units=10,
                    unit=ARTIFACT_PROGRESS_UNIT_BYTES,
                )
            )
        return self.report

    def transcribe(
        self,
        audio: object,
        params: RuntimeParams | None = None,
    ) -> TranscriptionResult:
        del audio, params
        self.transcribe_calls += 1
        return TranscriptionResult(text="artifact transcript")


class _ArtifactCliLegacyEngine(_ArtifactCliEngine):
    """Engine that predates the artifact-lifecycle protocol surface."""

    properties: ClassVar[BaseProperties] = _ArtifactCliLegacyProperties()


class _ArtifactCliInvalidMetadataEngine(_ArtifactCliEngine):
    """Protocol 1.1 engine with a malformed metadata class declaration."""

    declared_metadata: ClassVar[Any] = {"artifacts": {}}


class _ArtifactCliMissingMetadataEngine(_ArtifactCliEngine):
    """Protocol 1.1 engine omitting its required metadata declaration."""

    declared_metadata: ClassVar[Any] = None


def _artifact_cli_factory() -> _ArtifactCliEngine:  # pyright: ignore[reportUnusedFunction]
    raise AssertionError("show must not instantiate the selected engine")


def _artifact_cli_legacy_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _ArtifactCliLegacyEngine
):
    raise AssertionError("show must not instantiate the selected engine")


def _artifact_cli_invalid_metadata_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _ArtifactCliInvalidMetadataEngine
):
    raise AssertionError("show must not instantiate the selected engine")


def _artifact_cli_missing_metadata_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _ArtifactCliMissingMetadataEngine
):
    raise AssertionError("show must not instantiate the selected engine")


def _patch_artifact_engine(
    monkeypatch: pytest.MonkeyPatch,
    engine: object,
) -> dict[str, object]:
    """Patch discovery with a recording single-engine registry."""

    seen: dict[str, object] = {}

    class _Registry:
        def create(self, name: str, **config: object) -> object:
            seen["name"] = name
            seen["config"] = config
            return engine

    registry = cast("ModelRegistry", _Registry())

    def _discover_models(**kwargs: object) -> ModelRegistry:
        seen["discovery"] = kwargs
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)
    return seen


def _artifact_requirement(
    *,
    artifact_id: str = "weights",
    label: str = "Recognizer weights",
    state: str = ARTIFACT_READY,
    required: bool = True,
    can_acquire_now: bool = False,
    may_acquire_during_inference: bool = False,
    mutable: bool = False,
    blocker: str | None = None,
    actions: tuple[ArtifactAction, ...] = (),
    location: Path | None = None,
    size_bytes: int | None = None,
    expected_size_bytes: int | None = None,
    artifact_version: str | None = None,
) -> ArtifactRequirement:
    """Build one internally coherent artifact CLI fixture."""

    return ArtifactRequirement(
        artifact_id=artifact_id,
        label=label,
        state=state,
        required_for_inference=required,
        can_acquire_now=can_acquire_now,
        may_acquire_during_inference=may_acquire_during_inference,
        source_is_mutable=mutable,
        acquisition_blocker=blocker,
        required_actions=actions,
        location=location,
        size_bytes=size_bytes,
        expected_size_bytes=expected_size_bytes,
        artifact_version=artifact_version,
    )


def _artifact_report(*requirements: ArtifactRequirement) -> ArtifactReport:
    """Build a batch artifact report with canonical aggregate readiness."""

    return ArtifactReport.from_requirements(
        mode="batch",
        applicable=True,
        requirements=requirements,
    )


def test_cli_models_list(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = _demo_registry()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["list"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "alpha/first" in output
    assert "engine=alpha" in output


def test_cli_models_list_empty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = ModelRegistry({})

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["list"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "No Standard ASR models were discovered." in output


def test_cli_models_show(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = _demo_registry()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["show", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Engine ID" in output
    assert "alpha/first" in output
    # models show MUST surface DeclaredCapabilities (no instantiation).
    assert "Capabilities:" in output
    assert "runtime_override" in output
    # It also surfaces the init-config schema; this engine declares none, so the
    # omission is stated explicitly rather than left silent.
    assert "Config schema: <none" in output


def test_cli_show_renders_canonical_declared_metadata_without_instantiation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = discover_models(
        eps=[
            EntryPoint(
                name="artifact-cli/demo",
                value="tests.test_cli:_artifact_cli_factory",
                group="standard_asr.models",
            )
        ],
        strict=True,
    )
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["show", "artifact-cli/demo"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Declared metadata:" in output
    assert '"artifacts": {' in output
    assert '"applicable": true' in output
    assert '"supports_explicit_acquisition": true' in output
    assert '"may_acquire_during_inference": true' in output


def test_cli_show_marks_legacy_declared_metadata_unsupported(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = discover_models(
        eps=[
            EntryPoint(
                name="artifact-cli/demo",
                value="tests.test_cli:_artifact_cli_legacy_factory",
                group="standard_asr.models",
            )
        ],
        strict=True,
    )
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["show", "artifact-cli/demo"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Declared metadata: <unsupported:" in output
    assert "requires protocol 1.1.0" in output
    assert '"applicable"' not in output


def test_cli_show_fault_bounds_invalid_declared_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = discover_models(
        eps=[
            EntryPoint(
                name="artifact-cli/demo",
                value="tests.test_cli:_artifact_cli_invalid_metadata_factory",
                group="standard_asr.models",
            )
        ],
        strict=True,
    )
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["show", "artifact-cli/demo"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Declared metadata: <invalid:" in output
    assert "DeclaredEngineMetadata model" in output
    assert "Config schema:" in output


def test_cli_show_fault_bounds_missing_protocol_1_1_declared_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = discover_models(
        eps=[
            EntryPoint(
                name="artifact-cli/demo",
                value="tests.test_cli:_artifact_cli_missing_metadata_factory",
                group="standard_asr.models",
            )
        ],
        strict=True,
    )
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["show", "artifact-cli/demo"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Declared metadata: <invalid:" in output
    assert "protocol 1.1 requires declared_metadata.artifacts" in output


def test_cli_declared_metadata_fault_boundary_reraises_factory_load_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _UnresolvableSpec:
        def engine_class(self) -> type[object]:
            raise FactoryLoadError("Selected plugin class could not be resolved.")

    with pytest.raises(FactoryLoadError):
        cli._print_declared_metadata(  # pyright: ignore[reportPrivateUsage]
            _UnresolvableSpec()
        )
    output = capsys.readouterr().out

    assert "Declared metadata: <unavailable:" in output
    assert "Selected plugin class could not be resolved." in output


def test_cli_declared_metadata_rejects_noncallable_canonical_projection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Spec:
        def engine_class(self) -> type[_ArtifactCliEngine]:
            return _ArtifactCliEngine

    monkeypatch.setattr(DeclaredEngineMetadata, "canonical_json", None)
    cli._print_declared_metadata(_Spec())  # pyright: ignore[reportPrivateUsage]

    assert "Declared metadata: <invalid: canonical_json is not callable>" in capsys.readouterr().out


def test_cli_models_show_unresolvable_class(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_unannotated_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["show", "alpha/first"])
    output = capsys.readouterr().out

    # Exit 1, not 0: the caller's key was fine and nothing about the engine
    # could be read, so reporting success told a script the model was usable.
    # The metadata and the sanitized unavailable line are still printed --
    # `show` gives the operator everything it could learn AND an honest code.
    assert exit_code == 1
    assert "Capabilities: <unavailable" in output
    assert "Engine ID" in output
    # No config-schema section: an unloadable class is reported ONCE, at the
    # capabilities line, and `show` then exits 1. The config schema needs that
    # same engine class, so a second unavailable line would only restate the
    # one fault the operator has already been given.
    assert "Config schema" not in output


def test_cli_models_show_config_schema(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An engine that declares a config_type has its init-config JSON Schema
    # rendered (read from the class -- _ConfigTypeASR.__init__ raises, so this
    # proves no instantiation happens).
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_ConfigTypeASR",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["show", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Config schema:" in output
    # Fields of the declared config model appear in the rendered schema.
    assert '"default_language"' in output
    assert '"strict"' in output


def test_cli_models_show_config_schema_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The engine class itself loads, so `show` does NOT abort at the
    # capabilities line -- but its config_type is un-schematizable
    # (arbitrary_types_allowed + an opaque handle field). The config-schema
    # section degrades to an explicit unavailable line and the rest of the
    # metadata still renders, rather than crashing mid-print.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_UnschematizableConfigTypeASR",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["show", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Config schema: <unavailable" in output
    assert "JSON Schema cannot be generated" in output
    # The rest of the metadata survived the degraded section.
    assert "Engine ID" in output


class _NoCapsClass:
    """An engine class that declares no capabilities (declared_capabilities=None)."""

    declared_capabilities = None

    def transcribe(self, audio: object, options: object = None) -> object:
        # A real engine must expose the defining 'transcribe' method even when
        # it declares no capabilities.
        return None


def _no_caps_factory() -> _NoCapsClass:  # pyright: ignore[reportUnusedFunction]
    return _NoCapsClass()


# --- Fixtures for (mis-typed caps) and (compliance run) ---


class _DictCapsASR:
    """Engine mis-declaring declared_capabilities as a dict (declaration bug)."""

    declared_capabilities: ClassVar[dict[str, dict[str, object]]] = {"batch": {}}

    def transcribe(self, audio: object, options: object = None) -> object:
        return None


def _dict_caps_factory() -> _DictCapsASR:  # pyright: ignore[reportUnusedFunction]
    return _DictCapsASR()


class _StreamConfig(BaseConfig[Literal["stream"]]):
    engine: Literal["stream"] = "stream"


class _StreamOkProps(BaseProperties):
    engine_id: str = "stream"
    model_name: str = "ok"  # model_id == ``stream/ok``
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    selectable_languages: list[str] = []  # no language axis -> no default needed
    wire_encodings: list[str] | None = ["pcm_s16le"]


class _StreamBadProps(_StreamOkProps):
    model_name: str = "bad"  # model_id == 'stream/bad'


_STREAM_CAPS = DeclaredCapabilities(
    streaming=StreamingCapabilities(),
    streaming_input=FlagCap(supported=True),
    streaming_output=FlagCap(supported=True),
)


class _GatingSession(TranscriptionSession):
    """Ends immediately (the base producer appends the terminal ``done``)."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        return
        yield  # pragma: no cover - makes this an async generator


class _GatingStreamEngine(EngineBase):
    """Streaming engine that relies on the base template's gating (compliant)."""

    properties: ClassVar[BaseProperties] = _StreamOkProps()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _STREAM_CAPS
    config_type: ClassVar[type[BaseConfig[str]] | None] = _StreamConfig

    def __init__(self) -> None:
        self.config = _StreamConfig(engine="stream")

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        return TranscriptionResult(text="")

    def _start_transcription(
        self,
        *,
        gated_params: RuntimeParams,
        audio_format: object = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        return _GatingSession()


class _UngatedStreamEngine(_GatingStreamEngine):
    """Non-compliant: overrides the PUBLIC start_transcription, bypassing gating."""

    properties: ClassVar[BaseProperties] = _StreamBadProps()

    def start_transcription(
        self,
        *,
        audio_format: object = None,
        params: object = None,
        audio: object = None,
        deadlines: object = None,
    ) -> TranscriptionSession:
        # Forgot to gate: returns a session for ANY params, no gate_params call.
        return _GatingSession()


class _BatchConfig(BaseConfig[Literal["batch"]]):
    engine: Literal["batch"] = "batch"


class _BatchOnlyProps(BaseProperties):
    engine_id: str = "batch"
    model_name: str = "only"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    selectable_languages: list[str] = []


class _BatchOnlyEngine(EngineBase):
    """Batch-only compliant engine (no streaming capabilities)."""

    properties: ClassVar[BaseProperties] = _BatchOnlyProps()
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities()
    config_type: ClassVar[type[BaseConfig[str]] | None] = _BatchConfig

    def __init__(self) -> None:
        self.config = _BatchConfig(engine="batch")

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        return TranscriptionResult(text="")


def _batch_only_factory() -> _BatchOnlyEngine:  # pyright: ignore[reportUnusedFunction]
    return _BatchOnlyEngine()


class _BatchOnlyBadWireEngine(_BatchOnlyEngine):
    """Batch-only engine whose wire recommendation its own Properties reject."""

    def recommended_wire_format(self) -> AudioFormat | None:
        """Recommend a rate outside the engine's accepted set.

        Returns:
            A self-inconsistent format (the F4 counterexample).
        """
        return AudioFormat(encoding="pcm_s16le", sample_rate=4321)


def _batch_only_bad_wire_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _BatchOnlyBadWireEngine
):
    return _BatchOnlyBadWireEngine()


def _gating_stream_factory() -> _GatingStreamEngine:  # pyright: ignore[reportUnusedFunction]
    return _GatingStreamEngine()


class _OtherConfig(BaseConfig[Literal["other"]]):
    engine: Literal["other"] = "other"


class _OtherProps(_StreamOkProps):
    engine_id: str = "other"
    model_name: str = "model"  # model_id == 'other/model'


#: Construction ledger for the co-installed-plugin scoping test: a named
#: `compliance run` must never construct (and thereby probe) this engine.
_unnamed_probe_constructions: list[str] = []


class _OtherRecordingEngine(_GatingStreamEngine):
    """A compliant co-installed engine that records every construction."""

    properties: ClassVar[BaseProperties] = _OtherProps()
    config_type: ClassVar[type[BaseConfig[str]] | None] = _OtherConfig

    def __init__(self) -> None:
        _unnamed_probe_constructions.append("other/model")
        self.config = _OtherConfig(engine="other")


def _other_recording_factory() -> _OtherRecordingEngine:  # pyright: ignore[reportUnusedFunction]
    return _OtherRecordingEngine()


class _ArbitraryFactoryFault(Exception):
    """A plugin-authored fault type the registry does not wrap."""


def _runtime_error_factory() -> _GatingStreamEngine:  # pyright: ignore[reportUnusedFunction]
    raise RuntimeError("SDK initialization failed")


def _type_error_factory() -> _GatingStreamEngine:  # pyright: ignore[reportUnusedFunction]
    raise TypeError("bad SDK signature")


def _os_error_factory() -> _GatingStreamEngine:  # pyright: ignore[reportUnusedFunction]
    raise OSError("model directory unreadable")


def _authored_error_factory() -> _GatingStreamEngine:  # pyright: ignore[reportUnusedFunction]
    raise _ArbitraryFactoryFault("plugin said no")


def _keyboard_interrupt_factory() -> _GatingStreamEngine:  # pyright: ignore[reportUnusedFunction]
    raise KeyboardInterrupt


def _ungated_stream_factory() -> _UngatedStreamEngine:  # pyright: ignore[reportUnusedFunction]
    return _UngatedStreamEngine()


class _StreamOutConfig(BaseConfig[Literal["streamout"]]):
    engine: Literal["streamout"] = "streamout"


class _StreamOutOnlyProps(_StreamOkProps):
    engine_id: str = "streamout"
    model_name: str = "only"


class _OutputOnlyStreamEngine(_GatingStreamEngine):
    """Output-only streaming engine: streaming_output WITHOUT streaming_input."""

    properties: ClassVar[BaseProperties] = _StreamOutOnlyProps()
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(),
        streaming_output=FlagCap(supported=True),
    )
    config_type: ClassVar[type[BaseConfig[str]] | None] = _StreamOutConfig

    def __init__(self) -> None:
        self.config = _StreamOutConfig(engine="streamout")


def _output_only_stream_factory() -> _OutputOnlyStreamEngine:  # pyright: ignore[reportUnusedFunction]
    return _OutputOnlyStreamEngine()


class _BrokenWireConfig(BaseConfig[Literal["brokenwire"]]):
    engine: Literal["brokenwire"] = "brokenwire"


class _BrokenWireProps(_StreamOkProps):
    engine_id: str = "brokenwire"
    model_name: str = "engine"


class _BrokenWireEngine(_GatingStreamEngine):
    """Streaming engine whose ``recommended_wire_format()`` raises.

    A structural implementation is free to override the derivation; a buggy
    one takes the whole CLI run down with it unless the runner guards.
    """

    properties: ClassVar[BaseProperties] = _BrokenWireProps()
    config_type: ClassVar[type[BaseConfig[str]] | None] = _BrokenWireConfig

    def __init__(self) -> None:
        self.config = _BrokenWireConfig(engine="brokenwire")

    def recommended_wire_format(self) -> AudioFormat | None:
        raise RuntimeError("wire format derivation exploded")


def _broken_wire_factory() -> _BrokenWireEngine:  # pyright: ignore[reportUnusedFunction]
    return _BrokenWireEngine()


def test_cli_models_show_no_capabilities(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_cli:_no_caps_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)

    def _discover(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover)

    exit_code = cli.main(["show", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Capabilities: <none declared>" in output


def test_cli_doctor(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # The doctor command prints the report and returns 0 with no conflicts.
    from standard_asr.toolchain import doctor as doctor_module

    def _entry_points(*, group: str) -> list[object]:
        return []

    monkeypatch.setattr(doctor_module, "entry_points", _entry_points)
    exit_code = cli.main(["doctor"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Standard ASR" in output or "plugins" in output.lower()


def test_cli_doctor_conflict_returns_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A numpy 1.x vs 2.x conflict makes the doctor command exit non-zero.
    from dataclasses import dataclass

    from standard_asr.toolchain import doctor as doctor_module

    @dataclass
    class _Dist:
        name: str
        requires: list[str] | None

    @dataclass
    class _EP:
        name: str
        dist: _Dist | None

    def _entry_points(*, group: str) -> list[_EP]:
        return [
            _EP("old/a", _Dist("std-a", ["numpy<2"])),
            _EP("new/b", _Dist("std-b", ["numpy>=2.1"])),
        ]

    monkeypatch.setattr(doctor_module, "entry_points", _entry_points)
    exit_code = cli.main(["doctor"])
    capsys.readouterr()
    assert exit_code == 1


def test_cli_doctor_packaging_unavailable_with_plugins_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Plugins installed but `packaging` missing: doctor cannot prove the
    # environment conflict-free, so the headline is non-clean and "cannot prove
    # clean" is operationally a failure -> exit 1.
    from dataclasses import dataclass

    from standard_asr.toolchain import doctor as doctor_module

    @dataclass
    class _Dist:
        name: str
        requires: list[str] | None

    @dataclass
    class _EP:
        name: str
        dist: _Dist | None

    def _entry_points(*, group: str) -> list[_EP]:
        return [_EP("a/x", _Dist("std-a", ["numpy>=1.26"]))]

    monkeypatch.setattr(doctor_module, "entry_points", _entry_points)
    monkeypatch.setattr(doctor_module, "packaging_available", lambda: False)
    exit_code = cli.main(["doctor"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "unavailable" in output


def test_cli_doctor_packaging_unavailable_no_plugins_exits_0(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # With no plugins there is nothing to analyze: `packaging` absence stays a
    # non-issue and the clean exit 0 is preserved.
    from standard_asr.toolchain import doctor as doctor_module

    def _entry_points(*, group: str) -> list[object]:
        return []

    monkeypatch.setattr(doctor_module, "entry_points", _entry_points)
    monkeypatch.setattr(doctor_module, "packaging_available", lambda: False)
    exit_code = cli.main(["doctor"])
    capsys.readouterr()

    assert exit_code == 0


def test_cli_models_cache(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "resolve_cache_dir", lambda: tmp_path)

    exit_code = cli.main(["cache"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert str(tmp_path) in output


def test_cli_artifact_status_human_uses_init_config_and_full_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    action = ArtifactAction.model_validate(
        {
            "kind": ARTIFACT_ACTION_ACCEPT_TERMS,
            "message": "Accept the model license.",
            "url": "https://example.test/license",
        }
    )
    engine = _ArtifactCliEngine(
        _artifact_report(
            _artifact_requirement(
                location=tmp_path,
                size_bytes=10,
                expected_size_bytes=20,
                artifact_version="commit-1",
            ),
            _artifact_requirement(
                artifact_id="aligner",
                label="Forced aligner",
                state=ARTIFACT_MISSING,
                required=False,
                blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
                actions=(action,),
            ),
        )
    )
    seen = _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(
        [
            "status",
            "artifact-cli/demo",
            "--strict-discovery",
            "--config",
            '{"device": "cpu", "compute_type": "float32"}',
            "--set",
            "compute_type=int8",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Artifact readiness: ready" in captured.out
    assert "Recognizer weights (weights): ready, required" in captured.out
    assert "Forced aligner (aligner): missing, optional" in captured.out
    assert "Accept the model license." in captured.out
    assert "https://example.test/license" in captured.out
    assert f"location: {tmp_path}" in captured.out
    assert "version: commit-1" in captured.out
    assert "size_bytes: 10" in captured.out
    assert "expected_size_bytes: 20" in captured.out
    assert captured.err == ""
    assert engine.status_contexts == [None]
    assert seen["name"] == "artifact-cli/demo"
    assert seen["discovery"] == {"strict": True}
    assert seen["config"] == {"device": "cpu", "compute_type": "int8"}


@pytest.mark.parametrize(
    ("require_ready", "expected_exit"),
    [(False, 0), (True, 1)],
)
def test_cli_artifact_status_json_and_require_ready(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    require_ready: bool,
    expected_exit: int,
) -> None:
    engine = _ArtifactCliEngine(
        _artifact_report(
            _artifact_requirement(
                state=ARTIFACT_MISSING,
                can_acquire_now=True,
            )
        )
    )
    _patch_artifact_engine(monkeypatch, engine)
    argv = ["status", "artifact-cli/demo", "--json"]
    if require_ready:
        argv.append("--require-ready")

    exit_code = cli.main(argv)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == expected_exit
    assert payload["readiness"] == "unavailable"
    assert payload["requirements"][0]["artifact_id"] == "weights"
    assert captured.err == ""


def test_cli_artifact_status_require_ready_accepts_not_applicable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _ArtifactCliEngine(ArtifactReport.from_requirements(mode="batch", applicable=False))
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(["status", "artifact-cli/demo", "--require-ready"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Artifact readiness: not_applicable" in captured.out
    assert "Applicable : no" in captured.out
    assert "Requirements: <none>" in captured.out


@pytest.mark.parametrize(
    ("command", "strict"),
    [("status", False), ("status", True), ("pull", False), ("pull", True)],
)
def test_cli_artifact_commands_thread_strict_discovery(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    strict: bool,
) -> None:
    engine = _ArtifactCliEngine(_artifact_report(_artifact_requirement()))
    seen = _patch_artifact_engine(monkeypatch, engine)
    argv = [command, "artifact-cli/demo"]
    if strict:
        argv.append("--strict-discovery")

    assert cli.main(argv) == 0
    capsys.readouterr()
    assert seen["discovery"] == {"strict": strict}


def test_cli_artifact_status_rejects_missing_operation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _ArtifactCliEngine(_artifact_report(_artifact_requirement()))
    setattr(engine, "artifact_status", None)
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(["status", "artifact-cli/demo"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "must expose callable artifact_status()" in captured.err


@pytest.mark.parametrize("command", [["status"], ["pull"], ["pull", "--refresh"]])
def test_cli_artifact_commands_guard_protocol_1_0_before_operation_lookup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: list[str],
) -> None:
    engine = _ArtifactCliLegacyEngine(_artifact_report(_artifact_requirement()))
    _patch_artifact_engine(monkeypatch, engine)
    argv = [command[0], "artifact-cli/demo", *command[1:]]

    exit_code = cli.main(argv)
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "requires protocol 1.1.0" in captured.err
    assert engine.status_contexts == []
    assert engine.refresh_values == []


def test_cli_artifact_pull_refresh_reports_progress_and_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _ArtifactCliEngine(_artifact_report(_artifact_requirement()))
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(["pull", "artifact-cli/demo", "--refresh", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["readiness"] == "ready"
    assert engine.refresh_values == [True]
    assert "Artifact progress for 'weights': transferring 5/10 bytes." in captured.err
    assert "[OK] Artifact acquisition complete." in captured.err


def test_cli_artifact_progress_renders_partial_and_unknown_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for progress in (
        ArtifactProgress(phase="resolving"),
        ArtifactProgress(
            phase="transferring",
            total_units=20,
            unit=ARTIFACT_PROGRESS_UNIT_BYTES,
        ),
        ArtifactProgress(
            phase="transferring",
            completed_units=5,
            unit=ARTIFACT_PROGRESS_UNIT_BYTES,
        ),
    ):
        cli._render_artifact_progress(progress)  # pyright: ignore[reportPrivateUsage]
    error = capsys.readouterr().err

    assert "Artifact progress: resolving." in error
    assert "Artifact progress: transferring total=20 bytes." in error
    assert "Artifact progress: transferring 5 bytes." in error


def test_cli_artifact_pull_optional_blocker_warns_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    action = ArtifactAction(
        kind=ARTIFACT_ACTION_ACCEPT_TERMS,
        message="Accept optional aligner terms.",
    )
    engine = _ArtifactCliEngine(
        _artifact_report(
            _artifact_requirement(),
            _artifact_requirement(
                artifact_id="aligner",
                label="Optional aligner",
                state=ARTIFACT_MISSING,
                required=False,
                blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
                actions=(action,),
            ),
        )
    )
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(["pull", "artifact-cli/demo"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Optional artifact 'Optional aligner' remains missing" in captured.err
    assert "Accept optional aligner terms." in captured.err
    assert "Artifact acquisition complete" in captured.err


def test_cli_artifact_pull_optional_warning_never_contradicts_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The optional warning states that the REQUIRED artifacts are ready, so it
    # belongs after the required verdict. Emitted before it, a report carrying
    # one non-ready required and one non-ready optional requirement printed the
    # readiness claim and the failure line together -- two stderr lines
    # contradicting each other on the surface an operator reads to know what to
    # fix.
    engine = _ArtifactCliEngine(
        _artifact_report(
            _artifact_requirement(
                state=ARTIFACT_MISSING,
                blocker=ARTIFACT_BLOCKER_UNSUPPORTED,
            ),
            _artifact_requirement(
                artifact_id="aligner",
                label="Optional aligner",
                state=ARTIFACT_MISSING,
                required=False,
                blocker=ARTIFACT_BLOCKER_UNSUPPORTED,
            ),
        )
    )
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(["pull", "artifact-cli/demo"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Required inference artifacts remain unavailable" in captured.err
    assert "required inference artifacts are ready" not in captured.err
    assert "Optional aligner" not in captured.err


@pytest.mark.parametrize("command", ["status", "pull"])
def test_cli_artifact_report_diagnostics_are_not_duplicated_in_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    # `--json` carries the report's diagnostics on stdout, so rendering them to
    # stderr too gave one run two representations of the same diagnostic. The
    # text view keeps the stderr rendering (that is the view that would
    # otherwise drop them) -- the rule `transcribe` already follows.
    from standard_asr.contract.results import Diagnostic

    diagnostic = Diagnostic(code="opaque_cache", message="native cache index is unreadable")
    report = ArtifactReport.from_requirements(
        mode="batch",
        applicable=True,
        requirements=(_artifact_requirement(),),
        diagnostics=(diagnostic,),
    )
    engine = _ArtifactCliEngine(report)
    _patch_artifact_engine(monkeypatch, engine)

    assert cli.main([command, "artifact-cli/demo", "--json"]) == 0
    as_json = capsys.readouterr()
    _patch_artifact_engine(monkeypatch, _ArtifactCliEngine(report))
    assert cli.main([command, "artifact-cli/demo"]) == 0
    as_text = capsys.readouterr()

    payload = json.loads(as_json.out)
    assert payload["diagnostics"][0]["code"] == "opaque_cache"
    assert "diagnostic [opaque_cache]" not in as_json.err
    assert "diagnostic [opaque_cache]" in as_text.err


@pytest.mark.parametrize(
    ("blocker", "expected_exit"),
    [
        (ARTIFACT_BLOCKER_ACTION_REQUIRED, 2),
        (ARTIFACT_BLOCKER_DOWNLOADS_DISABLED, 2),
        (ARTIFACT_BLOCKER_UNSUPPORTED, 1),
        ("x_vendor_policy", 1),
    ],
)
def test_cli_artifact_pull_aggregates_required_returned_blockers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    blocker: str,
    expected_exit: int,
) -> None:
    action = ArtifactAction(
        kind=ARTIFACT_ACTION_ACCEPT_TERMS,
        message="Accept terms before acquisition.",
    )
    actions = (action,) if blocker == ARTIFACT_BLOCKER_ACTION_REQUIRED else ()
    engine = _ArtifactCliEngine(
        _artifact_report(
            _artifact_requirement(
                state=ARTIFACT_MISSING,
                blocker=blocker,
                actions=actions,
            )
        )
    )
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(["pull", "artifact-cli/demo"])
    captured = capsys.readouterr()

    assert exit_code == expected_exit
    assert "Required inference artifacts remain unavailable" in captured.err
    if actions:
        assert "Accept terms before acquisition." in captured.err


@pytest.mark.parametrize(
    ("reason", "expected_exit"),
    [
        ("downloads_disabled", 2),
        ("action_required", 2),
        ("unsupported", 1),
        ("busy", 1),
        ("failed", 1),
    ],
)
def test_cli_artifact_acquisition_error_exit_mapping(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reason: Literal["downloads_disabled", "action_required", "unsupported", "busy", "failed"],
    expected_exit: int,
) -> None:
    action = ArtifactAction.model_validate(
        {
            "kind": ARTIFACT_ACTION_ACCEPT_TERMS,
            "message": "Complete the external action.",
            "url": "https://example.test/action",
        }
    )
    report = _artifact_report(
        _artifact_requirement(
            state=ARTIFACT_MISSING,
            blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
            actions=(action,),
        ),
        _artifact_requirement(
            artifact_id="aligner",
            label="Optional aligner",
            state=ARTIFACT_MISSING,
            required=False,
            blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
            actions=(action,),
        ),
    )
    error = ArtifactAcquisitionError(
        "Artifact acquisition did not complete.",
        reason=reason,
        report=None if reason == "busy" else report,
        required_actions=(action,),
    )
    engine = _ArtifactCliEngine(report, acquisition_error=error)
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(["pull", "artifact-cli/demo"])
    captured = capsys.readouterr()

    assert exit_code == expected_exit
    assert "Artifact acquisition did not complete." in captured.err
    assert captured.err.count("Complete the external action.") == 1
    assert captured.err.count("https://example.test/action") == 1


@pytest.mark.parametrize(
    ("reason", "expected_exit"),
    [
        ("downloads_disabled", 2),
        ("action_required", 2),
        ("missing", 1),
        ("incomplete", 1),
        ("corrupt", 1),
        ("unknown", 1),
    ],
)
def test_cli_artifact_unavailable_error_exit_mapping(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reason: Literal[
        "missing",
        "incomplete",
        "corrupt",
        "unknown",
        "action_required",
        "downloads_disabled",
    ],
    expected_exit: int,
) -> None:
    report = _artifact_report(_artifact_requirement(state=ARTIFACT_MISSING, can_acquire_now=True))
    error = ArtifactUnavailableError(
        "Required artifacts are unavailable.",
        reason=reason,
        report=report,
    )
    engine = _ArtifactCliEngine(report, acquisition_error=error)
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(["pull", "artifact-cli/demo"])
    captured = capsys.readouterr()

    assert exit_code == expected_exit
    assert "Required artifacts are unavailable." in captured.err


@pytest.mark.parametrize("command", ["status", "pull"])
def test_cli_artifact_status_error_is_exit_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    report = _artifact_report(_artifact_requirement())
    error = ArtifactStatusError("Native artifact inspection failed.")
    engine = _ArtifactCliEngine(
        report,
        status_error=error if command == "status" else None,
        acquisition_error=error if command == "pull" else None,
    )
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main([command, "artifact-cli/demo"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Native artifact inspection failed." in captured.err


def test_cli_artifact_progress_callback_error_is_exit_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _artifact_report(_artifact_requirement())
    engine = _ArtifactCliEngine(
        report,
        acquisition_error=ArtifactProgressCallbackError(
            "Artifact progress observer failed.",
            report=report,
        ),
    )
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(["pull", "artifact-cli/demo"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Artifact progress observer failed." in captured.err


def test_cli_transcribe_artifact_preflight_passes_runtime_context_and_notices(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _ArtifactCliEngine(
        _artifact_report(
            _artifact_requirement(
                state=ARTIFACT_MISSING,
                can_acquire_now=True,
                may_acquire_during_inference=True,
            )
        )
    )
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(
        [
            "transcribe",
            "artifact-cli/demo",
            "audio.wav",
            "--options",
            '{"language": "en"}',
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "artifact transcript\n"
    assert "The first transcription can acquire them" in captured.err
    assert "standard-asr pull artifact-cli/demo" in captured.err
    assert engine.transcribe_calls == 1
    assert len(engine.status_contexts) == 1
    context = engine.status_contexts[0]
    assert context is not None
    assert context.params.language == "en"


def test_cli_transcribe_unknown_artifact_preflight_uses_static_may_acquire(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _ArtifactCliEngine(
        _artifact_report(
            _artifact_requirement(
                state=ARTIFACT_UNKNOWN,
                blocker=ARTIFACT_BLOCKER_UNSUPPORTED,
            )
        )
    )
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(["transcribe", "artifact-cli/demo", "audio.wav"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Required artifact readiness is unknown" in captured.err
    assert "may acquire inference artifacts" in captured.err
    assert engine.transcribe_calls == 1


def test_cli_transcribe_artifact_status_error_warns_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _ArtifactCliEngine(
        _artifact_report(_artifact_requirement()),
        status_error=ArtifactStatusError("Native status probe failed."),
    )
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(["transcribe", "artifact-cli/demo", "audio.wav"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "artifact transcript\n"
    assert "[WARN] Artifact status inspection failed" in captured.err
    assert "Native status probe failed." in captured.err
    assert engine.transcribe_calls == 1


def test_cli_transcribe_protocol_1_0_skips_advisory_preflight(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _ArtifactCliLegacyEngine(
        _artifact_report(
            _artifact_requirement(
                state=ARTIFACT_MISSING,
                can_acquire_now=True,
                may_acquire_during_inference=True,
            )
        )
    )
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(["transcribe", "artifact-cli/demo", "audio.wav"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "artifact transcript\n"
    assert captured.err == ""
    assert engine.status_contexts == []
    assert engine.transcribe_calls == 1


@pytest.mark.parametrize(
    "report",
    [
        _artifact_report(_artifact_requirement()),
        _artifact_report(
            _artifact_requirement(
                state=ARTIFACT_MISSING,
                blocker=ARTIFACT_BLOCKER_UNSUPPORTED,
            )
        ),
    ],
)
def test_cli_transcribe_artifact_preflight_emits_no_false_notice(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    report: ArtifactReport,
) -> None:
    engine = _ArtifactCliEngine(report)
    _patch_artifact_engine(monkeypatch, engine)

    exit_code = cli.main(["transcribe", "artifact-cli/demo", "audio.wav"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "artifact transcript\n"
    assert captured.err == ""
    assert engine.transcribe_calls == 1


def test_cli_transcribe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = _demo_registry()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["transcribe", "alpha/first", "dummy.wav"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "dummy" in output


@pytest.mark.parametrize(
    ("argv", "expected_strict"),
    [
        (["transcribe", "alpha/first", "dummy.wav"], False),
        (["transcribe", "alpha/first", "dummy.wav", "--strict-discovery"], True),
        (["prepare", "alpha/first"], False),
        (["prepare", "alpha/first", "--strict-discovery"], True),
        (["list"], False),
        (["list", "--strict-discovery"], True),
        (["show", "alpha/first"], False),
        (["show", "alpha/first", "--strict-discovery"], True),
    ],
)
def test_cli_strict_discovery_threads_into_discovery(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    expected_strict: bool,
) -> None:
    """Every discovery-facing subcommand is uniform: --strict-discovery makes
    discovery FAIL on an invalid entry point instead of skipping it, and its
    absence keeps the lenient default. Captured at the discovery call so the
    flag cannot be parsed-but-dropped.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
            argv: The command-line variant under test (parametrized).
            expected_strict: The strictness expected to reach discovery (parametrized).
    """
    registry = _demo_registry()
    seen: dict[str, object] = {}

    def _discover_models(**kwargs: object) -> ModelRegistry:
        seen.update(kwargs)
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(argv)
    capsys.readouterr()

    assert exit_code == 0
    assert seen["strict"] is expected_strict


@pytest.mark.parametrize(
    "argv",
    [
        ["list", "--strict"],
        ["show", "alpha/first", "--strict"],
        ["prepare", "alpha/first", "--strict"],
        ["status", "alpha/first", "--strict"],
        ["pull", "alpha/first", "--strict"],
        ["transcribe", "alpha/first", "dummy.wav", "--strict"],
        ["compliance", "entrypoints", "--strict"],
        ["compliance", "run", "--strict"],
    ],
)
def test_cli_bare_strict_flag_is_rejected(argv: list[str]) -> None:
    """A bare ``--strict`` is a usage error (exit 2), never an abbreviation.

    ``strict`` alone already names a DIFFERENT setting: the engine's
    strict/best_effort PARAMETER-gating policy, an init-config field set via
    ``--set strict=...``. The discovery flag is therefore spelled
    ``--strict-discovery``, and argparse prefix abbreviation is disabled
    (``allow_abbrev=False``) so ``--strict`` cannot silently resolve to it and
    resurrect exactly the confusion the rename exists to prevent.

        Args:
            argv: The command-line variant under test (parametrized).
    """
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(argv)
    assert excinfo.value.code == 2


def test_cli_does_not_abbreviate_long_flags() -> None:
    """allow_abbrev=False is set on the top-level parser AND every subparser: an
    abbreviation a user scripts today can turn ambiguous the day a new flag
    lands, which would be a silent behavior change.
    """
    parser = cli.build_parser()
    for argv in (
        ["list", "--strict-disc"],
        ["compliance", "run", "--include"],
        ["compliance", "run", "--bridge", "5"],
        ["serve", "--ho", "0.0.0.0"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(argv)
        assert excinfo.value.code == 2


def test_cli_transcribe_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = _demo_registry()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["transcribe", "alpha/first", "dummy.wav", "--json"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"text"' in output


def test_cli_transcribe_invalid_options(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = _demo_registry()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["transcribe", "alpha/first", "dummy.wav", "--options", "not-json"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.err != ""


def test_cli_transcribe_options_portable_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = _demo_registry()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(
        ["transcribe", "alpha/first", "dummy.wav", "--options", '{"language": "en"}']
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "dummy" in output


def test_cli_transcribe_options_provider_params_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Mirrors the server's untyped-wire rule: provider_params cannot be
    # validated from untyped JSON, so the CLI rejects the key itself loudly as
    # a usage / validation error (exit 2) instead of passing it to the engine.
    # An empty object is the regression case: the old RuntimeParams path
    # silently accepted it as a bare ProviderParams().
    registry = _demo_registry()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(
        ["transcribe", "alpha/first", "dummy.wav", "--options", '{"provider_params": {}}']
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "provider_params" in captured.err


def test_cli_models_list_entrypoint_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _discover_models(**_: object) -> ModelRegistry:
        raise EntrypointValidationError("bad entrypoint")

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["list"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "bad entrypoint" in captured.err


def test_cli_transcribe_audio_processing_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = _demo_registry()

    class _BadAudioASR:
        def transcribe(self, audio: object, params: object = None) -> None:
            raise AudioProcessingError("bad audio")

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    def _create(*_: object, **__: object) -> _BadAudioASR:
        return _BadAudioASR()

    monkeypatch.setattr(cli, "discover_models", _discover_models)
    monkeypatch.setattr(registry, "create", _create)

    exit_code = cli.main(["transcribe", "alpha/first", "dummy.wav"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "bad audio" in captured.err


def test_cli_transcribe_unsupported_feature_is_usage_exit_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A strict-mode rejection is a usage error (exit 2), not a runtime failure.

    ``UnsupportedFeatureError`` is a ``StructuredError`` -- NOT a ``ValueError``
    -- so without its explicit entry in ``main()``'s usage-error branch it fell
    into the generic runtime-failure branch (exit 1) and scripts misread a
    caller mistake (for example, a strict non-detectable candidate language) as an
    internal failure.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    registry = _demo_registry()

    class _StrictRejectASR:
        def transcribe(self, audio: object, params: object = None) -> None:
            raise UnsupportedFeatureError(
                "Candidate language 'zz' is not detectable by this engine.",
                param="candidate_languages",
                mode="batch",
            )

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    def _create(*_: object, **__: object) -> _StrictRejectASR:
        return _StrictRejectASR()

    monkeypatch.setattr(cli, "discover_models", _discover_models)
    monkeypatch.setattr(registry, "create", _create)

    exit_code = cli.main(["transcribe", "alpha/first", "dummy.wav"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "not detectable" in captured.err


def test_cli_transcribe_transcription_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = _demo_registry()

    class _FailASR:
        def transcribe(self, audio: object, params: object = None) -> None:
            raise TranscriptionError("boom")

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    def _create(*_: object, **__: object) -> _FailASR:
        return _FailASR()

    monkeypatch.setattr(cli, "discover_models", _discover_models)
    monkeypatch.setattr(registry, "create", _create)

    exit_code = cli.main(["transcribe", "alpha/first", "dummy.wav"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "boom" in captured.err


def test_cli_debug_shows_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {"traceback": False}

    def _raise(_: object) -> int:
        raise RuntimeError("boom")

    def _print_exc() -> None:
        called["traceback"] = True

    monkeypatch.setattr(cli, "_cmd_list", _raise)
    monkeypatch.setattr(cli.traceback, "print_exc", _print_exc)

    exit_code = cli.main(["--debug", "list"])

    assert exit_code == 1
    assert called["traceback"] is True


def test_cli_generic_exception_no_debug(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise(_: object) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_cmd_list", _raise)

    exit_code = cli.main(["list"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "boom" in captured.err


def test_cli_compliance_entrypoints_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = ModelRegistry({})
    report = ComplianceReport(
        registry=registry,
        issues=[
            ComplianceIssue(
                level="error",
                code="entrypoint_factory_failed",
                message="Factory invocation failed with RuntimeError('boom').",
                model="alpha/first",
            )
        ],
    )
    monkeypatch.setattr(
        cli,
        "check_entrypoints",
        lambda strict_discovery=False, instantiate=True: report,
    )

    exit_code = cli.main(["compliance", "entrypoints"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "failed" in output
    assert "alpha/first" in output


def test_cli_compliance_entrypoints_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = ModelRegistry({})
    report = ComplianceReport(
        registry=registry,
        issues=[
            ComplianceIssue(
                level="warning",
                code="factory_requires_config",
                message="Minor warning",
                model="alpha/first",
            )
        ],
    )
    monkeypatch.setattr(
        cli,
        "check_entrypoints",
        lambda strict_discovery=False, instantiate=True: report,
    )

    exit_code = cli.main(["compliance", "entrypoints"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "passed" in output
    assert "Warning" in output


def test_cli_compliance_entrypoints_quiet(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = ModelRegistry({})
    report = ComplianceReport(
        registry=registry,
        issues=[
            ComplianceIssue(
                level="warning",
                code="factory_requires_config",
                message="Minor warning",
                model="alpha/first",
            )
        ],
    )
    monkeypatch.setattr(
        cli,
        "check_entrypoints",
        lambda strict_discovery=False, instantiate=True: report,
    )

    exit_code = cli.main(["compliance", "entrypoints", "--quiet"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Warning" not in output


def test_cli_serve_uses_server_module(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("standard_asr.toolchain.server")
    called: dict[str, object] = {}

    def _run(**kwargs: object) -> None:
        called.update(dict(kwargs))

    setattr(module, "run", _run)

    monkeypatch.setitem(__import__("sys").modules, "standard_asr.toolchain.server", module)

    exit_code = cli.main(["serve", "--host", "0.0.0.0", "--port", "9001"])

    assert exit_code == 0
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 9001


def test_cli_serve_missing_server_dependency(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Simulate the server module failing to import (deterministic, no import
    # warnings): a None entry in sys.modules makes `from standard_asr.toolchain.server import run`
    # raise ImportError, exercising the missing-server-deps branch.
    monkeypatch.setitem(sys.modules, "standard_asr.toolchain.server", None)

    exit_code = cli.main(["serve"])
    captured = capsys.readouterr()

    assert exit_code == 1
    # Errors go to stderr, never stdout.
    assert "dependencies are missing" in captured.err
    assert captured.out == ""


def test_cli_serve_missing_dependency_honors_debug(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--debug prints the trace on the handler-caught serve error path too.

    The flag's help and ``docs/content/specification/cli.md`` promise a trace on EVERY error
    path; ``_cmd_serve`` catches its ImportErrors itself and returned 1
    without ever consulting ``--debug``, so this documented path printed
    nothing. Pin the promise.
    """
    monkeypatch.setitem(sys.modules, "standard_asr.toolchain.server", None)

    exit_code = cli.main(["--debug", "serve"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "dependencies are missing" in captured.err
    assert "Traceback" in captured.err


def test_cli_debug_does_not_claim_to_cover_argument_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An argument error ends in the parser, so `--debug` prints usage, not a trace.

    ``main()`` calls ``parse_args`` before the ``try`` that routes failures
    through ``_debug_traceback``, so argparse exits first. The flag's help and
    ``docs/content/specification/cli.md`` said "any error path", which promised a trace here;
    both now name this boundary. Lock the claim to the behavior.
    """
    import pathlib

    with pytest.raises(SystemExit):
        cli.main(["--debug", "transcribe", "--no-such-flag"])
    assert "Traceback" not in capsys.readouterr().err

    help_text = cli.build_parser().format_help()
    assert "any error path" not in help_text

    cli_md = (
        pathlib.Path(__file__).resolve().parents[1]
        / "docs"
        / "content"
        / "specification"
        / "cli.md"
    )
    doc = cli_md.read_text(encoding="utf-8")
    assert "argument errors are reported by the parser" in doc.lower()


def test_cli_serve_importerror_from_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = types.ModuleType("standard_asr.toolchain.server")

    def _run(**_: object) -> None:
        raise ImportError("boom")

    setattr(module, "run", _run)
    monkeypatch.setitem(__import__("sys").modules, "standard_asr.toolchain.server", module)

    exit_code = cli.main(["serve"])
    captured = capsys.readouterr()

    assert exit_code == 1
    # Errors go to stderr, never stdout.
    assert "boom" in captured.err
    assert captured.out == ""


def test_cli_models_prepare_no_prepare_hook_is_noop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An engine without a prepare() hook must NOT trigger a real transcribe as
    # a stand-in (that would be a billable request with side effects for cloud
    # engines). It is a reported no-op instead.
    registry = _demo_registry()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["prepare", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "nothing to warm up" in output.lower()


def test_cli_models_prepare_calls_prepare(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _PrepASR:
        def __init__(self) -> None:
            self.called = False

        def prepare(self) -> None:
            self.called = True

    prep = _PrepASR()
    registry = _demo_registry()

    def _create(*_: object) -> _PrepASR:
        return prep

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(registry, "create", _create)
    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["prepare", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert prep.called is True
    assert "prepare" in output.lower()


def test_parse_options() -> None:
    from standard_asr.contract.params import RuntimeParams

    params = cli._parse_options('{"language": "en"}')  # pyright: ignore[reportPrivateUsage]
    assert isinstance(params, RuntimeParams)
    assert params.language == "en"

    with pytest.raises(ValueError):
        cli._parse_options("[1, 2, 3]")  # pyright: ignore[reportPrivateUsage]

    # The non-portable provider_params key is rejected outright: even an
    # empty object -- which the old RuntimeParams path silently accepted as a
    # bare ProviderParams() -- must fail through WireRuntimeParams.
    with pytest.raises(ValueError, match="provider_params"):
        cli._parse_options('{"provider_params": {}}')  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# --options validation errors MUST NOT echo the submitted value
# (a mis-pasted secret would otherwise reach stderr / CI logs / bug reports).
# ---------------------------------------------------------------------------


def test_parse_options_does_not_echo_secret_value() -> None:
    # A secret mis-placed in --options (rejected by extra="forbid")
    # must not be reflected in the error. pydantic's str(ValidationError) echoes
    # input_value by default; the CLI must scrub it.
    secret = "sk-SUPERSECRET123"  # noqa: S105 - test fixture, not a real credential
    with pytest.raises(ValueError) as excinfo:
        cli._parse_options(  # pyright: ignore[reportPrivateUsage]
            '{"api_key": "' + secret + '"}'
        )
    message = str(excinfo.value)
    assert secret not in message
    # The field name (a credential token) is redacted in the message too.
    assert "[redacted]" in message


def test_cli_transcribe_invalid_options_no_secret_echo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # End-to-end, the mis-placed secret never reaches stderr.
    registry = _demo_registry()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    secret = "sk-LEAKME-999"  # noqa: S105 - test fixture, not a real credential
    exit_code = cli.main(
        ["transcribe", "alpha/first", "dummy.wav", "--options", '{"api_key": "' + secret + '"}']
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert secret not in captured.err
    assert secret not in captured.out


def test_cli_models_prepare_construction_error_no_secret_echo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # RR-014: when engine CONSTRUCTION fails pydantic validation on a credential
    # field, str(ValidationError) echoes the plaintext input_value. main() must route
    # a ValidationError through the shared scrub (the same one --options gets and the
    # server applies on its construction path), so the secret never reaches stderr.
    from pydantic import BaseModel, SecretStr, ValidationError, field_validator

    secret = "sk-CONSTRUCT-LEAK-123"  # noqa: S105 - test fixture, not a real credential

    class _CredConfig(BaseModel):
        api_key: SecretStr

        @field_validator("api_key")
        @classmethod
        def _reject(cls, _v: SecretStr) -> SecretStr:
            raise ValueError("provider rejected the key")

    # Sanity: the raw ValidationError really does leak the plaintext (the bug exists).
    with pytest.raises(ValidationError) as raw:
        _CredConfig(api_key=secret)  # pyright: ignore[reportArgumentType]
    assert secret in str(raw.value)

    registry = _demo_registry()

    def _create(_name: str, /, *args: object, **kwargs: object) -> object:
        return _CredConfig(api_key=secret)  # pyright: ignore[reportArgumentType]

    _patch_discover(monkeypatch, registry)
    monkeypatch.setattr(registry, "create", _create)

    exit_code = cli.main(["prepare", "alpha/first"])
    captured = capsys.readouterr()

    # A RAW ValidationError escaping construction is an ENGINE fault (exit 1),
    # not a usage error: caller-originating pydantic failures are classified
    # upstream, and registry.create wraps real config failures as ConfigError.
    assert exit_code == 1
    assert secret not in captured.err
    assert secret not in captured.out
    assert "engine fault" in captured.err


class _RawValidationErrorASR:
    """A structural engine whose internals raise a raw ``ValidationError``.

    Models a plugin that does not inherit ``EngineBase`` (so nothing wraps
    the seam) and constructs an invalid internal model -- the CLI twin of
    the server's scrubbed-500 case.
    """

    def transcribe(self, audio: Any, options: Any = None) -> Any:
        """Build an invalid internal model.

        Args:
            audio: Ignored.
            options: Ignored.

        Returns:
            Never returns.

        Raises:
            ValidationError: Always, from the engine's own model.
        """
        from pydantic import BaseModel

        class _Internal(BaseModel):
            settings: dict[str, int]

        _Internal.model_validate(
            {"settings": {"sk-PASTED-SECRET-0123456789abcdef0123456789": "bad"}}
        )
        raise AssertionError("unreachable")  # pragma: no cover

    def prepare(self) -> None:
        """Fail the same way during warm-up.

        Raises:
            ValidationError: Always.
        """
        self.transcribe(None)


def test_cli_structural_engine_raw_validation_error_is_an_engine_fault(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The round-7 counterexample: a raw engine VE was reported as usage exit 2.

    Every caller-originating pydantic failure is classified upstream
    (``--options`` by ``_parse_options``, init config by ``registry.create``'s
    ``ConfigError`` wrap, flags by argparse), so a raw ``ValidationError``
    at the top level is a structural engine constructing an invalid model.
    The server maps that seam to a scrubbed 500; the CLI now agrees (exit 1),
    and reports it on the OPERATOR audience -- so a caller-derived mapping
    key inside the engine's own path never reaches stderr or a CI log.

    Args:
        monkeypatch: Pytest fixture for attribute patching.
        capsys: Pytest fixture capturing stdout/stderr.
    """
    registry = _demo_registry()

    def _create(*_: object, **__: object) -> _RawValidationErrorASR:
        return _RawValidationErrorASR()

    monkeypatch.setattr(registry, "create", _create)
    _patch_discover(monkeypatch, registry)

    for argv, label in (
        (["transcribe", "alpha/first", "a.wav"], "transcribe"),
        (["prepare", "alpha/first"], "prepare"),
    ):
        exit_code = cli.main(argv)
        captured = capsys.readouterr()
        assert exit_code == 1, label  # engine fault, not usage
        assert exit_code != 2, label
        assert "engine fault" in captured.err, label
        # Operator surface: the pasted-key-shaped mapping key is masked
        # (identifier-shaped values are the trust model's accepted limit).
        assert "sk-PASTED-SECRET" not in captured.err, label
        assert "[redacted-key]" in captured.err, label


def test_cli_registered_plugin_load_failure_is_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A REGISTERED model whose plugin fails to import is an installation fault.

    ``FactoryLoadError`` subclasses ``DiscoveryError``, so the usage arm
    (exit 2) used to swallow it -- but the caller cannot fix a broken
    installation, and the server maps the same state to a scrubbed 500,
    never a 404 (the round-5 fault-ownership verdict). The CLI now agrees:
    exit 1, with the sanitized summary and no raw chained text.

    Args:
        monkeypatch: Pytest fixture for attribute patching.
        capsys: Pytest fixture capturing stdout/stderr.
    """
    registry = _demo_registry()

    def _create(*_: object, **__: object) -> object:
        raise FactoryLoadError(
            "Failed to load entry point 'alpha/first': No module named 'alpha_plugin'"
        )

    monkeypatch.setattr(registry, "create", _create)
    _patch_discover(monkeypatch, registry)

    for argv, label in (
        (["transcribe", "alpha/first", "a.wav"], "transcribe"),
        (["prepare", "alpha/first"], "prepare"),
    ):
        exit_code = cli.main(argv)
        captured = capsys.readouterr()
        assert exit_code == 1, label
        assert "alpha_plugin" in captured.err, label

    # An UNKNOWN model key stays the caller's usage error (exit 2): only
    # EntrypointValidationError means "fix your model name". A fresh
    # registry, so the real spec lookup runs instead of the patched create.
    _patch_discover(monkeypatch, _demo_registry())
    exit_code = cli.main(["transcribe", "alpha/does-not-exist", "a.wav"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "does-not-exist" in captured.err


class _ShowEntryPoint:
    """The entry-point view `show` prints (module / attr / value)."""

    module = "alpha_plugin"
    attr = "create"
    value = "alpha_plugin:create"


class _ShowSpec:
    """A minimal ModelSpec stand-in: the real one is a frozen dataclass."""

    model_id = "alpha/first"
    engine_id = "alpha"
    model_name = "first"
    entry_point = _ShowEntryPoint()


def test_cli_show_reports_a_broken_plugin_as_an_engine_fault(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``show`` was the consumer that swallowed ``FactoryLoadError``.

    ``main``'s exit-1 arm never saw it: the capabilities helper caught the
    error, printed an ``<unavailable: ...>`` line, and ``_cmd_show``
    returned 0 -- so a script probing a registered-but-broken installation
    read success. The metadata and the sanitized line still print (the
    operator loses no diagnostics), and the fault now propagates.

    Args:
        monkeypatch: Pytest fixture for attribute patching.
        capsys: Pytest fixture capturing stdout/stderr.
    """

    class _BrokenSpec(_ShowSpec):
        def engine_class(self) -> object:
            """Fail like a registered plugin whose import is broken.

            Returns:
                Never returns.

            Raises:
                FactoryLoadError: Always.
            """
            raise FactoryLoadError(
                "Failed to load entry point 'alpha/first': No module named 'alpha_plugin'"
            )

    def _spec(_name: str) -> Any:
        return _BrokenSpec()

    registry = _demo_registry()
    monkeypatch.setattr(registry, "spec", _spec)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["show", "alpha/first"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Engine ID" in captured.out
    assert "Capabilities: <unavailable" in captured.out
    assert "alpha_plugin" in captured.out + captured.err


def test_cli_show_class_attribute_descriptor_is_an_engine_fault(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reading ``declared_capabilities`` is plugin code too.

    ``show`` never constructs the engine, but the CLASS-level lookup can
    still run a plugin descriptor (a metaclass property). Outside the
    engine-fault seam its ``ValueError`` was reported as the invoker's
    usage error.

    Args:
        monkeypatch: Pytest fixture for attribute patching.
        capsys: Pytest fixture capturing stdout/stderr.
    """

    class _CapsMeta(type):
        @property
        def declared_capabilities(cls) -> object:
            raise ValueError("capabilities descriptor exploded")

    class _EngineClass(metaclass=_CapsMeta):
        pass

    class _DescriptorSpec(_ShowSpec):
        def engine_class(self) -> object:
            """Return the engine class whose metaclass property raises.

            Returns:
                The engine class.
            """
            return _EngineClass

    def _spec(_name: str) -> Any:
        return _DescriptorSpec()

    registry = _demo_registry()
    monkeypatch.setattr(registry, "spec", _spec)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["show", "alpha/first"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "capabilities descriptor exploded" in captured.err


def test_cli_prepare_attribute_lookup_runs_inside_the_engine_seam(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ``prepare`` DESCRIPTOR raising is an engine fault, not usage.

    The round-8 envelope wrapped the prepare CALL but not the binding, and
    ``prepare`` may legitimately be a property: its body is engine code, so
    a ``ValueError`` escaping the lookup was mapped to exit 2 by the broad
    usage arm. A raw ``ValidationError`` from the same lookup keeps the
    operator-audience scrub.

    Args:
        monkeypatch: Pytest fixture for attribute patching.
        capsys: Pytest fixture capturing stdout/stderr.
    """

    class _RaisingDescriptorASR:
        @property
        def prepare(self) -> object:
            raise ValueError("prepare descriptor failed internally")

    class _ValidationDescriptorASR:
        @property
        def prepare(self) -> object:
            from pydantic import BaseModel

            class _Internal(BaseModel):
                settings: dict[str, int]

            _Internal.model_validate(
                {"settings": {"sk-PASTED-SECRET-0123456789abcdef0123456789": "bad"}}
            )
            raise AssertionError("unreachable")  # pragma: no cover

    class _CoroutineDescriptorASR:
        @property
        def prepare(self) -> object:
            async def _warm() -> None:  # pragma: no cover - never awaited
                return None

            return _warm

    registry = _demo_registry()
    _patch_discover(monkeypatch, registry)

    for engine_factory, needle in (
        (_RaisingDescriptorASR, "prepare descriptor failed internally"),
        (_ValidationDescriptorASR, "engine fault"),
        (_CoroutineDescriptorASR, "coroutine"),
    ):

        def _create(*_args: object, **_kwargs: object) -> Any:
            return engine_factory()

        monkeypatch.setattr(registry, "create", _create)
        exit_code = cli.main(["prepare", "alpha/first"])
        captured = capsys.readouterr()
        assert exit_code == 1, engine_factory.__name__
        assert needle in captured.err, engine_factory.__name__
        assert "sk-PASTED-SECRET" not in captured.err
        assert "complete" not in captured.out.lower()


def test_cli_engine_bare_value_error_at_execution_is_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare ``ValueError`` escaping the ENGINE call is an engine fault.

    The class is ambiguous -- ``_parse_options`` raises usage ``ValueError``
    too -- so the seam classifies: everything caller-fixable is validated
    before the engine runs, and the server maps the same engine-internal
    ``ValueError`` to a scrubbed 500. Exit 2 here misread an SDK bug as the
    user's mistake (the round-8 counterexample).

    Args:
        monkeypatch: Pytest fixture for attribute patching.
        capsys: Pytest fixture capturing stdout/stderr.
    """
    registry = _demo_registry()

    class _SdkBugASR:
        def transcribe(self, audio: object, params: object = None) -> None:
            raise ValueError("SDK returned malformed response")

        def prepare(self) -> None:
            raise ValueError("SDK returned malformed response")

    def _create(*_: object, **__: object) -> _SdkBugASR:
        return _SdkBugASR()

    monkeypatch.setattr(registry, "create", _create)
    _patch_discover(monkeypatch, registry)

    for argv, label in (
        (["transcribe", "alpha/first", "a.wav"], "transcribe"),
        (["prepare", "alpha/first"], "prepare"),
    ):
        exit_code = cli.main(argv)
        captured = capsys.readouterr()
        assert exit_code == 1, label
        assert "SDK returned malformed response" in captured.err, label

    # The usage ValueError from --options still exits 2: same class,
    # different seam, correctly separated.
    exit_code = cli.main(["transcribe", "alpha/first", "a.wav", "--options", "[1]"])
    captured = capsys.readouterr()
    assert exit_code == 2


def test_cli_config_error_is_invoker_owned_at_every_seam(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``ConfigError`` exits 2 whether it surfaces at construction or later.

    At the CLI the invoking user owns the configuration -- the flags AND the
    env vars -- so "the configuration is invalid/absent" is caller-actionable
    there regardless of WHEN the engine checks it: a factory rejecting a
    supplied value, and a deferred credential check surfacing
    ``ConfigurationRequiredError`` at first transcribe, both name something
    the invoker can fix. (The same errors are a scrubbed 500/503 on the
    server, whose clients cannot supply config: ownership follows the
    supplier, not the exception site.)

    Args:
        monkeypatch: Pytest fixture for attribute patching.
        capsys: Pytest fixture capturing stdout/stderr.
    """
    registry = _demo_registry()

    def _reject(*_: object, **__: object) -> object:
        raise ConfigError("Invalid configuration for 'alpha/first': device: unknown value")

    monkeypatch.setattr(registry, "create", _reject)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["transcribe", "alpha/first", "a.wav", "--set", "device=warp"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "device" in captured.err

    class _LazyCredentialASR:
        def transcribe(self, audio: object, params: object = None) -> None:
            raise ConfigurationRequiredError(
                "Engine 'alpha' requires configuration: set STANDARD_ASR_ALPHA__API_KEY."
            )

    def _create(*_: object, **__: object) -> _LazyCredentialASR:
        return _LazyCredentialASR()

    monkeypatch.setattr(registry, "create", _create)

    exit_code = cli.main(["transcribe", "alpha/first", "a.wav"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "STANDARD_ASR_ALPHA__API_KEY" in captured.err


def test_cli_serve_doc_does_not_list_unparsed_reload_flag() -> None:
    # removed --reload from the serve parser; a later doc commit
    # re-listed it, promising a flag the CLI rejects with SystemExit(2). Lock the doc
    # and parser together: the parser rejects --reload, so cli.md must omit it.
    import pathlib

    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--reload"])  # parser intentionally has no --reload

    cli_md = (
        pathlib.Path(__file__).resolve().parents[1]
        / "docs"
        / "content"
        / "specification"
        / "cli.md"
    )
    doc = cli_md.read_text(encoding="utf-8")
    serve_section = doc.split("### `standard-asr serve`", 1)[1].split("\n### ", 1)[0]
    assert "--reload" not in serve_section


def test_cli_no_command_prints_help_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """A bare `standard-asr` prints help (with examples) and exits 0, not an error.

    The flat-verb redesign registers the subparsers with ``required=False`` so the
    first-run experience is the help screen rather than an argparse "arguments are
    required" error.
    """
    exit_code = cli.main([])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "usage: standard-asr" in out
    assert "Examples:" in out  # the epilog is shown


def test_cli_flat_verbs_resolve_to_handlers() -> None:
    """The common verbs are flat top-level commands (no nested `models` group).

    Asserted against the parser's actual registration -- each verb resolves to its
    handler -- rather than help-text substrings, which the epilog examples would
    satisfy even if a subparser were dropped or re-nested under `models`.
    """
    parser = cli.build_parser()
    resolved = {
        "list": parser.parse_args(["list"]).func.__name__,
        "show": parser.parse_args(["show", "e/m"]).func.__name__,
        "cache": parser.parse_args(["cache"]).func.__name__,
        "prepare": parser.parse_args(["prepare", "e/m"]).func.__name__,
        "status": parser.parse_args(["status", "e/m"]).func.__name__,
        "pull": parser.parse_args(["pull", "e/m"]).func.__name__,
        "transcribe": parser.parse_args(["transcribe", "e/m", "a.wav"]).func.__name__,
        "serve": parser.parse_args(["serve"]).func.__name__,
        "doctor": parser.parse_args(["doctor"]).func.__name__,
    }
    assert resolved == {
        "list": "_cmd_list",
        "show": "_cmd_show",
        "cache": "_cmd_cache",
        "prepare": "_cmd_prepare",
        "status": "_cmd_status",
        "pull": "_cmd_pull",
        "transcribe": "_cmd_transcribe",
        "serve": "_cmd_serve",
        "doctor": "_cmd_doctor",
    }
    # The old nested `models` group is gone: it is no longer a valid command.
    with pytest.raises(SystemExit):
        parser.parse_args(["models"])


# ---------------------------------------------------------------------------
# Prepare warm-up hook contract (sync, zero-arg; reject coroutine).
# ---------------------------------------------------------------------------


def test_cli_models_prepare_rejects_coroutine_hook(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An `async def prepare` would be callable and return an
    # un-awaited coroutine; calling it must NOT report a false "prepare complete".
    # Exit 1, not 2: a declaration-shape defect is the ENGINE author's
    # contract violation (EngineContractError) -- no flag or env var the CLI
    # user owns can fix it (round-8 fault-ownership rule).
    class _AsyncPrepASR:
        async def prepare(self) -> None:  # noqa: D401 - test double
            return None

    registry = _demo_registry()

    def _create(*_: object) -> _AsyncPrepASR:
        return _AsyncPrepASR()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(registry, "create", _create)
    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["prepare", "alpha/first"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "coroutine" in captured.err
    assert "complete" not in captured.out.lower()


def test_cli_models_prepare_rejects_non_callable_hook(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A non-callable 'prepare' attribute is a declaration bug, not a
    # "no hook" case -- reject it loudly, as an engine fault (exit 1).
    class _BadPrepASR:
        prepare = "not callable"

    registry = _demo_registry()

    def _create(*_: object) -> _BadPrepASR:
        return _BadPrepASR()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(registry, "create", _create)
    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["prepare", "alpha/first"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "non-callable" in captured.err


def test_cli_models_prepare_rejects_required_args_hook(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The warm-up hook MUST be invocable with no arguments. A prepare()
    # that requires a parameter can never be driven by the CLI -- reject it with a
    # structured error rather than letting the call blow up with a bare TypeError
    # (mirrors the compliance suite's 'prepare_hook_requires_args'); an
    # engine fault (exit 1).
    class _ArgPrepASR:
        def prepare(self, warmup_level: int) -> None:  # noqa: D401 - test double
            return None

    registry = _demo_registry()

    def _create(*_: object) -> _ArgPrepASR:
        return _ArgPrepASR()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(registry, "create", _create)
    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["prepare", "alpha/first"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "required parameters" in captured.err
    assert "complete" not in captured.out.lower()


def test_cli_models_prepare_engine_base_default_is_noop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # EngineBase now provides a default no-op prepare; an engine
    # that did NOT override it must be reported as "nothing to warm up", not a
    # misleading "prepare complete".
    engine = _GatingStreamEngine()

    registry = _demo_registry()

    def _create(*_: object) -> _GatingStreamEngine:
        return engine

    monkeypatch.setattr(registry, "create", _create)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["prepare", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "nothing to warm up" in output.lower()


# ---------------------------------------------------------------------------
# The runtime sync-call boundary at the CLI's REAL consumer sites: a plugin
# whose sync member hands back a coroutine (or a wrong-typed value) must be an
# ENGINE fault (exit 1) surfaced at the boundary -- never a false success, a
# secondary AttributeError, or a leaked never-awaited coroutine.
# ---------------------------------------------------------------------------


def test_cli_transcribe_async_engine_is_engine_fault_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sync-wrapper ``transcribe`` returning a coroutine exits 1, loudly.

    Pre-fix the coroutine flowed into ``result.text`` -- an unhandled
    ``AttributeError`` traceback plus a never-awaited ``RuntimeWarning``.
    The boundary closes the coroutine and names the real defect.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """

    class _AsyncASR:
        async def _impl(self) -> None:
            """Async implementation (never driven)."""
            return None  # pragma: no cover - never awaited

        def transcribe(self, audio: Any, options: Any = None) -> Any:
            """Delegate to the async implementation (returns a coroutine)."""
            return self._impl()

    registry = _demo_registry()

    def _create(*_: object) -> _AsyncASR:
        return _AsyncASR()

    monkeypatch.setattr(registry, "create", _create)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["transcribe", "alpha/first", "dummy.wav"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "transcribe() returned an awaitable" in captured.err
    assert "engine/plugin bug" in captured.err


def test_cli_transcribe_wrong_result_type_is_engine_fault_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-``TranscriptionResult`` return is an engine fault named by type.

    The message carries type names only -- the engine's value (which could
    embed arbitrary payload text) is never echoed.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """

    class _DictASR:
        def transcribe(self, audio: Any, options: Any = None) -> Any:
            """Return the wrong type (a plain dict)."""
            return {"text": "dict-not-result"}

    registry = _demo_registry()

    def _create(*_: object) -> _DictASR:
        return _DictASR()

    monkeypatch.setattr(registry, "create", _create)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["transcribe", "alpha/first", "dummy.wav"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "transcribe() returned dict, not TranscriptionResult" in captured.err
    assert "dict-not-result" not in captured.err


def test_cli_models_prepare_sync_wrapper_returning_coroutine_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sync ``prepare`` delegating to an ``async def`` must not read as success.

    ``iscoroutinefunction`` cannot see this shape (the declaration IS sync);
    pre-fix the CLI printed "Model prepare complete." while nothing warmed up
    and the coroutine leaked. The strict require-``None`` boundary kills the
    false success: engine fault, exit 1.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """

    class _WrapperPrepASR:
        async def _async_prepare(self) -> None:
            """Async warm-up implementation (never driven)."""
            return None  # pragma: no cover - never awaited

        def prepare(self) -> Any:
            """Delegate to the async implementation (returns a coroutine)."""
            return self._async_prepare()

    registry = _demo_registry()

    def _create(*_: object) -> _WrapperPrepASR:
        return _WrapperPrepASR()

    monkeypatch.setattr(registry, "create", _create)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["prepare", "alpha/first"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "complete" not in captured.out.lower()
    assert "prepare() returned an awaitable" in captured.err


def test_cli_models_prepare_non_none_return_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``prepare()`` is pinned to return ``None``; any other value is a defect.

    Args:
        monkeypatch: Pytest fixture for attribute patching.
        capsys: Pytest fixture capturing stdout/stderr.
    """

    class _ChattyPrepASR:
        def prepare(self) -> Any:
            """Return a non-None value (a contract violation)."""
            return "warmed"

    registry = _demo_registry()

    def _create(*_: object) -> _ChattyPrepASR:
        return _ChattyPrepASR()

    monkeypatch.setattr(registry, "create", _create)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["prepare", "alpha/first"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "complete" not in captured.out.lower()
    assert "prepare() returned str, not None" in captured.err


def test_cli_debug_traceback_does_not_reopen_the_pydantic_echo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--debug`` must not double as a credential-leak opt-in.

    The normal path scrubs the ``--options`` ValidationError; pre-fix,
    ``--debug``'s ``traceback.print_exc()`` re-rendered the chained raw
    error -- ``input_value=`` echo included -- into stderr/CI logs. The debug
    trace now uses the safe chain rendering when a ValidationError is in the
    chain (a plain exception keeps the full ordinary traceback).

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    registry = _demo_registry()
    _patch_discover(monkeypatch, registry)
    secret = "sk-DEBUG-SENTINEL"  # noqa: S105 - test fixture, not a real credential

    exit_code = cli.main(
        ["--debug", "transcribe", "alpha/first", "a.wav", "--options", f'{{"api_key": "{secret}"}}']
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert secret not in captured.err
    assert secret not in captured.out
    # The debug trace still exists and still names the fault structure.
    assert "ValidationError" in captured.err
    assert "test_cli" not in captured.out  # sanity: trace goes to stderr


def test_cli_debug_traceback_keeps_full_trace_for_plain_exceptions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A chain without a ValidationError keeps the ordinary full traceback.

    Args:
        monkeypatch: Pytest fixture for attribute patching.
        capsys: Pytest fixture capturing stdout/stderr.
    """

    class _BoomASR:
        def transcribe(self, audio: Any, options: Any = None) -> Any:
            """Raise a plain engine fault."""
            raise RuntimeError("boom: plain engine fault")

    registry = _demo_registry()

    def _create(*_: object) -> _BoomASR:
        return _BoomASR()

    monkeypatch.setattr(registry, "create", _create)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["--debug", "transcribe", "alpha/first", "a.wav"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Traceback (most recent call last)" in captured.err
    assert "boom: plain engine fault" in captured.err


class _EchoCopyingASR:
    """An engine that wraps a pydantic failure by copying its text.

    The classic third-party shape ``raise RuntimeError(f"...: {exc}") from
    exc``: the wrapper's own message embeds pydantic's (truncated) input
    echo, so any surface printing ``str(exc)`` re-leaks it.
    """

    def __init__(self, secret: str) -> None:
        """Store the credential to mis-place into a validation failure.

        Args:
            secret: The sentinel credential.
        """
        self._secret = secret

    def transcribe(self, audio: Any, options: Any = None) -> Any:
        """Fail validation with the secret as input, then copy the echo.

        Args:
            audio: Ignored.
            options: Ignored.

        Returns:
            Never returns.

        Raises:
            RuntimeError: Always, message embedding the pydantic echo.
        """
        from pydantic import BaseModel, ValidationError

        class _Params(BaseModel):
            beam: int

        try:
            _Params.model_validate({"beam": self._secret})
        except ValidationError as exc:
            raise RuntimeError(f"plugin failed: {exc}") from exc
        raise AssertionError("unreachable")  # pragma: no cover


def test_cli_normal_error_line_scrubs_wrapped_validation_echo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The NORMAL error line (no ``--debug``) is the leak that mattered.

    ``_debug_traceback`` only guards the opt-in trace; the unconditional
    ``_print_error(str(exc))`` line printed the copying wrapper's message --
    truncated input echo included -- before any traceback logic ran. Every
    catch arm now reports through the safe boundary. The secret is LONG so
    pydantic truncates it: a substring check against the full value can
    never catch this copy (the round-3 counterexample).

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    secret = "sk-" + "A" * 300 + "-END"  # noqa: S105 - test fixture
    registry = _demo_registry()

    def _create(*_: object) -> _EchoCopyingASR:
        return _EchoCopyingASR(secret)

    monkeypatch.setattr(registry, "create", _create)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["transcribe", "alpha/first", "a.wav"])
    captured = capsys.readouterr()

    assert exit_code == 1
    for stream in (captured.err, captured.out):
        assert "sk-AAAAA" not in stream  # truncated prefix must not leak
        assert "-END" not in stream  # truncated suffix must not leak
    # The line still names the fault structure for the operator.
    assert "RuntimeError" in captured.err
    assert "ValidationError" in captured.err


def test_cli_debug_error_line_scrubs_wrapped_validation_echo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With ``--debug`` the normal line AND the trace stay echo-free.

    Args:
        monkeypatch: Pytest fixture for attribute patching.
        capsys: Pytest fixture capturing stdout/stderr.
    """
    secret = "sk-" + "B" * 300 + "-TAIL"  # noqa: S105 - test fixture
    registry = _demo_registry()

    def _create(*_: object) -> _EchoCopyingASR:
        return _EchoCopyingASR(secret)

    monkeypatch.setattr(registry, "create", _create)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["--debug", "transcribe", "alpha/first", "a.wav"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "sk-BBBBB" not in captured.err
    assert "-TAIL" not in captured.err
    # The safe chain rendering still shows structure and locations.
    assert "ValidationError" in captured.err


def test_cli_hostile_exception_str_reports_placeholder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An exception whose ``__str__`` raises cannot crash error reporting.

    The message is WITHHELD rather than probed: an author-defined display
    is unauditable code, so the summary never dispatches it at all (a
    hostile ``__str__`` that returned a credential would have been rendered
    by the old "call it and see" placeholder path). Reporting still
    completes -- the type name and the usage exit code survive.

    Args:
        monkeypatch: Pytest fixture for attribute patching.
        capsys: Pytest fixture capturing stdout/stderr.
    """

    class _HostileError(Exception):
        def __str__(self) -> str:
            """Raise unconditionally.

            Returns:
                Never returns.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("boom from __str__")

    def _raise(_: object) -> int:
        raise _HostileError()

    monkeypatch.setattr(cli, "_cmd_list", _raise)

    exit_code = cli.main(["list"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "_HostileError: <exception str() failed>" in captured.err

    # A display set to None dispatches no author code, so the untainted
    # branch still renders it -- through the totality placeholder, since
    # the protocol itself raises.
    class _NoDisplayError(Exception):
        __str__ = None  # pyright: ignore[reportAssignmentType]

    def _raise_no_display(_: object) -> int:
        raise _NoDisplayError()

    monkeypatch.setattr(cli, "_cmd_list", _raise_no_display)
    assert cli.main(["list"]) == 1
    assert "_NoDisplayError: <exception str() failed>" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Redirected/piped output must not crash on non-ASCII text; status
# markers are ASCII and the streams are forced to UTF-8.
# ---------------------------------------------------------------------------


def test_ensure_utf8_stream_reconfigures_non_utf8() -> None:
    # A cp1252-backed strict stream (the Windows redirect default)
    # must be switched to UTF-8 so non-Latin transcripts print loss-lessly
    # instead of raising UnicodeEncodeError.
    import io

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    cli._ensure_utf8_stream(stream)  # pyright: ignore[reportPrivateUsage]
    assert stream.encoding == "utf-8"

    stream.write("你好 mañana")
    stream.flush()
    assert raw.getvalue().decode("utf-8") == "你好 mañana"


def test_ensure_utf8_stream_noop_on_utf8() -> None:
    # Already-UTF-8 streams (the POSIX default) are left untouched.
    import io

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8")
    cli._ensure_utf8_stream(stream)  # pyright: ignore[reportPrivateUsage]
    assert stream.encoding == "utf-8"


def test_ensure_utf8_stream_tolerates_missing_reconfigure() -> None:
    # A stream without reconfigure() (for example, a plain StringIO) must not crash.
    import io

    cli._ensure_utf8_stream(io.StringIO())  # pyright: ignore[reportPrivateUsage]


def test_cli_status_markers_are_ascii() -> None:
    # The decorative status markers must be ASCII so a redirected
    # ANSI-code-page stream never raises on them.
    for marker in (cli._OK, cli._FAIL, cli._WARN, cli._INFO):  # pyright: ignore[reportPrivateUsage]
        marker.encode("ascii")  # raises if any marker is non-ASCII


# ---------------------------------------------------------------------------
# Text mode renders TranscriptionResult.diagnostics to stderr
# (stdout stays a clean, pipeable transcript).
# ---------------------------------------------------------------------------


def test_cli_transcribe_text_mode_renders_diagnostics_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A lossy step's diagnostic must not vanish on the default text
    # surface -- it goes to stderr, stdout stays the bare transcript.
    from standard_asr.contract.results import Diagnostic, TranscriptionResult

    result = TranscriptionResult(
        text="hello world",
        diagnostics=[Diagnostic(code="resampled_with", message="resampled 8000->16000 via scipy")],
    )

    class _DiagASR:
        def transcribe(self, audio: object, params: object = None) -> TranscriptionResult:
            return result

    registry = _demo_registry()

    def _create(*_: object) -> _DiagASR:
        return _DiagASR()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(registry, "create", _create)
    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["transcribe", "alpha/first", "dummy.wav"])
    captured = capsys.readouterr()

    assert exit_code == 0
    # stdout: bare transcript only (pipeable).
    assert captured.out.strip() == "hello world"
    # stderr: the diagnostic surfaced with its code and message.
    assert "resampled_with" in captured.err
    assert "8000->16000" in captured.err


def test_cli_transcribe_json_mode_keeps_diagnostics_off_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # --json already carries diagnostics on the result; the text-mode
    # stderr rendering must NOT also fire (no double reporting).
    from standard_asr.contract.results import Diagnostic, TranscriptionResult

    result = TranscriptionResult(
        text="hi",
        diagnostics=[Diagnostic(code="audio_conversion", message="decoded wav")],
    )

    class _DiagASR:
        def transcribe(self, audio: object, params: object = None) -> TranscriptionResult:
            return result

    registry = _demo_registry()

    def _create(*_: object) -> _DiagASR:
        return _DiagASR()

    monkeypatch.setattr(registry, "create", _create)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["transcribe", "alpha/first", "dummy.wav", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "audio_conversion" in captured.out  # present in JSON
    assert captured.err == ""  # not duplicated to stderr


# ---------------------------------------------------------------------------
# Models show renders canonical_json (derived `supported` at every
# node) and defends against a mis-typed declared_capabilities.
# ---------------------------------------------------------------------------


def test_cli_models_show_uses_canonical_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The rendered capabilities must be the CANONICAL shape --
    # every node, INCLUDING container nodes like the `batch` domain, carries a
    # derived `supported` boolean (REST and CLI agree). A bare `"supported" in
    # output` is non-discriminating: model_dump(mode="json") also emits `supported`
    # on leaf flags. Assert it on a container node, which only canonical_json injects.
    import json

    registry = _demo_registry()
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["show", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Capabilities:" in output
    # Isolate the capabilities JSON block (the metadata section follows it).
    caps_block = output.split("Capabilities:", 1)[1].split("Declared metadata:", 1)[0]
    caps = json.loads(caps_block)
    # The `batch` domain is a container with no `supported` field of its own;
    # canonical_json derives one (true here), model_dump(mode="json") would not.
    assert caps["batch"]["supported"] is True


def test_cli_models_show_defends_mistyped_capabilities(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An engine that mis-declares declared_capabilities as a dict must
    # not crash `show` with an opaque AttributeError; the rest of the
    # metadata still renders and the author is pointed at compliance.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_cli:_dict_caps_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["show", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    # Other metadata still renders.
    assert "Engine ID" in output
    # The capabilities line names the problem and points at compliance.
    assert "invalid" in output
    assert "compliance entrypoints" in output


# ---------------------------------------------------------------------------
# --debug emits a stack trace for errors caught by a named branch,
# not only the final generic branch.
# ---------------------------------------------------------------------------


def test_cli_debug_traceback_for_named_branch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A ValueError (caught by the ConfigError/DiscoveryError/ValueError
    # branch -> exit 2) must still print a trace under --debug. Previously only
    # the final `except Exception` branch honored --debug.
    called: dict[str, bool] = {"traceback": False}

    def _raise(_: object) -> int:
        raise ValueError("engine internal value error")

    def _print_exc() -> None:
        called["traceback"] = True

    monkeypatch.setattr(cli, "_cmd_list", _raise)
    monkeypatch.setattr(cli.traceback, "print_exc", _print_exc)

    exit_code = cli.main(["--debug", "list"])

    assert exit_code == 2
    assert called["traceback"] is True


def test_cli_no_debug_no_traceback_for_named_branch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Without --debug, the named branch prints the message but no trace.
    called: dict[str, bool] = {"traceback": False}

    def _raise(_: object) -> int:
        raise ValueError("engine internal value error")

    monkeypatch.setattr(cli, "_cmd_list", _raise)
    monkeypatch.setattr(cli.traceback, "print_exc", lambda: called.__setitem__("traceback", True))

    exit_code = cli.main(["list"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert called["traceback"] is False
    assert "engine internal value error" in captured.err


# ---------------------------------------------------------------------------
# `compliance run` aggregates entry points + streaming gating, and
# names the event-sequence dimension it cannot run.
# ---------------------------------------------------------------------------


def _compliant_dummy_registry() -> ModelRegistry:
    # _GatingStreamEngine is a fully compliant EngineBase subclass (passes
    # check_entrypoints), unlike the minimal structural _dummy_factory.
    eps = [
        EntryPoint(
            name="stream/ok",
            value="tests.test_cli:_gating_stream_factory",
            group="standard_asr.models",
        )
    ]
    return discover_models(eps=eps, strict=True)


def test_cli_compliance_run_aggregates_and_passes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `compliance run` reports the entry point result AND points the author at
    # BOTH checks the CLI cannot synthesize (event-sequence and result).
    registry = _compliant_dummy_registry()
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["compliance", "run", "stream/ok"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Entry point compliance checks passed" in output
    assert "check_event_sequence" in output
    assert "check_transcription_result" in output
    assert "Compliance run passed" in output


def test_cli_compliance_run_batch_only_engine(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A batch-only engine (no streaming capabilities) skips the streaming
    # compliance checks entirely — exercises the non-streaming branch.
    eps = [
        EntryPoint(
            name="batch/only",
            value="tests.test_cli:_batch_only_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["compliance", "run", "batch/only"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Compliance run passed" in output
    # A batch-only engine must still be told the result check is not run here --
    # the omission that was previously never disclosed for its shape.
    assert "check_transcription_result" in output
    assert "check_event_sequence" in output


def test_cli_compliance_run_batch_only_bad_wire_format_fails_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A batch-only engine's broken wire recommendation FAILS a full run — once.

    The F4 counterexample: `recommended_wire_format()` is unconditional (spec
    §3.1), but the CLI used to run its round-trip only for streaming-axis
    engines, so a batch-only engine shipped this defect through a green
    `compliance run`. The round-trip now lives in the entrypoint instance
    checks — and ONLY there, so one run reports the defect exactly once (the
    old CLI-side call would now double-report).

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    eps = [
        EntryPoint(
            name="batch/badwire",
            value="tests.test_cli:_batch_only_bad_wire_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["compliance", "run", "batch/badwire"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert output.count("recommended_wire_format_self_inconsistent") == 1


def test_parse_init_config_merges_config_under_set() -> None:
    """--config supplies a base; --set overrides/adds (and wins). --set values
    stay strings (the engine's pydantic config coerces them, like the env path).
    """
    ns = argparse.Namespace(
        config='{"device": "cpu", "beam_size": 1}',
        set_=["beam_size=5", "compute_type=int8"],
    )
    assert cli._parse_init_config(ns) == {  # pyright: ignore[reportPrivateUsage]
        "device": "cpu",
        "beam_size": "5",
        "compute_type": "int8",
    }


def test_parse_init_config_empty_when_unset() -> None:
    """No --config / --set -> empty mapping (create() called with no init config)."""
    assert cli._parse_init_config(argparse.Namespace(config=None, set_=None)) == {}  # pyright: ignore[reportPrivateUsage]


def test_parse_init_config_rejects_non_object_config() -> None:
    ns = argparse.Namespace(config="[1, 2]", set_=None)
    with pytest.raises(ConfigError, match="JSON object"):
        cli._parse_init_config(ns)  # pyright: ignore[reportPrivateUsage]


def test_parse_init_config_rejects_invalid_json_config() -> None:
    ns = argparse.Namespace(config="{not json", set_=None)
    with pytest.raises(ConfigError, match="JSON object"):
        cli._parse_init_config(ns)  # pyright: ignore[reportPrivateUsage]


def test_parse_init_config_rejects_set_without_equals() -> None:
    ns = argparse.Namespace(config=None, set_=["noequals"])
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        cli._parse_init_config(ns)  # pyright: ignore[reportPrivateUsage]


def test_parse_init_config_rejects_set_with_empty_key() -> None:
    # An empty key must NOT echo the (possibly secret) value back.
    ns = argparse.Namespace(config=None, set_=["=sk-secret"])
    with pytest.raises(ConfigError, match="non-empty key") as excinfo:
        cli._parse_init_config(ns)  # pyright: ignore[reportPrivateUsage]
    assert "sk-secret" not in str(excinfo.value)


def test_cli_compliance_run_skips_engine_needing_args(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An engine whose factory needs arguments (for example, credentials)
    # cannot be exercised for the streaming checks; the skip is reported (the
    # streaming checks cannot supply real credentials), not failed.
    registry = _compliant_dummy_registry()

    def _spec_is_zero_arg(_spec: object) -> bool:
        return False

    _patch_discover(monkeypatch, registry)
    monkeypatch.setattr(cli, "_spec_is_zero_arg", _spec_is_zero_arg)

    exit_code = cli.main(["compliance", "run", "stream/ok"])
    output = capsys.readouterr().out

    # The streaming-check skip is reported; entry points still validated.
    assert exit_code == 0
    assert "skipped streaming checks" in output


def test_cli_compliance_run_streaming_engine_gating(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A zero-arg streaming engine has its gating check executed by
    # `compliance run` (not just entry points). A compliant gating engine passes.
    eps = [
        EntryPoint(
            name="stream/ok",
            value="tests.test_cli:_gating_stream_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["compliance", "run", "stream/ok"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Compliance run passed" in output


def test_cli_compliance_run_executes_swap_safety_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Integration: `compliance run` MUST exercise
    # the provider_params swap-safety dimension for
    # every constructed engine, not just streaming ones -- it previously wired in
    # only entrypoints + streaming gating, silently omitting an unconditional
    # MUST. Spy the check to prove it runs for the constructed engine.
    called: list[object] = []

    def _spy(engine: object) -> ComplianceReport:
        called.append(engine)
        return ComplianceReport(registry=ModelRegistry({}), issues=[])

    monkeypatch.setattr(cli, "check_provider_params_swap_safety", _spy)
    registry = _compliant_dummy_registry()
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["compliance", "run"])

    assert exit_code == 0
    assert called, "compliance run did not execute the provider_params swap-safety check"


def test_cli_compliance_run_detects_ungated_streaming_engine(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The whole point -- a streaming engine that bypassed gating is
    # caught by `compliance run` even though its entry points are valid.
    eps = [
        EntryPoint(
            name="stream/bad",
            value="tests.test_cli:_ungated_stream_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["compliance", "run", "stream/bad"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Compliance run failed" in output


def test_cli_compliance_run_include_bridge_runs_sync_bridge(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # --include-bridge additionally drives the sync-bridge check
    # (which opens a streaming session). A compliant streaming engine passes.
    eps = [
        EntryPoint(
            name="stream/ok",
            value="tests.test_cli:_gating_stream_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["compliance", "run", "stream/ok", "--include-bridge"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Compliance run passed" in output


def test_cli_compliance_run_bridge_not_applicable_for_output_only_engine(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--include-bridge on an output-only engine yields a STRUCTURED warning.

    The bridge feeds bare PCM frames; an output-only engine's
    ``start_transcription(audio_format=...)`` fails the ``streaming_input``
    capability gate. The CLI no longer pre-gates with an ad-hoc ``print`` a
    script could never see: it always runs the bridge and hands it ``engine=``
    so ``check_sync_bridge`` itself classifies the refusal, putting a
    ``sync_bridge_not_applicable`` warning in the REPORT SET -- one layer, one
    message, visible to machine consumers and honoring ``--quiet``.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    eps = [
        EntryPoint(
            name="streamout/only",
            value="tests.test_cli:_output_only_stream_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    # Record the engine each check receives, delegating to the REAL checks so
    # the classification below is the shipped behavior, not a stub's.
    constructed: list[object] = []
    real_swap_check = cli.check_provider_params_swap_safety
    real_check_sync_bridge = cli.check_sync_bridge
    bridge_kwargs: list[dict[str, object]] = []

    def _spy_swap(engine: object) -> ComplianceReport:
        constructed.append(engine)
        return real_swap_check(cast(StandardASR, engine))

    def _spy_bridge(factory: object, **kwargs: object) -> ComplianceReport:
        bridge_kwargs.append(dict(kwargs))
        return real_check_sync_bridge(
            cast(Callable[[], TranscriptionSession], factory),
            **cast(Any, kwargs),
        )

    monkeypatch.setattr(cli, "check_provider_params_swap_safety", _spy_swap)
    monkeypatch.setattr(cli, "check_sync_bridge", _spy_bridge)

    exit_code = cli.main(["compliance", "run", "streamout/only", "--include-bridge"])
    output = capsys.readouterr().out

    assert exit_code == 0
    # The bridge RUNS now -- no CLI-side capability pre-gate.
    assert len(bridge_kwargs) == 1
    # ...and it is handed THE constructed engine, so the check can read the
    # engine's own streaming_input declaration (the only thing that earns the
    # passing not-applicable verdict).
    assert bridge_kwargs[0]["engine"] is constructed[0]
    assert bridge_kwargs[0]["model"] == "streamout/only"
    # Structured warning in the report set, not an ad-hoc informational print.
    assert "[WARN] Warning streamout/only [sync_bridge_not_applicable]:" in output
    assert "streaming_input" in output
    assert "skipped sync-bridge" not in output
    assert "Compliance run passed" in output


def test_cli_compliance_run_quiet_suppresses_the_not_applicable_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--quiet hides the not-applicable warning line; the run still passes.

    This is the DX the old ad-hoc ``print`` could not deliver: a warning that
    lives in the report set obeys the run's own verbosity flag.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    eps = [
        EntryPoint(
            name="streamout/only",
            value="tests.test_cli:_output_only_stream_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["compliance", "run", "streamout/only", "--include-bridge", "--quiet"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "sync_bridge_not_applicable" not in output
    assert "Compliance run passed" in output


@pytest.mark.parametrize("bad", ["0", "-1", "-0.5", "nan", "inf", "-inf", "abc", "1e300"])
def test_cli_bridge_timeout_rejects_nonpositive_and_nonfinite(bad: str) -> None:
    """--bridge-timeout must be finite, > 0, and within the platform's wait cap.

    A bare ``type=float`` would accept these; fed into ``Thread.join`` they
    become an immediately expiring timeout (a false "did not terminate"
    verdict blamed on the engine), a hang (``inf``), or an ``OverflowError``
    mid-check (a finite value beyond ``threading.TIMEOUT_MAX``, for example,
    ``1e300``). argparse rejects them as usage errors (exit 2) at parse time.

        Args:
            bad: A value the rule must reject (parametrized).
    """
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["compliance", "run", "--include-bridge", "--bridge-timeout", bad])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_cli_bridge_timeout_surfaces_the_librarys_own_rule_text(bad: float) -> None:
    """The CLI parser delegates the rule to ``compliance.validate_bridge_timeout``.

    One rule, one owner: the CLI only wraps the library's ``ValueError`` into
    an argparse usage error, so the two layers can never drift on WHICH values
    are legal or on how the rejection is explained.

        Args:
            bad: A value the rule must reject (parametrized).
    """
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        cli._positive_finite_seconds(str(bad))  # pyright: ignore[reportPrivateUsage]


def test_cli_bridge_timeout_rejects_a_non_numeric_value_before_the_rule() -> None:
    """A non-float never reaches validate_bridge_timeout: the conversion error is
    its own message, naming the offending text.
    """
    with pytest.raises(argparse.ArgumentTypeError, match="invalid float value"):
        cli._positive_finite_seconds("abc")  # pyright: ignore[reportPrivateUsage]


def test_cli_bridge_timeout_accepts_positive_finite_value() -> None:
    """A positive finite --bridge-timeout parses to its float value."""
    parser = cli.build_parser()
    args = parser.parse_args(["compliance", "run", "--bridge-timeout", "12.5"])
    assert args.bridge_timeout == 12.5


def test_cli_bridge_not_applicable_when_output_only_engine_has_no_wire_format(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No recommendable wire format + no streaming_input = not applicable.

    An output-only engine that recommends no bare-frame format (a structural
    engine may legitimately return None) must earn the same passing
    ``sync_bridge_not_applicable`` verdict as one that CAN construct a format
    -- the old hard ``sync_bridge_no_wire_format`` error failed a compliant
    output-only engine on a property of the check's shape. A streaming_input
    engine without a usable format keeps the hard error (a real declaration
    problem).

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    eps = [
        EntryPoint(
            name="streamout/only",
            value="tests.test_cli:_output_only_stream_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    def _no_wire_format(engine: object) -> None:
        return None

    monkeypatch.setattr(cli, "_streaming_audio_format", _no_wire_format)

    exit_code = cli.main(["compliance", "run", "streamout/only", "--include-bridge"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "sync_bridge_not_applicable" in output
    assert "sync_bridge_no_wire_format" not in output


def test_cli_bridge_unclassifiable_wire_format_gets_its_own_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A wire-format value the boundary cannot classify is reported not re-read.

    The sync-bridge gate is a REAL consumer of the boundary verdict: a
    hostile ``__mro__`` metaclass makes every metadata read raise, so the
    report must come from the VERDICT alone (never a second
    ``type(value).__name__`` -- that read would crash the error path).

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    eps = [
        EntryPoint(
            name="streamout/only",
            value="tests.test_cli:_output_only_stream_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    class _HostileMroMeta(type):
        @property
        def __mro__(cls) -> tuple[type, ...]:
            """Fail any introspection that reads the MRO.

            Returns:
                Never returns.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("hostile mro read")

    class _Hostile(metaclass=_HostileMroMeta):
        pass

    def _hostile_wire_format(engine: object) -> object:
        return _Hostile()

    monkeypatch.setattr(cli, "_streaming_audio_format", _hostile_wire_format)

    exit_code = cli.main(["compliance", "run", "streamout/only", "--include-bridge"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "protocol_member_unclassifiable_result" in output
    assert "could not be safely classified" in output


def test_cli_no_wire_format_with_broken_supports_keeps_the_hard_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A raising supports() cannot buy the not-applicable pass here either.

    The no-wire-format branch takes the passing not-applicable verdict only
    when the engine VERIFIABLY does not declare streaming_input; an engine
    whose supports() raises is unverifiable and keeps the hard error --
    mirroring check_sync_bridge's own fail-closed stance. The message must
    name the ACTUAL fault (supports() raised, declaration unverifiable), not
    assert "declares streaming_input" -- a declaration that was never
    observed.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    eps = [
        EntryPoint(
            name="streamout/only",
            value="tests.test_cli:_output_only_stream_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    def _no_wire_format(engine: object) -> None:
        return None

    monkeypatch.setattr(cli, "_streaming_audio_format", _no_wire_format)

    real_create = registry.create

    def _create_with_broken_supports(name: str, /, *args: object, **kwargs: object) -> object:
        engine = real_create(name, *args, **kwargs)

        def _boom(dot_path: str) -> bool:
            if dot_path == "streaming_input":
                raise RuntimeError("capability tree exploded")
            return type(engine).supports(engine, dot_path)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        monkeypatch.setattr(engine, "supports", _boom)
        return engine

    monkeypatch.setattr(registry, "create", _create_with_broken_supports)

    exit_code = cli.main(["compliance", "run", "streamout/only", "--include-bridge"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "sync_bridge_no_wire_format" in output
    assert "sync_bridge_not_applicable" not in output
    # Honest attribution: supports() raised, so the declaration is
    # unverifiable -- the message must not claim the engine declares
    # streaming_input.
    assert "supports() raised" in output
    assert "engine declares streaming_input" not in output


def test_cli_compliance_run_bridge_timeout_parses_to_a_none_sentinel() -> None:
    """The PARSED default is a None sentinel, not the literal 5.0: only that
    distinguishes "user omitted the flag" from "user explicitly asked for 5 s",
    which _cmd_compliance_run needs to reject an explicit --bridge-timeout
    without --include-bridge instead of silently ignoring it. The documented
    5.0 s default is applied there (see the effective-default test below).
    """
    parser = cli.build_parser()
    args = parser.parse_args(["compliance", "run"])
    assert args.bridge_timeout is None
    assert parser.parse_args(["compliance", "run", "--bridge-timeout", "30"]).bridge_timeout == 30.0


def test_cli_compliance_run_bridge_timeout_without_include_bridge_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--bridge-timeout only means anything with --include-bridge. Accepting it
    alone would print "Compliance run passed" to an author who reads it as
    "the bridge ran within my budget" -- a silently inert flag. It is a usage
    error (exit 2) naming the flag that would make it effective.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    eps = [
        EntryPoint(
            name="stream/ok",
            value="tests.test_cli:_gating_stream_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    def _fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("the bridge must not run without --include-bridge")

    monkeypatch.setattr(cli, "check_sync_bridge", _fail_if_called)

    exit_code = cli.main(["compliance", "run", "stream/ok", "--bridge-timeout", "60"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "--include-bridge" in captured.err
    assert "Compliance run passed" not in captured.out


def test_cli_compliance_run_bridge_timeout_reaches_check_sync_bridge(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--bridge-timeout is threaded all the way to check_sync_bridge: without
    this the check's own advice ("re-run with a larger timeout") named a setting
    no CLI user could reach.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    eps = [
        EntryPoint(
            name="stream/ok",
            value="tests.test_cli:_gating_stream_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    seen: dict[str, object] = {}

    def _check_sync_bridge(
        factory: object,
        *,
        timeout: float = 5.0,
        model: str | None = None,
        engine: object = None,
    ) -> ComplianceReport:
        seen["timeout"] = timeout
        seen["model"] = model
        seen["engine"] = engine
        return ComplianceReport(registry=ModelRegistry({}), issues=[])

    monkeypatch.setattr(cli, "check_sync_bridge", _check_sync_bridge)

    exit_code = cli.main(
        ["compliance", "run", "stream/ok", "--include-bridge", "--bridge-timeout", "12.5"]
    )
    capsys.readouterr()

    assert exit_code == 0
    assert seen["timeout"] == 12.5
    # The model key travels with it: in a multi-model run an unattributed bridge
    # issue renders as <registry> and the user cannot tell which engine failed.
    assert seen["model"] == "stream/ok"
    # The engine travels too: without it the check cannot tell an output-only
    # engine's honest refusal from a streaming engine's capability lie, and
    # fails closed on both.
    assert isinstance(seen["engine"], _GatingStreamEngine)


def test_cli_compliance_run_without_bridge_timeout_uses_the_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Omitting the flag hands the library's DEFAULT_SYNC_BRIDGE_TIMEOUT down, not
    the None sentinel argparse now parses to (check_sync_bridge rejects a
    non-float). The default is READ from compliance, never re-literalled here,
    so --help and the applied budget can never advertise different numbers.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    eps = [
        EntryPoint(
            name="stream/ok",
            value="tests.test_cli:_gating_stream_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    seen: dict[str, object] = {}

    def _check_sync_bridge(
        factory: object,
        *,
        timeout: float = -1.0,
        model: str | None = None,
        engine: object = None,
    ) -> ComplianceReport:
        seen["timeout"] = timeout
        seen["model"] = model
        seen["engine"] = engine
        return ComplianceReport(registry=ModelRegistry({}), issues=[])

    monkeypatch.setattr(cli, "check_sync_bridge", _check_sync_bridge)

    exit_code = cli.main(["compliance", "run", "stream/ok", "--include-bridge"])
    capsys.readouterr()

    assert exit_code == 0
    assert seen["timeout"] == DEFAULT_SYNC_BRIDGE_TIMEOUT
    assert seen["model"] == "stream/ok"
    assert isinstance(seen["engine"], _GatingStreamEngine)


def test_cli_bridge_timeout_help_advertises_the_shared_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--help`` prints the constant's value, not a hand-copied literal.

    A drifted help string tells the author their engine got a budget it never
    got; the help text is built from ``DEFAULT_SYNC_BRIDGE_TIMEOUT`` itself.

        Args:
            capsys: Pytest fixture capturing stdout/stderr.
    """
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["compliance", "run", "--help"])
    assert excinfo.value.code == 0

    output = capsys.readouterr().out
    assert f"default: {DEFAULT_SYNC_BRIDGE_TIMEOUT}" in output


def test_cli_compliance_run_failure_headline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # When entry points fail, `compliance run` prints the failure headline and
    # exits 1 (covers the failed-entrypoints path).
    registry = ModelRegistry({})
    report = ComplianceReport(
        registry=registry,
        issues=[
            ComplianceIssue(
                level="error", code="no_entrypoints", message="No entry points.", model=None
            )
        ],
    )
    _patch_discover(monkeypatch, registry)
    _patch_check_entrypoints(monkeypatch, report)

    exit_code = cli.main(["compliance", "run"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Entry point compliance checks failed" in output
    assert "Compliance run failed" in output


def test_cli_compliance_run_quiet_suppresses_warnings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # --quiet suppresses warning emission (covers the not-quiet false branch).
    registry = ModelRegistry({})
    report = ComplianceReport(
        registry=registry,
        issues=[
            ComplianceIssue(
                level="warning", code="demo_warning", message="a warning", model="stream/ok"
            )
        ],
    )
    _patch_discover(monkeypatch, registry)
    _patch_check_entrypoints(monkeypatch, report)

    exit_code = cli.main(["compliance", "run", "--quiet"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "a warning" not in output


def test_cli_compliance_run_unknown_model_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An explicitly named model that is not in the registry is reported as an
    # error for that name, not an unhandled crash.
    registry = _compliant_dummy_registry()
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["compliance", "run", "does/not-exist"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "unknown model" in output


def test_cli_compliance_run_configuration_required_is_skipped_not_failed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A zero-arg factory raising ConfigurationRequiredError is a SKIP.

    check_entrypoints classifies exactly this state (required credential
    absent from the environment; from_env raises the narrow subtype
    automatically) as a factory_requires_config WARNING skip; the instance
    layer failing the same engine in the same `compliance run` gave one
    engine two contradictory verdicts in one command. The verdict must not
    depend on which layer looked first.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    from standard_asr.contract.exceptions import ConfigurationRequiredError

    registry = _compliant_dummy_registry()

    def _boom(_name: str, /, *args: object, **kwargs: object) -> object:
        raise ConfigurationRequiredError("missing credential")

    _patch_discover(monkeypatch, registry)
    monkeypatch.setattr(registry, "create", _boom)

    exit_code = cli.main(["compliance", "run", "stream/ok"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "skipped instance checks" in output
    assert "missing credential" in output
    assert "could not construct engine" not in output


def test_cli_compliance_run_plain_config_error_still_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A plain ConfigError (invalid config, NOT absence) stays a failure.

    The skip is scoped to the narrow ConfigurationRequiredError: waiving
    every ConfigError would let a broken plugin (an inconsistent declaration,
    an invalid supplied value, a wrapped construction ValidationError) read
    as green-with-warning -- a false clean verdict on a defective engine.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    from standard_asr.contract.exceptions import ConfigError

    registry = _compliant_dummy_registry()

    def _boom(_name: str, /, *args: object, **kwargs: object) -> object:
        raise ConfigError("engine declaration is internally inconsistent")

    _patch_discover(monkeypatch, registry)
    monkeypatch.setattr(registry, "create", _boom)

    exit_code = cli.main(["compliance", "run", "stream/ok"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "could not construct engine" in output
    assert "skipped instance checks" not in output


def test_cli_compliance_run_non_config_construction_error_still_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The skip is scoped to ConfigurationRequiredError (absent configuration)
    ONLY: a factory failing for any other reason is a broken plugin and
    stays a per-model error rather than aborting the whole run.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    registry = _compliant_dummy_registry()

    def _boom(_name: str, /, *args: object, **kwargs: object) -> object:
        raise ValueError("factory exploded")

    _patch_discover(monkeypatch, registry)
    monkeypatch.setattr(registry, "create", _boom)

    exit_code = cli.main(["compliance", "run", "stream/ok"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "could not construct engine" in output


def test_cli_compliance_run_named_subset_scopes_probes_at_the_source(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `compliance run <named>` on a machine with a co-installed plugin must
    # scope the per-engine checks AT THE SOURCE, not filter the report
    # afterwards: the instance checks execute engine code (construction, the
    # supports() sweep, the start_transcription() refusal probe -- a model
    # load; for a cloud engine a billable call), and the old post-hoc filter
    # discarded only the verdicts while the user still paid those side
    # effects on an engine they never named.
    eps = [
        EntryPoint(
            name="stream/ok",
            value="tests.test_cli:_gating_stream_factory",
            group="standard_asr.models",
        ),
        EntryPoint(
            name="other/model",
            value="tests.test_cli:_other_recording_factory",
            group="standard_asr.models",
        ),
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)
    _unnamed_probe_constructions.clear()

    exit_code = cli.main(["compliance", "run", "stream/ok"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Compliance run passed" in output
    # The engine the user never named was never constructed, so none of its
    # instance probes ran either.
    assert _unnamed_probe_constructions == []

    # Unnamed run: both engines are checked (the scope is the user's choice,
    # not a new default narrowing).
    exit_code = cli.main(["compliance", "run"])
    capsys.readouterr()
    assert exit_code == 0
    assert _unnamed_probe_constructions != []
    assert "alpha/first" not in output
    assert "Compliance run passed" in output


def test_cli_compliance_run_named_model_still_reports_global_collision(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # guard: scoping MUST NOT silence a registry-global invariant. An
    # engine_id collision is keyed by a bare engine_id in shadowed_engine_ids and is
    # kept even for a named run (a naive `model in named` filter would drop it).
    base = _compliant_dummy_registry()
    registry = ModelRegistry({k: base.spec(k) for k in base.names()}, shadowed_engine_ids={"zeta"})
    _patch_discover(monkeypatch, registry)
    crafted = ComplianceReport(
        registry=None,
        issues=[
            ComplianceIssue(
                level="error",
                code="engine_id_collision",
                message="engine_id 'zeta' is shadowed by more than one distribution",
                model="zeta",
            )
        ],
    )
    _patch_check_entrypoints(monkeypatch, crafted)

    exit_code = cli.main(["compliance", "run", "stream/ok"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "zeta" in output


def test_cli_compliance_run_broken_wire_format_does_not_abort_the_whole_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A raising ``recommended_wire_format()`` is a per-model report, not an abort.

    The bridge runner derives its bare-frame format from the engine; an
    exception there used to escape ``_run_instance_checks`` and take down a
    multi-model run, so ONE broken plugin hid every other plugin's verdict.
    It is now contained as a ``sync_bridge_setup_failed`` error for that model
    while the co-installed model's checks still report.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            capsys: Pytest fixture capturing stdout/stderr.
    """
    eps = [
        EntryPoint(
            name="brokenwire/engine",
            value="tests.test_cli:_broken_wire_factory",
            group="standard_asr.models",
        ),
        EntryPoint(
            name="stream/ok",
            value="tests.test_cli:_gating_stream_factory",
            group="standard_asr.models",
        ),
    ]
    registry = discover_models(eps=eps, strict=True)
    _patch_discover(monkeypatch, registry)

    bridged: list[object] = []
    real_check_sync_bridge = cli.check_sync_bridge

    def _spy_bridge(factory: object, **kwargs: object) -> ComplianceReport:
        bridged.append(kwargs.get("model"))
        return real_check_sync_bridge(
            cast(Callable[[], TranscriptionSession], factory),
            **cast(Any, kwargs),
        )

    monkeypatch.setattr(cli, "check_sync_bridge", _spy_bridge)

    exit_code = cli.main(["compliance", "run", "--include-bridge"])
    output = capsys.readouterr().out

    # The run COMPLETES (no traceback escaping) and fails on the broken model.
    assert exit_code == 1
    assert "Compliance run failed" in output
    assert "[FAIL] Error brokenwire/engine [sync_bridge_setup_failed]:" in output
    assert "recommended_wire_format() raised" in output
    # The same fault is ALSO reported by the dedicated wire-format check, so the
    # guard duplicates a verdict rather than inventing one.
    assert "recommended_wire_format_raised" in output
    # The healthy co-installed model was still bridged for real (the broken one
    # never reaches the check -- there is no format to bridge with), and its
    # verdict is clean, so nothing is attributed to it.
    assert bridged == ["stream/ok"]
    assert "stream/ok [sync_bridge" not in output


def test_run_sync_bridge_broken_wire_format_is_a_contained_error() -> None:
    """Unit-level companion to the run-level test above: the helper returns the
    report instead of propagating, and names the engine's own exception.
    """
    report = cli._run_sync_bridge(_BrokenWireEngine(), "brokenwire/engine")  # pyright: ignore[reportPrivateUsage]

    assert report.passed is False
    assert [(i.level, i.code) for i in report.issues] == [("error", "sync_bridge_setup_failed")]
    assert "wire format derivation exploded" in report.issues[0].message
    assert report.issues[0].model == "brokenwire/engine"


class _CliFakeAwaitable:
    """An awaitable that is not a coroutine (nothing to close)."""

    def __await__(self) -> Any:  # pragma: no cover - never actually awaited
        yield


def test_engine_supports_fails_closed_on_malformed_answers() -> None:
    """The CLI pre-gate counts ONLY a literal True as supported.

    bool() coercion silently promoted a truthy coroutine or a "false" string
    to a capability verdict (and leaked the coroutine unawaited -- the
    suite's warnings-as-errors policy is the leak oracle). The compliance
    checks report the underlying defect; this pre-gate's job is to not act
    on a lie and not leak.
    """

    class _Answers:
        def __init__(self, value: Any) -> None:
            self._value = value

        def supports(self, dot_path: str) -> Any:
            return self._value

    async def _coro() -> bool:
        return True  # pragma: no cover - never awaited by design

    assert cli._engine_supports(_Answers(True), "x") is True  # pyright: ignore[reportPrivateUsage]
    assert cli._engine_supports(_Answers(False), "x") is False  # pyright: ignore[reportPrivateUsage]
    assert cli._engine_supports(_Answers("false"), "x") is False  # pyright: ignore[reportPrivateUsage]
    assert cli._engine_supports(_Answers(1), "x") is False  # pyright: ignore[reportPrivateUsage]
    assert cli._engine_supports(_Answers(_coro()), "x") is False  # pyright: ignore[reportPrivateUsage]
    assert cli._engine_supports(_Answers(_CliFakeAwaitable()), "x") is False  # pyright: ignore[reportPrivateUsage]


def test_run_sync_bridge_async_wire_format_is_modality_error() -> None:
    """An async recommended_wire_format hands the bridge an awaitable instead
    of an AudioFormat: reported under the same code the compliance layer
    uses, with the stray coroutine closed (warnings-as-errors would fail
    this test on a leak), never fed into a session open.
    """

    class _AsyncWireEngine(_GatingStreamEngine):
        async def recommended_wire_format(self) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
            raise AssertionError("never awaited")  # pragma: no cover

    report = cli._run_sync_bridge(_AsyncWireEngine(), "stream/ok")  # pyright: ignore[reportPrivateUsage]
    assert report.passed is False
    assert [i.code for i in report.issues] == ["protocol_member_not_synchronous"]
    assert "SYNCHRONOUS" in report.issues[0].message


def test_run_sync_bridge_non_coroutine_awaitable_wire_format_same_error() -> None:
    """The non-coroutine awaitable shape (nothing to close) draws the same
    verdict.
    """

    class _AwaitableWireEngine(_GatingStreamEngine):
        def recommended_wire_format(self) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
            return _CliFakeAwaitable()

    report = cli._run_sync_bridge(_AwaitableWireEngine(), "stream/ok")  # pyright: ignore[reportPrivateUsage]
    assert report.passed is False
    assert [i.code for i in report.issues] == ["protocol_member_not_synchronous"]


def test_run_sync_bridge_wrong_typed_wire_format_never_reaches_the_engine() -> None:
    """A non-``AudioFormat`` recommendation stops at the bridge boundary.

    This is a REAL consumer: pre-fix only the awaitable shape was caught, so
    a str/dict/duck recommendation was handed straight into
    ``start_transcription(audio_format=...)`` -- potentially triggering model
    loads or a secondary crash misblamed on the bridge. The value's own text
    is never echoed (type name only).
    """
    opened: list[object] = []

    class _StrWireEngine(_GatingStreamEngine):
        def recommended_wire_format(self) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
            return "pcm_s16le;sk-WIRE-SECRET"

        def start_transcription(  # pyright: ignore[reportIncompatibleMethodOverride]
            self, **kwargs: Any
        ) -> Any:
            opened.append(kwargs)  # pragma: no cover - must never be reached
            raise AssertionError("engine must not be opened")  # pragma: no cover

    report = cli._run_sync_bridge(_StrWireEngine(), "stream/ok")  # pyright: ignore[reportPrivateUsage]
    assert report.passed is False
    assert [i.code for i in report.issues] == ["protocol_member_wrong_return_type"]
    assert "returned str, not AudioFormat / None" in report.issues[0].message
    assert "sk-WIRE-SECRET" not in report.issues[0].message
    assert opened == []


def test_run_sync_bridge_async_start_transcription_is_invalid_session() -> None:
    """The CLI bridge path reports an async opener at the factory boundary.

    The CLI's canonical factory wraps start_transcription; an `async def`
    opener used to hand check_sync_bridge a coroutine that was stored as the
    session, misreported as a bridge lifecycle fault deep in SyncSession, and
    leaked unawaited (warnings-as-errors would fail this test on the leak).
    """

    class _AsyncStartEngine(_GatingStreamEngine):
        async def start_transcription(  # pyright: ignore[reportIncompatibleMethodOverride]
            self,
            *,
            audio_format: Any = None,
            params: Any = None,
            audio: Any = None,
            deadlines: Any = None,
        ) -> Any:
            raise AssertionError("never awaited")  # pragma: no cover

    report = cli._run_sync_bridge(_AsyncStartEngine(), "stream/ok", timeout=5.0)  # pyright: ignore[reportPrivateUsage]
    assert report.passed is False
    assert [i.code for i in report.issues] == ["sync_bridge_invalid_session"]
    assert not any("did not terminate" in i.message for i in report.issues)


def test_run_sync_bridge_awaitable_supports_is_unverifiable_hard_error() -> None:
    """No usable wire format + supports() answering an awaitable: the
    declaration is unverifiable -- fail-closed hard error with an honest
    message (never a fabricated "declares streaming_input" from a truthy
    coroutine, never a not-applicable pass).
    """

    class _NoFormatAsyncSupportsEngine(_GatingStreamEngine):
        def recommended_wire_format(self) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
            return None

        def supports(self, dot_path: str) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
            async def _answer() -> bool:
                return False  # pragma: no cover - never awaited by design

            return _answer()

    report = cli._run_sync_bridge(_NoFormatAsyncSupportsEngine(), "stream/ok")  # pyright: ignore[reportPrivateUsage]
    assert report.passed is False
    assert [i.code for i in report.issues] == ["sync_bridge_no_wire_format"]
    assert "non-boolean/awaitable" in report.issues[0].message

    class _NoFormatAwaitableSupportsEngine(_NoFormatAsyncSupportsEngine):
        def supports(self, dot_path: str) -> Any:
            return _CliFakeAwaitable()  # non-coroutine awaitable: nothing to close

    report = cli._run_sync_bridge(_NoFormatAwaitableSupportsEngine(), "stream/ok")  # pyright: ignore[reportPrivateUsage]
    assert report.passed is False
    assert [i.code for i in report.issues] == ["sync_bridge_no_wire_format"]


def test_run_sync_bridge_non_bool_supports_is_unverifiable_hard_error() -> None:
    """Same fail-closed arm for a truthy non-bool answer."""

    class _NoFormatStringSupportsEngine(_GatingStreamEngine):
        def recommended_wire_format(self) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
            return None

        def supports(self, dot_path: str) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
            return "false"

    report = cli._run_sync_bridge(_NoFormatStringSupportsEngine(), "stream/ok")  # pyright: ignore[reportPrivateUsage]
    assert report.passed is False
    assert [i.code for i in report.issues] == ["sync_bridge_no_wire_format"]


def test_run_sync_bridge_no_wire_format_is_error() -> None:
    # A streaming engine that declares no usable wire sample rate cannot be bridged
    # from the CLI (no bare-frame format to open with); reported as an error, not a
    # crash. (Missing wire_encodings is NOT this case -- it falls back to pcm_s16le.)
    class _NoRateEngine(_GatingStreamEngine):
        properties: ClassVar[BaseProperties] = _StreamOkProps.model_construct(
            wire_encodings=["pcm_s16le"], native_sample_rate=0, required_input_sample_rate=None
        )

    report = cli._run_sync_bridge(_NoRateEngine(), "stream/ok")  # pyright: ignore[reportPrivateUsage]
    assert report.passed is False
    assert any("no usable" in i.message for i in report.issues)


def test_spec_is_zero_arg_handles_unloadable_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _spec_is_zero_arg returns False when the factory cannot be loaded or its
    # signature cannot be read, never raising.
    from standard_asr.contract.exceptions import FactoryLoadError

    class _Spec:
        def load_factory(self) -> object:
            raise FactoryLoadError("nope")

    assert cli._spec_is_zero_arg(_Spec()) is False  # pyright: ignore[reportPrivateUsage,reportArgumentType]

    class _SpecBadSig:
        def load_factory(self) -> object:
            # A builtin whose signature cannot be introspected (raises ValueError).
            return type

    assert cli._spec_is_zero_arg(_SpecBadSig()) is False  # pyright: ignore[reportPrivateUsage,reportArgumentType]


def test_engine_supports_defensive() -> None:
    # _engine_supports is fail-closed: no supports() method, or one that raises,
    # both yield False.
    class _NoSupports:
        pass

    assert cli._engine_supports(_NoSupports(), "streaming_input") is False  # pyright: ignore[reportPrivateUsage]

    class _RaisingSupports:
        def supports(self, dot_path: str) -> bool:
            raise RuntimeError("boom")

    assert cli._engine_supports(_RaisingSupports(), "streaming_input") is False  # pyright: ignore[reportPrivateUsage]


def test_ensure_utf8_stream_tolerates_reconfigure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _ensure_utf8_stream must not crash when reconfigure() raises (for example, a
    # detached buffer); it leaves the stream as-is.
    import io

    class _BadReconfigure(io.StringIO):
        encoding = "cp1252"

        def reconfigure(self, **_: object) -> None:  # type: ignore[override]
            raise io.UnsupportedOperation("cannot reconfigure")

    # Must return without raising.
    cli._ensure_utf8_stream(_BadReconfigure())  # pyright: ignore[reportPrivateUsage,reportArgumentType]


@pytest.mark.parametrize(
    ("factory", "marker"),
    [
        ("_runtime_error_factory", "SDK initialization failed"),
        ("_type_error_factory", "bad SDK signature"),
        ("_os_error_factory", "model directory unreadable"),
        ("_authored_error_factory", "plugin said no"),
    ],
)
def test_cli_compliance_run_contains_any_factory_fault_per_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    factory: str,
    marker: str,
) -> None:
    # `compliance run`'s contract is a verdict for EVERY model in one run
    # (G2.1). registry.create wraps only a construction-time ValidationError,
    # so a factory's own RuntimeError/TypeError/OSError/authored type
    # propagates verbatim -- and a named-types catch let exactly those abort
    # the loop, denying every LATER model its verdict.
    eps = [
        EntryPoint(
            name="broken/first",
            value=f"tests.test_cli:{factory}",
            group="standard_asr.models",
        ),
        EntryPoint(
            name="later/gating",
            value="tests.test_cli:_ungated_stream_factory",
            group="standard_asr.models",
        ),
    ]
    _patch_discover(monkeypatch, discover_models(eps=eps, strict=True))

    exit_code = cli.main(["compliance", "run"])
    captured = capsys.readouterr()
    output = captured.out + captured.err

    assert exit_code == 1
    # The broken model is reported, with its own reason...
    assert "broken/first [engine_construction_failed]" in output
    assert marker in output
    # ...and the LATER model was still reached and judged on its own merits
    # (its gating defect is a verdict only the loop's continuation produces).
    assert "gating_strict_accepted" in output
    assert "Compliance run failed" in output


def test_cli_compliance_run_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The containment is `except Exception`, never BaseException: the
    # operator's own control flow must still stop the run.
    eps = [
        EntryPoint(
            name="interrupt/first",
            value="tests.test_cli:_keyboard_interrupt_factory",
            group="standard_asr.models",
        )
    ]
    _patch_discover(monkeypatch, discover_models(eps=eps, strict=True))

    with pytest.raises(KeyboardInterrupt):
        cli.main(["compliance", "run"])


def test_cli_compliance_run_contains_a_crashing_check_implementation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Defense in depth behind the construction arm: if a CHECK itself falls
    # over on a shape nobody anticipated, that is still one model's verdict,
    # reported under its own code so it stays distinguishable from a plugin
    # construction fault.
    eps = [
        EntryPoint(
            name="stream/ok",
            value="tests.test_cli:_gating_stream_factory",
            group="standard_asr.models",
        ),
        EntryPoint(
            name="stream/two",
            value="tests.test_cli:_gating_stream_factory",
            group="standard_asr.models",
        ),
    ]
    _patch_discover(monkeypatch, discover_models(eps=eps, strict=True))

    real_checks = cli._run_instance_checks  # pyright: ignore[reportPrivateUsage]

    def _crash_on_first(
        registry: ModelRegistry, name: str, **kwargs: object
    ) -> list[ComplianceReport]:
        if name == "stream/ok":
            raise RuntimeError("check implementation exploded")
        return real_checks(registry, name, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(cli, "_run_instance_checks", _crash_on_first)

    exit_code = cli.main(["compliance", "run"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "compliance_check_crashed" in output
    assert "check implementation exploded" in output
    # The second model still ran.
    assert "stream/two" in output
