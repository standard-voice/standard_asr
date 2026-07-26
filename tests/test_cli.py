# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""CLI coverage for Standard ASR entrypoint tooling."""

from __future__ import annotations

import argparse
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
from standard_asr.contract.capabilities import (
    DeclaredCapabilities,
    FlagCap,
    StreamingCapabilities,
)
from standard_asr.contract.exceptions import (
    AudioProcessingError,
    ConfigError,
    EntrypointValidationError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from standard_asr.engine import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    InputKind,
    PreparedAudio,
    SampleRateRange,
)
from standard_asr.plugins.discovery import ModelRegistry, discover_models
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

    assert exit_code == 0
    assert "Capabilities: <unavailable" in output


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
    model_name: str = "ok"  # model_id == 'stream/ok'
    protocol_version: str = "0.2.0"
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
    protocol_version: str = "0.2.0"
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


def _gating_stream_factory() -> _GatingStreamEngine:  # pyright: ignore[reportUnusedFunction]
    return _GatingStreamEngine()


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
    # Every discovery-facing subcommand is uniform: --strict-discovery makes
    # discovery FAIL on an invalid entry point instead of skipping it, and its
    # absence keeps the lenient default. Captured at the discovery call so the
    # flag cannot be parsed-but-dropped.
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
        ["transcribe", "alpha/first", "dummy.wav", "--strict"],
        ["compliance", "entrypoints", "--strict"],
        ["compliance", "run", "--strict"],
    ],
)
def test_cli_bare_strict_flag_is_rejected(argv: list[str]) -> None:
    """A bare ``--strict`` is a usage error (exit 2), never an abbreviation.

    ``strict`` alone already names a DIFFERENT knob: the engine's
    strict/best_effort PARAMETER-gating policy, an init-config field set via
    ``--set strict=...``. The discovery flag is therefore spelled
    ``--strict-discovery``, and argparse prefix abbreviation is disabled
    (``allow_abbrev=False``) so ``--strict`` cannot silently resolve to it and
    resurrect exactly the confusion the rename exists to prevent.
    """
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(argv)
    assert excinfo.value.code == 2


def test_cli_does_not_abbreviate_long_flags() -> None:
    # allow_abbrev=False is set on the top-level parser AND every subparser: an
    # abbreviation a user scripts today can turn ambiguous the day a new flag
    # lands, which would be a silent behavior change.
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
    caller mistake (e.g. a strict non-detectable candidate language) as an
    internal failure.
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

    assert exit_code == 2
    assert secret not in captured.err
    assert secret not in captured.out


def test_cli_serve_doc_does_not_list_unparsed_reload_flag() -> None:
    # removed --reload from the serve parser; a later doc commit
    # re-listed it, promising a flag the CLI rejects with SystemExit(2). Lock the doc
    # and parser together: the parser rejects --reload, so cli.md must omit it.
    import pathlib

    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--reload"])  # parser intentionally has no --reload

    cli_md = pathlib.Path(__file__).resolve().parents[1] / "docs" / "spec" / "cli.md"
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
        "transcribe": parser.parse_args(["transcribe", "e/m", "a.wav"]).func.__name__,
        "serve": parser.parse_args(["serve"]).func.__name__,
        "doctor": parser.parse_args(["doctor"]).func.__name__,
    }
    assert resolved == {
        "list": "_cmd_list",
        "show": "_cmd_show",
        "cache": "_cmd_cache",
        "prepare": "_cmd_prepare",
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

    assert exit_code == 2
    assert "coroutine" in captured.err
    assert "complete" not in captured.out.lower()


def test_cli_models_prepare_rejects_non_callable_hook(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A non-callable 'prepare' attribute is a declaration bug, not a
    # "no hook" case -- reject it loudly.
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

    assert exit_code == 2
    assert "non-callable" in captured.err


def test_cli_models_prepare_rejects_required_args_hook(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The warm-up hook MUST be invocable with no arguments. A prepare()
    # that requires a parameter can never be driven by the CLI -- reject it with a
    # structured error rather than letting the call blow up with a bare TypeError
    # (mirrors the compliance suite's 'prepare_hook_requires_args').
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

    assert exit_code == 2
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
    # A stream without reconfigure() (e.g. a plain StringIO) must not crash.
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
    caps = json.loads(output.split("Capabilities:", 1)[1])
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
    # the final `except Exception` branch honoured --debug.
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
    # _GatingStreamEngine is a fully-compliant EngineBase subclass (passes
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
    # `compliance run` reports the entry point result AND points the
    # author at the event-sequence check the CLI cannot synthesize.
    registry = _compliant_dummy_registry()
    _patch_discover(monkeypatch, registry)

    exit_code = cli.main(["compliance", "run", "stream/ok"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Entry point compliance checks passed" in output
    assert "event-sequence" in output
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


def test_parse_init_config_merges_config_under_set() -> None:
    # --config supplies a base; --set overrides/adds (and wins). --set values
    # stay strings (the engine's pydantic config coerces them, like the env path).
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
    # No --config / --set -> empty mapping (create() called with no init config).
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
    # An engine whose factory needs arguments (e.g. credentials)
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
    message, visible to machine consumers and honouring ``--quiet``.
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
    # the classification below is the shipped behaviour, not a stub's.
    constructed: list[object] = []
    real_swap_check = cli.check_provider_params_swap_safety
    real_check_sync_bridge = cli.check_sync_bridge
    bridge_kwargs: list[dict[str, object]] = []

    def _spy_swap(engine: object) -> ComplianceReport:
        constructed.append(engine)
        return real_swap_check(cast(EngineBase, engine))

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


@pytest.mark.parametrize("bad", ["0", "-1", "-0.5", "nan", "inf", "-inf", "abc"])
def test_cli_bridge_timeout_rejects_nonpositive_and_nonfinite(bad: str) -> None:
    """--bridge-timeout must be a finite number of seconds > 0.

    A bare ``type=float`` would accept these; fed into ``Thread.join`` they
    become an immediately-expiring timeout (a false "did not terminate"
    verdict blamed on the engine) or a hang (``inf``). argparse rejects them
    as usage errors (exit 2) at parse time.
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
    """
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        cli._positive_finite_seconds(str(bad))  # pyright: ignore[reportPrivateUsage]


def test_cli_bridge_timeout_rejects_a_non_numeric_value_before_the_rule() -> None:
    # A non-float never reaches validate_bridge_timeout: the conversion error is
    # its own message, naming the offending text.
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


def test_cli_no_wire_format_with_broken_supports_keeps_the_hard_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A raising supports() cannot buy the not-applicable pass here either.

    The no-wire-format branch takes the passing not-applicable verdict only
    when the engine VERIFIABLY does not declare streaming_input; an engine
    whose supports() raises is unverifiable and keeps the hard error --
    mirroring check_sync_bridge's own fail-closed stance.
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


def test_cli_compliance_run_bridge_timeout_parses_to_a_none_sentinel() -> None:
    # The PARSED default is a None sentinel, not the literal 5.0: only that
    # distinguishes "user omitted the flag" from "user explicitly asked for 5 s",
    # which _cmd_compliance_run needs to reject an explicit --bridge-timeout
    # without --include-bridge instead of silently ignoring it. The documented
    # 5.0 s default is applied there (see the effective-default test below).
    parser = cli.build_parser()
    args = parser.parse_args(["compliance", "run"])
    assert args.bridge_timeout is None
    assert parser.parse_args(["compliance", "run", "--bridge-timeout", "30"]).bridge_timeout == 30.0


def test_cli_compliance_run_bridge_timeout_without_include_bridge_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # --bridge-timeout only means anything with --include-bridge. Accepting it
    # alone would print "Compliance run passed" to an author who reads it as
    # "the bridge ran within my budget" -- a silently inert flag. It is a usage
    # error (exit 2) naming the flag that would make it effective.
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
    # --bridge-timeout is threaded all the way to check_sync_bridge: without
    # this the check's own advice ("re-run with a larger timeout") named a knob
    # no CLI user could reach.
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
    # Omitting the flag hands the library's DEFAULT_SYNC_BRIDGE_TIMEOUT down, not
    # the None sentinel argparse now parses to (check_sync_bridge rejects a
    # non-float). The default is READ from compliance, never re-literalled here,
    # so --help and the applied budget can never advertise different numbers.
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


def test_cli_compliance_run_construction_error_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A zero-arg engine whose construction raises a client-side config error is
    # reported for that model rather than aborting the whole run.
    from standard_asr.contract.exceptions import ConfigError

    registry = _compliant_dummy_registry()

    def _boom(_name: str, /, *args: object, **kwargs: object) -> object:
        raise ConfigError("missing credential")

    _patch_discover(monkeypatch, registry)
    monkeypatch.setattr(registry, "create", _boom)

    exit_code = cli.main(["compliance", "run", "stream/ok"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "could not construct engine" in output


def test_scope_entrypoints_report_keeps_named_global_and_collision() -> None:
    # Scoping an entry-point report to a named subset keeps (a) the named
    # models, (b) registry-global invariants (model is None), and (c) engine_id
    # collisions (shadowed_engine_ids); it drops an unrelated co-installed plugin's
    # per-engine issue so it cannot fail a named run.
    base = _compliant_dummy_registry()
    registry = ModelRegistry({k: base.spec(k) for k in base.names()}, shadowed_engine_ids={"zeta"})
    report = ComplianceReport(
        registry=None,
        issues=[
            ComplianceIssue(level="error", code="a", message="m", model="stream/ok"),
            ComplianceIssue(level="error", code="b", message="m", model="alpha/first"),
            ComplianceIssue(level="error", code="c", message="m", model=None),
            ComplianceIssue(level="error", code="engine_id_collision", message="m", model="zeta"),
        ],
    )
    scoped = cli._scope_entrypoints_report(  # pyright: ignore[reportPrivateUsage]
        report, registry, {"stream/ok"}
    )
    assert {issue.model for issue in scoped.issues} == {"stream/ok", None, "zeta"}


def test_cli_compliance_run_named_model_ignores_unrelated_plugin_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A named run must NOT fail because an UNRELATED co-installed plugin has
    # a per-engine entry-point problem; the headline and exit code reflect only the
    # named subset plus registry-global invariants.
    registry = _compliant_dummy_registry()
    _patch_discover(monkeypatch, registry)
    crafted = ComplianceReport(
        registry=None,
        issues=[
            ComplianceIssue(
                level="error",
                code="provider_params_type_not_closed",
                message="alpha/first is broken",
                model="alpha/first",
            )
        ],
    )
    _patch_check_entrypoints(monkeypatch, crafted)

    exit_code = cli.main(["compliance", "run", "stream/ok"])
    output = capsys.readouterr().out

    assert exit_code == 0
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
    # Unit-level companion to the run-level test above: the helper returns the
    # report instead of propagating, and names the engine's own exception.
    report = cli._run_sync_bridge(_BrokenWireEngine(), "brokenwire/engine")  # pyright: ignore[reportPrivateUsage]

    assert report.passed is False
    assert [(i.level, i.code) for i in report.issues] == [("error", "sync_bridge_setup_failed")]
    assert "wire format derivation exploded" in report.issues[0].message
    assert report.issues[0].model == "brokenwire/engine"


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


def test_streaming_audio_format_helpers() -> None:
    # _streaming_audio_format delegates to EngineBase.recommended_wire_format: it
    # builds a valid AudioFormat from declared encodings + sample rate, falls back
    # to canonical pcm_s16le when wire_encodings is unconstrained, and returns None
    # only when no usable sample rate is declared.
    engine = _GatingStreamEngine()
    fmt = cli._streaming_audio_format(engine)  # pyright: ignore[reportPrivateUsage]
    assert fmt is not None
    assert fmt.encoding == "pcm_s16le"
    assert fmt.sample_rate == 16000

    class _NoWireProps(_StreamOkProps):
        wire_encodings: list[str] | None = None

    class _NoWire(_GatingStreamEngine):
        properties: ClassVar[BaseProperties] = _NoWireProps()

    # Unconstrained wire_encodings -> canonical pcm_s16le fallback (NOT None: the
    # engine accepts any encoding, so a bare-frame session can still open).
    no_wire_fmt = cli._streaming_audio_format(_NoWire())  # pyright: ignore[reportPrivateUsage]
    assert no_wire_fmt is not None
    assert no_wire_fmt.encoding == "pcm_s16le"

    class _BadRate(_GatingStreamEngine):
        properties: ClassVar[BaseProperties] = _StreamOkProps.model_construct(
            wire_encodings=["pcm_s16le"], native_sample_rate=0, required_input_sample_rate=None
        )

    assert cli._streaming_audio_format(_BadRate()) is None  # pyright: ignore[reportPrivateUsage]


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
    # _ensure_utf8_stream must not crash when reconfigure() raises (e.g. a
    # detached buffer); it leaves the stream as-is.
    import io

    class _BadReconfigure(io.StringIO):
        encoding = "cp1252"

        def reconfigure(self, **_: object) -> None:  # type: ignore[override]
            raise io.UnsupportedOperation("cannot reconfigure")

    # Must return without raising.
    cli._ensure_utf8_stream(_BadReconfigure())  # pyright: ignore[reportPrivateUsage,reportArgumentType]
