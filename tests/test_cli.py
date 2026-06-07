"""CLI coverage for Standard ASR entrypoint tooling."""

from __future__ import annotations

import sys
import types
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from standard_asr import cli
from standard_asr.compliance import ComplianceIssue, ComplianceReport
from standard_asr.discovery import ModelRegistry, discover_models
from standard_asr.exceptions import (
    AudioProcessingError,
    EntrypointValidationError,
    TranscriptionError,
)


def _demo_registry() -> ModelRegistry:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    return discover_models(eps=eps, strict=True)


def test_cli_models_list(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = _demo_registry()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["models", "list"])
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

    exit_code = cli.main(["models", "list"])
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

    exit_code = cli.main(["models", "show", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Engine ID" in output
    assert "alpha/first" in output
    # §264: models show MUST surface DeclaredCapabilities (no instantiation).
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

    exit_code = cli.main(["models", "show", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Capabilities: <unavailable" in output


def test_cli_models_cache(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "resolve_cache_dir", lambda: tmp_path)

    exit_code = cli.main(["models", "cache"])
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


def test_cli_models_list_entrypoint_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _discover_models(**_: object) -> ModelRegistry:
        raise EntrypointValidationError("bad entrypoint")

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["models", "list"])
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

    monkeypatch.setattr(cli, "_cmd_models_list", _raise)
    monkeypatch.setattr(cli.traceback, "print_exc", _print_exc)

    exit_code = cli.main(["--debug", "models", "list"])

    assert exit_code == 1
    assert called["traceback"] is True


def test_cli_generic_exception_no_debug(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise(_: object) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_cmd_models_list", _raise)

    exit_code = cli.main(["models", "list"])
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
    module = types.ModuleType("standard_asr.server")
    called: dict[str, object] = {}

    def _run(**kwargs: object) -> None:
        called.update(dict(kwargs))

    setattr(module, "run", _run)

    monkeypatch.setitem(__import__("sys").modules, "standard_asr.server", module)

    exit_code = cli.main(["serve", "--host", "0.0.0.0", "--port", "9001"])

    assert exit_code == 0
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 9001


def test_cli_serve_missing_server_dependency(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Simulate the server module failing to import (deterministic, no import
    # warnings): a None entry in sys.modules makes `from .server import run`
    # raise ImportError, exercising the missing-server-deps branch.
    monkeypatch.setitem(sys.modules, "standard_asr.server", None)

    exit_code = cli.main(["serve"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "dependencies are missing" in output


def test_cli_serve_importerror_from_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = types.ModuleType("standard_asr.server")

    def _run(**_: object) -> None:
        raise ImportError("boom")

    setattr(module, "run", _run)
    monkeypatch.setitem(__import__("sys").modules, "standard_asr.server", module)

    exit_code = cli.main(["serve"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "boom" in output


def test_cli_models_prepare_calls_transcribe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = _demo_registry()

    def _discover_models(**_: object) -> ModelRegistry:
        return registry

    monkeypatch.setattr(cli, "discover_models", _discover_models)

    exit_code = cli.main(["models", "prepare", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "prepare" in output.lower()


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

    exit_code = cli.main(["models", "prepare", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert prep.called is True
    assert "prepare" in output.lower()


def test_parse_options() -> None:
    from standard_asr.runtime_params import RuntimeParams

    params = cli._parse_options('{"language": "en"}')  # pyright: ignore[reportPrivateUsage]
    assert isinstance(params, RuntimeParams)
    assert params.language == "en"

    with pytest.raises(ValueError):
        cli._parse_options("[1, 2, 3]")  # pyright: ignore[reportPrivateUsage]
