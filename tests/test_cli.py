"""CLI coverage for Standard ASR entrypoint tooling."""

from __future__ import annotations

import types
from importlib.metadata import EntryPoint
from pathlib import Path

import numpy as np
import pytest

from standard_asr import cli
from standard_asr.compliance import ComplianceIssue, ComplianceReport
from standard_asr.discovery import ModelRegistry, discover_models


def _demo_registry() -> ModelRegistry:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    return discover_models(eps=eps, strict=True)


def test_cli_models_list(monkeypatch, capsys) -> None:
    registry = _demo_registry()
    monkeypatch.setattr(cli, "discover_models", lambda **_: registry)

    exit_code = cli.main(["models", "list"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "alpha/first" in output
    assert "engine=alpha" in output


def test_cli_models_show(monkeypatch, capsys) -> None:
    registry = _demo_registry()
    monkeypatch.setattr(cli, "discover_models", lambda **_: registry)

    exit_code = cli.main(["models", "show", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Engine ID" in output
    assert "alpha/first" in output


def test_cli_models_cache(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "resolve_cache_dir", lambda: tmp_path)

    exit_code = cli.main(["models", "cache"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert str(tmp_path) in output


def test_cli_transcribe(monkeypatch, capsys) -> None:
    registry = _demo_registry()
    monkeypatch.setattr(cli, "discover_models", lambda **_: registry)
    monkeypatch.setattr(cli, "load_audio", lambda _: np.zeros(16000, dtype=np.float32))

    exit_code = cli.main(["transcribe", "alpha/first", "dummy.wav"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "dummy" in output


def test_cli_compliance_entrypoints_failure(monkeypatch, capsys) -> None:
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


def test_cli_serve_uses_server_module(monkeypatch) -> None:
    module = types.ModuleType("standard_asr.server")
    called = {}

    def _run(**kwargs):
        called.update(kwargs)

    module.run = _run

    monkeypatch.setitem(__import__("sys").modules, "standard_asr.server", module)

    exit_code = cli.main(["serve", "--host", "0.0.0.0", "--port", "9001"])

    assert exit_code == 0
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 9001


def test_cli_models_prepare_calls_transcribe(monkeypatch, capsys) -> None:
    registry = _demo_registry()
    monkeypatch.setattr(cli, "discover_models", lambda **_: registry)

    exit_code = cli.main(["models", "prepare", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "prepare" in output.lower()


def test_cli_models_prepare_calls_prepare(monkeypatch, capsys) -> None:
    class _PrepASR:
        def __init__(self) -> None:
            self.called = False

        def prepare(self) -> None:
            self.called = True

    prep = _PrepASR()
    registry = _demo_registry()
    monkeypatch.setattr(registry, "create", lambda *_: prep)
    monkeypatch.setattr(cli, "discover_models", lambda **_: registry)

    exit_code = cli.main(["models", "prepare", "alpha/first"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert prep.called is True
    assert "prepare" in output.lower()


def test_parse_options() -> None:
    options = cli._parse_options('{"language": "en"}')
    assert isinstance(options, dict)
    assert options["language"] == "en"

    with pytest.raises(ValueError):
        cli._parse_options("[1, 2, 3]")
