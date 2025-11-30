"""CLI coverage for Standard ASR entrypoint tooling."""

from __future__ import annotations

from importlib.metadata import EntryPoint

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
