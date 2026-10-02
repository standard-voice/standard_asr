# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for custom provider-parameter validation errors."""

from __future__ import annotations

import pytest
from pydantic import ValidationError, field_validator, model_validator
from pydantic_core import PydanticCustomError

from standard_asr.contract.params import ProviderParams, WireRuntimeParams
from standard_asr.toolchain import cli
from standard_asr.toolchain import server as server_module

_SECRET = "sk-CUSTOM-PROVIDER-SECRET"


class _FieldValidatedParams(ProviderParams):
    """Reject one field with a safe custom message and private context."""

    mode: str

    @field_validator("mode")
    @classmethod
    def _reject_mode(cls, _value: str) -> str:
        raise PydanticCustomError(
            "provider_mode_disallowed",
            "Mode violates the {rule}.",
            {"rule": "provider allowlist", "debug_value": _SECRET},
        )


class _ModelValidatedParams(ProviderParams):
    """Reject a field combination with a message that echoes caller input."""

    primary: str
    secondary: str

    @model_validator(mode="after")
    def _reject_combination(self) -> _ModelValidatedParams:
        raise PydanticCustomError(
            "provider_combination_disallowed",
            "The combination contains {primary}.",
            {"primary": self.primary, "debug_value": _SECRET},
        )


class _NestedContextParams(ProviderParams):
    """Expose a rendered placeholder supplied by another context value."""

    mode: str

    @field_validator("mode")
    @classmethod
    def _reject_mode(cls, _value: str) -> str:
        raise PydanticCustomError(
            "provider_nested_context",
            "Mode violates {first}.",
            {"second": "rewritten", "first": "{second}"},
        )


@pytest.mark.parametrize(
    ("params_type", "payload", "code", "loc", "message", "context"),
    [
        (
            _FieldValidatedParams,
            {"mode": "restricted"},
            "provider_mode_disallowed",
            ("provider_params", "mode"),
            "Mode violates the provider allowlist.",
            {"rule": "provider allowlist", "debug_value": _SECRET},
        ),
        (
            _ModelValidatedParams,
            {"primary": "caller-secret", "secondary": "other"},
            "provider_combination_disallowed",
            ("provider_params",),
            "The combination contains caller-secret.",
            {"primary": "caller-secret", "debug_value": _SECRET},
        ),
        (
            _NestedContextParams,
            {"mode": "restricted"},
            "provider_nested_context",
            ("provider_params", "mode"),
            "Mode violates {second}.",
            {"first": "{second}", "second": "rewritten"},
        ),
    ],
)
def test_promotion_preserves_custom_provider_error(
    params_type: type[ProviderParams],
    payload: dict[str, object],
    code: str,
    loc: tuple[str, ...],
    message: str,
    context: dict[str, object],
) -> None:
    """Promotion keeps custom error identity while prefixing its location."""
    wire = WireRuntimeParams(provider_params=payload)

    with pytest.raises(ValidationError) as exc_info:
        wire.to_runtime_params(params_type)

    error = exc_info.value.errors(include_url=False)[0]
    assert error["type"] == code
    assert error["loc"] == loc
    assert error["msg"] == message
    assert "ctx" not in error
    cause = exc_info.value.__cause__
    assert isinstance(cause, ValidationError)
    original_error = cause.errors(include_url=False)[0]
    assert original_error["msg"] == message
    assert original_error.get("ctx") == context


class _Engine:
    """Minimal engine exposing one provider params type."""

    def __init__(self, params_type: type[ProviderParams]) -> None:
        self.provider_params_type = params_type

    def close(self) -> None:
        pass


class _Registry:
    """Registry stand-in for validation that fails before transcription."""

    def __init__(self, params_type: type[ProviderParams]) -> None:
        self._params_type = params_type

    def names(self) -> list[str]:
        return ["dummy/model"]

    def create(self, model: str, **_config: object) -> _Engine:
        assert model == "dummy/model"
        return _Engine(self._params_type)


@pytest.mark.parametrize(
    ("params_type", "provider_params", "code", "loc", "message"),
    [
        (
            _FieldValidatedParams,
            {"mode": "restricted"},
            "provider_mode_disallowed",
            ["options", "provider_params", "mode"],
            "Mode violates the provider allowlist.",
        ),
        (
            _ModelValidatedParams,
            {"primary": _SECRET, "secondary": "other"},
            "provider_combination_disallowed",
            ["options", "provider_params"],
            "[redacted]",
        ),
    ],
)
def test_rest_returns_custom_provider_errors_as_sanitized_422(
    params_type: type[ProviderParams],
    provider_params: dict[str, object],
    code: str,
    loc: list[str],
    message: str,
) -> None:
    """REST preserves typed custom errors and removes input and context."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_Registry(params_type))  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/v1/transcribe:json",
            json={
                "model": "dummy/model",
                "audio": "ZmFrZQ==",
                "options": {"provider_params": provider_params},
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == [{"type": code, "loc": loc, "msg": message}]
    assert _SECRET not in response.text
    assert "ctx" not in response.text
    assert "input" not in response.text


class _CliRegistry:
    """Return an engine whose custom provider validation fails before use."""

    def __init__(self, params_type: type[ProviderParams]) -> None:
        self._params_type = params_type

    def create(self, _model: str, **_config: object) -> _Engine:
        return _Engine(self._params_type)


@pytest.mark.parametrize(
    ("params_type", "options", "expected"),
    [
        (
            _FieldValidatedParams,
            '{"provider_params":{"mode":"restricted"}}',
            "Mode violates the provider allowlist.",
        ),
        (
            _ModelValidatedParams,
            f'{{"provider_params":{{"primary":"{_SECRET}","secondary":"other"}}}}',
            "[redacted]",
        ),
    ],
)
def test_cli_reports_custom_provider_errors_as_usage_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    params_type: type[ProviderParams],
    options: str,
    expected: str,
) -> None:
    """CLI custom validation errors stay sanitized caller errors with exit 2."""

    def discover(**_kwargs: object) -> _CliRegistry:
        return _CliRegistry(params_type)

    monkeypatch.setattr(_Engine, "provider_params_type", params_type, raising=False)
    monkeypatch.setattr(cli, "discover_models", discover)

    exit_code = cli.main(["transcribe", "dummy/model", "audio.wav", "--options", options])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert expected in captured.err
    assert _SECRET not in captured.err
    assert captured.out == ""
