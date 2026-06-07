# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for init config: mixins, credentials, env fallback."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import SecretStr

from standard_asr.asr_config import (
    BaseConfig,
    CredentialsConfigMixin,
    DeviceConfigMixin,
    LanguageConfigMixin,
    env_var_name,
    secret_field,
)
from standard_asr.exceptions import ConfigError


class _CloudConfig(CredentialsConfigMixin, LanguageConfigMixin, BaseConfig[Literal["acme"]]):
    engine: Literal["acme"] = "acme"


class _LocalConfig(DeviceConfigMixin, BaseConfig[Literal["local"]]):
    engine: Literal["local"] = "local"


def test_strict_defaults_true() -> None:
    assert _LocalConfig().strict is True


def test_secret_is_masked_in_public_dump() -> None:
    cfg = _CloudConfig(api_key=SecretStr("super-secret"))
    dumped = cfg.public_dump()
    assert "super-secret" not in str(dumped)
    # Plaintext only on explicit reveal.
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "super-secret"


def test_secret_field_marks_schema() -> None:
    schema = _CloudConfig.model_json_schema()
    assert schema["properties"]["api_key"].get("secret") is True


def test_env_var_name_normalization() -> None:
    assert env_var_name("acme-cloud", "api_key") == "STANDARD_ASR_ACME_CLOUD_API_KEY"


def test_env_overrides_picks_up_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STANDARD_ASR_ACME_BASE_URL", "https://api.acme.test")
    overrides = _CloudConfig.env_overrides("acme")
    assert overrides["base_url"] == "https://api.acme.test"


def test_env_overrides_collision_detected() -> None:
    class _Collide(BaseConfig[Literal["x"]]):
        engine: Literal["x"] = "x"
        apikey: str | None = None
        apiKey: str | None = None  # noqa: N815 - intentional case collision

    with pytest.raises(ConfigError, match="collision"):
        _Collide.env_overrides("x")


def test_endpoint_routing_not_secret() -> None:
    cfg = _CloudConfig(base_url="https://api.acme.test", region="us-east")
    dumped = cfg.public_dump()
    assert dumped["base_url"] == "https://api.acme.test"
    assert dumped["region"] == "us-east"


def test_extra_forbidden() -> None:
    with pytest.raises(ValueError):
        _LocalConfig(unknown=1)  # type: ignore[call-arg]


def test_secret_field_helper_default_none() -> None:
    class _C(BaseConfig[Literal["c"]]):
        engine: Literal["c"] = "c"
        token: SecretStr | None = secret_field(description="tok")

    assert _C().token is None
