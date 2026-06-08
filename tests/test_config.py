# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for init config: mixins, credentials, env fallback."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import SecretStr

from standard_asr.asr_config import (
    SECRET_MASK,
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


def test_reveal_dump_materializes_secret_plaintext() -> None:
    # reveal_dump() is the explicit, symmetric counterpart to public_dump() for
    # in-process SDK calls (INIT-4): secrets are materialized as plaintext.
    cfg = _CloudConfig(api_key=SecretStr("super-secret"), base_url="https://api.acme.test")
    revealed = cfg.reveal_dump()
    assert revealed["api_key"] == "super-secret"
    assert revealed["base_url"] == "https://api.acme.test"
    # public_dump() stays masked for the same instance.
    assert "super-secret" not in str(cfg.public_dump())


def test_reveal_dump_leaves_unset_secret_as_none() -> None:
    cfg = _CloudConfig()
    revealed = cfg.reveal_dump()
    assert revealed["api_key"] is None


def test_secret_field_marks_schema() -> None:
    schema = _CloudConfig.model_json_schema()
    assert schema["properties"]["api_key"].get("secret") is True


def test_secret_marked_non_secretstr_field_rejected_at_definition() -> None:
    # A secret-marked field annotated as plain str leaks plaintext everywhere;
    # the framework MUST fail loud at class-definition time (IC.3).
    with pytest.raises(TypeError, match="marked secret"):

        class _BadCfg(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            api_key: str | None = secret_field()  # type: ignore[assignment]


def test_public_dump_redacts_secret_marked_field_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Defensive masking: even if a plaintext value (hypothetically) slipped past
    # the annotation guard, public_dump() must never emit it. We simulate that by
    # patching model_dump to return a plaintext value for the secret field.
    cfg = _CloudConfig(api_key=SecretStr("super-secret"))
    raw = dict(cfg.model_dump(mode="json"))
    raw["api_key"] = "leaked-plaintext"

    def _leaky_dump(_self: _CloudConfig, **_kw: object) -> dict[str, object]:
        return raw

    monkeypatch.setattr(_CloudConfig, "model_dump", _leaky_dump)
    dumped = cfg.public_dump()
    assert dumped["api_key"] == SECRET_MASK
    assert "leaked-plaintext" not in str(dumped)


def test_secretstr_config_roundtrips_masked() -> None:
    cfg = _CloudConfig(api_key=SecretStr("super-secret"))
    dumped = cfg.public_dump()
    assert dumped["api_key"] == SECRET_MASK
    assert "super-secret" not in str(dumped)
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "super-secret"


def test_public_dump_leaves_unset_secret_as_none() -> None:
    cfg = _CloudConfig()
    dumped = cfg.public_dump()
    assert dumped["api_key"] is None


def test_secret_whitespace_not_stripped_direct() -> None:
    # str_strip_whitespace MUST NOT silently trim a credential's contents, which
    # could mask a paste error (X-EL-5). Plain routing fields still strip.
    cfg = _CloudConfig(api_key=SecretStr("  pad-secret  "), base_url="  https://x  ")
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "  pad-secret  "
    assert cfg.base_url == "https://x"


def test_secret_whitespace_not_stripped_from_env() -> None:
    # The from_env path hands the constructor a plain str; it must still not be
    # trimmed for a secret-marked field.
    env = {"STANDARD_ASR_ACME_API_KEY": "  pad-secret  "}
    cfg = _CloudConfig.from_env("acme", environ=env)
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "  pad-secret  "


def test_secret_whitespace_preserved_via_alias() -> None:
    # Wrapping must also find the value under the field alias.
    from pydantic import Field

    class _Aliased(BaseConfig[Literal["al"]]):
        engine: Literal["al"] = "al"
        xi_api_key: SecretStr | None = Field(
            default=None,
            alias="xi-api-key",
            json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
        )

    cfg = _Aliased.model_validate({"xi-api-key": "  tok  "})
    assert cfg.xi_api_key is not None
    assert cfg.xi_api_key.get_secret_value() == "  tok  "


def test_secret_validator_passes_non_mapping_input_through() -> None:
    # The before-validator must not choke on non-dict input (e.g. model_validate
    # of a non-mapping); it returns it unchanged so normal validation reports the
    # error. This exercises the non-dict guard branch.
    with pytest.raises(ValueError):
        _CloudConfig.model_validate(["not", "a", "dict"])


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


def test_from_env_explicit_wins_over_env() -> None:
    env = {"STANDARD_ASR_ACME_BASE_URL": "https://from-env.test"}
    cfg = _CloudConfig.from_env("acme", environ=env, base_url="https://explicit.test")
    assert cfg.base_url == "https://explicit.test"


def test_from_env_fills_unset_from_env() -> None:
    env = {"STANDARD_ASR_ACME_BASE_URL": "https://from-env.test"}
    cfg = _CloudConfig.from_env("acme", environ=env)
    assert cfg.base_url == "https://from-env.test"


def test_from_env_wraps_secret_and_masks() -> None:
    env = {"STANDARD_ASR_ACME_API_KEY": "super-secret"}
    cfg = _CloudConfig.from_env("acme", environ=env)
    # Secret was wrapped in SecretStr -> masked everywhere, plaintext only on
    # explicit reveal (no plaintext dict leak path).
    assert isinstance(cfg.api_key, SecretStr)
    assert "super-secret" not in str(cfg)
    assert "super-secret" not in str(cfg.public_dump())
    assert cfg.api_key.get_secret_value() == "super-secret"


def test_from_env_does_not_downgrade_strict_policy() -> None:
    # Env fallback MUST NOT let the environment flip the fail-loud `strict`
    # safety policy to best_effort (X-EL-1).
    env = {"STANDARD_ASR_ACME_STRICT": "false"}
    assert "strict" not in _CloudConfig.env_overrides("acme", environ=env)
    cfg = _CloudConfig.from_env("acme", environ=env)
    assert cfg.strict is True


def test_from_env_loads_aliased_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    # IC.4: a credential declaring a provider-native alias (e.g. ElevenLabs
    # `xi-api-key`) must still load from its STANDARD_ASR_<ENGINE>_<FIELD> env var
    # (keyed by attribute name), even under extra="forbid" (INIT-1).
    from pydantic import Field

    class _ElevenConfig(BaseConfig[Literal["eleven"]]):
        engine: Literal["eleven"] = "eleven"
        api_key: SecretStr | None = secret_field(description="key")

        # Re-declare with an alias to mimic an aliased credential field.
        xi_api_key: SecretStr | None = Field(
            default=None,
            alias="xi-api-key",
            json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
        )

    env = {"STANDARD_ASR_ELEVEN_XI_API_KEY": "secret-token"}
    cfg = _ElevenConfig.from_env("eleven", environ=env)
    assert isinstance(cfg.xi_api_key, SecretStr)
    assert cfg.xi_api_key.get_secret_value() == "secret-token"
    assert "secret-token" not in str(cfg.public_dump())


def test_from_env_does_not_read_engine_discriminator() -> None:
    env = {"STANDARD_ASR_ACME_ENGINE": "evil"}
    cfg = _CloudConfig.from_env("acme", environ=env)
    assert cfg.engine == "acme"


def test_from_env_missing_required_raises() -> None:
    class _NeedsKey(BaseConfig[Literal["n"]]):
        engine: Literal["n"] = "n"
        required_field: str

    with pytest.raises(ValueError):
        _NeedsKey.from_env("n", environ={})
