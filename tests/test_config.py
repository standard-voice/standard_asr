# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for init config: mixins, credentials, env fallback."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, SecretStr

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
    # in-process SDK calls: secrets are materialized as plaintext.
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


def test_secret_marked_container_of_secrets_rejected_at_definition() -> None:
    # A container parametrized by a secret type satisfied the old
    # recursive __args__ unwrap, but the whitespace-preserving wrapper and the
    # masking dumps only handle scalar secrets -- half-protected. The check is
    # scalar-only, so a secret-marked container fails loud at class definition.
    with pytest.raises(TypeError, match="separate scalar fields"):

        class _BadListCfg(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            api_keys: list[SecretStr] = secret_field(default=[])  # type: ignore[assignment]


def test_secret_marked_optional_container_of_secrets_rejected_at_definition() -> None:
    # A union does not launder a container: `list[SecretStr] | None` is still a
    # container of secrets, not a scalar secret type.
    with pytest.raises(TypeError, match="separate scalar fields"):

        class _BadOptListCfg(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            api_keys: list[SecretStr] | None = secret_field()  # type: ignore[assignment]


def test_secret_marked_scalar_annotations_still_pass_definition() -> None:
    # Bare SecretStr and SecretStr | None remain the supported scalar shapes.
    class _ScalarCfg(BaseConfig[Literal["ok"]]):
        engine: Literal["ok"] = "ok"
        required_key: SecretStr = secret_field(default=SecretStr("preset"))
        optional_key: SecretStr | None = secret_field()

    cfg = _ScalarCfg(optional_key=SecretStr("tok"))
    assert cfg.required_key.get_secret_value() == "preset"
    assert cfg.optional_key is not None
    assert cfg.optional_key.get_secret_value() == "tok"


def test_secret_marked_field_in_nested_submodel_rejected_at_definition() -> None:
    # A secret marker on a NESTED submodel field (IC.8 encourages
    # per-model-family submodels) bypassed both definition-time guards -- the
    # SecretStr enforcement and public_dump's by-name masking only walk a
    # BaseConfig's OWN fields -- so the plaintext leaked through public_dump /
    # repr / model_dump. The hook now rejects it, naming the offending path and
    # directing the author to a top-level scalar SecretStr.
    class _Auth(BaseModel):
        token: str | None = secret_field(description="oops, plain str in a submodel")

    with pytest.raises(TypeError, match=r"nested submodel.*_Auth\.token.*top-level scalar"):

        class _NestedCfg(BaseConfig[Literal["nested"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["nested"] = "nested"
            auth: _Auth | None = None


def test_secret_marked_field_nested_in_container_rejected_at_definition() -> None:
    # The nested-secret guard unwraps containers too: a submodel carrying a
    # secret reached through list[...] / dict[...] is just as unprotected.
    class _Auth(BaseModel):
        token: SecretStr | None = secret_field()

    with pytest.raises(TypeError, match="nested submodel"):

        class _ListCfg(BaseConfig[Literal["lst"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["lst"] = "lst"
            auths: list[_Auth] | None = None

    with pytest.raises(TypeError, match="nested submodel"):

        class _DictCfg(BaseConfig[Literal["dct"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["dct"] = "dct"
            auths: dict[str, _Auth] | None = None


def test_forward_ref_submodel_rejected_fail_closed_at_definition() -> None:
    # A field whose submodel is defined AFTER the config (or not imported)
    # is left by pydantic as an unresolved ForwardRef. Guard 2b's RESOLVED nested
    # scan cannot see a secret buried behind it, so guard 2a fails closed and
    # rejects the annotation at definition -- mirroring guard 1's fail-closed stance.
    # Pre-fix the class defined silently and the buried credential leaked through
    # public_dump / repr / model_dump.
    with pytest.raises(TypeError, match=r"unresolved forward-reference.*_LaterAuth"):

        class _ForwardRefCfg(BaseConfig[Literal["fwd"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["fwd"] = "fwd"
            auth: _LaterAuth | None = None

    class _LaterAuth(BaseModel):  # the submodel, defined AFTER the config above
        token: str | None = secret_field()  # plain str behind secret_field -> the real leak shape


def test_forward_ref_submodel_in_container_rejected_fail_closed() -> None:
    # The same fail-closed behavior reached through a container annotation; under
    # PEP 563 string annotations the whole annotation is one unresolved ForwardRef.
    with pytest.raises(TypeError, match="unresolved forward-reference"):

        class _ForwardRefListCfg(BaseConfig[Literal["fwdl"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["fwdl"] = "fwdl"
            auths: list[_LaterAuthList] = []

    class _LaterAuthList(BaseModel):  # the submodel, defined AFTER the config above
        token: str | None = secret_field()  # plain str behind secret_field -> the real leak shape


def test_nested_submodel_without_secret_marker_allowed() -> None:
    # IC.8 nested submodels are fully supported as long as they carry no secret
    # marker; the guard must not flag an ordinary (non-credential) submodel.
    class _ModelOpts(BaseModel):
        beam_size: int = 5
        temperature: float = 0.0

    class _Cfg(BaseConfig[Literal["plain"]]):
        engine: Literal["plain"] = "plain"
        opts: _ModelOpts | None = None

    cfg = _Cfg(opts=_ModelOpts(beam_size=3))
    assert cfg.public_dump()["opts"] == {"beam_size": 3, "temperature": 0.0}


def test_deeply_nested_secret_marker_rejected_at_definition() -> None:
    # The guard recurses through multiple submodel levels: a secret two layers
    # down is still detected (and reported with its full dotted path).
    class _Inner(BaseModel):
        token: str | None = secret_field()

    class _Outer(BaseModel):
        inner: _Inner | None = None

    # The reported path is rooted at the config field's submodel and walks down:
    # _Outer.inner.token (the secret two levels below the config field `outer`).
    with pytest.raises(TypeError, match=r"_Outer\.inner\.token"):

        class _Cfg(BaseConfig[Literal["deep"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["deep"] = "deep"
            outer: _Outer | None = None


def test_self_referential_submodel_does_not_loop() -> None:
    # The nested-secret walk uses a visited set, so a self-referential submodel
    # graph (a recursive config shape) terminates instead of recursing forever.
    class _Node(BaseModel):
        child: "_Node | None" = None

    _Node.model_rebuild()

    class _Cfg(BaseConfig[Literal["rec"]]):
        engine: Literal["rec"] = "rec"
        root: _Node | None = None

    # Defining the class (which runs the recursive guard) did not hang or raise.
    assert _Cfg(root=_Node()).engine == "rec"


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
    # could mask a paste error. Plain routing fields still strip.
    cfg = _CloudConfig(api_key=SecretStr("  pad-secret  "), base_url="  https://x  ")
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "  pad-secret  "
    assert cfg.base_url == "https://x"


def test_secret_whitespace_not_stripped_from_env() -> None:
    # The from_env path hands the constructor a plain str; it must still not be
    # trimmed for a secret-marked field.
    env = {"STANDARD_ASR_ACME__API_KEY": "  pad-secret  "}
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


def test_secret_validator_does_not_mutate_caller_input() -> None:
    # No spooky action at a distance: validating a caller's mapping MUST NOT
    # mutate it (the before-validator wraps raw secret strings on a shallow copy).
    # The caller's dict keeps its original plain-str api_key, and the resulting
    # model still has the secret preserved/correct.
    user_input = {"api_key": "  pad-secret  ", "base_url": "https://x"}
    cfg = _CloudConfig.model_validate(user_input)
    assert user_input["api_key"] == "  pad-secret  "  # original str, untouched.
    assert not isinstance(user_input["api_key"], SecretStr)
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "  pad-secret  "


def test_secret_validator_does_not_mutate_caller_input_via_alias() -> None:
    # The same non-mutation guarantee holds when the secret is supplied under a
    # field alias (the wrap-on-copy path keys by alias too).
    from pydantic import Field

    class _Aliased(BaseConfig[Literal["al"]]):
        engine: Literal["al"] = "al"
        xi_api_key: SecretStr | None = Field(
            default=None,
            alias="xi-api-key",
            json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
        )

    user_input: dict[str, object] = {"xi-api-key": "tok"}
    cfg = _Aliased.model_validate(user_input)
    assert user_input["xi-api-key"] == "tok"
    assert not isinstance(user_input["xi-api-key"], SecretStr)
    assert cfg.xi_api_key is not None
    assert cfg.xi_api_key.get_secret_value() == "tok"


def test_env_var_name_normalization() -> None:
    assert env_var_name("acme-cloud", "api_key") == "STANDARD_ASR_ACME_CLOUD__API_KEY"


def test_env_var_name_double_underscore_disambiguates_boundary() -> None:
    # The engine/field boundary uses a DOUBLE underscore, so an
    # engine/field split that collided under the old single-underscore scheme is
    # now distinct: ("openai", "api_key") vs ("openai-api", "key") no longer
    # both produce STANDARD_ASR_OPENAI_API_KEY (which let one engine silently
    # read another's credentials).
    assert env_var_name("openai", "api_key") == "STANDARD_ASR_OPENAI__API_KEY"
    assert env_var_name("openai-api", "key") == "STANDARD_ASR_OPENAI_API__KEY"
    assert env_var_name("openai", "api_key") != env_var_name("openai-api", "key")


def test_env_var_name_collapses_non_alphanumeric_runs() -> None:
    # A run of non-alphanumerics collapses to a SINGLE underscore so no segment
    # can contain "__" and forge a false boundary.
    assert env_var_name("a--b", "c") == "STANDARD_ASR_A_B__C"
    assert env_var_name("a", "b..c") == "STANDARD_ASR_A__B_C"


def test_env_fallback_decodes_list_field_from_json() -> None:
    # Default_candidate_languages is a spec-named Init Config field
    # (list[str]), but an env var is a bare string that never coerces into a
    # list -- previously a list_type ValidationError, leaving a standard field
    # unreachable through its own env convention. The env value is now JSON-
    # decoded for composite fields.
    class _LangCfg(LanguageConfigMixin, BaseConfig[Literal["lg"]]):
        engine: Literal["lg"] = "lg"
        default_language: str | None = "auto"

    env = {env_var_name("lg", "default_candidate_languages"): '["en", "ja"]'}
    cfg = _LangCfg.from_env("lg", environ=env)
    assert cfg.default_candidate_languages == ["en", "ja"]


def test_env_fallback_malformed_json_list_fails_loud() -> None:
    # On a JSON-decode error the raw string is kept so construction still FAILS
    # LOUDLY (a list_type ValidationError), never silently dropping the value.
    class _LangCfg(LanguageConfigMixin, BaseConfig[Literal["lg"]]):
        engine: Literal["lg"] = "lg"
        default_language: str | None = "auto"

    env = {env_var_name("lg", "default_candidate_languages"): "en,ja"}  # not JSON
    with pytest.raises(ValueError):
        _LangCfg.from_env("lg", environ=env)


def test_env_fallback_scalar_string_not_json_decoded() -> None:
    # A scalar str field is NOT JSON-decoded: a base_url that happens to look
    # like JSON (or a credential) must pass through verbatim, not be reinterpreted.
    env = {env_var_name("acme", "base_url"): "[1,2,3]"}
    cfg = _CloudConfig.from_env("acme", environ=env)
    assert cfg.base_url == "[1,2,3]"


def test_env_fallback_literal_field_not_json_decoded() -> None:
    # A non-union, non-container, non-model annotation (e.g. Literal) is scalar:
    # its env value passes through verbatim, never JSON-decoded. A
    # bare token like "fast" is not valid JSON, so JSON-decoding it would drop
    # the value; it must reach the field unchanged.
    class _ModeCfg(BaseConfig[Literal["m"]]):
        engine: Literal["m"] = "m"
        mode: Literal["fast", "accurate"] = "fast"

    env = {env_var_name("m", "mode"): "accurate"}
    assert _ModeCfg.env_overrides("m", environ=env) == {"mode": "accurate"}
    assert _ModeCfg.from_env("m", environ=env).mode == "accurate"


def test_env_overrides_picks_up_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STANDARD_ASR_ACME__BASE_URL", "https://api.acme.test")
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
    env = {"STANDARD_ASR_ACME__BASE_URL": "https://from-env.test"}
    cfg = _CloudConfig.from_env("acme", environ=env, base_url="https://explicit.test")
    assert cfg.base_url == "https://explicit.test"


def test_from_env_explicit_none_overrides_env() -> None:
    # minor: "explicit > env" treats an explicitly-passed None as a
    # value (the key IS present), so it wins over env -- the rule is "explicit
    # wins", not "explicit-non-None wins". This locks the documented semantics so
    # a wrapper forwarding optional None kwargs gets predictable behavior.
    env = {"STANDARD_ASR_ACME__BASE_URL": "https://from-env.test"}
    cfg = _CloudConfig.from_env("acme", environ=env, base_url=None)
    assert cfg.base_url is None


def test_from_env_omitted_key_still_falls_back_to_env() -> None:
    # Contrast with the None case: a key entirely OMITTED from explicit is
    # "absent", so the env value fills it.
    env = {"STANDARD_ASR_ACME__BASE_URL": "https://from-env.test"}
    cfg = _CloudConfig.from_env("acme", environ=env)  # base_url omitted
    assert cfg.base_url == "https://from-env.test"


def test_env_fallback_covers_engine_declared_field() -> None:
    # minor: env fallback covers the FULL config surface, not just the
    # standard mixin fields -- an engine-declared field (e.g. beam_size) gets a
    # STANDARD_ASR_<ENGINE>_<FIELD> entry too (intentional DX, now documented in
    # IC.4 and the _ENV_EXCLUDED_FIELDS comment).
    class _EngineCfg(BaseConfig[Literal["eng"]]):
        engine: Literal["eng"] = "eng"
        beam_size: int = 1

    env = {"STANDARD_ASR_ENG__BEAM_SIZE": "5"}
    assert _EngineCfg.env_overrides("eng", environ=env) == {"beam_size": "5"}
    assert _EngineCfg.from_env("eng", environ=env).beam_size == 5


def test_from_env_accepts_read_only_mapping() -> None:
    # minor: environ is typed Mapping[str, str], so a read-only mapping
    # (os.environ is os._Environ, a Mapping -- not a dict) is a valid argument.
    from types import MappingProxyType

    env = MappingProxyType({"STANDARD_ASR_ACME__BASE_URL": "https://ro.test"})
    cfg = _CloudConfig.from_env("acme", environ=env)
    assert cfg.base_url == "https://ro.test"


def test_from_env_fills_unset_from_env() -> None:
    env = {"STANDARD_ASR_ACME__BASE_URL": "https://from-env.test"}
    cfg = _CloudConfig.from_env("acme", environ=env)
    assert cfg.base_url == "https://from-env.test"


def test_from_env_wraps_secret_and_masks() -> None:
    env = {"STANDARD_ASR_ACME__API_KEY": "super-secret"}
    cfg = _CloudConfig.from_env("acme", environ=env)
    # Secret was wrapped in SecretStr -> masked everywhere, plaintext only on
    # explicit reveal (no plaintext dict leak path).
    assert isinstance(cfg.api_key, SecretStr)
    assert "super-secret" not in str(cfg)
    assert "super-secret" not in str(cfg.public_dump())
    assert cfg.api_key.get_secret_value() == "super-secret"


def test_from_env_does_not_downgrade_strict_policy() -> None:
    # Env fallback MUST NOT let the environment flip the fail-loud `strict`
    # safety policy to best_effort.
    env = {"STANDARD_ASR_ACME__STRICT": "false"}
    assert "strict" not in _CloudConfig.env_overrides("acme", environ=env)
    cfg = _CloudConfig.from_env("acme", environ=env)
    assert cfg.strict is True


def test_from_env_loads_aliased_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    # IC.4: a credential declaring a provider-native alias (e.g. ElevenLabs
    # `xi-api-key`) must still load from its STANDARD_ASR_<ENGINE>_<FIELD> env var
    # (keyed by attribute name), even under extra="forbid".
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

    env = {"STANDARD_ASR_ELEVEN__XI_API_KEY": "secret-token"}
    cfg = _ElevenConfig.from_env("eleven", environ=env)
    assert isinstance(cfg.xi_api_key, SecretStr)
    assert cfg.xi_api_key.get_secret_value() == "secret-token"
    assert "secret-token" not in str(cfg.public_dump())


def test_from_env_does_not_read_engine_discriminator() -> None:
    env = {"STANDARD_ASR_ACME__ENGINE": "evil"}
    cfg = _CloudConfig.from_env("acme", environ=env)
    assert cfg.engine == "acme"


def test_from_env_missing_required_raises_config_error() -> None:
    class _NeedsKey(BaseConfig[Literal["n"]]):
        engine: Literal["n"] = "n"
        required_field: str

    # EC-1: construction failure is catchable as ConfigError (not a raw pydantic
    # ValidationError an app cannot catch as the standard type) -- and still a
    # ValueError subclass so existing handlers keep working -- carrying the
    # structured entries for transports to render.
    with pytest.raises(ConfigError) as excinfo:
        _NeedsKey.from_env("n", environ={})
    assert isinstance(excinfo.value, ValueError)
    details = excinfo.value.details
    assert details is not None
    assert any(entry["loc"] == ["required_field"] for entry in details)


def test_from_env_construction_error_does_not_echo_secret() -> None:
    class _TypedField(BaseConfig[Literal["n"]]):
        engine: Literal["n"] = "n"
        count: int = 0

    # A mis-placed secret in an env value must not be reflected back by from_env
    # (the EC-1 wrap scrubs it), mirroring the create() / CLI guards.
    secret = "sk-ENV-CONSTRUCT-LEAK"  # noqa: S105 - test fixture
    with pytest.raises(ConfigError) as excinfo:
        _TypedField.from_env("n", environ={"STANDARD_ASR_N__COUNT": secret})
    assert secret not in str(excinfo.value)
    assert secret not in repr(excinfo.value.details)
