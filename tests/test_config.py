# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for init config: mixins, credentials, env fallback."""

from __future__ import annotations

import enum
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import pytest
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretBytes,
    SecretStr,
    ValidationError,
)

from standard_asr.contract.exceptions import ConfigError, ConfigurationRequiredError
from standard_asr.runtime.config import (
    SECRET_MASK,
    BaseConfig,
    CredentialsConfigMixin,
    DeviceConfigMixin,
    LanguageConfigMixin,
    env_var_name,
    secret_field,
)


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
    # the framework MUST fail loud at class-definition time.
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


def test_secret_marked_plaintext_union_rejected_at_definition() -> None:
    """``SecretStr | int`` must fail loud at class definition.

    The old ``any()``-over-members check accepted it, so ``Config(token=123)``
    produced a plain ``int`` -- plaintext in ``repr``/``model_dump`` -- while
    the schema advertised a password field. Exactly one carrier, optionally
    with ``None``, is the whole contract.
    """
    with pytest.raises(TypeError, match="exactly SecretStr or SecretBytes"):

        class _StrIntCfg(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            token: SecretStr | int = secret_field(default=...)  # type: ignore[assignment]

    with pytest.raises(TypeError, match="exactly SecretStr or SecretBytes"):

        class _BytesPlainCfg(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            token: SecretBytes | bytes = secret_field(default=...)  # type: ignore[assignment]

    with pytest.raises(TypeError, match="exactly SecretStr or SecretBytes"):

        class _OptStrIntCfg(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            token: SecretStr | int | None = secret_field()  # type: ignore[assignment]


def test_secret_marked_dual_carrier_union_rejected_at_definition() -> None:
    """``SecretStr | SecretBytes`` (with or without None) is ambiguous.

    The whitespace-preserving pre-validator must wrap a raw string into THE
    field's carrier; two carriers make that wrap a guess. Rejected loud.
    """
    with pytest.raises(TypeError, match="exactly SecretStr or SecretBytes"):

        class _DualCfg(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            token: SecretStr | SecretBytes = secret_field(default=...)

    with pytest.raises(TypeError, match="exactly SecretStr or SecretBytes"):

        class _DualNoneCfg(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            token: SecretStr | SecretBytes | None = secret_field()


def test_secret_plain_string_default_rejected_at_definition() -> None:
    """A plaintext default must fail loud at class definition.

    pydantic does not validate defaults, so ``secret_field(default="s")``
    lived on instances as a raw ``str``: plaintext ``repr``/``model_dump``,
    and a crash inside ``model_dump_json``/``public_dump`` when the secret
    serializer called ``get_secret_value()`` on it.
    """
    with pytest.raises(TypeError, match="default is a str"):

        class _PlainDefaultCfg(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            token: SecretStr = secret_field(default="SUPERSECRET")  # noqa: S106

    # The guard must NOT echo the plaintext default into the error message
    # (definition-time TypeErrors land in CI logs). pytest.raises, not a bare
    # try/except: an except-scoped assertion silently stops running the day
    # the guard stops raising.
    with pytest.raises(TypeError) as excinfo:

        class _PlainDefaultCfg2(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            token: SecretStr = secret_field(default="SUPERSECRET")  # noqa: S106

    assert "SUPERSECRET" not in str(excinfo.value)


def test_secret_none_default_on_non_optional_annotation_rejected() -> None:
    """Default ``None`` with a non-optional carrier annotation is a lie.

    Instances built without the field would carry ``None`` where the
    annotation promises ``SecretStr`` (defaults are unvalidated). The field
    must either union ``None`` or be required.
    """
    with pytest.raises(TypeError, match="does not admit None"):

        class _NoneDefaultCfg(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            token: SecretStr = secret_field()


def test_secret_default_factory_rejected_at_definition() -> None:
    """A ``default_factory`` on a secret field is rejected outright.

    Factories run at construction, unvetted by the definition-time guard; a
    factory returning plaintext would bypass masking silently.
    """
    from pydantic import Field

    with pytest.raises(TypeError, match="default_factory"):

        class _FactoryCfg(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            token: SecretStr | None = Field(
                default_factory=lambda: SecretStr("generated"),
                json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
            )


def test_required_secret_via_ellipsis_default() -> None:
    """``secret_field(default=...)`` declares a required credential."""

    class _RequiredCfg(BaseConfig[Literal["req"]]):
        engine: Literal["req"] = "req"
        token: SecretStr = secret_field(default=...)

    cfg = _RequiredCfg(token=SecretStr("tok"))
    assert cfg.token.get_secret_value() == "tok"
    with pytest.raises(ValidationError):
        _RequiredCfg()  # pyright: ignore[reportCallIssue]


def test_secret_bytes_field_constructs_from_raw_string() -> None:
    """A ``SecretBytes`` field accepts env/alias-style raw strings.

    The old pre-validator wrapped every raw secret string into ``SecretStr``,
    which the ``SecretBytes`` field then rejected (``bytes_type``) -- while
    plain pydantic would have coerced the bare str. The wrap now targets the
    field's own carrier via UTF-8, whitespace preserved exactly.
    """

    class _BytesCfg(BaseConfig[Literal["byt"]]):
        engine: Literal["byt"] = "byt"
        token: SecretBytes | None = secret_field()

    cfg = _BytesCfg(token=" padded-cred\t")  # pyright: ignore[reportArgumentType]
    assert cfg.token is not None
    assert cfg.token.get_secret_value() == b" padded-cred\t"

    env_cfg = _BytesCfg.from_env("byt", environ={"STANDARD_ASR_BYT__TOKEN": " env cred "})
    assert env_cfg.token is not None
    assert env_cfg.token.get_secret_value() == b" env cred "


def test_secret_bytes_field_via_alias_and_choices_keeps_bytes() -> None:
    """The carrier-aware wrap covers every flat input key, not just canonical."""
    from pydantic import AliasChoices, Field

    class _AliasBytesCfg(BaseConfig[Literal["ab"]]):
        engine: Literal["ab"] = "ab"
        token: SecretBytes | None = Field(
            default=None,
            validation_alias=AliasChoices("token", "xi-token"),
            json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
        )

    via_choice = _AliasBytesCfg.model_validate({"xi-token": " choice cred "})
    assert via_choice.token is not None
    assert via_choice.token.get_secret_value() == b" choice cred "


def test_secret_marked_field_in_nested_submodel_rejected_at_definition() -> None:
    # A secret marker on a NESTED submodel field (the standard encourages
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
    # secret reached through ``list[...]`` / ``dict[...]`` is just as unprotected.
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


def test_bare_carrier_in_nested_submodel_rejected_at_definition() -> None:
    # An UNMARKED SecretStr in a submodel is half-broken rather than leaky:
    # pydantic masks its dumps, but str_strip_whitespace silently rewrites its
    # raw string input, reveal_dump never unwraps it, and the submodel's own
    # hooks could unwrap it inside the "masked" dump. Carrier presence alone
    # (not just the marker) is therefore rejected in nested models.
    class _Auth(BaseModel):
        token: SecretStr | None = None

    with pytest.raises(TypeError, match=r"_Auth\.token is annotated with a secret carrier"):

        class _NestedCarrierCfg(BaseConfig[Literal["nc"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["nc"] = "nc"
            auth: _Auth | None = None


def test_top_level_carrier_without_marker_rejected_at_definition() -> None:
    # The same half-protection exists at the top level: an unmarked SecretStr
    # field bypasses the whitespace-preserving pre-validator, so a padded
    # credential is silently stripped (the exact rewrite the pipeline
    # forbids) and auto-UI renders an ordinary input instead of a password.
    with pytest.raises(TypeError, match="not marked secret"):

        class _UnmarkedCfg(BaseConfig[Literal["um"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["um"] = "um"
            api_key: SecretStr | None = None

    # Carriers hidden inside a container annotation are caught the same way
    # (marking them then routes into guard 1: exactly one carrier per secret).
    with pytest.raises(TypeError, match="not marked secret"):

        class _ListCarrierCfg(BaseConfig[Literal["ulc"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["ulc"] = "ulc"
            api_keys: list[SecretStr] | None = None


def test_author_serialization_hooks_rejected_at_definition() -> None:
    # public_dump == model_dump + by-name masking, and that equation is only
    # sound because author serialization hooks cannot exist: each of these
    # three ran INSIDE the "masked" dump and rematerialized the plaintext
    # under a key the mask never touches (computed 'authorization', a
    # model_serializer's shadow key, a field_serializer reading a sibling
    # secret into base_url).
    from pydantic import computed_field, field_serializer, model_serializer

    with pytest.raises(TypeError, match=r"@computed_field 'authorization'"):

        class _ComputedCfg(BaseConfig[Literal["comp"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["comp"] = "comp"
            api_key: SecretStr = secret_field(...)

            @computed_field  # pyright: ignore[reportAny]
            @property
            def authorization(self) -> str:
                """Derived header (the leak shape).

                Returns:
                    The bearer header with the plaintext secret.
                """
                return "Bearer " + self.api_key.get_secret_value()

    with pytest.raises(TypeError, match="@model_serializer"):

        class _ModelSerCfg(BaseConfig[Literal["mser"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["mser"] = "mser"
            api_key: SecretStr = secret_field(...)

            @model_serializer
            def _ser(self) -> dict[str, str]:
                """Rewrites the whole dump (the shadow-key leak shape).

                Returns:
                    A dict with the secret under an unmasked key.
                """
                return {"shadow": self.api_key.get_secret_value()}

    with pytest.raises(TypeError, match=r"@field_serializer '_ser_url'"):

        class _FieldSerCfg(BaseConfig[Literal["fser"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["fser"] = "fser"
            api_key: SecretStr = secret_field(...)
            base_url: str = "https://x"

            @field_serializer("base_url")
            def _ser_url(self, value: str) -> str:
                """Reads the sibling secret (the cross-field leak shape).

                Args:
                    value: The field value being serialized.

                Returns:
                    The URL with the plaintext secret appended.
                """
                return value + "?key=" + self.api_key.get_secret_value()

    # The closure is uniform: hooks are rejected on secret-free configs too
    # (the dump must be the declared input surface -- a computed key cannot
    # re-validate under extra='forbid', and auto-UI renders model_fields).
    with pytest.raises(TypeError, match="@computed_field"):

        class _NoSecretCfg(BaseConfig[Literal["nsc"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["nsc"] = "nsc"
            beam: int = 5

            @computed_field  # pyright: ignore[reportAny]
            @property
            def doubled(self) -> int:
                """A derived view field.

                Returns:
                    Twice the beam.
                """
                return self.beam * 2


def test_inherited_serialization_hook_rejected_at_definition() -> None:
    # The decorator registry aggregates base classes, so a hook smuggled in
    # through an innocent-looking mixin is caught identically.
    from pydantic import model_serializer

    class _HookMixin(BaseModel):
        @model_serializer
        def _ser(self) -> dict[str, str]:
            """Rewrites the dump from a base class.

            Returns:
                An arbitrary replacement payload.
            """
            return {}

    with pytest.raises(TypeError, match="declares .or inherits. author serialization"):

        class _InheritingCfg(  # pyright: ignore[reportUnusedClass]
            BaseConfig[Literal["inh"]], _HookMixin
        ):
            engine: Literal["inh"] = "inh"


def test_annotated_serializer_metadata_rejected_at_definition() -> None:
    # ``Annotated[..., PlainSerializer(...)]`` is the same open lane in field
    # metadata form: on a secret field the serializer receives the UNWRAPPED
    # carrier value; on any field it silently diverges the dump from the
    # declared input. Both variants are rejected at definition.
    from pydantic import PlainSerializer, WrapSerializer

    with pytest.raises(TypeError, match="PlainSerializer/WrapSerializer"):

        class _AnnotSecretCfg(BaseConfig[Literal["ans"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["ans"] = "ans"
            api_key: Annotated[SecretStr, PlainSerializer(lambda v: v.get_secret_value())] = (
                secret_field(...)
            )

    def _wrap(value: str, handler: Any) -> str:
        """Pass-through wrap serializer (shape only).

        Args:
            value: The value being serialized.
            handler: The next serializer in the chain.

        Returns:
            The handler's rendering.
        """
        return cast("str", handler(value))

    with pytest.raises(TypeError, match="PlainSerializer/WrapSerializer"):

        class _AnnotPlainCfg(BaseConfig[Literal["anp"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["anp"] = "anp"
            note: Annotated[str, WrapSerializer(_wrap)] = "x"


def test_closed_serialization_keeps_ordinary_shapes_working() -> None:
    # The closure must cost legal configs nothing: stdlib scalars, enums,
    # paths, nested non-secret submodels, and plain @property all construct,
    # dump JSON-safely, and round-trip through public_dump -> validate.
    class _Level(enum.Enum):
        FAST = "fast"
        ACCURATE = "accurate"

    class _Decode(BaseModel):
        model_config = ConfigDict(extra="forbid")

        beam: int = 4
        patience: float | None = None

    class _RichCfg(BaseConfig[Literal["rich"]]):
        engine: Literal["rich"] = "rich"
        api_key: SecretStr = secret_field(...)
        model_dir: Path | None = None
        level: _Level = _Level.FAST
        decode: _Decode = _Decode()

        @property
        def cache_key(self) -> str:
            """An in-process convenience view (NOT serialized).

            Returns:
                A stable cache key.
            """
            return f"{self.engine}:{self.level.value}"

    cfg = _RichCfg(
        engine="rich",
        api_key=SecretStr("sk-real"),
        model_dir=Path("/models"),
        level=_Level.ACCURATE,
        decode=_Decode(beam=8),
    )
    dumped = cfg.public_dump()
    assert dumped["api_key"] == SECRET_MASK
    # Path serialization is platform-native (\ on Windows): compare the
    # round-trip, not a POSIX literal.
    assert dumped["model_dir"] == str(Path("/models"))
    assert dumped["level"] == "accurate"
    assert dumped["decode"] == {"beam": 8, "patience": None}
    assert "cache_key" not in dumped  # a property is not a dump field
    assert cfg.cache_key == "rich:accurate"
    # The dump is the declared input surface: it re-validates (masked secret
    # aside, which a caller would replace with a real credential).
    reloaded = dict(dumped)
    reloaded["api_key"] = "sk-real"
    assert _RichCfg.model_validate(reloaded).decode.beam == 8


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
    # Nested submodels are fully supported as long as they carry no secret
    # marker; the guard must not flag an ordinary (non-credential) submodel.
    class _ModelOpts(BaseModel):
        model_config = ConfigDict(extra="forbid")

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
        model_config = ConfigDict(extra="forbid")

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
    # the annotation guard, public_dump() must never emit it. public_dump now
    # calls BaseModel.model_dump UNBOUND, so the leak is simulated by patching
    # THAT (the actual call target), not the instance-level method.
    cfg = _CloudConfig(api_key=SecretStr("super-secret"))
    raw = dict(BaseModel.model_dump(cfg, mode="json"))
    raw["api_key"] = "leaked-plaintext"

    def _leaky_dump(_self: object, **_kw: object) -> dict[str, object]:
        return raw

    monkeypatch.setattr(BaseModel, "model_dump", _leaky_dump)
    dumped = cfg.public_dump()
    assert dumped["api_key"] == SECRET_MASK
    assert "leaked-plaintext" not in str(dumped)


@pytest.mark.parametrize(
    "method",
    ["public_dump", "reveal_dump", "model_dump", "model_dump_json", "__iter__"],
)
def test_security_owned_serialization_method_override_rejected(method: str) -> None:
    # public_dump's "safe for /v1/models" contract rests on the DISPATCH, not
    # only the serializer schema: an ordinary Python override of any of these
    # runs inside the dump and can rematerialize a sibling secret under a key
    # the by-name mask never sees. Rejected at class definition (Guard 8).
    def _override(self: object, *args: object, **kwargs: object) -> dict[str, object]:
        return {}

    namespace: dict[str, object] = {
        "engine": "ovr",
        "__annotations__": {"engine": Literal["ovr"]},
        method: _override,
    }
    with pytest.raises(TypeError, match=f"security-owned serialization method {method!r}"):
        type("_OverrideCfg", (BaseConfig[Literal["ovr"]],), namespace)


def test_mixin_carried_serialization_override_rejected() -> None:
    # The override is resolved statically over the whole MRO, so one smuggled
    # in through an innocent-looking (non-BaseConfig) mixin is caught too.
    class _LeakMixin:
        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            """Rewrite the dump from a base class.

            Returns:
                An arbitrary replacement payload.
            """
            return {"leak": "x"}

    with pytest.raises(TypeError, match="security-owned serialization method 'model_dump'"):

        class _MixedCfg(_LeakMixin, BaseConfig[Literal["mix"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["mix"] = "mix"


def test_public_dump_ignores_a_post_definition_model_dump_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The RUNTIME half of the defense: even if an override is installed AFTER
    # definition (bypassing Guard 8, for example, by monkeypatch or a metaclass trick),
    # public_dump calls BaseModel.model_dump UNBOUND and never dispatches to it.
    cfg = _CloudConfig(api_key=SecretStr("sk-RUNTIME-VERIFY"))

    def _leaky_dump(self: object, **kwargs: object) -> dict[str, object]:
        base = dict(BaseModel.model_dump(cfg, mode="json"))
        base["authorization"] = "Bearer sk-RUNTIME-VERIFY"
        return base

    monkeypatch.setattr(_CloudConfig, "model_dump", _leaky_dump)
    dumped = cfg.public_dump()
    assert "authorization" not in dumped
    assert "sk-RUNTIME-VERIFY" not in str(dumped)
    assert dumped["api_key"] == SECRET_MASK


def test_reveal_dump_ignores_a_post_definition_iter_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # reveal_dump reads the DECLARED fields from instance state, never dict(self)
    # (which walks __iter__): a post-definition __iter__ override that injects an
    # extra key cannot smuggle it into the revealed dict.
    cfg = _CloudConfig(api_key=SecretStr("sk-REAL"))

    def _leaky_iter(self: object) -> object:
        yield ("engine", "acme")
        yield ("smuggled", "sk-VIA-ITER")

    monkeypatch.setattr(_CloudConfig, "__iter__", _leaky_iter)
    revealed = cfg.reveal_dump()
    assert "smuggled" not in revealed
    assert "sk-VIA-ITER" not in str(revealed)
    # The real declared secret is still materialized.
    assert revealed["api_key"] == "sk-REAL"


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


def test_config_input_domain_is_closed_to_mappings() -> None:
    # The before-validator REJECTS non-mapping input outright (fixed,
    # input-echo-free message) instead of passing it through: pass-through let
    # `model_validate(obj, from_attributes=True)` extract raw attribute
    # strings behind the secret wrap and silently strip them (see the
    # object-extraction test below). A wrong-typed input still fails loudly.
    with pytest.raises(ValidationError) as exc_info:
        _CloudConfig.model_validate(["not", "a", "dict"])
    assert exc_info.value.errors()[0]["type"] == "standard_asr_config_mapping_required"


def test_config_object_extraction_fails_loudly_not_silently_trimmed() -> None:
    # `model_validate(obj, from_attributes=True)` extracted raw attribute
    # strings BEHIND the whitespace-preserving wrap: str_strip_whitespace then
    # trimmed the credential before SecretStr coercion, so
    # "  sk-PADDED-SECRET  " validated as "sk-PADDED-SECRET" -- a silently
    # different key from the one the object held (the exact silent rewrite
    # the wrap forbids on the mapping path). Object extraction is therefore
    # not part of the config input contract: it MUST fail loudly.
    class _Settings:
        engine = "acme"
        api_key = "  sk-STANDARD-ASR-REVIEW-F5  "

    with pytest.raises(ValidationError) as exc_info:
        _CloudConfig.model_validate(_Settings(), from_attributes=True)
    assert exc_info.value.errors()[0]["type"] == "standard_asr_config_mapping_required"
    # The rejection message is a fixed authored string: no input echo.
    entry_msg = exc_info.value.errors()[0]["msg"]
    assert "sk-STANDARD-ASR-REVIEW-F5" not in entry_msg


def test_secret_whitespace_preserved_via_mapping_proxy() -> None:
    # The closed domain is Mapping, not dict: an immutable mapping view still
    # takes the whitespace-preserving wrap path.
    import types as _types

    cfg = _CloudConfig.model_validate(_types.MappingProxyType({"api_key": "  pad-secret  "}))
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "  pad-secret  "


def test_config_model_validate_json_preserves_secret_whitespace() -> None:
    # pydantic's native JSON pipeline rejected the wrap's SecretStr instance
    # (its JSON-mode secret schema accepts only the string form), so
    # model_validate_json failed for EVERY config document carrying a secret.
    # The override validates the parsed document in python mode, where the
    # wrap is defined -- and the padded credential survives byte-for-byte.
    cfg = _CloudConfig.model_validate_json('{"api_key": "  pad-secret  "}')
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "  pad-secret  "
    # Non-secret fields keep the convenience strip on the JSON path too.
    cfg2 = _CloudConfig.model_validate_json('{"base_url": "  https://x  "}')
    assert cfg2.base_url == "https://x"


def test_config_model_validate_json_malformed_raises_canonical_error() -> None:
    # A malformed document surfaces pydantic's canonical json_invalid error
    # (the delegation parses-and-fails before any field validation).
    with pytest.raises(ValidationError) as exc_info:
        _CloudConfig.model_validate_json('{"api_key": ')
    assert exc_info.value.errors()[0]["type"] == "json_invalid"


def test_config_model_validate_json_nonfinite_token_parity() -> None:
    # Both json.loads and pydantic's JSON parser accept the NaN token; on
    # either route the resulting float then fails string-field validation
    # identically. The override must not widen OR narrow that grammar.
    with pytest.raises(ValidationError) as exc_info:
        _CloudConfig.model_validate_json('{"api_key": NaN}')
    assert exc_info.value.errors()[0]["type"] == "string_type"


def test_config_model_validate_json_non_object_document_fails_loudly() -> None:
    # A valid JSON document that is not an object hits the closed input
    # domain (mapping-required), never a silent pass.
    with pytest.raises(ValidationError) as exc_info:
        _CloudConfig.model_validate_json('["not", "an", "object"]')
    assert exc_info.value.errors()[0]["type"] == "standard_asr_config_mapping_required"


def test_config_model_validate_strings_preserves_secret_whitespace() -> None:
    # The strings-mode sibling of the model_validate_json incompatibility:
    # the wrap's SecretStr carrier fails strings-mode field validation, so
    # model_validate_strings failed for EVERY config supplying a secret --
    # blaming the caller's own credential field for a wrong type it never
    # passed. The override validates under the config surface's own string
    # grammar (the from_env/env_overrides one) in python mode, where the
    # wrap is defined -- and the padded credential survives byte-for-byte.
    cfg = _CloudConfig.model_validate_strings({"api_key": "  pad-secret  "})
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "  pad-secret  "
    # Non-secret fields keep the convenience strip on the strings path too.
    cfg2 = _CloudConfig.model_validate_strings({"base_url": "  https://x  "})
    assert cfg2.base_url == "https://x"


def test_config_model_validate_strings_env_grammar() -> None:
    # One string grammar for every string-valued source: scalar fields take
    # the raw string (lax coercion), structured fields take a JSON document
    # exactly as the STANDARD_ASR_* env route does, and a malformed JSON
    # document for a structured field fails loudly with the field named.
    class _Structured(BaseConfig[Literal["structured"]]):
        engine: Literal["structured"] = "structured"
        threads: int = 1
        tags: list[str] = []

    cfg = _Structured.model_validate_strings({"threads": "4", "tags": '["a", "b"]'})
    assert cfg.threads == 4
    assert cfg.tags == ["a", "b"]

    with pytest.raises(ValidationError) as exc_info:
        _Structured.model_validate_strings({"tags": "[not json"})
    assert exc_info.value.errors()[0]["loc"] == ("tags",)

    # A non-mapping input hits the closed input domain (mapping-required).
    with pytest.raises(ValidationError) as exc_info:
        _Structured.model_validate_strings(["not", "a", "mapping"])
    assert exc_info.value.errors()[0]["type"] == "standard_asr_config_mapping_required"

    # An unknown key is judged by field validation (extra="forbid"), loudly.
    with pytest.raises(ValidationError):
        _Structured.model_validate_strings({"thraeds": "4"})


def test_config_instance_revalidation_still_short_circuits() -> None:
    # Closing the input domain must not break revalidating an existing config
    # instance: pydantic recognizes the instance before any validator runs.
    original = _CloudConfig(api_key=SecretStr("  pad-secret  "))
    revalidated = _CloudConfig.model_validate(original)
    assert revalidated is original
    assert revalidated.api_key is not None
    assert revalidated.api_key.get_secret_value() == "  pad-secret  "


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
    # A non-union, non-container, non-model annotation (for example, Literal) is scalar:
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
    # minor: "explicit > env" treats an explicitly passed None as a
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
    # standard mixin fields -- an engine-declared field (for example, beam_size) gets a
    # STANDARD_ASR_<ENGINE>_<FIELD> entry too (intentional DX, documented on
    # the _ENV_EXCLUDED_FIELDS comment).
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
    # A credential declaring a provider-native alias (for example, ElevenLabs
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


class _AliasedKeyConfig(BaseConfig[Literal["ali"]]):
    """A config whose credential declares a provider-native alias."""

    engine: Literal["ali"] = "ali"
    api_key: SecretStr | None = Field(
        default=None,
        alias="apiKey",
        json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
    )


def test_from_env_explicit_alias_wins_over_env_canonical() -> None:
    """Explicit-under-alias suppresses the field's canonical env fallback.

    The old blind merge kept BOTH keys: pydantic populated from the alias
    and rejected the canonical env key as extra (``extra="forbid"``) -- a
    loud failure where the documented contract says the explicit value wins.
    """
    env = {"STANDARD_ASR_ALI__API_KEY": "from-env"}
    cfg = _AliasedKeyConfig.from_env("ali", environ=env, **{"apiKey": "explicit"})
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "explicit"


def test_from_env_explicit_validation_alias_wins_over_env_canonical() -> None:
    """The same rule through a string ``validation_alias``."""

    class _VaConfig(BaseConfig[Literal["va"]]):
        engine: Literal["va"] = "va"
        api_key: SecretStr | None = Field(
            default=None,
            validation_alias="x-va-key",
            json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
        )

    env = {"STANDARD_ASR_VA__API_KEY": "from-env"}
    cfg = _VaConfig.from_env("va", environ=env, **{"x-va-key": "explicit"})
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "explicit"


def test_from_env_explicit_alias_choice_wins_over_env_canonical() -> None:
    """The same rule through ANY ``AliasChoices`` choice."""

    class _ChoicesConfig(BaseConfig[Literal["cho"]]):
        engine: Literal["cho"] = "cho"
        api_key: SecretStr | None = Field(
            default=None,
            validation_alias=AliasChoices("api_key", "xi-api-key"),
            json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
        )

    env = {"STANDARD_ASR_CHO__API_KEY": "from-env"}
    cfg = _ChoicesConfig.from_env("cho", environ=env, **{"xi-api-key": "explicit"})
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "explicit"


def test_from_env_explicit_none_under_alias_disables_env_fallback() -> None:
    """Explicit ``None`` under an alias key is a VALUE and wins over env.

    Mirrors the canonical-key None rule: "explicit wins", not
    "explicit-non-None wins" -- now alias-aware.
    """
    env = {"STANDARD_ASR_ALI__API_KEY": "from-env"}
    cfg = _AliasedKeyConfig.from_env("ali", environ=env, **{"apiKey": None})
    assert cfg.api_key is None


def test_from_env_dual_explicit_keys_for_one_field_still_loudly_rejected() -> None:
    """A caller passing canonical AND alias keys is a mistake, not merge policy.

    The alias-aware env drop must not swallow it: with no env involved at
    all, pydantic still rejects the duplicate under ``extra="forbid"``, with
    the echoed duplicate masked (it arrives pre-wrapped as ``SecretStr``).
    """
    with pytest.raises(ConfigError):
        _AliasedKeyConfig.from_env(
            "ali", environ={}, **{"apiKey": "via-alias", "api_key": "via-canonical"}
        )


def test_from_env_does_not_read_engine_discriminator() -> None:
    env = {"STANDARD_ASR_ACME__ENGINE": "evil"}
    cfg = _CloudConfig.from_env("acme", environ=env)
    assert cfg.engine == "acme"


def test_from_env_missing_required_raises_configuration_required() -> None:
    """A failure caused SOLELY by absent required fields is the narrow subtype.

    "Absent" is a fact about the environment (credential not set) that the
    compliance suite skips rather than fails; the body's comment records the
    full classification contract.
    """

    class _NeedsKey(BaseConfig[Literal["n"]]):
        engine: Literal["n"] = "n"
        required_field: str

    # EC-1 plus classification: a failure caused SOLELY by absent required
    # fields is the narrow ConfigurationRequiredError -- a fact about the
    # environment (credential not set), machine-distinguishable from an
    # invalid supplied value so the compliance suite can skip rather than
    # fail it. Still a ConfigError and a ValueError (existing handlers keep
    # working), still carrying the structured entries.
    with pytest.raises(ConfigurationRequiredError) as excinfo:
        _NeedsKey.from_env("n", environ={})
    assert isinstance(excinfo.value, ConfigError)
    assert isinstance(excinfo.value, ValueError)
    details = excinfo.value.details
    assert details is not None
    assert any(entry["loc"] == ["required_field"] for entry in details)


def test_from_env_missing_engine_discriminator_is_not_absence() -> None:
    """A subclass that forgot its ``engine`` default is a DECLARATION bug.

    ``engine`` is env-excluded (it can never be environment-supplied), so its
    absence is not a fact about the environment -- classifying it as
    ConfigurationRequiredError made compliance SKIP a broken plugin as
    "credentials missing". It must stay a plain ConfigError (a defect).
    """

    class _ForgotEngine(BaseConfig[Literal["broken"]]):
        pass  # no `engine: Literal["broken"] = "broken"` pin

    with pytest.raises(ConfigError) as excinfo:
        _ForgotEngine.from_env("broken", environ={})
    assert not isinstance(excinfo.value, ConfigurationRequiredError)


def test_from_env_nested_missing_field_is_not_absence() -> None:
    """A supplied-but-incomplete nested value (outer object given, inner
    required field missing) is a defect in the supplied value, never
    environment absence: the loc is deeper than one level.
    """

    class _Auth(BaseModel):
        model_config = ConfigDict(extra="forbid")

        token: str

    class _NestedCfg(BaseConfig[Literal["n"]]):
        engine: Literal["n"] = "n"
        auth: _Auth

    with pytest.raises(ConfigError) as excinfo:
        _NestedCfg.from_env("n", environ={}, auth={})
    assert not isinstance(excinfo.value, ConfigurationRequiredError)


def test_from_env_missing_aliased_credential_is_absence() -> None:
    """A provider-native alias resolves back to its env-fillable own field.

    Pydantic keys its errors by ALIAS: a required
    ``Field(alias="xi-api-key")`` credential reports ``loc=("xi-api-key",)``,
    which is not a ``model_fields`` key. Rejecting that token as unknown
    re-created exactly the env-dependent verdict the classifier exists to
    kill (plain ConfigError -> compliance FAIL on a clean CI; pass with the
    env var set). The alias must resolve to ``xi_api_key`` and classify as
    absence.
    """

    from pydantic import Field

    class _AliasedCfg(BaseConfig[Literal["eleven"]]):
        engine: Literal["eleven"] = "eleven"
        xi_api_key: SecretStr = Field(
            alias="xi-api-key",
            json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
        )

    with pytest.raises(ConfigurationRequiredError):
        _AliasedCfg.from_env("eleven", environ={})
    # The attribute-named env var (env_var_name derives from the FIELD name,
    # populate_by_name accepts it) still constructs normally.
    cfg = _AliasedCfg.from_env("eleven", environ={"STANDARD_ASR_ELEVEN__XI_API_KEY": "sk-e2e"})
    assert cfg.xi_api_key.get_secret_value() == "sk-e2e"


def test_alias_path_is_rejected_at_class_definition() -> None:
    """An ``AliasPath`` validation alias is a definition-time ``TypeError``.

    A path alias populates a field from NESTED input, which neither the flat
    ``STANDARD_ASR_<ENGINE>__<FIELD>`` env convention (IC.4) nor the
    absent-vs-invalid classifier's single-token loc resolution can express:
    pre-guard, a pure absence produced ``loc=("auth", "token")`` and was
    misclassified as a plugin defect -- an env-dependent compliance verdict.
    Fail loudly where the author can fix it: at class definition.
    """
    from pydantic import AliasPath, Field

    with pytest.raises(TypeError, match="AliasPath"):

        class _PathAliasCfg(BaseConfig[Literal["p"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["p"] = "p"
            token: str = Field(validation_alias=AliasPath("auth", "token"))


def test_alias_choices_with_path_entry_is_rejected_at_class_definition() -> None:
    """An ``AliasChoices`` smuggling an ``AliasPath`` entry is rejected too."""
    from pydantic import AliasChoices, AliasPath, Field

    with pytest.raises(TypeError, match="non-string"):

        class _MixedChoicesCfg(BaseConfig[Literal["m"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["m"] = "m"
            token: str = Field(validation_alias=AliasChoices("token", AliasPath("auth", "token")))


def test_all_string_alias_choices_classifies_absence() -> None:
    """All-string ``AliasChoices`` is supported end to end.

    Each choice resolves like a plain string alias, so pure absence (pydantic
    keys the missing error by the FIRST choice) classifies as
    ``ConfigurationRequiredError``, and the canonical field name still
    env-fills via ``populate_by_name``.
    """
    from pydantic import AliasChoices, Field

    class _ChoicesCfg(BaseConfig[Literal["ch"]]):
        engine: Literal["ch"] = "ch"
        api_key: SecretStr = Field(
            validation_alias=AliasChoices("apiKey", "api-key"),
            json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
        )

    with pytest.raises(ConfigurationRequiredError):
        _ChoicesCfg.from_env("ch", environ={})
    cfg = _ChoicesCfg.from_env("ch", environ={"STANDARD_ASR_CH__API_KEY": "sk-choice"})
    assert cfg.api_key.get_secret_value() == "sk-choice"


def test_secret_whitespace_survives_every_alias_input_key() -> None:
    """Whitespace fidelity holds for EVERY key a secret can arrive under.

    The whitespace-preserving pre-validator used to wrap only the canonical
    name (or, failing that, `alias`); a raw credential passed under a string
    `validation_alias` or an `AliasChoices` choice fell through to
    ``str_strip_whitespace``'s silent trim -- a silently wrong secret, the
    exact state the wrapper exists to prevent. All flat input keys now share
    one vocabulary (`_flat_input_keys`) with the absence classifier.
    """
    from pydantic import AliasChoices, Field

    padded = "  sk-padded-secret  "

    class _ChoicesSecretCfg(BaseConfig[Literal["cw"]]):
        engine: Literal["cw"] = "cw"
        api_key: SecretStr = Field(
            validation_alias=AliasChoices("apiKey", "api-key"),
            json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
        )

    for key in ("apiKey", "api-key"):
        cfg = _ChoicesSecretCfg.model_validate({"engine": "cw", key: padded})
        assert cfg.api_key.get_secret_value() == padded, key

    class _ValidationAliasSecretCfg(BaseConfig[Literal["va"]]):
        engine: Literal["va"] = "va"
        token: SecretStr = Field(
            validation_alias="x-token",
            json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
        )

    cfg2 = _ValidationAliasSecretCfg.model_validate({"engine": "va", "x-token": padded})
    assert cfg2.token.get_secret_value() == padded


def test_secret_supplied_under_both_keys_is_rejected_loudly_and_masked() -> None:
    """Alias + canonical key together: a loud duplicate, with BOTH values masked.

    ``extra="forbid"`` makes the dual supply a construction error (pydantic
    consumes the alias and rejects the unconsumed canonical key) -- there is
    no silent winner to mistrust. What the multi-key wrap buys here is the
    ERROR SURFACE: the rejected duplicate's echoed ``input_value`` is a
    masked ``SecretStr``, not the raw credential, because every present key
    was wrapped before validation ran.
    """
    from pydantic import Field

    padded = "  sk-alias-dup  "

    class _AliasedSecretCfg(BaseConfig[Literal["aw"]]):
        engine: Literal["aw"] = "aw"
        xi_api_key: SecretStr = Field(
            alias="xi-api-key",
            json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
        )

    with pytest.raises(ValidationError) as excinfo:
        _AliasedSecretCfg.model_validate(
            {"engine": "aw", "xi-api-key": padded, "xi_api_key": "  other-secret  "}
        )
    rendered = str(excinfo.value)
    assert "sk-alias-dup" not in rendered
    assert "other-secret" not in rendered
    # The single-key path (either key alone) still constructs with full
    # whitespace fidelity.
    alone = _AliasedSecretCfg.model_validate({"engine": "aw", "xi-api-key": padded})
    assert alone.xi_api_key.get_secret_value() == padded


def test_absence_classifier_alias_choices_other_choice_supplied_is_not_absence() -> None:
    """A value under ANY choice key counts as supplied for the classifier.

    The shared flat-key vocabulary keeps the classifier honest: an invalid
    value supplied under one `AliasChoices` key must classify as a defect
    (plain ConfigError), never as environment absence, regardless of which
    choice pydantic keyed the error by.
    """
    from pydantic import AliasChoices, Field

    class _ChoicesIntCfg(BaseConfig[Literal["ci"]]):
        engine: Literal["ci"] = "ci"
        retries: int = Field(validation_alias=AliasChoices("retries_alias", "retry-count"))

    with pytest.raises(ConfigError) as excinfo:
        _ChoicesIntCfg.from_env("ci", environ={}, **{"retry-count": "not-an-int"})
    assert not isinstance(excinfo.value, ConfigurationRequiredError)


def test_absence_classifier_alias_ambiguity_is_now_impossible_by_construction() -> None:
    """Two fields sharing one alias no longer reach the classifier at all.

    The classifier's ambiguity arm used to fail closed at ERROR time; guard 4
    moves the failure to class DEFINITION (the ambiguous vocabulary is the
    defect, not the error that later trips over it), so the runtime never has
    to disambiguate a loc token.
    """
    from pydantic import Field as PField

    with pytest.raises(TypeError, match="claimed by both"):

        class _Ambiguous(BaseConfig[Literal["n"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["n"] = "n"
            first: str = PField(default="x", validation_alias="shared")
            second: str = PField(default="y", validation_alias="shared")


def test_absence_classifier_alias_supplied_in_merged_is_not_absence() -> None:
    """A field whose ALIAS appears in the merged input yet still reports
    missing is a population mismatch -- a defect, never absence.
    """
    from pydantic import Field as PField
    from pydantic import ValidationError as PydanticValidationError

    class _Aliased(BaseConfig[Literal["n"]]):
        engine: Literal["n"] = "n"
        api_key: str = PField(alias="apiKey")

    exc = PydanticValidationError.from_exception_data(
        "X", [{"type": "missing", "loc": ("apiKey",), "input": {}}]
    )
    assert (
        _Aliased._failure_is_absent_env_config(  # pyright: ignore[reportPrivateUsage]
            exc, {"apiKey": "supplied"}
        )
        is False
    )


def test_absence_classifier_rejects_supplied_and_unknown_names() -> None:
    """Direct unit coverage of the classifier's remaining conservative
    rejections: a field reported missing DESPITE appearing in the merged
    input (an alias/population mismatch) and a loc naming no own field
    (for example, an alias string) are both defects, never absence.
    """
    from pydantic import ValidationError as PydanticValidationError

    class _NeedsKey(BaseConfig[Literal["n"]]):
        engine: Literal["n"] = "n"
        api_key: str

    supplied_yet_missing = PydanticValidationError.from_exception_data(
        "X", [{"type": "missing", "loc": ("api_key",), "input": {}}]
    )
    assert (
        _NeedsKey._failure_is_absent_env_config(  # pyright: ignore[reportPrivateUsage]
            supplied_yet_missing, {"api_key": "supplied"}
        )
        is False
    )
    unknown_name = PydanticValidationError.from_exception_data(
        "X", [{"type": "missing", "loc": ("apiKeyAlias",), "input": {}}]
    )
    assert (
        _NeedsKey._failure_is_absent_env_config(  # pyright: ignore[reportPrivateUsage]
            unknown_name, {}
        )
        is False
    )


def test_from_env_invalid_supplied_value_stays_plain_config_error() -> None:
    """An invalid SUPPLIED value is a defect, never the absence state.

    Even when a required field is ALSO missing, the supplied defect is the
    actionable fault, so ``ConfigurationRequiredError`` must not be raised.
    """

    class _NeedsKey(BaseConfig[Literal["n"]]):
        engine: Literal["n"] = "n"
        required_field: str
        count: int = 0

    # An invalid SUPPLIED value is a real configuration defect (someone must
    # fix the value), never the absence state -- even when a required field
    # is ALSO missing, the supplied defect is the actionable fault, so the
    # narrow subtype must NOT be raised.
    with pytest.raises(ConfigError) as excinfo:
        _NeedsKey.from_env("n", environ={"STANDARD_ASR_N__COUNT": "not-an-int"})
    assert not isinstance(excinfo.value, ConfigurationRequiredError)


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


# --------------------------------------------------------------------------- #
# Round-4: bytes-input secret fidelity and cross-field key uniqueness.
# --------------------------------------------------------------------------- #


class _BytesFidelityCfg(BaseConfig[Literal["bf"]]):
    """A config with one carrier of each kind for bytes-input tests."""

    engine: Literal["bf"] = "bf"
    str_token: SecretStr | None = secret_field()
    bytes_token: SecretBytes | None = secret_field()


def test_secret_str_field_preserves_bytes_input_exactly() -> None:
    """A bytes credential for a ``SecretStr`` field keeps its exact contents.

    pydantic's lax ``bytes -> str`` coercion runs the decoded text through
    ``str_strip_whitespace``, silently trimming a padded credential --
    the same paste-error masking the raw-str wrap already prevents. The
    pre-validator now decodes (UTF-8, mirroring pydantic) and wraps first.
    """
    cfg = _BytesFidelityCfg(str_token=b"  padded  ")  # pyright: ignore[reportArgumentType]
    assert cfg.str_token is not None
    assert cfg.str_token.get_secret_value() == "  padded  "

    via_bytearray = _BytesFidelityCfg(str_token=bytearray(b"\tabc\n"))  # pyright: ignore[reportArgumentType]
    assert via_bytearray.str_token is not None
    assert via_bytearray.str_token.get_secret_value() == "\tabc\n"


def test_secret_str_field_rejects_non_utf8_bytes_loudly() -> None:
    """Invalid UTF-8 bytes for a ``SecretStr`` field fail loudly, never guess."""
    with pytest.raises(ValidationError):
        _BytesFidelityCfg(str_token=b"\xff\xfe\x00")  # pyright: ignore[reportArgumentType]


def test_secret_bytes_field_preserves_bytes_and_bytearray_unchanged() -> None:
    """Bytes-like input for a ``SecretBytes`` field is wrapped verbatim."""
    cfg = _BytesFidelityCfg(bytes_token=b"  raw \xff bytes  ")  # pyright: ignore[reportArgumentType]
    assert cfg.bytes_token is not None
    assert cfg.bytes_token.get_secret_value() == b"  raw \xff bytes  "

    via_bytearray = _BytesFidelityCfg(bytes_token=bytearray(b" ba "))  # pyright: ignore[reportArgumentType]
    assert via_bytearray.bytes_token is not None
    assert via_bytearray.bytes_token.get_secret_value() == b" ba "


def test_cross_field_alias_collision_with_canonical_name_rejected() -> None:
    """An alias claiming another field's canonical name fails at definition.

    ``populate_by_name`` fills BOTH fields from the one input key -- one
    caller value silently controlling two independent settings.
    """
    with pytest.raises(TypeError, match="claimed by both"):

        class _AliasVsName(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            token: str = "a"
            other: str = Field(default="b", alias="token")


def test_cross_field_alias_vs_alias_collision_rejected() -> None:
    """Two fields sharing one alias spelling fail at definition."""
    with pytest.raises(TypeError, match="claimed by both"):

        class _AliasVsAlias(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            first: str = Field(default="a", alias="shared-key")
            second: str = Field(default="b", alias="shared-key")


def test_cross_field_alias_choices_overlap_rejected() -> None:
    """An ``AliasChoices`` choice overlapping another field's key fails."""
    with pytest.raises(TypeError, match="claimed by both"):

        class _ChoicesOverlap(BaseConfig[Literal["bad"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["bad"] = "bad"
            token: str = "a"
            other: str = Field(default="b", validation_alias=AliasChoices("alt", "token"))


def test_same_field_duplicate_keys_are_fine() -> None:
    """A field whose alias repeats its own name dedupes, never collides."""

    class _SelfAlias(BaseConfig[Literal["ok"]]):
        engine: Literal["ok"] = "ok"
        token: str = Field(default="a", validation_alias=AliasChoices("token", "alt-token"))

    assert _SelfAlias.model_validate({"alt-token": "x"}).token == "x"


# --------------------------------------------------------------------------- #
# Round-5: secret exactness holds for EVERY Mapping input, not just dict.
# --------------------------------------------------------------------------- #


class _ReadOnlyMapping(Mapping[str, object]):
    """A minimal custom read-only Mapping (not a dict subclass)."""

    def __init__(self, data: dict[str, object]) -> None:
        """Wrap the given data.

        Args:
            data: The underlying key/value pairs.
        """
        self._data = data

    def __getitem__(self, key: str) -> object:
        """Return the value for ``key``.

        Args:
            key: The lookup key.

        Returns:
            The stored value.
        """
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate the keys.

        Returns:
            An iterator over the keys.
        """
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of entries.

        Returns:
            The entry count.
        """
        return len(self._data)


def test_secret_exactness_holds_for_mapping_proxy_input() -> None:
    """The round-5 counterexample: ``MappingProxyType`` bypassed the wrapper.

    The pre-validator gated on ``isinstance(data, dict)``, but
    ``model_validate`` accepts any ``Mapping`` -- a read-only proxy sailed
    past the wrap and ``str_strip_whitespace`` silently rewrote
    ``"  padded  "`` to ``"padded"``: the exact silent credential rewrite
    the validator exists to forbid.
    """
    from types import MappingProxyType

    cfg = _BytesFidelityCfg.model_validate(
        MappingProxyType({"engine": "bf", "str_token": "  padded  "})
    )
    assert cfg.str_token is not None
    assert cfg.str_token.get_secret_value() == "  padded  "


def test_secret_exactness_holds_for_custom_read_only_mapping() -> None:
    """A custom non-dict Mapping gets the same exact-contents contract.

    Covers both carriers, str/bytes/bytearray inputs, and alias keys --
    the full wrap matrix, through the Mapping path.
    """
    from pydantic import AliasChoices as _AC
    from pydantic import Field as _F

    cfg = _BytesFidelityCfg.model_validate(
        _ReadOnlyMapping({"engine": "bf", "str_token": b"  by \tte  ", "bytes_token": " raw str "})
    )
    assert cfg.str_token is not None
    assert cfg.str_token.get_secret_value() == "  by \tte  "
    assert cfg.bytes_token is not None
    assert cfg.bytes_token.get_secret_value() == b" raw str "

    class _AliasedMapCfg(BaseConfig[Literal["amc"]]):
        engine: Literal["amc"] = "amc"
        token: SecretStr | None = _F(
            default=None,
            validation_alias=_AC("token", "xi-token"),
            json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
        )

    via_alias = _AliasedMapCfg.model_validate(_ReadOnlyMapping({"xi-token": "  spaced  "}))
    assert via_alias.token is not None
    assert via_alias.token.get_secret_value() == "  spaced  "

    # The caller's read-only mapping is untouched (and no TypeError from
    # attempting to write through it): the wrap works on a copy.
    frozen = _ReadOnlyMapping({"engine": "bf", "str_token": " x "})
    cfg2 = _BytesFidelityCfg.model_validate(frozen)
    assert cfg2.str_token is not None
    assert cfg2.str_token.get_secret_value() == " x "
    assert isinstance(frozen["str_token"], str)  # still the caller's plain str


def test_input_key_vocabulary_is_validation_only() -> None:
    """A serialization-only ``alias`` is not an input key.

    Pydantic treats ``alias`` as both directions only until a
    ``validation_alias`` overrides it; from then on ``alias`` is
    serialization-only and pydantic REJECTS it on input. Treating it as an
    input key made all three consumers of this vocabulary believe a key
    could populate a field that pydantic never accepts from it.
    """
    from pydantic import Field

    class _SplitAliasCfg(BaseConfig[Literal["sa"]]):
        engine: Literal["sa"] = "sa"
        api_key: SecretStr | None = Field(
            default=None,
            alias="serializedKey",
            validation_alias="apiKey",
            json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
        )

    keys = _SplitAliasCfg._flat_input_keys(  # pyright: ignore[reportPrivateUsage]
        "api_key", _SplitAliasCfg.model_fields["api_key"]
    )
    assert keys == ("api_key", "apiKey")

    # Pydantic's actual behavior, pinned: the validation alias populates and
    # the serialization alias is extra.
    assert _SplitAliasCfg.model_validate({"apiKey": "sk-1"}).api_key is not None
    with pytest.raises(ValidationError):
        _SplitAliasCfg.model_validate({"serializedKey": "sk-1"})

    # `alias` alone IS a validation key (nothing overrides it).
    class _PlainAliasCfg(BaseConfig[Literal["pa"]]):
        engine: Literal["pa"] = "pa"
        region: str = Field(default="eu", alias="x-region")

    plain_keys = _PlainAliasCfg._flat_input_keys(  # pyright: ignore[reportPrivateUsage]
        "region", _PlainAliasCfg.model_fields["region"]
    )
    assert plain_keys == ("region", "x-region")
    assert _PlainAliasCfg.model_validate({"x-region": "us"}).region == "us"


def test_serialization_alias_may_spell_another_fields_name() -> None:
    """The collision guard polices INPUT keys, so this declaration is legal.

    ``b``'s serialization alias spelling ``a``'s field name cannot make one
    input key populate two fields -- ``a`` validates from ``a``, ``b`` from
    ``bb`` -- yet the guard rejected the class at definition time because its
    vocabulary conflated the two directions.
    """
    from pydantic import Field

    class _SplitCfg(BaseConfig[Literal["sc"]]):
        engine: Literal["sc"] = "sc"
        a: str = "x"
        b: str = Field(default="y", alias="a", validation_alias="bb")

    cfg = _SplitCfg.model_validate({"a": "from-a", "bb": "from-bb"})
    assert (cfg.a, cfg.b) == ("from-a", "from-bb")

    # A genuine input-key collision is still rejected.
    with pytest.raises(TypeError, match="claimed by both"):

        class _CollidingCfg(BaseConfig[Literal["cc"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["cc"] = "cc"
            a: str = "x"
            b: str = Field(default="y", validation_alias="a")


def test_duck_typed_serialization_is_rejected_at_definition() -> None:
    # THE bypass the decorator/metadata enumeration could not see:
    # SerializeAsAny serializes the RUNTIME object, so a subclass that adds a
    # credential plus a computed field rematerialized the plaintext inside
    # public_dump -- under a key the by-name mask never looks at, from a
    # class the nested-carrier scan never sees (the declared type is clean).
    from pydantic import SerializeAsAny

    class PublicView(BaseModel):
        model_config = ConfigDict(extra="forbid")

        name: str = "account"

    with pytest.raises(TypeError, match="DUCK-TYPED"):

        class _DuckCfg(BaseConfig[Literal["duck"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["duck"] = "duck"
            view: SerializeAsAny[PublicView] = PublicView()

    # Reachable through containers, unions, and mappings alike -- the proof
    # walks the schema, so nesting cannot hide the marker.
    with pytest.raises(TypeError, match="DUCK-TYPED"):

        class _ListCfg(BaseConfig[Literal["dl"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["dl"] = "dl"
            views: list[SerializeAsAny[PublicView]] | None = None

    with pytest.raises(TypeError, match="DUCK-TYPED"):

        class _OptCfg(BaseConfig[Literal["do"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["do"] = "do"
            view: SerializeAsAny[PublicView] | None = None

    with pytest.raises(TypeError, match="DUCK-TYPED"):

        class _DictCfg(BaseConfig[Literal["dd"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["dd"] = "dd"
            views: dict[str, SerializeAsAny[PublicView]] | None = None

    # The same declared field WITHOUT the marker is fine: pydantic then
    # serializes by the declared type, so a runtime subclass adds nothing.
    class _PlainCfg(BaseConfig[Literal["dp"]]):
        engine: Literal["dp"] = "dp"
        view: PublicView = PublicView()

    class _Extended(PublicView):
        api_key: SecretStr = secret_field(...)

    cfg = _PlainCfg(engine="dp", view=_Extended(name="account", api_key=SecretStr("sk-RUNTIME")))
    dumped = cfg.public_dump()
    assert "sk-RUNTIME" not in str(dumped)
    assert dumped["view"] == {"name": "account"}


def test_nested_submodel_serialization_hooks_are_rejected() -> None:
    # A nested submodel's OWN hooks run inside the parent's dump (its
    # computed field is emitted as a key of the nested object), so the class
    # hook's decorator registry -- which only sees the config class itself --
    # was blind to them.
    from pydantic import PlainSerializer, computed_field, model_serializer

    class ComputedLeaf(BaseModel):
        a: int = 1

        @computed_field  # pyright: ignore[reportAny]
        @property
        def derived(self) -> int:
            """A derived key inside the nested object.

            Returns:
                Twice ``a``.
            """
            return self.a * 2

    with pytest.raises(TypeError, match="computed field"):

        class _NestedComputedCfg(BaseConfig[Literal["nc2"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["nc2"] = "nc2"
            leaf: ComputedLeaf = ComputedLeaf()

    class SerializingLeaf(BaseModel):
        a: int = 1

        @model_serializer
        def _ser(self) -> dict[str, int]:
            """Rewrite the nested object's dump.

            Returns:
                An arbitrary replacement payload.
            """
            return {"rewritten": self.a}

    with pytest.raises(TypeError, match="author-defined serializer"):

        class _NestedSerCfg(BaseConfig[Literal["ns2"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["ns2"] = "ns2"
            leaf: SerializingLeaf = SerializingLeaf()

    # Annotated serializer metadata on a NESTED submodel's field, and a hook
    # buried two levels down: the walk recurses through submodel fields.
    class AnnotatedLeaf(BaseModel):
        a: Annotated[int, PlainSerializer(lambda v: v + 1)] = 1

    with pytest.raises(TypeError, match="PlainSerializer/WrapSerializer"):

        class _NestedAnnCfg(BaseConfig[Literal["na2"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["na2"] = "na2"
            leaf: AnnotatedLeaf = AnnotatedLeaf()

    class DeepMiddle(BaseModel):
        inner: ComputedLeaf = ComputedLeaf()

    with pytest.raises(TypeError, match="computed field"):

        class _DeepCfg(BaseConfig[Literal["dp2"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["dp2"] = "dp2"
            middle: DeepMiddle = DeepMiddle()


def test_excluded_field_is_rejected_at_definition() -> None:
    # exclude=True is the other direction of the same round-trip: the dump
    # documented for persistence silently loses a declared input, so
    # reloading it yields a different config with no diagnostic.
    with pytest.raises(TypeError, match="excluded from the dump"):

        class _ExcludingCfg(BaseConfig[Literal["ex"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["ex"] = "ex"
            beam: int = Field(default=5, exclude=True)


def test_undeclared_value_shapes_are_rejected_at_definition() -> None:
    # The duck-typing channel reached from the declared TYPE rather than
    # from a marker: an `Any`/`object`/unparametrized-container field has no
    # serialization node at all, yet pydantic hands whatever object it holds
    # at runtime to that object's OWN serializer -- so a submodel stored
    # there emits its computed fields, credentials included. It also makes
    # the config unrenderable (G3.1: the JSON Schema for `Any` is empty).
    for annotation in (Any, object, dict, list):
        with pytest.raises(TypeError, match="undeclared value shape"):
            type(
                "_AnyCfg",
                (BaseConfig[Literal["any"]],),
                {"__annotations__": {"engine": Literal["any"], "opts": annotation | None}},
            )

    with pytest.raises(TypeError, match="undeclared value shape"):

        class _MappingCfg(BaseConfig[Literal["m2"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["m2"] = "m2"
            opts: dict[str, Any] | None = None

    # An untyped DISCRIMINATOR is the same defect reached through the
    # generic parameter, and is refused with the same rule.
    with pytest.raises(TypeError, match="undeclared value shape"):

        class _UntypedEngine(BaseConfig[Any]):  # pyright: ignore[reportUnusedClass]
            pass

    # Fully declared shapes -- including a heterogeneous mapping spelled as
    # a union, the practical replacement for dict[str, Any] -- are fine, and
    # so is a discriminator typed as a plain str.
    class _DeclaredCfg(BaseConfig[Literal["ok"]]):
        engine: Literal["ok"] = "ok"
        opts: dict[str, str | int | bool | None] | None = None
        items: list[str] | None = None

    class _StrEngine(BaseConfig[str]):
        engine: str = "ok"

    cfg = _DeclaredCfg(engine="ok", opts={"beam": 5, "vad": True, "lang": "en", "none": None})
    assert cfg.public_dump()["opts"] == {"beam": 5, "vad": True, "lang": "en", "none": None}
    assert _StrEngine().public_dump()["engine"] == "ok"


def test_any_typed_field_would_have_leaked_a_runtime_credential() -> None:
    # The counterexample the rejection exists for, pinned on a plain model
    # (a config can no longer declare the field): a value parked in an
    # untyped slot is dumped by ITS OWN serializer, so a computed field
    # rematerializes the credential -- and a config's by-name mask, which
    # only knows the config's own secret fields, never sees the key.
    from pydantic import computed_field

    class _CredentialView(BaseModel):
        api_key: SecretStr

        @computed_field  # pyright: ignore[reportAny]
        @property
        def authorization(self) -> str:
            """Derive the header the way an ordinary view model would.

            Returns:
                The bearer header, credential included.
            """
            return "Bearer " + self.api_key.get_secret_value()

    class _Untyped(BaseModel):
        opts: Any = None

    leaked = _Untyped(opts=_CredentialView(api_key=SecretStr("sk-DUCK"))).model_dump(mode="json")
    assert leaked["opts"]["api_key"] == "**********"
    assert leaked["opts"]["authorization"] == "Bearer sk-DUCK"


def test_secret_extraction_is_the_closure_boundary() -> None:
    # Executable statement of the closure proof's BOUNDARY (public_dump's
    # docstring + spec 776): the proof bounds what the SCHEMA installs in
    # the dump, not the CONTENTS of values author code constructed. Author
    # code that copies a secret out of its carrier into non-secret state
    # leaks it through a dump the proof accepts -- and must, because no
    # serialization mechanism can classify a plain str's content as secret.
    # Two costumes of the SAME author act are pinned as equivalent; if a
    # future mechanism moves this boundary, this test, the docstring, and
    # the spec must move together.
    from pydantic import field_validator, model_validator

    # Costume 1: the secret is copied into a plain DECLARED str field. Every
    # value the dump touches has a declared shape and an audited serializer;
    # the leak is pure content, invisible to any type or dispatch audit.
    # The frozen config makes the NAIVE assignment loud (a real mitigation,
    # pinned below), so the extraction must announce itself with an explicit
    # override -- but it remains schema-legal and the dump emits it.
    class _CopiedCfg(BaseConfig[Literal["bd1"]]):
        engine: Literal["bd1"] = "bd1"
        api_key: SecretStr = secret_field(...)
        note: str = ""

        @model_validator(mode="after")
        def _copy_out(self) -> "_CopiedCfg":
            object.__setattr__(self, "note", "Bearer " + self.api_key.get_secret_value())
            return self

    with pytest.raises(ValidationError, match="frozen"):

        class _NaiveCfg(BaseConfig[Literal["bd0"]]):
            engine: Literal["bd0"] = "bd0"
            api_key: SecretStr = secret_field(...)
            note: str = ""

            @model_validator(mode="after")
            def _copy_out(self) -> "_NaiveCfg":
                self.note = "x"  # plain assignment: frozen rejects it loudly
                return self

        _NaiveCfg(api_key=SecretStr("sk-REAL"))

    copied = _CopiedCfg(api_key=SecretStr("sk-REAL")).public_dump()
    assert copied["api_key"] == SECRET_MASK  # the carrier itself stays masked
    assert copied["note"] == "Bearer sk-REAL"  # the extracted copy is out of reach

    # Costume 2: the secret rides display state that a TRUSTED serializer
    # renders by builtin dispatch -- a validator-returned Path SUBCLASS
    # whose __str__ embeds it (pydantic's own audited ser_path calls
    # str() on the runtime value, on the 2.5 floor and current alike).
    # Same act (get_secret_value into non-secret state), same leak -- and
    # this one mutates the VALUE, which the frozen config never covers.
    concrete_path = type(Path())

    class _LeakyPath(concrete_path):  # pyright: ignore[reportGeneralTypeIssues,reportUntypedBaseClass]
        leaked: str = ""

        def __str__(self) -> str:
            return "Bearer " + self.leaked

    class _SmuggledCfg(BaseConfig[Literal["bd2"]]):
        engine: Literal["bd2"] = "bd2"
        api_key: SecretStr = secret_field(...)
        model_dir: Path

        @field_validator("model_dir", mode="before")
        @classmethod
        def _smuggle(cls, value: object) -> Path:
            return _LeakyPath(cast("Any", value))

        @model_validator(mode="after")
        def _copy_out(self) -> "_SmuggledCfg":
            cast("Any", self.model_dir).leaked = self.api_key.get_secret_value()
            return self

    smuggled = _SmuggledCfg(api_key=SecretStr("sk-REAL"), model_dir=Path("/m")).public_dump()
    assert smuggled["api_key"] == SECRET_MASK
    assert smuggled["model_dir"] == "Bearer sk-REAL"  # same leak, different costume


def test_dict_default_payload_is_data_not_schema() -> None:
    """A default value spelling schema words must not be misread as schema.

    The core-schema walks (env-codec classification, the nested
    extra-keys depth guard) visit every node value -- and a ``default``
    wrapper node stores the field's LITERAL default under its ``default``
    key. A legal config whose dict default contains ``{"type": "str"}`` was
    classified as both scalar and structured (spurious Guard 7 ambiguity),
    and ``{"type": "model"}`` read as an open nested input container --
    definition-time TypeErrors against data, not schema.
    """

    class _Profiles(BaseConfig[Literal["pd"]]):
        engine: Literal["pd"] = "pd"
        profiles: dict[str, str] = {"type": "str"}

    class _Table(BaseConfig[Literal["td"]]):
        engine: Literal["td"] = "td"
        table: dict[str, str] = {"type": "model"}

    assert _Profiles._ENV_CODECS["profiles"] == "json"  # pyright: ignore[reportPrivateUsage]
    assert _Table().public_dump()["table"] == {"type": "model"}


def test_reopened_input_surface_is_rejected_at_definition() -> None:
    # extra="forbid" is load-bearing, not stylistic: the flat input-key
    # vocabulary, the absent-vs-invalid classifier, the typo-names-the-key
    # DX, and public_dump's by-name mask all assume every accepted key
    # belongs to a declared field.
    with pytest.raises(TypeError, match="input surface must stay closed"):

        class _AllowCfg(BaseConfig[Literal["al2"]]):  # pyright: ignore[reportUnusedClass]
            model_config = ConfigDict(extra="allow")
            engine: Literal["al2"] = "al2"

    # 'ignore' is refused too: it swallows a mistyped credential key, which
    # then reads as an absent credential instead of a loud error.
    with pytest.raises(TypeError, match="input surface must stay closed"):

        class _IgnoreCfg(BaseConfig[Literal["ig"]]):  # pyright: ignore[reportUnusedClass]
            model_config = ConfigDict(extra="ignore")
            engine: Literal["ig"] = "ig"

    # A nested submodel storing undeclared keys carries them into the
    # parent's dump the same way; the depth guard (Guard 0 at depth)
    # refuses it.
    class _LooseLeaf(BaseModel):
        model_config = ConfigDict(extra="allow")
        beam: int = 4

    with pytest.raises(TypeError, match="extra-keys policy"):

        class _NestedLooseCfg(BaseConfig[Literal["nl"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["nl"] = "nl"
            decode: _LooseLeaf = _LooseLeaf()

    # The counterexample that motivates it, on a plain model: pydantic keeps
    # the smuggled key and the dump emits it verbatim.
    smuggled = _LooseLeaf.model_validate({"beam": 4, "api_key": "sk-SMUGGLED"})
    assert smuggled.model_dump()["api_key"] == "sk-SMUGGLED"

    # A closed submodel (extra='forbid') is the accepted shape; the default
    # ('ignore') is refused at depth too -- see
    # test_nested_input_container_must_forbid_undeclared_keys.
    class _TightLeaf(BaseModel):
        model_config = ConfigDict(extra="forbid")

        beam: int = 4

    class _OkCfg(BaseConfig[Literal["ok2"]]):
        engine: Literal["ok2"] = "ok2"
        decode: _TightLeaf = _TightLeaf()

    assert _OkCfg().public_dump()["decode"] == {"beam": 4}


def test_nested_input_container_must_forbid_undeclared_keys() -> None:
    # The counterexample that motivates the rule, on a plain model: pydantic's
    # DEFAULT ('ignore') accepts a typo'd key and silently drops it -- the
    # caller reads their setting as applied while the engine runs on the
    # field's default. Same for a misplaced credential key, which then reads
    # as an absent credential instead of a loud error.
    class _Loose(BaseModel):
        beam: int = 4

    swallowed = _Loose.model_validate({"beam": 8, "baem": 999})
    assert swallowed.beam == 8
    assert swallowed.__pydantic_extra__ is None  # "baem" vanished, no diagnostic

    # A config reaching that submodel is therefore refused at definition.
    with pytest.raises(TypeError, match=r"nested input container _Loose.*extra-keys policy"):

        class _DefaultCfg(BaseConfig[Literal["nd1"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["nd1"] = "nd1"
            decode: _Loose = _Loose()

    # An EXPLICIT 'ignore' is the same behavior spelled out.
    class _Ignoring(BaseModel):
        model_config = ConfigDict(extra="ignore")

        beam: int = 4

    with pytest.raises(TypeError, match="nested input container _Ignoring"):

        class _IgnoreCfg2(BaseConfig[Literal["nd2"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["nd2"] = "nd2"
            decode: _Ignoring = _Ignoring()

    # The walk is over the SCHEMA, so depth and containers cannot hide the
    # open leaf: a closed middle model with an open inner one, and a leaf
    # reached only through ``dict[str, ...]``, are both found.
    class _OpenLeaf(BaseModel):
        rate: int = 16000

    class _ClosedMiddle(BaseModel):
        model_config = ConfigDict(extra="forbid")

        inner: _OpenLeaf = _OpenLeaf()

    with pytest.raises(TypeError, match="nested input container _OpenLeaf"):

        class _DeepCfg(BaseConfig[Literal["nd3"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["nd3"] = "nd3"
            middle: _ClosedMiddle = _ClosedMiddle()

    with pytest.raises(TypeError, match="nested input container _OpenLeaf"):

        class _MappedCfg(BaseConfig[Literal["nd4"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["nd4"] = "nd4"
            profiles: dict[str, _OpenLeaf] = Field(default_factory=dict)

    # The walk itself is fail-closed: a schema that cannot be introspected
    # is a gap, never a silent pass.
    from standard_asr.runtime.config import (
        _nested_input_surface_gap,  # pyright: ignore[reportPrivateUsage]
    )

    class _Unreadable:
        @property
        def __pydantic_core_schema__(self) -> object:
            """Fail introspection the way a mock/unbuilt schema would.

            Returns:
                Never returns.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("no schema")

    assert _nested_input_surface_gap(cast("Any", _Unreadable())) is not None


def test_nested_typo_key_is_rejected_loudly_not_swallowed() -> None:
    # End to end on the accepted shape: with the nested surface closed, the
    # typo'd key and the misplaced credential key both fail LOUDLY at the
    # nested level instead of vanishing.
    class _Decode(BaseModel):
        model_config = ConfigDict(extra="forbid")

        beam: int = 4

    class _Cfg(BaseConfig[Literal["nt"]]):
        engine: Literal["nt"] = "nt"
        decode: _Decode = _Decode()

    assert _Cfg.model_validate({"decode": {"beam": 8}}).decode.beam == 8
    with pytest.raises(ValidationError) as typo:
        _Cfg.model_validate({"decode": {"beam": 8, "baem": 999}})
    assert typo.value.errors()[0]["type"] == "extra_forbidden"
    with pytest.raises(ValidationError) as misplaced:
        _Cfg.model_validate({"decode": {"api_key": "sk-MISPLACED"}})
    assert misplaced.value.errors()[0]["type"] == "extra_forbidden"


def test_nested_typed_dict_and_dataclass_input_surfaces_must_be_closed() -> None:
    # The rule is about KEYED INPUT, not about BaseModel: a TypedDict and a
    # dataclass accept mapping input with the same silent-drop default. The
    # guard reads the EFFECTIVE policy from the core schema, so pydantic's
    # config-propagation rules are honored rather than re-derived: a bare
    # TypedDict and a bare STDLIB dataclass inherit the enclosing config's
    # extra='forbid' (closed for free, pinned here on both), while a PYDANTIC
    # dataclass owns its config (no inheritance) and must close it itself.
    import dataclasses

    import pydantic.dataclasses
    from typing_extensions import TypedDict  # 3.10-compatible for pydantic

    class _BareTD(TypedDict):
        beam: int

    @dataclasses.dataclass
    class _BareStdDC:
        rate: int = 16000

    class _InheritingCfg(BaseConfig[Literal["td1"]]):
        engine: Literal["td1"] = "td1"
        decode: _BareTD = {"beam": 4}
        io: _BareStdDC = dataclasses.field(default_factory=_BareStdDC)  # pyright: ignore[reportAssignmentType]

    with pytest.raises(ValidationError):
        _InheritingCfg.model_validate({"decode": {"beam": 8, "baem": 9}})
    with pytest.raises(ValidationError):
        _InheritingCfg.model_validate({"io": {"rate": 8000, "rte": 1}})

    # A TypedDict that REOPENS itself (own __pydantic_config__) is refused.
    class _ReopenedTD(TypedDict):
        __pydantic_config__ = ConfigDict(extra="ignore")  # pyright: ignore[reportGeneralTypeIssues]
        beam: int

    with pytest.raises(TypeError, match="nested input container"):

        class _TDCfg(BaseConfig[Literal["td2"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["td2"] = "td2"
            decode: _ReopenedTD = {"beam": 4}

    # A pydantic dataclass does NOT inherit the enclosing config: bare is
    # open (silently drops keys), so it is refused until it closes itself.
    @pydantic.dataclasses.dataclass
    class _OpenDC:
        beam: int = 4

    with pytest.raises(TypeError, match="nested input container _OpenDC"):

        class _DCCfg(BaseConfig[Literal["dc1"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["dc1"] = "dc1"
            decode: _OpenDC = _OpenDC()

    @pydantic.dataclasses.dataclass(config=ConfigDict(extra="forbid"))
    class _ClosedDC:
        beam: int = 4

    class _DCOkCfg(BaseConfig[Literal["dc2"]]):
        engine: Literal["dc2"] = "dc2"
        decode: _ClosedDC = _ClosedDC()

    with pytest.raises(ValidationError):
        _DCOkCfg.model_validate({"decode": {"beam": 8, "baem": 9}})


def test_env_codec_covers_every_shape_the_definition_guards_admit() -> None:
    # The origin whitelist was fail-open in the DX direction: a Mapping, a
    # Sequence, a TypedDict and a dataclass all pass every class-definition
    # guard -- they are ordinary, fully declared config shapes -- yet had no
    # origin entry, so each was unreachable through its own documented env
    # convention (the bare string hit the container validator and raised).
    # The codec now reads the core schema, so what the guards admit, the env
    # convention serves.
    import dataclasses
    from collections.abc import Sequence

    from typing_extensions import TypedDict

    class _Sub(BaseModel):
        model_config = ConfigDict(extra="forbid")

        beam: int = 4

    class _TD(TypedDict):
        beam: int

    @dataclasses.dataclass
    class _DC:
        beam: int = 4

    class _Cfg(BaseConfig[Literal["codec"]]):
        engine: Literal["codec"] = "codec"
        mapping: Mapping[str, int] = {}
        sequence: Sequence[str] = ()
        td: _TD = {"beam": 4}
        dc: _DC = dataclasses.field(default_factory=_DC)  # pyright: ignore[reportAssignmentType]
        sub: _Sub = _Sub()
        listed: list[str] = Field(default_factory=list)
        # ...and the scalars must keep passing through VERBATIM: JSON-decoding
        # a scalar would silently reinterpret "123" as the integer 123.
        name: str = ""
        count: int = 0
        where: Path | None = None
        api_key: SecretStr | None = secret_field()

    assert _Cfg._ENV_CODECS["mapping"] == "json"  # pyright: ignore[reportPrivateUsage]
    assert _Cfg._ENV_CODECS["sequence"] == "json"  # pyright: ignore[reportPrivateUsage]
    assert _Cfg._ENV_CODECS["td"] == "json"  # pyright: ignore[reportPrivateUsage]
    assert _Cfg._ENV_CODECS["dc"] == "json"  # pyright: ignore[reportPrivateUsage]
    assert _Cfg._ENV_CODECS["sub"] == "json"  # pyright: ignore[reportPrivateUsage]
    assert _Cfg._ENV_CODECS["listed"] == "json"  # pyright: ignore[reportPrivateUsage]
    for scalar in ("name", "count", "where", "api_key"):
        assert _Cfg._ENV_CODECS[scalar] == "raw", scalar  # pyright: ignore[reportPrivateUsage]

    built = _Cfg.from_env(
        "codec",
        environ={
            "STANDARD_ASR_CODEC__MAPPING": '{"a": 1}',
            "STANDARD_ASR_CODEC__SEQUENCE": '["x"]',
            "STANDARD_ASR_CODEC__TD": '{"beam": 8}',
            "STANDARD_ASR_CODEC__DC": '{"beam": 8}',
            "STANDARD_ASR_CODEC__SUB": '{"beam": 8}',
            "STANDARD_ASR_CODEC__LISTED": '["a", "b"]',
            "STANDARD_ASR_CODEC__NAME": "123",
            "STANDARD_ASR_CODEC__API_KEY": " sk-padded ",
        },
    )
    assert dict(built.mapping) == {"a": 1}
    assert list(built.sequence) == ["x"]
    assert built.td == {"beam": 8}
    assert built.dc.beam == 8
    assert built.sub.beam == 8
    assert built.listed == ["a", "b"]
    # The scalar was NOT reinterpreted as a JSON number...
    assert built.name == "123"
    # ...and a credential keeps its exact bytes, whitespace included.
    assert built.api_key is not None
    assert built.api_key.get_secret_value() == " sk-padded "


def test_ambiguous_env_shape_is_rejected_at_definition() -> None:
    # A field accepting BOTH a scalar and a structured shape has no defined
    # env reading: "123" is either that string or that JSON number, and
    # decoding it would disagree with the explicit constructor, which always
    # takes the string. The old helper picked JSON silently, so the two
    # construction paths diverged on the same input.
    with pytest.raises(TypeError, match="accepts BOTH a scalar and a structured shape"):

        class _MixedCfg(BaseConfig[Literal["mixed"]]):  # pyright: ignore[reportUnusedClass]
            engine: Literal["mixed"] = "mixed"
            value: str | list[str] = ""

    # The explicit constructor's reading is the one being protected: it takes
    # the string, and that must stay true of every accepted shape.
    class _ScalarCfg(BaseConfig[Literal["scalar"]]):
        engine: Literal["scalar"] = "scalar"
        value: str = ""

    assert _ScalarCfg(value="123").value == "123"
    assert _ScalarCfg.from_env("scalar", environ={"STANDARD_ASR_SCALAR__VALUE": "123"}).value == (
        "123"
    )


def test_env_codec_classification_is_fail_closed_and_scalar_biased() -> None:
    # The two wrong answers are not symmetric: guessing "raw" for something
    # structured fails LOUDLY at construction (pydantic rejects the string),
    # while guessing "json" for a scalar silently REINTERPRETS it. So "json"
    # is returned only on positive evidence and anything unrecognized stays
    # "raw" -- including a shape this pydantic version does not describe.
    from standard_asr.runtime.config import (
        _env_codec,  # pyright: ignore[reportPrivateUsage]
        _env_codecs,  # pyright: ignore[reportPrivateUsage]
    )

    assert _env_codec({"type": "future-kind"}) == "raw"
    assert _env_codec(None) == "raw"
    assert _env_codec({"type": "list", "items_schema": {"type": "str"}}) == "json"
    # A structured node's MEMBERS describe the document's contents, not how
    # the document arrives: a dict of scalars is still one JSON document.
    assert _env_codec({"type": "dict", "values_schema": {"type": "str"}}) == "json"
    # Nested wrappers and lists of choices are walked.
    assert (
        _env_codec({"type": "nullable", "schema": {"type": "default", "schema": {"type": "model"}}})
        == "json"
    )
    assert (
        _env_codec({"type": "union", "choices": [{"type": "str"}, {"type": "list"}]}) == "ambiguous"
    )

    # An unreadable schema is a definition-time error, not a silent default.
    class _Unreadable:
        __name__ = "_Unreadable"
        model_fields: dict[str, object] = {}

        @property
        def __pydantic_core_schema__(self) -> object:
            """Fail introspection the way a mock/unbuilt schema would.

            Returns:
                Never returns.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("no schema")

    with pytest.raises(TypeError, match="core schema could not be introspected"):
        _env_codecs(cast("Any", _Unreadable()))


def test_json_annotated_fields_read_the_env_document_as_text() -> None:
    # ``Json[T]`` is the one structured-looking shape whose env reading is
    # RAW: the annotation's contract is that the input IS the JSON document
    # text, decoded by pydantic's own ``json`` validator -- the explicit
    # constructor takes the string and REJECTS the decoded value
    # (``json_type``). Classifying off the inner document schema made the
    # codec pre-decode, so the field passed the class-definition guards and
    # the explicit constructor yet could not be fed through its own
    # documented env convention.
    from pydantic import Json

    class _JsonCfg(BaseConfig[Literal["jsonfield"]]):
        engine: Literal["jsonfield"] = "jsonfield"
        hints: Json[list[str]] | None = None
        weights: Json[dict[str, int]] | None = None

    assert _JsonCfg._ENV_CODECS["hints"] == "raw"  # pyright: ignore[reportPrivateUsage]
    assert _JsonCfg._ENV_CODECS["weights"] == "raw"  # pyright: ignore[reportPrivateUsage]

    built = _JsonCfg.from_env(
        "jsonfield",
        environ={
            "STANDARD_ASR_JSONFIELD__HINTS": '["a", "b"]',
            "STANDARD_ASR_JSONFIELD__WEIGHTS": '{"a": 1}',
        },
    )
    assert built.hints == ["a", "b"]
    assert built.weights == {"a": 1}
    # The explicit constructor is the reference semantics: same string in,
    # same value out -- and the decoded list is REJECTED there (proof that
    # pre-decoding was the wrong reading, not merely a different one).
    assert _JsonCfg(hints='["a", "b"]').hints == ["a", "b"]  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError, match="JSON input should be"):
        _JsonCfg(hints=["a", "b"])  # pyright: ignore[reportArgumentType]


def test_env_codec_treats_the_json_kind_as_terminal_raw() -> None:
    # The walker must STOP at a ``json`` node: its inner schema describes
    # the DECODED document's contents, not how the env string arrives.
    from standard_asr.runtime.config import (
        _env_codec,  # pyright: ignore[reportPrivateUsage]
    )

    assert (
        _env_codec({"type": "json", "schema": {"type": "list", "items_schema": {"type": "str"}}})
        == "raw"
    )
    # ...under wrappers like any other terminal kind.
    assert (
        _env_codec(
            {
                "type": "default",
                "schema": {
                    "type": "nullable",
                    "schema": {"type": "json", "schema": {"type": "dict"}},
                },
            }
        )
        == "raw"
    )
    # A union with a genuinely structured alternative is still ambiguous
    # (no defined reading), not silently raw.
    assert (
        _env_codec(
            {
                "type": "union",
                "choices": [{"type": "json", "schema": {"type": "list"}}, {"type": "list"}],
            }
        )
        == "ambiguous"
    )
