# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests covering plugin discovery and compliance helpers."""

from __future__ import annotations

import asyncio
import inspect
import runpy
from importlib.metadata import EntryPoint, EntryPoints
from typing import Any, AsyncIterator, ClassVar, Literal

import pytest
from pydantic import ConfigDict

import standard_asr.compliance as compliance
from standard_asr import TranscriptionResult
from standard_asr.audio.input import InputKind
from standard_asr.compliance import check_entrypoints
from standard_asr.contract.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    FlagCap,
    LanguageCaps,
)
from standard_asr.contract.exceptions import (
    ConfigError,
    EngineContractError,
    EntrypointValidationError,
    FactoryLoadError,
    ProtocolCompatibilityError,
)
from standard_asr.contract.identifiers import validate_engine_id, validate_model_name
from standard_asr.contract.params import ProviderParams
from standard_asr.engine import (
    NO_ARTIFACT_LIFECYCLE,
    ArtifactReport,
    BaseConfig,
    BaseProperties,
    DeclaredEngineMetadata,
    SampleRateRange,
)
from standard_asr.plugins import discovery as discovery_module
from standard_asr.plugins.discovery import (
    ENTRYPOINT_GROUP,
    ModelRegistry,
    ModelSpec,
    _gather_entry_points,  # pyright: ignore[reportPrivateUsage]
    discover_models,
    parse_entrypoint_name,
    pep503_normalize,
)
from standard_asr.runtime.interface import StandardASR
from standard_asr.runtime.streaming import TranscriptionEvent, TranscriptionSession


class _DummyConfig(BaseConfig[Literal["dummy"]]):
    engine: Literal["dummy"] = "dummy"
    # _DummyProperties exposes a language axis, so a compliant config provides
    # default_language -- keeps these fixtures clean under the
    # check_entrypoints language-axis check.
    default_language: str | None = "en"


class _DummyProperties(BaseProperties):
    engine_id: str = "dummy"
    model_name: str = "demo"
    protocol_version: str = "0.2.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    selectable_languages: list[str] = ["en"]


_DUMMY_CAPS = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
    )
)


class _DummyASR:
    properties: ClassVar[_DummyProperties] = _DummyProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _DUMMY_CAPS
    # The supported protocol line carries the artifact lifecycle, so even this
    # minimal structural fixture authors the metadata and answers both
    # lifecycle operations (the CLI transcribe preflight calls
    # artifact_status on any engine that declares the line).
    declared_metadata: ClassVar[DeclaredEngineMetadata] = DeclaredEngineMetadata(
        artifacts=NO_ARTIFACT_LIFECYCLE
    )

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.config = _DummyConfig(engine="dummy")

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="dummy")

    def artifact_status(self, context: Any = None) -> ArtifactReport:
        return ArtifactReport.from_requirements(mode="batch", applicable=False)

    def acquire_artifacts(
        self,
        context: Any = None,
        *,
        refresh: bool = False,
        progress: Any = None,
    ) -> ArtifactReport:
        return self.artifact_status(context)


def _dummy_factory(**kwargs: Any) -> _DummyASR:  # pyright: ignore[reportUnusedFunction]
    return _DummyASR(**kwargs)


class _CovariantDummyASR(_DummyASR):
    """Subclass an honest covariant factory returns instead of its annotation."""


def _covariant_dummy_factory(**kwargs: Any) -> _DummyASR:  # pyright: ignore[reportUnusedFunction]
    return _CovariantDummyASR(**kwargs)


class _OutsideLineDummyProperties(_DummyProperties):
    protocol_version: str = "0.1.0"


class _OutsideLineDummyASR(_DummyASR):
    properties: ClassVar[_DummyProperties] = _OutsideLineDummyProperties()


def _outside_line_factory(**kwargs: Any) -> _OutsideLineDummyASR:  # pyright: ignore[reportUnusedFunction]
    return _OutsideLineDummyASR(**kwargs)


#: Construction ledger for the preflight test: an outside-line class must be
#: refused BEFORE its factory runs -- otherwise a construction-time fault (a
#: missing credential, say) masks the line mismatch behind a configuration
#: diagnosis.
_outside_line_constructions: list[str] = []


def _counting_outside_line_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _OutsideLineDummyASR
):
    _outside_line_constructions.append("dummy/old")
    return _OutsideLineDummyASR()


class _DuckPropertiesASR:
    """Engine-shaped object whose ``properties`` is not a ``BaseProperties``.

    The create-time gate fails closed on it: an engine whose protocol line
    cannot be established is less knowable than one on a wrong line, so it
    must not be handed back ready to transcribe (round-25 review).
    """

    properties: ClassVar[dict[str, str]] = {"protocol_version": "9.9.9"}

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="duck")


def _duck_properties_factory() -> _DuckPropertiesASR:  # pyright: ignore[reportUnusedFunction]
    return _DuckPropertiesASR()


class _InstanceOnlyPropertiesASR:
    """Engine whose only typed declaration is built on the instance.

    The class-level slot holds an untyped dict, so every class-read surface
    (show, compliance, the per-model endpoints) fails closed on this engine;
    creation must join that verdict instead of handing back a running engine
    whose class declaration the rest of the toolchain rejects.
    """

    properties: dict[str, str] | BaseProperties = {"protocol_version": "0.2.0"}

    def __init__(self) -> None:
        self.properties = _DummyProperties()

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="instance-only")


#: Construction ledger for the class-declaration preflight test: an engine
#: whose resolvable class declaration is untyped must be refused BEFORE its
#: factory runs -- the same masking rationale as the line preflight.
_instance_only_constructions: list[str] = []


def _instance_only_properties_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _InstanceOnlyPropertiesASR
):
    _instance_only_constructions.append("dummy/instance-only")
    return _InstanceOnlyPropertiesASR()


def _opaque_instance_only_factory(**kwargs: Any) -> Any:  # pyright: ignore[reportUnusedFunction]
    # Opaque return annotation: the class is unresolvable without calling the
    # factory, so the class-declaration verdict cannot preflight and must land
    # on type(engine) after construction (the fallthrough seam).
    return _InstanceOnlyPropertiesASR()


def _opaque_duck_factory(**kwargs: Any) -> Any:  # pyright: ignore[reportUnusedFunction]
    # Nothing is resolvable statically AND the instance declaration is
    # untyped: the last net is the instance gate inside create().
    return _DuckPropertiesASR()


class _NotAnEngine:
    """A class an entry point might resolve to that is NOT a Standard ASR engine.

    It lacks the required class surface (``properties`` /
    ``declared_capabilities``), so ``engine_class`` must reject it with a clear
    ``FactoryLoadError`` instead of casting it through.
    """


def _not_an_engine_factory() -> _NotAnEngine:  # pyright: ignore[reportUnusedFunction]
    return _NotAnEngine()


class _LookAlikeConfig:
    """A non-engine class that happens to expose generic engine-ish names.

    A misconfigured entry point pointed at an engine's Config object would
    resolve here. It exposes ``properties`` / ``supports`` but NOT the defining
    ``transcribe`` method, so it must be rejected.
    """

    properties: ClassVar[dict[str, str]] = {}

    def supports(self, dot_path: str) -> bool:
        return False


def _look_alike_config_factory() -> _LookAlikeConfig:  # pyright: ignore[reportUnusedFunction]
    return _LookAlikeConfig()


def _protocol_annotated_factory() -> StandardASR:  # pyright: ignore[reportUnusedFunction]
    # The most common authoring mistake -- annotating the factory
    # with the StandardASR *protocol* instead of the concrete engine class.
    # ``StandardASR`` is runtime_checkable, so its ``transcribe`` would pass the
    # duck-type, but it carries no class-level metadata; engine_class must reject
    # the protocol loudly instead of returning it and silently losing metadata.
    raise NotImplementedError


def _unannotated_factory():  # type: ignore[no-untyped-def]  # pyright: ignore[reportUnusedFunction]
    return _DummyASR()


def _bad_annotation_factory():  # type: ignore[no-untyped-def]  # pyright: ignore[reportUnusedFunction]
    return _DummyASR()


# A return annotation naming a type that does not exist: resolving it raises
# NameError, so engine_class must surface a FactoryLoadError rather than crash.
_bad_annotation_factory.__annotations__ = {"return": "ThisTypeDoesNotExistAnywhere"}


def _dotted_annotation_factory():  # type: ignore[no-untyped-def]  # pyright: ignore[reportUnusedFunction]
    return _DummyASR()


# A dotted-name return annotation: the resolver walks the factory module's
# globals by attribute (``asyncio`` -> ``Future``) with no eval. ``Future`` is a
# real class but not an engine, so the class-surface guard rejects it -- proving
# the dotted path resolves.
_dotted_annotation_factory.__annotations__ = {"return": "asyncio.Future"}


def _subscripted_annotation_factory():  # type: ignore[no-untyped-def]  # pyright: ignore[reportUnusedFunction]
    return _DummyASR()


# A subscripted/generic return annotation is not a concrete engine class. The
# resolver refuses to eval arbitrary expressions, so a return type that is not a
# plain/dotted name is reported as an authoring error.
_subscripted_annotation_factory.__annotations__ = {"return": "list[_DummyASR]"}


def _bad_param_annotation_factory(  # pyright: ignore[reportUnusedFunction]
    required: ThisParamTypeDoesNotExist,  # type: ignore[name-defined]  # noqa: F821
) -> _DummyASR:  # pragma: no cover - instantiation skipped
    # The parameter annotation is an unresolvable forward reference, but the
    # RETURN annotation is concrete. engine_class must read the return type
    # without choking on the unrelated parameter.
    return _DummyASR()


class _OpenParams(ProviderParams):
    model_config = ConfigDict(extra="allow")  # violates the closed-params rule


class _OpenParamsASR(_DummyASR):
    provider_params_type: ClassVar[type[ProviderParams] | None] = _OpenParams


class _ConfigTypeASR(_DummyASR):  # pyright: ignore[reportUnusedClass]
    """Engine declaring config_type; construction must never be needed to read it."""

    config_type: ClassVar[type[BaseConfig[str]] | None] = _DummyConfig

    def __init__(self, **kwargs: Any) -> None:
        raise AssertionError("config_schema must not instantiate the engine")


class _GarbageConfigTypeASR(_DummyASR):  # pyright: ignore[reportUnusedClass]
    """config_type set to a non-BaseConfig object (a broken engine)."""

    config_type: ClassVar[Any] = _NotAnEngine


class _OpaqueHandle:
    """Stand-in for a client/model handle: not JSON-Schema representable."""


class _UnschematizableConfig(BaseConfig[Literal["dummy"]]):
    """Legitimate BaseConfig subclass whose JSON Schema cannot be generated."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine: Literal["dummy"] = "dummy"
    default_language: str | None = "en"
    handle: _OpaqueHandle | None = None


class _UnschematizableConfigTypeASR(_DummyASR):  # pyright: ignore[reportUnusedClass]
    """config_type is a valid BaseConfig, but schema generation fails."""

    config_type: ClassVar[type[BaseConfig[str]] | None] = _UnschematizableConfig


def _open_params_factory() -> _OpenParamsASR:  # pyright: ignore[reportUnusedFunction]
    return _OpenParamsASR()


class _FakeDist:
    def __init__(self, name: str) -> None:
        self.name = name


def _ep_with_dist(name: str, dist_name: str) -> EntryPoint:
    ep = EntryPoint(
        name=name,
        value="tests.test_discovery:_dummy_factory",
        group="standard_asr.models",
    )
    object.__setattr__(ep, "dist", _FakeDist(dist_name))
    return ep


class _BridgeSession(TranscriptionSession):
    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        async for _ in self.audio_chunks():
            pass
        yield TranscriptionEvent.final("s0", "done", start=0.0, end=1.0)


class _HangBridgeSession(TranscriptionSession):
    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        # Never terminates and never yields: simulates a deadlocking engine.
        await asyncio.Event().wait()
        yield TranscriptionEvent.done()  # pragma: no cover


def _requires_argument_factory(
    required: str,
) -> _DummyASR:  # pragma: no cover - instantiation skipped
    return _DummyASR(required=required)


def _error_factory() -> _DummyASR:  # pyright: ignore[reportUnusedFunction]
    raise RuntimeError("boom")


class _MissingMetaASR:
    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="missing")


def _missing_meta_factory() -> _MissingMetaASR:  # pyright: ignore[reportUnusedFunction]
    return _MissingMetaASR()


class _GatedButBareASR:
    """Typed supported-line properties; everything else missing.

    Passes the instance protocol gate, so the per-attribute instance checks
    (config, capabilities, the required surface) actually run and report
    their own arms instead of being short-circuited behind the gate.
    """

    properties: ClassVar[_DummyProperties] = _DummyProperties()

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="bare")


def _gated_but_bare_factory() -> _GatedButBareASR:  # pyright: ignore[reportUnusedFunction]
    return _GatedButBareASR()


# pyright: ignore[reportUnusedFunction]
def _non_callable_factory() -> str:  # pragma: no cover
    return "not-an-asr"


def test_pep503_normalize_and_parse_roundtrip() -> None:
    assert pep503_normalize("Foo.Bar_baz") == "foo-bar-baz"
    engine, model = parse_entrypoint_name("engine-only")
    assert engine == "engine-only"
    assert model == ""
    engine2, model2 = parse_entrypoint_name("faster-whisper/whisper")
    assert engine2 == "faster-whisper"
    assert model2 == "whisper"


def test_parse_entrypoint_name_rejects_bad_engine() -> None:
    with pytest.raises(EntrypointValidationError):
        parse_entrypoint_name("BadCaps/model")


def test_declared_engine_id_docstring_example_is_reachable() -> None:
    # (code is the contract): the ModelSpec.declared_engine_id
    # docstring example must be a value the field can actually hold. The lower-
    # case ``faster_whisper`` is accepted and folded to ``faster-whisper`` (so it
    # is reachable), while the previously documented upper-case ``Faster_Whisper``
    # is rejected at validation and can never appear -- the asymmetry the
    # docstring and plugin-entry-points.md now spell out.
    eps = [
        EntryPoint(
            name="faster_whisper/large",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    spec = registry.spec("faster-whisper/large")
    assert spec.declared_engine_id == "faster_whisper"  # the docstring example
    assert spec.engine_id == "faster-whisper"  # its canonical routing identity

    # The old, unreachable upper-case example must stay rejected (declared side).
    with pytest.raises(EntrypointValidationError):
        parse_entrypoint_name("Faster_Whisper/large")


def test_lookup_accepts_slashless_as_default_alias() -> None:
    # The LOOKUP parser stays lenient -- a bare engine id is a
    # convenience alias for the engine's default model, so spec()/keys_by_engine()
    # callers need not type the trailing slash.
    engine, model = parse_entrypoint_name("faster-whisper")
    assert engine == "faster-whisper"
    assert model == ""


def test_declaration_rejects_slashless_entrypoint_when_strict() -> None:
    # A plugin KEY must use the explicit <engine_id>/<model_name>
    # form. A slash-less key (an unspecified third form, usually a dropped
    # /<model_name>) is rejected in strict discovery rather than silently
    # accepted as the engine default and slipping past compliance.
    eps = [
        EntryPoint(
            name="faster-whisper",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    with pytest.raises(EntrypointValidationError) as excinfo:
        discover_models(eps=eps, strict=True)
    assert "has no '/'" in str(excinfo.value)


def test_declaration_warns_and_skips_slashless_entrypoint_by_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # In default (non-strict) discovery the slash-less key is not
    # silently registered as ``engine_id/``; it warns (telling the author the
    # exact fix) and is skipped, exactly like any other malformed key.
    eps = [
        EntryPoint(
            name="faster-whisper",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    caplog.set_level("WARNING")
    registry = discover_models(eps=eps, strict=False)
    assert registry.names() == []
    assert any("has no '/'" in r.message for r in caplog.records)


def test_discover_models_supports_multiple_entries() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        ),
        EntryPoint(
            name="alpha/second",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        ),
        EntryPoint(
            name="beta/",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        ),
    ]
    registry = discover_models(eps=eps, strict=True)
    assert registry.names() == ["alpha/first", "alpha/second", "beta/"]
    assert registry.keys_by_engine("alpha") == ["alpha/first", "alpha/second"]
    spec = registry.spec("beta/")
    assert spec.model_name == ""


def test_discover_models_duplicate_strategy_replace() -> None:
    # ``replace`` is the same provider overriding its own registration, so both
    # entry points carry the SAME distribution identity -- otherwise this would
    # (correctly) be a cross-distribution engine-identity collision. Distinct targets
    # let the test assert that the latter factory wins.
    ep_a = EntryPoint(
        name="alpha/only",
        value="tests.test_discovery:_dummy_factory",
        group="standard_asr.models",
    )
    ep_b = EntryPoint(
        name="alpha/only",
        value="tests.test_discovery:_requires_argument_factory",
        group="standard_asr.models",
    )
    object.__setattr__(ep_a, "dist", _FakeDist("one-dist"))
    object.__setattr__(ep_b, "dist", _FakeDist("one-dist"))
    registry = discover_models(eps=[ep_a, ep_b], strict=True, on_conflict="replace")
    spec = registry.spec("alpha/only")
    factory = spec.load_factory()
    assert factory is _requires_argument_factory


def test_discover_models_invalid_name_raises_when_strict() -> None:
    eps = [
        EntryPoint(
            name="bad/name/with/slashes",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    with pytest.raises(EntrypointValidationError) as excinfo:
        discover_models(eps=eps, strict=True)
    assert "bad/name/with/slashes" in str(excinfo.value)


def test_compliance_reports_expected_issues() -> None:
    eps = [
        EntryPoint(
            name="dummy/demo",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        ),
        EntryPoint(
            name="needs-arg/model",
            value="tests.test_discovery:_requires_argument_factory",
            group="standard_asr.models",
        ),
        EntryPoint(
            name="bad/model",
            value="tests.test_discovery:_non_callable_factory",
            group="standard_asr.models",
        ),
    ]
    registry = discover_models(eps=eps, strict=True)
    report = check_entrypoints(registry=registry, instantiate=True)

    warnings = list(report.iter_level("warning"))
    assert any(issue.model == "needs-arg/model" for issue in warnings)

    errors = list(report.iter_level("error"))
    assert any(issue.model == "bad/model" for issue in errors)
    assert report.passed is False


def test_model_registry_create_forwards_arguments() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        ),
    ]
    registry = discover_models(eps=eps, strict=True)
    instance = registry.create("alpha/first", foo="bar")
    assert isinstance(instance, _DummyASR)
    assert instance.kwargs["foo"] == "bar"


class _NamedConfigEngine(_DummyASR):
    """Engine whose declared config_type feeds create()'s error rendering."""

    config_type: ClassVar[type[BaseConfig[str]] | None] = _DummyConfig

    def __init__(self, **kwargs: Any) -> None:
        # A bare-constructor config failure: raw pydantic ValidationError.
        _DummyConfig(engine="dummy", default_language=object())  # type: ignore[arg-type]
        super().__init__(**kwargs)  # pragma: no cover - never reached


def _named_config_engine_factory(**kwargs: Any) -> _NamedConfigEngine:  # pyright: ignore[reportUnusedFunction]
    return _NamedConfigEngine(**kwargs)


def _opaque_validation_factory(**kwargs: Any) -> Any:  # pyright: ignore[reportUnusedFunction]
    # A FACTORY with an opaque return annotation (engine class unreachable
    # without calling it) whose ValidationError create() must still wrap --
    # with no name source at all.
    _DummyConfig(engine="dummy", default_language=object())  # type: ignore[arg-type]
    return _DummyASR(**kwargs)  # pragma: no cover - never reached


def _no_config_type_factory(**kwargs: Any) -> _DummyASR:  # pyright: ignore[reportUnusedFunction]
    # Engine class resolvable but declares NO config_type: the name-source
    # resolution finds nothing and the wrap stays fully masked.
    _DummyConfig(engine="dummy", default_language=object())  # type: ignore[arg-type]
    return _DummyASR(**kwargs)  # pragma: no cover - never reached


def test_create_config_error_names_declared_fields_via_config_type() -> None:
    """create()'s ConfigError names the failing field, never the input.

    The wrap is uniform for every factory shape -- an annotated engine
    class, an opaque factory function, a class with no config_type: the
    accident-model scrubber names field-name-shaped loc components (the
    typo-names-the-key DX) and drops the input echo, with no engine-class
    resolution needed.
    """
    eps = [
        EntryPoint(
            name="named/model",
            value="tests.test_discovery:_named_config_engine_factory",
            group="standard_asr.models",
        ),
        EntryPoint(
            name="opaque/model",
            value="tests.test_discovery:_opaque_validation_factory",
            group="standard_asr.models",
        ),
        EntryPoint(
            name="bare/model",
            value="tests.test_discovery:_no_config_type_factory",
            group="standard_asr.models",
        ),
    ]
    registry = discover_models(eps=eps, strict=True)
    for key in ("named/model", "opaque/model", "bare/model"):
        with pytest.raises(ConfigError) as excinfo:
            registry.create(key)
        assert "default_language" in str(excinfo.value), key
        assert "object object" not in str(excinfo.value), key


def test_compliance_reports_error_when_registry_empty() -> None:
    registry = discover_models(eps=[], strict=True)
    report = check_entrypoints(registry=registry)
    assert report.passed is False
    errors = list(report.iter_level("error"))
    assert errors[0].model is None


def test_non_callable_factory_returns_string() -> None:
    assert _non_callable_factory() == "not-an-asr"


def test_validate_engine_id_rejects_slash() -> None:
    with pytest.raises(EntrypointValidationError):
        validate_engine_id("bad/name")


def test_validate_model_name_rejects_slash() -> None:
    with pytest.raises(EntrypointValidationError):
        validate_model_name("bad/name")


def test_validate_model_name_rejects_invalid_chars() -> None:
    with pytest.raises(EntrypointValidationError):
        parse_entrypoint_name("engine/bad*name")


def test_validate_model_name_position_defect_names_the_rule_and_value() -> None:
    """A leading '.', ':', or '-' fails on POSITION, not on the character set.

    The rejection message once listed the offending character as *allowed*
    ("Allowed characters: letters, digits, '.', ...") and never echoed the
    value -- pointing a plugin author away from the only thing wrong with
    ``.v1``. The message must state the leading-character rule from
    ``docs/content/engine-authors/plugin-entry-points.md`` and echo the rejected value.
    """
    for bad in (".v1", "-int8", ":cpu"):
        with pytest.raises(EntrypointValidationError, match="must start with") as exc_info:
            validate_model_name(bad)
        assert repr(bad) in str(exc_info.value)


def test_validate_engine_id_position_defect_names_the_rule_and_value() -> None:
    with pytest.raises(EntrypointValidationError, match="must start with") as exc_info:
        validate_engine_id(".dotted")
    assert repr(".dotted") in str(exc_info.value)


def test_validate_engine_id_rejects_empty_with_a_plain_message() -> None:
    # An empty id used to land on "contains unsupported characters" -- there
    # is no character in an empty string; the defect is emptiness itself.
    with pytest.raises(EntrypointValidationError, match="must not be empty"):
        validate_engine_id("")


def test_validate_engine_id_accepts_non_canonical() -> None:
    # A non-canonical-but-valid id passes surface validation; canonicalization
    # to the routing identity happens in parse_entrypoint_name / discover_models.
    validate_engine_id("my_engine")
    validate_model_name("model")


def test_parse_entrypoint_name_canonicalizes_engine_id() -> None:
    # The routing identity is the PEP 503 canonical form, not the verbatim
    # declared segment (runs of [-_.] collapse to a single '-').
    engine_id, model_name = parse_entrypoint_name("my_engine/large.v3")
    assert engine_id == "my-engine"
    assert model_name == "large.v3"


def test_discover_canonicalizes_engine_id_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    eps = [
        EntryPoint(
            name="my_engine/first",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    caplog.set_level("INFO")
    registry = discover_models(eps=eps, strict=True)

    # The routing key and engine_id are canonical; the declared form is retained.
    assert registry.names() == ["my-engine/first"]
    spec = registry.spec("my-engine/first")
    assert spec.engine_id == "my-engine"
    assert spec.declared_engine_id == "my_engine"
    assert registry.keys_by_engine("my-engine") == ["my-engine/first"]
    assert any("not PEP 503 normalized" in r.message for r in caplog.records)


def test_by_engine_normalizes_non_canonical_argument() -> None:
    # Engine-identity consistency: keys_by_engine() must PEP 503-normalize its argument the same
    # way spec()/create() do, so a non-canonical query form (for example, "my_engine")
    # resolves to the same engine -- not an empty list while spec()/create()
    # still resolve it.
    eps = [
        EntryPoint(
            name="my-engine/first",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)

    expected = ["my-engine/first"]
    assert registry.keys_by_engine("my-engine") == expected  # canonical form.
    # Lowercase non-canonical forms ([-_.] separators) are exactly what
    # spec()/create() accept and fold to the canonical id; keys_by_engine() now agrees
    # instead of returning [].
    for non_canonical in ("my_engine", "my.engine", "my--engine"):
        assert registry.keys_by_engine(non_canonical) == expected, non_canonical
        assert registry.spec(f"{non_canonical}/first").engine_id == "my-engine"
        assert registry.create(f"{non_canonical}/first") is not None
    # pep503_normalize also lowercases, so keys_by_engine tolerates a mixed-case form
    # too (a lookup key, unlike an entry-point name, need not be lowercase).
    assert registry.keys_by_engine("My-Engine") == expected


def test_model_spec_load_factory_error_on_load() -> None:
    class _BadEntryPoint:
        def load(self) -> object:
            raise RuntimeError("boom")

    spec = ModelSpec(
        model_id="alpha/first",
        engine_id="alpha",
        model_name="first",
        entry_point=_BadEntryPoint(),  # type: ignore[arg-type]
    )

    with pytest.raises(FactoryLoadError):
        spec.load_factory()


def test_model_spec_load_factory_rejects_non_callable() -> None:
    class _BadEntryPoint:
        def load(self) -> object:
            return "not-callable"

    spec = ModelSpec(
        model_id="alpha/first",
        engine_id="alpha",
        model_name="first",
        entry_point=_BadEntryPoint(),  # type: ignore[arg-type]
    )

    with pytest.raises(FactoryLoadError):
        spec.load_factory()


def test_model_registry_missing_spec_raises() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)

    with pytest.raises(EntrypointValidationError):
        registry.spec("alpha/missing")


def test_gather_entry_points_override() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    gathered = _gather_entry_points(eps)

    assert len(gathered) == 1


def test_gather_entry_points_default(monkeypatch: pytest.MonkeyPatch) -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_dummy_factory",
            group=ENTRYPOINT_GROUP,
        )
    ]

    def _entry_points(group: str) -> EntryPoints:
        assert group == ENTRYPOINT_GROUP
        return EntryPoints(eps)

    monkeypatch.setattr("standard_asr.plugins.discovery.entry_points", _entry_points)

    gathered = _gather_entry_points()

    assert len(gathered) == 1


def test_discover_models_invalid_on_conflict() -> None:
    with pytest.raises(ValueError):
        discover_models(eps=[], strict=True, on_conflict="bad")


def test_discover_models_skips_wrong_group() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_dummy_factory",
            group="other.group",
        )
    ]
    registry = discover_models(eps=eps, strict=True)

    assert len(registry.names()) == 0


def test_discover_models_warn_keep_first() -> None:
    # Same provider registering the key twice (shared distribution identity), so
    # the duplicate is resolved by ``warn_keep_first`` rather than flagged as an
    # cross-distribution engine-identity collision. Distinct targets prove the first is kept.
    ep_a = EntryPoint(
        name="alpha/dup",
        value="tests.test_discovery:_dummy_factory",
        group="standard_asr.models",
    )
    ep_b = EntryPoint(
        name="alpha/dup",
        value="tests.test_discovery:_requires_argument_factory",
        group="standard_asr.models",
    )
    object.__setattr__(ep_a, "dist", _FakeDist("one-dist"))
    object.__setattr__(ep_b, "dist", _FakeDist("one-dist"))
    registry = discover_models(eps=[ep_a, ep_b], strict=True)

    spec = registry.spec("alpha/dup")
    assert spec.entry_point.value == "tests.test_discovery:_dummy_factory"


def test_discover_models_invalid_entrypoint_non_strict() -> None:
    eps = [
        EntryPoint(
            name="bad/name/with/slashes",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=False)

    assert len(registry.names()) == 0


def test_can_call_without_args_signature_error() -> None:
    assert compliance._can_call_without_args(object()) is False  # pyright: ignore[reportPrivateUsage]


def test_check_entrypoints_registry_none_calls_discover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ModelRegistry({})
    called: dict[str, bool] = {"called": False}

    def _discover_models(strict: bool = False) -> ModelRegistry:
        called["called"] = True
        return registry

    monkeypatch.setattr("standard_asr.compliance.discover_models", _discover_models)

    report = check_entrypoints(registry=None, strict_discovery=True)

    assert called["called"] is True
    assert report.registry is registry


def test_check_entrypoints_factory_load_error() -> None:
    class _Spec:
        def load_factory(self) -> object:
            raise FactoryLoadError("boom")

    registry = ModelRegistry({"alpha/first": _Spec()})  # type: ignore[arg-type]
    report = check_entrypoints(registry=registry)

    errors = list(report.iter_level("error"))
    assert any("boom" in issue.message for issue in errors)


def test_check_entrypoints_instantiate_false() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_requires_argument_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    report = check_entrypoints(registry=registry, instantiate=False)

    assert report.passed is True


def test_check_entrypoints_factory_invocation_error() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_error_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    report = check_entrypoints(registry=registry, instantiate=True)

    errors = list(report.iter_level("error"))
    assert any("Factory invocation failed" in issue.message for issue in errors)


def test_check_entrypoints_missing_metadata() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_missing_meta_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    report = check_entrypoints(registry=registry, instantiate=True)

    errors = list(report.iter_level("error"))
    # The class-level gate fails closed on the untyped declaration and stops
    # there, exactly as ModelRegistry.create() refuses the class before its
    # factory runs: with no protocol line the gate can establish, measuring
    # the engine against this core's contract would be noise behind the root
    # cause, and constructing it would run plugin code the runtime never runs.
    assert [issue.code for issue in errors] == ["missing_class_properties"]
    assert any("BaseProperties" in issue.message for issue in errors)
    assert not any("BaseConfig" in issue.message for issue in errors)


def test_check_entrypoints_reports_per_attribute_gaps_behind_the_gate() -> None:
    # A gate-passing instance still gets the per-attribute arms: missing
    # config and missing capabilities each report their own error rather
    # than hiding behind the protocol verdict.
    eps = [
        EntryPoint(
            name="dummy/demo",
            value="tests.test_discovery:_gated_but_bare_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    report = check_entrypoints(registry=registry, instantiate=True)

    errors = list(report.iter_level("error"))
    assert any("BaseConfig" in issue.message for issue in errors)
    assert any("DeclaredCapabilities" in issue.message for issue in errors)


def test_check_entrypoints_model_id_mismatch() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    report = check_entrypoints(registry=registry, instantiate=True)

    errors = list(report.iter_level("error"))
    assert any("model_id" in issue.message for issue in errors)


# ----- no-instantiation engine class resolution --------------------------- #


def test_engine_class_resolves_from_factory_return_annotation() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    cls = registry.engine_class("alpha/first")
    assert cls is _DummyASR
    # Reading ClassVars must not require instantiation.
    assert getattr(cls, "declared_capabilities") is _DUMMY_CAPS


def test_engine_class_resolves_when_entrypoint_is_a_class() -> None:
    """A directly exposed engine class needs no factory return annotation.

    Both engine-author guides once carried a blanket "the entry-point factory
    MUST be annotated with your concrete engine class", which would tell a
    plugin author that this -- registering the class with no factory at all --
    is noncompliant. It is not: ``engine_class`` returns a class target
    directly, and compliance only reports ``class_metadata_unreadable`` when
    NEITHER form resolves. Keep this test as the contract the guides
    (``adapt-an-asr-system.md``, ``plugin-entry-points.md``) must match.
    """
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_DummyASR",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    assert registry.engine_class("alpha/first") is _DummyASR


def test_engine_class_rejects_entrypoint_class_without_engine_surface() -> None:
    # An entry point resolving to a class that does not expose the StandardASR
    # class surface must fail loudly (FactoryLoadError), not be cast through to a
    # later AttributeError when its metadata is read.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_NotAnEngine",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(FactoryLoadError, match="does not expose"):
        registry.engine_class("alpha/first")


def test_engine_class_rejects_factory_returning_non_engine() -> None:
    # Same guard via the factory-return-annotation path.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_not_an_engine_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(FactoryLoadError, match="does not expose"):
        registry.engine_class("alpha/first")


def test_engine_class_rejects_factory_annotated_with_protocol() -> None:
    # Annotating the factory '-> StandardASR' (the protocol, not a
    # concrete engine) is the most common authoring mistake. The protocol is
    # runtime_checkable so its 'transcribe' would pass the duck-type, but it
    # carries no class-level metadata -- engine_class must reject it loudly
    # instead of returning the protocol and silently reporting MISSING metadata.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_protocol_annotated_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(FactoryLoadError, match="Protocol"):
        registry.engine_class("alpha/first")


def test_engine_class_rejects_protocol_entrypoint_directly() -> None:
    # The same rejection must hold when the entry point resolves straight to the
    # protocol class (not via a factory return annotation).
    eps = [
        EntryPoint(
            name="alpha/first",
            value="standard_asr.runtime.interface:StandardASR",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(FactoryLoadError, match="Protocol"):
        registry.engine_class("alpha/first")


def test_config_schema_protocol_annotation_does_not_silently_return_none() -> None:
    # consequence: before the fix, config_schema would read the
    # protocol's (absent) config_type and return None -- semantically "this
    # engine has no config" -- masking the authoring error. It must raise.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_protocol_annotated_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(FactoryLoadError, match="Protocol"):
        registry.config_schema("alpha/first")


def test_config_schema_reads_class_without_instantiation() -> None:
    # The whole point of config_schema: render a settings UI for an engine
    # BEFORE construction (construction may require the very credentials the
    # form collects). _ConfigTypeASR.__init__ raises, proving no instantiation.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_ConfigTypeASR",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    schema = registry.config_schema("alpha/first")
    assert schema is not None
    assert "engine" in schema["properties"]
    assert "strict" in schema["properties"]


def test_config_schema_returns_none_when_undeclared() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_DummyASR",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    assert registry.config_schema("alpha/first") is None


def test_config_schema_rejects_non_baseconfig_config_type() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_GarbageConfigTypeASR",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(FactoryLoadError, match="not a BaseConfig subclass"):
        registry.config_schema("alpha/first")


def test_config_schema_surfaces_unschematizable_config_type() -> None:
    # A legitimate BaseConfig subclass can still be un-schematizable
    # (arbitrary_types_allowed + an opaque handle field). pydantic's
    # PydanticInvalidForJsonSchema must surface as FactoryLoadError so both
    # schema consumers (`show` and ``GET /v1/config-schema/...``) degrade loudly
    # instead of crashing with a raw pydantic error.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_UnschematizableConfigTypeASR",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(FactoryLoadError, match="JSON Schema cannot be generated"):
        registry.config_schema("alpha/first")


def test_engine_class_raises_when_annotation_not_concrete() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_unannotated_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(FactoryLoadError):
        registry.engine_class("alpha/first")


def test_engine_class_rejects_look_alike_with_only_generic_markers() -> None:
    # A class exposing only generic names (properties/supports) but not the
    # defining 'transcribe' method must be rejected -- the previous ``any(...)``
    # gate accepted it.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_look_alike_config_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(FactoryLoadError, match="transcribe"):
        registry.engine_class("alpha/first")


def test_engine_class_accepts_engine_with_only_transcribe() -> None:
    # A real engine exposing 'transcribe' passes even if other ClassVars are
    # absent (completeness is the compliance suite's job).
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_missing_meta_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    assert registry.engine_class("alpha/first") is _MissingMetaASR


def test_engine_class_raises_when_return_annotation_unresolvable() -> None:
    # A factory whose *return* annotation references an undefined name cannot be
    # resolved; that must become a FactoryLoadError, not crash.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_bad_annotation_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(FactoryLoadError, match="return annotation"):
        registry.engine_class("alpha/first")


def test_engine_class_resolves_dotted_return_annotation_without_eval() -> None:
    # A dotted-name return annotation is resolved by an attribute walk over the
    # factory module's globals (no eval). It lands on a real-but-non-engine type
    # here, so the class-surface guard rejects it -- exercising the dotted path.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_dotted_annotation_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(FactoryLoadError, match="does not expose"):
        registry.engine_class("alpha/first")


def test_engine_class_rejects_non_name_return_annotation() -> None:
    # A subscripted/generic return annotation is not a concrete engine class and
    # is never eval'd; engine_class must fail loudly with FactoryLoadError rather
    # than execute the annotation string.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_subscripted_annotation_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(FactoryLoadError, match="not a plain name or dotted name"):
        registry.engine_class("alpha/first")


def test_engine_class_resolves_live_class_return_annotation() -> None:
    # A factory whose return annotation is already a live class object (no
    # ``from __future__ import annotations`` stringification) resolves directly,
    # without the eval path.
    def _live_annotation_factory() -> object:  # pragma: no cover - never invoked
        return _DummyASR()

    _live_annotation_factory.__annotations__ = {"return": _DummyASR}

    class _LoadsLiveFactory:
        def load(self) -> object:
            return _live_annotation_factory

    spec = ModelSpec(
        model_id="alpha/first",
        engine_id="alpha",
        model_name="first",
        entry_point=_LoadsLiveFactory(),  # type: ignore[arg-type]
    )

    assert spec.engine_class() is _DummyASR


def test_engine_class_raises_when_factory_has_no_signature() -> None:
    # A callable factory whose signature cannot be introspected (for example, an invalid
    # ``__signature__``) must surface a FactoryLoadError, not crash.
    class _NoSignatureFactory:
        __signature__ = "not a signature"  # makes inspect.signature raise

        def __call__(self) -> _DummyASR:  # pragma: no cover - never invoked
            return _DummyASR()

    class _LoadsNoSignature:
        def load(self) -> object:
            return _NoSignatureFactory()

    spec = ModelSpec(
        model_id="alpha/first",
        engine_id="alpha",
        model_name="first",
        entry_point=_LoadsNoSignature(),  # type: ignore[arg-type]
    )

    with pytest.raises(FactoryLoadError, match="inspectable signature"):
        spec.engine_class()


def test_engine_class_ignores_unresolvable_param_annotation() -> None:
    # An unrelated parameter carrying an unresolvable forward reference must
    # NOT block reading the engine class -- only the return annotation is
    # resolved, so static metadata stays readable without instantiation.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_bad_param_annotation_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    cls = registry.engine_class("alpha/first")
    assert cls is _DummyASR


# ----- engine-identity collision detection -------------------------------- #


def test_discover_detects_engine_id_collision_across_dists(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ep_a = _ep_with_dist("whisper/a", "dist-one")
    ep_b = _ep_with_dist("whisper/b", "dist-two")

    caplog.set_level("WARNING")
    registry = discover_models(eps=[ep_a, ep_b])
    assert registry.shadowed_engine_ids == {"whisper"}
    assert any("Engine-identity collision" in r.message for r in caplog.records)


def test_engine_id_collision_strict_raises() -> None:
    ep_a = _ep_with_dist("whisper/a", "dist-one")
    ep_b = _ep_with_dist("whisper/b", "dist-two")

    with pytest.raises(EntrypointValidationError):
        discover_models(eps=[ep_a, ep_b], strict=True)


def test_same_model_name_across_dists_is_shadowed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Regression guard: two DISTINCT distributions providing the SAME model key
    # (``whisper/large-v3``) are the most common engine-identity collision. The
    # ``on_conflict`` drop must not erase one provider before the collision is
    # counted, or the collision would silently survive (re-opening the mis-routing
    # the engine-identity guards protect against).
    ep_a = _ep_with_dist("whisper/large-v3", "dist-one")
    ep_b = _ep_with_dist("whisper/large-v3", "dist-two")

    caplog.set_level("WARNING")
    registry = discover_models(eps=[ep_a, ep_b])
    assert registry.shadowed_engine_ids == {"whisper"}
    assert any("Engine-identity collision" in r.message for r in caplog.records)


def test_same_model_name_across_dists_strict_raises() -> None:
    # The strict-mode counterpart of the regression above must still fail loud.
    ep_a = _ep_with_dist("whisper/large-v3", "dist-one")
    ep_b = _ep_with_dist("whisper/large-v3", "dist-two")

    with pytest.raises(EntrypointValidationError):
        discover_models(eps=[ep_a, ep_b], strict=True)


def test_single_dist_many_models_is_not_a_collision() -> None:
    # A single distribution legitimately exposing several models under one
    # engine_id must NOT be falsely flagged: set semantics dedupe its identity.
    ep_a = _ep_with_dist("whisper/large-v3", "one-dist")
    ep_b = _ep_with_dist("whisper/medium", "one-dist")
    ep_c = _ep_with_dist("whisper/small", "one-dist")

    registry = discover_models(eps=[ep_a, ep_b, ep_c], strict=True)
    assert registry.shadowed_engine_ids == set()
    assert set(registry.names()) == {
        "whisper/large-v3",
        "whisper/medium",
        "whisper/small",
    }


def test_normalized_engine_id_collision_across_dists_is_shadowed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Two distributions whose engine_ids only differ by PEP 503 normalization
    # (``my_engine`` vs ``my-engine``) route to the same canonical id and the
    # same env-var prefix, so they MUST be flagged as a collision.
    ep_a = _ep_with_dist("my_engine/a", "dist-one")
    ep_b = _ep_with_dist("my-engine/b", "dist-two")

    caplog.set_level("WARNING")
    registry = discover_models(eps=[ep_a, ep_b])
    assert registry.shadowed_engine_ids == {"my-engine"}
    assert any("Engine-identity collision" in r.message for r in caplog.records)


def test_normalized_engine_id_collision_strict_raises() -> None:
    ep_a = _ep_with_dist("my_engine/a", "dist-one")
    ep_b = _ep_with_dist("my-engine/b", "dist-two")

    with pytest.raises(EntrypointValidationError):
        discover_models(eps=[ep_a, ep_b], strict=True)


def test_dist_less_distinct_providers_collide(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Two entry points without distribution metadata but distinct module:attr
    # targets are genuinely different providers of the same engine id; they
    # must NOT collapse to a single "<unknown>" identity that hides the
    # collision.
    ep_a = EntryPoint(
        name="whisper/a",
        value="tests.test_discovery:_dummy_factory",
        group="standard_asr.models",
    )
    ep_b = EntryPoint(
        name="whisper/b",
        value="tests.test_discovery:_requires_argument_factory",
        group="standard_asr.models",
    )

    caplog.set_level("WARNING")
    registry = discover_models(eps=[ep_a, ep_b])
    assert registry.shadowed_engine_ids == {"whisper"}
    assert any("Engine-identity collision" in r.message for r in caplog.records)


def test_dist_less_same_provider_is_not_a_collision() -> None:
    # Two models from the SAME dist-less provider (identical module:attr target)
    # share an identity and must not be flagged.
    ep_a = EntryPoint(
        name="whisper/a",
        value="tests.test_discovery:_dummy_factory",
        group="standard_asr.models",
    )
    ep_b = EntryPoint(
        name="whisper/b",
        value="tests.test_discovery:_dummy_factory",
        group="standard_asr.models",
    )

    registry = discover_models(eps=[ep_a, ep_b], strict=True)
    assert registry.shadowed_engine_ids == set()


def test_same_dist_same_engine_id_is_not_a_collision() -> None:
    ep_a = _ep_with_dist("whisper/a", "one-dist")
    ep_b = _ep_with_dist("whisper/b", "one-dist")

    registry = discover_models(eps=[ep_a, ep_b], strict=True)
    assert registry.shadowed_engine_ids == set()


def test_create_shadowed_engine_id_warns_at_routing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The ambiguity is surfaced again at the point of use, not only at discovery.
    ep_a = _ep_with_dist("whisper/a", "dist-one")
    ep_b = _ep_with_dist("whisper/b", "dist-two")
    registry = discover_models(eps=[ep_a, ep_b])

    caplog.clear()
    caplog.set_level("WARNING")
    registry.create("whisper/a")
    assert any("routing is ambiguous" in r.message for r in caplog.records)


# ----- compliance class-level + sync-bridge checks ------------------------- #


def test_compliance_flags_unreadable_class_metadata() -> None:
    # Typed properties (so the class gate passes) but no class-level
    # declared_capabilities: the class pass flags it without instantiation.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_gated_but_bare_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    report = check_entrypoints(registry=registry, instantiate=False)

    errors = list(report.iter_level("error"))
    assert any("class-level 'declared_capabilities'" in i.message for i in errors)


def test_compliance_flags_open_provider_params() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_open_params_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    report = check_entrypoints(registry=registry, instantiate=False)

    errors = list(report.iter_level("error"))
    assert any("closed type" in i.message for i in errors)


def test_check_sync_bridge_passes_for_clean_session() -> None:
    report = compliance.check_sync_bridge(_BridgeSession)
    assert report.passed is True


def test_check_sync_bridge_detects_deadlock() -> None:
    report = compliance.check_sync_bridge(_HangBridgeSession, timeout=0.5)
    assert report.passed is False
    assert any("deadlock" in i.message for i in report.iter_level("error"))


def test_running_the_discovery_module_points_at_the_real_cli() -> None:
    """``python -m standard_asr.plugins.discovery`` exits with a signpost.

    The module has no CLI. Exiting silently would read as "no models
    discovered" to someone debugging plugin visibility -- the exact wrong
    conclusion -- so it exits loudly and names the tool that DOES list them.
    """
    module_file = inspect.getsourcefile(discovery_module)
    assert module_file is not None
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(module_file, run_name="__main__")

    message = str(excinfo.value.code)
    assert "standard-asr list" in message
    assert "--strict-discovery" in message


def test_create_refuses_an_outside_line_engine() -> None:
    # AR.1: the registry is the one construction seam every toolchain
    # consumer shares, so a mismatched installed plugin fails loudly at
    # creation instead of transcribing with possibly drifted semantics.
    eps = [
        EntryPoint(
            name="dummy/old",
            value="tests.test_discovery:_outside_line_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(ProtocolCompatibilityError, match="pre-stable line 0.2"):
        registry.create("dummy/old")


def test_create_preflights_the_class_line_before_the_factory() -> None:
    # The gate's placement is part of the contract: gating only after
    # construction let a construction-time fault (a missing credential, an
    # SDK failure) mask the line mismatch, sending the operator to debug
    # configuration for an engine this core cannot use at all. With the
    # class resolvable, the factory must never run.
    eps = [
        EntryPoint(
            name="dummy/old",
            value="tests.test_discovery:_counting_outside_line_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _outside_line_constructions.clear()
    with pytest.raises(ProtocolCompatibilityError, match="pre-stable line 0.2"):
        registry.create("dummy/old")
    assert _outside_line_constructions == []


def test_create_refuses_untyped_properties() -> None:
    # Fail closed: strictly less information must never earn strictly more
    # permission. A typed engine declaring "9.9.9" is refused, so a dict
    # declaration quacking the same field cannot be handed back transcribing.
    # The class is resolvable, so the preflight lands the precise class-level
    # verdict before construction.
    eps = [
        EntryPoint(
            name="dummy/duck",
            value="tests.test_discovery:_duck_properties_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(EngineContractError, match="class-level 'properties'"):
        registry.create("dummy/duck")


def test_create_refuses_untyped_properties_behind_an_opaque_factory() -> None:
    # With nothing resolvable statically and an untyped instance declaration,
    # the instance gate inside create() is the last net -- fail closed there
    # too, with the gate's own establishment message.
    eps = [
        EntryPoint(
            name="dummy/opaque-duck",
            value="tests.test_discovery:_opaque_duck_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(EngineContractError, match="cannot be established"):
        registry.create("dummy/opaque-duck")


class _ShadowedDummyASR(_DummyASR):
    """Class on the supported line; ``__init__`` shadows a divergent copy.

    The shadow stays ON the supported line (only ``model_name``
    differs) so the failure exercised is the equality check itself, not the
    line gate that would fire first for an off-line shadow.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.properties = _DummyProperties(model_name="shadow")  # type: ignore[misc]


def _shadowed_dummy_factory(**kwargs: Any) -> _ShadowedDummyASR:  # pyright: ignore[reportUnusedFunction]
    return _ShadowedDummyASR(**kwargs)


def test_create_refuses_an_instance_that_shadows_the_class_declaration() -> None:
    # One authoritative declaration: the class-read surfaces (show, metadata)
    # and the runtime gates must see the same properties, so a divergent
    # instance fails at the shared construction seam.
    eps = [
        EntryPoint(
            name="dummy/shadow",
            value="tests.test_discovery:_shadowed_dummy_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(EngineContractError, match="class-level declaration"):
        registry.create("dummy/shadow")


def test_create_refuses_a_typed_instance_without_a_class_declaration() -> None:
    # The middle case between "both untyped" and "typed but divergent": an
    # instance-built typed declaration with no class-level BaseProperties
    # passed the instance gate while the equality check silently skipped --
    # creation handed back an engine that show, compliance, and the
    # per-model endpoints all fail closed on (AR.1: the class declaration
    # is the authoritative one). The class is resolvable, so the refusal is
    # certain before construction -- running the factory first could only
    # mask this verdict behind a construction fault, so it must never run.
    eps = [
        EntryPoint(
            name="dummy/instance-only",
            value="tests.test_discovery:_instance_only_properties_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    _instance_only_constructions.clear()
    with pytest.raises(EngineContractError, match="class-level 'properties'"):
        registry.create("dummy/instance-only")
    assert _instance_only_constructions == []


def test_create_refuses_an_untyped_class_declaration_behind_an_opaque_factory() -> None:
    # The preflight needs a resolvable class; an opaque factory defers the
    # class-declaration verdict to the post-construction checks, which must
    # reach the same fail-closed refusal on type(engine).
    eps = [
        EntryPoint(
            name="dummy/opaque-instance-only",
            value="tests.test_discovery:_opaque_instance_only_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(EngineContractError, match="class-level 'properties'"):
        registry.create("dummy/opaque-instance-only")


def test_create_refuses_a_factory_returning_an_undeclared_class() -> None:
    # The class-read surfaces (show, compliance, the per-model endpoints)
    # project the class the entry point resolves to; the runtime executes
    # what the factory returns. create() binds the two with exact identity:
    # a subclass may override any class-level contract surface
    # (declared_metadata, capabilities, a public template), so isinstance
    # would re-open the certification/execution split (round-28 review, B2).
    eps = [
        EntryPoint(
            name="dummy/covariant",
            value="tests.test_discovery:_covariant_dummy_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(EngineContractError, match="not the declared class") as excinfo:
        registry.create("dummy/covariant")
    assert "_CovariantDummyASR" in str(excinfo.value)
    assert "_DummyASR" in str(excinfo.value)
