"""Tests covering plugin discovery and compliance helpers."""

from __future__ import annotations

import asyncio
from importlib.metadata import EntryPoint, EntryPoints
from typing import Any, AsyncIterator, ClassVar, Literal

import pytest
from pydantic import ConfigDict

import standard_asr.compliance as compliance
from standard_asr import BaseConfig, BaseProperties, TranscriptionResult
from standard_asr.audio_input import InputKind
from standard_asr.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    FlagCap,
    LanguageCaps,
)
from standard_asr.compliance import check_entrypoints
from standard_asr.discovery import (
    ENTRYPOINT_GROUP,
    ModelRegistry,
    ModelSpec,
    _gather_entry_points,  # pyright: ignore[reportPrivateUsage]
    discover_models,
    parse_entrypoint_name,
    pep503_normalize,
    validate_engine_id,
    validate_model_name,
)
from standard_asr.exceptions import EntrypointValidationError, FactoryLoadError
from standard_asr.runtime_params import ProviderParams
from standard_asr.streaming import TranscriptionEvent, TranscriptionSession


class _DummyConfig(BaseConfig[Literal["dummy"]]):
    engine: Literal["dummy"] = "dummy"


class _DummyProperties(BaseProperties):
    engine_id: str = "dummy"
    model_name: str = "demo"
    protocol_version: str = "0.2.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | Literal["any"] = [16000]
    selectable_languages: list[str] = ["en"]


_DUMMY_CAPS = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
    )
)


class _DummyASR:
    properties: ClassVar[_DummyProperties] = _DummyProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _DUMMY_CAPS

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.config = _DummyConfig(engine="dummy")

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="dummy")


def _dummy_factory(**kwargs: Any) -> _DummyASR:  # pyright: ignore[reportUnusedFunction]
    return _DummyASR(**kwargs)


class _NotAnEngine:
    """A class an entry point might resolve to that is NOT a Standard ASR engine.

    It lacks the required class surface (``properties`` /
    ``declared_capabilities``), so ``engine_class`` must reject it with a clear
    ``FactoryLoadError`` instead of casting it through.
    """


def _not_an_engine_factory() -> _NotAnEngine:  # pyright: ignore[reportUnusedFunction]
    return _NotAnEngine()


def _unannotated_factory():  # type: ignore[no-untyped-def]  # pyright: ignore[reportUnusedFunction]
    return _DummyASR()


def _bad_annotation_factory():  # type: ignore[no-untyped-def]  # pyright: ignore[reportUnusedFunction]
    return _DummyASR()


# A return annotation naming a type that does not exist: typing.get_type_hints
# raises NameError when resolving it, so engine_class must surface a
# FactoryLoadError rather than crash.
_bad_annotation_factory.__annotations__ = {"return": "ThisTypeDoesNotExistAnywhere"}


class _OpenParams(ProviderParams):
    model_config = ConfigDict(extra="allow")  # violates §R.4 R1 (must be closed)


class _OpenParamsASR(_DummyASR):
    provider_params_type: ClassVar[type[ProviderParams] | None] = _OpenParams


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
        # Never terminates and never yields: simulates a deadlocking adapter.
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
    assert registry.by_engine("alpha") == ["alpha/first", "alpha/second"]
    spec = registry.spec("beta/")
    assert spec.model_name == ""


def test_discover_models_duplicate_strategy_replace() -> None:
    eps = [
        EntryPoint(
            name="alpha/only",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        ),
        EntryPoint(
            name="alpha/only",
            value="tests.test_discovery:_requires_argument_factory",
            group="standard_asr.models",
        ),
    ]
    registry = discover_models(eps=eps, strict=True, on_conflict="replace")
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


def test_validate_engine_id_accepts_non_canonical() -> None:
    # A non-canonical-but-valid id passes surface validation; canonicalisation
    # to the routing identity happens in parse_entrypoint_name / discover_models.
    validate_engine_id("my_engine")
    validate_model_name("model")


def test_parse_entrypoint_name_canonicalizes_engine_id() -> None:
    # IC.2: the routing identity is the PEP 503 canonical form, not the verbatim
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
    assert registry.by_engine("my-engine") == ["my-engine/first"]
    assert any("not PEP 503 normalized" in r.message for r in caplog.records)


def test_model_spec_load_factory_error_on_load() -> None:
    class _BadEntryPoint:
        def load(self) -> object:
            raise RuntimeError("boom")

    spec = ModelSpec(
        key="alpha/first",
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
        key="alpha/first",
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

    monkeypatch.setattr("standard_asr.discovery.entry_points", _entry_points)

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
    eps = [
        EntryPoint(
            name="alpha/dup",
            value="tests.test_discovery:_dummy_factory",
            group="standard_asr.models",
        ),
        EntryPoint(
            name="alpha/dup",
            value="tests.test_discovery:_requires_argument_factory",
            group="standard_asr.models",
        ),
    ]
    registry = discover_models(eps=eps, strict=True)

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
    assert any("BaseProperties" in issue.message for issue in errors)
    assert any("BaseConfig" in issue.message for issue in errors)


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


# ----- H6: no-instantiation engine class resolution ----------------------- #


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


def test_engine_class_raises_when_type_hints_unresolvable() -> None:
    # A factory whose return annotation references an undefined name makes
    # typing.get_type_hints raise; that must become a FactoryLoadError, not crash.
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_bad_annotation_factory",
            group="standard_asr.models",
        )
    ]
    registry = discover_models(eps=eps, strict=True)
    with pytest.raises(FactoryLoadError, match="type hints"):
        registry.engine_class("alpha/first")


# ----- IC.2: engine-identity collision detection -------------------------- #


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


def test_normalized_engine_id_collision_across_dists_is_shadowed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # DISC-1/H3: two distributions whose engine_ids only differ by PEP 503
    # normalisation (``my_engine`` vs ``my-engine``) route to the same canonical
    # id and the same env-var prefix, so they MUST be flagged as a collision.
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


# ----- H14: compliance class-level + sync-bridge checks -------------------- #


def test_compliance_flags_unreadable_class_metadata() -> None:
    eps = [
        EntryPoint(
            name="alpha/first",
            value="tests.test_discovery:_missing_meta_factory",
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
