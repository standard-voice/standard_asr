"""Tests covering plugin discovery and compliance helpers."""

from __future__ import annotations

from importlib.metadata import EntryPoint, EntryPoints
from typing import Any, ClassVar, Literal

import pytest

from standard_asr import BaseConfig, BaseProperties, TranscriptionResult
from standard_asr.audio_input import InputKind
from standard_asr.compliance import check_entrypoints
import standard_asr.compliance as compliance
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


class _DummyConfig(BaseConfig[Literal["dummy"]]):
    engine: Literal["dummy"] = "dummy"


class _DummyProperties(BaseProperties):
    engine_id: str = "dummy"
    model_name: str = "demo"
    protocol_version: str = "0.2.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] = [16000]
    selectable_languages: list[str] = ["en"]


class _DummyASR:
    properties: ClassVar[_DummyProperties] = _DummyProperties()

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.config = _DummyConfig(engine="dummy")

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="dummy")


def _dummy_factory(**kwargs: Any) -> _DummyASR:  # pyright: ignore[reportUnusedFunction]
    return _DummyASR(**kwargs)


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


def test_validate_engine_id_logs_guidance(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    validate_engine_id("my_engine")
    validate_model_name("model")

    assert any("PEP 503" in record.message for record in caplog.records)


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
