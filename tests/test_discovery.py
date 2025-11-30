"""Tests covering plugin discovery and compliance helpers."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from typing import Any

import pytest

from standard_asr.compliance import check_entrypoints
from standard_asr.discovery import (
    discover_models,
    parse_entrypoint_name,
    pep503_normalize,
)
from standard_asr.exceptions import EntrypointValidationError


class _DummyASR:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def transcribe(self, audio: Any) -> str:  # pragma: no cover - dummy implementation
        return "dummy"


def _dummy_factory(**kwargs: Any) -> _DummyASR:
    return _DummyASR(**kwargs)


def _requires_argument_factory(
    required: str,
) -> _DummyASR:  # pragma: no cover - instantiation skipped
    return _DummyASR(required=required)


def _non_callable_factory() -> (
    str
):  # pragma: no cover - used to trigger compliance error
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
    # Default model is represented by empty model name
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
    try:
        discover_models(eps=eps, strict=True)
    except Exception:
        pass
    else:  # pragma: no cover - explicit failure message for readability
        raise AssertionError("strict discovery should raise for invalid names")


def test_compliance_reports_expected_issues() -> None:
    eps = [
        EntryPoint(
            name="good/model",
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

    # Should emit a warning for the factory that needs an argument
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
