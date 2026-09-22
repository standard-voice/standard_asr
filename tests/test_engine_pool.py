# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for application-lifetime engine ownership."""

from __future__ import annotations

import asyncio
from importlib.metadata import EntryPoint
from typing import Any, ClassVar, Literal

import pytest
from pydantic import BaseModel

from standard_asr import TranscriptionResult
from standard_asr.audio.input import InputKind
from standard_asr.contract.capabilities import BatchCapabilities, DeclaredCapabilities
from standard_asr.contract.exceptions import EngineContractError
from standard_asr.engine import BaseConfig, BaseProperties, SampleRateRange
from standard_asr.plugins.discovery import discover_models
from standard_asr.runtime.engine_pool import EnginePool, EnginePoolClosedError


class _Config(BaseConfig[str]):
    engine: str = "pooled"


class _Properties(BaseProperties):
    engine_id: str = "pooled"
    model_name: str = "demo"
    protocol_version: str = "0.2.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]


_CONSTRUCTIONS: list[_Engine] = []


class _Engine:
    properties: ClassVar[BaseProperties] = _Properties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        batch=BatchCapabilities()
    )
    provider_params_type = None

    def __init__(self, **config: Any) -> None:
        self.config = _Config(engine="pooled", **config)
        self.close_calls = 0
        _CONSTRUCTIONS.append(self)

    def transcribe(self, audio: Any, params: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="pooled")

    def close(self) -> None:
        self.close_calls += 1


def _factory(**config: Any) -> _Engine:  # pyright: ignore[reportUnusedFunction]
    return _Engine(**config)


def _registry():
    return _registry_for("_factory")


def _registry_for(factory: str):
    return discover_models(
        eps=[
            EntryPoint(
                name="pooled/demo",
                value=f"tests.test_engine_pool:{factory}",
                group="standard_asr.models",
            )
        ],
        strict=True,
    )


class _CloseValidationEngine(_Engine):
    """Raises a validation error carrying an input value during close."""

    def close(self) -> None:
        class _Secret(BaseModel):
            api_key: int

        _Secret.model_validate({"api_key": "sk-CLOSE-SECRET"})


def _close_validation_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _CloseValidationEngine
):
    return _CloseValidationEngine()


class _AsyncCallable:
    async def __call__(self) -> None:
        return None


class _AsyncCallableCloseEngine(_Engine):
    close: ClassVar[Any] = _AsyncCallable()


def _async_callable_close_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _AsyncCallableCloseEngine
):
    return _AsyncCallableCloseEngine()


class _NestedConfig(BaseConfig[str]):
    engine: str = "pooled"
    settings: dict[str, list[str]]


class _NestedEngine(_Engine):
    def __init__(self, **config: Any) -> None:
        self.config = _NestedConfig(engine="pooled", **config)
        self.close_calls = 0


def _nested_factory(**config: Any) -> _NestedEngine:  # pyright: ignore[reportUnusedFunction]
    return _NestedEngine(**config)


_MUTATING_FACTORY_INPUTS: list[list[str]] = []


def _mutating_failure_factory(  # pyright: ignore[reportUnusedFunction]
    **config: Any,
) -> _NestedEngine:
    items = config["settings"]["items"]
    items.append("factory")
    _MUTATING_FACTORY_INPUTS.append(list(items))
    if len(_MUTATING_FACTORY_INPUTS) == 1:
        raise RuntimeError("first construction fails after mutating its kwargs")
    return _NestedEngine(**config)


def _compliance_registry(factory: str):
    return discover_models(
        eps=[
            EntryPoint(
                name="dummy/demo",
                value=f"tests.test_compliance:{factory}",
                group="standard_asr.models",
            )
        ],
        strict=True,
    )


def test_concurrent_acquire_constructs_once_and_shutdown_waits() -> None:
    async def scenario() -> None:
        _CONSTRUCTIONS.clear()
        pool = EnginePool(_registry())
        first, second = await asyncio.gather(
            pool.acquire("pooled/demo"), pool.acquire("pooled/demo")
        )
        assert first.engine is second.engine
        assert len(_CONSTRUCTIONS) == 1

        closing = asyncio.create_task(pool.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        await first.release()
        await asyncio.sleep(0)
        assert not closing.done()
        await second.release()
        await closing

        assert _CONSTRUCTIONS[0].close_calls == 1
        with pytest.raises(EnginePoolClosedError):
            await pool.acquire("pooled/demo")

    asyncio.run(scenario())


def test_pool_passes_fixed_model_config() -> None:
    async def scenario() -> None:
        _CONSTRUCTIONS.clear()
        pool = EnginePool(_registry(), engine_configs={"pooled/demo": {"strict": False}})
        lease = await pool.acquire("pooled/demo")
        assert lease.engine.config.strict is False
        await lease.release()
        await pool.aclose()

    asyncio.run(scenario())


def test_pool_rejects_config_for_unknown_model() -> None:
    with pytest.raises(ValueError, match="not in the registry"):
        EnginePool(_registry(), engine_configs={"missing/model": {}})


@pytest.mark.parametrize("factory", ["no_close_factory", "async_close_factory"])
def test_invalid_close_contract_is_rejected_before_service(factory: str) -> None:
    async def scenario() -> None:
        pool = EnginePool(_compliance_registry(factory))
        with pytest.raises(EngineContractError, match="close"):
            await pool.acquire("dummy/demo")
        await pool.aclose()

    asyncio.run(scenario())


def test_async_callable_close_is_rejected_before_service() -> None:
    async def scenario() -> None:
        pool = EnginePool(_registry_for("_async_callable_close_factory"))
        with pytest.raises(EngineContractError, match="synchronous"):
            await pool.acquire("pooled/demo")
        await pool.aclose()

    asyncio.run(scenario())


def test_pool_snapshots_nested_config_before_first_use() -> None:
    async def scenario() -> None:
        supplied = {"pooled/demo": {"settings": {"items": ["base"]}}}
        pool = EnginePool(_registry_for("_nested_factory"), engine_configs=supplied)
        supplied["pooled/demo"]["settings"]["items"].append("caller")

        lease = await pool.acquire("pooled/demo")
        assert isinstance(lease.engine.config, _NestedConfig)
        assert lease.engine.config.settings == {"items": ["base"]}
        await lease.release()
        await pool.aclose()

    asyncio.run(scenario())


def test_failed_factory_mutation_does_not_change_retry_config() -> None:
    async def scenario() -> None:
        _MUTATING_FACTORY_INPUTS.clear()
        pool = EnginePool(
            _registry_for("_mutating_failure_factory"),
            engine_configs={"pooled/demo": {"settings": {"items": ["base"]}}},
        )
        with pytest.raises(RuntimeError, match="first construction"):
            await pool.acquire("pooled/demo")

        lease = await pool.acquire("pooled/demo")
        assert _MUTATING_FACTORY_INPUTS == [
            ["base", "factory"],
            ["base", "factory"],
        ]
        assert isinstance(lease.engine.config, _NestedConfig)
        assert lease.engine.config.settings == {"items": ["base", "factory"]}
        await lease.release()
        await pool.aclose()

    asyncio.run(scenario())


def test_close_failure_is_sanitized_and_does_not_escape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        pool = EnginePool(_registry_for("_close_validation_factory"))
        lease = await pool.acquire("pooled/demo")
        await lease.release()
        await pool.aclose()

    with caplog.at_level("ERROR"):
        asyncio.run(scenario())

    assert "failed to close" in caplog.text
    assert "sk-CLOSE-SECRET" not in caplog.text


def test_canceled_release_still_unblocks_shutdown() -> None:
    async def scenario() -> None:
        _CONSTRUCTIONS.clear()
        pool = EnginePool(_registry())
        lease = await pool.acquire("pooled/demo")

        await pool._condition.acquire()  # pyright: ignore[reportPrivateUsage]
        releasing = asyncio.create_task(lease.release())
        await asyncio.sleep(0)
        releasing.cancel()
        await asyncio.gather(releasing, return_exceptions=True)
        pool._condition.release()  # pyright: ignore[reportPrivateUsage]

        await asyncio.wait_for(pool.aclose(), timeout=1)
        assert _CONSTRUCTIONS[0].close_calls == 1

    asyncio.run(scenario())
