# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for pooled-engine cancellation and defensive boundaries."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import pytest

from standard_asr.contract.exceptions import EngineContractError
from standard_asr.runtime.engine_pool import EnginePool, EnginePoolClosedError


class _Engine:
    """Minimal engine whose shutdown is observable."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _Registry:
    """Small registry stand-in that preserves the pool's public contract."""

    def __init__(self, create: Callable[..., Any]) -> None:
        self._create = create

    def names(self) -> list[str]:
        return ["dummy/model"]

    def create(self, model: str, **config: Any) -> Any:
        assert model == "dummy/model"
        return self._create(**config)


def _pool(create: Callable[..., Any]) -> EnginePool:
    return EnginePool(_Registry(create))  # type: ignore[arg-type]


def test_release_can_be_retried_after_caller_cancellation() -> None:
    """A canceled waiter can rejoin the retained release and release only once."""

    async def scenario() -> None:
        engine = _Engine()

        def create_engine(**_config: Any) -> _Engine:
            return engine

        pool = _pool(create_engine)
        lease = await pool.acquire("dummy/model")

        await pool._condition.acquire()  # pyright: ignore[reportPrivateUsage]
        first_release = asyncio.create_task(lease.release())
        await asyncio.sleep(0)
        first_release.cancel()
        await asyncio.gather(first_release, return_exceptions=True)
        pool._condition.release()  # pyright: ignore[reportPrivateUsage]

        await lease.release()
        await lease.release()
        await pool.aclose()

        assert engine.close_calls == 1

    asyncio.run(scenario())


def test_pool_rejects_config_that_cannot_be_snapshotted() -> None:
    """Operator config must be safely copied before the pool accepts it."""

    class _Uncopyable:
        def __deepcopy__(self, _memo: dict[int, object]) -> object:
            raise RuntimeError("copy failed")

    with pytest.raises(TypeError, match="could not be copied safely"):
        EnginePool(
            _Registry(lambda **_config: _Engine()),  # type: ignore[arg-type]
            engine_configs={"dummy/model": {"value": _Uncopyable()}},
        )


def test_pool_rejects_config_that_cannot_be_recopied_for_construction() -> None:
    """A stored snapshot must remain copyable for every construction attempt."""

    class _FailsOnSecondCopy:
        copies = 0

        def __deepcopy__(self, _memo: dict[int, object]) -> object:
            type(self).copies += 1
            if type(self).copies == 2:
                raise RuntimeError("second copy failed")
            return self

    async def scenario() -> None:
        _FailsOnSecondCopy.copies = 0
        pool = EnginePool(
            _Registry(lambda **_config: _Engine()),  # type: ignore[arg-type]
            engine_configs={"dummy/model": {"value": _FailsOnSecondCopy()}},
        )

        with pytest.raises(EngineContractError, match="stored engine config"):
            await pool.acquire("dummy/model")
        await pool.aclose()

    asyncio.run(scenario())


def test_shutdown_refuses_a_lease_that_finishes_constructing_too_late() -> None:
    """Construction may finish during shutdown, but no new lease may escape."""

    async def scenario() -> None:
        started = threading.Event()
        unblock = threading.Event()
        engine = _Engine()

        def create(**_config: Any) -> _Engine:
            started.set()
            assert unblock.wait(timeout=1)
            return engine

        pool = _pool(create)
        acquiring = asyncio.create_task(pool.acquire("dummy/model"))
        assert await asyncio.to_thread(started.wait, 1)
        closing = asyncio.create_task(pool.aclose())
        await asyncio.sleep(0)
        unblock.set()

        with pytest.raises(EnginePoolClosedError):
            await acquiring
        await closing
        assert engine.close_calls == 1

    asyncio.run(scenario())


def test_canceled_construction_callback_does_not_read_an_exception() -> None:
    """The pool's completion callback accepts a canceled construction task."""

    async def scenario() -> None:
        blocker = asyncio.Event()

        async def wait_forever() -> Any:
            await blocker.wait()
            return None

        task = asyncio.create_task(wait_forever())
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        from standard_asr.runtime.engine_pool import (
            _consume_task_exception,  # pyright: ignore[reportPrivateUsage]
        )

        _consume_task_exception(task)  # type: ignore[arg-type]

    asyncio.run(scenario())


def test_close_lookup_failure_rejects_engine_before_publication() -> None:
    """A broken close descriptor becomes a stable engine contract error."""

    class _BrokenCloseEngine:
        def __getattribute__(self, name: str) -> object:
            if name == "close":
                raise RuntimeError("descriptor failed")
            return super().__getattribute__(name)

    async def scenario() -> None:
        def create_engine(**_config: Any) -> _BrokenCloseEngine:
            return _BrokenCloseEngine()

        pool = _pool(create_engine)
        with pytest.raises(EngineContractError, match="close lookup failed") as exc_info:
            await pool.acquire("dummy/model")
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        await pool.aclose()

    asyncio.run(scenario())
