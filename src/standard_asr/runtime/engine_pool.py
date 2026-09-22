# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Application-lifetime ownership for reusable Standard ASR engines."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import TracebackType
from typing import Any, cast

from standard_asr.contract.exceptions import EngineContractError
from standard_asr.plugins.discovery import ModelRegistry
from standard_asr.runtime.interface import StandardASR
from standard_asr.runtime.protocol_boundary import require_sync_result
from standard_asr.runtime.redaction import log_exception_safely

logger = logging.getLogger(__name__)


class EnginePoolClosedError(RuntimeError):
    """The application started shutting down before an engine lease began."""


@dataclass
class _Entry:
    """One configured engine and its active lease count."""

    engine: StandardASR
    active: int = 0


class EngineLease:
    """An active reference to one pooled engine.

    Release is idempotent so a transport error cannot decrement the pool twice.
    Use the lease as an async context manager around the complete operation,
    including a WebSocket session's teardown.
    """

    def __init__(self, pool: EnginePool, model: str, engine: StandardASR) -> None:
        self._pool = pool
        self._model = model
        self.engine = engine
        self._released = False
        self._release_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> StandardASR:
        """Return the leased engine.

        Returns:
            The shared engine instance.
        """
        return self.engine

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the lease after the operation ends.

        Args:
            exc_type: Exception type from the managed operation, if any.
            exc: Exception from the managed operation, if any.
            traceback: Exception traceback from the managed operation, if any.

        Returns:
            None.
        """
        del exc_type, exc, traceback
        await self.release()

    async def release(self) -> None:
        """Release this lease once.

        Returns:
            None.
        """
        if self._released:
            return
        if self._release_task is None:
            self._release_task = asyncio.create_task(
                self._pool._release(self._model)  # pyright: ignore[reportPrivateUsage]
            )
            self._pool._retain_release_task(  # pyright: ignore[reportPrivateUsage]
                self._release_task
            )
        await asyncio.shield(self._release_task)
        self._released = True

    def release_when_done(self, task: asyncio.Task[Any]) -> None:
        """Schedule release after an owning route task finishes.

        Args:
            task: Task whose complete lifetime owns this lease.

        Returns:
            None.
        """
        self._pool._release_when_done(task, self)  # pyright: ignore[reportPrivateUsage]


class EnginePool:
    """Reuse one configured engine per model for one application lifetime.

    Each application fixes at most one init-config mapping for a model. The
    resulting cache is therefore bounded by the discovered model set. Concurrent
    first requests share one construction task, and shutdown refuses new leases,
    waits for active work, then closes each constructed engine once.

    Args:
        registry: Registry used to construct engines.
        engine_configs: Optional operator-owned init config by full model key.

    Raises:
        ValueError: If config names a model absent from the registry.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        engine_configs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._registry = registry
        try:
            self._configs = deepcopy(
                {model: dict(config) for model, config in (engine_configs or {}).items()}
            )
        except Exception:  # noqa: BLE001
            raise TypeError("engine_configs could not be copied safely.") from None
        if not set(self._configs).issubset(registry.names()):
            raise ValueError("engine_configs contains a model key that is not in the registry.")
        self._condition = asyncio.Condition()
        self._entries: dict[str, _Entry] = {}
        self._pending: dict[str, asyncio.Task[_Entry]] = {}
        self._release_tasks: set[asyncio.Task[None]] = set()
        self._closing = False

    async def acquire(self, model: str) -> EngineLease:
        """Acquire one engine, constructing it once under concurrency.

        Args:
            model: Full model key.

        Returns:
            A lease that must remain active for the complete operation.

        Raises:
            EnginePoolClosedError: If shutdown has started.
            Exception: Any registry construction error.
        """
        async with self._condition:
            if self._closing:
                raise EnginePoolClosedError("The engine pool is shutting down.")
            entry = self._entries.get(model)
            if entry is None:
                pending = self._pending.get(model)
                if pending is None:
                    pending = asyncio.create_task(self._construct(model))
                    pending.add_done_callback(_consume_task_exception)
                    self._pending[model] = pending
            else:
                entry.active += 1
                return EngineLease(self, model, entry.engine)

        entry = await asyncio.shield(pending)
        async with self._condition:
            if self._closing:
                raise EnginePoolClosedError("The engine pool is shutting down.")
            entry.active += 1
            return EngineLease(self, model, entry.engine)

    async def _construct(self, model: str) -> _Entry:
        """Construct and publish one entry independently of caller cancellation.

        Args:
            model: Full model key.

        Returns:
            The published entry.

        Raises:
            Exception: Any registry construction error.
        """
        try:
            try:
                config = deepcopy(self._configs.get(model, {}))
            except Exception:  # noqa: BLE001
                raise EngineContractError(
                    "The stored engine config could not be copied for construction."
                ) from None
            engine = await asyncio.to_thread(self._registry.create, model, **config)
            _require_synchronous_close(engine)
            entry = _Entry(engine=engine)
            async with self._condition:
                self._entries[model] = entry
                return entry
        finally:
            async with self._condition:
                self._pending.pop(model, None)
                self._condition.notify_all()

    async def _release(self, model: str) -> None:
        """Release one active reference and wake shutdown.

        Args:
            model: Full model key.

        Returns:
            None.
        """
        async with self._condition:
            entry = self._entries[model]
            if entry.active <= 0:  # pragma: no cover - EngineLease prevents this state.
                raise RuntimeError("Engine lease count became negative.")
            entry.active -= 1
            self._condition.notify_all()

    def _release_when_done(self, owner: asyncio.Task[Any], lease: EngineLease) -> None:
        """Retain the background release task created for an ASGI route.

        Args:
            owner: Route task that owns the lease.
            lease: Lease to release when the route finishes.

        Returns:
            None.
        """

        def schedule(_task: asyncio.Task[Any]) -> None:
            release_task = owner.get_loop().create_task(lease.release())
            self._release_tasks.add(release_task)
            release_task.add_done_callback(self._release_tasks.discard)

        owner.add_done_callback(schedule)

    def _retain_release_task(self, task: asyncio.Task[None]) -> None:
        """Keep a release alive until it updates the active count.

        Args:
            task: Release task to retain.

        Returns:
            None.
        """
        self._release_tasks.add(task)
        task.add_done_callback(self._release_tasks.discard)

    async def aclose(self) -> None:
        """Wait for active work and close every constructed engine once.

        A close failure is an engine/resource fault. It is safe-logged without
        aborting later engines' cleanup or masking application shutdown.

        Returns:
            None.
        """
        async with self._condition:
            self._closing = True
            await self._condition.wait_for(
                lambda: (
                    not self._pending and all(entry.active == 0 for entry in self._entries.values())
                )
            )
            entries = list(self._entries.items())
            self._entries.clear()

        for model, entry in entries:
            try:
                result = await asyncio.to_thread(entry.engine.close)
                require_sync_result(result, "close()", expected_type=type(None))
            except Exception:  # noqa: BLE001
                log_exception_safely(logger, "Engine %r failed to close during shutdown", model)


def _consume_task_exception(task: asyncio.Task[_Entry]) -> None:
    """Retrieve an abandoned construction task's exception.

    Args:
        task: Completed construction task.

    Returns:
        None.
    """
    if not task.cancelled():
        task.exception()


def _require_synchronous_close(engine: StandardASR) -> None:
    """Reject an engine whose required close member cannot be called synchronously.

    The check does not invoke cleanup during construction. Shutdown still checks
    the actual return value so a synchronous wrapper that returns an awaitable is
    caught at the call boundary.

    Args:
        engine: Newly constructed engine, before pool publication.

    Returns:
        None.

    Raises:
        EngineContractError: If close is missing, non-callable, or declared async.
    """
    try:
        close = getattr(engine, "close")
        close_type = cast(Any, type(close))
        call = inspect.getattr_static(close_type, "__call__", None)
    except Exception as exc:  # noqa: BLE001
        raise EngineContractError(
            "Engine close lookup failed before the engine entered the server pool."
        ) from exc
    if not callable(close):
        raise EngineContractError(
            "The StandardASR close member must be callable before the engine enters the pool."
        )
    if inspect.iscoroutinefunction(close) or inspect.iscoroutinefunction(call):
        raise EngineContractError(
            "The StandardASR close member must be synchronous before the engine enters the pool."
        )


__all__ = ["EngineLease", "EnginePool", "EnginePoolClosedError"]
