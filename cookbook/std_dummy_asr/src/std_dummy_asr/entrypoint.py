# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint targets exposed under ``standard_asr.models``."""

from __future__ import annotations

from typing import Any

from .engine import DummyASR, DummyDefaultASR


def create_echo(**kwargs: Any) -> DummyASR:
    """Create the ``dummy/echo`` preset.

    The return annotation is the **concrete** engine class so the registry can
    resolve it and read its class-level metadata without instantiation
    (``ModelRegistry.engine_class``; spec §3.1 / §C).

    Args:
        **kwargs: Optional overrides forwarded to :class:`~std_dummy_asr.engine.DummyASR`.

    Returns:
        Configured :class:`DummyASR` instance.

    Raises:
        ValueError: If configuration validation fails.
    """

    return DummyASR(**kwargs)


def create_default(**kwargs: Any) -> DummyDefaultASR:
    """Return the default dummy model aliasing :func:`create_echo`.

    Args:
        **kwargs: Optional overrides forwarded to :func:`create_echo`.

    Returns:
        Configured :class:`DummyDefaultASR` instance.

    Raises:
        ValueError: If configuration validation fails.
    """

    return DummyDefaultASR(**kwargs)
