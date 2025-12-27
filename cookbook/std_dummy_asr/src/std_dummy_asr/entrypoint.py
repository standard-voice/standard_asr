"""Entrypoint targets exposed under ``standard_asr.models``."""

from __future__ import annotations

from typing import Any

from standard_asr import StandardASR

from .engine import DummyASR


def create_echo(**kwargs: Any) -> StandardASR:
    """Create the ``dummy/echo`` preset.

    Args:
        **kwargs: Optional overrides forwarded to :class:`~std_dummy_asr.engine.DummyASR`.

    Returns:
        Configured :class:`DummyASR` instance.

    Raises:
        ValueError: If configuration validation fails.
    """

    return DummyASR(**kwargs)


def create_default(**kwargs: Any) -> StandardASR:
    """Return the default dummy model aliasing :func:`create_echo`.

    Args:
        **kwargs: Optional overrides forwarded to :func:`create_echo`.

    Returns:
        Configured :class:`DummyASR` instance.

    Raises:
        ValueError: If configuration validation fails.
    """

    return create_echo(**kwargs)
