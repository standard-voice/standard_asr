# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Entry point targets for the std-faster-whisper plugin demo."""

from __future__ import annotations

from typing import Any

from .std_asr_faster_whisper import FasterWhisperASR


def create(**kwargs: Any) -> FasterWhisperASR:
    """Return a configured :class:`FasterWhisperASR` instance.

    The return annotation is the **concrete** engine class (not the
    ``StandardASR`` protocol) so the registry can resolve the class -- and read
    its class-level ``declared_capabilities`` / ``provider_params_type`` -- without
    instantiating the engine (spec §3.1 / §C, ``ModelRegistry.engine_class``).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`FasterWhisperASR`.

    Returns:
        Configured Standard ASR implementation.

    Raises:
        ValueError: If configuration validation fails.
    """

    return FasterWhisperASR(**kwargs)
