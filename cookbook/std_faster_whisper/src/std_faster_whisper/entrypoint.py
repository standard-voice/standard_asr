# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Entry point targets for the std-faster-whisper plugin demo.

One factory per preset (spec IC.7: model selection = entry-point preset, never an
init ``model`` field), so ``standard-asr models list`` / the registry / a settings
UI can enumerate the available models. Each factory's return annotation is the
**concrete** preset class (not the ``StandardASR`` protocol) so the registry can
resolve the class -- and read its class-level ``properties`` /
``declared_capabilities`` / ``provider_params_type`` -- WITHOUT instantiating the
engine (spec §3.1 / §C, ``ModelRegistry.engine_class``).
"""

from __future__ import annotations

from typing import Any

from .std_asr_faster_whisper import DistilLargeV3ASR, FasterWhisperASR, TurboASR


def create(**kwargs: Any) -> FasterWhisperASR:
    """Return the ``faster-whisper/large-v3`` preset (canonical multilingual).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`FasterWhisperASR`.

    Returns:
        A configured large-v3 engine.

    Raises:
        ValueError: If configuration validation fails.
    """

    return FasterWhisperASR(**kwargs)


def create_distil_large_v3(**kwargs: Any) -> DistilLargeV3ASR:
    """Return the ``faster-whisper/distil-large-v3`` preset (faster, distilled).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`DistilLargeV3ASR`.

    Returns:
        A configured distil-large-v3 engine.

    Raises:
        ValueError: If configuration validation fails.
    """

    return DistilLargeV3ASR(**kwargs)


def create_turbo(**kwargs: Any) -> TurboASR:
    """Return the ``faster-whisper/turbo`` preset (large-v3-turbo, fastest).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`TurboASR`.

    Returns:
        A configured turbo engine.

    Raises:
        ValueError: If configuration validation fails.
    """

    return TurboASR(**kwargs)
