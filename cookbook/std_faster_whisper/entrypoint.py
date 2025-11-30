"""Entry point targets for the std-faster-whisper plugin demo."""

from __future__ import annotations

from typing import Any

from standard_asr import StandardASR

from .std_asr_faster_whisper import FasterWhisperASR


def create(**kwargs: Any) -> StandardASR:
    """Return a configured :class:`FasterWhisperASR` instance.

    Args:
        **kwargs: Keyword arguments forwarded to :class:`FasterWhisperASR`.

    Returns:
        Configured Standard ASR implementation.
    """

    return FasterWhisperASR(**kwargs)
