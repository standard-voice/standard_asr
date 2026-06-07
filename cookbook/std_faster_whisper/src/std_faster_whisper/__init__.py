# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard ASR plugin wrapping faster-whisper."""

from .entrypoint import create
from .std_asr_faster_whisper import (
    FasterWhisperASR,
    FasterWhisperConfig,
    FasterWhisperParams,
    FasterWhisperProperties,
)

__all__ = [
    "FasterWhisperASR",
    "FasterWhisperConfig",
    "FasterWhisperParams",
    "FasterWhisperProperties",
    "create",
]
