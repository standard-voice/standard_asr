# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard ASR plugin wrapping faster-whisper."""

from .entrypoint import create, create_distil_large_v3, create_turbo
from .std_asr_faster_whisper import (
    DistilLargeV3ASR,
    DistilLargeV3Properties,
    FasterWhisperASR,
    FasterWhisperConfig,
    FasterWhisperParams,
    FasterWhisperProperties,
    TurboASR,
    TurboProperties,
)

__all__ = [
    "DistilLargeV3ASR",
    "DistilLargeV3Properties",
    "FasterWhisperASR",
    "FasterWhisperConfig",
    "FasterWhisperParams",
    "FasterWhisperProperties",
    "TurboASR",
    "TurboProperties",
    "create",
    "create_distil_large_v3",
    "create_turbo",
]
