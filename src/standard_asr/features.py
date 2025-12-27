"""Feature flag definitions for Standard ASR.

This module defines the standardized optional capabilities that an ASR engine
may support. Engines should declare supported features via ``BaseProperties``.
"""

from __future__ import annotations

from enum import Enum


class FeatureFlag(str, Enum):
    """Enumerate standardized optional ASR features.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.

    Attributes:
        STREAMING_INPUT: Supports chunked/streaming audio input.
        STREAMING_OUTPUT: Supports incremental/streaming transcription output.
        WORD_TIMESTAMPS: Supports word-level timestamps in results.
        SPEAKER_DIARIZATION: Supports speaker diarization (speaker labels).
        TRANSLATION: Supports translate task (speech -> target language text).
        LANGUAGE_DETECTION: Supports automatic language detection.
        VAD: Supports voice activity detection controls.
    """

    STREAMING_INPUT = "streaming_input"
    STREAMING_OUTPUT = "streaming_output"
    WORD_TIMESTAMPS = "word_timestamps"
    SPEAKER_DIARIZATION = "speaker_diarization"
    TRANSLATION = "translation"
    LANGUAGE_DETECTION = "language_detection"
    VAD = "vad"


__all__ = ["FeatureFlag"]
