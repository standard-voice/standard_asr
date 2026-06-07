# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Subtitle renderers for transcription results (SRT / VTT).

The core library renders the constant :class:`~standard_asr.results.TranscriptionResult`
into SRT and VTT, so every compliant engine gets subtitle output for free
(spec, section "Transcription Result", rule TR.6). This replaces the old
``response_format`` knob: rendering is a post-hoc transformation, not a request
parameter. Provider-rendered high-fidelity formats remain available only via
``result.extra["provider_formats"]``.
"""

from __future__ import annotations

from .results import Segment, TranscriptionResult


def _format_timestamp(seconds: float, *, millis_sep: str) -> str:
    """Format a time offset as ``HH:MM:SS<sep>mmm``.

    Negative values are clamped to zero.

    Args:
        seconds: Time offset in seconds.
        millis_sep: Separator before milliseconds (``","`` SRT, ``"."`` VTT).

    Returns:
        The formatted timestamp string.
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_sep}{millis:03d}"


def _cues(result: TranscriptionResult) -> list[Segment]:
    """Return the segments to render, falling back to a single full-text cue.

    Args:
        result: The transcription result.

    Returns:
        A list of segments. When the result has no segments, a single synthetic
        segment spanning ``[0, duration]`` with the full text is returned.
    """
    if result.segments:
        return result.segments
    end = result.duration if result.duration is not None else 0.0
    return [Segment(start=0.0, end=end, text=result.text)]


def to_srt(result: TranscriptionResult) -> str:
    """Render a transcription result as SRT.

    Args:
        result: The transcription result to render.

    Returns:
        The SRT document as a string.
    """
    blocks: list[str] = []
    for index, segment in enumerate(_cues(result), start=1):
        start = _format_timestamp(segment.start, millis_sep=",")
        end = _format_timestamp(segment.end, millis_sep=",")
        blocks.append(f"{index}\n{start} --> {end}\n{segment.text.strip()}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(result: TranscriptionResult) -> str:
    """Render a transcription result as WebVTT.

    Args:
        result: The transcription result to render.

    Returns:
        The WebVTT document as a string.
    """
    blocks: list[str] = ["WEBVTT"]
    for segment in _cues(result):
        start = _format_timestamp(segment.start, millis_sep=".")
        end = _format_timestamp(segment.end, millis_sep=".")
        blocks.append(f"{start} --> {end}\n{segment.text.strip()}")
    return "\n\n".join(blocks) + "\n"


__all__ = ["to_srt", "to_vtt"]
