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

import re

from .results import Segment, TranscriptionResult

#: Matches runs of two-or-more newlines (optionally with intervening blank
#: whitespace), i.e. the blank-line cue separator. Transcript text containing
#: such a run could otherwise forge or split cue blocks (cues are blank-line
#: delimited in both SRT and WebVTT).
_BLANK_LINE_RUN = re.compile(r"(?:\r?\n[ \t]*){2,}")


def _sanitize_cue_text(text: str, *, neutralize_arrow: bool) -> str:
    """Sanitize segment text so it cannot forge or break cue structure.

    Cue blocks in SRT and WebVTT are separated by blank lines, so a transcript
    containing an interior blank line (a double newline) followed by an index
    and a timestamp line could forge a new cue. WebVTT additionally treats
    ``-->`` as the cue timing delimiter, so it MUST NOT appear in cue payload.

    Args:
        text: Raw segment text.
        neutralize_arrow: Whether to neutralize ``-->`` (required for WebVTT).

    Returns:
        Text safe to interpolate into a cue block: leading/trailing whitespace
        stripped, interior blank-line runs collapsed to a single newline, and
        (for WebVTT) ``-->`` replaced so it cannot be read as cue timing.
    """
    collapsed = _BLANK_LINE_RUN.sub("\n", text.strip())
    if neutralize_arrow:
        collapsed = collapsed.replace("-->", "->")
    return collapsed


def _format_timestamp(seconds: float, *, millis_sep: str) -> str:
    """Format a time offset as ``HH:MM:SS<sep>mmm``.

    The renderer trusts the validated data model: :class:`~standard_asr.results.Segment`
    / :class:`~standard_asr.results.Word` guarantee a non-negative finite
    ``start`` / ``end`` (spec TR.2), so no negative offset can reach here. The
    renderer therefore does NOT clamp negatives -- clamping would silently mask
    an upstream timestamp bug (a wrong result), and the model already rejects one
    loudly at construction.

    Args:
        seconds: Time offset in seconds (non-negative, finite).
        millis_sep: Separator before milliseconds (``","`` SRT, ``"."`` VTT).

    Returns:
        The formatted timestamp string.
    """
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_sep}{millis:03d}"


def _cues(result: TranscriptionResult) -> list[Segment]:
    """Return the segments to render, falling back to a single full-text cue.

    The §TR.1 null rule distinguishes the two empty states: ``segments is None``
    means segmentation was *not requested / not applicable*, whereas
    ``segments == []`` means it *was requested but is empty* (e.g. confirmed
    silence). Only the former may fall back to a synthetic whole-text cue; an
    explicit ``[]`` yields zero cues, never a fabricated full-span cue.

    Segments are sorted by ``(start, channel)`` to enforce the §TR.2 top-level
    ordering invariant at the rendering boundary, so out-of-order input still
    produces correctly ordered subtitles. ``channel`` may be ``None``; it sorts
    before any explicit channel index (which the data model constrains to
    ``>= 0``).

    Args:
        result: The transcription result.

    Returns:
        The segments to render, ordered by ``(start, channel)``. For
        ``segments == []`` this is empty. When ``segments is None`` and
        ``text`` is non-empty, a single synthetic segment spanning
        ``[0, duration]`` with the full text is returned; when ``text`` is
        empty too, no cues are produced.
    """
    if result.segments is not None:
        return sorted(
            result.segments,
            key=lambda s: (s.start, s.channel if s.channel is not None else -1),
        )
    if not result.text:
        return []
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
    index = 1
    for segment in _cues(result):
        text = _sanitize_cue_text(segment.text, neutralize_arrow=False)
        if not text:
            # An empty / whitespace-only segment would yield a cue with no
            # payload (an index + timing line followed by a blank line), which
            # strict SRT parsers reject. Skip it; indices stay contiguous
            # because they are only advanced for emitted cues.
            continue
        start = _format_timestamp(segment.start, millis_sep=",")
        end = _format_timestamp(segment.end, millis_sep=",")
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
        index += 1
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
        text = _sanitize_cue_text(segment.text, neutralize_arrow=True)
        if not text:
            # A WebVTT cue with no payload line is malformed; skip empty /
            # whitespace-only segments rather than emit a payload-less block.
            continue
        start = _format_timestamp(segment.start, millis_sep=".")
        end = _format_timestamp(segment.end, millis_sep=".")
        blocks.append(f"{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


__all__ = ["to_srt", "to_vtt"]
