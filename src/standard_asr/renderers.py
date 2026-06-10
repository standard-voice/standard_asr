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
#: delimited in both SRT and WebVTT). Line terminators are normalized to ``\n``
#: by :func:`_sanitize_cue_text` *before* this runs, so matching ``\n`` alone is
#: sufficient (a lone ``\r`` -- a line terminator in both WebVTT and many SRT
#: parsers -- can no longer slip past as an unrecognized newline form).
_BLANK_LINE_RUN = re.compile(r"(?:\n[ \t]*){2,}")

#: End time (seconds) of the synthetic whole-text cue when ``duration`` is
#: unknown. Players (ffmpeg, VLC, browser WebVTT) silently drop zero-duration
#: cues, so the fallback cue MUST have a non-zero span to display at all; 3 s
#: is long enough to be visible and short enough to read as synthetic.
_SYNTHETIC_CUE_FALLBACK_END = 3.0


def _sanitize_cue_text(text: str, *, escape_markup: bool) -> str:
    """Sanitize segment text so it cannot forge or break cue structure.

    Cue blocks in SRT and WebVTT are separated by blank lines, so a transcript
    containing an interior blank line followed by an index and a timestamp line
    could forge a new cue. WebVTT additionally parses ``&`` and ``<`` as markup:
    a bare ``<`` opens a cue-span tag that the browser's WebVTT tokenizer
    consumes up to the next ``>``, so unescaped ``<`` in cue text (e.g. an
    engine-leaked ``<unk>`` token, ``<i>``, or "a < b") makes the browser
    *silently drop* that span -- the cardinal silent-wrong-result sin. ``&``
    likewise begins a character reference. Per the W3C WebVTT cue-text grammar
    the standard renderer therefore escapes ``&`` -> ``&amp;``, ``<`` ->
    ``&lt;``, and ``>`` -> ``&gt;`` (the ``&`` substitution runs first so the
    ``&`` it introduces is not re-escaped; escaping ``>`` also neutralizes any
    ``-->`` so it can never be read as cue timing).

    SRT has no entity-reference mechanism, so ``escape_markup`` is ``False`` for
    SRT: escaping there would surface the literal ``&amp;`` / ``&lt;`` to the
    viewer. (Angle-bracket text is passed through verbatim in SRT; see
    :func:`to_srt`.)

    Line terminators are normalized to ``\\n`` first so a lone ``\\r`` -- a
    valid line terminator in WebVTT and many SRT parsers -- cannot slip past the
    blank-line collapse and forge a cue via ``\\r\\r``.

    Args:
        text: Raw segment text.
        escape_markup: Whether to escape WebVTT markup metacharacters
            (``& < >``). ``True`` for WebVTT, ``False`` for SRT.

    Returns:
        Text safe to interpolate into a cue block: line terminators normalized,
        leading/trailing whitespace stripped, interior blank-line runs collapsed
        to a single newline, and (for WebVTT) ``& < >`` escaped as character
        references so payload text can neither be parsed as markup/cue-timing
        nor be silently dropped.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    collapsed = _BLANK_LINE_RUN.sub("\n", normalized.strip())
    if escape_markup:
        collapsed = collapsed.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
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
        ``[0, duration]`` with the full text is returned -- or
        ``[0, 3 s]`` when ``duration`` is unknown (e.g. a reduced stream),
        because players silently drop zero-duration cues; when ``text`` is
        empty too, no cues are produced.
    """
    if result.segments is not None:
        return sorted(
            result.segments,
            key=lambda s: (s.start, s.channel if s.channel is not None else -1),
        )
    if not result.text:
        return []
    end = result.duration if result.duration is not None else _SYNTHETIC_CUE_FALLBACK_END
    return [Segment(start=0.0, end=end, text=result.text)]


def to_srt(result: TranscriptionResult) -> str:
    """Render a transcription result as SRT.

    Cue text is sanitized so it cannot forge cue structure (line terminators
    normalized, interior blank-line runs collapsed). Unlike :func:`to_vtt`, SRT
    has **no character-reference mechanism**, so ``&`` and angle brackets in
    transcript text are emitted verbatim: an engine-leaked ``<unk>`` token or
    ``<i>`` is passed through as-is. Most SRT players render angle-bracket text
    literally, but some interpret a subset of HTML-like tags; if a downstream
    consumer must neutralize tags, do so on the transcript text before
    rendering. (WebVTT, which mandates escaping, is handled by :func:`to_vtt`.)

    Segment fallback (spec TR.6): when ``result.segments is None`` (segmentation
    not requested/applicable) but ``result.text`` is non-empty, a single cue
    spanning the whole text is synthesized -- ``[0, duration]``, or ``[0, 3 s]``
    when ``duration`` is unknown (players silently drop zero-duration cues).
    ``segments == []`` (requested but empty, e.g. silence) yields no cues. Pass a
    segmented result for time-accurate subtitles.

    Args:
        result: The transcription result to render.

    Returns:
        The SRT document as a string.
    """
    blocks: list[str] = []
    index = 1
    for segment in _cues(result):
        text = _sanitize_cue_text(segment.text, escape_markup=False)
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

    Cue text is escaped per the W3C WebVTT cue-text grammar (``&`` -> ``&amp;``,
    ``<`` -> ``&lt;``, ``>`` -> ``&gt;``) so payload text -- including
    engine-leaked ``<unk>`` / ``<|...|>`` tokens or "AT&T" -- is shown verbatim
    instead of being silently dropped by the browser's cue-span tokenizer. Cue
    structure is also protected (line terminators normalized, blank-line runs
    collapsed, ``-->`` neutralized by the ``>`` escape).

    Segment fallback (spec TR.6): when ``result.segments is None`` (segmentation
    not requested/applicable) but ``result.text`` is non-empty, a single cue
    spanning the whole text is synthesized -- ``[0, duration]``, or ``[0, 3 s]``
    when ``duration`` is unknown (players silently drop zero-duration cues).
    ``segments == []`` (requested but empty, e.g. silence) yields no cues. Pass a
    segmented result for time-accurate subtitles.

    Args:
        result: The transcription result to render.

    Returns:
        The WebVTT document as a string.
    """
    blocks: list[str] = ["WEBVTT"]
    for segment in _cues(result):
        text = _sanitize_cue_text(segment.text, escape_markup=True)
        if not text:
            # A WebVTT cue with no payload line is malformed; skip empty /
            # whitespace-only segments rather than emit a payload-less block.
            continue
        start = _format_timestamp(segment.start, millis_sep=".")
        end = _format_timestamp(segment.end, millis_sep=".")
        blocks.append(f"{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


__all__ = ["to_srt", "to_vtt"]
