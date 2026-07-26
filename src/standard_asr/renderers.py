# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Subtitle renderers for transcription results (SRT / VTT).

The core library renders the constant :class:`~standard_asr.contract.results.TranscriptionResult`
into SRT and VTT, so every compliant engine gets subtitle output for free
(spec, section "Transcription Result"). This replaces the old
``response_format`` knob: rendering is a post-hoc transformation, not a request
parameter. Provider-rendered high-fidelity formats remain available only via
``result.extra["provider_formats"]``. Speaker labels are rendered only on
explicit opt-in (``include_speakers=True``): SRT prefixes the cue
text with ``[<label>]: ``, WebVTT wraps the cue body in a ``<v <label>>`` span.
"""

from __future__ import annotations

import re

from standard_asr.contract.results import (
    DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE,
    Segment,
    TranscriptionResult,
)

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


def _sanitize_speaker_label(label: str, *, escape_markup: bool) -> str:
    """Sanitize a speaker label for interpolation into a cue block.

    The model validators (:func:`~standard_asr.contract.results.validate_speaker_label`)
    reject empty, whitespace-only, and edge-whitespace labels, but NOT interior
    line terminators: ``"A\\nB"`` is a construction-valid label that, spliced
    verbatim into a cue, would introduce a line break -- and a line break in the
    SRT ``[<label>]: `` prefix or the WebVTT ``<v <label>>`` annotation can
    forge or split cue structure exactly like unsanitized cue text. Every line
    terminator is therefore collapsed to a single space (a label must never
    span lines).

    For the WebVTT voice-span annotation context (``escape_markup=True``) the
    label additionally escapes ``&`` -> ``&amp;``, ``<`` -> ``&lt;``, ``>`` ->
    ``&gt;`` (``&`` first, so the ``&`` it introduces is not re-escaped): a raw
    ``>`` inside the annotation would terminate the ``<v>`` tag early and leak
    the label remainder into the payload, and character references are legal in
    the W3C cue-span annotation. SRT has no entity mechanism (see
    :func:`_sanitize_cue_text`), so ``escape_markup`` is ``False`` there.

    The result is never empty: a valid label has non-whitespace at both ends
    (edge whitespace is construction-rejected), so collapsing interior
    terminators cannot strip it to ``""``.

    Args:
        label: A construction-valid speaker label (non-empty, no edge
            whitespace).
        escape_markup: Whether to escape WebVTT markup metacharacters
            (``& < >``). ``True`` for the VTT voice-tag annotation, ``False``
            for the SRT text prefix.

    Returns:
        The label, safe for its target context.
    """
    collapsed = label.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
    if escape_markup:
        collapsed = collapsed.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return collapsed


def _format_timestamp(seconds: float, *, millis_sep: str) -> str:
    """Format a time offset as ``HH:MM:SS<sep>mmm``.

    The renderer trusts the validated data model: :class:`~standard_asr.contract.results.Segment`
    / :class:`~standard_asr.contract.results.Word` guarantee a non-negative finite
    ``start`` / ``end``, so no negative offset can reach here. The
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

    The null rule distinguishes the two empty states: ``segments is None``
    means segmentation was *not requested / not applicable*, whereas
    ``segments == []`` means it *was requested but is empty* (e.g. confirmed
    silence). Only the former may fall back to a synthetic whole-text cue; an
    explicit ``[]`` yields zero cues, never a fabricated full-span cue.

    Segments are sorted by ``(start, channel, speaker)`` to enforce the
    top-level ordering invariant at the rendering boundary, so out-of-order
    input still produces correctly ordered subtitles. ``channel`` may be
    ``None``; it sorts before any explicit channel index (which the data model
    constrains to ``>= 0``). ``speaker`` is the final tie-break; ``None`` sorts
    before any real label (mapped to ``""``, which no valid label can be --
    empty labels are construction-rejected).

    A result carrying the reducer's
    :data:`~standard_asr.contract.results.DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE`
    diagnostic declares that some retained segment's ``start``/``end`` are
    ``0.0`` **placeholders** (the engine emitted no timestamps;
    :class:`~standard_asr.contract.results.Segment` requires finite times).
    That DIAGNOSTIC -- never the span values -- is the signal this function
    keys on: sniffing ``0.0`` spans cannot distinguish a placeholder from a
    genuine zero-length segment at ``t=0`` (e.g. an event with ``start=0.0``
    and no ``end``), and misreading real timing as placeholder would fabricate
    cue timing silently. When the diagnostic is present the whole segment
    timeline is untrustworthy for subtitles -- rendering it per-segment would
    emit zero-duration cues players silently drop and, in the mixed shape,
    reorder text against ``result.text`` -- so the result is routed to the same
    synthetic whole-text fallback as ``segments is None``, refusing to
    fabricate a partial timeline. Genuine zero-span segments in a
    diagnostic-free result render faithfully, one cue each.

    Args:
        result: The transcription result.

    Returns:
        The segments to render, ordered by ``(start, channel, speaker)``. For
        ``segments == []`` this is empty. When ``segments is None`` -- or the
        result carries the ``segment_timestamps_unavailable`` diagnostic (its
        timeline holds placeholders, e.g. a reduced timestamp-less stream) --
        and ``text`` is non-empty, a single synthetic segment spanning
        ``[0, duration]`` with the full text is returned -- or ``[0, 3 s]``
        when ``duration`` is unknown, because players silently drop
        zero-duration cues; when ``text`` is empty too, no cues are produced.
    """
    # ``segments == []`` (requested but empty) takes this branch too (an empty
    # reduce emits no timestamp diagnostic) and sorts to zero cues, never
    # reaching the synthetic fallback below.
    if result.segments is not None and not _timing_unavailable(result):
        return sorted(
            result.segments,
            key=lambda s: (
                s.start,
                s.channel if s.channel is not None else -1,
                s.speaker if s.speaker is not None else "",
            ),
        )
    if not result.text:
        return []
    end = result.duration if result.duration is not None else _SYNTHETIC_CUE_FALLBACK_END
    return [Segment(start=0.0, end=end, text=result.text)]


def _timing_unavailable(result: TranscriptionResult) -> bool:
    """Return whether the result declares placeholder segment timestamps.

    Args:
        result: The transcription result.

    Returns:
        ``True`` when the result carries the reducer's
        ``segment_timestamps_unavailable`` diagnostic -- the authoritative
        signal that segment ``start``/``end`` values include ``0.0``
        placeholders rather than real timing. Span values are deliberately not
        inspected (a genuine zero-length segment at ``t=0`` is
        indistinguishable from a placeholder by value).
    """
    return any(d.code == DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE for d in result.diagnostics)


def to_srt(result: TranscriptionResult, *, include_speakers: bool = False) -> str:
    """Render a transcription result as SRT.

    Cue text is sanitized so it cannot forge cue structure (line terminators
    normalized, interior blank-line runs collapsed). Unlike :func:`to_vtt`, SRT
    has **no character-reference mechanism**, so ``&`` and angle brackets in
    transcript text are emitted verbatim: an engine-leaked ``<unk>`` token or
    ``<i>`` is passed through as-is. Most SRT players render angle-bracket text
    literally, but some interpret a subset of HTML-like tags; if a downstream
    consumer must neutralize tags, do so on the transcript text before
    rendering. (WebVTT, which mandates escaping, is handled by :func:`to_vtt`.)

    Speaker rendering: SRT has no speaker syntax, so opting in
    mutates the cue text itself -- each labelled cue is prefixed with
    ``[<label>]: ``. The default is ``False`` on text-purity grounds (the
    renderer is a projection of the transcript; injecting labels uninvited
    would surprise every consumer that treats the cue text as speech), not for
    backward compatibility. Cues whose ``speaker`` is ``None`` are rendered
    unchanged; empty-text segments are skipped even when labelled (a label
    with no payload is not a cue).

    Segment fallback: when ``result.segments is None`` (segmentation
    not requested/applicable) -- or when the result carries the reducer's
    ``segment_timestamps_unavailable`` diagnostic, the authoritative signal
    that segment spans include ``0.0`` placeholders (a timestamp-less or
    partially timestamp-less streaming reduce) -- but ``result.text`` is
    non-empty, a single cue spanning the whole text is synthesized --
    ``[0, duration]``, or ``[0, 3 s]`` when ``duration`` is unknown (players
    silently drop zero-duration cues, so rendering placeholder spans
    per-segment would display nothing and, in the mixed shape, reorder text
    against ``result.text``; the renderer refuses to fabricate a partial
    timeline). Genuine zero-span segments in a diagnostic-free result render
    faithfully. ``segments == []`` (requested but empty, e.g. silence) yields
    no cues. The synthetic cue carries no speaker label -- a whole-text cue
    has no single attributable speaker -- so ``include_speakers`` has no
    effect on it. Pass a segmented result for time-accurate (and
    speaker-attributed) subtitles.

    Args:
        result: The transcription result to render.
        include_speakers: Render ``Segment.speaker`` labels as ``[<label>]: ``
            cue-text prefixes. Default ``False`` (pure transcript text).

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
            # because they are only advanced for emitted cues. A speaker label
            # never resurrects a skipped cue: no payload means no cue.
            continue
        if include_speakers and segment.speaker is not None:
            # Prefix AFTER text sanitization: the prefix must not be subject to
            # the blank-line collapse, and the label is sanitized for its own
            # context (interior newlines collapsed so it cannot forge cue
            # structure; no markup escaping -- SRT has no entity mechanism).
            label = _sanitize_speaker_label(segment.speaker, escape_markup=False)
            text = f"[{label}]: {text}"
        start = _format_timestamp(segment.start, millis_sep=",")
        end = _format_timestamp(segment.end, millis_sep=",")
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
        index += 1
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(result: TranscriptionResult, *, include_speakers: bool = False) -> str:
    """Render a transcription result as WebVTT.

    Cue text is escaped per the W3C WebVTT cue-text grammar (``&`` -> ``&amp;``,
    ``<`` -> ``&lt;``, ``>`` -> ``&gt;``) so payload text -- including
    engine-leaked ``<unk>`` / ``<|...|>`` tokens or "AT&T" -- is shown verbatim
    instead of being silently dropped by the browser's cue-span tokenizer. Cue
    structure is also protected (line terminators normalized, blank-line runs
    collapsed, ``-->`` neutralized by the ``>`` escape).

    Speaker rendering: opting in wraps the whole cue body of each
    labelled cue in WebVTT's native voice span, ``<v <label>>``. The default is
    ``False`` on text-purity grounds (the renderer is a projection of the
    transcript), not for backward compatibility. Cues whose ``speaker`` is
    ``None`` are rendered unchanged; empty-text segments are skipped even when
    labelled.

    Segment fallback: when ``result.segments is None`` (segmentation
    not requested/applicable) -- or when the result carries the reducer's
    ``segment_timestamps_unavailable`` diagnostic, the authoritative signal
    that segment spans include ``0.0`` placeholders (a timestamp-less or
    partially timestamp-less streaming reduce) -- but ``result.text`` is
    non-empty, a single cue spanning the whole text is synthesized --
    ``[0, duration]``, or ``[0, 3 s]`` when ``duration`` is unknown (players
    silently drop zero-duration cues, so rendering placeholder spans
    per-segment would display nothing and, in the mixed shape, reorder text
    against ``result.text``; the renderer refuses to fabricate a partial
    timeline). Genuine zero-span segments in a diagnostic-free result render
    faithfully. ``segments == []`` (requested but empty, e.g. silence) yields
    no cues. The synthetic cue carries no speaker label -- a whole-text cue
    has no single attributable speaker -- so ``include_speakers`` has no
    effect on it. Pass a segmented result for time-accurate (and
    speaker-attributed) subtitles.

    Args:
        result: The transcription result to render.
        include_speakers: Render ``Segment.speaker`` labels as ``<v <label>>``
            voice spans wrapping the cue body. Default ``False`` (pure
            transcript text).

    Returns:
        The WebVTT document as a string.
    """
    blocks: list[str] = ["WEBVTT"]
    for segment in _cues(result):
        text = _sanitize_cue_text(segment.text, escape_markup=True)
        if not text:
            # A WebVTT cue with no payload line is malformed; skip empty /
            # whitespace-only segments rather than emit a payload-less block.
            # A speaker label never resurrects a skipped cue.
            continue
        if include_speakers and segment.speaker is not None:
            # Inject the voice tag AFTER _sanitize_cue_text escaped the payload,
            # so the tag itself is not escaped away; the label is separately
            # sanitized for the annotation context (newlines collapsed; & < >
            # escaped -- a raw '>' would terminate the tag early). An unclosed
            # <v> span legally runs to the end of the cue payload per WebVTT,
            # so a single opening tag attributes a multi-line body without
            # per-line closing-tag ambiguity.
            label = _sanitize_speaker_label(segment.speaker, escape_markup=True)
            text = f"<v {label}>{text}"
        start = _format_timestamp(segment.start, millis_sep=".")
        end = _format_timestamp(segment.end, millis_sep=".")
        blocks.append(f"{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


__all__ = ["to_srt", "to_vtt"]
