# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Subtitle renderers for transcription results (SRT / VTT).

The core library renders the constant :class:`~standard_asr.contract.results.TranscriptionResult`
into SRT and VTT, so every compliant engine gets subtitle output for free
(spec, section "Transcription Result"). This replaces the old
``response_format`` option: rendering is a post-hoc transformation, not a request
parameter. Provider-rendered high-fidelity formats remain available only via
``result.extra["provider_formats"]``. Speaker labels are rendered only on
explicit opt-in (``include_speakers=True``): SRT prefixes the cue
text with ``[<label>]: ``, WebVTT wraps the cue body in a ``<v <label>>`` span.

Timing policy: a subtitle cue is an interval claim that must survive the
output's millisecond grid to be seen at all. A segment is UNRENDERABLE when
it lacks a measured span
(:attr:`~standard_asr.contract.results.Segment.timestamp_status`
``"start_only"`` / ``"unavailable"``) or when its measured span quantizes to
zero milliseconds (players silently drop a ``T --> T`` cue -- the render
would report success while the text never appears). Neither dropping text
nor fabricating timing is the renderer's to choose silently: the default
(``on_unrenderable="error"``) raises
:class:`~standard_asr.contract.exceptions.SubtitleRenderingError`, and the
caller picks the loss explicitly (``"omit"`` / ``"collapse"``). The nullable
``start``/``end`` values and the output grid are the only signals consulted
-- no diagnostics, no markers, no value-sniffing.
"""

from __future__ import annotations

import re
from typing import Iterable, Literal, NamedTuple

from standard_asr.contract.exceptions import SubtitleRenderingError
from standard_asr.contract.results import Segment, TranscriptionResult

#: The renderers' policy for segments that cannot render as VISIBLE cues:
#: no measured span, or a measured span that quantizes to zero milliseconds
#: on the output grid (players silently drop zero-duration cues). ``"error"``
#: (the default) raises; ``"omit"`` keeps only the renderable cues (dropping
#: the other segments' text from the file); ``"collapse"`` renders one
#: whole-text cue with no per-segment timeline.
UnrenderablePolicy = Literal["error", "omit", "collapse"]

#: Matches runs of two-or-more newlines (optionally with intervening blank
#: whitespace), i.e. the blank-line cue separator. Transcript text containing
#: such a run could otherwise forge or split cue blocks (cues are blank-line
#: delimited in both SRT and WebVTT). Line terminators are normalized to ``\n``
#: by :func:`_sanitize_cue_text` *before* this runs, so matching ``\n`` alone is
#: sufficient (a lone ``\r`` -- a line terminator in both WebVTT and many SRT
#: parsers -- can no longer slip past as an unrecognized newline form).
_BLANK_LINE_RUN = re.compile(r"(?:\n[ \t]*){2,}")

#: End time (seconds) of the synthetic whole-text cue when ``duration`` is
#: unknown or quantizes to zero on the output millisecond grid. Players
#: (ffmpeg, VLC, browser WebVTT) silently drop zero-duration cues, so the
#: fallback cue MUST have a non-zero span to display at all; 3 s is long
#: enough to be visible and short enough to read as synthetic.
_SYNTHETIC_CUE_FALLBACK_END = 3.0


class _TimedCue(NamedTuple):
    """A renderable cue: a grid-surviving measured span plus its source segment.

    Built only where renderability has been established
    (:func:`_renderable`), so the renderers consume real ``float`` bounds --
    no ``None`` narrowing, no dead defensive branches, and no way to feed an
    unrenderable segment into a timestamp formatter by accident.
    """

    start: float
    end: float
    segment: Segment


def _sanitize_cue_text(text: str, *, escape_markup: bool) -> str:
    r"""Sanitize segment text so it cannot forge or break cue structure.

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

    Line terminators are normalized to ``\n`` first so a lone ``\r`` -- a
    valid line terminator in WebVTT and many SRT parsers -- cannot slip past the
    blank-line collapse and forge a cue via ``\r\r``.

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
    r"""Sanitize a speaker label for interpolation into a cue block.

    The model validators (:func:`~standard_asr.contract.results.validate_speaker_label`)
    reject empty, whitespace-only, and edge-whitespace labels, but NOT interior
    line terminators: ``"A\nB"`` is a construction-valid label that, spliced
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


def _to_millis(seconds: float) -> int:
    """Quantize a time offset to the subtitle output's millisecond grid.

    THE single quantization rule: :func:`_format_timestamp` renders exactly
    this value, so any visibility decision (is a span non-zero AFTER
    formatting?) must consult the same function -- comparing raw float
    seconds instead re-created the invisible-cue bug for sub-millisecond
    spans (``duration=0.0005`` passes ``> 0`` but formats to ``00:00:00,000``).

    Args:
        seconds: Time offset in seconds (non-negative, finite).

    Returns:
        The offset in whole milliseconds (banker's rounding, as ``round``).
    """
    return int(round(seconds * 1000))


def _has_visible_payload(text: str) -> bool:
    """Report whether a segment's text would produce any cue payload at all.

    Target-independent by construction: :func:`_sanitize_cue_text` differs
    between SRT and WebVTT only in markup escaping, which can never turn
    non-blank text blank, so the emptiness question has one answer for both
    renderers -- and it is the same one they already apply when they skip a
    payload-less cue.

    Args:
        text: Raw segment text.

    Returns:
        ``True`` when the text survives normalization as something visible.
    """
    return bool(text.replace("\r\n", "\n").replace("\r", "\n").strip())


def _renderable(segment: Segment) -> bool:
    """Report whether a segment can render as a VISIBLE subtitle cue.

    Two requirements, both decided here so no caller re-derives them:

    * a measured span (``start`` and ``end`` both present -- an interval
      claim needs an interval);
    * a span that survives the output grid: ``_to_millis(end) >
      _to_millis(start)``. A zero-or-sub-millisecond span formats to
      ``T --> T``, which players (ffmpeg, VLC, browser WebVTT) silently
      drop -- the renderer would report success while the text never
      appears, the exact silent loss the policy parameter exists to make
      explicit. Renderability is a RENDERER property (it depends on the
      output quantum), distinct from the model's ``timestamp_status``
      (which reports what was measured).

    Args:
        segment: The segment to test.

    Returns:
        ``True`` if the segment's cue would be visible in the output.
    """
    return (
        segment.start is not None
        and segment.end is not None
        and _to_millis(segment.end) > _to_millis(segment.start)
    )


def _format_timestamp(seconds: float, *, millis_sep: str) -> str:
    """Format a time offset as ``HH:MM:SS<sep>mmm``.

    The renderer trusts the validated data model: :class:`~standard_asr.contract.results.Segment`
    / :class:`~standard_asr.contract.results.Word` guarantee a non-negative finite
    ``start`` / ``end`` wherever one is measured, so no negative offset can
    reach here. The
    renderer therefore does NOT clamp negatives -- clamping would silently mask
    an upstream timestamp bug (a wrong result), and the model already rejects one
    loudly at construction.

    Args:
        seconds: Time offset in seconds (non-negative, finite).
        millis_sep: Separator before milliseconds (``","`` SRT, ``"."`` VTT).

    Returns:
        The formatted timestamp string.
    """
    hours, rem = divmod(_to_millis(seconds), 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_sep}{millis:03d}"


def _sorted_timed(segments: Iterable[Segment]) -> list[_TimedCue]:
    """Project measured segments into time-ordered renderable cues.

    Args:
        segments: Segments to project; every entry MUST be renderable
            (callers establish that via :func:`_renderable` -- the
            comprehension's guard is the type narrowing, not a filter that
            silently drops data).

    Returns:
        The measured cues sorted by ``(start, channel, speaker)`` -- ``channel``
        ``None`` first, ``speaker`` ``None`` first (mapped to ``""``, which no
        valid label can be).
    """
    timed = [
        _TimedCue(segment.start, segment.end, segment)
        for segment in segments
        if segment.start is not None and segment.end is not None
    ]
    return sorted(
        timed,
        key=lambda cue: (
            cue.start,
            cue.segment.channel if cue.segment.channel is not None else -1,
            cue.segment.speaker if cue.segment.speaker is not None else "",
        ),
    )


def _whole_text_cue(result: TranscriptionResult, *, text: str | None = None) -> list[_TimedCue]:
    """Build the single synthetic whole-text cue (or none for no visible text).

    Args:
        result: The transcription result (the cue span's source).
        text: The text the cue carries; defaults to ``result.text`` (the
            ``segments is None`` fallback shape). The ``"collapse"`` policy
            passes the text it owes the caller instead.

    Returns:
        One cue spanning ``[0, duration]`` (or ``[0, 3 s]`` when ``duration``
        is unknown or quantizes to zero milliseconds -- players silently drop
        zero-duration cues) carrying the text; empty when no visible payload
        would survive rendering (the renderers skip a payload-less cue, so
        emptiness is decided by the same rule, up front).
    """
    carried = result.text if text is None else text
    if not _has_visible_payload(carried):
        return []
    # A zero-duration synthetic cue is exactly the invisible artifact
    # (players silently drop it) this fallback's non-zero span exists to
    # avoid. "Usable" is decided on the OUTPUT grid (_to_millis -- the same
    # quantization _format_timestamp renders), not on the raw float: a
    # model-legal sub-millisecond duration (0.0005) is positive as a float
    # yet formats to 00:00:00,000 --> 00:00:00,000 all the same. A non-empty
    # transcript always gets a span that survives formatting: the real
    # duration when it quantizes non-zero, else the fallback.
    duration = result.duration
    end = (
        duration
        if duration is not None and _to_millis(duration) > 0
        else _SYNTHETIC_CUE_FALLBACK_END
    )
    return [_TimedCue(0.0, end, Segment(start=0.0, end=end, text=carried))]


def _cues(result: TranscriptionResult, *, on_unrenderable: UnrenderablePolicy) -> list[_TimedCue]:
    """Select the cues to render under the caller's unrenderable-cue policy.

    The null rule distinguishes the two empty states: ``segments is None``
    means segmentation was *not requested / not applicable* -- the synthetic
    whole-text fallback applies -- whereas ``segments == []`` means it *was
    requested but is empty* (e.g. confirmed silence): zero cues,
    unconditionally, never a fabricated full-span cue.

    Segments with no VISIBLE PAYLOAD are dropped first
    (:func:`_has_visible_payload`) and never reach the policy: both renderers
    skip a payload-less segment anyway, so an empty one produces no cue
    whether or not it was measured -- there is no text to lose and no timing
    to fabricate, which is the whole premise of the policy. (Zero
    payload-bearing segments therefore render zero cues, never the
    ``segments is None`` whole-text fallback: segmentation ran and reported
    nothing visible.)

    Renderability is then read from the values alone via :func:`_renderable`
    (a measured span that survives the output millisecond grid); result
    diagnostics are deliberately NOT consulted -- the model is the single
    source of truth, so a stale or missing diagnostic can neither drop real
    timing nor resurrect unmeasured timing. When every segment is renderable
    the policy is inert and the segments render faithfully. Otherwise the
    policy decides, because every remaining option loses something a caller
    must knowingly accept (fabricating a wider span is not offered -- that
    would be unauthorized timing):

    * ``"error"`` (default): raise
      :class:`~standard_asr.contract.exceptions.SubtitleRenderingError` naming
      the counts -- no silent loss, the caller chooses.
    * ``"omit"``: render only the renderable segments; the other segments'
      text is absent from the file (possibly ALL of it -- zero cues). A
      start-only segment counts as unrenderable: with no ``end``, any
      placement fabricates an interval. A zero/sub-millisecond measured
      span counts too: its cue would format as ``T --> T`` and be silently
      dropped by players.
    * ``"collapse"``: one synthetic whole-text cue (the ``segments is None``
      fallback shape) -- every text survives, the per-segment timeline does
      not. The cue carries ``result.text`` when it is visibly non-empty,
      else the payload segments' texts in model order: "every text
      survives" is a guarantee, not a hope that the engine mirrored its
      segments into ``text``.

    Args:
        result: The transcription result.
        on_unrenderable: The caller's policy for unrenderable segments.

    Returns:
        The cues to render, time-ordered.

    Raises:
        ValueError: If ``on_unrenderable`` is not one of the three
            policies. The ``Literal`` type is not enforced at runtime, and an
            unrecognized value falling through to a policy arm would be
            exactly the implicit policy selection the parameter exists to
            eliminate (``"omti"`` silently collapsing a timeline).
        SubtitleRenderingError: Under the default ``"error"`` policy, when any
            segment cannot render as a visible cue.
    """
    if on_unrenderable not in ("error", "omit", "collapse"):
        raise ValueError(
            f"on_unrenderable must be 'error', 'omit', or 'collapse'; got "
            f"{on_unrenderable!r}. An unrecognized policy must fail loudly -- "
            "silently picking one would reintroduce the implicit data-loss choice "
            "this parameter exists to eliminate."
        )
    if result.segments == []:
        return []
    if result.segments is not None:
        # Segments with no visible payload are dropped BEFORE renderability is
        # judged. Both renderers already skip them (a cue with no payload line
        # is malformed), so an empty segment produces no cue whether or not it
        # was measured -- there is no text to lose and no timing to fabricate,
        # which is the entire premise of the policy. Judging them first made
        # the same empty segment silently skipped when it carried timestamps
        # and a hard SubtitleRenderingError when it did not.
        payload = [s for s in result.segments if _has_visible_payload(s.text)]
        if not payload:
            return []
        unrenderable = [s for s in payload if not _renderable(s)]
        if not unrenderable:
            return _sorted_timed(payload)
        if on_unrenderable == "error":
            raise SubtitleRenderingError(
                f"{len(unrenderable)} of {len(payload)} segments cannot "
                "render as visible subtitle cues. A segment is unrenderable when "
                "its span is unmeasured (start and/or end is None), or when it "
                "quantizes to zero milliseconds on the output grid (players "
                "silently drop a 'T --> T' cue). Rendering "
                "anyway would drop text or fabricate timing. Choose explicitly: "
                "on_unrenderable='omit' keeps only the renderable cues (their "
                "text alone reaches the file), or 'collapse' renders one "
                "whole-text cue with no per-segment timeline.",
                unrenderable=len(unrenderable),
                total=len(payload),
            )
        if on_unrenderable == "omit":
            return _sorted_timed([s for s in payload if _renderable(s)])
        # "collapse": the caller trades the timeline for completeness -- so
        # completeness is owed. ``result.text`` is the engine's canonical
        # whole-transcript rendering and carries the cue when it is visibly
        # non-empty; but an engine that populated only ``segments`` leaves
        # it blank, and collapsing to it rendered an empty file while every
        # word sat in the payload segments -- the exact silent total loss
        # the caller's explicit choice was scoped to exclude. The cue then
        # carries the payload segments' texts in model order (the
        # transcript's own reading order) instead.
        return _whole_text_cue(
            result,
            text=(
                result.text
                if _has_visible_payload(result.text)
                else " ".join(s.text.strip() for s in payload)
            ),
        )
    return _whole_text_cue(result)


def to_srt(
    result: TranscriptionResult,
    *,
    include_speakers: bool = False,
    on_unrenderable: UnrenderablePolicy = "error",
) -> str:
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
    mutates the cue text itself -- each labeled cue is prefixed with
    ``[<label>]: ``. The default is ``False`` on text-purity grounds (the
    renderer is a projection of the transcript; injecting labels uninvited
    would surprise every consumer that treats the cue text as speech), not for
    backward compatibility. Cues whose ``speaker`` is ``None`` are rendered
    unchanged; empty-text segments are skipped even when labeled (a label
    with no payload is not a cue).

    Unrenderable segments: a segment without a measured span, or whose
    measured span quantizes to zero milliseconds on the output grid (its
    ``T --> T`` cue would be silently dropped by players), cannot render as
    a visible cue. By default that raises
    :class:`~standard_asr.contract.exceptions.SubtitleRenderingError` -- the
    render never silently drops, hides, or fabricates -- and the caller opts
    into a loss explicitly: ``on_unrenderable="omit"`` renders only the
    renderable cues (the other segments' text is absent from the file),
    ``"collapse"`` renders one synthetic whole-text cue ``[0, duration]``
    (or ``[0, 3 s]`` when ``duration`` is unknown or quantizes to zero
    milliseconds; the fallback exists for the same player behavior).
    ``result.segments is None`` (segmentation not requested/applicable) uses
    the same whole-text fallback under every policy; ``segments == []``
    (requested but empty, e.g. silence) ALWAYS yields no cues. The synthetic
    cue carries no speaker label -- a whole-text cue has no single
    attributable speaker -- so ``include_speakers`` has no effect on it.
    Pass a fully renderable result for time-accurate (and
    speaker-attributed) subtitles.

    Args:
        result: The transcription result to render.
        include_speakers: Render ``Segment.speaker`` labels as ``[<label>]: ``
            cue-text prefixes. Default ``False`` (pure transcript text).
        on_unrenderable: Policy for segments that cannot render as visible
            cues (unmeasured span, or a span that quantizes to zero
            milliseconds): ``"error"`` (default) raises; ``"omit"`` keeps
            only renderable cues; ``"collapse"`` renders one whole-text cue.

    Returns:
        The SRT document as a string.

    Raises:
        SubtitleRenderingError: Under the default ``"error"`` policy, when any
            segment cannot render as a visible cue.
        ValueError: If ``on_unrenderable`` is not one of the three policies
            (see :func:`_cues`).
    """
    blocks: list[str] = []
    index = 1
    for cue in _cues(result, on_unrenderable=on_unrenderable):
        text = _sanitize_cue_text(cue.segment.text, escape_markup=False)
        if not text:  # pragma: no cover - _cues guarantees visible payload
            # Unreachable while _cues holds its invariant (every cue it
            # returns carries visible payload -- _has_visible_payload is
            # decided on the same normalization, and escaping never blanks
            # non-blank text). Kept as the loop's own guard: a cue with no
            # payload (an index + timing line followed by a blank line) is
            # malformed SRT that strict parsers reject, so a future cue
            # source that broke the invariant must skip, never emit.
            # Indices stay contiguous (advanced only for emitted cues); a
            # speaker label never resurrects a skipped cue.
            continue
        if include_speakers and cue.segment.speaker is not None:
            # Prefix AFTER text sanitization: the prefix must not be subject to
            # the blank-line collapse, and the label is sanitized for its own
            # context (interior newlines collapsed so it cannot forge cue
            # structure; no markup escaping -- SRT has no entity mechanism).
            label = _sanitize_speaker_label(cue.segment.speaker, escape_markup=False)
            text = f"[{label}]: {text}"
        start = _format_timestamp(cue.start, millis_sep=",")
        end = _format_timestamp(cue.end, millis_sep=",")
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
        index += 1
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(
    result: TranscriptionResult,
    *,
    include_speakers: bool = False,
    on_unrenderable: UnrenderablePolicy = "error",
) -> str:
    """Render a transcription result as WebVTT.

    Cue text is escaped per the W3C WebVTT cue-text grammar (``&`` -> ``&amp;``,
    ``<`` -> ``&lt;``, ``>`` -> ``&gt;``) so payload text -- including
    engine-leaked ``<unk>`` / ``<|...|>`` tokens or "AT&T" -- is shown verbatim
    instead of being silently dropped by the browser's cue-span tokenizer. Cue
    structure is also protected (line terminators normalized, blank-line runs
    collapsed, ``-->`` neutralized by the ``>`` escape).

    Speaker rendering: opting in wraps the whole cue body of each
    labeled cue in WebVTT's native voice span, ``<v <label>>``. The default is
    ``False`` on text-purity grounds (the renderer is a projection of the
    transcript), not for backward compatibility. Cues whose ``speaker`` is
    ``None`` are rendered unchanged; empty-text segments are skipped even when
    labeled.

    Unrenderable segments: identical policy contract to :func:`to_srt` -- a
    segment without a measured span, or whose span quantizes to zero
    milliseconds on the output grid, raises by default
    (:class:`~standard_asr.contract.exceptions.SubtitleRenderingError`), and
    the caller opts into ``"omit"`` (renderable cues only) or ``"collapse"``
    (one synthetic whole-text cue) explicitly. ``segments is None`` uses the
    whole-text fallback under every policy; ``segments == []`` always yields
    zero cues; the synthetic cue carries no speaker label.

    Args:
        result: The transcription result to render.
        include_speakers: Render ``Segment.speaker`` labels as ``<v <label>>``
            voice spans wrapping the cue body. Default ``False`` (pure
            transcript text).
        on_unrenderable: Policy for segments that cannot render as visible
            cues (unmeasured span, or a span that quantizes to zero
            milliseconds): ``"error"`` (default) raises; ``"omit"`` keeps
            only renderable cues; ``"collapse"`` renders one whole-text cue.

    Returns:
        The WebVTT document as a string.

    Raises:
        SubtitleRenderingError: Under the default ``"error"`` policy, when any
            segment cannot render as a visible cue.
        ValueError: If ``on_unrenderable`` is not one of the three policies
            (see :func:`_cues`).
    """
    blocks: list[str] = ["WEBVTT"]
    for cue in _cues(result, on_unrenderable=on_unrenderable):
        text = _sanitize_cue_text(cue.segment.text, escape_markup=True)
        if not text:  # pragma: no cover - _cues guarantees visible payload
            # Unreachable while _cues holds its invariant (see to_srt's
            # twin guard). Kept as the loop's own guard: a WebVTT cue with
            # no payload line is malformed, so a future cue source that
            # broke the invariant must skip, never emit a payload-less
            # block. A speaker label never resurrects a skipped cue.
            continue
        if include_speakers and cue.segment.speaker is not None:
            # Inject the voice tag AFTER _sanitize_cue_text escaped the payload,
            # so the tag itself is not escaped away; the label is separately
            # sanitized for the annotation context (newlines collapsed; & < >
            # escaped -- a raw '>' would terminate the tag early). An unclosed
            # <v> span legally runs to the end of the cue payload per WebVTT,
            # so a single opening tag attributes a multi-line body without
            # per-line closing-tag ambiguity.
            label = _sanitize_speaker_label(cue.segment.speaker, escape_markup=True)
            text = f"<v {label}>{text}"
        start = _format_timestamp(cue.start, millis_sep=".")
        end = _format_timestamp(cue.end, millis_sep=".")
        blocks.append(f"{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


__all__ = ["UnrenderablePolicy", "to_srt", "to_vtt"]
