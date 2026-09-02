# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for transcription result models and renderers."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from pydantic import BaseModel, ValidationError

import standard_asr as standard_asr_package
from standard_asr.contract import results as results_module
from standard_asr.contract.exceptions import SubtitleRenderingError
from standard_asr.contract.results import (
    DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE,
    ChannelResult,
    Diagnostic,
    Segment,
    TranscriptionResult,
    Word,
    synthesize_segment_speaker,
)
from standard_asr.renderers import to_srt, to_vtt
from standard_asr.runtime import streaming as streaming_module
from standard_asr.runtime.streaming import StreamReducer, TranscriptionEvent


def test_minimal_result() -> None:
    result = TranscriptionResult(text="hello")
    assert result.text == "hello"
    assert result.detected_language is None
    assert result.diagnostics == []


def test_no_blanket_metadata_pocket_on_the_result() -> None:
    """The blanket `metadata: dict[str, Any]` field was removed: a "standardized
    engine-agnostic metadata" dict with no standardized keys, no writer, and no
    reader is the same unstructured-data disease the spec dropped from
    Properties and Capabilities. Standardized result data earns a real field;
    everything engine-specific goes in `extra`. The model is extra="forbid",
    so a caller still passing metadata= is rejected, never silently absorbed.
    (model_validate keeps the removed key a runtime rejection, not a static
    type error.)
    """
    assert "metadata" not in TranscriptionResult.model_fields
    with pytest.raises(ValidationError):
        TranscriptionResult.model_validate({"text": "x", "metadata": {"cost": 1}})
    # The engine-specific pocket that DID survive still accepts it.
    assert TranscriptionResult(text="x", extra={"cost": 1}).extra == {"cost": 1}


def test_result_rejects_negative_duration() -> None:
    with pytest.raises(ValueError):
        TranscriptionResult(text="x", duration=-1.0)


def test_result_rejects_nonfinite_duration() -> None:
    with pytest.raises(ValueError):
        TranscriptionResult(text="x", duration=math.nan)
    with pytest.raises(ValueError):
        TranscriptionResult(text="x", duration=math.inf)


def test_result_accepts_zero_duration() -> None:
    assert TranscriptionResult(text="x", duration=0.0).duration == 0.0


def test_result_rejects_malformed_detected_language() -> None:
    # A native language name is not a BCP-47 tag; reject loudly, do not echo it.
    with pytest.raises(ValueError):
        TranscriptionResult(text="x", detected_language="English")


def test_result_rejects_auto_as_detected_language() -> None:
    # 'auto' is the detect-me directive, never a detection *result*.
    with pytest.raises(ValueError):
        TranscriptionResult(text="x", detected_language="auto")
    with pytest.raises(ValueError):
        TranscriptionResult(text="x", detected_language="AUTO")


def test_result_canonicalizes_detected_language() -> None:
    # A valid tag is accepted and normalized to canonical casing.
    result = TranscriptionResult(text="x", detected_language="zh-hans")
    assert result.detected_language == "zh-Hans"


def test_segment_and_word_models() -> None:
    word = Word(start=0.0, end=0.5, text="hi", probability=0.9)
    segment = Segment(start=0.0, end=1.0, text="hi", words=[word], channel=0)
    result = TranscriptionResult(text="hi", segments=[segment], words=[word])
    assert result.segments is not None
    assert result.segments[0].words is not None
    assert result.words is not None
    assert result.words[0].text == "hi"


def test_probability_bounds() -> None:
    with pytest.raises(ValueError):
        Word(start=0.0, end=0.1, text="x", probability=1.5)


# --------------------------------------------------------------------------- #
# Timestamp invariants: non-negative, finite, ordered floats.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", [Word, Segment])
def test_time_rejects_negative_start(model: type[Word | Segment]) -> None:
    with pytest.raises(ValueError):
        model(start=-0.1, end=1.0, text="x")


@pytest.mark.parametrize("model", [Word, Segment])
def test_time_rejects_negative_end(model: type[Word | Segment]) -> None:
    with pytest.raises(ValueError):
        model(start=0.0, end=-0.1, text="x")


@pytest.mark.parametrize("model", [Word, Segment])
def test_time_rejects_inverted_span(model: type[Word | Segment]) -> None:
    with pytest.raises(ValueError):
        model(start=1.0, end=0.5, text="x")


@pytest.mark.parametrize("model", [Word, Segment])
def test_time_rejects_nan(model: type[Word | Segment]) -> None:
    with pytest.raises(ValueError):
        model(start=math.nan, end=1.0, text="x")
    with pytest.raises(ValueError):
        model(start=0.0, end=math.nan, text="x")


@pytest.mark.parametrize("model", [Word, Segment])
def test_time_rejects_inf(model: type[Word | Segment]) -> None:
    with pytest.raises(ValueError):
        model(start=0.0, end=math.inf, text="x")
    with pytest.raises(ValueError):
        model(start=-math.inf, end=0.0, text="x")


@pytest.mark.parametrize("model", [Word, Segment])
def test_time_allows_zero_duration_span(model: type[Word | Segment]) -> None:
    # end == start (zero duration) is a valid span, not an inverted one.
    item = model(start=1.5, end=1.5, text="x")
    assert item.start == 1.5
    assert item.end == 1.5


def test_logprob_separate_from_probability() -> None:
    word = Word(start=0.0, end=0.1, text="x", probability=0.8, logprob=-0.2)
    assert word.probability == 0.8
    assert word.logprob == -0.2


def test_channels_field() -> None:
    chan = ChannelResult(channel=1, text="left")
    result = TranscriptionResult(text="left right", channels=[chan])
    assert result.channels is not None
    assert result.channels[0].channel == 1


def test_channel_segments_require_top_level_segments() -> None:
    # Ignoring `channels` must be lossless. A channel entry carrying
    # segments while the top level has none would make channel-agnostic
    # consumers (for example, the renderers) silently drop all per-channel timing, so
    # the shape is rejected at construction.
    chan = ChannelResult(channel=0, text="hi", segments=[Segment(start=0.0, end=1.0, text="hi")])
    with pytest.raises(ValueError, match="time-merged union"):
        TranscriptionResult(text="hi", channels=[chan])


def test_channel_words_require_top_level_words() -> None:
    # Same derivability invariant for the flattened word-level view.
    chan = ChannelResult(channel=0, text="hi", words=[Word(start=0.0, end=0.5, text="hi")])
    with pytest.raises(ValueError, match="time-merged union"):
        TranscriptionResult(text="hi", channels=[chan])


def test_channels_with_top_level_segments_and_words_construct() -> None:
    # The conformant shape (top level = time-merge of all channels) is
    # accepted; per-channel detail with a populated top level is the contract.
    word = Word(start=0.0, end=0.5, text="hi")
    seg = Segment(start=0.0, end=1.0, text="hi")
    chan = ChannelResult(channel=0, text="hi", segments=[seg], words=[word])
    result = TranscriptionResult(text="hi", segments=[seg], words=[word], channels=[chan])
    assert result.channels is not None
    assert result.segments == [seg]
    assert result.words == [word]


def test_duplicate_channel_index_rejected() -> None:
    # The standard defines `channels` as one ChannelResult per channel, so a
    # duplicate index is a semantically illegal shape -- a consumer keying a dict
    # by channel index would silently drop one entry. The model refuses it.
    with pytest.raises(ValueError, match="duplicate entries for one channel index"):
        TranscriptionResult(
            text="x",
            channels=[
                ChannelResult(channel=0, text="a"),
                ChannelResult(channel=0, text="b"),
            ],
        )


def test_distinct_channel_indices_accepted() -> None:
    # The legitimate multi-channel shape (distinct indices) still constructs.
    result = TranscriptionResult(
        text="a b",
        channels=[
            ChannelResult(channel=0, text="a"),
            ChannelResult(channel=1, text="b"),
        ],
    )
    assert result.channels is not None
    assert [c.channel for c in result.channels] == [0, 1]


def test_out_of_order_segments_accepted_at_construction() -> None:
    # The (start, channel, speaker) ordering is an ENGINE obligation, NOT
    # a construct-time invariant -- and the compliance suite does not check it
    # either. The streaming reducer legitimately keeps arrival order for
    # timestamp-less engines and sorts only by start, so a strict ordering
    # check (at construction or in the suite) would reject valid reduced
    # results. This test locks the deliberate non-enforcement so a future
    # change does not silently add a breaking validator (the renderers
    # re-sort defensively -- see test_srt_sorts_out_of_order_segments).
    out_of_order = [
        Segment(start=5.0, end=6.0, text="second"),
        Segment(start=0.0, end=1.0, text="first"),
    ]
    result = TranscriptionResult(text="x", segments=out_of_order)
    assert result.segments is not None
    assert [s.start for s in result.segments] == [5.0, 0.0]


def test_diagnostic_model() -> None:
    diag = Diagnostic(
        level="warning",
        code="audio_conversion",
        message="lossy",
        param="audio",
        provided="float32",
        effective="int16",
    )
    result = TranscriptionResult(text="hi", diagnostics=[diag])
    assert result.diagnostics[0].code == "audio_conversion"


def test_to_srt_from_segments() -> None:
    segs = [
        Segment(start=0.0, end=1.5, text="Hello"),
        Segment(start=1.5, end=3.25, text="world"),
    ]
    srt = to_srt(TranscriptionResult(text="Hello world", segments=segs))
    assert "1\n00:00:00,000 --> 00:00:01,500\nHello" in srt
    assert "2\n00:00:01,500 --> 00:00:03,250\nworld" in srt


def test_to_vtt_from_segments() -> None:
    segs = [Segment(start=0.0, end=1.0, text="Hi")]
    vtt = to_vtt(TranscriptionResult(text="Hi", segments=segs))
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.000\nHi" in vtt


def test_renderers_fallback_to_full_text() -> None:
    result = TranscriptionResult(text="No segments here", duration=2.0)
    srt = to_srt(result)
    assert "No segments here" in srt
    assert "00:00:00,000 --> 00:00:02,000" in srt
    vtt = to_vtt(result)
    assert "No segments here" in vtt


def test_to_srt_empty_text_no_duration() -> None:
    # segments=None + empty text: nothing to render, so no fabricated cue.
    result = TranscriptionResult(text="")
    assert to_srt(result) == ""
    assert to_vtt(result) == "WEBVTT\n"


def test_empty_segments_list_yields_no_cues() -> None:
    # segments=[] means segmentation ran and found nothing (for example, silence). Per
    # the null rule this must NOT fabricate a full-span cue from text.
    result = TranscriptionResult(text="some text", segments=[], duration=5.0)
    assert to_srt(result) == ""
    assert to_vtt(result) == "WEBVTT\n"


def test_none_segments_with_text_synthesizes_one_cue() -> None:
    # segments=None (not requested) + non-empty text: synthesize a single cue.
    result = TranscriptionResult(text="whole text", duration=2.0)
    srt = to_srt(result)
    assert "1\n00:00:00,000 --> 00:00:02,000\nwhole text" in srt
    # Exactly one cue.
    assert "2\n" not in srt


def test_synthetic_cue_without_duration_has_visible_span() -> None:
    # segments=None + unknown duration (for example, a reduced stream): the synthetic
    # cue must not be zero-duration -- ffmpeg / VLC / browser WebVTT silently
    # drop zero-duration cues, hiding the only transcript content. The
    # renderer falls back to a fixed 3 s span.
    result = TranscriptionResult(text="only text")
    srt = to_srt(result)
    assert "1\n00:00:00,000 --> 00:00:03,000\nonly text" in srt
    vtt = to_vtt(result)
    assert "00:00:00.000 --> 00:00:03.000\nonly text" in vtt


def test_zero_span_measured_segments_are_unrenderable_by_default() -> None:
    """A measured zero-length span cannot render as a VISIBLE cue.

    ``00:00:00,000 --> 00:00:00,000`` is a well-formed cue that players
    (ffmpeg, VLC, browser WebVTT) silently drop -- the earlier "render
    faithfully" contract produced a successful string whose text never
    appeared on screen, violating "renderers never silently drop text" one
    step downstream. Renderability is decided on the output grid; the
    default policy is loud, and the losses are explicit opt-ins.
    """
    segs = [
        Segment(start=0.0, end=0.0, text="hello"),
        Segment(start=0.0, end=0.0, text="world"),
    ]
    result = TranscriptionResult(text="hello world", segments=segs)
    assert result.diagnostics == []

    for render in (to_srt, to_vtt):
        with pytest.raises(SubtitleRenderingError) as excinfo:
            render(result)
        assert excinfo.value.unrenderable == 2
        assert excinfo.value.total == 2

    # "omit" drops the invisible cues explicitly (here: everything).
    assert to_srt(result, on_unrenderable="omit") == ""
    # "collapse" keeps every text in the one visible whole-text cue.
    assert to_srt(result, on_unrenderable="collapse") == (
        "1\n00:00:00,000 --> 00:00:03,000\nhello world\n"
    )


def test_zero_span_segments_with_known_duration_still_raise_by_default() -> None:
    """A known duration never silently rewrites reported segment timing.

    The renderer does not repair invisible spans with ``duration`` on its
    own -- that would be unauthorized timing fabrication. The duration is
    used only where the caller explicitly collapses the timeline.
    """
    segs = [
        Segment(start=0.0, end=0.0, text="a"),
        Segment(start=0.0, end=0.0, text="b"),
    ]
    result = TranscriptionResult(text="a b", segments=segs, duration=7.5)

    with pytest.raises(SubtitleRenderingError):
        to_srt(result)
    assert to_srt(result, on_unrenderable="collapse") == ("1\n00:00:00,000 --> 00:00:07,500\na b\n")


def test_renderable_empty_segment_text_renders_nothing() -> None:
    """A payload-less RENDERABLE segment yields no cue (the empty-payload rule).

    An empty ``text`` is no synthesis source either: nothing to show means
    no cue, not a fabricated one.
    """
    result = TranscriptionResult(text="", segments=[Segment(start=0.0, end=1.0, text="")])
    assert to_srt(result) == ""
    assert to_vtt(result) == "WEBVTT\n"


def test_zero_span_segment_text_cannot_silently_vanish() -> None:
    """A transcript living only in an invisible-span segment fails LOUDLY.

    Under the retired "render faithfully" contract this shape produced a
    successful file whose only cue no player displays -- the transcript
    silently vanished at playback. The default policy now refuses; with a
    grid-surviving span the segment renders normally.
    """
    result = TranscriptionResult(text="", segments=[Segment(start=0.0, end=0.0, text="hello")])
    with pytest.raises(SubtitleRenderingError):
        to_srt(result)

    renderable = TranscriptionResult(text="", segments=[Segment(start=0.0, end=1.0, text="hello")])
    assert to_srt(renderable) == "1\n00:00:00,000 --> 00:00:01,000\nhello\n"
    assert "hello" in to_vtt(renderable)


def test_renderable_segment_keeps_its_speaker_label() -> None:
    """A renderable segment renders as a real cue, so include_speakers
    still attributes it (the synthetic whole-text fallback drops labels --
    it has no single attributable speaker).
    """
    result = TranscriptionResult(
        text="x", segments=[Segment(start=0.0, end=1.0, text="hi", speaker="Alice")]
    )
    assert (
        to_srt(result, include_speakers=True) == "1\n00:00:00,000 --> 00:00:01,000\n[Alice]: hi\n"
    )
    assert "<v Alice>hi" in to_vtt(result, include_speakers=True)


def _unavailable_segment(text: str, **kwargs: object) -> Segment:
    """Build a segment with no measured timing, as the reducer stores them.

    Args:
        text: The segment text.
        **kwargs: Extra :class:`Segment` fields (for example, ``speaker``).

    Returns:
        A ``start=None, end=None`` segment (``timestamp_status="unavailable"``).
    """
    return Segment(start=None, end=None, text=text, **kwargs)  # pyright: ignore[reportArgumentType]


def test_segment_timing_shapes() -> None:
    """The legal timing shapes are pinned; the illegal one is unrepresentable.

    ``(float, float)`` is measured, ``(float, None)`` start-only,
    ``(None, None)`` unavailable; ``(None, float)`` -- an end without a start
    -- is rejected at construction, and ``timestamp_status`` derives from the
    values (no stored field, no side-channel marker to disagree with them).
    """
    measured = Segment(start=1.0, end=2.0, text="m")
    assert measured.timestamp_status == "measured"
    start_only = Segment(start=12.5, end=None, text="s")
    assert start_only.timestamp_status == "start_only"
    unavailable = Segment(start=None, end=None, text="u")
    assert unavailable.timestamp_status == "unavailable"
    with pytest.raises(ValidationError, match="end is set without a start"):
        Segment(start=None, end=2.0, text="x")
    # The measured-shape invariants still hold on the float branch.
    with pytest.raises(ValidationError, match="must be >= start"):
        Segment(start=2.0, end=1.0, text="x")


def test_segment_wire_schema_start_end_nullable() -> None:
    """Two-layer isomorphism: the nullable timing is visible in the JSON schema.

    A cross-language client discovers ``start``/``end`` as number-or-null from
    the schema itself -- the Python-layer shape change has a direct wire-layer
    expression (no reserved ``extra`` key to know about out-of-band).
    """
    schema = Segment.model_json_schema()
    for field_name in ("start", "end"):
        any_of = schema["properties"][field_name]["anyOf"]
        assert {"type": "null"} in any_of
        assert any(entry.get("type") == "number" for entry in any_of)
    # The retired reserved-marker surface is gone from the package root too.
    assert not hasattr(standard_asr_package, "SEGMENT_EXTRA_TIMESTAMP_PLACEHOLDER")
    assert "SEGMENT_EXTRA_TIMESTAMP_PLACEHOLDER" not in standard_asr_package.__all__


def test_missing_timing_raises_by_default() -> None:
    """The default policy is loud: unmeasured spans never render silently.

    A mixed result (one measured cue, one unmeasured segment) cannot be
    rendered without dropping text or fabricating timing; the renderer must
    not choose a loss for the caller. Both renderers raise the standard
    error, which carries the counts and is caller-fixable (a ValueError).
    """
    result = TranscriptionResult(
        text="real missing",
        segments=[Segment(start=1.0, end=2.0, text="real"), _unavailable_segment("missing")],
    )
    for render in (to_srt, to_vtt):
        with pytest.raises(SubtitleRenderingError) as excinfo:
            render(result)
        assert excinfo.value.unrenderable == 1
        assert excinfo.value.total == 2
        assert "on_unrenderable" in str(excinfo.value)
    assert issubclass(SubtitleRenderingError, ValueError)


def test_omit_policy_keeps_the_real_cue_and_drops_the_unmeasured_text() -> None:
    """Mixed shape under explicit "omit": the real 5-6 s cue survives with its
    true timing while the unmeasured segment's text is absent from the file
    -- a loss the caller knowingly accepted at the call site (the default
    would have raised). The full text remains in result.text.
    """
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "b", start=5.0, end=6.0))
    reducer.add(TranscriptionEvent.final("s2", "a"))
    result = reducer.result()

    assert result.text == "b a"
    assert [d.code for d in result.diagnostics] == [DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE]

    srt = to_srt(result, on_unrenderable="omit")
    assert srt == "1\n00:00:05,000 --> 00:00:06,000\nb\n"
    assert srt.count("-->") == 1
    # The unmeasured text is not rendered at a fabricated time...
    assert "\na\n" not in srt
    assert "00:00:00,000 --> 00:00:00,000" not in srt
    # ...and the whole transcript is NOT collapsed into one synthetic cue either.
    assert "b a" not in srt
    assert to_vtt(result, on_unrenderable="omit") == (
        "WEBVTT\n\n00:00:05.000 --> 00:00:06.000\nb\n"
    )


def test_omit_policy_with_zero_measured_cues_renders_an_empty_file() -> None:
    """ "omit" means omit: when nothing is measured, nothing reaches the file.
    The whole-text resurrection is exclusively "collapse"'s behavior -- an
    implicit fallback here would render text the caller asked to have
    dropped when unplaceable.
    """
    result = TranscriptionResult(text="hello", segments=[_unavailable_segment("hello")])
    assert to_srt(result, on_unrenderable="omit") == ""
    assert to_vtt(result, on_unrenderable="omit") == "WEBVTT\n"


def test_collapse_policy_renders_one_whole_text_cue() -> None:
    """ "collapse" trades the per-segment timeline for completeness: one
    synthetic cue carries the full text -- [0, 3 s] when duration is unknown
    (players silently drop zero-duration cues), [0, duration] when known.
    """
    segs = [_unavailable_segment("hello"), _unavailable_segment("world")]
    result = TranscriptionResult(text="hello world", segments=segs)

    srt = to_srt(result, on_unrenderable="collapse")
    assert srt == "1\n00:00:00,000 --> 00:00:03,000\nhello world\n"
    assert "2\n" not in srt
    vtt = to_vtt(result, on_unrenderable="collapse")
    assert "00:00:00.000 --> 00:00:03.000\nhello world" in vtt
    assert vtt.count("-->") == 1

    known = TranscriptionResult(text="hello world", segments=segs, duration=7.5)
    assert to_srt(known, on_unrenderable="collapse") == (
        "1\n00:00:00,000 --> 00:00:07,500\nhello world\n"
    )
    # Even when SOME segments are measured, "collapse" is what the caller
    # asked for: one cue, no partial timeline.
    mixed = TranscriptionResult(
        text="real missing",
        segments=[Segment(start=1.0, end=2.0, text="real"), _unavailable_segment("missing")],
        duration=9.0,
    )
    assert to_srt(mixed, on_unrenderable="collapse") == (
        "1\n00:00:00,000 --> 00:00:09,000\nreal missing\n"
    )


def test_collapse_carries_segment_texts_when_whole_text_is_blank() -> None:
    """ "collapse" owes completeness even when the engine left ``text`` blank.

    An engine that populated only ``segments`` (empty or whitespace-only
    ``result.text``) used to collapse into ``_whole_text_cue``'s empty
    return: a successful ``""`` / bare ``WEBVTT`` file while every word of
    the transcript sat in the segments -- silent total loss from the very
    policy whose contract is "every text survives, only the timeline does
    not". The collapsed cue must carry the payload segments' texts in model
    order instead.
    """
    segs = [_unavailable_segment("hello"), _unavailable_segment("world")]
    for blank in ("", " "):
        result = TranscriptionResult(text=blank, segments=segs, duration=7.5)
        assert to_srt(result, on_unrenderable="collapse") == (
            "1\n00:00:00,000 --> 00:00:07,500\nhello world\n"
        )
        vtt = to_vtt(result, on_unrenderable="collapse")
        assert "00:00:00.000 --> 00:00:07.500\nhello world" in vtt

    # A visibly non-empty result.text stays canonical: the engine's own
    # whole-transcript rendering wins over a re-join of the segments.
    canonical = TranscriptionResult(text="Hello, world.", segments=segs)
    assert "Hello, world." in to_srt(canonical, on_unrenderable="collapse")

    # Empty text with NO payload segments still renders nothing: there is
    # genuinely no text to carry, and nothing is fabricated.
    silent = TranscriptionResult(
        text="", segments=[Segment(start=None, end=None, text=" ")], duration=2.0
    )
    assert to_srt(silent, on_unrenderable="collapse") == ""
    assert to_vtt(silent, on_unrenderable="collapse") == "WEBVTT\n"


def test_one_unmeasured_never_discards_a_predominantly_real_timeline() -> None:
    """With three measured segments and one unmeasured, "omit" keeps the three
    real cues correctly timed and attributed; only the unplaceable segment's
    text drops out -- and the default policy would have raised rather than
    decide that silently.
    """
    segs = [
        Segment(start=0.0, end=1.0, text="one", speaker="Alice"),
        Segment(start=1.0, end=2.0, text="two", speaker="Bob"),
        Segment(start=2.0, end=3.0, text="three", speaker="Alice"),
        _unavailable_segment("four", speaker="Bob"),
    ]
    result = TranscriptionResult(text="one two three four", segments=segs)

    srt = to_srt(result, include_speakers=True, on_unrenderable="omit")
    assert srt == (
        "1\n00:00:00,000 --> 00:00:01,000\n[Alice]: one\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n[Bob]: two\n\n"
        "3\n00:00:02,000 --> 00:00:03,000\n[Alice]: three\n"
    )
    assert "four" not in srt
    vtt = to_vtt(result, include_speakers=True, on_unrenderable="omit")
    assert vtt.count("-->") == 3
    assert "four" not in vtt


def test_start_only_final_keeps_its_real_start_but_is_not_a_cue() -> None:
    """A start-only final stores its real onset verbatim -- and no fabricated end.

    The measurement survives on the model (``start=12.5, end=None``,
    ``timestamp_status="start_only"``) for API consumers; for cue purposes it
    counts as unrenderable -- with no end, any interval would be fabricated
    (and the retired design's ``end=start`` zero-duration cues were invisible
    in every player, which is why grid-zero measured spans are unrenderable
    too).
    """
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "hello", start=12.5))
    result = reducer.result()

    assert [d.code for d in result.diagnostics] == [DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE]
    assert result.segments is not None
    assert result.segments[0].start == 12.5
    assert result.segments[0].end is None
    assert result.segments[0].timestamp_status == "start_only"
    with pytest.raises(SubtitleRenderingError):
        to_srt(result)
    assert to_srt(result, on_unrenderable="omit") == ""

    # An EXPLICIT engine-sent zero-length span survives on the MODEL as a
    # measurement (no diagnostic -- the engine did measure it), but it is
    # still unrenderable output-side: its cue would be invisible in every
    # player, so the renderer stays loud rather than emitting it.
    explicit = StreamReducer()
    explicit.add(TranscriptionEvent.final("s1", "hello", start=0.0, end=0.0))
    explicit_result = explicit.result()
    assert explicit_result.diagnostics == []
    assert explicit_result.segments is not None
    assert explicit_result.segments[0].timestamp_status == "measured"
    with pytest.raises(SubtitleRenderingError):
        to_srt(explicit_result)


def test_reducer_stores_unmeasured_timing_verbatim() -> None:
    """The engine's measurement is the single source of truth: an untimed final
    reduces to real None values (nothing fabricated, no marker, extra stays
    engine-owned and empty), while a timed final keeps its floats.
    """
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "timed", start=5.0, end=6.0))
    reducer.add(TranscriptionEvent.final("s2", "untimed"))
    segments = reducer.result().segments
    assert segments is not None

    by_text = {s.text: s for s in segments}
    assert by_text["untimed"].start is None
    assert by_text["untimed"].end is None
    assert by_text["untimed"].extra == {}
    assert by_text["timed"].start == 5.0
    assert by_text["timed"].end == 6.0


def test_reducer_sorts_only_when_every_segment_has_a_start() -> None:
    """A real onset is a real time position: start-only segments sort too.

    When every retained segment carries a ``start`` (measured or start-only)
    the reducer orders by it -- and ``text`` joins in the same order, so the
    transcript and the segment list can never disagree. One ``start=None``
    segment switches the whole result to arrival (reading) order instead of
    fabricating it a position.
    """
    sortable = StreamReducer()
    sortable.add(TranscriptionEvent.final("s1", "late", start=9.0, end=10.0))
    sortable.add(TranscriptionEvent.final("s2", "early", start=1.0))  # start-only
    sorted_result = sortable.result()
    assert sorted_result.text == "early late"
    assert sorted_result.segments is not None
    assert [s.text for s in sorted_result.segments] == ["early", "late"]

    unsortable = StreamReducer()
    unsortable.add(TranscriptionEvent.final("s1", "late", start=9.0, end=10.0))
    unsortable.add(TranscriptionEvent.final("s2", "untimed"))
    arrival = unsortable.result()
    assert arrival.text == "late untimed"
    assert arrival.segments is not None
    assert [s.text for s in arrival.segments] == ["late", "untimed"]


def test_empty_segments_with_the_diagnostic_still_render_zero_cues() -> None:
    """The null rule is unconditional and policy-independent: an app that
    deliberately emptied `segments` must never see the removed text
    resurrected as a fabricated full-span cue -- and never a policy error
    for segments it removed.
    """
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "hello"))
    reduced = reducer.result()
    assert [d.code for d in reduced.diagnostics] == [DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE]

    emptied = reduced.model_copy(update={"segments": []})

    assert emptied.text == "hello"
    assert to_srt(emptied) == ""
    assert to_vtt(emptied) == "WEBVTT\n"


def test_stale_diagnostic_on_fully_measured_result_is_ignored() -> None:
    """The values are the truth: a stray diagnostic cannot degrade real timing.

    Under the retired marker design a result-level diagnostic ALONE collapsed
    a fully measured timeline into one synthetic cue. Timing now lives only
    in the nullable values, so a stale/wrong diagnostic (for example, surviving a
    model_copy that replaced the segments) changes nothing: every measured
    cue renders faithfully under the default policy.
    """
    result = TranscriptionResult(
        text="hello world",
        segments=[
            Segment(start=0.0, end=1.0, text="hello"),
            Segment(start=1.0, end=2.0, text="world"),
        ],
        diagnostics=[
            Diagnostic(
                level="warning",
                code=DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE,
                message="stale disclosure from an earlier segment list",
            )
        ],
    )
    srt = to_srt(result)
    assert srt.count("-->") == 2
    assert "hello" in srt and "world" in srt


def test_timestamps_unavailable_constant_is_shared_and_exported() -> None:
    """The constant is homed in contract.results (it describes a property of the
    RESULT) and re-exported by runtime.streaming (its emitter) and the
    package root. All names MUST be the same object: a copy would let the
    emitter and consumers drift apart silently after a rename.
    """
    assert streaming_module.DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE is (
        results_module.DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE
    )
    assert (
        standard_asr_package.DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE
        is DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE
    )
    assert "DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE" in standard_asr_package.__all__
    assert DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE == "segment_timestamps_unavailable"


def test_unknown_missing_timestamps_policy_is_a_loud_value_error() -> None:
    """A typo'd policy must not silently become ``collapse``.

    ``Literal`` is not enforced at runtime; pre-fix, any unrecognized value
    fell through the ``error``/``omit`` arms into the collapse behavior --
    silently reintroducing the implicit data-loss choice the parameter exists
    to eliminate. Rejected loudly even when the policy would be inert.
    """
    mixed = TranscriptionResult(
        text="real missing",
        segments=[Segment(start=1.0, end=2.0, text="real"), _unavailable_segment("missing")],
    )
    measured = TranscriptionResult(text="ok", segments=[Segment(start=0.0, end=1.0, text="ok")])
    for render in (to_srt, to_vtt):
        with pytest.raises(ValueError, match="on_unrenderable"):
            render(mixed, on_unrenderable="omti")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="on_unrenderable"):
            render(measured, on_unrenderable=None)  # type: ignore[arg-type]


def test_synthetic_cue_with_zero_duration_still_gets_a_visible_span() -> None:
    """``duration=0.0`` must not synthesize the invisible cue it exists to avoid.

    The model legally allows a zero duration; using it verbatim would emit
    ``00:00:00 --> 00:00:00`` -- the zero-duration cue players silently drop.
    A non-empty transcript always gets the visible fallback span instead.
    """
    no_segments = TranscriptionResult(text="hello", duration=0.0)
    assert to_srt(no_segments) == "1\n00:00:00,000 --> 00:00:03,000\nhello\n"

    collapsed = TranscriptionResult(
        text="hello", segments=[_unavailable_segment("hello")], duration=0.0
    )
    assert to_srt(collapsed, on_unrenderable="collapse") == (
        "1\n00:00:00,000 --> 00:00:03,000\nhello\n"
    )
    assert "00:00:00.000 --> 00:00:03.000\nhello" in to_vtt(collapsed, on_unrenderable="collapse")


def test_synthetic_cue_visibility_is_decided_on_the_millisecond_grid() -> None:
    """A sub-millisecond ``duration`` is as invisible as an exact ``0.0``.

    ``duration=0.0005`` passes a raw ``> 0`` float check yet formats to
    ``00:00:00,000 --> 00:00:00,000`` -- the same silently dropped cue.
    The fallback decision must consult the SAME quantization the timestamp
    formatter renders (``int(round(s * 1000))``, ties-to-even), so the
    boundary cases pin the actual output grid:

    * ``0.0``, ``1e-9``, ``0.0005`` (rounds to 0 ms) -> fallback span;
    * ``0.0005001``, ``0.001`` (round to 1 ms) -> the real duration.
    """
    for invisible in (0.0, 1e-9, 0.0005):
        result = TranscriptionResult(text="hello", duration=invisible)
        assert to_srt(result) == "1\n00:00:00,000 --> 00:00:03,000\nhello\n", invisible

    for visible in (0.0005001, 0.001):
        result = TranscriptionResult(text="hello", duration=visible)
        assert to_srt(result) == "1\n00:00:00,000 --> 00:00:00,001\nhello\n", visible


def test_single_zero_start_segment_with_real_end_is_not_the_fallback() -> None:
    """A genuine segment that merely STARTS at 0.0 carries real timing: it must
    render as itself, never be swallowed by the timing-less fallback.
    """
    result = TranscriptionResult(
        text="real", segments=[Segment(start=0.0, end=1.25, text="real")], duration=9.0
    )
    assert to_srt(result) == "1\n00:00:00,000 --> 00:00:01,250\nreal\n"


def test_reduced_timestampless_stream_default_raises_collapse_renders() -> None:
    """End-to-end: a fully untimed reduce reaches the renderers as an explicit
    decision point. The default refuses to guess; "collapse" renders the ONE
    visible whole-text cue (never a file of invisible zero-duration cues).
    """
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "hello"))
    reducer.add(TranscriptionEvent.final("s2", "world"))
    result = reducer.result()

    assert result.duration is None
    assert [d.code for d in result.diagnostics] == [DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE]
    with pytest.raises(SubtitleRenderingError) as excinfo:
        to_srt(result)
    assert (excinfo.value.unrenderable, excinfo.value.total) == (2, 2)
    assert to_srt(result, on_unrenderable="collapse") == (
        "1\n00:00:00,000 --> 00:00:03,000\nhello world\n"
    )
    assert "00:00:00.000 --> 00:00:03.000\nhello world" in to_vtt(
        result, on_unrenderable="collapse"
    )


def test_srt_skips_empty_segment_and_renumbers() -> None:
    # An empty / whitespace-only segment among real ones must not produce a
    # payload-less cue, and the surviving SRT indices must stay contiguous.
    segs = [
        Segment(start=0.0, end=1.0, text="first"),
        Segment(start=1.0, end=2.0, text="   "),
        Segment(start=2.0, end=3.0, text="third"),
    ]
    srt = to_srt(TranscriptionResult(text="x", segments=segs))
    assert "1\n00:00:00,000 --> 00:00:01,000\nfirst" in srt
    assert "2\n00:00:02,000 --> 00:00:03,000\nthird" in srt
    # No third index (the whitespace cue was dropped, not emitted blank).
    assert "3\n" not in srt
    # No payload-less / empty cue (would manifest as a stray blank-line run).
    assert "\n\n\n" not in srt


def test_vtt_skips_empty_segment() -> None:
    segs = [
        Segment(start=0.0, end=1.0, text="first"),
        Segment(start=1.0, end=2.0, text=""),
        Segment(start=2.0, end=3.0, text="third"),
    ]
    vtt = to_vtt(TranscriptionResult(text="x", segments=segs))
    # WEBVTT header + two real cues = exactly two blank-line separators.
    assert vtt.count("\n\n") == 2
    assert "first" in vtt
    assert "third" in vtt


# --------------------------------------------------------------------------- #
# Renderer sanitization: transcript text must not forge / break cue structure.
# --------------------------------------------------------------------------- #
def test_srt_adversarial_blank_line_cannot_forge_cue() -> None:
    # A transcript with an interior blank line followed by digits + a timestamp
    # line would, unsanitized, forge a second SRT cue. After sanitization the
    # whole thing stays inside cue 1 and there is exactly one cue.
    evil = "Hello\n\n2\n00:00:05,000 --> 00:00:09,000\nInjected"
    srt = to_srt(TranscriptionResult(text="x", segments=[Segment(start=0.0, end=1.0, text=evil)]))
    # SRT cues are blank-line-delimited; with the interior blank line collapsed
    # there is no separator, so the injected content stays inside cue 1 and
    # cannot forge a second cue. (SRT, unlike VTT, does not treat "-->" in a
    # payload line as cue timing, so it need not be neutralized.)
    assert srt.count("\n\n") == 0
    assert srt.startswith("1\n")
    assert "Injected" in srt


def test_srt_collapses_interior_blank_lines() -> None:
    seg = Segment(start=0.0, end=1.0, text="line one\n\n\nline two")
    srt = to_srt(TranscriptionResult(text="x", segments=[seg]))
    assert "line one\nline two" in srt
    assert "line one\n\n" not in srt


def test_vtt_neutralizes_arrow_in_text() -> None:
    seg = Segment(start=0.0, end=1.0, text="a --> b")
    vtt = to_vtt(TranscriptionResult(text="x", segments=[seg]))
    # Only the cue timing line may contain "-->"; the payload arrow is
    # neutralized by the WebVTT ">" escape, so "-->" in the payload
    # becomes "--&gt;" and can never be read as cue timing.
    assert vtt.count("-->") == 1
    assert "a --&gt; b" in vtt


def test_vtt_adversarial_blank_line_cannot_forge_cue() -> None:
    evil = "Hi\n\n00:00:05.000 --> 00:00:09.000\nInjected"
    vtt = to_vtt(TranscriptionResult(text="x", segments=[Segment(start=0.0, end=1.0, text=evil)]))
    # WEBVTT header + one real cue: blank-line count is exactly one.
    assert vtt.count("\n\n") == 1
    assert vtt.count("-->") == 1


def test_vtt_escapes_markup_metacharacters() -> None:
    # WebVTT parses "<" as a cue-span tag start and "&" as a
    # character-reference start. Unescaped, the browser's cue-text tokenizer
    # silently drops "< b & AT&T <i" (everything up to the next ">"), so the
    # viewer loses transcript text with no error -- the cardinal silent-wrong
    # sin. The renderer MUST escape & -> &amp;, < -> &lt;, > -> &gt; per the W3C
    # WebVTT cue-text grammar so the literal text survives.
    seg = Segment(start=0.0, end=1.0, text="a < b & AT&T <i>x")
    vtt = to_vtt(TranscriptionResult(text="x", segments=[seg]))
    assert "a &lt; b &amp; AT&amp;T &lt;i&gt;x" in vtt
    # No raw markup metacharacters survive in the payload line.
    payload = vtt.split("\n")[-2]
    assert "<" not in payload
    # "&" only ever appears as the start of an escaped entity, never bare.
    for token in payload.split("&")[1:]:
        assert token.startswith(("amp;", "lt;", "gt;"))


def test_vtt_escapes_engine_leaked_special_tokens() -> None:
    # The realistic input: a Whisper-family engine leaks ``<unk>`` / ``<|...|>``
    # special tokens. They must be shown verbatim (escaped), not eaten as tags.
    seg = Segment(start=0.0, end=1.0, text="<unk> hi <|endoftext|>")
    vtt = to_vtt(TranscriptionResult(text="x", segments=[seg]))
    assert "&lt;unk&gt; hi &lt;|endoftext|&gt;" in vtt
    assert "<unk>" not in vtt.split("WEBVTT")[1]


def test_vtt_escape_order_no_double_escaping() -> None:
    # "&" must be escaped FIRST so the "&" it introduces (in &lt; / &gt;) is not
    # itself re-escaped into &amp;lt;. A literal "&lt;" in the source text must
    # round-trip as "&amp;lt;" (the ampersand escaped, the rest literal).
    seg = Segment(start=0.0, end=1.0, text="&lt; and < ")
    vtt = to_vtt(TranscriptionResult(text="x", segments=[seg]))
    assert "&amp;lt; and &lt;" in vtt
    assert "&amp;lt;lt;" not in vtt  # no double-escaping artifact


def test_srt_does_not_escape_markup() -> None:
    # SRT has no character-reference mechanism, so escaping would
    # surface a literal "&amp;" / "&lt;" to the viewer. The renderer passes "&"
    # and angle brackets through verbatim on the SRT path.
    seg = Segment(start=0.0, end=1.0, text="AT&T <i>bold</i>")
    srt = to_srt(TranscriptionResult(text="x", segments=[seg]))
    assert "AT&T <i>bold</i>" in srt
    assert "&amp;" not in srt
    assert "&lt;" not in srt


@pytest.mark.parametrize("render", [to_srt, to_vtt])
def test_lone_cr_normalized_cannot_forge_cue(
    render: Callable[[TranscriptionResult], str],
) -> None:
    # A lone CR ("\r") is a line terminator in WebVTT and many SRT
    # parsers, so "\r\r" is an effective blank line. The old sanitizer only
    # collapsed "\r?\n" runs, letting CR-delimited blank lines slip through and
    # forge a cue. The renderer now normalizes "\r\n"/"\r" -> "\n" before
    # collapsing, so no raw CR survives and no cue can be forged via CR.
    evil = "hello\r\r2\r00:00:05,000 --> 00:00:09,000\rEVIL"
    out = render(TranscriptionResult(text="x", segments=[Segment(start=0.0, end=1.0, text=evil)]))
    assert "\r" not in out
    # The CR-forged blank line is gone: payload stays in one cue. SRT emits no
    # blank-line separator for a single cue; VTT has exactly one (after WEBVTT).
    expected_separators = 1 if render is to_vtt else 0
    assert out.count("\n\n") == expected_separators


# --------------------------------------------------------------------------- #
# Renderer ordering: cues sorted by (start, channel, speaker).
# --------------------------------------------------------------------------- #
def test_srt_sorts_out_of_order_segments() -> None:
    segs = [
        Segment(start=2.0, end=3.0, text="second"),
        Segment(start=0.0, end=1.0, text="first"),
    ]
    srt = to_srt(TranscriptionResult(text="x", segments=segs))
    assert srt.index("first") < srt.index("second")
    assert srt.startswith("1\n00:00:00,000")


def test_srt_sorts_by_channel_on_tie() -> None:
    segs = [
        Segment(start=0.0, end=1.0, text="ch1", channel=1),
        Segment(start=0.0, end=1.0, text="ch0", channel=0),
    ]
    srt = to_srt(TranscriptionResult(text="x", segments=segs))
    assert srt.index("ch0") < srt.index("ch1")


def test_channel_rejects_negative_index() -> None:
    # channel is constrained to >= 0, so the renderer's None=-1 sort sentinel
    # can never collide with a real channel index.
    with pytest.raises(ValueError):
        Segment(start=0.0, end=1.0, text="x", channel=-1)
    with pytest.raises(ValueError):
        Word(start=0.0, end=1.0, text="x", channel=-1)
    with pytest.raises(ValueError):
        ChannelResult(channel=-1, text="x")


def test_srt_sorts_none_channel_before_real_channel() -> None:
    # A None channel sorts before any real channel (>= 0); channel=0 must keep
    # its real ordering and never be treated as if it were None.
    segs = [
        Segment(start=0.0, end=1.0, text="ch0", channel=0),
        Segment(start=0.0, end=1.0, text="none", channel=None),
    ]
    srt = to_srt(TranscriptionResult(text="x", segments=segs))
    assert srt.index("none") < srt.index("ch0")


def test_renderer_rejects_negative_preroll_time() -> None:
    # The data model now forbids negative times, so a "pre-roll"
    # segment can never reach the renderer: it is rejected at construction. This
    # is why the renderer no longer needs to clamp negative timestamps.
    with pytest.raises(ValueError):
        Segment(start=-0.5, end=0.5, text="pre-roll")


# --------------------------------------------------------------------------- #
# Speaker labels: shared construct-time validation + THE pinned
# segment-speaker synthesis rule.
# --------------------------------------------------------------------------- #
def _with_speaker(model: type[Word | Segment], speaker: str | None) -> Word | Segment:
    return model(start=0.0, end=1.0, text="x", speaker=speaker)


@pytest.mark.parametrize("model", [Word, Segment])
def test_speaker_label_rejects_empty(model: type[Word | Segment]) -> None:
    # "" would be an undefined third state between None and a real label.
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        _with_speaker(model, "")


@pytest.mark.parametrize("model", [Word, Segment])
@pytest.mark.parametrize("label", ["   ", "\t", "\n"])
def test_speaker_label_rejects_whitespace_only(model: type[Word | Segment], label: str) -> None:
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        _with_speaker(model, label)


@pytest.mark.parametrize("model", [Word, Segment])
@pytest.mark.parametrize("label", ["A ", " A", " A ", "A\n"])
def test_speaker_label_rejects_padded(model: type[Word | Segment], label: str) -> None:
    # Edge whitespace silently breaks within-result label consistency: "A " and
    # "A" read as two different speakers. Rejected -- never normalized (a
    # silent rewrite would hide the adapter bug producing the padding).
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        _with_speaker(model, label)


@pytest.mark.parametrize("model", [Word, Segment])
@pytest.mark.parametrize("label", [None, "A", "Guest-1", "John Doe", "说话人一"])
def test_speaker_label_valid_and_none(model: type[Word | Segment], label: str | None) -> None:
    # None (no attribution) and real labels -- including interior spaces --
    # construct unchanged.
    assert _with_speaker(model, label).speaker == label


@pytest.mark.parametrize("model", [Word, Segment])
def test_speaker_label_error_never_echoes_raw_value(model: type[Word | Segment]) -> None:
    # A speaker label can carry a personal name; the rejection message names
    # the rule, never the value (echoed verbatim by server 422 bodies / logs --
    # same redaction stance as the language-tag validator).
    sentinel = "Very Secret Name"
    with pytest.raises(ValidationError) as exc_info:
        _with_speaker(model, f" {sentinel} ")
    assert all(sentinel not in err["msg"] for err in exc_info.value.errors())


def _w(text: str, speaker: str | None, start: float) -> Word:
    return Word(start=start, end=start + 0.5, text=text, speaker=speaker)


def test_synthesize_speaker_majority() -> None:
    words = [_w("a", "A", 0.0), _w("b", "A", 0.5), _w("c", "B", 1.0)]
    assert synthesize_segment_speaker(words) == "A"


def test_synthesize_speaker_tie_earliest_word() -> None:
    # Equal counts -> the speaker of the earliest (lowest-index) word wins.
    assert synthesize_segment_speaker([_w("a", "A", 0.0), _w("b", "B", 0.5)]) == "A"
    assert synthesize_segment_speaker([_w("a", "B", 0.0), _w("b", "A", 0.5)]) == "B"
    # Count still beats earliest position: B first, but A carries more words.
    words = [_w("a", "B", 0.0), _w("b", "A", 0.5), _w("c", "A", 1.0)]
    assert synthesize_segment_speaker(words) == "A"


def test_synthesize_speaker_none_words_do_not_vote() -> None:
    # None-speaker words abstain: a None "majority" never beats one real label.
    words = [_w("a", None, 0.0), _w("b", None, 0.5), _w("c", "B", 1.0)]
    assert synthesize_segment_speaker(words) == "B"


def test_synthesize_speaker_no_votes_returns_none() -> None:
    assert synthesize_segment_speaker(None) is None
    assert synthesize_segment_speaker([]) is None
    assert synthesize_segment_speaker([_w("a", None, 0.0), _w("b", None, 0.5)]) is None


# --------------------------------------------------------------------------- #
# Speaker rendering (include_speakers): SRT "[label]: " prefix, VTT
# <v label> voice span, opt-in default, label sanitization, sort tie-break.
# --------------------------------------------------------------------------- #
def _speaker_result(*speakers: str | None) -> TranscriptionResult:
    segs = [
        Segment(start=float(i), end=float(i) + 1.0, text=f"line {i}", speaker=speaker)
        for i, speaker in enumerate(speakers)
    ]
    return TranscriptionResult(text=" ".join(s.text for s in segs), segments=segs)


def test_srt_include_speakers_prefixes_label() -> None:
    srt = to_srt(_speaker_result("Alice", "Bob"), include_speakers=True)
    assert "1\n00:00:00,000 --> 00:00:01,000\n[Alice]: line 0" in srt
    assert "2\n00:00:01,000 --> 00:00:02,000\n[Bob]: line 1" in srt


def test_vtt_include_speakers_wraps_voice_tag() -> None:
    vtt = to_vtt(_speaker_result("Alice", "Bob"), include_speakers=True)
    assert "00:00:00.000 --> 00:00:01.000\n<v Alice>line 0" in vtt
    assert "00:00:01.000 --> 00:00:02.000\n<v Bob>line 1" in vtt


def test_renderers_default_omits_speakers() -> None:
    # No flag -> byte-identical to the output for the same segments without
    # speakers: rendering stays a pure projection of the transcript text.
    labelled = _speaker_result("Alice", "Bob")
    unlabelled = _speaker_result(None, None)
    assert to_srt(labelled) == to_srt(unlabelled)
    assert to_vtt(labelled) == to_vtt(unlabelled)


def test_include_speakers_skips_none_speaker_segments() -> None:
    # Mixed result: only the labeled cue changes; a None speaker renders the
    # cue unchanged (no "[None]" fabrication).
    srt = to_srt(_speaker_result("Alice", None), include_speakers=True)
    assert "[Alice]: line 0" in srt
    assert "\nline 1" in srt
    assert "None" not in srt
    vtt = to_vtt(_speaker_result("Alice", None), include_speakers=True)
    assert "<v Alice>line 0" in vtt
    # The unlabeled cue's payload starts directly after its timing line -- no
    # voice tag was fabricated for it.
    assert "\nline 1" in vtt


def test_vtt_label_sanitized_for_voice_tag() -> None:
    # A raw '>' in the label would terminate the <v> tag early; '&'/'<' begin
    # markup. All three are escaped in the annotation, text untouched.
    result = _speaker_result("A>B&C<D")
    vtt = to_vtt(result, include_speakers=True)
    assert "<v A&gt;B&amp;C&lt;D>line 0" in vtt


def test_vtt_voice_tag_injected_after_text_sanitization() -> None:
    # The payload is escaped by _sanitize_cue_text; the voice tag is injected
    # AFTERWARD so it is not itself escaped away.
    seg = Segment(start=0.0, end=1.0, text="<unk> token", speaker="A")
    vtt = to_vtt(TranscriptionResult(text="x", segments=[seg]), include_speakers=True)
    assert "<v A>&lt;unk&gt; token" in vtt


@pytest.mark.parametrize("label", ["A\nB", "A\r\nB", "A\rB"])
def test_srt_label_newline_collapsed(label: str) -> None:
    # Interior line terminators pass the model validator (only edge whitespace
    # is construction-rejected) but must not forge cue structure: every
    # terminator form collapses to a single space in the rendered label.
    seg = Segment(start=0.0, end=1.0, text="hello", speaker=label)
    srt = to_srt(TranscriptionResult(text="x", segments=[seg]), include_speakers=True)
    assert "[A B]: hello" in srt
    vtt = to_vtt(TranscriptionResult(text="x", segments=[seg]), include_speakers=True)
    assert "<v A B>hello" in vtt


def test_srt_empty_text_with_speaker_still_skipped() -> None:
    # A label never resurrects a payload-less cue: no payload -> no cue, so no
    # index is consumed either.
    segs = [
        Segment(start=0.0, end=1.0, text="   ", speaker="Alice"),
        Segment(start=1.0, end=2.0, text="real", speaker="Bob"),
    ]
    srt = to_srt(TranscriptionResult(text="x", segments=segs), include_speakers=True)
    assert "Alice" not in srt
    assert srt.startswith("1\n00:00:01,000")
    vtt = to_vtt(TranscriptionResult(text="x", segments=segs), include_speakers=True)
    assert "Alice" not in vtt


def test_cues_sort_speaker_tie_break() -> None:
    # Ordering: (start, channel, speaker) -- speaker is the FINAL tie-break; None
    # sorts before any real label.
    segs = [
        Segment(start=0.0, end=1.0, text="from B", channel=0, speaker="B"),
        Segment(start=0.0, end=1.0, text="from A", channel=0, speaker="A"),
        Segment(start=0.0, end=1.0, text="no speaker", channel=0),
    ]
    srt = to_srt(TranscriptionResult(text="x", segments=segs))
    assert srt.index("no speaker") < srt.index("from A") < srt.index("from B")


def test_cues_sort_channel_still_beats_speaker() -> None:
    # speaker is the LAST key: a lower channel wins regardless of labels.
    segs = [
        Segment(start=0.0, end=1.0, text="ch1 A", channel=1, speaker="A"),
        Segment(start=0.0, end=1.0, text="ch0 B", channel=0, speaker="B"),
    ]
    srt = to_srt(TranscriptionResult(text="x", segments=segs))
    assert srt.index("ch0 B") < srt.index("ch1 A")


def test_vtt_multiline_cue_single_voice_tag_wraps_whole_body() -> None:
    # An unclosed <v> span legally runs to the end of the cue payload, so ONE
    # opening tag attributes the whole (multi-line) body.
    seg = Segment(start=0.0, end=1.0, text="line one\nline two", speaker="A")
    vtt = to_vtt(TranscriptionResult(text="x", segments=[seg]), include_speakers=True)
    assert "<v A>line one\nline two" in vtt
    assert vtt.count("<v A>") == 1


def test_sub_millisecond_measured_span_is_unrenderable() -> None:
    """The round-4 counterexample: a POSITIVE raw span can still be invisible.

    ``Segment(start=1.0, end=1.0004)`` passes every model invariant and the
    old measured-only check, yet formats to ``00:00:01,000 --> 00:00:01,000``
    -- the cue players silently drop. Renderability is decided on the same
    millisecond grid the formatter renders (ties-to-even included), for
    every cue, not only the synthetic fallback.
    """
    result = TranscriptionResult(
        text="real ghost",
        segments=[
            Segment(start=0.0, end=1.0, text="real"),
            Segment(start=1.0, end=1.0004, text="ghost"),
        ],
    )
    for render in (to_srt, to_vtt):
        with pytest.raises(SubtitleRenderingError) as excinfo:
            render(result)
        assert excinfo.value.unrenderable == 1
        assert excinfo.value.total == 2

    # "omit" keeps the visible cue only; the ghost text is an explicit loss.
    srt = to_srt(result, on_unrenderable="omit")
    assert "real" in srt
    assert "ghost" not in srt
    # "collapse" keeps every text, no per-segment timeline.
    assert "real ghost" in to_srt(result, on_unrenderable="collapse")

    # The grid boundary: a span that rounds to >= 1 ms renders normally.
    boundary = TranscriptionResult(text="ok", segments=[Segment(start=1.0, end=1.0006, text="ok")])
    assert to_srt(boundary) == "1\n00:00:01,000 --> 00:00:01,001\nok\n"


def test_equal_millisecond_distinct_floats_are_unrenderable() -> None:
    """Distinct floats quantizing to the SAME millisecond are invisible too.

    ``5.0001 -> 5.0004`` is a positive raw span whose bounds both format to
    ``00:00:05,000``; the output grid, not raw float arithmetic, decides.
    """
    result = TranscriptionResult(text="x", segments=[Segment(start=5.0001, end=5.0004, text="x")])
    with pytest.raises(SubtitleRenderingError):
        to_srt(result)


def test_empty_segments_are_skipped_not_unrenderable() -> None:
    """An empty segment has no text to lose, so it is not a policy decision.

    Both renderers already skip a payload-less segment (a cue with no payload
    line is malformed), so an empty segment produces no cue whether or not it
    carries timestamps. Judging renderability first made the SAME segment
    silently skipped when measured and a hard SubtitleRenderingError when
    not -- the default policy firing for a segment that would have produced
    nothing either way, while the policy exists to stop LOST TEXT and
    FABRICATED TIMING.
    """
    result = TranscriptionResult(
        text="hello",
        segments=[
            Segment(start=0.0, end=1.0, text="hello"),
            Segment(start=None, end=None, text=""),
            Segment(start=None, end=None, text="   \n  "),
        ],
    )
    # Default policy: no error, and the empty segments contribute no cues.
    srt = to_srt(result)
    assert srt.count("-->") == 1
    assert "hello" in srt
    assert to_vtt(result).count("-->") == 1

    # A measured-but-empty segment behaved this way already; the unmeasured
    # one now agrees.
    measured_empty = TranscriptionResult(
        text="hello",
        segments=[
            Segment(start=0.0, end=1.0, text="hello"),
            Segment(start=1.0, end=2.0, text=""),
        ],
    )
    assert to_srt(measured_empty).count("-->") == 1


def test_all_empty_segments_render_no_cues_and_never_fabricate() -> None:
    """Segmentation happened and every segment is empty: zero cues.

    Not the whole-text fallback -- that is the ``segments is None`` shape.
    Falling through to it would fabricate a full-span cue for a result whose
    segmentation deliberately reported nothing visible.
    """
    result = TranscriptionResult(
        text="hello",
        segments=[Segment(start=None, end=None, text=""), Segment(start=None, end=None, text="  ")],
    )
    assert to_srt(result) == ""
    assert to_vtt(result) == "WEBVTT\n"


def test_unrenderable_counts_exclude_empty_segments() -> None:
    """The raised counts describe segments that could actually have rendered."""
    result = TranscriptionResult(
        text="a b",
        segments=[
            Segment(start=0.0, end=1.0, text="a"),
            Segment(start=None, end=None, text="b"),  # real text, no span: a genuine policy case
            Segment(start=None, end=None, text=""),  # no payload: not part of the decision
        ],
    )
    with pytest.raises(SubtitleRenderingError) as excinfo:
        to_srt(result)
    assert excinfo.value.unrenderable == 1
    assert excinfo.value.total == 2

    # 'omit' keeps the renderable one; 'collapse' still carries all the text.
    assert to_srt(result, on_unrenderable="omit").count("-->") == 1
    assert "a b" in to_srt(result, on_unrenderable="collapse")


def test_whitespace_only_whole_text_yields_no_cue() -> None:
    """The synthetic whole-text cue is skipped when its payload is blank.

    ``segments is None`` uses the whole-text fallback, and a whitespace-only
    transcript sanitizes to nothing -- a payload-less cue block, which strict
    SRT parsers reject and WebVTT calls malformed.
    """
    blank = TranscriptionResult(text="   \n  ")
    assert to_srt(blank) == ""
    assert to_vtt(blank) == "WEBVTT\n"


# --------------------------------------------------------------------------- #
# Wire-visible JSON slots enforce a string key domain (Python == JSON layer)
# --------------------------------------------------------------------------- #
_F7_SECRET = b"sk-STANDARD-ASR-REVIEW-F7"  # a bytes key that must never coerce

#: Each wire model plus the minimal valid base fields to construct it. Invalid
#: ``extra`` payloads go in via ``model_validate`` (which takes ``Any``) so the
#: runtime-rejection tests stay static-type-clean -- the same before-validator
#: governs direct construction (verified: the constructor path rejects too).
_WIRE_MODELS: list[tuple[type[BaseModel], dict[str, object]]] = [
    (Word, {"start": 0.0, "end": 1.0, "text": "w"}),
    (Segment, {"start": 0.0, "end": 1.0, "text": "s"}),
    (TranscriptionResult, {"text": "t"}),
]


@pytest.mark.parametrize(("model", "base"), _WIRE_MODELS)
def test_extra_rejects_bytes_object_keys(model: type[BaseModel], base: dict[str, object]) -> None:
    """A ``bytes`` key in ``extra`` is refused, never coerced to ``str``.

    pydantic's lax ``dict[str, JsonValue]`` COERCED ``b"x"`` to ``"x"`` -- a key
    no JSON document can express, silently admitted on the Python layer only.
    """
    with pytest.raises(ValidationError) as exc_info:
        model.model_validate({**base, "extra": {_F7_SECRET: 1}})
    assert exc_info.value.errors()[0]["type"] == "standard_asr_json_object_key"


def test_extra_bytes_key_collision_no_longer_collapses_silently() -> None:
    """``{"x": 1, b"x": 2}`` MUST fail loudly, not collapse to one key.

    The coercion collapsed two distinct Python keys into a single ``"x"`` (last
    wins) -- the silent wrong result a wire-visible slot must never produce.
    """
    with pytest.raises(ValidationError) as exc_info:
        Segment.model_validate({"start": 0.0, "end": 1.0, "text": "s", "extra": {"x": 1, b"x": 2}})
    assert exc_info.value.errors()[0]["type"] == "standard_asr_json_object_key"


def test_extra_rejects_int_object_keys() -> None:
    """A non-string, non-bytes key is refused as well (JSON keys are strings)."""
    with pytest.raises(ValidationError) as exc_info:
        Segment.model_validate({"start": 0.0, "end": 1.0, "text": "s", "extra": {1: "v"}})
    assert exc_info.value.errors()[0]["type"] in (
        "standard_asr_json_object_key",
        # pydantic's own int-key rejection is also acceptable; the point is a
        # loud failure, never a coercion. (The contract's before-validator fires first.)
        "invalid_key",
    )


def test_extra_rejects_nested_bytes_keys_at_every_depth() -> None:
    """The key domain holds at every nesting level, not only the top."""
    for extra in (
        {"outer": {b"inner": 1}},
        {"list": [{b"inner": 1}]},
        {"deep": {"a": [{"b": {b"c": 1}}]}},
    ):
        with pytest.raises(ValidationError) as exc_info:
            Segment.model_validate({"start": 0.0, "end": 1.0, "text": "s", "extra": extra})
        assert exc_info.value.errors()[0]["type"] == "standard_asr_json_object_key"


def test_extra_rejects_hostile_str_subclass_keys() -> None:
    """An exact ``str`` is required: a hostile subclass could dodge de-dup.

    A ``str`` subclass that overrides ``__eq__`` / ``__hash__`` keeps two keys
    that both serialize to ``"x"`` DISTINCT in the input mapping -- so accepting
    it would reintroduce the wire collision on serialization. Exact-type keying
    denies it.
    """

    class _Wedge(str):
        def __hash__(self) -> int:
            return 999

        def __eq__(self, other: object) -> bool:
            return self is other

    with pytest.raises(ValidationError) as exc_info:
        Segment.model_validate(
            {"start": 0.0, "end": 1.0, "text": "s", "extra": {"x": 1, _Wedge("x"): 2}}
        )
    assert exc_info.value.errors()[0]["type"] == "standard_asr_json_object_key"


def test_diagnostic_json_slots_reject_bytes_keys() -> None:
    """``Diagnostic.provided`` / ``effective`` share the same key domain."""
    with pytest.raises(ValidationError) as exc_info:
        Diagnostic.model_validate({"code": "c", "message": "m", "provided": {b"k": 1}})
    assert exc_info.value.errors()[0]["type"] == "standard_asr_json_object_key"
    with pytest.raises(ValidationError):
        Diagnostic.model_validate({"code": "c", "message": "m", "effective": {"ok": {b"k": 1}}})


def test_extra_string_keys_and_json_roundtrip_are_unchanged() -> None:
    """Every legitimate (string-keyed) value still constructs byte-for-byte.

    The rule only closes the Python-only key domain; a wire document (whose
    object keys are always strings) is untouched, and nested string-keyed JSON
    round-trips exactly.
    """
    nested: dict[str, object] = {"cost": 1, "meta": {"model": "x", "scores": [{"k": 0.5}]}}
    built = Segment.model_validate({"start": 0.0, "end": 1.0, "text": "s", "extra": nested})
    assert built.extra == nested
    assert Segment(start=0.0, end=1.0, text="s").extra == {}
    parsed = Word.model_validate_json('{"start":0,"end":1,"text":"w","extra":{"a":{"b":[1,2]}}}')
    assert parsed.extra == {"a": {"b": [1, 2]}}


def test_string_key_walk_is_cycle_safe() -> None:
    """The key walk terminates on a self-referential mapping (no hang).

    Exercised directly (pydantic's own JsonValue validation would separately
    reject a cyclic structure): the safety property under test is that the
    standard layer's OWN walk never spins on a cycle.
    """
    cyclic: dict[str, object] = {"self": None}
    cyclic["self"] = cyclic
    assert results_module.require_json_string_keys(cyclic) is cyclic
    # A shared/cyclic LIST is memoized too (visited once, not re-walked).
    shared: list[object] = [{"k": 1}]
    shared.append(shared)  # self-referential list
    assert results_module.require_json_string_keys({"a": shared, "b": shared}) is not None
    # A non-string key anywhere in a cyclic structure is still caught.
    cyclic_bad: dict[object, object] = {"a": None}
    cyclic_bad["a"] = cyclic_bad
    cyclic_bad[b"bad"] = 1
    with pytest.raises(Exception, match="JSON object keys must be strings"):
        results_module.require_json_string_keys(cyclic_bad)
