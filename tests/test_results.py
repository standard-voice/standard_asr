# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for transcription result models and renderers."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from pydantic import ValidationError

import standard_asr as standard_asr_package
from standard_asr.contract import results as results_module
from standard_asr.contract.results import (
    DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE,
    SEGMENT_EXTRA_TIMESTAMP_PLACEHOLDER,
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
    # The blanket `metadata: dict[str, Any]` field was removed: a "standardized
    # engine-agnostic metadata" dict with no standardized keys, no writer and no
    # reader is the same unstructured-data disease the spec dropped from
    # Properties and Capabilities. Standardized result data earns a real field;
    # everything engine-specific goes in `extra`. The model is extra="forbid",
    # so a caller still passing metadata= is rejected, never silently absorbed.
    # (model_validate keeps the removed key a runtime rejection, not a static
    # type error.)
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
    # consumers (e.g. the renderers) silently drop all per-channel timing, so
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
    with pytest.raises(ValueError, match="duplicate entries for channel index 0"):
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
    # segments=[] means segmentation ran and found nothing (e.g. silence). Per
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
    # segments=None + unknown duration (e.g. a reduced stream): the synthetic
    # cue must not be zero-duration -- ffmpeg / VLC / browser WebVTT silently
    # drop zero-duration cues, hiding the only transcript content. The
    # renderer falls back to a fixed 3 s span.
    result = TranscriptionResult(text="only text")
    srt = to_srt(result)
    assert "1\n00:00:00,000 --> 00:00:03,000\nonly text" in srt
    vtt = to_vtt(result)
    assert "00:00:00.000 --> 00:00:03.000\nonly text" in vtt


def test_zero_span_segments_without_the_diagnostic_render_faithfully() -> None:
    # Span VALUES are never sniffed: a [0.0, 0.0] span in a result that carries
    # no segment_timestamps_unavailable diagnostic is REAL timing the engine
    # reported, so each segment renders as its own faithful zero-length cue.
    # (The renderer previously guessed "all-zero == placeholder" and fabricated
    # a 3 s whole-text cue here, silently inventing timing for genuinely
    # timestamped segments -- the fix is to key on the diagnostic instead.)
    segs = [
        Segment(start=0.0, end=0.0, text="hello"),
        Segment(start=0.0, end=0.0, text="world"),
    ]
    result = TranscriptionResult(text="hello world", segments=segs)
    assert result.diagnostics == []

    srt = to_srt(result)
    assert srt == (
        "1\n00:00:00,000 --> 00:00:00,000\nhello\n\n2\n00:00:00,000 --> 00:00:00,000\nworld\n"
    )

    vtt = to_vtt(result)
    assert "00:00:00.000 --> 00:00:00.000\nhello" in vtt
    assert "00:00:00.000 --> 00:00:00.000\nworld" in vtt
    assert vtt.count("-->") == 2


def test_zero_span_segments_without_the_diagnostic_ignore_the_known_duration() -> None:
    # A known duration does not license rewriting reported segment timing: with
    # no diagnostic these are real spans, so `duration` never leaks into a cue.
    segs = [
        Segment(start=0.0, end=0.0, text="a"),
        Segment(start=0.0, end=0.0, text="b"),
    ]
    result = TranscriptionResult(text="a b", segments=segs, duration=7.5)

    srt = to_srt(result)
    assert srt == "1\n00:00:00,000 --> 00:00:00,000\na\n\n2\n00:00:00,000 --> 00:00:00,000\nb\n"
    assert "00:00:07,500" not in srt
    assert "00:00:07.500" not in to_vtt(result)


def test_diagnostic_free_empty_segment_text_renders_nothing() -> None:
    # A payload-less segment yields no cue (the empty-payload rule), and with no
    # diagnostic present the empty `text` is NOT a synthesis source either.
    result = TranscriptionResult(text="", segments=[Segment(start=0.0, end=0.0, text="")])
    assert to_srt(result) == ""
    assert to_vtt(result) == "WEBVTT\n"


def test_zero_span_segment_text_survives_an_empty_result_text() -> None:
    # A diagnostic-free result whose `text` is empty but whose segment carries
    # the transcript must still render that segment. Under the old value-sniffing
    # fallback this shape routed to the synthetic whole-text cue and, with `text`
    # empty, produced an EMPTY subtitle file -- the transcript was dropped.
    result = TranscriptionResult(text="", segments=[Segment(start=0.0, end=0.0, text="hello")])
    assert to_srt(result) == "1\n00:00:00,000 --> 00:00:00,000\nhello\n"
    assert "hello" in to_vtt(result)


def test_zero_span_segment_keeps_its_speaker_label() -> None:
    # A diagnostic-free zero-span segment renders as a real cue, so
    # include_speakers still attributes it. (The synthetic whole-text fallback
    # drops labels -- it has no single attributable speaker -- so the old
    # value-sniffing route silently discarded diarization here.)
    result = TranscriptionResult(
        text="x", segments=[Segment(start=0.0, end=0.0, text="hi", speaker="Alice")]
    )
    assert (
        to_srt(result, include_speakers=True) == "1\n00:00:00,000 --> 00:00:00,000\n[Alice]: hi\n"
    )
    assert "<v Alice>hi" in to_vtt(result, include_speakers=True)


def _placeholder_diagnostic(message: str) -> Diagnostic:
    """Build the reducer's placeholder-timestamp disclosure.

    Args:
        message: The human-readable detail.

    Returns:
        A warning :class:`Diagnostic` carrying the standard code.
    """
    return Diagnostic(level="warning", code=DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE, message=message)


def _placeholder_segment(text: str, **kwargs: object) -> Segment:
    """Build a segment marked as a placeholder span, as the reducer marks them.

    Args:
        text: The segment text.
        **kwargs: Extra :class:`Segment` fields (e.g. ``speaker``).

    Returns:
        A ``0.0``-span segment carrying the reserved placeholder marker.
    """
    return Segment(
        start=0.0,
        end=0.0,
        text=text,
        extra={SEGMENT_EXTRA_TIMESTAMP_PLACEHOLDER: True},
        **kwargs,  # pyright: ignore[reportArgumentType]
    )


def test_timestamps_unavailable_diagnostic_forces_the_single_synthetic_cue() -> None:
    # ALL segments marked as placeholders: there is zero usable timing, so the
    # result routes to the synthetic whole-text fallback -- rendering the 0.0
    # spans per-segment would emit only zero-duration cues players silently
    # drop. Both signals are required: the result-level diagnostic says
    # placeholders exist, the per-segment marker says WHICH.
    segs = [_placeholder_segment("hello"), _placeholder_segment("world")]
    diagnostic = _placeholder_diagnostic("2 of 2 segments carry placeholder timestamps.")
    result = TranscriptionResult(text="hello world", segments=segs, diagnostics=[diagnostic])

    srt = to_srt(result)
    assert srt == "1\n00:00:00,000 --> 00:00:03,000\nhello world\n"
    assert "2\n" not in srt
    vtt = to_vtt(result)
    assert "00:00:00.000 --> 00:00:03.000\nhello world" in vtt
    assert vtt.count("-->") == 1

    # A known duration spans the synthetic cue instead of the 3 s fallback.
    known = TranscriptionResult(
        text="hello world", segments=segs, duration=7.5, diagnostics=[diagnostic]
    )
    assert to_srt(known) == "1\n00:00:00,000 --> 00:00:07,500\nhello world\n"
    assert "00:00:00.000 --> 00:00:07.500\nhello world" in to_vtt(known)


def test_diagnostic_with_no_marked_segment_renders_every_span_faithfully() -> None:
    # The diagnostic ALONE never discards timing: with no segment marked, every
    # span is real and each renders as its own cue. (The reducer always marks
    # the placeholders it created; a diagnostic with nothing marked describes a
    # timeline whose retained segments all carry real timestamps.) Collapsing
    # this to one synthetic cue would throw away a correct timeline -- and the
    # speaker labels with it.
    segs = [
        Segment(start=0.0, end=1.0, text="hello", speaker="Alice"),
        Segment(start=1.0, end=2.0, text="world", speaker="Bob"),
    ]
    result = TranscriptionResult(
        text="hello world",
        segments=segs,
        diagnostics=[_placeholder_diagnostic("some spans were placeholders upstream")],
    )

    assert to_srt(result) == (
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n2\n00:00:01,000 --> 00:00:02,000\nworld\n"
    )
    assert to_srt(result, include_speakers=True).count("[Alice]: hello") == 1
    assert to_vtt(result).count("-->") == 2


def test_final_with_start_and_no_end_renders_its_real_zero_length_span() -> None:
    # A final carrying start=0.0 and no end is a REAL timestamp: the reducer
    # stores end=start and emits no diagnostic, so the renderer must show the
    # engine's own 00:00:00,000 --> 00:00:00,000 span rather than fabricate a
    # 3 s cue from a value that merely looks like a placeholder.
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "hello", start=0.0))
    result = reducer.result()

    assert result.diagnostics == []
    assert to_srt(result) == "1\n00:00:00,000 --> 00:00:00,000\nhello\n"


def test_partially_timestamped_reduce_keeps_the_real_cue_and_omits_the_placeholder() -> None:
    # Mixed shape: one timestamped final and one timestamp-less final. The
    # per-segment marker says exactly WHICH span is a placeholder, so the real
    # 5-6 s cue survives with its true timing while the timing-less segment is
    # OMITTED from the timed projection -- any placement for it would be
    # fabricated, and a zero-duration cue at a made-up point is both a lie in
    # the file and invisible in players. The omission is disclosed by the
    # diagnostic, and the full text remains in result.text.
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "b", start=5.0, end=6.0))
    reducer.add(TranscriptionEvent.final("s2", "a"))
    result = reducer.result()

    assert result.text == "b a"
    assert [d.code for d in result.diagnostics] == [DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE]

    srt = to_srt(result)
    assert srt == "1\n00:00:05,000 --> 00:00:06,000\nb\n"
    assert srt.count("-->") == 1
    # The placeholder's text is not rendered at a fabricated time...
    assert "\na\n" not in srt
    assert "00:00:00,000 --> 00:00:00,000" not in srt
    # ...and the whole transcript is NOT collapsed into one synthetic cue either.
    assert "b a" not in srt
    assert to_vtt(result) == "WEBVTT\n\n00:00:05.000 --> 00:00:06.000\nb\n"


def test_one_placeholder_never_discards_a_predominantly_real_timeline() -> None:
    # Round-2 regression: with three real-timestamped segments and one
    # placeholder, the whole timeline used to collapse into a single synthetic
    # cue -- three correct spans (and their speaker labels) thrown away because
    # of one missing timestamp. The three real cues must survive, correctly
    # timed and attributed; only the unplaceable segment drops out.
    segs = [
        Segment(start=0.0, end=1.0, text="one", speaker="Alice"),
        Segment(start=1.0, end=2.0, text="two", speaker="Bob"),
        Segment(start=2.0, end=3.0, text="three", speaker="Alice"),
        _placeholder_segment("four", speaker="Bob"),
    ]
    result = TranscriptionResult(
        text="one two three four",
        segments=segs,
        diagnostics=[_placeholder_diagnostic("1 of 4 segments carry placeholder timestamps.")],
    )

    srt = to_srt(result, include_speakers=True)
    assert srt == (
        "1\n00:00:00,000 --> 00:00:01,000\n[Alice]: one\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n[Bob]: two\n\n"
        "3\n00:00:02,000 --> 00:00:03,000\n[Alice]: three\n"
    )
    assert "four" not in srt
    vtt = to_vtt(result, include_speakers=True)
    assert vtt.count("-->") == 3
    assert "four" not in vtt


def test_empty_segments_with_the_diagnostic_still_render_zero_cues() -> None:
    # Round-2 regression: the null rule is unconditional. An app that
    # deliberately emptied `segments` must never see the removed text resurrected
    # as a fabricated full-span cue just because the result also carries the
    # placeholder diagnostic.
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "hello"))
    reduced = reducer.result()
    assert [d.code for d in reduced.diagnostics] == [DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE]

    emptied = reduced.model_copy(update={"segments": []})

    assert emptied.text == "hello"
    assert to_srt(emptied) == ""
    assert to_vtt(emptied) == "WEBVTT\n"


def test_reducer_marks_only_the_placeholder_segments() -> None:
    # The per-segment marker is what makes the mixed shape decidable: value
    # sniffing cannot tell a 0.0 placeholder from a genuine zero-length span at
    # t=0, so the reducer records the fact at the moment it invents the span.
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "timed", start=5.0, end=6.0))
    reducer.add(TranscriptionEvent.final("s2", "untimed"))
    segments = reducer.result().segments
    assert segments is not None

    by_text = {s.text: s for s in segments}
    assert by_text["untimed"].extra == {SEGMENT_EXTRA_TIMESTAMP_PLACEHOLDER: True}
    # A real span carries NO marker -- the key is only ever a positive claim.
    assert by_text["timed"].extra == {}


def test_placeholder_marker_constant_and_its_top_level_export() -> None:
    # The reserved key is part of the wire contract engines must not repurpose,
    # so its value is pinned; and both placeholder signals are reachable from
    # the package root -- an app consuming results should not have to import
    # from `standard_asr.contract.results` to interpret them.
    assert SEGMENT_EXTRA_TIMESTAMP_PLACEHOLDER == "timestamp_placeholder"
    assert (
        standard_asr_package.SEGMENT_EXTRA_TIMESTAMP_PLACEHOLDER
        is SEGMENT_EXTRA_TIMESTAMP_PLACEHOLDER
    )
    assert (
        standard_asr_package.DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE
        is DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE
    )
    assert "SEGMENT_EXTRA_TIMESTAMP_PLACEHOLDER" in standard_asr_package.__all__
    assert "DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE" in standard_asr_package.__all__


def test_timestamps_unavailable_constant_is_shared_by_results_and_streaming() -> None:
    # The constant is homed in contract.results (it describes a property of the
    # RESULT that the renderers consume) and re-exported by runtime.streaming
    # (its emitter). Both names MUST be the same object: a copy would let the
    # emitter and the renderer drift apart silently after a rename.
    assert streaming_module.DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE is (
        results_module.DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE
    )
    assert DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE == "segment_timestamps_unavailable"


def test_single_zero_start_segment_with_real_end_is_not_the_fallback() -> None:
    # A genuine segment that merely STARTS at 0.0 carries real timing: it must
    # render as itself, never be swallowed by the timing-less fallback.
    result = TranscriptionResult(
        text="real", segments=[Segment(start=0.0, end=1.25, text="real")], duration=9.0
    )
    assert to_srt(result) == "1\n00:00:00,000 --> 00:00:01,250\nreal\n"


def test_reduced_timestampless_stream_renders_one_visible_cue_end_to_end() -> None:
    # End-to-end: the StreamReducer's timestamp-less output (0.0 placeholders +
    # the segment_timestamps_unavailable disclosure) must reach the subtitle
    # renderers as ONE visible cue rather than a file of dropped zero-duration
    # cues.
    reducer = StreamReducer()
    reducer.add(TranscriptionEvent.final("s1", "hello"))
    reducer.add(TranscriptionEvent.final("s2", "world"))
    result = reducer.result()

    assert result.duration is None
    assert [d.code for d in result.diagnostics] == [DIAG_SEGMENT_TIMESTAMPS_UNAVAILABLE]
    assert to_srt(result) == "1\n00:00:00,000 --> 00:00:03,000\nhello world\n"
    assert "00:00:00.000 --> 00:00:03.000\nhello world" in to_vtt(result)


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
    # The realistic input: a Whisper-family engine leaks "<unk>" / "<|...|>"
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
    # Mixed result: only the labelled cue changes; a None speaker renders the
    # cue unchanged (no "[None]" fabrication).
    srt = to_srt(_speaker_result("Alice", None), include_speakers=True)
    assert "[Alice]: line 0" in srt
    assert "\nline 1" in srt
    assert "None" not in srt
    vtt = to_vtt(_speaker_result("Alice", None), include_speakers=True)
    assert "<v Alice>line 0" in vtt
    # The unlabelled cue's payload starts directly after its timing line -- no
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
    # AFTERWARDS so it is not itself escaped away.
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
