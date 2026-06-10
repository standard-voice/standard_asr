# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for transcription result models and renderers."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from standard_asr.renderers import to_srt, to_vtt
from standard_asr.results import (
    ChannelResult,
    Diagnostic,
    Segment,
    TranscriptionResult,
    Word,
)


def test_minimal_result() -> None:
    result = TranscriptionResult(text="hello")
    assert result.text == "hello"
    assert result.detected_language is None
    assert result.diagnostics == []


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
    # 'auto' is the detect-me directive, never a detection *result* (spec TR.1).
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
# Timestamp invariants (spec TR.1/TR.2): non-negative, finite, ordered floats.
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
    # Spec TR.4: ignoring `channels` must be lossless. A channel entry carrying
    # segments while the top level has none would make channel-agnostic
    # consumers (e.g. the renderers) silently drop all per-channel timing, so
    # the shape is rejected at construction.
    chan = ChannelResult(channel=0, text="hi", segments=[Segment(start=0.0, end=1.0, text="hi")])
    with pytest.raises(ValueError, match="time-merged union"):
        TranscriptionResult(text="hi", channels=[chan])


def test_channel_words_require_top_level_words() -> None:
    # Same TR.4 derivability invariant for the flattened word-level view.
    chan = ChannelResult(channel=0, text="hi", words=[Word(start=0.0, end=0.5, text="hi")])
    with pytest.raises(ValueError, match="time-merged union"):
        TranscriptionResult(text="hi", channels=[chan])


def test_channels_with_top_level_segments_and_words_construct() -> None:
    # The TR.4-conformant shape (top level = time-merge of all channels) is
    # accepted; per-channel detail with a populated top level is the contract.
    word = Word(start=0.0, end=0.5, text="hi")
    seg = Segment(start=0.0, end=1.0, text="hi")
    chan = ChannelResult(channel=0, text="hi", segments=[seg], words=[word])
    result = TranscriptionResult(text="hi", segments=[seg], words=[word], channels=[chan])
    assert result.channels is not None
    assert result.segments == [seg]
    assert result.words == [word]


def test_duplicate_channel_index_rejected() -> None:
    # TR.4 defines `channels` as one ChannelResult per channel, so a
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
    # The TR.2 (start, channel) ordering is an ENGINE obligation
    # (compliance-verified), NOT a construct-time invariant. The streaming
    # reducer legitimately keeps arrival order for timestamp-less engines and
    # sorts only by start, so a construct-time ordering validator would reject
    # valid reduced results. This test locks the deliberate non-enforcement so a
    # future change does not silently add a breaking validator (the renderers
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
    # the §TR.1 null rule this must NOT fabricate a full-span cue from text.
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
# Renderer ordering: cues sorted by (start, channel) per spec TR.2.
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
    # The data model now forbids negative times (spec TR.2), so a "pre-roll"
    # segment can never reach the renderer: it is rejected at construction. This
    # is why the renderer no longer needs to clamp negative timestamps.
    with pytest.raises(ValueError):
        Segment(start=-0.5, end=0.5, text="pre-roll")
