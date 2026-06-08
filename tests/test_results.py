# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for transcription result models and renderers."""

from __future__ import annotations

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


def test_logprob_separate_from_probability() -> None:
    word = Word(start=0.0, end=0.1, text="x", probability=0.8, logprob=-0.2)
    assert word.probability == 0.8
    assert word.logprob == -0.2


def test_channels_field() -> None:
    chan = ChannelResult(channel=1, text="left")
    result = TranscriptionResult(text="left right", channels=[chan])
    assert result.channels is not None
    assert result.channels[0].channel == 1


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
    # Only the cue timing line may contain "-->"; payload arrow neutralized.
    assert vtt.count("-->") == 1
    assert "a -> b" in vtt


def test_vtt_adversarial_blank_line_cannot_forge_cue() -> None:
    evil = "Hi\n\n00:00:05.000 --> 00:00:09.000\nInjected"
    vtt = to_vtt(TranscriptionResult(text="x", segments=[Segment(start=0.0, end=1.0, text=evil)]))
    # WEBVTT header + one real cue: blank-line count is exactly one.
    assert vtt.count("\n\n") == 1
    assert vtt.count("-->") == 1


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


def test_renderer_clamps_negative_preroll_time() -> None:
    # Data model allows negative (pre-roll) start; renderer clamps to zero for
    # the SRT/VTT grammar (documented format constraint, not a silent mask).
    seg = Segment(start=-0.5, end=0.5, text="pre-roll")
    srt = to_srt(TranscriptionResult(text="x", segments=[seg]))
    assert "00:00:00,000 --> 00:00:00,500" in srt


def test_word_segment_allow_negative_times() -> None:
    # ge=0 decision: negatives are permitted (streaming pre-roll).
    w = Word(start=-0.2, end=0.0, text="pre")
    s = Segment(start=-0.2, end=0.0, text="pre")
    assert w.start == -0.2
    assert s.start == -0.2
