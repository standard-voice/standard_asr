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
    result = TranscriptionResult(text="")
    srt = to_srt(result)
    assert "00:00:00,000 --> 00:00:00,000" in srt
