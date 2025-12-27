"""Tests for transcription result models."""

from standard_asr.results import Segment, TranscriptionResult, Word


def test_transcription_result_text_only() -> None:
    result = TranscriptionResult(text="hello")
    assert result.text_only() == "hello"


def test_segment_and_word_models() -> None:
    word = Word(start=0.0, end=0.5, text="hi", probability=0.9)
    segment = Segment(start=0.0, end=1.0, text="hi", words=[word])
    result = TranscriptionResult(text="hi", segments=[segment], words=[word])

    assert result.segments is not None
    assert result.segments[0].words is not None
    assert result.words is not None
    assert result.words[0].text == "hi"
