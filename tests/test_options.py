"""Tests for transcription options helpers."""

from standard_asr.options import BaseTranscribeOptions, coerce_options


def test_coerce_options_default() -> None:
    options = coerce_options(None, BaseTranscribeOptions)
    assert isinstance(options, BaseTranscribeOptions)
    assert options.task == "transcribe"


def test_coerce_options_from_mapping() -> None:
    options = coerce_options(
        {"language": "en", "word_timestamps": True}, BaseTranscribeOptions
    )
    assert options.language == "en"
    assert options.word_timestamps is True


def test_coerce_options_existing_instance() -> None:
    original = BaseTranscribeOptions(language="en")
    options = coerce_options(original, BaseTranscribeOptions)
    assert options is original


def test_coerce_options_base_to_subclass() -> None:
    class _Custom(BaseTranscribeOptions):
        pass

    original = BaseTranscribeOptions(language="en")
    options = coerce_options(original, _Custom)
    assert isinstance(options, _Custom)
    assert options.language == "en"
