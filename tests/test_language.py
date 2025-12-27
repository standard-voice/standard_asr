"""Tests for BCP 47 language helpers."""

from standard_asr.language import is_valid_bcp47, normalize_bcp47


def test_normalize_bcp47() -> None:
    assert normalize_bcp47("EN-US") == "en-us"
    assert normalize_bcp47(" zh_cn ") == "zh-cn"


def test_is_valid_bcp47() -> None:
    assert is_valid_bcp47("en") is True
    assert is_valid_bcp47("en-US") is True
    assert is_valid_bcp47("und") is True
    assert is_valid_bcp47("x-private") is True
    assert is_valid_bcp47("") is False
    assert is_valid_bcp47("en--US") is False
    assert is_valid_bcp47("en@US") is False
