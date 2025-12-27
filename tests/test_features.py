"""Tests for feature flag definitions."""

from standard_asr.features import FeatureFlag


def test_feature_flag_values() -> None:
    assert FeatureFlag.WORD_TIMESTAMPS.value == "word_timestamps"
    assert FeatureFlag.STREAMING_INPUT.value == "streaming_input"
