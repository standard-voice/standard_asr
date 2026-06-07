# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dummy Standard ASR cookbook plugin."""

from __future__ import annotations

import numpy as np
import pytest
from std_dummy_asr import (
    DummyASR,
    DummyASRConfig,
    DummyASRProperties,
    DummyDefaultASR,
    DummyDefaultProperties,
    create_default,
    create_echo,
)

from standard_asr import RuntimeParams, StandardASR
from standard_asr.audio_input import AudioArray
from standard_asr.capabilities import DeclaredCapabilities


def _audio(n: int = 8) -> AudioArray:
    return AudioArray(np.zeros(n, dtype=np.float32), 16000)


def test_engine_is_standard_asr() -> None:
    assert isinstance(DummyASR(), StandardASR)


def test_default_message_echo() -> None:
    # No explicit message and no env override -> the config default "echo".
    result = DummyASR().transcribe(_audio(4))
    assert result.text == "echo: 4 samples"
    assert result.metadata == {"samples": 4}
    assert result.duration == pytest.approx(4 / 16000)


def test_explicit_message_overrides_default() -> None:
    result = DummyASR(message="hi").transcribe(_audio(2))
    assert result.text == "hi: 2 samples"


def test_message_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unset message falls back to the STANDARD_ASR_DUMMY_* environment.
    monkeypatch.setenv("STANDARD_ASR_DUMMY_MESSAGE", "fromenv")
    result = DummyASR().transcribe(_audio(1))
    assert result.text == "fromenv: 1 samples"


def test_explicit_message_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STANDARD_ASR_DUMMY_MESSAGE", "fromenv")
    result = DummyASR(message="explicit").transcribe(_audio(1))
    assert result.text == "explicit: 1 samples"


def test_runtime_language_override_reflected() -> None:
    # runtime_override is supported, so an explicit request language is honored
    # and reported as the detected language.
    result = DummyASR().transcribe(_audio(), RuntimeParams(language="en"))
    assert result.detected_language == "en"


def test_default_language_used_when_no_override() -> None:
    result = DummyASR().transcribe(_audio())
    assert result.detected_language == "en"


def test_zero_samples_when_array_missing() -> None:
    # A bare (rate-less) array in best-effort mode still transcribes; an empty
    # buffer reports zero samples.
    result = DummyASR(message="m").transcribe(AudioArray(np.zeros(0, dtype=np.float32), 16000))
    assert result.text == "m: 0 samples"
    assert result.metadata == {"samples": 0}


def test_properties_and_capabilities_class_level() -> None:
    assert isinstance(DummyASR.properties, DummyASRProperties)
    assert isinstance(DummyASR.declared_capabilities, DeclaredCapabilities)
    assert DummyASR.properties.model_id == "dummy/echo"


def test_config_model_defaults() -> None:
    config = DummyASRConfig()
    assert config.engine == "dummy"
    assert config.message == "echo"
    assert config.default_language == "en"


def test_create_echo_factory() -> None:
    engine = create_echo()
    assert isinstance(engine, DummyASR)
    assert engine.transcribe(_audio(3)).text == "echo: 3 samples"


def test_create_echo_forwards_kwargs() -> None:
    engine = create_echo(message="custom")
    assert engine.transcribe(_audio(1)).text == "custom: 1 samples"


def test_create_default_factory() -> None:
    engine = create_default()
    assert isinstance(engine, DummyDefaultASR)
    # The default preset's model_id matches the "dummy/" entry-point key.
    assert engine.properties.model_id == "dummy/"


def test_default_preset_properties() -> None:
    props = DummyDefaultProperties()
    assert props.model_name == ""
    assert props.engine_id == "dummy"
    assert isinstance(DummyDefaultASR.properties, DummyDefaultProperties)
