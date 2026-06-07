# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the audio input negotiation matrix."""

from __future__ import annotations

import numpy as np
import pytest

from standard_asr.audio_input import (
    AudioArray,
    AudioBase64,
    AudioBytes,
    AudioPath,
    AudioUrl,
    InputKind,
)
from standard_asr.audio_negotiation import (
    ConversionOp,
    ConversionPlan,
    NoViablePath,
    can_accept,
    negotiate,
    negotiate_or_raise,
)
from standard_asr.exceptions import IncompatibleAudioInputError

ARR = InputKind.ARRAY
FILE = InputKind.ENCODED_FILE
BYTES = InputKind.ENCODED_BYTES
URL = InputKind.FETCHABLE_URL


def _arr() -> AudioArray:
    return AudioArray(np.zeros(8, dtype=np.float32), 16000)


def test_array_passthrough_to_array_engine() -> None:
    plan = negotiate(_arr(), {ARR})
    assert isinstance(plan, ConversionPlan)
    assert plan.is_passthrough
    assert plan.target_kind is ARR


def test_array_encodes_to_wav_for_file_engine() -> None:
    plan = negotiate(_arr(), {FILE, BYTES})
    assert isinstance(plan, ConversionPlan)
    assert plan.operations == (ConversionOp.ENCODE_WAV,)
    assert plan.lossy is True
    assert plan.target_kind is BYTES


def test_array_to_url_only_engine_fails() -> None:
    result = negotiate(_arr(), {URL})
    assert isinstance(result, NoViablePath)


def test_path_passthrough_file() -> None:
    plan = negotiate(AudioPath("a.wav"), {FILE, BYTES})
    assert isinstance(plan, ConversionPlan)
    assert plan.is_passthrough
    assert plan.target_kind is FILE


def test_path_read_to_bytes_when_only_bytes() -> None:
    plan = negotiate(AudioPath("a.wav"), {BYTES})
    assert isinstance(plan, ConversionPlan)
    assert plan.operations == (ConversionOp.READ_FILE,)


def test_path_decode_to_array_needs_extra() -> None:
    plan = negotiate(AudioPath("a.mp3"), {ARR})
    assert isinstance(plan, ConversionPlan)
    assert plan.operations == (ConversionOp.DECODE,)
    assert plan.requires_audio_extra is True


def test_bytes_passthrough() -> None:
    plan = negotiate(AudioBytes(b"x"), {BYTES})
    assert isinstance(plan, ConversionPlan)
    assert plan.is_passthrough


def test_bytes_decode_to_array() -> None:
    plan = negotiate(AudioBytes(b"x"), {ARR})
    assert isinstance(plan, ConversionPlan)
    assert plan.operations == (ConversionOp.DECODE,)


def test_base64_decode_to_bytes() -> None:
    plan = negotiate(AudioBase64("AAAA"), {BYTES})
    assert isinstance(plan, ConversionPlan)
    assert plan.operations == (ConversionOp.B64_DECODE,)


def test_base64_to_array_two_steps() -> None:
    plan = negotiate(AudioBase64("AAAA"), {ARR})
    assert isinstance(plan, ConversionPlan)
    assert plan.operations == (ConversionOp.B64_DECODE, ConversionOp.DECODE)


def test_url_passthrough_when_accepted() -> None:
    plan = negotiate(AudioUrl("https://x/a.wav"), {URL, FILE, BYTES})
    assert isinstance(plan, ConversionPlan)
    assert plan.is_passthrough
    assert plan.target_kind is URL


def test_url_to_array_engine_v1_fails() -> None:
    result = negotiate(AudioUrl("https://x/a.wav"), {ARR})
    assert isinstance(result, NoViablePath)


def test_can_accept() -> None:
    assert can_accept(_arr(), {ARR}) is True
    assert can_accept(_arr(), {URL}) is False


def test_negotiate_or_raise_ok() -> None:
    assert isinstance(negotiate_or_raise(_arr(), {ARR}), ConversionPlan)


def test_negotiate_or_raise_raises() -> None:
    with pytest.raises(IncompatibleAudioInputError) as exc:
        negotiate_or_raise(_arr(), {URL})
    assert "AudioArray" in str(exc.value)
