# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the audio conversion executor."""

from __future__ import annotations

import base64
import io
import wave
from pathlib import Path

import numpy as np
import pytest

from standard_asr.audio_conversion import PreparedAudio, execute_plan
from standard_asr.audio_input import (
    AudioArray,
    AudioBase64,
    AudioBytes,
    AudioPath,
    AudioUrl,
    InputKind,
)
from standard_asr.audio_negotiation import negotiate
from standard_asr.exceptions import AudioProcessingError


def _exec(provided: object, accepted: set[InputKind], **kw: object) -> PreparedAudio:
    plan = negotiate(provided, accepted)  # type: ignore[arg-type]
    assert not isinstance(plan, type(None))
    return execute_plan(
        provided,  # type: ignore[arg-type]
        plan,  # type: ignore[arg-type]
        accepted_sample_rates=kw.get("accepted_sample_rates", "any"),
        native_sample_rate=kw.get("native_sample_rate", 16000),  # type: ignore[arg-type]
        required_input_sample_rate=kw.get("required_input_sample_rate"),  # type: ignore[arg-type]
        max_file_size=kw.get("max_file_size"),  # type: ignore[arg-type]
        strict=kw.get("strict", True),  # type: ignore[arg-type]
    )


def _wav_bytes(samples: int = 8, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(np.zeros(samples, dtype=np.int16).tobytes())
    return buf.getvalue()


def test_array_passthrough() -> None:
    prepared = _exec(AudioArray(np.zeros(8, dtype=np.float32), 16000), {InputKind.ARRAY})
    assert prepared.kind is InputKind.ARRAY
    assert prepared.array is not None
    assert prepared.sample_rate == 16000


def test_array_encode_to_wav() -> None:
    prepared = _exec(
        AudioArray(np.zeros(8, dtype=np.float32), 16000),
        {InputKind.ENCODED_BYTES},
    )
    assert prepared.kind is InputKind.ENCODED_BYTES
    assert prepared.container == "wav"
    assert any(d.code == "audio_conversion" for d in prepared.diagnostics)


def test_array_encode_downmix_diagnostic() -> None:
    stereo = np.zeros((8, 2), dtype=np.float32)
    prepared = _exec(AudioArray(stereo, 16000), {InputKind.ENCODED_BYTES})
    assert sum(d.code == "audio_conversion" for d in prepared.diagnostics) == 2


def test_array_encode_oversize_raises() -> None:
    with pytest.raises(AudioProcessingError):
        _exec(
            AudioArray(np.zeros(100000, dtype=np.float32), 16000),
            {InputKind.ENCODED_BYTES},
            max_file_size=128,
        )


def test_path_passthrough_file() -> None:
    prepared = _exec(AudioPath("a.wav"), {InputKind.ENCODED_FILE})
    assert prepared.kind is InputKind.ENCODED_FILE
    assert prepared.path == "a.wav"


def test_path_read_to_bytes(tmp_path: Path) -> None:
    f = tmp_path / "a.wav"
    f.write_bytes(_wav_bytes())
    prepared = _exec(AudioPath(f), {InputKind.ENCODED_BYTES})
    assert prepared.kind is InputKind.ENCODED_BYTES
    assert prepared.data == _wav_bytes()
    assert prepared.container == "wav"


def test_bytes_passthrough() -> None:
    prepared = _exec(AudioBytes(b"xyz", "mp3"), {InputKind.ENCODED_BYTES})
    assert prepared.data == b"xyz"
    assert prepared.container == "mp3"


def test_base64_to_bytes() -> None:
    payload = base64.b64encode(b"hello").decode()
    prepared = _exec(AudioBase64(payload), {InputKind.ENCODED_BYTES})
    assert prepared.data == b"hello"


def test_base64_data_uri() -> None:
    payload = "data:audio/wav;base64," + base64.b64encode(b"hello").decode()
    prepared = _exec(AudioBase64(payload), {InputKind.ENCODED_BYTES})
    assert prepared.data == b"hello"


def test_base64_invalid_raises() -> None:
    with pytest.raises(AudioProcessingError):
        _exec(AudioBase64("!!!notb64!!!"), {InputKind.ENCODED_BYTES})


def test_url_passthrough() -> None:
    prepared = _exec(AudioUrl("https://x/a.wav"), {InputKind.FETCHABLE_URL})
    assert prepared.kind is InputKind.FETCHABLE_URL
    assert prepared.url == "https://x/a.wav"


def test_decode_path_to_array(tmp_path: Path) -> None:
    f = tmp_path / "a.wav"
    f.write_bytes(_wav_bytes(rate=8000))
    prepared = _exec(AudioPath(f), {InputKind.ARRAY}, native_sample_rate=16000)
    assert prepared.kind is InputKind.ARRAY
    assert prepared.array is not None
    assert any(d.code == "audio_conversion" for d in prepared.diagnostics)


def test_bare_array_strict_raises() -> None:
    with pytest.raises(AudioProcessingError):
        _exec(
            AudioArray(np.zeros(8, dtype=np.float32)),
            {InputKind.ARRAY},
            accepted_sample_rates=[16000],
            strict=True,
        )


def test_bare_array_best_effort_assumes() -> None:
    prepared = _exec(
        AudioArray(np.zeros(8, dtype=np.float32)),
        {InputKind.ARRAY},
        accepted_sample_rates=[16000],
        strict=False,
    )
    assert any(d.code == "assumed_sample_rate" for d in prepared.diagnostics)


def test_array_resampled_to_accepted_rate() -> None:
    prepared = _exec(
        AudioArray(np.zeros(48000, dtype=np.float32), 48000),
        {InputKind.ARRAY},
        accepted_sample_rates=[16000],
        native_sample_rate=16000,
    )
    assert prepared.sample_rate == 16000
    assert any(d.code == "resampled_with" for d in prepared.diagnostics)
