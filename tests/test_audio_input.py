# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for audio input types and coercion."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from standard_asr.audio_input import (
    AudioArray,
    AudioBase64,
    AudioBytes,
    AudioPath,
    AudioUrl,
    InputKind,
    coerce_audio_input,
)


def test_str_coerces_to_path_never_url() -> None:
    coerced = coerce_audio_input("https://example.com/a.wav")
    assert isinstance(coerced, AudioPath)
    assert coerced.provided_kind is InputKind.ENCODED_FILE


def test_pathlike_coerces_to_path() -> None:
    coerced = coerce_audio_input(Path("/tmp/a.wav"))
    assert isinstance(coerced, AudioPath)


def test_bytes_coerces_to_bytes() -> None:
    coerced = coerce_audio_input(b"RIFF....")
    assert isinstance(coerced, AudioBytes)
    assert coerced.container is None


def test_bare_array_coerces_without_sample_rate() -> None:
    arr = np.zeros(8, dtype=np.float32)
    coerced = coerce_audio_input(arr)
    assert isinstance(coerced, AudioArray)
    assert coerced.sample_rate is None


def test_array_tuple_coerces_with_sample_rate() -> None:
    arr = np.zeros(8, dtype=np.float32)
    coerced = coerce_audio_input((arr, 16000))
    assert isinstance(coerced, AudioArray)
    assert coerced.sample_rate == 16000


def test_existing_variant_returned_unchanged() -> None:
    url = AudioUrl("https://example.com/a.wav")
    assert coerce_audio_input(url) is url
    assert url.provided_kind is InputKind.FETCHABLE_URL


def test_base64_provided_kind() -> None:
    assert AudioBase64("AAAA").provided_kind is InputKind.ENCODED_BYTES


def test_array_uses_identity_equality() -> None:
    arr = np.zeros(4, dtype=np.float32)
    a = AudioArray(arr)
    b = AudioArray(arr)
    assert a == a
    assert a != b


def test_unsupported_type_raises() -> None:
    with pytest.raises(TypeError):
        coerce_audio_input(12345)  # type: ignore[arg-type]


def test_bad_tuple_length_raises() -> None:
    with pytest.raises(TypeError):
        coerce_audio_input((np.zeros(4),))  # type: ignore[arg-type]


def test_tuple_non_array_first_raises() -> None:
    with pytest.raises(TypeError):
        coerce_audio_input(("nope", 16000))  # type: ignore[arg-type]


def test_tuple_bool_sample_rate_raises() -> None:
    with pytest.raises(TypeError):
        coerce_audio_input((np.zeros(4), True))  # type: ignore[arg-type]


def test_array_rejects_int_dtype() -> None:
    int16 = np.zeros(8, dtype=np.int16)
    with pytest.raises(TypeError, match="floating dtype"):
        AudioArray(int16)  # type: ignore[arg-type]


def test_array_rejects_int_dtype_via_coercion() -> None:
    int16 = np.zeros(8, dtype=np.int16)
    with pytest.raises(TypeError, match="floating dtype"):
        coerce_audio_input(int16)  # type: ignore[arg-type]


def test_array_accepts_float32_and_float64() -> None:
    assert AudioArray(np.zeros(8, dtype=np.float32)).samples.dtype == np.float32
    assert AudioArray(np.zeros(8, dtype=np.float64)).samples.dtype == np.float64


def test_path_accepts_os_pathlike() -> None:
    class _P(os.PathLike[str]):
        def __fspath__(self) -> str:
            return "/tmp/x.wav"

    coerced = coerce_audio_input(_P())
    assert isinstance(coerced, AudioPath)
