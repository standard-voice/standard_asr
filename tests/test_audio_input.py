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
    AudioBytes,
    AudioPath,
    AudioStorageUri,
    AudioUrl,
    coerce_audio_input,
)


def test_str_coerces_to_path_never_url() -> None:
    coerced = coerce_audio_input("https://example.com/a.wav")
    assert isinstance(coerced, AudioPath)


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


@pytest.mark.parametrize("dtype", [np.int64, np.int32])
def test_array_tuple_accepts_numpy_integer_sample_rate(dtype: type) -> None:
    arr = np.zeros(8, dtype=np.float32)
    coerced = coerce_audio_input((arr, dtype(16000)))
    assert isinstance(coerced, AudioArray)
    assert coerced.sample_rate == 16000
    # Normalized to a builtin int -- downstream never sees a NumPy scalar.
    assert type(coerced.sample_rate) is int


def test_array_tuple_rejects_float_sample_rate() -> None:
    with pytest.raises(TypeError):
        coerce_audio_input((np.zeros(4, dtype=np.float32), 16000.0))  # type: ignore[arg-type]


def test_existing_variant_returned_unchanged() -> None:
    url = AudioUrl("https://example.com/a.wav")
    assert coerce_audio_input(url) is url


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


@pytest.mark.parametrize("bad_rate", [0, -1, -44100])
def test_array_rejects_non_positive_sample_rate(bad_rate: int) -> None:
    # A zero/negative sample rate is rejected at construction. Left
    # unchecked, an "any"-rate engine would be handed the bogus rate silently
    # (the cardinal sin) and a concrete-list engine would crash deep in
    # resampling (ZeroDivisionError on the duration check / bare ValueError) --
    # the failure mode drifting with engine metadata. Failing here makes the
    # application unit/shape bug loud and engine-independent (matches
    # AudioFormat.sample_rate gt=0).
    with pytest.raises(ValueError, match="positive number of Hz"):
        AudioArray(np.zeros(8, dtype=np.float32), bad_rate)


def test_array_accepts_none_sample_rate() -> None:
    # None (rate unknown) stays valid -- it is resolved by the R6
    # strict/best_effort policy downstream, not rejected at construction.
    assert AudioArray(np.zeros(8, dtype=np.float32), None).sample_rate is None


def test_array_allows_empty_samples_at_construction() -> None:
    # (per the verdict): emptiness is NOT rejected at construction --
    # an empty array can be a legitimate passthrough boundary input. It is
    # handled where it matters (resample time), not here.
    assert AudioArray(np.zeros(0, dtype=np.float32), 16000).samples.size == 0


def test_path_accepts_os_pathlike() -> None:
    class _P(os.PathLike[str]):
        def __fspath__(self) -> str:
            return "/tmp/x.wav"

    coerced = coerce_audio_input(_P())
    assert isinstance(coerced, AudioPath)


# --- AudioStorageUri ---


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/key.wav",
        "gs://bucket/key.flac",
        "gcs://bucket/key.flac",
        "oss://bucket/key.mp3",
        "abfs://container/key.wav",
        "abfss://container/key.wav",
        "az://container/key.wav",
        "wasb://container/key.wav",
        "wasbs://container/key.wav",
    ],
)
def test_storage_uri_accepts_allowlisted_schemes(uri: str) -> None:
    su = AudioStorageUri(uri)
    assert su.value == uri


def test_storage_uri_scheme_is_case_insensitive() -> None:
    assert AudioStorageUri("S3://Bucket/Key.WAV").value == "S3://Bucket/Key.WAV"


def test_storage_uri_round_trips_through_coercion() -> None:
    su = AudioStorageUri("s3://bucket/key.wav")
    assert coerce_audio_input(su) is su


def test_bare_str_never_coerces_to_storage_uri() -> None:
    # A bare s3:// string is still treated as a local path -- the same explicit
    # construction safety stance as AudioUrl/AudioBase64.
    coerced = coerce_audio_input("s3://bucket/key.wav")
    assert isinstance(coerced, AudioPath)


@pytest.mark.parametrize(
    "bad",
    [
        "http://example.com/a.wav",
        "https://example.com/a.wav",
        "file:///tmp/a.wav",
        "ftp://host/a.wav",
        "s3:/bucket/key.wav",  # missing the second slash -> no '://'
        "bucket/key.wav",  # schemeless
        "",  # empty
        "://bucket/key.wav",  # empty scheme
    ],
)
def test_storage_uri_rejects_bad_scheme(bad: str) -> None:
    with pytest.raises(ValueError):
        AudioStorageUri(bad)
