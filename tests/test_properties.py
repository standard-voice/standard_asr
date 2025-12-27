"""Tests for BaseProperties validation helpers."""

from __future__ import annotations

from typing import TypedDict

import pytest

from standard_asr.asr_properties import BaseProperties


class _BaseKwargs(TypedDict):
    engine_id: str
    model_name: str
    protocol_version: str
    supported_languages: list[str]
    supported_devices: list[str]
    supported_sample_rates: list[int]
    supported_channels: list[int]
    audio_dtype: str


def _base_kwargs() -> _BaseKwargs:
    return {
        "engine_id": "engine",
        "model_name": "model",
        "protocol_version": "0.2.0",
        "supported_languages": ["en-US"],
        "supported_devices": ["cpu"],
        "supported_sample_rates": [16000],
        "supported_channels": [1],
        "audio_dtype": "float32",
    }


def test_properties_language_normalization() -> None:
    data = _base_kwargs()
    data["supported_languages"] = ["EN_us"]

    props = BaseProperties(**data)

    assert props.supported_languages == ["en-us"]


def test_properties_language_validation_errors() -> None:
    data = _base_kwargs()
    data["supported_languages"] = ["bad@@tag"]

    with pytest.raises(ValueError):
        BaseProperties(**data)

    data = _base_kwargs()
    data["supported_languages"] = []

    with pytest.raises(ValueError):
        BaseProperties(**data)


def test_properties_sample_rate_validation_errors() -> None:
    data = _base_kwargs()
    data["supported_sample_rates"] = []

    with pytest.raises(ValueError):
        BaseProperties(**data)

    data = _base_kwargs()
    data["supported_sample_rates"] = [0]

    with pytest.raises(ValueError):
        BaseProperties(**data)


def test_properties_channel_validation_errors() -> None:
    data = _base_kwargs()
    data["supported_channels"] = []

    with pytest.raises(ValueError):
        BaseProperties(**data)

    data = _base_kwargs()
    data["supported_channels"] = [-1]

    with pytest.raises(ValueError):
        BaseProperties(**data)


def test_properties_numpy_dtype_and_model_id() -> None:
    props = BaseProperties(**_base_kwargs())

    assert props.numpy_dtype.name == "float32"
    assert props.model_id == "engine/model"
