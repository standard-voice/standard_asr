# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Dummy Standard ASR plugin used for entrypoint demos and tests."""

from .engine import (
    DummyASR,
    DummyASRConfig,
    DummyASRProperties,
    DummyDefaultASR,
    DummyDefaultProperties,
)
from .entrypoint import create_default, create_echo

__all__ = [
    "DummyASR",
    "DummyASRConfig",
    "DummyASRProperties",
    "DummyDefaultASR",
    "DummyDefaultProperties",
    "create_default",
    "create_echo",
]
