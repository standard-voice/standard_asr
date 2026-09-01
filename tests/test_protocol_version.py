# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for protocol-version syntax and feature compatibility."""

from __future__ import annotations

import pytest

from standard_asr.contract.exceptions import ProtocolCompatibilityError
from standard_asr.contract.protocol_version import (
    CURRENT_PROTOCOL_VERSION,
    PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE,
    PROTOCOL_FEATURE_MINIMUMS,
    SUPPORTED_PROTOCOL_MAJOR,
    parse_protocol_version,
    require_protocol_feature,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.0.0", (0, 0, 0)),
        ("1.1.0", (1, 1, 0)),
        ("12.345.6789", (12, 345, 6789)),
    ],
)
def test_parse_protocol_version_accepts_canonical_triplets(
    value: str, expected: tuple[int, int, int]
) -> None:
    assert parse_protocol_version(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "1",
        "1.1",
        "1.1.0.0",
        "01.1.0",
        "1.01.0",
        "1.1.00",
        "v1.1.0",
        "1.1.0rc1",
        " 1.1.0",
        "1.1.0 ",
        "1.-1.0",
        "123456789012345678901234567890123.1.0",
    ],
)
def test_parse_protocol_version_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        parse_protocol_version(value)


def test_protocol_constants_define_artifact_feature_floor() -> None:
    assert CURRENT_PROTOCOL_VERSION == "1.1.0"
    assert SUPPORTED_PROTOCOL_MAJOR == 1
    assert PROTOCOL_FEATURE_MINIMUMS[PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE] == "1.1.0"


@pytest.mark.parametrize("version", ["1.1.0", "1.1.9", "1.2.0", "1.999.0"])
def test_require_protocol_feature_accepts_floor_and_later_minor(version: str) -> None:
    assert require_protocol_feature(version, PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE) is None


def test_require_protocol_feature_rejects_older_minor_with_context() -> None:
    with pytest.raises(ProtocolCompatibilityError) as caught:
        require_protocol_feature("1.0.9", PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE)
    error = caught.value
    assert error.protocol_version == "1.0.9"
    assert error.feature == PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE
    assert error.required_protocol_version == "1.1.0"
    assert "requires protocol 1.1.0" in str(error)


@pytest.mark.parametrize("version", ["0.9.0", "2.0.0"])
def test_require_protocol_feature_rejects_other_majors(version: str) -> None:
    with pytest.raises(ProtocolCompatibilityError, match="supports protocol major 1"):
        require_protocol_feature(version, PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE)


def test_require_protocol_feature_wraps_malformed_version() -> None:
    with pytest.raises(ProtocolCompatibilityError, match="invalid protocol_version") as caught:
        require_protocol_feature("1.1", PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE)
    assert isinstance(caught.value.__cause__, ValueError)


def test_require_protocol_feature_rejects_unknown_feature_token() -> None:
    with pytest.raises(ValueError, match="Unknown protocol feature"):
        require_protocol_feature("1.1.0", "artifact_lifecyle")
