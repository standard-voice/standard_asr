# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for protocol-version syntax and feature compatibility."""

from __future__ import annotations

from types import MappingProxyType

import pytest

import standard_asr.contract.protocol_version as protocol_version_module
from standard_asr.contract.exceptions import ProtocolCompatibilityError
from standard_asr.contract.protocol_version import (
    CURRENT_PROTOCOL_VERSION,
    PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE,
    PROTOCOL_FEATURE_MINIMUMS,
    SUPPORTED_PROTOCOL_MAJOR,
    parse_protocol_version,
    require_protocol_feature,
    require_supported_protocol,
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
    ],
)
def test_parse_protocol_version_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        parse_protocol_version(value)


def test_parse_protocol_version_names_the_bound_for_an_overlong_token() -> None:
    # The length cap is the length of the largest canonical token, so it
    # admits that token and refuses anything longer before the grammar runs.
    # A 33-digit major still matches the grammar; its real defect is the
    # component bound, and the reason must say so instead of blaming syntax.
    assert parse_protocol_version("4294967295.4294967295.4294967295") == (
        4294967295,
        4294967295,
        4294967295,
    )
    with pytest.raises(ValueError, match="4294967295") as caught:
        parse_protocol_version("123456789012345678901234567890123.1.0")
    assert "syntax" not in str(caught.value)


def test_parse_protocol_version_enforces_the_cross_language_component_bound() -> None:
    # The protocol version is a cross-language wire token: components beyond
    # 32-bit unsigned cannot be compared exactly by common JSON
    # implementations, so the parser bounds each component.
    assert parse_protocol_version("4294967295.0.0") == (4294967295, 0, 0)
    with pytest.raises(ValueError, match="4294967295"):
        parse_protocol_version("4294967296.0.0")
    with pytest.raises(ValueError, match="4294967295"):
        parse_protocol_version("0.2.99999999999")


def test_production_constants_are_a_coherent_release_state() -> None:
    # Guards the constants themselves, not the algebra: a mis-set release
    # constant satisfies every synthetic-branch test yet ships an incoherent
    # state. Each rule here is one the runtime silently assumes.
    current = parse_protocol_version(CURRENT_PROTOCOL_VERSION)
    assert current[0] == SUPPORTED_PROTOCOL_MAJOR
    for feature, minimum in PROTOCOL_FEATURE_MINIMUMS.items():
        parsed = parse_protocol_version(minimum)
        assert parsed[0] == SUPPORTED_PROTOCOL_MAJOR, feature
        assert parsed <= current, feature


def test_protocol_constants_define_artifact_feature_floor() -> None:
    assert CURRENT_PROTOCOL_VERSION == "0.2.0"
    assert SUPPORTED_PROTOCOL_MAJOR == 0
    assert PROTOCOL_FEATURE_MINIMUMS[PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE] == "0.2.0"


@pytest.mark.parametrize("version", ["0.2.0", "0.2.9"])
def test_require_protocol_feature_accepts_the_supported_line(version: str) -> None:
    assert require_protocol_feature(version, PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE) is None


def test_require_supported_protocol_accepts_the_line_and_rejects_others() -> None:
    # Within protocol major 0 the minor is the breaking axis, so an OLDER line
    # and a NEWER line are equally unsupported -- with direction-aware
    # remediation, because "upgrade the plugin" is wrong advice when the
    # PLUGIN is the newer party.
    assert require_supported_protocol("0.2.0") is None
    assert require_supported_protocol("0.2.7") is None

    with pytest.raises(ProtocolCompatibilityError) as caught:
        require_supported_protocol("0.1.9")
    error = caught.value
    assert error.protocol_version == "0.1.9"
    assert error.feature is None
    assert "outside the supported pre-stable line 0.2" in str(error)
    assert "Upgrade the plugin." in str(error)

    with pytest.raises(ProtocolCompatibilityError) as caught:
        require_supported_protocol("0.3.0")
    assert "outside the supported pre-stable line 0.2" in str(caught.value)
    assert "Upgrade the core" in str(caught.value)


def test_require_supported_protocol_wraps_malformed_version() -> None:
    with pytest.raises(ProtocolCompatibilityError, match="invalid protocol_version") as caught:
        require_supported_protocol("1.1")
    assert isinstance(caught.value.__cause__, ValueError)


def test_require_protocol_feature_shares_the_line_gate() -> None:
    # The feature gate delegates the line rule to require_supported_protocol,
    # so a line mismatch raises the same line error (no feature context).
    with pytest.raises(ProtocolCompatibilityError) as caught:
        require_protocol_feature("0.1.9", PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE)
    assert caught.value.feature is None
    assert "outside the supported pre-stable line 0.2" in str(caught.value)


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0", "2.0.0"])
def test_require_protocol_feature_rejects_other_majors(version: str) -> None:
    with pytest.raises(ProtocolCompatibilityError, match="supports protocol major 0"):
        require_protocol_feature(version, PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE)


def test_require_protocol_feature_rejects_a_predating_version_within_the_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The feature-minimum comparison is only reachable within the supported
    # line while a minimum carries a nonzero patch; pin one synthetically, in
    # a COHERENT release state (a core never knows a floor above its own
    # version, per the production-constants invariant), so the branch stays
    # proven until a real minimum exercises it. The engine between the floor
    # and the core's own version proves the comparison is against the floor,
    # not against CURRENT_PROTOCOL_VERSION.
    monkeypatch.setattr(protocol_version_module, "CURRENT_PROTOCOL_VERSION", "0.2.9")
    monkeypatch.setattr(
        protocol_version_module,
        "PROTOCOL_FEATURE_MINIMUMS",
        MappingProxyType({PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE: "0.2.5"}),
    )
    with pytest.raises(ProtocolCompatibilityError) as caught:
        require_protocol_feature("0.2.4", PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE)
    error = caught.value
    assert error.required_protocol_version == "0.2.5"
    assert "requires protocol 0.2.5" in str(error)
    assert require_protocol_feature("0.2.5", PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE) is None
    assert require_protocol_feature("0.2.7", PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE) is None


def test_stable_major_uses_additive_minor_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pin a coherent stable-era state a later 1.x release installs (supported
    # major 1, the frozen artifact entry RETAINED and rewritten to 1.0.0 per
    # AR.1, current raised to the minor that added a feature -- a core never
    # knows a floor above its own version, per the production-constants
    # invariant), and prove the algebra that activates with it: every minor
    # in the stable major passes the line gate; a feature added in a later
    # minor is refused for an older engine by its minimum, not by the line
    # rule.
    monkeypatch.setattr(protocol_version_module, "SUPPORTED_PROTOCOL_MAJOR", 1)
    monkeypatch.setattr(protocol_version_module, "CURRENT_PROTOCOL_VERSION", "1.1.0")
    monkeypatch.setattr(
        protocol_version_module,
        "PROTOCOL_FEATURE_MINIMUMS",
        MappingProxyType(
            {
                # The frozen baseline entry is RETAINED and rewritten to
                # 1.0.0 (AR.1): the table is a public introspection surface,
                # so a known feature's compatibility query must not turn
                # into ValueError on freeze day.
                PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE: "1.0.0",
                "future_feature": "1.1.0",
            }
        ),
    )
    assert require_supported_protocol("1.0.0") is None
    assert require_protocol_feature("1.0.0", PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE) is None
    assert require_protocol_feature("1.4.2", PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE) is None
    assert require_supported_protocol("1.5.3") is None
    with pytest.raises(ProtocolCompatibilityError, match="supports protocol major 1"):
        require_supported_protocol("0.2.0")

    assert require_protocol_feature("1.2.0", "future_feature") is None
    with pytest.raises(ProtocolCompatibilityError, match="requires protocol 1.1.0"):
        require_protocol_feature("1.0.9", "future_feature")


def test_require_protocol_feature_wraps_malformed_version() -> None:
    with pytest.raises(ProtocolCompatibilityError, match="invalid protocol_version") as caught:
        require_protocol_feature("1.1", PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE)
    assert isinstance(caught.value.__cause__, ValueError)


def test_require_protocol_feature_rejects_unknown_feature_token() -> None:
    with pytest.raises(ValueError, match="Unknown protocol feature"):
        require_protocol_feature("0.2.0", "artifact_lifecyle")
