# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Protocol-version parsing and feature compatibility checks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from standard_asr.contract.exceptions import ProtocolCompatibilityError

CURRENT_PROTOCOL_VERSION: Final = "1.1.0"
"""Protocol version implemented by this core contract."""

SUPPORTED_PROTOCOL_MAJOR: Final = 1
"""Protocol major version that this core understands."""

PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE: Final = "artifact_lifecycle"
"""Feature token for inference-artifact status and acquisition."""

PROTOCOL_FEATURE_MINIMUMS: Mapping[str, str] = MappingProxyType(
    {PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE: "1.1.0"}
)
"""Earliest protocol version that defines each guarded feature."""

_PROTOCOL_VERSION_PATTERN = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_MAX_PROTOCOL_VERSION_LENGTH = 32


def parse_protocol_version(protocol_version: str) -> tuple[int, int, int]:
    """Parse a canonical ``MAJOR.MINOR.PATCH`` protocol version.

    Args:
        protocol_version: Protocol version to parse.

    Returns:
        A ``(major, minor, patch)`` integer tuple.

    Raises:
        ValueError: If the value is not canonical ``MAJOR.MINOR.PATCH`` syntax.
    """
    if len(protocol_version) > _MAX_PROTOCOL_VERSION_LENGTH:
        raise ValueError("protocol_version must use canonical MAJOR.MINOR.PATCH syntax.")
    match = _PROTOCOL_VERSION_PATTERN.fullmatch(protocol_version)
    if match is None:
        raise ValueError("protocol_version must use canonical MAJOR.MINOR.PATCH syntax.")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def require_protocol_feature(protocol_version: str, feature: str) -> None:
    """Require an engine protocol version to define a core feature.

    A later minor version in the supported major remains usable for an older
    feature. A different major is incompatible because the feature can change
    meaning across a major-version boundary.

    Args:
        protocol_version: Protocol version declared by the engine.
        feature: Stable feature token from :data:`PROTOCOL_FEATURE_MINIMUMS`.

    Returns:
        None.

    Raises:
        ProtocolCompatibilityError: If the version is malformed, has an
            unsupported major, or predates the requested feature.
        ValueError: If ``feature`` is not a known core feature token.
    """
    try:
        required_version = PROTOCOL_FEATURE_MINIMUMS[feature]
    except KeyError as exc:
        raise ValueError(f"Unknown protocol feature {feature!r}.") from exc

    try:
        declared = parse_protocol_version(protocol_version)
    except (TypeError, ValueError) as exc:
        raise ProtocolCompatibilityError(
            "The engine declares an invalid protocol_version. Expected "
            "canonical MAJOR.MINOR.PATCH syntax.",
            protocol_version=protocol_version,
            feature=feature,
            required_protocol_version=required_version,
        ) from exc

    required = parse_protocol_version(required_version)
    if declared[0] != SUPPORTED_PROTOCOL_MAJOR:
        raise ProtocolCompatibilityError(
            f"Engine protocol {protocol_version!r} cannot provide feature "
            f"{feature!r} to a core that supports protocol major "
            f"{SUPPORTED_PROTOCOL_MAJOR}.",
            protocol_version=protocol_version,
            feature=feature,
            required_protocol_version=required_version,
        )
    if declared < required:
        raise ProtocolCompatibilityError(
            f"Feature {feature!r} requires protocol {required_version} or later "
            f"within major {SUPPORTED_PROTOCOL_MAJOR}; the engine declares "
            f"{protocol_version}.",
            protocol_version=protocol_version,
            feature=feature,
            required_protocol_version=required_version,
        )


__all__ = [
    "CURRENT_PROTOCOL_VERSION",
    "PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE",
    "PROTOCOL_FEATURE_MINIMUMS",
    "SUPPORTED_PROTOCOL_MAJOR",
    "parse_protocol_version",
    "require_protocol_feature",
]
