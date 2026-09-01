# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Protocol-version parsing and feature compatibility checks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, cast

from standard_asr.contract.exceptions import ProtocolCompatibilityError

CURRENT_PROTOCOL_VERSION: Final = "0.2.0"
"""Protocol version implemented by this core contract.

Protocol major 0 is the pre-stable line: each ``0.MINOR`` generation may
change the contract incompatibly, so the minor is the breaking axis and the
core supports exactly one line at a time. The first stable release promotes
the then-current contract verbatim to ``1.0.0``; from protocol major 1 on,
a higher minor within the supported major is an additive, compatible change.
The protocol version and the core package version are independent lines:
one names the contract generation an engine implements, the other names a
release of this implementation.
"""

SUPPORTED_PROTOCOL_MAJOR: Final = 0
"""Protocol major version that this core understands."""

PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE: Final = "artifact_lifecycle"
"""Feature token for inference-artifact status and acquisition."""

PROTOCOL_FEATURE_MINIMUMS: Mapping[str, str] = MappingProxyType(
    {PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE: "0.2.0"}
)
"""Minimum version, within the supported major, for each guarded feature.

Not a historical record: the first stable release rewrites the frozen
baseline entries to ``1.0.0`` (AR.1), so an entry names the floor an engine
must declare under THIS core's supported major, not the version that first
defined the feature.
"""

_PROTOCOL_VERSION_PATTERN = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
#: Upper bound for each version component. The protocol version is a
#: cross-language wire token, and common JSON implementations cannot compare
#: integers beyond 2**53 exactly; a 32-bit unsigned bound keeps every
#: component exactly representable and comparable in every implementation
#: language without arbitrary-precision arithmetic.
MAX_PROTOCOL_VERSION_COMPONENT: Final = 4_294_967_295

#: Length of the largest canonical token (``4294967295.4294967295.4294967295``,
#: 32 characters): any longer value breaks the syntax or the component bound,
#: so the parser can refuse it before running the grammar or converting digits.
_MAX_PROTOCOL_VERSION_LENGTH: Final = len(".".join([str(MAX_PROTOCOL_VERSION_COMPONENT)] * 3))


def parse_protocol_version(protocol_version: str) -> tuple[int, int, int]:
    """Parse a canonical ``MAJOR.MINOR.PATCH`` protocol version.

    Args:
        protocol_version: Protocol version to parse.

    Returns:
        A ``(major, minor, patch)`` integer tuple.

    Raises:
        ValueError: If the value is not canonical ``MAJOR.MINOR.PATCH``
            syntax, is longer than the largest canonical token, or a
            component exceeds :data:`MAX_PROTOCOL_VERSION_COMPONENT`.
    """
    if len(protocol_version) > _MAX_PROTOCOL_VERSION_LENGTH:
        # A token this long may still match the grammar (a 33-digit major
        # does), so the reason names the bound it certainly breaks rather
        # than blaming the syntax.
        raise ValueError(
            f"protocol_version is longer than {_MAX_PROTOCOL_VERSION_LENGTH} characters, "
            "the length of the largest canonical MAJOR.MINOR.PATCH whose components "
            f"stay within {MAX_PROTOCOL_VERSION_COMPONENT}."
        )
    match = _PROTOCOL_VERSION_PATTERN.fullmatch(protocol_version)
    if match is None:
        raise ValueError("protocol_version must use canonical MAJOR.MINOR.PATCH syntax.")
    parsed = tuple(int(component) for component in match.groups())
    if any(component > MAX_PROTOCOL_VERSION_COMPONENT for component in parsed):
        raise ValueError(
            "protocol_version components must not exceed "
            f"{MAX_PROTOCOL_VERSION_COMPONENT} (the 32-bit cross-language bound)."
        )
    return cast("tuple[int, int, int]", parsed)


def _remediation(declared: tuple[int, int, int], current: tuple[int, int, int]) -> str:
    """Return the direction-aware remediation sentence for a version mismatch.

    Args:
        declared: Parsed version the engine declares.
        current: Parsed version this core implements.

    Returns:
        An actionable instruction matching the mismatch direction.
    """
    if declared[:2] > current[:2]:
        return (
            "Upgrade the core, or install a plugin release that declares "
            f"protocol {current[0]}.{current[1]}."
        )
    return "Upgrade the plugin."


def require_supported_protocol(protocol_version: str) -> None:
    """Require an engine's declared protocol line to be usable by this core.

    The general version gate, independent of any feature: a different major
    is incompatible because contracts can change meaning across a
    major-version boundary, and within protocol major 0 the minor itself is
    the breaking axis -- each pre-stable ``0.MINOR`` generation may change
    the contract incompatibly -- so an engine must declare the core's exact
    ``0.MINOR`` line. The major-0 rule is permanent: it goes dormant on its
    own once the first stable release promotes the contract to major 1, and
    within a stable major every minor passes this gate (additive evolution
    is the feature table's job, see :func:`require_protocol_feature`).

    Args:
        protocol_version: Protocol version declared by the engine.

    Returns:
        None.

    Raises:
        ProtocolCompatibilityError: If the version is malformed, has an
            unsupported major, or is outside the supported pre-stable line.
    """
    try:
        declared = parse_protocol_version(protocol_version)
    except (TypeError, ValueError) as exc:
        raise ProtocolCompatibilityError(
            "The engine declares an invalid protocol_version. Expected "
            "canonical MAJOR.MINOR.PATCH syntax.",
            protocol_version=protocol_version,
        ) from exc

    current = parse_protocol_version(CURRENT_PROTOCOL_VERSION)
    if declared[0] != SUPPORTED_PROTOCOL_MAJOR:
        raise ProtocolCompatibilityError(
            f"Engine protocol {protocol_version!r} cannot be used by a core "
            f"that supports protocol major {SUPPORTED_PROTOCOL_MAJOR} (this "
            f"core implements {CURRENT_PROTOCOL_VERSION}). " + _remediation(declared, current),
            protocol_version=protocol_version,
        )
    if declared[0] == 0 and declared[:2] != current[:2]:
        raise ProtocolCompatibilityError(
            f"Engine protocol {protocol_version!r} is outside the supported "
            f"pre-stable line {current[0]}.{current[1]}. Within protocol "
            "major 0 each minor may change the contract incompatibly, so the "
            "core supports exactly one 0.MINOR line. " + _remediation(declared, current),
            protocol_version=protocol_version,
        )


def require_protocol_feature(protocol_version: str, feature: str) -> None:
    """Require an engine protocol version to define a core feature.

    Validates the caller's ``feature`` token first (an unknown token is a
    code bug and fails fast as ``ValueError``), then runs the general line
    gate (:func:`require_supported_protocol`), then the feature minimum:
    within a stable major, a later minor remains usable for an older
    feature, so the minimum -- not the line gate -- is what refuses an
    engine that predates the feature.

    Args:
        protocol_version: Protocol version declared by the engine.
        feature: Stable feature token from :data:`PROTOCOL_FEATURE_MINIMUMS`.

    Returns:
        None.

    Raises:
        ProtocolCompatibilityError: If the version is malformed, has an
            unsupported major, is outside the supported pre-stable line, or
            predates the requested feature.
        ValueError: If ``feature`` is not a known core feature token.
    """
    try:
        required_version = PROTOCOL_FEATURE_MINIMUMS[feature]
    except KeyError as exc:
        raise ValueError(f"Unknown protocol feature {feature!r}.") from exc

    require_supported_protocol(protocol_version)
    declared = parse_protocol_version(protocol_version)
    required = parse_protocol_version(required_version)
    if declared < required:
        raise ProtocolCompatibilityError(
            f"Feature {feature!r} requires protocol {required_version} or later "
            f"within major {SUPPORTED_PROTOCOL_MAJOR}; the engine declares "
            f"{protocol_version}. Upgrade the plugin to a release that "
            f"implements protocol {required_version} or later.",
            protocol_version=protocol_version,
            feature=feature,
            required_protocol_version=required_version,
        )


__all__ = [
    "CURRENT_PROTOCOL_VERSION",
    "MAX_PROTOCOL_VERSION_COMPONENT",
    "PROTOCOL_FEATURE_ARTIFACT_LIFECYCLE",
    "PROTOCOL_FEATURE_MINIMUMS",
    "SUPPORTED_PROTOCOL_MAJOR",
    "parse_protocol_version",
    "require_protocol_feature",
    "require_supported_protocol",
]
