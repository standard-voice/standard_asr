# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Classify the version being released, and name the next development version.

The release workflow (``.github/workflows/release.yml``) reads
``project.version`` from the branch it runs on. Two questions about that
version need PEP 440 arithmetic, which the ``packaging`` library does and the
standard library does not, so the workflow runs this script with
``uv run --no-project --with packaging``:

* ``classify``: can the version be released, and is it a pre-release? A
  development version such as ``0.2.1.dev0`` cannot: between releases the
  branch declares one, so an accidental run stops here with nothing public.
  A version outside the project's policy (``X.Y.Z`` with an optional ``a``,
  ``b``, or ``rc`` suffix; ``RELEASING.md``, "Versioning policy") cannot
  either. The pre-release answer sets the GitHub Release's pre-release flag.
* ``next``: the development version the dev-bump pull request sets after a
  release. It sorts after the released version and before every version the
  next release pull request can choose: after ``0.2.0`` it is ``0.2.1.dev0``,
  and after ``0.2.0rc1`` it is ``0.2.0rc2.dev0``, so the branch's version
  does not go backwards through a pre-release series.
"""

from __future__ import annotations

import argparse
import sys

from packaging.version import InvalidVersion, Version


class ReleaseVersionError(ValueError):
    """The version on the branch cannot be released; the message says why."""


def releasable(version: str) -> Version:
    """Parse ``version`` and refuse one the workflow must not release.

    Args:
        version: ``project.version`` as the branch declares it.

    Returns:
        The parsed version.

    Raises:
        ReleaseVersionError: When the version is not a PEP 440 version, is a
            development version, or is outside the versioning policy.
    """
    try:
        parsed = Version(version)
    except InvalidVersion as error:
        raise ReleaseVersionError(f"{version!r} is not a PEP 440 version.") from error
    if parsed.is_devrelease:
        raise ReleaseVersionError(
            f"The version on the branch is {parsed}, a development version. Between "
            "releases that is the branch's normal state: merge the release pull request "
            "that sets the version, then run this workflow."
        )
    if parsed.epoch or parsed.local or parsed.post is not None or len(parsed.release) != 3:
        raise ReleaseVersionError(
            f"The version on the branch is {parsed}, which is outside the versioning "
            "policy: X.Y.Z with an optional a, b, or rc suffix (RELEASING.md)."
        )
    return parsed


def next_development_version(version: str) -> str:
    """Return the development version that follows a released ``version``.

    Args:
        version: The version just released.

    Returns:
        ``X.Y.(Z+1).dev0`` after a final release, or the next pre-release
        number as a development version after a pre-release.

    Raises:
        ReleaseVersionError: When ``version`` could not have been released.
    """
    released = releasable(version)
    major, minor, patch = released.release
    if released.pre is not None:
        label, number = released.pre
        return f"{major}.{minor}.{patch}{label}{number + 1}.dev0"
    return f"{major}.{minor}.{patch + 1}.dev0"


def main(argv: list[str] | None = None) -> int:
    """Run ``classify`` or ``next``; see the module docstring.

    Args:
        argv: The command line, without the program name.

    Returns:
        The process exit code: 0 with the answer on stdout, 1 with a
        ``::error::`` line on stderr.
    """
    parser = argparse.ArgumentParser(description="Classify a release version.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("classify", "next"):
        commands.add_parser(name).add_argument("version")
    args = parser.parse_args(argv)
    try:
        if args.command == "classify":
            print("true" if releasable(args.version).is_prerelease else "false")
        else:
            print(next_development_version(args.version))
    except ReleaseVersionError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
