# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Find the release commit for a version, and the build recipe it carries.

The release workflow (``.github/workflows/release.yml``) releases the commit
that set ``project.version``, not the tip of the branch. The design case
``docs/internal/feat_plan/release-pipeline-redesign.md`` says why: later merges
belong to the next version, and a second run of the same version has to
rebuild the same commit so that it produces the same files.

The release commit is the newest commit on the branch's first-parent history
whose ``project.version`` is the version, while its first parent's is not.
First-parent history is the sequence of merges and squashed commits on the
branch itself, so a merge of the release pull request counts as one commit;
a ``git log -S`` string search would instead find a commit inside the pull
request, or a comment that happens to contain the string. ``pyproject.toml``
is parsed as TOML, so a comment, a quoting style, or another table never
changes the answer.

The build recipe is the exact uv version the release uses, read from the
release commit itself: the lower bound of the ``uv-build`` requirement in
``[build-system]``. A workflow change after the release commit therefore
cannot change what a resumed run builds.

Only the standard library is used, so the workflow runs it with the runner's
``python3`` (3.11 or later, for ``tomllib``) and nothing to install.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable
from typing import cast

# The project's type-checking floor is Python 3.10, where tomllib is not in
# the standard library; this script runs on the release runner's python3.
import tomllib  # pyright: ignore[reportMissingTypeStubs]

#: Runs ``git`` with the given arguments and returns its stdout, or ``None``
#: when git exits non-zero (an absent object, a root commit's parent).
Git = Callable[[list[str]], "str | None"]


class IncompleteHistoryError(RuntimeError):
    """The checkout is shallow, so a first-parent walk would stop early."""


_UV_BUILD_FLOOR = re.compile(
    r"^uv[-_]build\s*(?:\[[^\]]*\])?\s*>=\s*(?P<floor>[0-9][0-9A-Za-z.!+-]*)"
)


def run_git(args: list[str]) -> str | None:
    """Run ``git`` and return its stdout, or ``None`` when it fails.

    Args:
        args: The arguments after ``git``.

    Returns:
        The command's stdout, or ``None`` on a non-zero exit.
    """
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout if result.returncode == 0 else None


def _table(pyproject: str, name: str) -> dict[str, object] | None:
    try:
        data = tomllib.loads(pyproject)
    except tomllib.TOMLDecodeError:
        return None
    table = data.get(name)
    return cast("dict[str, object]", table) if isinstance(table, dict) else None


def project_version(pyproject: str) -> str | None:
    """Return ``project.version`` from a ``pyproject.toml`` text, or ``None``.

    Args:
        pyproject: The file's text.

    Returns:
        The version string, or ``None`` when the file does not parse or the
        table or the key is absent or not a string.
    """
    project = _table(pyproject, "project")
    version = project.get("version") if project else None
    return version if isinstance(version, str) else None


def uv_build_floor(pyproject: str) -> str | None:
    """Return the lower bound of the ``uv-build`` requirement, or ``None``.

    Args:
        pyproject: The file's text.

    Returns:
        The version after ``uv-build>=`` in ``[build-system].requires``, or
        ``None`` when there is no such requirement or no lower bound.
    """
    build_system = _table(pyproject, "build-system")
    requires = build_system.get("requires") if build_system else None
    if not isinstance(requires, list):
        return None
    for requirement in cast("list[object]", requires):
        if isinstance(requirement, str):
            match = _UV_BUILD_FLOOR.match(requirement.strip())
            if match:
                return match.group("floor")
    return None


def version_at(commit: str, git: Git = run_git) -> str | None:
    """Return ``project.version`` at ``commit``, or ``None`` when unreadable.

    Args:
        commit: A commit reference.
        git: The git runner, replaceable in tests.

    Returns:
        The version, or ``None`` when the file or the commit does not exist.
    """
    text = git(["show", f"{commit}:pyproject.toml"])
    return project_version(text) if text is not None else None


def find_release_commit(version: str, head: str = "HEAD", git: Git = run_git) -> str | None:
    """Return the commit that set ``project.version`` to ``version``.

    Args:
        version: The version being released.
        head: The branch tip to walk back from.
        git: The git runner, replaceable in tests.

    Returns:
        The newest first-parent commit whose version is ``version`` while its
        first parent's is not, or ``None`` when no such commit exists.

    Raises:
        IncompleteHistoryError: When the checkout is shallow. The walk would
            end at the fetch boundary and take the wrong commit.
    """
    shallow = git(["rev-parse", "--is-shallow-repository"])
    if shallow is None or shallow.strip() != "false":
        raise IncompleteHistoryError(
            "The checkout is shallow, so the first-parent history ends at the fetch "
            "boundary and the walk would take the wrong commit. Check out with "
            "fetch-depth: 0."
        )
    log = git(["log", "--first-parent", "--format=%H", head, "--", "pyproject.toml"])
    if not log:
        return None
    for commit in log.split():
        if version_at(commit, git) != version:
            continue
        if version_at(f"{commit}^", git) != version:
            return commit
    return None


#: The directory GitHub guards: a token may create a tag at a commit whose
#: files here differ from the default branch's only if it may change workflows.
WORKFLOWS_DIRECTORY = ".github/workflows/"


def workflows_changed(sha: str, ref: str, git: Git = run_git) -> bool:
    """Return whether the workflow files at ``sha`` differ from those at ``ref``.

    GitHub lets a token create a tag or a release at a commit whose files
    under ``.github/workflows/`` differ from the default branch's only if
    the token may change workflows, and the workflow token never may (GitHub
    REST reference, "Create a release"). The release commit can fall behind
    that way whenever a workflow change merges after it.

    Args:
        sha: The release commit.
        ref: The default branch's tip, such as ``HEAD`` or
            ``origin/main``.
        git: The git runner, replaceable in tests.

    Returns:
        ``True`` when any file under ``.github/workflows/`` differs.

    Raises:
        RuntimeError: When git cannot compare the two commits.
    """
    changed = git(["diff", "--name-only", sha, ref, "--", WORKFLOWS_DIRECTORY])
    if changed is None:
        raise RuntimeError(f"git cannot compare {sha} with {ref}.")
    return bool(changed.strip())


def main(argv: list[str] | None = None) -> int:
    """Print the release commit and its build recipe for the workflow.

    Args:
        argv: The command line, without the program name.

    Returns:
        The process exit code: 0 with ``sha=`` and ``uv=`` lines on stdout,
        and a ``workflows_changed=`` line when ``--workflows-against`` is
        given; 1 with an error on stderr.
    """
    parser = argparse.ArgumentParser(description="Find the commit that set project.version.")
    parser.add_argument("--version", required=True, help="the version being released")
    parser.add_argument("--head", default="HEAD", help="the branch tip to walk back from")
    parser.add_argument(
        "--workflows-against",
        metavar="REF",
        help="also report whether the workflow files at the release commit differ from REF",
    )
    args = parser.parse_args(argv)
    # Pass the runner at call time, so a test can replace ``run_git``.
    try:
        sha = find_release_commit(args.version, args.head, run_git)
    except IncompleteHistoryError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    if sha is None:
        print(
            f"::error::No commit on the first-parent history of {args.head} sets "
            f'project.version to "{args.version}" in pyproject.toml.',
            file=sys.stderr,
        )
        return 1
    uv = uv_build_floor(run_git(["show", f"{sha}:pyproject.toml"]) or "")
    if uv is None:
        print(
            f"::error::pyproject.toml at {sha} names no uv-build lower bound in "
            "[build-system].requires, so the build recipe is unknown.",
            file=sys.stderr,
        )
        return 1
    changed: bool | None = None
    if args.workflows_against is not None:
        try:
            changed = workflows_changed(sha, args.workflows_against, run_git)
        except RuntimeError as error:
            print(f"::error::{error}", file=sys.stderr)
            return 1
    print(f"sha={sha}")
    print(f"uv={uv}")
    if changed is not None:
        print(f"workflows_changed={'true' if changed else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
