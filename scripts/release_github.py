# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Check and create the GitHub tag and release for one version.

The release workflow (``.github/workflows/release.yml``) creates the tag and
the GitHub Release last, after PyPI holds the files, and it has to be safe to
run twice. GitHub ignores ``--target`` when the tag already exists, so a tag
left behind by an earlier attempt could put the release on the wrong commit;
and a release left behind could carry other notes. This script reads the
state first and refuses anything it cannot reconcile:

* ``check``: before anything is published. The tag, if it exists, points at
  the release commit; the release, if it exists, has its tag, carries the
  CHANGELOG section as its notes, and has the right pre-release flag.
  Anything else is an error a person has to look at. A lookup that fails for
  any reason other than "not found" is an error too: an answer the script
  cannot read is not an absence.
* ``ensure``: after PyPI is verified. The same checks, then the release is
  created when absent, and checked again afterwards.

Workflow files: GitHub lets a token create a tag at a commit whose files
under ``.github/workflows/`` differ from the default branch's only if the
token may change workflows, and the workflow token never may. When a workflow
change merged after the release commit and the tag does not exist yet,
``check`` warns before anything is published, and ``ensure`` stops with the
command a person runs instead, after PyPI holds the files. When the tag
exists, ``ensure`` creates the release without ``--target``, which GitHub
ignores for an existing tag.
* ``pr-state``: the state of the dev-bump pull request for a branch.
* ``branch-head``: the commit a branch points at, or ``absent``, for the
  dev-bump job.

Drafts: GitHub lists a draft release only to a token with write access to
the repository. The ``check`` command runs with read access, so it sees
published releases and tags; the ``ensure`` command runs with write access
and sees drafts as well. A stray draft therefore stops the run at the
release step, after the indexes hold the files, and a person publishes or
deletes the draft before the next run.

It talks to GitHub through the ``gh`` command, which the runner has, with the
workflow's token. Only the standard library is used otherwise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

#: Runs ``gh`` with the given arguments; returns the exit code, stdout, and
#: stderr.
Runner = Callable[[list[str]], tuple[int, str, str]]


class GitHubError(RuntimeError):
    """A ``gh`` call failed for a reason other than "not found"."""


def run_gh(args: list[str]) -> tuple[int, str, str]:
    """Run ``gh`` and return its exit code, stdout, and stderr.

    Args:
        args: The arguments after ``gh``.

    Returns:
        The exit code, the command's stdout, and its stderr.
    """
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    return result.returncode, result.stdout, result.stderr


def _api(repo: str, path: str, run: Runner) -> object | None:
    """Return the JSON a GitHub endpoint answers, or ``None`` on a 404.

    Args:
        repo: ``owner/name``.
        path: The endpoint path under ``repos/{repo}/``.
        run: The ``gh`` runner.

    Returns:
        The decoded JSON, or ``None`` when GitHub answers 404.

    Raises:
        GitHubError: When ``gh`` fails for any other reason.
    """
    code, out, err = run(["api", f"repos/{repo}/{path}"])
    if code == 0:
        return json.loads(out)
    if "HTTP 404" in err:
        return None
    raise GitHubError(f"gh api repos/{repo}/{path} failed: {err.strip() or f'exit {code}'}")


def _object(repo: str, path: str, run: Runner) -> dict[str, object] | None:
    document = _api(repo, path, run)
    if document is None:
        return None
    if not isinstance(document, dict):
        raise GitHubError(f"GitHub answered {path} with something other than an object")
    return cast("dict[str, object]", document)


def tag_commit(repo: str, tag: str, run: Runner = run_gh) -> str | None:
    """Return the commit a tag points at, or ``None`` when the tag is absent.

    An annotated tag is followed to its commit.

    Args:
        repo: ``owner/name``.
        tag: The tag name, such as ``v1.0.0``.
        run: The ``gh`` runner, replaceable in tests.

    Returns:
        The commit sha, or ``None``.

    Raises:
        GitHubError: When the lookup fails, or the tag points at something
            other than a commit.
    """
    ref = _object(repo, f"git/ref/tags/{tag}", run)
    if ref is None:
        return None
    target = ref.get("object")
    for _ in range(3):
        if not isinstance(target, dict):
            raise GitHubError(f"GitHub answered tag {tag} without an object")
        pointed = cast("dict[str, object]", target)
        kind = pointed.get("type")
        sha = pointed.get("sha")
        if not isinstance(sha, str):
            raise GitHubError(f"GitHub answered tag {tag} without a sha")
        if kind == "commit":
            return sha
        if kind != "tag":
            raise GitHubError(f"tag {tag} points at a {kind!r}, not a commit")
        annotated = _object(repo, f"git/tags/{sha}", run)
        if annotated is None:
            raise GitHubError(f"annotated tag {tag} ({sha}) cannot be read")
        target = annotated.get("object")
    raise GitHubError(f"tag {tag} is nested more than three levels deep")


def branch_head(repo: str, branch: str, run: Runner = run_gh) -> str | None:
    """Return the commit a branch points at, or ``None`` when the branch is absent.

    Args:
        repo: ``owner/name``.
        branch: The branch name, such as ``release/dev-bump-v1.0.0``.
        run: The ``gh`` runner, replaceable in tests.

    Returns:
        The commit sha, or ``None``.

    Raises:
        GitHubError: When the lookup fails.
    """
    ref = _object(repo, f"git/ref/heads/{branch}", run)
    if ref is None:
        return None
    target = ref.get("object")
    sha = cast("dict[str, object]", target).get("sha") if isinstance(target, dict) else None
    if not isinstance(sha, str):
        raise GitHubError(f"GitHub answered branch {branch} without a sha")
    return sha


def release(repo: str, tag: str, run: Runner = run_gh) -> dict[str, object] | None:
    """Return the GitHub Release for a tag, or ``None`` when there is none.

    The release list is searched, because the by-tag endpoint answers only
    published releases, and a draft has to be found too where the token can
    see it.

    Args:
        repo: ``owner/name``.
        tag: The tag name.
        run: The ``gh`` runner, replaceable in tests.

    Returns:
        The release object GitHub returns, or ``None``.

    Raises:
        GitHubError: When the lookup fails.
    """
    document = _api(repo, "releases?per_page=100", run)
    if document is None:
        return None
    if not isinstance(document, list):
        raise GitHubError("GitHub answered the release list with something other than a list")
    for entry in cast("list[object]", document):
        if isinstance(entry, dict):
            found = cast("dict[str, object]", entry)
            if found.get("tag_name") == tag:
                return found
    return None


def _normalized(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def problems(
    repo: str, version: str, sha: str, notes: str, prerelease: bool, run: Runner = run_gh
) -> tuple[list[str], bool]:
    """Compare the tag and the release with what this run would create.

    Args:
        repo: ``owner/name``.
        version: The version being released.
        sha: The release commit.
        notes: The release notes this run carries (the CHANGELOG section).
        prerelease: Whether the version is a pre-release.
        run: The ``gh`` runner, replaceable in tests.

    Returns:
        The problems found, and whether a release already exists.

    Raises:
        GitHubError: When a lookup fails.
    """
    tag = f"v{version}"
    found: list[str] = []
    commit = tag_commit(repo, tag, run)
    if commit is not None and commit != sha:
        found.append(
            f"tag {tag} points at {commit}, not at the release commit {sha}. A tag name is "
            "burned once used, so look at that tag before anything else."
        )
    existing = release(repo, tag, run)
    if existing is None:
        return found, False
    if existing.get("draft") is True:
        found.append(f"a draft release for {tag} exists; publish it or delete it by hand.")
    elif commit is None:
        found.append(f"release {tag} exists but its tag does not; look at the release by hand.")
    body = existing.get("body")
    if _normalized(body if isinstance(body, str) else "") != _normalized(notes):
        found.append(
            f"release {tag} exists with notes that differ from the CHANGELOG section at "
            f"{sha}; make the two agree by hand."
        )
    if bool(existing.get("prerelease")) != prerelease:
        found.append(
            f"release {tag} is marked prerelease={existing.get('prerelease')!r}, and this "
            f"version is {'a pre-release' if prerelease else 'not a pre-release'}."
        )
    return found, True


def manual_create_command(repo: str, version: str, sha: str, prerelease: bool) -> str:
    """Return the ``gh`` command that creates the release by hand.

    Args:
        repo: ``owner/name``.
        version: The version being released.
        sha: The release commit.
        prerelease: Whether the version is a pre-release.

    Returns:
        The command, reading the notes from ``release-notes.md``, the file in
        the workflow artifact ``standard-asr-release``.
    """
    tag = f"v{version}"
    command = f"gh release create {tag} --repo {repo} --target {sha} --title {tag}"
    command += " --notes-file release-notes.md"
    return command + (" --prerelease" if prerelease else "")


def check(
    repo: str,
    version: str,
    sha: str,
    notes_path: Path,
    prerelease: bool,
    run: Runner = run_gh,
    *,
    workflows_changed: bool = False,
) -> int:
    """Report the tag and release state; see the module docstring.

    Args:
        repo: ``owner/name``.
        version: The version being released.
        sha: The release commit.
        notes_path: The file with the release notes.
        prerelease: Whether the version is a pre-release.
        run: The ``gh`` runner, replaceable in tests.
        workflows_changed: Whether the workflow files at ``sha`` differ from
            the default branch's; with no tag yet, this prints a warning.

    Returns:
        The process exit code: 0 when the state can be completed, 1 otherwise.
    """
    try:
        found, exists = problems(repo, version, sha, notes_path.read_text("utf-8"), prerelease, run)
        tag_missing = (
            not exists and workflows_changed and tag_commit(repo, f"v{version}", run) is None
        )
    except GitHubError as error:
        print(f"::error::{error}")
        return 1
    for problem in found:
        print(f"::error::{problem}")
    if found:
        return 1
    if tag_missing:
        print(
            f"::warning::The workflow files at the release commit {sha} differ from the "
            "default branch's, so the workflow token cannot create the tag there. The run "
            "publishes to PyPI as usual; then the GitHub Release step stops and prints the "
            'command that creates the release by hand (RELEASING.md, "When a run fails").'
        )
    print(f"release v{version}: {'exists at the release commit' if exists else 'absent'}")
    return 0


def ensure(
    repo: str,
    version: str,
    sha: str,
    notes_path: Path,
    prerelease: bool,
    run: Runner = run_gh,
    *,
    workflows_changed: bool = False,
) -> int:
    """Create the release when absent, and check it either way.

    Args:
        repo: ``owner/name``.
        version: The version being released.
        sha: The release commit.
        notes_path: The file with the release notes.
        prerelease: Whether the version is a pre-release.
        run: The ``gh`` runner, replaceable in tests.
        workflows_changed: Whether the workflow files at ``sha`` differ from
            the default branch's. With no tag yet, the workflow token cannot
            create one there, so the function stops with the command a
            person runs instead.

    Returns:
        The process exit code: 0 when the release exists at the release
        commit with the right notes, 1 otherwise.
    """
    if check(repo, version, sha, notes_path, prerelease, run) != 0:
        return 1
    tag = f"v{version}"
    try:
        existing = release(repo, tag, run)
        tagged = existing is None and tag_commit(repo, tag, run) is not None
    except GitHubError as error:
        print(f"::error::{error}")
        return 1
    if existing is None and not tagged and workflows_changed:
        print(
            f"::error::The workflow files at the release commit {sha} differ from the "
            "default branch's, and GitHub lets a token create a tag at such a commit only "
            "if the token may change workflows, which the workflow token never may. PyPI "
            f"holds {tag}. Create the release yourself, with credentials that may change "
            "workflows, then run this workflow again to open the dev-bump pull request. "
            "release-notes.md is in the workflow artifact standard-asr-release:"
        )
        print(manual_create_command(repo, version, sha, prerelease))
        return 1
    if existing is None:
        # GitHub ignores --target for a tag that exists, and checks it for
        # workflow changes all the same, so pass it only for a new tag.
        args = ["release", "create", tag, "--repo", repo]
        args += [] if tagged else ["--target", sha]
        args += ["--title", tag, "--notes-file", str(notes_path)]
        if prerelease:
            args.append("--prerelease")
        code, out, err = run(args)
        if code != 0:
            print(f"::error::gh release create {tag} failed: {err.strip() or f'exit {code}'}")
            print("::error::Read the release state before retrying.")
            return 1
        print(f"created release {tag} at {sha}: {out.strip()}")
    return check(repo, version, sha, notes_path, prerelease, run)


def pr_state(repo: str, branch: str, run: Runner = run_gh) -> str:
    """Return the state of the pull request from ``branch``.

    Args:
        repo: ``owner/name``.
        branch: The head branch name.
        run: The ``gh`` runner, replaceable in tests.

    Returns:
        ``"absent"``, ``"open"``, ``"merged"``, or ``"closed"``; the newest
        pull request from the branch decides.

    Raises:
        GitHubError: When the lookup fails.
    """
    args = ["pr", "list", "--repo", repo, "--head", branch, "--state", "all"]
    code, out, err = run([*args, "--json", "number,state", "--limit", "20"])
    if code != 0:
        raise GitHubError(f"gh pr list failed for {branch}: {err.strip() or f'exit {code}'}")
    entries: object = json.loads(out or "[]")
    pulls = [
        cast("dict[str, object]", entry)
        for entry in (cast("list[object]", entries) if isinstance(entries, list) else [])
        if isinstance(entry, dict)
    ]
    if not pulls:
        return "absent"

    def number(entry: dict[str, object]) -> int:
        value = entry.get("number", 0)
        return value if isinstance(value, int) else 0

    newest = max(pulls, key=number)
    state = str(newest.get("state", "")).lower()
    return "merged" if state == "merged" else "open" if state == "open" else "closed"


def main(argv: list[str] | None = None) -> int:
    """Run one of the four commands; see the module docstring.

    Args:
        argv: The command line, without the program name.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description="Check and create the GitHub tag and release.")
    parser.add_argument("--repo", required=True, help="owner/name")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "ensure"):
        sub = commands.add_parser(name)
        sub.add_argument("--version", required=True)
        sub.add_argument("--sha", required=True, help="the release commit")
        sub.add_argument("--notes", type=Path, required=True, help="the release notes file")
        sub.add_argument("--prerelease", choices=("true", "false"), required=True)
        sub.add_argument(
            "--workflows-changed",
            choices=("true", "false"),
            default="false",
            help="whether the release commit's workflow files differ from the default branch's",
        )
    for name in ("pr-state", "branch-head"):
        commands.add_parser(name).add_argument("--branch", required=True)
    args = parser.parse_args(argv)
    if args.command in ("pr-state", "branch-head"):
        try:
            if args.command == "pr-state":
                print(pr_state(args.repo, args.branch, run_gh))
            else:
                print(branch_head(args.repo, args.branch, run_gh) or "absent")
        except GitHubError as error:
            print(f"::error::{error}")
            return 1
        return 0
    command = check if args.command == "check" else ensure
    return command(
        args.repo,
        args.version,
        args.sha,
        args.notes,
        args.prerelease == "true",
        run_gh,
        workflows_changed=args.workflows_changed == "true",
    )


if __name__ == "__main__":
    sys.exit(main())
