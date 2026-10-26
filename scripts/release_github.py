# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Check and create the GitHub tag, release, and dev-bump pull request.

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

Drafts: GitHub lists a draft release only to a token with write access to
the repository. The ``check`` command runs with read access, so it sees
published releases and tags; the ``ensure`` command runs with write access
and sees drafts as well. A stray draft therefore stops the run at the
release step, after the indexes hold the files, and a person publishes or
deletes the draft before the next run.

The dev-bump job, which runs after the release, uses two more commands:

* ``branch-head``: the commit a branch points at, or ``absent``.
* ``dev-bump-pr``: find or open the pull request that moves the default
  branch to the next development version, and assign it to the person who
  started the run (``dev_bump_pull_request`` says what it does with a
  merged or closed one).

It talks to GitHub through the ``gh`` command, which the runner has, with the
workflow's token. Only the standard library is used otherwise.
"""

from __future__ import annotations

import argparse
import base64
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


#: The repository setting that lets the workflow token open a pull request.
PULL_REQUEST_SETTING = (
    "Settings > Actions > General > Workflow permissions > "
    '"Allow GitHub Actions to create and approve pull requests"'
)


class PullRequestRefusedError(GitHubError):
    """GitHub refused to open a pull request for the workflow token."""


def _number(entry: dict[str, object]) -> int:
    value = entry.get("number")
    return value if isinstance(value, int) else 0


def _field(entry: object, *path: str) -> str:
    """Return the string at ``path`` in a GitHub object, or ``""``."""
    value = entry
    for key in path:
        value = cast("dict[str, object]", value).get(key) if isinstance(value, dict) else None
    return value if isinstance(value, str) else ""


def pull_requests_from(repo: str, branch: str, run: Runner = run_gh) -> list[dict[str, object]]:
    """Return the pull requests whose head is ``branch`` in ``repo`` itself.

    GitHub filters by ``owner:branch``, which leaves out a fork that another
    account owns but not one the same owner holds, so the function also
    keeps only the pull requests whose head repository is ``repo``.

    Args:
        repo: ``owner/name``.
        branch: The head branch name.
        run: The ``gh`` runner, replaceable in tests.

    Returns:
        The pull requests, open and closed, newest first, up to 100.

    Raises:
        GitHubError: When the lookup fails.
    """
    owner = repo.split("/", 1)[0]
    document = _api(repo, f"pulls?head={owner}:{branch}&state=all&per_page=100", run)
    if not isinstance(document, list):
        raise GitHubError(
            f"GitHub answered the pull requests from {branch} with something other than a list"
        )
    pulls = [
        cast("dict[str, object]", entry)
        for entry in cast("list[object]", document)
        if isinstance(entry, dict)
    ]
    own = repo.lower()
    return [pull for pull in pulls if _field(pull, "head", "repo", "full_name").lower() == own]


def _post(repo: str, path: str, fields: list[str], run: Runner) -> tuple[int, str, str]:
    args = ["api", f"repos/{repo}/{path}", "--method", "POST"]
    for field in fields:
        args += ["-f", field]
    return run(args)


def assign(repo: str, number: int, login: str, run: Runner = run_gh) -> None:
    """Assign a pull request to ``login``, and check that GitHub did.

    GitHub can leave out an assignee without reporting an error, so the
    function reads the assignees back from GitHub's answer.

    Args:
        repo: ``owner/name``.
        number: The pull request number.
        login: The account to assign.
        run: The ``gh`` runner, replaceable in tests.

    Raises:
        GitHubError: When the call fails, or the answer does not list
            ``login`` among the assignees.
    """
    code, out, err = _post(repo, f"issues/{number}/assignees", [f"assignees[]={login}"], run)
    if code != 0:
        raise GitHubError(
            f"Assigning pull request #{number} to {login} failed: {err.strip() or f'exit {code}'}"
        )
    document: object = json.loads(out or "{}")
    entries = (
        cast("dict[str, object]", document).get("assignees") if isinstance(document, dict) else None
    )
    logins = [
        str(cast("dict[str, object]", entry).get("login", "")).lower()
        for entry in (cast("list[object]", entries) if isinstance(entries, list) else [])
        if isinstance(entry, dict)
    ]
    if login.lower() not in logins:
        raise GitHubError(
            f"GitHub did not assign pull request #{number} to {login}: its answer does not "
            f"list {login} among the assignees, and GitHub can leave an assignee out without "
            "an error. Assign the pull request yourself."
        )


def dev_bump_pull_request(
    repo: str,
    branch: str,
    base: str,
    title: str,
    body: str,
    assignee: str,
    run: Runner = run_gh,
) -> tuple[int, str, bool]:
    """Find or open the pull request from ``branch``, and assign it.

    The newest pull request from the branch decides. An open one into
    ``base`` is kept and assigned again, so a second run opens nothing new.
    An open one into another branch, a merged one, and a closed one are
    errors: the dev-bump job calls this only while ``base`` still declares
    the released version, so a merged one means something set the version
    back or it merged elsewhere, and a closed one means a person closed it.
    When the branch has no pull request, the function opens one.

    Args:
        repo: ``owner/name``.
        branch: The head branch, in ``repo`` itself.
        base: The branch the pull request merges into.
        title: The title of a new pull request.
        body: The body of a new pull request.
        assignee: The account to assign.
        run: The ``gh`` runner, replaceable in tests.

    Returns:
        The pull request number, its URL, and whether this call opened it.

    Raises:
        PullRequestRefusedError: When GitHub refuses to open the pull request.
        GitHubError: When a lookup or the assignment fails, or the newest
            pull request from the branch is merged or closed.
    """
    pulls = pull_requests_from(repo, branch, run)
    if pulls:
        newest = max(pulls, key=_number)
        number = _number(newest)
        into = _field(newest, "base", "ref")
        if newest.get("merged_at"):
            cause = (
                "something set the version back"
                if into == base
                else f"the pull request merged into {into}, not {base}"
            )
            raise GitHubError(
                f"Pull request #{number} from {branch} was merged, yet {base} still declares "
                f"the released version: {cause}. Find out why before you run the workflow again."
            )
        if newest.get("state") != "open":
            raise GitHubError(
                f"Pull request #{number} from {branch} was closed without merging. This job "
                "does not open another, because a person closed it: reopen it and merge it, "
                f"or open a new one from {branch} yourself."
            )
        if into != base:
            raise GitHubError(
                f"Pull request #{number} from {branch} merges into {into}, not {base}. "
                f"Change its base branch to {base}, then run the workflow again."
            )
        url = str(newest.get("html_url", ""))
        created = False
    else:
        fields = [f"head={branch}", f"base={base}", f"title={title}", f"body={body}"]
        code, out, err = _post(repo, "pulls", fields, run)
        if code != 0:
            raise PullRequestRefusedError(
                f"GitHub refused to open a pull request from {branch}: "
                f"{err.strip() or f'exit {code}'}. When GitHub says that Actions may not "
                f"create pull requests, turn on {PULL_REQUEST_SETTING} and run the workflow "
                "again, or open the pull request yourself from the link that follows."
            )
        document: object = json.loads(out or "{}")
        opened = cast("dict[str, object]", document) if isinstance(document, dict) else {}
        number = _number(opened)
        url = str(opened.get("html_url", ""))
        if not number:
            raise GitHubError(
                f"GitHub answered the new pull request from {branch} without its number"
            )
        created = True
    assign(repo, number, assignee, run)
    return number, url, created


def declared_version(repo: str, ref: str, run: Runner = run_gh) -> str:
    """Return ``project.version`` from ``pyproject.toml`` at ``ref``.

    Args:
        repo: ``owner/name``.
        ref: A branch, a tag, or a commit.
        run: The ``gh`` runner, replaceable in tests.

    Returns:
        The version, as the file writes it.

    Raises:
        GitHubError: When the lookup fails, or the file or its version is
            missing or unreadable.
    """
    # The type-checking floor is Python 3.10, which has no tomllib; this
    # function runs on the release runner's python3.
    import tomllib  # pyright: ignore[reportMissingTypeStubs]  # noqa: PLC0415

    document = _object(repo, f"contents/pyproject.toml?ref={ref}", run)
    if document is None:
        raise GitHubError(f"{ref} has no pyproject.toml")
    content = document.get("content")
    if document.get("encoding") != "base64" or not isinstance(content, str):
        raise GitHubError(f"GitHub answered pyproject.toml at {ref} without base64 content")
    try:
        data: object = tomllib.loads(base64.b64decode(content).decode("utf-8"))
    except ValueError as error:
        raise GitHubError(f"pyproject.toml at {ref} cannot be read: {error}") from error
    version = _field(data, "project", "version")
    if not version:
        raise GitHubError(f"pyproject.toml at {ref} declares no project.version")
    return version


def dev_bump_body(version: str, next_version: str, base: str) -> str:
    """Return the body of the dev-bump pull request.

    Args:
        version: The version just released.
        next_version: The development version the pull request sets.
        base: The branch the pull request merges into.

    Returns:
        The Markdown body.
    """
    return (
        f"The release workflow opened this pull request after it released v{version}. It "
        f"sets the version on `{base}` to {next_version}. Until it merges, `{base}` declares "
        f"{version} while its content moves past that release, so an install from `{base}` "
        "reports the wrong version. After it merges, a run of the release workflow started "
        "by mistake stops at its first guard.\n\n"
        "GitHub holds the CI runs of a pull request that the workflow token opens. Select "
        "**Approve workflows to run** in the merge box, wait for `checks-complete` to pass, "
        "and merge."
    )


def open_dev_bump(
    repo: str,
    *,
    branch: str,
    base: str,
    version: str,
    next_version: str,
    assignee: str,
    compare_url: str,
    summary: Path | None,
    run: Runner = run_gh,
) -> int:
    """Check the dev-bump branch, find or open its pull request, and report it.

    The branch must declare ``next_version``: a branch that already held
    commits of its own is kept as it is, so this is where the job finds out
    whether those commits set the version.

    Args:
        repo: ``owner/name``.
        branch: The dev-bump branch.
        base: The default branch.
        version: The version just released.
        next_version: The development version the branch sets.
        assignee: The account that started the run.
        compare_url: The page where a person opens the pull request by hand,
            printed when GitHub refuses to open it.
        summary: The file GitHub shows as the run's summary page, or ``None``.
        run: The ``gh`` runner, replaceable in tests.

    Returns:
        The process exit code: 0 when the branch declares ``next_version`` and
        its pull request is open and assigned, 1 otherwise.
    """
    try:
        declared = declared_version(repo, branch, run)
    except GitHubError as error:
        print(f"::error::{error}")
        return 1
    if declared != next_version:
        print(
            f"::error::Branch {branch} declares version {declared}, not {next_version}. "
            "The job keeps commits a person added to the branch, so set the version there "
            f"to {next_version} (uv version {next_version}, then uv lock), or delete the "
            "branch so that the next run commits it, then run the workflow again."
        )
        return 1
    title = f"chore: bump version to {next_version}"
    body = dev_bump_body(version, next_version, base)
    try:
        number, url, created = dev_bump_pull_request(repo, branch, base, title, body, assignee, run)
    except PullRequestRefusedError as error:
        print(f"::error::{error}")
        print(f"::error title=Open the dev-bump pull request yourself::{compare_url}")
        return 1
    except GitHubError as error:
        print(f"::error::{error}")
        return 1
    found = "Opened pull request" if created else "Found the open pull request"
    print(f"{found} #{number} and assigned it to {assignee}: {url}")
    print(f"::notice title=Merge the dev-bump pull request::{url}")
    if summary is not None:
        with summary.open("a", encoding="utf-8") as file:
            file.write(
                "### Merge the dev-bump pull request\n\n"
                f"Pull request #{number} sets the version on `{base}` to {next_version}, and "
                f"it is assigned to @{assignee}. Open it; if GitHub holds its CI runs, select "
                "**Approve workflows to run**. Wait for `checks-complete` to pass, and merge "
                f"it: {url}\n"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one of the four commands; see the module docstring.

    Args:
        argv: The command line, without the program name.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Check and create the GitHub tag, release, and dev-bump pull request."
    )
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
    commands.add_parser("branch-head").add_argument("--branch", required=True)
    bump = commands.add_parser("dev-bump-pr")
    bump.add_argument("--branch", required=True)
    bump.add_argument("--base", required=True, help="the default branch")
    bump.add_argument("--version", required=True, help="the version just released")
    bump.add_argument("--next", required=True, help="the development version the branch sets")
    bump.add_argument("--assignee", required=True, help="the person who started the run")
    bump.add_argument("--compare-url", required=True, help="the page that opens it by hand")
    bump.add_argument("--summary", type=Path, help="the run's summary file")
    args = parser.parse_args(argv)
    if args.command == "dev-bump-pr":
        return open_dev_bump(
            args.repo,
            branch=args.branch,
            base=args.base,
            version=args.version,
            next_version=args.next,
            assignee=args.assignee,
            compare_url=args.compare_url,
            summary=args.summary,
            run=run_gh,
        )
    if args.command == "branch-head":
        try:
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
