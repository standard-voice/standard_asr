# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""The release workflow's GitHub tag, release, and pull request checks."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_github.py"

Runner = Callable[[list[str]], tuple[int, str, str]]
Answers = dict[str, object]
REPO = "standard-voice/standard_asr"
SHA = "a" * 40
OTHER = "b" * 40
NOTES = "### Added\n\n- A thing.\n"
NOT_FOUND = "gh: Not Found (HTTP 404)\n"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_github", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


release_github = _load()


@pytest.fixture(autouse=True)
def _no_real_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    # The script's functions bind the real ``run_gh`` as their default
    # ``run`` when the module loads, so replacing the module attribute does
    # not reach a call that leaves ``run`` out. Such a call would reach
    # GitHub; this fixture makes it fail instead.
    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"a test ran a real command: {args}")

    monkeypatch.setattr(release_github.subprocess, "run", refuse)


def _runner(answers: Answers, created: list[list[str]] | None = None) -> Runner:
    # ``answers`` maps a suffix of the ``gh api`` path to the JSON to answer
    # with, to None for a 404, or to the string "fail" for another failure;
    # the ``releases?per_page=100`` key holds the release list.
    def run(args: list[str]) -> tuple[int, str, str]:
        if args[0] == "api":
            for suffix, answer in answers.items():
                if args[1].endswith(suffix):
                    if answer is None:
                        return 1, "", NOT_FOUND
                    if answer == "fail":
                        return 1, "", "gh: Forbidden (HTTP 403)\n"
                    return 0, json.dumps(answer), ""
            raise AssertionError(f"unexpected api call {args[1]}")
        if args[:2] == ["release", "create"]:
            assert created is not None
            created.append(args)
            releases = answers.setdefault("releases?per_page=100", [])
            assert isinstance(releases, list)
            cast("list[object]", releases).append(
                {
                    "tag_name": args[2],
                    "draft": False,
                    "prerelease": "--prerelease" in args,
                    "body": NOTES,
                }
            )
            answers[f"git/ref/tags/{args[2]}"] = {"object": {"type": "commit", "sha": SHA}}
            return 0, f"https://github.com/{REPO}/releases/tag/{args[2]}\n", ""
        raise AssertionError(f"unexpected gh call {args}")

    return run


def _notes(tmp_path: Path) -> Path:
    notes = tmp_path / "notes.md"
    notes.write_text(NOTES, encoding="utf-8")
    return notes


def _nothing() -> Answers:
    return {"git/ref/tags/v1.0.0": None, "releases?per_page=100": []}


def test_tag_commit_follows_an_annotated_tag() -> None:
    run = _runner(
        {
            "git/ref/tags/v1.0.0": {"object": {"type": "tag", "sha": "t" * 40}},
            "git/tags/" + "t" * 40: {"object": {"type": "commit", "sha": SHA}},
            "git/ref/tags/v9.9.9": None,
        }
    )
    assert release_github.tag_commit(REPO, "v1.0.0", run) == SHA
    assert release_github.tag_commit(REPO, "v9.9.9", run) is None


def test_tag_commit_rejects_a_tag_that_points_at_a_tree() -> None:
    run = _runner({"git/ref/tags/v1.0.0": {"object": {"type": "tree", "sha": "x" * 40}}})
    with pytest.raises(release_github.GitHubError, match="not a commit"):
        release_github.tag_commit(REPO, "v1.0.0", run)


def test_a_lookup_failure_is_not_an_absence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A 403, a rate limit, or a network failure must stop the run: an
    # answer the script cannot read says nothing about the tag.
    with pytest.raises(release_github.GitHubError, match="HTTP 403"):
        release_github.tag_commit(REPO, "v1.0.0", _runner({"git/ref/tags/v1.0.0": "fail"}))
    run = _runner({"git/ref/tags/v1.0.0": None, "releases?per_page=100": "fail"})
    assert release_github.check(REPO, "1.0.0", SHA, _notes(tmp_path), False, run) == 1
    assert "::error::gh api repos/standard-voice/standard_asr/releases" in capsys.readouterr().out


def test_check_passes_when_nothing_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        release_github.check(REPO, "1.0.0", SHA, _notes(tmp_path), False, _runner(_nothing())) == 0
    )
    assert "release v1.0.0: absent" in capsys.readouterr().out


def test_check_fails_when_the_tag_points_elsewhere(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # GitHub keeps the existing tag and ignores --target, so a release
    # created now would sit on the wrong commit: stop before publishing.
    answers = _nothing()
    answers["git/ref/tags/v1.0.0"] = {"object": {"type": "commit", "sha": OTHER}}
    assert release_github.check(REPO, "1.0.0", SHA, _notes(tmp_path), False, _runner(answers)) == 1
    out = capsys.readouterr().out
    assert f"::error::tag v1.0.0 points at {OTHER}, not at the release commit {SHA}" in out


def test_check_compares_an_existing_release(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = _notes(tmp_path)
    same: Answers = {
        "git/ref/tags/v1.0.0": {"object": {"type": "commit", "sha": SHA}},
        "releases?per_page=100": [
            {"tag_name": "v0.9.0", "draft": False, "prerelease": False, "body": "old"},
            {
                "tag_name": "v1.0.0",
                "draft": False,
                "prerelease": False,
                "body": "\n" + NOTES + "\n",
            },
        ],
    }
    # Trailing whitespace and blank edges do not count as a difference.
    assert release_github.check(REPO, "1.0.0", SHA, notes, False, _runner(same)) == 0
    assert "exists at the release commit" in capsys.readouterr().out
    different: Answers = dict(same)
    different["releases?per_page=100"] = [
        {"tag_name": "v1.0.0", "draft": True, "prerelease": True, "body": "other"}
    ]
    assert release_github.check(REPO, "1.0.0", SHA, notes, False, _runner(different)) == 1
    out = capsys.readouterr().out
    assert "a draft release for v1.0.0 exists" in out
    assert "notes that differ from the CHANGELOG section" in out
    assert "is marked prerelease=True" in out


def test_check_fails_when_the_release_exists_without_its_tag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers: Answers = {
        "git/ref/tags/v1.0.0": None,
        "releases?per_page=100": [
            {"tag_name": "v1.0.0", "draft": False, "prerelease": False, "body": NOTES}
        ],
    }
    assert release_github.check(REPO, "1.0.0", SHA, _notes(tmp_path), False, _runner(answers)) == 1
    assert "exists but its tag does not" in capsys.readouterr().out


def test_ensure_creates_the_release_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = _notes(tmp_path)
    created: list[list[str]] = []
    run = _runner({"git/ref/tags/v1.0.0rc1": None, "releases?per_page=100": []}, created)
    assert release_github.ensure(REPO, "1.0.0rc1", SHA, notes, True, run) == 0
    assert len(created) == 1
    assert created[0][2:] == [
        "v1.0.0rc1",
        "--repo",
        REPO,
        "--target",
        SHA,
        "--title",
        "v1.0.0rc1",
        "--notes-file",
        str(notes),
        "--prerelease",
    ]
    # The second run finds the release and creates nothing.
    assert release_github.ensure(REPO, "1.0.0rc1", SHA, notes, True, run) == 0
    assert len(created) == 1
    out = capsys.readouterr().out
    assert "created release v1.0.0rc1" in out
    # The check after the creation, then the two checks of the second run.
    assert out.count("exists at the release commit") == 3


def test_ensure_stops_on_a_problem_and_on_a_failed_create(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = _notes(tmp_path)
    answers = _nothing()
    answers["git/ref/tags/v1.0.0"] = {"object": {"type": "commit", "sha": OTHER}}
    assert release_github.ensure(REPO, "1.0.0", SHA, notes, False, _runner(answers)) == 1

    def failing_create(args: list[str]) -> tuple[int, str, str]:
        if args[0] == "api":
            return 1, "", NOT_FOUND
        return 1, "", "gh: boom\n"

    assert release_github.ensure(REPO, "1.0.0", SHA, notes, False, failing_create) == 1
    assert "gh release create v1.0.0 failed: gh: boom" in capsys.readouterr().out


def test_branch_head_reports_the_commit_or_the_absence() -> None:
    run = _runner(
        {
            "git/ref/heads/release/dev-bump-v1.0.0": {"object": {"type": "commit", "sha": SHA}},
            "git/ref/heads/gone": None,
            "git/ref/heads/broken": {"object": {}},
        }
    )
    assert release_github.branch_head(REPO, "release/dev-bump-v1.0.0", run) == SHA
    assert release_github.branch_head(REPO, "gone", run) is None
    with pytest.raises(release_github.GitHubError, match="without a sha"):
        release_github.branch_head(REPO, "broken", run)


def test_main_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = _notes(tmp_path)
    answers = _nothing()
    answers["git/ref/heads/x"] = None
    monkeypatch.setattr(release_github, "run_gh", _runner(answers, []))
    common = ["--repo", REPO]
    args = ["--version", "1.0.0", "--sha", SHA, "--notes", str(notes), "--prerelease", "false"]
    assert release_github.main([*common, "check", *args]) == 0
    assert release_github.main([*common, "ensure", *args]) == 0
    capsys.readouterr()
    assert release_github.main([*common, "branch-head", "--branch", "x"]) == 0
    assert capsys.readouterr().out == "absent\n"
    monkeypatch.setattr(release_github, "run_gh", _runner({"git/ref/heads/x": "fail"}))
    assert release_github.main([*common, "branch-head", "--branch", "x"]) == 1
    assert "::error::" in capsys.readouterr().out


def test_check_warns_when_the_workflow_token_cannot_create_the_tag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A workflow change after the release commit: the workflow token cannot
    # create the tag there, so the check warns before anything is published.
    notes = _notes(tmp_path)
    run = _runner(_nothing())
    assert release_github.check(REPO, "1.0.0", SHA, notes, False, run, workflows_changed=True) == 0
    assert "::warning::The workflow files at the release commit" in capsys.readouterr().out
    # With the tag already there, nothing needs creating at the release commit.
    tagged = _nothing()
    tagged["git/ref/tags/v1.0.0"] = {"object": {"type": "commit", "sha": SHA}}
    assert (
        release_github.check(
            REPO, "1.0.0", SHA, notes, False, _runner(tagged), workflows_changed=True
        )
        == 0
    )
    assert "::warning::" not in capsys.readouterr().out


def test_ensure_stops_with_the_manual_command_when_workflows_changed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = _notes(tmp_path)
    created: list[list[str]] = []
    run = _runner({"git/ref/tags/v1.0.0rc1": None, "releases?per_page=100": []}, created)
    assert (
        release_github.ensure(REPO, "1.0.0rc1", SHA, notes, True, run, workflows_changed=True) == 1
    )
    assert created == []
    out = capsys.readouterr().out
    assert "the workflow token never may" in out
    assert (
        f"gh release create v1.0.0rc1 --repo {REPO} --target {SHA} --title v1.0.0rc1 "
        "--notes-file release-notes.md --prerelease"
    ) in out


def test_ensure_creates_the_release_without_target_for_an_existing_tag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A person created the tag at the release commit; GitHub ignores
    # --target for an existing tag and would still check it for workflow
    # changes, so the release is created without it.
    notes = _notes(tmp_path)
    created: list[list[str]] = []
    answers = _nothing()
    answers["git/ref/tags/v1.0.0"] = {"object": {"type": "commit", "sha": SHA}}
    run = _runner(answers, created)
    assert release_github.ensure(REPO, "1.0.0", SHA, notes, False, run, workflows_changed=True) == 0
    assert len(created) == 1
    assert "--target" not in created[0]
    assert "created release v1.0.0" in capsys.readouterr().out


def test_main_passes_the_workflow_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = _notes(tmp_path)
    monkeypatch.setattr(release_github, "run_gh", _runner(_nothing(), []))
    args = ["--version", "1.0.0", "--sha", SHA, "--notes", str(notes), "--prerelease", "false"]
    command = ["--repo", REPO, "check", *args, "--workflows-changed", "true"]
    assert release_github.main(command) == 0
    assert "::warning::" in capsys.readouterr().out


BRANCH = "release/dev-bump-v1.0.0"
NEXT = "1.0.1.dev0"
PULLS = f"repos/{REPO}/pulls?head=standard-voice:{BRANCH}&state=all&per_page=100"
CONTENTS = f"repos/{REPO}/contents/pyproject.toml?ref={BRANCH}"
COMPARE = f"https://github.com/{REPO}/compare/main...{BRANCH}?expand=1"


def _pull(
    number: int,
    state: str = "open",
    *,
    merged: bool = False,
    base: str = "main",
    head_repo: str = REPO,
) -> dict[str, object]:
    return {
        "number": number,
        "state": state,
        "merged_at": "2026-10-26T00:00:00Z" if merged else None,
        "html_url": f"https://github.com/{REPO}/pull/{number}",
        "base": {"ref": base},
        "head": {"ref": BRANCH, "repo": {"full_name": head_repo}},
    }


def _pyproject(version: str) -> dict[str, object]:
    text = f'[project]\nname = "standard-asr"\nversion = "{version}"\n'
    return {"encoding": "base64", "content": base64.b64encode(text.encode()).decode()}


def _pull_runner(
    pulls: object,
    calls: list[list[str]],
    *,
    create: str = "ok",
    assignees: list[str] | None = None,
    declared: str | None = NEXT,
) -> Runner:
    # ``pulls`` is the answer to the lookup, or "fail" for a 403; a pull
    # request this runner opens joins it, as on GitHub. ``create`` is
    # ``"ok"`` or ``"refuse"``. ``assignees`` is the list GitHub answers
    # after the assignment; None answers with the requested login.
    # ``declared`` is the version pyproject.toml declares on the branch, or
    # None for a branch without the file.
    def run(args: list[str]) -> tuple[int, str, str]:
        calls.append(args)
        assert args[0] == "api", args
        if args[1] == PULLS:
            if pulls == "fail":
                return 1, "", "gh: Forbidden (HTTP 403)\n"
            return 0, json.dumps(pulls), ""
        if args[1] == CONTENTS:
            if declared is None:
                return 1, "", NOT_FOUND
            return 0, json.dumps(_pyproject(declared)), ""
        fields = dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "-f")
        assert args[2:4] == ["--method", "POST"], args
        if args[1] == f"repos/{REPO}/pulls":
            if create == "refuse":
                refused = "GitHub Actions is not permitted to create or approve pull requests."
                return 1, "", f"gh: {refused} (HTTP 403)\n"
            opened = _pull(7)
            assert isinstance(pulls, list)
            cast("list[object]", pulls).append(opened)
            return 0, json.dumps(opened), ""
        if args[1].startswith(f"repos/{REPO}/issues/") and args[1].endswith("/assignees"):
            listed = [fields["assignees[]"]] if assignees is None else assignees
            return 0, json.dumps({"assignees": [{"login": login} for login in listed]}), ""
        raise AssertionError(f"unexpected gh call {args}")

    return run


def _open(run: Runner) -> tuple[int, str, bool]:
    return release_github.dev_bump_pull_request(
        REPO, BRANCH, "main", f"chore: bump version to {NEXT}", "Body.", "maintainer", run
    )


def test_dev_bump_pull_request_looks_only_at_this_repositorys_branch() -> None:
    # ``gh pr list --head`` also matches a fork's branch of the same name.
    # The REST filter ``head=owner:branch`` leaves out another owner's fork
    # but not a fork the same owner holds, so the head repository decides.
    calls: list[list[str]] = []
    pulls = [_pull(5), _pull(9, head_repo="standard-voice/standard_asr-fork")]
    found = release_github.pull_requests_from(REPO, BRANCH, _pull_runner(pulls, calls))
    assert calls == [["api", PULLS]]
    assert [entry["number"] for entry in found] == [5]
    number, _, created = _open(_pull_runner(pulls, []))
    assert (number, created) == (5, False)


def test_dev_bump_pull_request_opens_and_assigns_a_new_one() -> None:
    calls: list[list[str]] = []
    number, url, created = _open(_pull_runner([], calls))
    assert (number, url, created) == (7, f"https://github.com/{REPO}/pull/7", True)
    create = calls[1]
    assert create[1] == f"repos/{REPO}/pulls"
    for field in (f"head={BRANCH}", "base=main", f"title=chore: bump version to {NEXT}"):
        assert field in create
    assert calls[2][1] == f"repos/{REPO}/issues/7/assignees"
    assert "assignees[]=maintainer" in calls[2]


def test_dev_bump_pull_request_keeps_the_open_one() -> None:
    # A second run finds the pull request the first one opened, opens
    # nothing new, and assigns it again.
    calls: list[list[str]] = []
    number, _, created = _open(_pull_runner([_pull(3, "closed"), _pull(5)], calls))
    assert (number, created) == (5, False)
    assert [call[1] for call in calls] == [PULLS, f"repos/{REPO}/issues/5/assignees"]


def test_a_second_run_assigns_the_pull_request_the_first_one_opened() -> None:
    # The first run opens the pull request, and GitHub leaves the assignee
    # out; the second run finds that pull request and opens no other.
    pulls: list[object] = []
    calls: list[list[str]] = []
    with pytest.raises(release_github.GitHubError, match="did not assign"):
        _open(_pull_runner(pulls, calls, assignees=[]))
    assert _open(_pull_runner(pulls, calls)) == (7, f"https://github.com/{REPO}/pull/7", False)
    assert [call[1] for call in calls].count(f"repos/{REPO}/pulls") == 1


@pytest.mark.parametrize(
    ("pulls", "message"),
    [
        ([_pull(4, "closed", merged=True)], "something set the version back"),
        ([_pull(4, "closed", merged=True, base="maint")], "merged into maint, not main"),
        ([_pull(6), _pull(9, "closed")], "closed without merging"),
        ([_pull(6, base="maint")], "merges into maint, not main"),
        ("fail", "HTTP 403"),
    ],
)
def test_dev_bump_pull_request_stops_where_a_person_has_to_look(
    pulls: object, message: str
) -> None:
    calls: list[list[str]] = []
    with pytest.raises(release_github.GitHubError, match=message):
        _open(_pull_runner(pulls, calls))
    assert [call[1] for call in calls] == [PULLS]


def test_dev_bump_pull_request_names_the_setting_when_github_refuses() -> None:
    with pytest.raises(release_github.PullRequestRefusedError) as raised:
        _open(_pull_runner([], [], create="refuse"))
    assert "Allow GitHub Actions to create and approve pull requests" in str(raised.value)
    assert "not permitted" in str(raised.value)


def test_dev_bump_pull_request_reads_the_assignment_back() -> None:
    # GitHub can leave an assignee out without an error, so the answer is
    # read back.
    with pytest.raises(
        release_github.GitHubError, match="did not assign pull request #5 to maintainer"
    ):
        _open(_pull_runner([_pull(5)], [], assignees=[]))


def test_declared_version_reads_pyproject_on_the_branch() -> None:
    pytest.importorskip("tomllib")
    assert release_github.declared_version(REPO, BRANCH, _pull_runner([], [])) == NEXT
    with pytest.raises(release_github.GitHubError, match="has no pyproject.toml"):
        release_github.declared_version(REPO, BRANCH, _pull_runner([], [], declared=None))
    without = {"encoding": "base64", "content": base64.b64encode(b"[project]\n").decode()}
    run = _runner({CONTENTS.removeprefix(f"repos/{REPO}/"): without})
    with pytest.raises(release_github.GitHubError, match="declares no project.version"):
        release_github.declared_version(REPO, BRANCH, run)


def _dev_bump_args(summary: Path) -> list[str]:
    command = ["--repo", REPO, "dev-bump-pr", "--branch", BRANCH, "--base", "main"]
    versions = ["--version", "1.0.0", "--next", NEXT, "--assignee", "maintainer"]
    return [*command, *versions, "--compare-url", COMPARE, "--summary", str(summary)]


def test_main_opens_the_dev_bump_pull_request_and_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("tomllib")
    calls: list[list[str]] = []
    monkeypatch.setattr(release_github, "run_gh", _pull_runner([], calls))
    summary = tmp_path / "summary.md"
    assert release_github.main(_dev_bump_args(summary)) == 0
    assert [call[1] for call in calls][:2] == [CONTENTS, PULLS]
    create = calls[2]
    body = dict(create[i + 1].split("=", 1) for i, a in enumerate(create) if a == "-f")["body"]
    assert "v1.0.0" in body and NEXT in body and "Approve workflows to run" in body
    out = capsys.readouterr().out
    assert (
        f"::notice title=Merge the dev-bump pull request::https://github.com/{REPO}/pull/7" in out
    )
    text = summary.read_text(encoding="utf-8")
    assert "#7" in text and "Approve workflows to run" in text and NEXT in text


def test_main_stops_when_the_branch_does_not_declare_the_next_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A branch that already held commits is kept as it is; when those
    # commits do not set the next version, the job opens nothing.
    pytest.importorskip("tomllib")
    calls: list[list[str]] = []
    monkeypatch.setattr(release_github, "run_gh", _pull_runner([], calls, declared="1.0.0"))
    assert release_github.main(_dev_bump_args(tmp_path / "summary.md")) == 1
    assert [call[1] for call in calls] == [CONTENTS]
    assert f"declares version 1.0.0, not {NEXT}" in capsys.readouterr().out


def test_main_prints_the_compare_link_when_github_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("tomllib")
    monkeypatch.setattr(release_github, "run_gh", _pull_runner([], [], create="refuse"))
    summary = tmp_path / "summary.md"
    assert release_github.main(_dev_bump_args(summary)) == 1
    out = capsys.readouterr().out
    assert "Allow GitHub Actions to create and approve pull requests" in out
    assert COMPARE in out
    assert not summary.exists()


def test_main_does_not_offer_the_compare_link_for_a_closed_pull_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A person closed the pull request. The error names both ways on, and
    # the job does not offer the link that opens another.
    pytest.importorskip("tomllib")
    monkeypatch.setattr(release_github, "run_gh", _pull_runner([_pull(9, "closed")], []))
    assert release_github.main(_dev_bump_args(tmp_path / "summary.md")) == 1
    out = capsys.readouterr().out
    assert "closed without merging" in out
    assert COMPARE not in out
