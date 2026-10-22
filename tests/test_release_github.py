# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""The release workflow's GitHub tag, release, and pull request checks."""

from __future__ import annotations

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


def _runner(answers: Answers, created: list[list[str]] | None = None) -> Runner:
    # ``answers`` maps a suffix of the ``gh api`` path to the JSON to answer
    # with, to None for a 404, or to the string "fail" for another failure;
    # ``releases`` holds the release list, ``pr-list`` the pull requests.
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
        if args[:2] == ["pr", "list"]:
            if answers.get("pr-list") == "fail":
                return 1, "", "gh: boom\n"
            return 0, json.dumps(answers.get("pr-list", [])), ""
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


def test_pr_state_reports_the_newest_pull_request() -> None:
    def with_pulls(*pulls: dict[str, object]) -> Runner:
        return _runner({"pr-list": list(pulls)})

    newest_merged = with_pulls({"number": 3, "state": "CLOSED"}, {"number": 5, "state": "MERGED"})
    assert release_github.pr_state(REPO, "release/dev-bump-v1.0.0", newest_merged) == "merged"
    assert release_github.pr_state(REPO, "x", with_pulls({"number": 1, "state": "OPEN"})) == "open"
    closed = with_pulls({"number": 1, "state": "CLOSED"})
    assert release_github.pr_state(REPO, "x", closed) == "closed"
    assert release_github.pr_state(REPO, "x", with_pulls()) == "absent"
    with pytest.raises(release_github.GitHubError, match="gh pr list failed"):
        release_github.pr_state(REPO, "x", _runner({"pr-list": "fail"}))


def test_main_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = _notes(tmp_path)
    answers = _nothing()
    answers["pr-list"] = []
    answers["git/ref/heads/x"] = None
    monkeypatch.setattr(release_github, "run_gh", _runner(answers, []))
    common = ["--repo", REPO]
    args = ["--version", "1.0.0", "--sha", SHA, "--notes", str(notes), "--prerelease", "false"]
    assert release_github.main([*common, "check", *args]) == 0
    assert release_github.main([*common, "ensure", *args]) == 0
    assert release_github.main([*common, "pr-state", "--branch", "x"]) == 0
    assert capsys.readouterr().out.strip().endswith("absent")
    assert release_github.main([*common, "branch-head", "--branch", "x"]) == 0
    assert capsys.readouterr().out == "absent\n"
    monkeypatch.setattr(release_github, "run_gh", _runner({"pr-list": "fail"}))
    assert release_github.main([*common, "pr-state", "--branch", "x"]) == 1


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
