# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""The release workflow's release-commit selector and build recipe."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

# The script parses TOML with the standard library, which the release runner
# has; the project's floor interpreter does not, so these tests skip there.
pytest.importorskip("tomllib")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_commit.py"

Git = Callable[[list[str]], "str | None"]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_commit", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


release_commit = _load()


def _pyproject(version: str, floor: str = "0.11.21") -> str:
    return (
        "[project]\n"
        'name = "standard-asr"\n'
        f'version = "{version}"\n'
        "\n"
        "[build-system]\n"
        f'requires = ["uv-build>={floor},<0.12"]\n'
        'build-backend = "uv_build"\n'
    )


def test_project_version_is_read_from_parsed_toml() -> None:
    text = (
        '# version = "9.9.9" in a comment\n'
        "[tool.other]\n"
        'version = "1.2.3"\n'
        "[project]\n"
        'name = "x"\n'
        "version = '0.2.0' # a comment after a single-quoted value\n"
        "[tool.uv]\n"
        'version = "7.7.7"\n'
    )
    assert release_commit.project_version(text) == "0.2.0"
    assert release_commit.project_version('[tool.uv]\nversion = "1"\n') is None
    assert release_commit.project_version("[project]\nname = 'x'\n") is None
    assert release_commit.project_version("[project]\nversion = 1\n") is None
    assert release_commit.project_version("not toml = = [\n") is None


def test_uv_build_floor_reads_the_requirement_not_a_comment() -> None:
    text = (
        "[build-system]\n"
        '# earlier: requires = ["uv-build>=0.10.0"]\n'
        'requires = ["setuptools", "uv-build>=0.11.21,<0.12"]\n'
        'build-backend = "uv_build"\n'
    )
    assert release_commit.uv_build_floor(text) == "0.11.21"
    assert (
        release_commit.uv_build_floor('[build-system]\nrequires = ["uv_build >= 0.12.3"]\n')
        == "0.12.3"
    )
    assert release_commit.uv_build_floor('[build-system]\nrequires = ["hatchling"]\n') is None
    assert release_commit.uv_build_floor('[build-system]\nrequires = ["uv-build<0.12"]\n') is None
    assert release_commit.uv_build_floor("[project]\n") is None


def _fake_git(history: list[tuple[str, str | None]]) -> Git:
    # ``history`` lists (commit, version) from newest to oldest along the
    # first-parent line; a None version means pyproject.toml is absent.
    by_commit = dict(history)
    order = [commit for commit, _ in history]

    def git(args: list[str]) -> str | None:
        if args[0] == "rev-parse":
            return "false\n"
        if args[0] == "log":
            return "\n".join(order) + "\n"
        assert args[0] == "show"
        ref = args[1].removesuffix(":pyproject.toml")
        if ref.endswith("^"):
            index = order.index(ref[:-1]) + 1
            if index >= len(order):
                return None
            ref = order[index]
        version = by_commit.get(ref)
        return _pyproject(version) if version else None

    return git


def _silent(args: list[str]) -> str | None:
    # A complete checkout whose history never touched pyproject.toml.
    return "false\n" if args[0] == "rev-parse" else None


def test_find_release_commit_picks_the_commit_that_set_the_version() -> None:
    git = _fake_git([("c4", "0.2.0"), ("c3", "0.2.0"), ("c2", "0.2.0.dev0"), ("c1", "0.1.1")])
    # c3 set the version; c4 (a later merge) kept it.
    assert release_commit.find_release_commit("0.2.0", "HEAD", git) == "c3"
    assert release_commit.find_release_commit("0.1.1", "HEAD", git) == "c1"


def test_find_release_commit_takes_the_newest_change_to_the_version() -> None:
    # The version went to 0.2.0, back to a dev version, then to 0.2.0 again:
    # the newest change wins, so "same version" means "same commit" only
    # while the branch still declares it.
    git = _fake_git([("c5", "0.2.0"), ("c4", "0.2.0.dev1"), ("c3", "0.2.0"), ("c2", "0.1.1")])
    assert release_commit.find_release_commit("0.2.0", "HEAD", git) == "c5"


def test_find_release_commit_returns_none_when_no_commit_set_it() -> None:
    git = _fake_git([("c2", "0.2.0.dev0"), ("c1", "0.1.1")])
    assert release_commit.find_release_commit("0.2.0", "HEAD", git) is None
    assert release_commit.find_release_commit("0.2.0", "HEAD", _silent) is None


def test_a_shallow_checkout_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # On a shallow clone the walk would end at the fetch boundary and take
    # the tip for the release commit, so the script refuses to guess.
    def shallow(args: list[str]) -> str | None:
        return "true\n" if args[0] == "rev-parse" else None

    with pytest.raises(release_commit.IncompleteHistoryError, match="shallow"):
        release_commit.find_release_commit("0.2.0", "HEAD", shallow)
    monkeypatch.setattr(release_commit, "run_git", shallow)
    assert release_commit.main(["--version", "0.2.0"]) == 1
    assert "::error::The checkout is shallow" in capsys.readouterr().err


def test_find_release_commit_ignores_a_change_that_keeps_the_version() -> None:
    # A later commit that only changes how the line is written is not a new release
    # commit, because the parsed version did not change.
    def git(args: list[str]) -> str | None:
        if args[0] == "rev-parse":
            return "false\n"
        if args[0] == "log":
            return "c2\nc1\n"
        ref = args[1].removesuffix(":pyproject.toml")
        if ref == "c2":
            return (
                "[project]\nversion = '0.2.0'  # reformatted\n"
                "[build-system]\nrequires = ['uv-build>=0.11.21']\n"
            )
        if ref in ("c1", "c2^"):
            return _pyproject("0.2.0")
        return None

    assert release_commit.find_release_commit("0.2.0", "HEAD", git) == "c1"


def _clean_git_env() -> dict[str, str]:
    # A git hook exports GIT_DIR and GIT_WORK_TREE (the pre-push hook runs
    # the test suite). With them set, every git command below would act on
    # the real repository instead of the temporary one.
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _git_in(repo: Path) -> Callable[..., str]:
    env = _clean_git_env()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout

    return git


def test_find_release_commit_on_a_real_repository(tmp_path: Path) -> None:
    """A merge of the release branch is one first-parent commit.

    A string search would find the commit inside the branch; the selector
    returns the merge, which is what the branch's history shows.
    """
    git = _git_in(tmp_path)
    git("init", "-q", "-b", "main")
    # Refuse to go on unless git really works in the temporary directory.
    assert Path(git("rev-parse", "--show-toplevel").strip()).resolve() == tmp_path.resolve()
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("config", "commit.gpgsign", "false")
    (tmp_path / "pyproject.toml").write_text(_pyproject("0.1.1"), encoding="utf-8")
    git("add", "pyproject.toml")
    git("commit", "-q", "-m", "0.1.1")
    git("checkout", "-q", "-b", "release")
    (tmp_path / "pyproject.toml").write_text(_pyproject("0.2.0"), encoding="utf-8")
    git("commit", "-q", "-am", "set 0.2.0")
    inner = git("rev-parse", "HEAD").strip()
    (tmp_path / "CHANGELOG.md").write_text("## [0.2.0]\n", encoding="utf-8")
    git("add", "CHANGELOG.md")
    git("commit", "-q", "-m", "changelog")
    git("checkout", "-q", "main")
    git("merge", "-q", "--no-ff", "-m", "Merge release", "release")
    merge = git("rev-parse", "HEAD").strip()
    (tmp_path / "other.txt").write_text("later work\n", encoding="utf-8")
    git("add", "other.txt")
    git("commit", "-q", "-m", "later work")

    def runner(args: list[str]) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            capture_output=True,
            text=True,
            check=False,
            env=_clean_git_env(),
        )
        return result.stdout if result.returncode == 0 else None

    assert release_commit.find_release_commit("0.2.0", "HEAD", runner) == merge
    assert merge != inner
    # The string search the selector replaces finds the commit inside the branch.
    search = git("log", "-1", "--format=%H", '-Sversion = "0.2.0"', "--", "pyproject.toml")
    assert search.strip() == inner


def test_main_prints_the_sha_and_the_recipe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    git = _fake_git([("c2", "0.2.0"), ("c1", "0.1.1")])
    monkeypatch.setattr(release_commit, "run_git", git)
    assert release_commit.main(["--version", "0.2.0"]) == 0
    assert capsys.readouterr().out == "sha=c2\nuv=0.11.21\n"
    assert release_commit.main(["--version", "0.3.0"]) == 1
    assert "No commit on the first-parent history" in capsys.readouterr().err


def test_main_fails_without_a_build_recipe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def git(args: list[str]) -> str | None:
        if args[0] == "rev-parse":
            return "false\n"
        if args[0] == "log":
            return "c1\n"
        if args[1].startswith("c1^"):
            return None  # the root commit has no parent
        return '[project]\nversion = "0.2.0"\n[build-system]\nrequires = ["hatchling"]\n'

    monkeypatch.setattr(release_commit, "run_git", git)
    assert release_commit.main(["--version", "0.2.0"]) == 1
    assert "names no uv-build lower bound" in capsys.readouterr().err
