# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Shell guards of the release workflow, run as the runner runs them."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"

# The release workflow runs this step under bash on Linux. On Windows, the
# bash on PATH can be the launcher of the Windows Subsystem for Linux, which
# fails when no distribution is installed, so the test runs elsewhere only.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="needs a POSIX bash, as the release workflow's Linux runner has",
)


def _step_script(name: str) -> str:
    """Return the ``run`` block of the step called ``name``, with its indentation removed."""
    lines = _WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == f"- name: {name}")
    run_at = next(i for i in range(start, len(lines)) if lines[i].strip() == "run: |")
    indent = len(lines[run_at]) - len(lines[run_at].lstrip())
    body: list[str] = []
    for line in lines[run_at + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    return textwrap.dedent("\n".join(body))


@pytest.mark.parametrize(
    ("ref", "accepted"),
    [("refs/heads/main", True), ("refs/heads/feature", False), ("refs/tags/main", False)],
)
def test_default_branch_guard_compares_the_full_ref(ref: str, accepted: bool) -> None:
    # A tag can carry the branch's name, and GITHUB_REF_NAME is the same
    # short name for both, so the guard compares the full ref.
    script = _step_script("Assert the run targets the default branch")
    env = {"PATH": os.environ["PATH"], "GITHUB_REF": ref, "DEFAULT_BRANCH": "main"}
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is accepted, result.stdout + result.stderr
    if not accepted:
        assert "::error::Run the release workflow from 'main'." in result.stdout


def _install_commands(step: str) -> list[str]:
    """Return the ``uv pip install`` commands of a step, continuation lines joined."""
    script = _step_script(step).replace("\\\n", " ")
    return [" ".join(line.split()) for line in script.splitlines() if "uv pip install" in line]


def test_testpypi_install_takes_the_package_from_testpypi_only() -> None:
    # uv's --index adds an index in front of PyPI, so a package TestPyPI
    # does not serve would come from PyPI and the check would pass on the
    # wrong index. --default-index replaces PyPI (uv CLI reference).
    package, dependencies = _install_commands("Install from TestPyPI in isolation")
    assert "--default-index https://test.pypi.org/simple/" in package
    assert "standard-asr==" in package
    assert "--no-deps" in package
    assert "--default-index https://pypi.org/simple/" in dependencies
    for command in (package, dependencies):
        assert " --index " not in command, command
        assert "--extra-index-url" not in command, command


def _git(repo: Path, *args: str) -> str:
    """Run git in ``repo`` without the variables a git hook exports."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    result = subprocess.run(
        ["git", "-C", str(repo), *args], env=env, check=True, capture_output=True, text=True
    )
    return result.stdout


def test_workflow_comparison_sees_the_current_default_branch(tmp_path: Path) -> None:
    # The step runs scripts/release_commit.py with this interpreter, and the
    # script needs ``tomllib``, which Python 3.10 lacks.
    pytest.importorskip("tomllib")
    # actions/checkout sets origin/main back to the commit the run started
    # from when main has moved. A workflow change merged while the run waited
    # must still count, so the step fetches main before it compares.
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    _git(tmp_path, "init", "-q", "-b", "main", str(work))
    for key, value in (("user.email", "t@example.com"), ("user.name", "t")):
        _git(work, "config", key, value)
    _git(work, "config", "commit.gpgsign", "false")
    assert Path(_git(work, "rev-parse", "--show-toplevel").strip()).resolve() == work.resolve()
    (work / "scripts").mkdir()
    (work / "scripts" / "release_commit.py").write_text(
        (_WORKFLOW.parents[2] / "scripts" / "release_commit.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (work / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.2.0"\n'
        '[build-system]\nrequires = ["uv-build>=0.11.21,<0.12"]\n',
        encoding="utf-8",
    )
    workflows = work / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text("name: CI\n", encoding="utf-8")
    _git(work, "add", ".")
    _git(work, "commit", "-q", "-m", "release 0.2.0")
    release_commit = _git(work, "rev-parse", "HEAD").strip()
    _git(work, "remote", "add", "origin", str(remote))
    _git(work, "push", "-q", "origin", "main")
    (workflows / "ci.yml").write_text("name: CI, updated\n", encoding="utf-8")
    _git(work, "commit", "-q", "-am", "update a workflow")
    _git(work, "push", "-q", "origin", "main")
    # The state actions/checkout leaves: HEAD and origin/main at the start.
    _git(work, "checkout", "-q", "--detach", release_commit)
    _git(work, "update-ref", "refs/remotes/origin/main", release_commit)

    output = tmp_path / "github-output"
    env = {
        **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
        "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ["PATH"]]),
        "VERSION": "0.2.0",
        "DEFAULT_BRANCH": "main",
        "GITHUB_OUTPUT": str(output),
    }
    script = _step_script("Compare the workflow files with the default branch")
    result = subprocess.run(
        ["bash", "-c", script], cwd=work, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text(encoding="utf-8") == "changed=true\n"
    assert f"sha={release_commit}" in result.stdout


def test_dev_bump_opens_its_pull_request_for_the_person_who_ran_the_release() -> None:
    # The pull request itself reminds the person who released to merge it,
    # so the job opens it and assigns it to the account that started the
    # run. That needs pull-requests: write, which only this job holds.
    text = _WORKFLOW.read_text(encoding="utf-8")
    job = text[text.index("\n  dev-bump:\n") :]
    permissions = job[job.index("permissions:") : job.index("steps:")]
    assert "pull-requests: write" in permissions
    assert text.count("pull-requests: write") == 1
    script = " ".join(
        _step_script("Commit the next development version and open its pull request")
        .replace("\\\n", " ")
        .split()
    )
    assert 'release_github.py --repo "${GITHUB_REPOSITORY}" dev-bump-pr' in script
    assert '--assignee "${RELEASER}"' in script
    assert "RELEASER: ${{ github.actor }}" in job
    assert '--compare-url "${url}"' in script
