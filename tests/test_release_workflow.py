# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Shell guards of the release workflow, run as the runner runs them."""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")


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
