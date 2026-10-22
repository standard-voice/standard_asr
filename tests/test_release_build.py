# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""The release guard, fed with what ``uv build`` writes.

The unit tests of ``scripts/release_index.py`` hand it two files they wrote
themselves. This module builds the package for real, as the release workflow
does, and runs the pre-publish guard on the result against each state an
index can be in.
"""

from __future__ import annotations

import importlib.util
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "release_index.py"
_WORKFLOW = _ROOT / ".github" / "workflows" / "release.yml"
_BUILD_STEP = "- name: Build sdist and wheel from the release commit"

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")

Document = dict[str, object]
Fetch = Callable[[str], Document | None]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_index_for_build", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


release_index = _load()


def _workflow_build_command() -> list[str]:
    # The command of the workflow's build step, read from the workflow, so
    # that a change there (a dropped flag, say) is tested here.
    lines = _WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == _BUILD_STEP)
    line = next(line.strip() for line in lines[start:] if line.strip().startswith("uv build "))
    return shlex.split(line)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    dist = tmp_path_factory.mktemp("dist")
    command = _workflow_build_command()
    command[command.index("--out-dir") + 1] = str(dist)
    subprocess.run(command, cwd=_ROOT, check=True, capture_output=True, text=True)
    version = subprocess.run(
        ["uv", "version", "--short"], cwd=_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return dist, version


def _index(files: dict[str, str]) -> Document:
    return {"urls": [{"filename": name, "digests": {"sha256": sha}} for name, sha in files.items()]}


def _fetch(pypi: Document | None, testpypi: Document | None) -> Fetch:
    def fetch(url: str) -> Document | None:
        return testpypi if url.startswith("https://test.pypi.org/") else pypi

    return fetch


def test_local_files_accepts_a_real_build(built: tuple[Path, str]) -> None:
    dist, version = built
    files = release_index.local_files(dist, version)
    assert set(files) == release_index.expected_files(version)
    assert all(len(digest) == 64 for digest in files.values())


def test_guard_walks_the_index_states_with_a_real_build(
    built: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    dist, version = built
    local = release_index.local_files(dist, version)
    wheel = next(name for name in local if name.endswith(".whl"))

    # Nothing published yet: the first run.
    assert release_index.guard(version, dist, _fetch(None, None)) == 0
    # A run that failed after TestPyPI took the wheel: the publish fills the rest.
    assert release_index.guard(version, dist, _fetch(None, _index({wheel: local[wheel]}))) == 0
    # A run that failed after both indexes hold the whole build.
    assert release_index.guard(version, dist, _fetch(_index(local), _index(local))) == 0
    capsys.readouterr()
    # The index holds the wheel's name with other bytes: the version is burned there.
    other = dict(local, **{wheel: "0" * 64})
    assert release_index.guard(version, dist, _fetch(_index(other), None)) == 1
    assert "::error::" in capsys.readouterr().out
