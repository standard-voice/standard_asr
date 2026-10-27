# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""The release workflow's version guard and dev-bump arithmetic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("packaging")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_version.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_version", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


release_version = _load()


def test_releasable_tells_a_pre_release_from_a_final_release() -> None:
    assert release_version.releasable("0.2.0").is_prerelease is False
    assert release_version.releasable("0.2.0rc1").is_prerelease is True
    assert release_version.releasable("1.0.0a2").is_prerelease is True


@pytest.mark.parametrize(
    ("version", "reason"),
    [
        ("0.2.1.dev0", "a development version"),
        ("0.2.0rc2.dev0", "a development version"),
        ("0.2.0+local", "outside the versioning policy"),
        ("0.2.0.post1", "outside the versioning policy"),
        ("1!0.2.0", "outside the versioning policy"),
        ("0.2", "outside the versioning policy"),
        ("0.2.0.1", "outside the versioning policy"),
        ("not-a-version", "not a PEP 440 version"),
    ],
)
def test_releasable_refuses_what_the_workflow_must_not_release(version: str, reason: str) -> None:
    with pytest.raises(release_version.ReleaseVersionError, match=reason):
        release_version.releasable(version)


@pytest.mark.parametrize(
    ("released", "next_version"),
    [
        ("0.2.0", "0.2.1.dev0"),
        ("1.0.0", "1.0.1.dev0"),
        ("0.2.0rc1", "0.2.0rc2.dev0"),
        ("0.2.0a1", "0.2.0a2.dev0"),
        ("1.0.0b3", "1.0.0b4.dev0"),
    ],
)
def test_next_development_version_sorts_between_the_release_and_its_successors(
    released: str, next_version: str
) -> None:
    from packaging.version import Version

    assert release_version.next_development_version(released) == next_version
    assert Version(released) < Version(next_version)
    # The next release pull request can choose any of these; the placeholder
    # sorts before all of them.
    final = Version(released).base_version
    successors = [final, f"{final}rc9"] if Version(released).pre else []
    major, minor, patch = Version(released).release
    successors += [f"{major}.{minor}.{patch + 1}", f"{major}.{minor + 1}.0", f"{major + 1}.0.0"]
    for successor in successors:
        if Version(successor) > Version(released):
            assert Version(next_version) < Version(successor), successor


def test_next_development_version_refuses_a_version_that_was_not_releasable() -> None:
    with pytest.raises(release_version.ReleaseVersionError):
        release_version.next_development_version("0.2.1.dev0")


def test_main_answers_on_stdout_and_fails_on_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    assert release_version.main(["classify", "0.2.0rc1"]) == 0
    assert capsys.readouterr().out == "true\n"
    assert release_version.main(["classify", "0.2.0"]) == 0
    assert capsys.readouterr().out == "false\n"
    assert release_version.main(["next", "0.2.0"]) == 0
    assert capsys.readouterr().out == "0.2.1.dev0\n"
    assert release_version.main(["classify", "0.2.1.dev0"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("::error::The version on the branch is 0.2.1.dev0")
