# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dependency-conflict doctor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import pytest

from standard_asr import doctor


class _VersionInfo(NamedTuple):
    major: int
    minor: int
    micro: int
    releaselevel: str
    serial: int


@dataclass
class _FakeDist:
    name: str
    requires: list[str] | None


@dataclass
class _FakeEP:
    name: str
    dist: _FakeDist | None


def _patch_eps(monkeypatch: pytest.MonkeyPatch, eps: list[_FakeEP]) -> None:
    def _entry_points(*, group: str) -> list[_FakeEP]:
        return eps

    monkeypatch.setattr(doctor, "entry_points", _entry_points)


def test_no_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_eps(monkeypatch, [])
    report = doctor.diagnose()
    assert report.plugins == []
    assert report.has_conflict is False
    assert "No Standard ASR plugins" in doctor.format_report(report)


def test_compatible_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("a/x", _FakeDist("std-a", ["numpy>=1.26", "pydantic>=2"])),
            _FakeEP("b/y", _FakeDist("std-b", ["numpy>=2.1"])),
        ],
    )
    report = doctor.diagnose()
    assert len(report.plugins) == 2
    assert report.has_conflict is False
    assert "No dependency conflicts" in doctor.format_report(report)


def test_numpy1_vs_2_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("old/funasr", _FakeDist("std-funasr", ["numpy<2"])),
            _FakeEP("new/qwen", _FakeDist("std-qwen", ["numpy>=2.1"])),
        ],
    )
    report = doctor.diagnose()
    assert report.has_conflict is True
    assert any("1.x vs 2.x" in c for c in report.conflicts)
    assert "!" in doctor.format_report(report)


def test_missing_dist(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_eps(monkeypatch, [_FakeEP("x/y", None)])
    report = doctor.diagnose()
    assert report.plugins[0].distribution == "<unknown>"
    assert report.plugins[0].numpy_spec is None


def test_classify_no_false_positive_for_bounded_range() -> None:
    """``>=1.26,<2.3`` admits both 1.x and 2.x -> NOT a hard numpy1-only split."""
    only1, req2 = doctor._classify_numpy(">=1.26,<2.3")  # pyright: ignore[reportPrivateUsage]
    assert only1 is False
    assert req2 is False


@pytest.mark.parametrize(
    "spec",
    ["==1.26.*", "~=1.26.0", ">=1.21,<1.27", "<2"],
)
def test_classify_detects_numpy1_only(spec: str) -> None:
    only1, req2 = doctor._classify_numpy(spec)  # pyright: ignore[reportPrivateUsage]
    assert only1 is True
    assert req2 is False


@pytest.mark.parametrize("spec", [">=2", ">=2.1", "==2.*"])
def test_classify_detects_numpy2_required(spec: str) -> None:
    only1, req2 = doctor._classify_numpy(spec)  # pyright: ignore[reportPrivateUsage]
    assert only1 is False
    assert req2 is True


@pytest.mark.parametrize("spec", [None, "(any)", "", "not-a-spec"])
def test_classify_returns_neutral_for_unconstrained_or_invalid(
    spec: str | None,
) -> None:
    assert doctor._classify_numpy(spec) == (False, False)  # pyright: ignore[reportPrivateUsage]


def test_bounded_range_pin_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``~=1.26.0`` pin vs ``>=2.1`` is a real conflict the old regex missed."""
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("old/funasr", _FakeDist("std-funasr", ["numpy~=1.26.0"])),
            _FakeEP("new/qwen", _FakeDist("std-qwen", ["numpy>=2.1"])),
        ],
    )
    report = doctor.diagnose()
    assert report.has_conflict is True
    assert any("1.x vs 2.x" in c for c in report.conflicts)


def test_numpy_extras_specifier_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["numpy[extra]<2"]))],
    )
    report = doctor.diagnose()
    assert report.plugins[0].numpy_spec == "<2"


def test_numpy_spec_skips_non_numpy_requirements_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The numpy requirement is not first in the list; the extractor must skip the
    # non-matching entries (the loop-continue branch) and still find numpy.
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["pydantic>=2", "typing-extensions", "numpy<2"]))],
    )
    report = doctor.diagnose()
    assert report.plugins[0].numpy_spec == "<2"


def test_py313_no_numpy1_wheel_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_eps(
        monkeypatch,
        [_FakeEP("old/funasr", _FakeDist("std-funasr", ["numpy<2"]))],
    )
    fake_vi = _VersionInfo(3, 13, 0, "final", 0)
    monkeypatch.setattr(doctor.sys, "version_info", fake_vi)
    report = doctor.diagnose()
    assert any("no numpy<2 wheel" in c for c in report.conflicts)


def test_packaging_available_false_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When `packaging` cannot be imported, packaging_available reports False.
    import builtins

    real_import = builtins.__import__

    def _import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("packaging"):
            raise ImportError("no packaging")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _import)
    assert doctor.packaging_available() is False


def test_classify_numpy_neutral_when_packaging_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without `packaging`, the classifier conservatively reports no hard split so
    # it never flags a conflict it cannot verify.
    import builtins

    real_import = builtins.__import__

    def _import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("packaging"):
            raise ImportError("no packaging")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _import)
    assert doctor._classify_numpy("<2") == (False, False)  # pyright: ignore[reportPrivateUsage]


def test_packaging_unavailable_adds_note(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["numpy<2"]))],
    )
    monkeypatch.setattr(doctor, "packaging_available", lambda: False)
    report = doctor.diagnose()
    assert any("packaging" in n for n in report.notes)
    assert "note:" in doctor.format_report(report)
