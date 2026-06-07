# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dependency-conflict doctor."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from standard_asr import doctor


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
