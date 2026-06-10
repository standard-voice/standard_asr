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


@pytest.mark.parametrize("spec", [">=1.26,<2.3", ">=1.26"])
def test_classify_no_false_positive_for_both_compatible_range(spec: str) -> None:
    """A range admitting both 1.x and 2.x (bounded or open) -> NOT a hard split."""
    only1, req2 = doctor._classify_numpy(spec)  # pyright: ignore[reportPrivateUsage]
    assert only1 is False
    assert req2 is False


@pytest.mark.parametrize(
    "spec",
    ["==1.26.*", "~=1.26.0", ">=1.21,<1.27", "<2", "==1.26.4"],
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


@pytest.mark.parametrize("spec", ["==2.2.0", "==2.4.0"])
def test_classify_exact_off_grid_pin_is_numpy2_required(spec: str) -> None:
    """An exact pin to a 2.x version absent from any probe grid must classify as
    numpy-2-only -- the old fixed grid read it as admitting neither major
    (R3-CLI-DOCTOR-04)."""
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


def test_disjoint_same_major_ranges_are_conflicting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``==2.0.*`` vs ``>=2.3`` share no satisfying numpy release: a real
    intersection conflict the 1.x/2.x major split alone would miss (CLI-4)."""
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("a/x", _FakeDist("std-a", ["numpy==2.0.*"])),
            _FakeEP("b/y", _FakeDist("std-b", ["numpy>=2.3"])),
        ],
    )
    report = doctor.diagnose()
    assert report.has_conflict is True
    assert any("no common satisfying version" in c for c in report.conflicts)
    # Both are numpy2 -> the dedicated 1.x-vs-2.x message must NOT fire.
    assert not any("1.x vs 2.x" in c for c in report.conflicts)


def test_compatible_overlapping_ranges_not_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-major ranges that DO overlap (e.g. share 2.3.x) are not conflicts."""
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("a/x", _FakeDist("std-a", ["numpy>=2.0,<2.5"])),
            _FakeEP("b/y", _FakeDist("std-b", ["numpy>=2.3"])),
        ],
    )
    report = doctor.diagnose()
    assert report.has_conflict is False


def test_intersection_empty_helper_direct() -> None:
    """The intersection helper reports emptiness for disjoint sets and
    non-emptiness for overlapping ones (covers the direct branch)."""
    from packaging.specifiers import SpecifierSet

    assert (
        doctor._intersection_is_empty(  # pyright: ignore[reportPrivateUsage]
            [SpecifierSet("==2.0.*"), SpecifierSet(">=2.3")]
        )
        is True
    )
    assert (
        doctor._intersection_is_empty(  # pyright: ignore[reportPrivateUsage]
            [SpecifierSet(">=1.26"), SpecifierSet(">=2.1")]
        )
        is False
    )


@pytest.mark.parametrize(
    "specs",
    [
        # High pins that all land ABOVE the old bounded grid but share a
        # satisfying version -- the false "empty intersection" of NEW-DOCTOR-1.
        [">=2.40", ">=2.1"],
        ["==2.45.*", ">=2.40"],
        [">=3.0", ">=2.1"],
        ["~=2.5", ">=2.9"],
    ],
)
def test_high_pins_are_not_falsely_empty(specs: list[str]) -> None:
    """A satisfiable intersection of high pins must NOT read as empty just
    because every satisfying version sits above any fixed grid (NEW-DOCTOR-1)."""
    from packaging.specifiers import SpecifierSet

    sets = [SpecifierSet(s) for s in specs]
    assert doctor._intersection_is_empty(sets) is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "specs",
    [
        # Genuinely disjoint -- must still be reported empty.
        ["==2.0.*", ">=2.3"],
        [">=3.0", "<3"],
        ["==2.45.*", ">=2.50"],
    ],
)
def test_disjoint_pins_are_empty(specs: list[str]) -> None:
    from packaging.specifiers import SpecifierSet

    sets = [SpecifierSet(s) for s in specs]
    assert doctor._intersection_is_empty(sets) is True  # pyright: ignore[reportPrivateUsage]


def test_arbitrary_equality_edge_is_skipped_not_crash() -> None:
    """A ``===`` arbitrary-equality edge carries a non-PEP440 version string; it
    must be skipped during candidate derivation rather than crash the probe, and
    the remaining real boundary still drives the verdict."""
    from packaging.specifiers import SpecifierSet

    # ``===foobar`` matches only the literal 'foobar', which no numeric release
    # satisfies, so intersecting it with ``>=2.1`` is genuinely empty.
    assert (
        doctor._intersection_is_empty(  # pyright: ignore[reportPrivateUsage]
            [SpecifierSet("===foobar"), SpecifierSet(">=2.1")]
        )
        is True
    )


def test_high_pins_no_conflict_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two plugins requiring ``>=2.40`` and ``>=2.1`` share every 2.40+ release,
    so doctor must report NO conflict (and exit-code-wise, no false positive)."""
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("a/x", _FakeDist("std-a", ["numpy>=2.40"])),
            _FakeEP("b/y", _FakeDist("std-b", ["numpy>=2.1"])),
        ],
    )
    report = doctor.diagnose()
    assert report.has_conflict is False, report.conflicts


def test_single_plugin_internally_unsatisfiable_is_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SINGLE plugin whose own numpy declaration is internally unsatisfiable
    (``<2`` AND ``>=2.1``) must be flagged, not silently passed (NEW-DOCTOR-2)."""
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["numpy<2", "numpy>=2.1"]))],
    )
    report = doctor.diagnose()
    assert report.has_conflict is True
    assert any("internally unsatisfiable" in c for c in report.conflicts)
    # A single offender is a self-contradiction, not a cross-plugin conflict.
    assert not any("cannot share one process" in c for c in report.conflicts)


def test_canonical_dual_line_resolves_to_numpy2_on_py313(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canonical interpreter-conditional dual-line form (DEP.1) must resolve
    to the line whose marker holds on the running interpreter -- on 3.13 that is
    ``>=2.1``, so no bogus 'no numpy<2 wheel' conflict fires (CLI-1 / H2)."""
    _patch_eps(
        monkeypatch,
        [
            _FakeEP(
                "x/y",
                _FakeDist(
                    "std-x",
                    [
                        'numpy<2; python_version < "3.13"',
                        'numpy>=2.1; python_version >= "3.13"',
                    ],
                ),
            )
        ],
    )
    fake_vi = _VersionInfo(3, 13, 0, "final", 0)
    monkeypatch.setattr(doctor.sys, "version_info", fake_vi)
    report = doctor.diagnose()
    assert report.plugins[0].numpy_spec == ">=2.1"
    assert report.has_conflict is False


def test_marker_false_line_imposes_no_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A numpy line whose marker is False on the running interpreter must be
    ignored entirely, not treated as an active constraint (CLI-1 / H2)."""
    # On 3.13 the marker python_version < "3.10" is False, so numpy is not
    # required at all -> no constraint, no conflict.
    _patch_eps(
        monkeypatch,
        [_FakeEP("x/y", _FakeDist("std-x", ['numpy<2; python_version < "3.10"']))],
    )
    fake_vi = _VersionInfo(3, 13, 0, "final", 0)
    monkeypatch.setattr(doctor.sys, "version_info", fake_vi)
    report = doctor.diagnose()
    assert report.plugins[0].numpy_spec is None
    assert report.has_conflict is False


def test_legacy_parenthesized_specifier_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy parenthesized Requires-Dist (``numpy (>=1.26)``) must parse to the
    real specifier rather than being swallowed as unconstrained (CLI-2)."""
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("old/funasr", _FakeDist("std-funasr", ["numpy (<2)"])),
            _FakeEP("new/qwen", _FakeDist("std-qwen", ["numpy (>=2.1)"])),
        ],
    )
    report = doctor.diagnose()
    assert report.plugins[0].numpy_spec == "<2"
    assert report.has_conflict is True
    assert any("1.x vs 2.x" in c for c in report.conflicts)


def test_invalid_requirement_line_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed Requires-Dist line must be skipped, not abort parsing."""
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["==not a requirement==", "numpy<2"]))],
    )
    report = doctor.diagnose()
    assert report.plugins[0].numpy_spec == "<2"


def test_numpy_spec_display_fallback_when_packaging_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without `packaging`, the spec is extracted display-only (no marker eval)
    and conflicts are not classified -- doctor degrades, never misreports."""
    import builtins

    real_import = builtins.__import__

    def _import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("packaging"):
            raise ImportError("no packaging")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _import)
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("old/funasr", _FakeDist("std-funasr", ["numpy<2"])),
            _FakeEP("new/qwen", _FakeDist("std-qwen", ["numpy>=2.1"])),
        ],
    )
    report = doctor.diagnose()
    # Display string is still rendered (best-effort regex).
    assert report.plugins[0].numpy_spec == "<2"
    # But with packaging absent, the real numpy1-vs-2 conflict is NOT classified.
    assert report.has_conflict is False
    # The unclassified state is explicit, never a silent "all clean" (M8).
    assert report.analysis_unavailable is True
    assert any("packaging" in n for n in report.notes)


def test_numpy_spec_display_fallback_returns_none_without_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The display fallback returns None when numpy is not required at all."""
    import builtins

    real_import = builtins.__import__

    def _import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("packaging"):
            raise ImportError("no packaging")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _import)
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["pydantic>=2"]))],
    )
    report = doctor.diagnose()
    assert report.plugins[0].numpy_spec is None


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


def test_packaging_unavailable_with_plugins_headline_is_not_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With plugins present but ``packaging`` missing, the report carries an
    explicit analysis-unavailable state and the headline must NOT claim "no
    conflicts" -- doctor cannot prove the environment conflict-free (M8)."""
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["numpy>=1.26"]))],
    )
    monkeypatch.setattr(doctor, "packaging_available", lambda: False)
    report = doctor.diagnose()
    assert report.analysis_unavailable is True
    rendered = doctor.format_report(report)
    assert "Conflict analysis unavailable" in rendered
    assert "No dependency conflicts detected." not in rendered


def test_packaging_unavailable_without_plugins_stays_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no plugins there is nothing to analyze, so ``packaging`` absence is
    a non-issue: the report stays clean and analysis is not flagged."""
    _patch_eps(monkeypatch, [])
    monkeypatch.setattr(doctor, "packaging_available", lambda: False)
    report = doctor.diagnose()
    assert report.analysis_unavailable is False
    assert report.has_conflict is False
    assert "No Standard ASR plugins" in doctor.format_report(report)
