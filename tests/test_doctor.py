# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dependency-conflict doctor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import pytest

from standard_asr.toolchain import doctor


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


def _patch_eps(
    monkeypatch: pytest.MonkeyPatch,
    eps: list[_FakeEP],
    *,
    core_spec: str | None = ">=1.26",
    core_unreadable: bool = False,
) -> None:
    def _entry_points(*, group: str) -> list[_FakeEP]:
        return eps

    monkeypatch.setattr(doctor, "entry_points", _entry_points)
    # Pin the core's own numpy requirement: the real value is
    # interpreter-conditional (>=1.26 below 3.13, >=2.1 at 3.13+), so an
    # unpinned diagnose() would make these tests change verdict with the
    # running interpreter. The default ">=1.26" admits both numpy majors, so
    # plugin-vs-plugin scenarios keep their original meaning; core-conflict
    # tests override it. ``core_unreadable=True`` simulates unreadable core
    # metadata (the pinned function RAISES, matching the real contract);
    # ``core_spec=None`` simulates readable metadata with no
    # interpreter-applicable numpy line.
    if core_unreadable:

        def _raise_unreadable() -> str | None:
            raise doctor.PackageNotFoundError(doctor._CORE_DISTRIBUTION)  # pyright: ignore[reportPrivateUsage]

        monkeypatch.setattr(doctor, "_core_numpy_spec", _raise_unreadable)
    else:
        monkeypatch.setattr(doctor, "_core_numpy_spec", lambda: core_spec)


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
    split = doctor._classify_numpy(spec)  # pyright: ignore[reportPrivateUsage]
    assert split.numpy1_only is False
    assert split.numpy2_required is False


@pytest.mark.parametrize(
    "spec",
    ["==1.26.*", "~=1.26.0", ">=1.21,<1.27", "<2", "==1.26.4"],
)
def test_classify_detects_numpy1_only(spec: str) -> None:
    split = doctor._classify_numpy(spec)  # pyright: ignore[reportPrivateUsage]
    assert split.numpy1_only is True
    assert split.numpy2_required is False


@pytest.mark.parametrize("spec", [">=2", ">=2.1", "==2.*"])
def test_classify_detects_numpy2_required(spec: str) -> None:
    split = doctor._classify_numpy(spec)  # pyright: ignore[reportPrivateUsage]
    assert split.numpy1_only is False
    assert split.numpy2_required is True


@pytest.mark.parametrize("spec", ["==2.2.0", "==2.4.0"])
def test_classify_exact_off_grid_pin_is_numpy2_required(spec: str) -> None:
    """An exact pin to a 2.x version absent from any probe grid must classify as
    numpy-2-only -- the old fixed grid read it as admitting neither major."""
    split = doctor._classify_numpy(spec)  # pyright: ignore[reportPrivateUsage]
    assert split.numpy1_only is False
    assert split.numpy2_required is True


@pytest.mark.parametrize("spec", [None, "(any)", "", "not-a-spec"])
def test_classify_returns_neutral_for_unconstrained_or_invalid(
    spec: str | None,
) -> None:
    assert doctor._classify_numpy(spec) == doctor._NumpyMajorSplit()  # pyright: ignore[reportPrivateUsage]


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
    intersection conflict the 1.x/2.x major split alone would miss."""
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


def test_intersection_satisfiability_helper_direct() -> None:
    """The helper answers UNSAT for disjoint sets and SAT for overlapping ones
    (exact-oracle path; the legacy fallback is simulated in dedicated tests)."""
    from packaging.specifiers import SpecifierSet

    assert (
        doctor._intersection_satisfiability(  # pyright: ignore[reportPrivateUsage]
            [SpecifierSet("==2.0.*"), SpecifierSet(">=2.3")]
        )
        is doctor.Satisfiability.UNSAT
    )
    assert (
        doctor._intersection_satisfiability(  # pyright: ignore[reportPrivateUsage]
            [SpecifierSet(">=1.26"), SpecifierSet(">=2.1")]
        )
        is doctor.Satisfiability.SAT
    )


@pytest.mark.parametrize(
    "specs",
    [
        # High pins that all land ABOVE the old bounded grid but share a
        # satisfying version -- previously misread as an empty intersection.
        [">=2.40", ">=2.1"],
        ["==2.45.*", ">=2.40"],
        [">=3.0", ">=2.1"],
        ["~=2.5", ">=2.9"],
    ],
)
def test_high_pins_are_not_falsely_empty(specs: list[str]) -> None:
    """A satisfiable intersection of high pins must NOT read as empty just
    because every satisfying version sits above any fixed grid."""
    from packaging.specifiers import SpecifierSet

    sets = [SpecifierSet(s) for s in specs]
    assert (
        doctor._intersection_satisfiability(sets)  # pyright: ignore[reportPrivateUsage]
        is doctor.Satisfiability.SAT
    )


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
    assert (
        doctor._intersection_satisfiability(sets)  # pyright: ignore[reportPrivateUsage]
        is doctor.Satisfiability.UNSAT
    )


def test_arbitrary_equality_edge_is_skipped_not_crash() -> None:
    """A ``===`` arbitrary-equality edge never crashes the analysis.

    ``===foobar`` matches only the literal 'foobar', which no numeric release
    satisfies, so intersecting it with ``>=2.1`` is genuinely empty -- the
    exact oracle (packaging >= 26) answers UNSAT. The legacy path (where
    membership itself re-parses 'foobar' and raises) is simulated in
    ``test_legacy_witness_loop_survives_raising_membership``.
    """
    from packaging.specifiers import SpecifierSet

    assert (
        doctor._intersection_satisfiability(  # pyright: ignore[reportPrivateUsage]
            [SpecifierSet("===foobar"), SpecifierSet(">=2.1")]
        )
        is doctor.Satisfiability.UNSAT
    )


def _force_legacy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a legacy ``packaging`` without the exact emptiness oracle.

    Patches the oracle seam (``_exact_unsatisfiable``) to answer "no oracle",
    so ``_intersection_satisfiability`` exercises the finite witness search
    exactly as it would on ``packaging < 26.1``.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """

    def _no_oracle(_combined: object) -> None:
        return None

    monkeypatch.setattr(doctor, "_exact_unsatisfiable", _no_oracle)


def test_epoch_range_is_satisfiable_exact_and_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """``>1!1.0,<1!1.1`` is satisfiable (witness ``1!1.0.1``) on BOTH paths.

    The release-derived candidates once dropped the edge's epoch, so every
    epoch interval over final releases -- even the trivially satisfiable
    ``>1!1.0`` -- was misreported as an internally unsatisfiable hard
    conflict. The exact oracle decides it; the fallback must now find the
    epoch-preserving witness itself.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    for spec in (">1!1.0,<1!1.1", ">1!1.0"):
        combined = SpecifierSet(spec)
        assert Version("1!1.0.1") in combined  # the interval really is satisfiable
        assert (
            doctor._intersection_satisfiability([combined])  # pyright: ignore[reportPrivateUsage]
            is doctor.Satisfiability.SAT
        )
    _force_legacy_fallback(monkeypatch)
    assert (
        doctor._intersection_satisfiability(  # pyright: ignore[reportPrivateUsage]
            [SpecifierSet(">1!1.0,<1!1.1")]
        )
        is doctor.Satisfiability.SAT
    )


def test_epoch_range_plugin_no_false_conflict_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin pinning an epoch range must not be branded self-contradictory.

    Args:
        monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["numpy>1!1.0,<1!1.1"]))],
    )
    report = doctor.diagnose()
    assert not any("internally unsatisfiable" in c for c in report.conflicts), report.conflicts


def test_legacy_fallback_answers_unknown_never_unsat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the exact oracle, no-witness-found is UNKNOWN -- never UNSAT.

    A finite witness search can prove SAT but cannot certify emptiness; the
    old bool collapse of this distinction is exactly what convicted
    satisfiable environments.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    from packaging.specifiers import SpecifierSet

    _force_legacy_fallback(monkeypatch)
    # A genuinely empty intersection: the fallback must refuse to convict.
    assert (
        doctor._intersection_satisfiability(  # pyright: ignore[reportPrivateUsage]
            [SpecifierSet(">2"), SpecifierSet("<1")]
        )
        is doctor.Satisfiability.UNKNOWN
    )
    # A satisfiable one is still proven by its witness.
    assert (
        doctor._intersection_satisfiability(  # pyright: ignore[reportPrivateUsage]
            [SpecifierSet(">=1.26"), SpecifierSet("<2")]
        )
        is doctor.Satisfiability.SAT
    )


def test_legacy_arbitrary_equality_edge_is_skipped_in_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the fallback path a ``===`` edge is skipped during candidate derivation.

    ``foobar`` is not a PEP 440 version, so it contributes no candidates; with
    no witness findable the verdict is UNKNOWN -- never a crash, never UNSAT.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    from packaging.specifiers import SpecifierSet

    _force_legacy_fallback(monkeypatch)
    verdict = doctor._intersection_satisfiability(  # pyright: ignore[reportPrivateUsage]
        [SpecifierSet("===foobar"), SpecifierSet(">=2.1")]
    )
    assert verdict is doctor.Satisfiability.UNKNOWN


def test_legacy_witness_loop_survives_raising_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate whose membership test raises is skipped, not a crash.

    On ``packaging <= 25.0`` an arbitrary-equality ``===`` specifier
    re-parses its non-PEP440 version inside ``__contains__`` and raises
    ``InvalidVersion`` for every probed candidate -- the old unguarded loop
    crashed the whole doctor run on it. Simulated by injecting an unparseable
    candidate (the raise happens at ``Version(candidate)``, inside the same
    guard); a later valid candidate must still prove SAT.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    from packaging.specifiers import SpecifierSet

    _force_legacy_fallback(monkeypatch)

    def _candidates_with_a_raiser(_combined: object) -> set[str]:
        return {"not!!a!!version", "1.26.4"}

    monkeypatch.setattr(doctor, "_emptiness_candidates", _candidates_with_a_raiser)
    verdict = doctor._intersection_satisfiability(  # pyright: ignore[reportPrivateUsage]
        [SpecifierSet("<2")]
    )
    assert verdict is doctor.Satisfiability.SAT

    def _only_raisers(_combined: object) -> set[str]:
        return {"not!!a!!version"}

    monkeypatch.setattr(doctor, "_emptiness_candidates", _only_raisers)
    verdict = doctor._intersection_satisfiability(  # pyright: ignore[reportPrivateUsage]
        [SpecifierSet("<2")]
    )
    assert verdict is doctor.Satisfiability.UNKNOWN


def test_legacy_undecidable_environment_is_analysis_unavailable_not_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end on legacy packaging: undecidable relations are disclosed.

    A 1.x plugin beside a 2.x plugin cannot be proven conflicting without the
    exact oracle, so the report must not convict -- and must not read clean
    either: analysis_unavailable (still a non-zero doctor exit) with ONE note
    naming every undecided relation (a self-satisfiability, a vs-core, the
    1.x/2.x classifications, and the joint intersection) plus the packaging
    upgrade path.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _force_legacy_fallback(monkeypatch)
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("old/funasr", _FakeDist("std-funasr", ["numpy<2"])),
            _FakeEP("new/qwen", _FakeDist("std-qwen", ["numpy>=2.1"])),
            # A genuinely self-contradictory pin: convictable only exactly.
            _FakeEP("bad/self", _FakeDist("std-selfbad", ["numpy<2", "numpy>=2.1"])),
            # Shares nothing with the core floor (>=1.26): vs-core undecided.
            _FakeEP("old/ancient", _FakeDist("std-ancient", ["numpy<1.0"])),
        ],
    )
    report = doctor.diagnose()
    assert report.has_conflict is False, report.conflicts
    assert report.analysis_unavailable is True
    assert report.is_clean is False
    note = next(n for n in report.notes if "could not be decided exactly" in n)
    assert "is_unsatisfiable" in note
    assert "std-selfbad (<2,>=2.1): self-satisfiability" in note
    assert "std-ancient (<1.0): vs standard-asr core" in note
    assert "1.x/2.x classification" in note
    rendered = doctor.format_report(report)
    assert "Conflict analysis unavailable or incomplete" in rendered


@pytest.mark.parametrize(
    ("spec", "witness"),
    [
        # The same pre/dev/post windows as the exact-path parametrization:
        # on legacy packaging only the boundary-derived candidates (segment
        # neighbors, the .dev0 of an upper edge, epoch-preserving release
        # neighbors) can prove these SAT -- this is the lane that keeps the
        # fallback generator honest now that the exact oracle answers first
        # on modern packaging.
        (">2.0rc1,<2.0rc3", "2.0rc2"),
        (">2.0a1,<2.0a3", "2.0a2"),
        (">2.0.dev1,<2.0.dev3", "2.0.dev2"),
        (">2.0.post1,<2.0.post3", "2.0.post2"),
        (">2.0rc1.dev1,<2.0rc1.dev3", "2.0rc1.dev2"),
        (">2.0rc1.post1,<2.0rc1.post3", "2.0rc1.post2"),
        (">2.0.post1.dev1,<2.0.post1.dev3", "2.0.post1.dev2"),
        (">1!2.0rc1,<1!2.0rc3", "1!2.0rc2"),
        (">2.0rc0,<2.0rc2", "2.0rc1"),
        (">2.0.post0,<2.0.post2", "2.0.post1"),
        (">2.0.dev0,<2.0.dev2", "2.0.dev1"),
        (">2.0rc1,<2.0rc2", "2.0rc2.dev1"),
        (">2.0,<2.1", "2.0.1"),
        ("==2.1.0rc1", "2.1.0rc1"),
    ],
)
def test_legacy_windows_are_witnessed_without_the_oracle(
    monkeypatch: pytest.MonkeyPatch, spec: str, witness: str
) -> None:
    """The fallback candidate generator still witnesses every covered window.

    With modern ``packaging`` the exact oracle answers first, so these
    parametrizations are the only lane exercising the boundary-derived
    candidates -- a regression there would otherwise hide behind the oracle
    until someone runs doctor on a legacy environment.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
            spec: The version specifier under test (parametrized).
            witness: The expected witness version (parametrized).
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    _force_legacy_fallback(monkeypatch)
    combined = SpecifierSet(spec)
    assert Version(witness) in combined  # the window really is satisfiable
    assert (
        doctor._intersection_satisfiability([combined])  # pyright: ignore[reportPrivateUsage]
        is doctor.Satisfiability.SAT
    )


def test_exact_oracle_defensive_shapes_answer_no_oracle() -> None:
    """An absent, raising, or non-bool ``is_unsatisfiable`` is "no oracle".

    The oracle is third-party surface; a broken answer must not crash the
    doctor nor be trusted as a verdict -- the caller falls back to witness
    probing. A real ``SpecifierSet`` on the pinned ``packaging`` answers a
    real bool.
    """
    from typing import Any, cast

    from packaging.specifiers import SpecifierSet

    class _NoOracle:
        pass

    class _RaisingOracle:
        def is_unsatisfiable(self) -> bool:
            raise RuntimeError("oracle exploded")

    class _NonBoolOracle:
        def is_unsatisfiable(self) -> str:
            return "yes"

    exact = doctor._exact_unsatisfiable  # pyright: ignore[reportPrivateUsage]
    assert exact(cast("Any", _NoOracle())) is None
    assert exact(cast("Any", _RaisingOracle())) is None
    assert exact(cast("Any", _NonBoolOracle())) is None
    assert exact(SpecifierSet(">2,<1")) is True
    assert exact(SpecifierSet(">=1.26")) is False


def test_classify_numpy_undecided_on_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """On legacy packaging a one-sided spec is undecided, not a hard side.

    ``<2`` admits 1.x by witness, but "admits no 2.x" needs the exact oracle;
    asserting numpy1-only from a failed witness search is the exact bool
    collapse this classifier no longer performs.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _force_legacy_fallback(monkeypatch)
    split = doctor._classify_numpy("<2")  # pyright: ignore[reportPrivateUsage]
    assert split == doctor._NumpyMajorSplit(undecided=True)  # pyright: ignore[reportPrivateUsage]
    # An unconstrained spec is witness-satisfiable on BOTH majors: fully
    # decided (all-False split), no false undecided noise.
    assert doctor._classify_numpy("(any)") == doctor._NumpyMajorSplit()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("specs", "witness"),
    [
        # Strict-lower vs next-edge upper. ``2.0.1`` satisfies both
        # but no candidate witnessed it before the one-level-deeper probe.
        ([">2.0", "<2.1"], "2.0.1"),
        ([">2.0,<2.1"], "2.0.1"),
        # Sub-release-width interval. ``2.0.0.1`` (or the high-filled
        # ``2.0.0.99999`` probe) satisfies both; the old just-below probe stopped
        # at ``2.0.0`` which fails the strict lower bound.
        ([">2.0", "<2.0.1"], "2.0.0.1"),
    ],
)
def test_strict_boundary_intervals_are_not_falsely_empty(specs: list[str], witness: str) -> None:
    """a satisfiable interval adjacent to a strict or sub-release
    boundary must NOT read as empty. ``packaging`` confirms the witness sits
    inside the intersection, so the empty verdict would be a false conflict."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    sets = [SpecifierSet(s) for s in specs]
    combined = SpecifierSet()
    for s in sets:
        combined &= s
    assert Version(witness) in combined  # the interval really is satisfiable
    assert (
        doctor._intersection_satisfiability(sets)  # pyright: ignore[reportPrivateUsage]
        is doctor.Satisfiability.SAT
    )


@pytest.mark.parametrize("spec", ["==2.1.0rc1", "==1!2.0", "==1!1.0"])
def test_prerelease_and_epoch_pins_are_not_falsely_empty(spec: str) -> None:
    """an epoch or pre-release pin is satisfiable by its own version,
    which the verbatim edge candidate now witnesses (the old bare-``release``
    candidate dropped the epoch/pre-release segment and misread it as empty)."""
    from packaging.specifiers import SpecifierSet

    assert (
        doctor._intersection_satisfiability([SpecifierSet(spec)])  # pyright: ignore[reportPrivateUsage]
        is doctor.Satisfiability.SAT
    )


@pytest.mark.parametrize(
    ("spec", "witness"),
    [
        # Same-kind pre/dev/post windows: every final-release candidate sorts
        # OUTSIDE the window, so only a within-segment neighbor
        # (_segment_neighbor_candidates) can witness satisfiability. These
        # four were real false 'hard conflict' verdicts before the fix.
        (">2.0rc1,<2.0rc3", "2.0rc2"),
        (">2.0a1,<2.0a3", "2.0a2"),
        (">2.0.dev1,<2.0.dev3", "2.0.dev2"),
        (">2.0.post1,<2.0.post3", "2.0.post2"),
        # Nested segments: the neighbor must be rebuilt around the OUTER
        # segments (dev under a pre-release; post under a pre-release; dev
        # under a post-release), and under a non-zero epoch.
        (">2.0rc1.dev1,<2.0rc1.dev3", "2.0rc1.dev2"),
        (">2.0rc1.post1,<2.0rc1.post3", "2.0rc1.post2"),
        (">2.0.post1.dev1,<2.0.post1.dev3", "2.0.post1.dev2"),
        (">1!2.0rc1,<1!2.0rc3", "1!2.0rc2"),
        # Zero-indexed edges: the downward neighbor would be negative and is
        # skipped; the upward neighbor still witnesses the window.
        (">2.0rc0,<2.0rc2", "2.0rc1"),
        (">2.0.post0,<2.0.post2", "2.0.post1"),
        (">2.0.dev0,<2.0.dev2", "2.0.dev1"),
        # ADJACENT same-kind window: no rc sits between rc1 and rc2, and PEP
        # 440 bars the rc1.post* family from ``>2.0rc1`` -- but a dev release
        # of the UPPER edge is admitted, and the edge-derived ``X.dev0``
        # candidate witnesses it.
        (">2.0rc1,<2.0rc2", "2.0rc2.dev1"),
    ],
)
def test_pre_dev_post_windows_are_not_falsely_empty(spec: str, witness: str) -> None:
    """A satisfiable window between two same-kind pre/dev/post releases must
    NOT read as empty: doctor turned these into hard 'internally
    unsatisfiable' / 'no common version' conflicts (and a non-zero exit)
    against perfectly valid PEP 440 requirements.

        Args:
            spec: The version specifier under test (parametrized).
            witness: The expected witness version (parametrized).
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    combined = SpecifierSet(spec)
    assert Version(witness) in combined  # the window really is satisfiable
    assert (
        doctor._intersection_satisfiability([combined])  # pyright: ignore[reportPrivateUsage]
        is doctor.Satisfiability.SAT
    )


def test_prerelease_window_requirement_no_false_conflict_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin pinning a pre-release window must not be branded
    self-contradictory by relation 0 (a false hard conflict, non-zero exit).

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["numpy>2.0rc1,<2.0rc3"]))],
    )
    report = doctor.diagnose()
    assert not any("internally unsatisfiable" in c for c in report.conflicts), report.conflicts


def test_strict_boundary_pin_no_false_conflict_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """two plugins pinned ``>2.0`` and ``<2.1`` share ``2.0.1``; the
    old probe falsely reported 'no common satisfying version' + exit 1."""
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("a/x", _FakeDist("std-a", ["numpy>2.0"])),
            _FakeEP("b/y", _FakeDist("std-b", ["numpy<2.1"])),
        ],
    )
    report = doctor.diagnose()
    assert report.has_conflict is False, report.conflicts


def test_single_plugin_internally_unsatisfiable_is_an_exact_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An 'internally unsatisfiable' conflict is exact-oracle-backed, said so.

    UNSAT can only come from packaging's specifier algebra (the fallback
    witness search answers UNKNOWN, never UNSAT), so the message asserts the
    verdict absolutely instead of hedging with a report-a-bug invitation.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["numpy<2", "numpy>=2.1"]))],
    )
    report = doctor.diagnose()
    assert report.has_conflict is True
    assert any("computed exactly" in c for c in report.conflicts)
    assert not any("report a bug" in c for c in report.conflicts)


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


def test_conflict_lists_each_distribution_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a distribution shipping several presets carries one
    PluginNumpy per entry point with the same numpy spec; the conflict text must
    list ``std-foo (<2)`` once, not repeat it per preset and inflate the
    apparent conflict size."""
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("std-foo/p1", _FakeDist("std-foo", ["numpy<2"])),
            _FakeEP("std-foo/p2", _FakeDist("std-foo", ["numpy<2"])),
            _FakeEP("std-foo/p3", _FakeDist("std-foo", ["numpy<2"])),
            _FakeEP("std-bar/x", _FakeDist("std-bar", ["numpy>=2.1"])),
        ],
    )
    report = doctor.diagnose()
    assert report.has_conflict is True
    conflict = next(c for c in report.conflicts if "1.x vs 2.x" in c)
    # Listed once on the numpy<2 side, not three times.
    assert conflict.count("std-foo (<2)") == 1
    assert conflict.count("std-bar (>=2.1)") == 1


def test_py313_wheel_note_lists_each_distribution_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """the Python 3.13 'no numpy<2 wheel' note is distribution-scoped
    too, so a multi-preset numpy1 distribution appears once."""
    # core_spec=">=2.1" is the REAL core floor on 3.13, so a numpy<2 plugin is
    # ALSO core-incompatible and gets excluded from the environment-level
    # participants. The wheel note is computed over ALL plugins precisely so
    # that exclusion cannot swallow it -- the missing wheel is an independent
    # fact about the interpreter. (An earlier ">=1.26" fixture paired the 3.13
    # interpreter with the sub-3.13 core floor: an impossible combination that
    # hid the interaction.)
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("std-foo/p1", _FakeDist("std-foo", ["numpy<2"])),
            _FakeEP("std-foo/p2", _FakeDist("std-foo", ["numpy<2"])),
        ],
        core_spec=">=2.1",
    )
    fake_vi = _VersionInfo(3, 13, 0, "final", 0)
    monkeypatch.setattr(doctor.sys, "version_info", fake_vi)
    report = doctor.diagnose()
    note = next(c for c in report.conflicts if "no numpy<2 wheel" in c)
    assert note.count("std-foo") == 1
    # Both conflicts fire: the plugin-vs-core incompatibility AND the wheel fact.
    assert any("conflict with standard-asr core" in c for c in report.conflicts)


def test_single_distribution_many_presets_unsatisfiable_is_one_offender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """an internally-unsatisfiable declaration shipped across several
    presets of ONE distribution is a single self-contradiction. The single-vs-
    cross framing keys on distinct distributions, so it must read as one
    offender ('declares an internally unsatisfiable'), not a cross-plugin
    'cannot share one process'."""
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("std-baz/p1", _FakeDist("std-baz", ["numpy<2", "numpy>=2.1"])),
            _FakeEP("std-baz/p2", _FakeDist("std-baz", ["numpy<2", "numpy>=2.1"])),
        ],
    )
    report = doctor.diagnose()
    assert report.has_conflict is True
    # Deduplicated to ONE conflict: relation 0 is per DISTRIBUTION, so both
    # presets of one broken distribution are a single fix, not two.
    assert len(report.conflicts) == 1, report.conflicts
    assert "internally unsatisfiable" in report.conflicts[0]
    assert not any("cannot share one process" in c for c in report.conflicts)


def test_self_contradictory_plugin_does_not_co_blame_an_innocent_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-2 regression: a self-contradictory distribution is judged ALONE.

    ``std-broken`` pins ``<2,>=2.1`` -- no numpy anywhere satisfies it -- while
    ``std-good`` (``>=2``) is perfectly compatible with the core floor. Keeping
    the broken one in the joint intersection made the whole environment read as
    a cross-plugin conflict: std-good was named as an offender it is not, and
    the remedy offered was out-of-process isolation, which cannot fix a pin no
    version satisfies. Relation 0 now reports the self-contradiction per
    distribution and excludes it from every later relation.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("broken/x", _FakeDist("std-broken", ["numpy<2", "numpy>=2.1"])),
            _FakeEP("good/y", _FakeDist("std-good", ["numpy>=2"])),
        ],
    )
    report = doctor.diagnose()

    assert report.has_conflict is True
    assert len(report.conflicts) == 1, report.conflicts
    conflict = report.conflicts[0]
    assert "internally unsatisfiable" in conflict
    assert "std-broken" in conflict
    # The innocent plugin is neither named nor implicated...
    assert "std-good" not in conflict
    # ...and no cross-plugin framing or isolation advice is offered for a fault
    # that isolation cannot fix.
    assert "cannot share one process" not in conflict
    assert "separate processes" not in conflict
    assert "out-of-process" not in conflict


def test_two_distinct_self_contradictory_plugins_each_get_their_own_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relation 0 is PER DISTRIBUTION: two broken pins are two separate fixes.

    Merging them into one message would make the author fix one and re-run into
    a verdict that looks unchanged.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("broken/x", _FakeDist("std-broken", ["numpy<2", "numpy>=2.1"])),
            _FakeEP("other/y", _FakeDist("std-other", ["numpy<1.20", "numpy>=2.3"])),
        ],
    )
    report = doctor.diagnose()

    assert len(report.conflicts) == 2, report.conflicts
    assert all("internally unsatisfiable" in c for c in report.conflicts)
    named = {"std-broken" if "std-broken" in c else "std-other" for c in report.conflicts}
    assert named == {"std-broken", "std-other"}


def test_single_plugin_internally_unsatisfiable_is_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SINGLE plugin whose own numpy declaration is internally unsatisfiable
    (``<2`` AND ``>=2.1``) must be flagged, not silently passed."""
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["numpy<2", "numpy>=2.1"]))],
    )
    report = doctor.diagnose()
    assert report.has_conflict is True
    conflict = next(c for c in report.conflicts if "internally unsatisfiable" in c)
    # A single offender is a self-contradiction, not a cross-plugin conflict --
    # and with core now in the intersection, the offender attribution must
    # still name ONLY the plugin (blaming standard-asr for a plugin's
    # self-contradictory pin would misdirect the fix).
    assert "standard-asr" not in conflict
    assert not any("cannot share one process" in c for c in report.conflicts)


def test_canonical_dual_line_resolves_to_numpy2_on_py313(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canonical interpreter-conditional dual-line form must resolve
    to the line whose marker holds on the running interpreter -- on 3.13 that is
    ``>=2.1``, so no bogus 'no numpy<2 wheel' conflict fires."""
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
    ignored entirely, not treated as an active constraint."""
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
    real specifier rather than being swallowed as unconstrained."""
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
    # The unclassified state is explicit, never a silent "all clean".
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
    """On 3.13 a numpy<2 plugin yields BOTH the core conflict and the wheel note.

    The two say different things and neither substitutes for the other: the
    core conflict is about this environment's resolution, the wheel note is
    about the interpreter (no numpy<2 wheel exists for 3.13 in ANY process).
    Since the core floor on 3.13 is ``>=2.1``, the plugin is excluded from the
    environment-level participants -- so a wheel note computed over the
    filtered participants would silently never fire on the exact interpreter it
    describes.
    """
    _patch_eps(
        monkeypatch,
        [_FakeEP("old/funasr", _FakeDist("std-funasr", ["numpy<2"]))],
        core_spec=">=2.1",
    )
    fake_vi = _VersionInfo(3, 13, 0, "final", 0)
    monkeypatch.setattr(doctor.sys, "version_info", fake_vi)
    report = doctor.diagnose()
    note = next(c for c in report.conflicts if "no numpy<2 wheel" in c)
    assert "std-funasr" in note
    assert any("conflict with standard-asr core" in c for c in report.conflicts)
    # The remediation is honest about scope: the wheel is missing in every
    # process on this interpreter, so in-process isolation is NOT a fix. The old
    # "or isolate the plugin" phrasing advertised exactly that dead end.
    assert "cannot be installed on this interpreter in any process" in note
    assert "Run that plugin under Python <3.13" in note
    assert "or isolate the plugin" not in note


def test_no_wheel_note_below_py313(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below 3.13 numpy<2 wheels exist, so the note must NOT fire.

    The note is interpreter-scoped, not a blanket "numpy<2 is bad" warning:
    with the sub-3.13 core floor (``>=1.26``) a numpy<2 plugin resolves fine and
    the report stays clean.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [_FakeEP("old/funasr", _FakeDist("std-funasr", ["numpy<2"]))],
    )
    monkeypatch.setattr(doctor.sys, "version_info", _VersionInfo(3, 12, 0, "final", 0))
    report = doctor.diagnose()
    assert not any("no numpy<2 wheel" in c for c in report.conflicts)
    assert report.has_conflict is False


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
    assert doctor._classify_numpy("<2") == doctor._NumpyMajorSplit()  # pyright: ignore[reportPrivateUsage]


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
    conflicts" -- doctor cannot prove the environment conflict-free."""
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["numpy>=1.26"]))],
    )
    monkeypatch.setattr(doctor, "packaging_available", lambda: False)
    report = doctor.diagnose()
    assert report.analysis_unavailable is True
    rendered = doctor.format_report(report)
    # The headline is deliberately GENERALIZED (it now covers the unreadable-core
    # gap too); the packaging-specific cause and its fix live in the note.
    assert "Conflict analysis unavailable or incomplete" in rendered
    assert "No dependency conflicts detected." not in rendered
    assert any("optional 'packaging' library" in n for n in report.notes)


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


def test_core_numpy_spec_reads_marker_evaluated_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core's interpreter-conditional dual-line numpy declaration must
    resolve to the line whose marker holds on the running interpreter.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """

    def _requires(name: str) -> list[str]:
        assert name == "standard-asr"
        return [
            'numpy>=1.26; python_version < "3.13"',
            'numpy>=2.1; python_version >= "3.13"',
            "pydantic>=2.5",
        ]

    monkeypatch.setattr(doctor, "requires", _requires)
    monkeypatch.setattr(doctor.sys, "version_info", _VersionInfo(3, 13, 0, "final", 0))
    assert doctor._core_numpy_spec() == ">=2.1"  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(doctor.sys, "version_info", _VersionInfo(3, 10, 0, "final", 0))
    assert doctor._core_numpy_spec() == ">=1.26"  # pyright: ignore[reportPrivateUsage]


def test_core_numpy_spec_raises_when_core_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreadable core metadata PROPAGATES so the caller can tell it apart.

    Returning None here would conflate a broken install with the legitimate
    "readable metadata, no interpreter-applicable numpy line" state -- the
    caller must report the first as a non-clean analysis gap and the second
    as a mere informational note.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """

    def _requires(name: str) -> list[str]:
        raise doctor.PackageNotFoundError(name)

    monkeypatch.setattr(doctor, "requires", _requires)
    with pytest.raises(doctor.PackageNotFoundError):
        doctor._core_numpy_spec()  # pyright: ignore[reportPrivateUsage]


def test_core_numpy_spec_none_when_no_applicable_numpy_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readable metadata without a numpy line returns None (a declaration fact).

    Args:
        monkeypatch: Pytest fixture for attribute patching.
    """

    def _requires(name: str) -> list[str]:
        return ["pydantic>=2.5"]

    monkeypatch.setattr(doctor, "requires", _requires)
    assert doctor._core_numpy_spec() is None  # pyright: ignore[reportPrivateUsage]


def test_diagnose_readable_core_without_numpy_line_stays_clean_with_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No applicable core numpy line is NOT "unreadable metadata".

    A repackaged/vendored core whose requirements were rewritten without a
    numpy line is a working environment: doctor must not false-alarm a CI
    gate with an "unreadable" claim and a red exit -- it discloses that the
    plugin-vs-core relation has nothing to check and stays clean.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["numpy>=1.26"]))],
        core_spec=None,
    )
    report = doctor.diagnose()
    assert report.core is None
    assert report.analysis_unavailable is False
    assert report.is_clean is True
    assert any("declares no numpy requirement" in n for n in report.notes)
    assert not any("unreadable" in n for n in report.notes)


#: The core's real declaration shape: interpreter-conditional, dual-line.
_CORE_DUAL_DECLARATION = [
    'numpy>=1.26; python_version < "3.13"',
    'numpy>=2.1; python_version >= "3.13"',
    "pydantic>=2.5",
]


def _block_packaging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``packaging`` import fail (the optional-dependency-absent mode)."""
    import builtins

    real_import = builtins.__import__

    def _import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("packaging"):
            raise ImportError("no packaging")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _import)


def _patch_core_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the core's real dual-line numpy declaration from ``requires``."""

    def _requires(name: str) -> list[str]:
        assert name == "standard-asr"
        return _CORE_DUAL_DECLARATION

    monkeypatch.setattr(doctor, "requires", _requires)


def _patch_raw_entry_points(monkeypatch: pytest.MonkeyPatch, eps: list[_FakeEP]) -> None:
    """Patch ONLY ``entry_points``, leaving the real ``_core_numpy_spec`` in play.

    ``_patch_eps`` pins ``_core_numpy_spec`` to a fixed value, which is exactly
    what the core-row tests below need to exercise for real.
    """

    def _entry_points(*, group: str) -> list[_FakeEP]:
        return eps

    monkeypatch.setattr(doctor, "entry_points", _entry_points)


def test_core_numpy_spec_none_when_packaging_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``packaging`` the core row is omitted rather than shown wrong.

    The display fallback the plugin rows use is marker-BLIND: it returns the
    first matching line, ``>=1.26``, which is factually the wrong core floor on
    a 3.13+ interpreter (the effective one is ``>=2.1``). Showing that in the
    very row added to expose core conflicts would misinform; ``None`` is the
    honest answer, and the report is already flagged analysis-unavailable.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _block_packaging(monkeypatch)
    _patch_core_declaration(monkeypatch)
    monkeypatch.setattr(doctor.sys, "version_info", _VersionInfo(3, 13, 0, "final", 0))

    # What the marker-blind fallback WOULD have produced, and why it is refused.
    assert (
        doctor._numpy_spec_for(_CORE_DUAL_DECLARATION)  # pyright: ignore[reportPrivateUsage]
        == ">=1.26"
    )
    assert doctor._core_numpy_spec() is None  # pyright: ignore[reportPrivateUsage]


def test_diagnose_without_packaging_omits_core_row_and_its_unreadable_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The packaging-absent mode shows no core row and does NOT claim it is unreadable.

    ``_core_numpy_spec`` returns ``None`` for two different reasons; the
    "metadata is unreadable" note describes only the anomalous-install one. It
    must not fire in the packaging-absent mode, where the metadata is perfectly
    readable and the analysis-unavailable state already discloses the gap --
    otherwise every packaging-less environment is told its install is broken.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _block_packaging(monkeypatch)
    _patch_raw_entry_points(monkeypatch, [_FakeEP("a/x", _FakeDist("std-a", ["numpy<2"]))])
    _patch_core_declaration(monkeypatch)

    report = doctor.diagnose()

    assert report.core is None
    assert not any("distribution metadata is unreadable" in n for n in report.notes)
    # The gap IS disclosed -- by the analysis-unavailable state plus the note
    # naming the ACTUAL cause (a missing optional dependency, not a broken install).
    assert report.analysis_unavailable is True
    assert any("optional 'packaging' library" in n for n in report.notes)


def test_core_metadata_unreadable_note_needs_packaging_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``packaging`` present, genuinely unreadable core metadata still notes it.

    The note names the ANOMALOUS-INSTALL gap specifically (distinct from the
    packaging-absent gap), and flags the analysis as incomplete so the run
    cannot exit clean.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_raw_entry_points(monkeypatch, [_FakeEP("a/x", _FakeDist("std-a", ["numpy>=1.26"]))])

    def _requires(name: str) -> list[str]:
        raise doctor.PackageNotFoundError(name)

    monkeypatch.setattr(doctor, "requires", _requires)

    report = doctor.diagnose()

    assert report.core is None
    assert any("distribution metadata is unreadable" in n for n in report.notes)
    assert any("plugin-vs-core numpy analysis could not run" in n for n in report.notes)
    # This is the packaging-PRESENT gap, so the packaging note must not fire.
    assert not any("optional 'packaging' library" in n for n in report.notes)
    assert report.analysis_unavailable is True
    assert report.is_clean is False


def test_core_floor_conflicts_with_numpy1_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin pinned ``numpy<2`` against a core floor of ``>=2.1`` (the real
    3.13+ floor) is a hard conflict even with zero plugin-vs-plugin conflicts --
    the exact gap doctor previously missed by leaving core out of the analysis.
    It is reported as a DEDICATED plugin-vs-core conflict (core is in every
    process, so the remediation must say isolation cannot help; out-of-process
    advice would be wrong here), and no environment-level message fires for a
    plugin that cannot run in any layout.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [_FakeEP("old/funasr", _FakeDist("std-funasr", ["numpy<2"]))],
        core_spec=">=2.1",
    )
    report = doctor.diagnose()
    assert report.has_conflict is True
    conflict = next(c for c in report.conflicts if "conflict with standard-asr core" in c)
    assert "std-funasr (<2)" in conflict
    assert "numpy >=2.1" in conflict
    assert "isolation cannot help" in conflict
    assert "out-of-process" not in conflict
    assert not any("1.x vs 2.x" in c for c in report.conflicts)


def test_plugin_only_conflict_still_advises_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When core admits both majors (the sub-3.13 floor), a plugin-vs-plugin
    split keeps the out-of-process remediation -- the core-specific text must
    fire only when core is genuinely a conflict side.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("old/funasr", _FakeDist("std-funasr", ["numpy<2"])),
            _FakeEP("new/qwen", _FakeDist("std-qwen", ["numpy>=2.1"])),
        ],
    )
    report = doctor.diagnose()
    conflict = next(c for c in report.conflicts if "1.x vs 2.x" in c)
    assert "out-of-process" in conflict
    assert "isolation cannot help" not in conflict


def test_core_in_range_intersection_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-major disjointness against the core floor (no 1.x/2.x split to
    catch it) is a plugin-vs-core incompatibility: reported as the dedicated
    core conflict with the core remediation.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["numpy==2.0.*"]))],
        core_spec=">=2.3",
    )
    report = doctor.diagnose()
    assert report.has_conflict is True
    conflict = next(c for c in report.conflicts if "conflict with standard-asr core" in c)
    assert "std-a (==2.0.*)" in conflict
    assert "numpy >=2.3" in conflict
    assert "isolation cannot help" in conflict


def test_core_conflict_reported_once_per_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A distribution shipping several presets carries one entry point each with
    the SAME numpy spec; a core incompatibility belongs to the distribution, so
    it must be reported once -- not once per preset (mirroring the
    distribution-scoped dedup of the environment-level messages).

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("std-foo/p1", _FakeDist("std-foo", ["numpy<2"])),
            _FakeEP("std-foo/p2", _FakeDist("std-foo", ["numpy<2"])),
        ],
        core_spec=">=2.1",
    )
    report = doctor.diagnose()
    core_conflicts = [c for c in report.conflicts if "conflict with standard-asr core" in c]
    assert len(core_conflicts) == 1
    assert "std-foo (<2)" in core_conflicts[0]


def test_mixed_shape_reports_core_conflict_not_misleading_isolation_advice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plugins that are ALSO mutually conflicting must not mask a plugin-vs-core
    incompatibility: ``std-a ==2.0.*`` conflicts with both ``std-b >=2.3`` and a
    ``>=2.1`` core. Isolating std-a cannot help (core is in the worker too), so
    the report must carry the dedicated std-a-vs-core conflict -- and must NOT
    advise isolating a plugin that is dead in any process layout. std-b remains
    compatible with core and with the environment, so no further conflict fires.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("a/x", _FakeDist("std-a", ["numpy==2.0.*"])),
            _FakeEP("b/y", _FakeDist("std-b", ["numpy>=2.3"])),
        ],
        core_spec=">=2.1",
    )
    report = doctor.diagnose()
    assert report.has_conflict is True
    conflict = next(c for c in report.conflicts if "conflict with standard-asr core" in c)
    assert "std-a (==2.0.*)" in conflict
    assert "std-b" not in conflict
    assert not any("out-of-process" in c for c in report.conflicts)
    assert not any("no common satisfying version" in c for c in report.conflicts)


def test_exotic_joint_emptiness_falls_back_to_core_attributed_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-interval shape: every plugin is individually core-compatible and the
    plugins agree with each other, yet the three-way intersection is empty
    (core ``!=2.2.*`` excludes exactly the plugins' common band ``[2.2, 2.3)``).
    No dedicated per-plugin core conflict applies, so the environment-level
    fallback must fire, list core as a participant, and -- because every
    plugin reaching this branch is by construction individually
    core-compatible -- advise that per-process isolation WORKS (here: one
    worker on 2.3.x, one on 2.1.x, each satisfying core). Saying "isolation
    cannot help" in this branch would be false in its only reachable shape.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("a/x", _FakeDist("std-a", ["numpy>=2.2,<2.4"])),
            _FakeEP("b/y", _FakeDist("std-b", ["numpy>=2.1,<2.3"])),
        ],
        core_spec="!=2.2.*",
    )
    report = doctor.diagnose()
    assert report.has_conflict is True
    assert not any("conflict with standard-asr core" in c for c in report.conflicts)
    conflict = next(c for c in report.conflicts if "no common satisfying version" in c)
    assert "standard-asr (!=2.2.*)" in conflict
    assert "separate processes" in conflict
    assert "isolation cannot help" not in conflict


def test_core_metadata_unreadable_is_a_non_clean_analysis_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreadable core metadata means the plugin-vs-core relation never ran.

    That relation is the half that catches an environment NO process layout can
    fix, so its absence is a non-clean ``analysis_unavailable`` state, not a
    footer note under a green "No dependency conflicts detected" headline --
    the core-floor gap this analysis exists to close would silently reopen.
    The rendered listing must also NOT show a core line it does not have.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["numpy>=1.26"]))],
        core_unreadable=True,
    )
    report = doctor.diagnose()
    assert report.core is None
    assert any("plugin-vs-core numpy analysis could not run" in n for n in report.notes)
    assert any("cannot be proven conflict-free" in n for n in report.notes)
    # No classified conflict -- but NOT clean, and the verdict/exit path agree.
    assert report.has_conflict is False
    assert report.analysis_unavailable is True
    assert report.is_clean is False

    rendered = doctor.format_report(report)
    assert "core:" not in rendered
    assert "No dependency conflicts detected." not in rendered
    # The headline is the generalized one: it covers every analysis gap, and the
    # specific gap is named by the note.
    assert "Conflict analysis unavailable or incomplete" in rendered


def test_plugin_vs_plugin_same_major_conflict_does_not_blame_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the plugin-only intersection is ALREADY empty, core is not a
    load-bearing participant: the message must list only the plugins and keep
    the out-of-process remediation -- the core-specific 'isolation cannot help'
    text here would be false and would misdirect the user away from the one
    fix that works.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("a/x", _FakeDist("std-a", ["numpy==2.0.*"])),
            _FakeEP("b/y", _FakeDist("std-b", ["numpy>=2.3"])),
        ],
    )
    report = doctor.diagnose()
    conflict = next(c for c in report.conflicts if "no common satisfying version" in c)
    assert "out-of-process" in conflict
    assert "isolation cannot help" not in conflict
    assert "standard-asr" not in conflict


def test_format_report_renders_core_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core requirement joins the intersection, so the listing must show it
    -- a conflict naming standard-asr must trace back to a visible line.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    _patch_eps(
        monkeypatch,
        [_FakeEP("a/x", _FakeDist("std-a", ["numpy>=1.26"]))],
    )
    rendered = doctor.format_report(doctor.diagnose())
    assert "core: [standard-asr] numpy >=1.26" in rendered


def test_is_clean_is_the_single_verdict() -> None:
    # The one verdict consumed by the CLI exit code and the report headline:
    # clean requires BOTH no detected conflict AND that analysis ran (an
    # unprovable environment must not read as clean).
    assert doctor.DoctorReport(python_version="3.12").is_clean is True
    assert doctor.DoctorReport(python_version="3.12", conflicts=["x vs y"]).is_clean is False
    assert doctor.DoctorReport(python_version="3.12", analysis_unavailable=True).is_clean is False
