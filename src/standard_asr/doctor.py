# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Read-only dependency conflict diagnostic (``standard-asr doctor``).

Enumerates installed Standard ASR plugins, reads each plugin distribution's
declared ``numpy`` requirement, and reports conflicts that cannot coexist in a
single process -- most importantly the numpy 1.x-vs-2.x split (spec DEP.5). It
never resolves or installs anything; it only diagnoses and suggests remediation
(out-of-process isolation when a conflict is real).

Scope (v1, spec DEP.5): doctor diagnoses ``numpy`` ONLY. numpy is the single
shared native dependency the standard itself has (DEP.1), and its 1.x-vs-2.x
break is a clean C-ABI split whose conflict is fully encoded in version
specifiers -- so a version-range intersection decides it. Other shared native
libraries (torch CUDA build variants; onnxruntime vs onnxruntime-gpu package
identity) have fundamentally different conflict models that version
intersection cannot decide, so they are explicitly known-uncovered in v1;
their hard conflicts fall under the general DEP.4 process-isolation guidance.
See the per-library seam in :func:`_numpy_spec_for` for the rationale.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from .discovery import ENTRYPOINT_GROUP

if TYPE_CHECKING:
    from packaging.specifiers import SpecifierSet

# Display-only fallback for the packaging-absent path. ``packaging`` is the
# authoritative parser (it evaluates environment markers and the legacy
# parenthesized form ``numpy (>=1.26)``); this regex is used solely to render a
# best-effort specifier string when ``packaging`` cannot be imported, in which
# case doctor degrades to listing-without-classifying, never reports a conflict
# it could not verify, and marks the report ``analysis_unavailable`` (a
# non-clean verdict) when plugins exist. It captures the text before any marker
# (``;``); the extras group (``numpy[foo]``) is discarded.
_NUMPY_REQ = re.compile(r"^\s*numpy\b(?:\[[^\]]*\])?(?P<spec>[^;]*)", re.IGNORECASE)

# Representative versions spanning the numpy 1.x / 2.x boundary, seeded into the
# emptiness probe's candidate set (see ``_emptiness_candidates``) alongside the
# boundary-derived candidates. They are NOT a classification grid: classifying a
# single spec as numpy1/numpy2 intersects it with the full major ranges below,
# because a fixed grid misreads an exact off-grid pin such as ``==2.2.0`` as
# admitting neither major.
_NUMPY1_PROBES = ("1.21.0", "1.24.0", "1.26.4", "1.26.99", "1.99.99")
_NUMPY2_PROBES = ("2.0.0", "2.1.0", "2.3.0", "2.99.99")

# The full numpy major-version ranges. A spec admits a 1.x (resp. 2.x) release
# iff its intersection with the corresponding range is non-empty, decided by the
# boundary-derived emptiness oracle (``_intersection_is_empty``).
_NUMPY1_RANGE = ">=1.0,<2.0"
_NUMPY2_RANGE = ">=2.0,<3.0"

# A sentinel "arbitrarily large" release, used to witness that an open upper
# bound (``>=``/``>`` with no ceiling) is satisfiable. Any real numpy pin is far
# below this, so it is a safe stand-in for "+infinity" when probing emptiness.
_OPEN_UPPER_SENTINEL = "100000.0.0"

# A large component value used to construct a version that sits *just below* a
# boundary at a given release position (e.g. just under ``2.1`` -> ``2.0.<big>``).
_JUST_BELOW_FILL = 99999


def _empty_plugins() -> list["PluginNumpy"]:
    """Return an empty plugin list (typed factory for dataclass default).

    Returns:
        An empty list.
    """
    return []


def _empty_strs() -> list[str]:
    """Return an empty string list (typed factory for dataclass default).

    Returns:
        An empty list.
    """
    return []


@dataclass
class PluginNumpy:
    """A plugin and its declared numpy requirement.

    Args:
        entrypoint: The plugin entry-point name.
        distribution: The distribution package name.
        numpy_spec: The raw numpy version specifier (e.g. ``"<2"``), or ``None``.
    """

    entrypoint: str
    distribution: str
    numpy_spec: str | None


@dataclass
class DoctorReport:
    """The result of a dependency diagnosis.

    Args:
        python_version: The running interpreter version (``X.Y``).
        plugins: The discovered plugins and their numpy requirements.
        conflicts: Human-readable conflict descriptions.
        notes: Supplementary remediation hints (non-verdict footer lines).
        analysis_unavailable: Whether conflict analysis could not run at all --
            plugins are installed but the optional ``packaging`` distribution is
            missing -- so the environment cannot be proven conflict-free. This
            is a non-clean state distinct from "no conflicts detected".
    """

    python_version: str
    plugins: list[PluginNumpy] = field(default_factory=_empty_plugins)
    conflicts: list[str] = field(default_factory=_empty_strs)
    notes: list[str] = field(default_factory=_empty_strs)
    analysis_unavailable: bool = False

    @property
    def has_conflict(self) -> bool:
        """Whether any conflict was detected.

        Returns:
            ``True`` if there is at least one conflict.
        """
        return bool(self.conflicts)

    @property
    def is_clean(self) -> bool:
        """Whether the environment is proven conflict-free (the verdict).

        The single source of the doctor verdict, consumed by both the CLI exit
        code and the :func:`format_report` headline: clean requires BOTH no
        detected conflict AND that conflict analysis actually ran
        (``analysis_unavailable`` is a non-clean state -- an unprovable
        environment must not read as clean). A future non-clean state is
        wired in here once, not re-derived in every consumer.

        Returns:
            ``True`` when no conflict was detected and analysis ran.
        """
        return not self.has_conflict and not self.analysis_unavailable


def packaging_available() -> bool:
    """Return whether the optional ``packaging`` library is importable.

    ``packaging`` is NOT a core dependency (core = pydantic + numpy only, spec
    DEP.1); doctor uses it for precise specifier analysis when present and
    degrades gracefully otherwise.

    Returns:
        ``True`` if ``packaging`` can be imported.
    """
    try:
        import packaging.specifiers  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError:
        return False
    return True


def _classify_numpy(numpy_spec: str | None) -> tuple[bool, bool]:
    """Classify a numpy specifier as numpy1-only and/or numpy2-required.

    Whether the spec admits a 1.x (resp. 2.x) release is decided by intersecting
    it with the full major range (``[1.0, 2.0)`` / ``[2.0, 3.0)``) and testing
    emptiness against the intersection's own boundary-derived candidates
    (:func:`_intersection_is_empty`) -- never against a fixed probe grid, which
    misreads an exact off-grid pin such as ``==2.2.0`` as admitting neither
    major. When ``packaging`` is absent the classifier
    conservatively returns ``(False, False)`` (no hard split) so it never
    reports a conflict it cannot verify.

    Args:
        numpy_spec: The raw numpy specifier (e.g. ``"<2"``, ``"~=1.26.0"``,
            ``"(any)"``), or ``None`` when numpy is not required.

    Returns:
        A ``(numpy1_only, numpy2_required)`` pair. ``numpy1_only`` is ``True``
        when the spec admits a 1.x but no 2.x; ``numpy2_required`` is ``True``
        when it admits a 2.x but no 1.x. An unconstrained / both-admitting /
        unparseable spec, or a missing ``packaging``, yields ``(False, False)``.
    """
    spec_set = _specset(numpy_spec)
    if spec_set is None:
        return (False, False)
    from packaging.specifiers import SpecifierSet

    admits1 = not _intersection_is_empty([spec_set, SpecifierSet(_NUMPY1_RANGE)])
    admits2 = not _intersection_is_empty([spec_set, SpecifierSet(_NUMPY2_RANGE)])
    return (admits1 and not admits2, admits2 and not admits1)


def _specset(numpy_spec: str | None) -> SpecifierSet | None:
    """Parse a numpy specifier string into a ``SpecifierSet``.

    Args:
        numpy_spec: The effective numpy specifier (e.g. ``"<2"``, ``"~=1.26.0"``,
            ``"(any)"``), or ``None`` when numpy is not required.

    Returns:
        The parsed :class:`packaging.specifiers.SpecifierSet`, or ``None`` when
        the spec is missing/unparseable or ``packaging`` is unavailable. The
        ``"(any)"`` sentinel parses to the empty (admit-all) set.
    """
    if not numpy_spec:
        return None
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
    except ImportError:
        return None
    raw = "" if numpy_spec == "(any)" else numpy_spec
    try:
        return SpecifierSet(raw)
    except InvalidSpecifier:
        return None


def _emptiness_candidates(combined: SpecifierSet) -> set[str]:
    """Derive probe versions that witness whether *combined* is satisfiable.

    A :class:`~packaging.specifiers.SpecifierSet` has no exact emptiness oracle,
    so emptiness is decided by probing. A *bounded* version grid is unsound here:
    a perfectly satisfiable high pin (``>=2.40``, ``==2.45.*``, ``>=3.0``) lands
    above any fixed grid and would be misread as empty. Instead the candidates
    are derived from the combined specifier's **own boundaries** -- each
    specifier's edge version (verbatim, so an ``epoch``/pre-release pin such as
    ``==1!2.0`` / ``==2.1.0rc1`` keeps the segment that makes it satisfiable),
    plus a version just *above* and just *below* every release position of that
    edge, plus two **one-level-deeper** witnesses (``release + (1,)`` and a
    ``release`` with one extra high-filled component) so a strict-boundary or
    sub-release-width interval such as ``>2.0,<2.1`` (witness ``2.0.1``) or
    ``>2.0,<2.0.1`` (witness ``2.0.0.99999``) is not misread as empty -- together
    with an open "arbitrarily large" sentinel (so an open ``>=``/``>`` lower
    bound is recognised as satisfiable) and ``0`` (so an open ``<``/``<=`` upper
    bound is too).

    Soundness is **approximate, not total**: PEP 440 release tuples have
    unbounded length, so no finite candidate set decides every conceivable
    interval (e.g. ``>2.0,<2.0.0.0.1`` needs a five-component witness). The
    candidate set is sound for release- and one-sub-release-granularity
    intervals -- which covers numpy's real pins and the canonical 1.x/2.x split
    -- and adding candidates is one-directionally safe: a new witness can only
    turn a false ``empty`` verdict into a correct ``non-empty`` one, never the
    reverse. A satisfiable interval narrower than this resolution may still be
    misreported as empty; :func:`diagnose` therefore frames a single-plugin
    "internally unsatisfiable" verdict as report-a-bug-able rather than
    absolute.

    Args:
        combined: The merged specifier whose satisfiability is being probed.

    Returns:
        A set of candidate version strings. Most are final releases; a verbatim
        edge string may carry an epoch or pre-release segment, which by PEP 440
        membership only matches a specifier that itself admits it -- so it never
        manufactures a false non-empty verdict.
    """
    from packaging.version import Version

    candidates: set[str] = {"0", _OPEN_UPPER_SENTINEL}
    candidates.update(_NUMPY1_PROBES)
    candidates.update(_NUMPY2_PROBES)
    for spec in combined:
        # ``==2.45.*`` / ``!=2.0.*`` carry a non-PEP440 ``2.45.*`` version; the
        # prefix (``2.45``) is the band edge and is itself a valid Version.
        edge = spec.version[:-2] if spec.version.endswith(".*") else spec.version
        try:
            parsed = Version(edge)
        except Exception:  # noqa: BLE001 - a non-version edge (e.g. ``===`` URL) is just skipped
            continue
        # Keep the edge verbatim (``str(Version(edge))`` canonicalises but
        # preserves epoch/pre-release), so an epoch/pre-release pin witnesses
        # itself. A bare ``release`` candidate would silently drop those segments.
        candidates.add(str(parsed))
        release = parsed.release
        candidates.add(".".join(str(r) for r in release) or "0")
        # One level deeper than the edge: ``release + (1,)`` sits inside the open
        # interval immediately above ``release`` (e.g. ``2.0`` -> ``2.0.1``),
        # witnessing strict-lower-bound / next-edge pairs like ``>2.0,<2.1``.
        deeper_above = (*release, 1)
        candidates.add(".".join(str(r) for r in deeper_above))
        for i in range(len(release)):
            # Just above this release position: bump component i, zero the tail.
            above = (*release[:i], release[i] + 1, *((0,) * (len(release) - i - 1)))
            candidates.add(".".join(str(r) for r in above))
            # Just below: decrement component i (when > 0), fill the tail high,
            # and append one extra high-filled component so a sub-release-width
            # upper bound (``<2.0.1`` -> witness ``2.0.0.99999``) is covered.
            if release[i] > 0:
                below = (
                    *release[:i],
                    release[i] - 1,
                    *((_JUST_BELOW_FILL,) * (len(release) - i)),
                )
                candidates.add(".".join(str(r) for r in below))
    return candidates


def _intersection_is_empty(specs: list[SpecifierSet]) -> bool:
    """Report whether the intersection of numpy ``SpecifierSet``s admits nothing.

    Computes the real combined specifier (``&``) across plugins and tests it
    against boundary-derived probe versions (:func:`_emptiness_candidates`). An
    empty intersection means no single numpy release satisfies every plugin -- a
    hard conflict. This catches disjoint same-major ranges (``==2.0.*`` vs
    ``>=2.3``) that a 1.x/2.x major-boundary classification alone would miss,
    *and* high pins (``>=2.40``, ``>=3.0``) that a bounded grid would have
    misreported as empty.

    Args:
        specs: The :class:`packaging.specifiers.SpecifierSet`s to intersect
            (per-plugin specs, or a spec paired with a major range for
            classification). Must be non-empty and contain only real specifier
            sets. A single internally-unsatisfiable set (e.g. ``<2`` and
            ``>=2.1`` declared by one plugin) is a valid -- and detected --
            input.

    Returns:
        ``True`` if no candidate version satisfies the combined specifier.
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    combined = SpecifierSet()
    for spec in specs:
        combined &= spec
    return not any(Version(v) in combined for v in _emptiness_candidates(combined))


def _numpy_spec_for(requires: list[str] | None) -> str | None:
    """Extract the *effective* numpy specifier for the running interpreter.

    Per-library seam: numpy is the only shared native dependency Standard ASR can
    diagnose precisely (spec DEP.5). Its 1.x-vs-2.x split is a clean
    C-ABI break with a clean version-range signature, so a Requires-Dist version
    specifier fully determines compatibility. torch (CUDA build variants),
    onnxruntime vs onnxruntime-gpu (package-identity conflicts) and similar do
    NOT encode their conflict in version specifiers, so this seam intentionally
    matches ``numpy`` only -- generalizing the version-intersection to them would
    be confidently wrong. See DEP.4 for the general isolation guidance.

    Each ``Requires-Dist`` line is parsed with :class:`packaging.requirements.
    Requirement`, which evaluates PEP 508 environment markers and accepts the
    legacy parenthesized form (``numpy (>=1.26)``). Only numpy lines whose marker
    holds on the running interpreter (or is absent) contribute, so the canonical
    interpreter-conditional dual-line declaration (spec DEP.1) resolves to the
    one line that actually applies here. Multiple applicable lines are
    intersected. When ``packaging`` is absent doctor degrades to a display-only
    regex extraction (no marker evaluation, no conflict classification).

    Args:
        requires: The distribution's ``Requires-Dist`` entries.

    Returns:
        The effective numpy specifier string (``"(any)"`` when numpy is required
        without a version bound), or ``None`` if numpy is not required (or no
        applicable line survives marker evaluation).
    """
    try:
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.specifiers import SpecifierSet
    except ImportError:
        return _numpy_spec_for_display(requires)

    combined = SpecifierSet()
    found = False
    # Evaluate markers against an environment derived from sys.version_info rather
    # than packaging's default (which reads the real interpreter via
    # platform.python_version()). This keeps marker resolution consistent with the
    # python_version doctor reports and makes it overridable -- e.g. a test that
    # simulates another interpreter by patching sys.version_info, or any caller
    # that wants the canonical interpreter-conditional dual line (DEP.1) resolved
    # for a specific Python.
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    marker_env = {
        "python_version": py,
        "python_full_version": f"{py}.{sys.version_info.micro}",
    }
    for raw in requires or []:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if req.name.lower() != "numpy":
            continue
        if req.marker is not None and not req.marker.evaluate(marker_env):
            continue
        found = True
        combined &= req.specifier
    if not found:
        return None
    return str(combined) or "(any)"


def _render_distributions(plugins: list[PluginNumpy]) -> str:
    """Render a conflict participant list, one entry per distribution.

    A single distribution that ships several presets (``plugin_entrypoints.md``
    encourages this) contributes one :class:`PluginNumpy` per entry point, all
    carrying the SAME ``Requires-Dist`` numpy spec. The numpy constraint belongs
    to the *distribution*, so the conflict text lists each ``(distribution,
    numpy_spec)`` once -- order-preserving dedup -- instead of repeating
    ``std-foo (<2), std-foo (<2), std-foo (<2)`` and inflating the apparent
    conflict size.

    Args:
        plugins: The plugins on one side of a conflict.

    Returns:
        A comma-joined ``"<distribution> (<spec>)"`` listing with duplicates
        (same distribution AND same spec) collapsed, original order kept.
    """
    unique = dict.fromkeys((p.distribution, p.numpy_spec) for p in plugins)
    return ", ".join(f"{dist} ({spec})" for dist, spec in unique)


def _unique_distributions(plugins: list[PluginNumpy]) -> list[str]:
    """Return the distinct distribution names among *plugins*, order-preserving.

    Used both to size a conflict (one distribution shipping many presets is a
    single participant, not many) and to render the distribution-only Python
    3.13 wheel note without repetition.

    Args:
        plugins: The plugins to reduce to their distributions.

    Returns:
        The distribution names with duplicates removed, in first-seen order.
    """
    return list(dict.fromkeys(p.distribution for p in plugins))


def _numpy_spec_for_display(requires: list[str] | None) -> str | None:
    """Best-effort numpy specifier extraction for the packaging-absent path.

    This does NOT evaluate environment markers and is used only to populate the
    human-readable plugin listing when ``packaging`` is unavailable; in that mode
    doctor never classifies conflicts (see :func:`diagnose`).

    Args:
        requires: The distribution's ``Requires-Dist`` entries.

    Returns:
        The first numpy specifier string, or ``None`` if numpy is not required.
    """
    for req in requires or []:
        match = _NUMPY_REQ.match(req)
        if match:
            return match.group("spec").strip() or "(any)"
    return None


def diagnose(*, group: str = ENTRYPOINT_GROUP) -> DoctorReport:
    """Diagnose numpy compatibility across installed plugins.

    Args:
        group: The entry-point group to inspect.

    Returns:
        A :class:`DoctorReport` describing plugins and any conflicts.
    """
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    report = DoctorReport(python_version=py)

    for ep in entry_points(group=group):
        dist = ep.dist
        dist_name = dist.name if dist is not None else "<unknown>"
        spec = _numpy_spec_for(dist.requires if dist is not None else None)
        report.plugins.append(PluginNumpy(ep.name, dist_name, spec))

    if report.plugins and not packaging_available():
        # With plugins present but no analyzer, doctor cannot prove the
        # environment conflict-free; the report must say so loudly rather than
        # let an empty conflict list read as a clean verdict. With no
        # plugins there is nothing to analyze and absence is a non-issue.
        report.analysis_unavailable = True
        report.notes.append(
            "Install the optional 'packaging' library for precise numpy "
            "conflict analysis; without it, version-range conflicts are not "
            "classified."
        )

    numpy1_only: list[PluginNumpy] = []
    numpy2_required: list[PluginNumpy] = []
    constrained: list[PluginNumpy] = []
    spec_sets: list[SpecifierSet] = []
    for p in report.plugins:
        spec_set = _specset(p.numpy_spec)
        if spec_set is not None:
            constrained.append(p)
            spec_sets.append(spec_set)
        only1, req2 = _classify_numpy(p.numpy_spec)
        if only1:
            numpy1_only.append(p)
        if req2:
            numpy2_required.append(p)

    if numpy1_only and numpy2_required:
        # Clean 1.x-vs-2.x split: the most actionable framing for the canonical
        # C-ABI break, named explicitly so the user knows which side to isolate.
        # Each side lists per distribution, not per preset.
        report.conflicts.append(
            "numpy 1.x vs 2.x conflict: "
            + _render_distributions(numpy1_only)
            + " require numpy<2 while "
            + _render_distributions(numpy2_required)
            + " require numpy>=2. They cannot share one process; run the "
            "conflicting plugin out-of-process (subprocess/server isolation)."
        )
    elif spec_sets and _intersection_is_empty(spec_sets):
        # Real-intersection conflict that the 1.x/2.x classification alone misses
        # -- e.g. disjoint same-major ranges (``==2.0.*`` vs ``>=2.3``) that share
        # no satisfying numpy release. A SINGLE distribution whose own numpy
        # declaration is internally unsatisfiable (e.g. ``<2`` and ``>=2.1``) is
        # checked too: an impossible self-pin is a real conflict the user must
        # see, not a silently-passed declaration. The single-vs-cross framing is
        # decided by the count of distinct *distributions*, so a
        # lone distribution shipping several presets still reads as one offender.
        listing = _render_distributions(constrained)
        if len(_unique_distributions(constrained)) == 1:
            report.conflicts.append(
                f"numpy version conflict: {listing} declares an internally "
                "unsatisfiable numpy range (no version satisfies it). Fix the "
                "plugin's numpy requirement. (Emptiness is decided by a "
                "boundary-derived probe sound to one-sub-release granularity; if "
                "you believe this range is satisfiable, please report a bug.)"
            )
        else:
            report.conflicts.append(
                f"numpy version conflict: {listing} declare numpy ranges with no "
                "common satisfying version. They cannot share one process; run the "
                "conflicting plugin out-of-process (subprocess/server isolation)."
            )

    if sys.version_info >= (3, 13) and numpy1_only:
        report.conflicts.append(
            "On Python 3.13+ there is no numpy<2 wheel: "
            + ", ".join(_unique_distributions(numpy1_only))
            + " cannot be installed here. Use Python <3.13 or isolate the plugin."
        )

    return report


def format_report(report: DoctorReport) -> str:
    """Render a doctor report as human-readable text.

    Args:
        report: The report to render.

    Returns:
        The formatted report.
    """
    lines = [f"Standard ASR doctor (Python {report.python_version})", ""]
    if not report.plugins:
        lines.append("No Standard ASR plugins are installed.")
    else:
        lines.append("Installed plugins:")
        for p in report.plugins:
            lines.append(f"  - {p.entrypoint} [{p.distribution}] numpy {p.numpy_spec}")
    lines.append("")
    if report.is_clean:
        # The clean claim is gated on the report's single verdict property, so
        # a new non-clean state added to is_clean can never read as clean here
        # (it still needs its own rendering branch below).
        lines.append("No dependency conflicts detected.")
    elif report.has_conflict:
        lines.append("Conflicts:")
        lines.extend(f"  ! {c}" for c in report.conflicts)
    else:
        # Non-clean without a classified conflict: analysis could not run.
        # Claiming "no conflicts" here would be a silent wrong result; the
        # headline must carry the non-clean state.
        lines.append(
            "Conflict analysis unavailable: the 'packaging' distribution is "
            "not installed (pip install packaging). Cannot prove the "
            "environment conflict-free."
        )
    if report.notes:
        lines.append("")
        lines.extend(f"  note: {n}" for n in report.notes)
    return "\n".join(lines)


__all__ = ["DoctorReport", "PluginNumpy", "diagnose", "format_report"]
