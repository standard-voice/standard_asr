# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Read-only dependency conflict diagnostic (``standard-asr doctor``).

Enumerates installed Standard ASR plugins, reads each plugin distribution's
declared ``numpy`` requirement, and reports conflicts that cannot coexist in a
single process -- most importantly the numpy 1.x-vs-2.x split. It
never resolves or installs anything; it only diagnoses and suggests remediation
(out-of-process isolation when a conflict is real).

Scope (v1): doctor diagnoses ``numpy`` ONLY. numpy is the single shared native
dependency the standard itself has, and its 1.x-vs-2.x break is a clean C-ABI
split whose conflict is fully encoded in version specifiers -- so a version-range
intersection decides it. Other shared native libraries (torch CUDA build
variants; onnxruntime vs onnxruntime-gpu package identity) have fundamentally
different conflict models that version intersection cannot decide, so they are
explicitly known-uncovered in v1; their hard conflicts fall under the general
out-of-process isolation guidance.
See the per-library seam in :func:`_numpy_spec_for` for the rationale.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, entry_points, requires
from typing import TYPE_CHECKING

from standard_asr.plugins.discovery import ENTRYPOINT_GROUP

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
        core: The standard-asr core's own effective numpy requirement, as a
            conflict participant (``None`` when no plugins are installed --
            nothing to conflict with -- or when the core distribution's
            metadata is unreadable). Core is ALWAYS in the process, so its
            interpreter-conditional numpy floor MUST join the intersection: a
            plugin whose numpy range excludes the core floor can never run
            with standard-asr on this interpreter, and omitting core from the
            analysis let doctor report such an environment clean.
        conflicts: Human-readable conflict descriptions.
        notes: Supplementary remediation hints (non-verdict footer lines).
        analysis_unavailable: Whether conflict analysis could not run at all --
            plugins are installed but the optional ``packaging`` distribution is
            missing -- so the environment cannot be proven conflict-free. This
            is a non-clean state distinct from "no conflicts detected".
    """

    python_version: str
    plugins: list[PluginNumpy] = field(default_factory=_empty_plugins)
    core: PluginNumpy | None = None
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

    ``packaging`` is NOT a core dependency (core = pydantic + numpy only);
    doctor uses it for precise specifier analysis when present and degrades
    gracefully otherwise.

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
    diagnose precisely. Its 1.x-vs-2.x split is a clean
    C-ABI break with a clean version-range signature, so a Requires-Dist version
    specifier fully determines compatibility. torch (CUDA build variants),
    onnxruntime vs onnxruntime-gpu (package-identity conflicts) and similar do
    NOT encode their conflict in version specifiers, so this seam intentionally
    matches ``numpy`` only -- generalizing the version-intersection to them would
    be confidently wrong. Those fall under the general out-of-process isolation
    guidance instead.

    Each ``Requires-Dist`` line is parsed with :class:`packaging.requirements.
    Requirement`, which evaluates PEP 508 environment markers and accepts the
    legacy parenthesized form (``numpy (>=1.26)``). Only numpy lines whose marker
    holds on the running interpreter (or is absent) contribute, so the canonical
    interpreter-conditional dual-line declaration resolves to the
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
    # that wants the canonical interpreter-conditional dual line resolved
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


#: The core distribution whose own numpy floor joins every conflict analysis.
_CORE_DISTRIBUTION = "standard-asr"


def _core_numpy_spec() -> str | None:
    """Return the core's effective numpy specifier for the running interpreter.

    The core declares an interpreter-conditional numpy floor (e.g.
    ``numpy>=1.26`` below Python 3.13, ``numpy>=2.1`` at 3.13+), and the core is
    in EVERY process that runs standard_asr -- including any isolated worker a
    plugin would be moved into. Its requirement therefore belongs in the same
    intersection as the plugins': without it, a plugin declaring e.g.
    ``numpy<1.26`` read as conflict-free while resolution against core is
    impossible on this interpreter.

    Returns:
        The effective numpy specifier string (marker-evaluated for the running
        interpreter, same rules as :func:`_numpy_spec_for`), or ``None`` when
        the core distribution's metadata is unreadable (an anomalous install;
        the caller reports the analysis gap rather than silently narrowing it).
    """
    try:
        core_requires = requires(_CORE_DISTRIBUTION)
    except PackageNotFoundError:
        return None
    return _numpy_spec_for(core_requires)


def diagnose(*, group: str = ENTRYPOINT_GROUP) -> DoctorReport:
    """Diagnose numpy compatibility across installed plugins AND the core.

    Two distinct conflict relations are analyzed, in order, because their
    remediations are different and conflating them misdirects the user:

    1. **Plugin vs core** (see :func:`_core_numpy_spec`): core is in every
       process -- including any isolated worker -- so a plugin whose numpy
       range has no version in common with the core floor can never run with
       standard-asr on this interpreter AT ALL. Each such distribution gets
       its own dedicated conflict, and is then EXCLUDED from the
       environment-level analysis below: it cannot run in any process layout,
       so letting it participate would pollute the isolation advice (e.g. a
       plugin-vs-plugin message telling the user to isolate a plugin that is
       equally dead in the isolated worker).
    2. **Environment level** among the remaining (core-compatible) plugins,
       plus core: the 1.x-vs-2.x split and the general empty-intersection
       check, whose remediation is out-of-process isolation.

    Args:
        group: The entry-point group to inspect.

    Returns:
        A :class:`DoctorReport` describing plugins, the core requirement, and
        any conflicts.
    """
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    report = DoctorReport(python_version=py)

    for ep in entry_points(group=group):
        dist = ep.dist
        dist_name = dist.name if dist is not None else "<unknown>"
        spec = _numpy_spec_for(dist.requires if dist is not None else None)
        report.plugins.append(PluginNumpy(ep.name, dist_name, spec))

    if report.plugins:
        core_spec = _core_numpy_spec()
        if core_spec is not None:
            report.core = PluginNumpy("(core)", _CORE_DISTRIBUTION, core_spec)
        else:
            # Core metadata unreadable: the intersection is missing a
            # participant that is always in the process. Say so rather than
            # letting the narrower analysis read as complete.
            report.notes.append(
                "standard-asr's own distribution metadata is unreadable, so the "
                "core numpy requirement could not join the conflict analysis."
            )

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

    # Relation 1 -- plugin vs core, per distribution. A plugin internally
    # unsatisfiable ON ITS OWN is skipped here: that is a plugin declaration
    # bug, and the environment-level branch below attributes it as one
    # (blaming core for a self-contradictory pin would misdirect the fix).
    core_spec_set = _specset(report.core.numpy_spec) if report.core is not None else None
    core_incompatible: list[PluginNumpy] = []
    if report.core is not None and core_spec_set is not None:
        seen_core_conflict_dists: set[str] = set()
        for p in report.plugins:
            if p.distribution in seen_core_conflict_dists:
                continue
            plugin_spec_set = _specset(p.numpy_spec)
            if plugin_spec_set is None or _intersection_is_empty([plugin_spec_set]):
                continue
            if _intersection_is_empty([plugin_spec_set, core_spec_set]):
                seen_core_conflict_dists.add(p.distribution)
                core_incompatible.append(p)
        for p in core_incompatible:
            report.conflicts.append(
                f"numpy conflict with standard-asr core: {p.distribution} "
                f"({p.numpy_spec}) shares no numpy version with the core "
                f"requirement (numpy {report.core.numpy_spec}) on this "
                "interpreter. Core is in every process -- isolation cannot "
                "help: the plugin must change its numpy requirement, or run "
                "under a Python version where the core floor differs."
            )

    # Relation 2 -- environment level. Core joins as a participant (it is in
    # every process), but the core-incompatible distributions reported above
    # are excluded: they cannot run in any layout, and keeping them here would
    # produce isolation advice that silently fails for exactly those plugins.
    excluded_dists = {p.distribution for p in core_incompatible}
    participants: list[PluginNumpy] = [
        p for p in report.plugins if p.distribution not in excluded_dists
    ]
    if report.core is not None:
        participants.append(report.core)

    numpy1_only: list[PluginNumpy] = []
    numpy2_required: list[PluginNumpy] = []
    constrained: list[PluginNumpy] = []
    spec_sets: list[SpecifierSet] = []
    for p in participants:
        spec_set = _specset(p.numpy_spec)
        if spec_set is not None:
            constrained.append(p)
            spec_sets.append(spec_set)
        only1, req2 = _classify_numpy(p.numpy_spec)
        if only1:
            numpy1_only.append(p)
        if req2:
            numpy2_required.append(p)

    def _remedy(sides: list[PluginNumpy]) -> str:
        """Return the remediation sentence for a conflict among ``sides``.

        Args:
            sides: Every participant in the conflict.

        Returns:
            Process-isolation advice. When core is among ``sides``, the
            conflict is necessarily JOINT-only: every plugin for which
            isolation genuinely cannot help (individually core-incompatible)
            already got its own dedicated conflict in relation 1 and was
            excluded from this analysis, so each plugin here IS individually
            core-compatible and per-process isolation works by construction --
            the sentence says so explicitly, naming core's participation.
        """
        if report.core is not None and report.core in sides:
            return (
                " The core requirement participates in the joint emptiness, "
                "but every plugin here is individually core-compatible (a "
                "plugin that is not gets its own dedicated conflict), so "
                "out-of-process isolation works: run the conflicting plugins "
                "in separate processes, each resolving its own numpy together "
                "with the core requirement."
            )
        return (
            " They cannot share one process; run the conflicting plugin "
            "out-of-process (subprocess/server isolation)."
        )

    if numpy1_only and numpy2_required:
        # Clean 1.x-vs-2.x split: the most actionable framing for the canonical
        # C-ABI break, named explicitly so the user knows which side to isolate.
        # Each side lists per distribution, not per preset.
        report.conflicts.append(
            "numpy 1.x vs 2.x conflict: "
            + _render_distributions(numpy1_only)
            + " require numpy<2 while "
            + _render_distributions(numpy2_required)
            + " require numpy>=2."
            + _remedy([*numpy1_only, *numpy2_required])
        )
    elif spec_sets and _intersection_is_empty(spec_sets):
        # Real-intersection conflict that the 1.x/2.x classification alone misses
        # -- e.g. disjoint same-major ranges (``==2.0.*`` vs ``>=2.3``) that share
        # no satisfying numpy release. A SINGLE distribution whose own numpy
        # declaration is internally unsatisfiable (e.g. ``<2`` and ``>=2.1``) is
        # checked too: an impossible self-pin is a real conflict the user must
        # see, not a silently-passed declaration. The single-vs-cross framing is
        # decided by the count of distinct *plugin* distributions and by whether
        # the plugin side is empty ON ITS OWN: with core in the intersection, a
        # lone self-contradictory plugin must still read as the one offender
        # (naming core would misattribute a plugin bug), so the plugin-only
        # emptiness is tested first.
        plugin_pairs = [(p, s) for p, s in zip(constrained, spec_sets) if p is not report.core]
        plugin_constrained = [p for p, _ in plugin_pairs]
        plugin_specs = [s for _, s in plugin_pairs]
        if (
            len(_unique_distributions(plugin_constrained)) == 1
            and plugin_specs
            and _intersection_is_empty(plugin_specs)
        ):
            listing = _render_distributions(plugin_constrained)
            report.conflicts.append(
                f"numpy version conflict: {listing} declares an internally "
                "unsatisfiable numpy range (no version satisfies it). Fix the "
                "plugin's numpy requirement. (Emptiness is decided by a "
                "boundary-derived probe sound to one-sub-release granularity; if "
                "you believe this range is satisfiable, please report a bug.)"
            )
        else:
            # Attribute the conflict to its LOAD-BEARING participants. If the
            # plugin-only intersection is already empty, this is a
            # plugin-vs-plugin conflict: core must be neither listed nor
            # blamed, because naming it would invert the remediation --
            # out-of-process isolation is exactly the fix for plugin-vs-plugin,
            # and the core-specific "isolation cannot help" text would tell the
            # user their one escape hatch is useless. Core joins the listing
            # (and flips the remedy) only when the plugins alone are
            # satisfiable and adding core is what empties the intersection.
            # (A single self-contradictory plugin never reaches here: the
            # internal-unsatisfiable branch above catches it first, so
            # plugin-only emptiness here implies >= 2 plugin distributions.)
            plugins_alone_empty = bool(plugin_specs) and _intersection_is_empty(plugin_specs)
            sides = plugin_constrained if plugins_alone_empty else constrained
            listing = _render_distributions(sides)
            report.conflicts.append(
                f"numpy version conflict: {listing} declare numpy ranges with no "
                "common satisfying version." + _remedy(sides)
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
        if report.core is not None:
            # Core participates in the intersection (it is in every process);
            # show its effective requirement so a core-involving conflict below
            # is traceable to a visible line, not an invisible participant.
            lines.append(f"  core: [{report.core.distribution}] numpy {report.core.numpy_spec}")
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
