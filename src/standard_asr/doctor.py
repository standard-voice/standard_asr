# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Read-only dependency conflict diagnostic (``standard-asr doctor``).

Enumerates installed Standard ASR plugins, reads each plugin distribution's
declared ``numpy`` requirement, and reports conflicts that cannot coexist in a
single process -- most importantly the numpy 1.x-vs-2.x split (spec DEP.5). It
never resolves or installs anything; it only diagnoses and suggests remediation
(out-of-process isolation when a conflict is real).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from importlib.metadata import entry_points

from .discovery import ENTRYPOINT_GROUP

# Match a leading ``numpy`` requirement and capture its version specifier (the
# part before any environment marker). The extras group (e.g. ``numpy[foo]``) is
# discarded; we only care about the version constraints.
_NUMPY_REQ = re.compile(r"^\s*numpy\b(?:\[[^\]]*\])?(?P<spec>[^;]*)", re.IGNORECASE)

# Representative probe versions spanning the numpy 1.x / 2.x boundary. We use
# proper specifier-set membership (not regex token matching) so that ranges like
# ``>=1.26,<2.3`` (admits both), ``==1.26.*`` / ``~=1.26.0`` / ``>=1.21,<1.27``
# (numpy-1 only) are classified correctly. The probes bracket the meaningful
# inflection points: oldest supported 1.x, the 1.x ceiling, the 2.0 boundary,
# and well beyond the current 2.x line.
_NUMPY1_PROBES = ("1.21.0", "1.24.0", "1.26.4", "1.26.99", "1.99.99")
_NUMPY2_PROBES = ("2.0.0", "2.1.0", "2.3.0", "2.99.99")


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
    """

    python_version: str
    plugins: list[PluginNumpy] = field(default_factory=_empty_plugins)
    conflicts: list[str] = field(default_factory=_empty_strs)
    notes: list[str] = field(default_factory=_empty_strs)

    @property
    def has_conflict(self) -> bool:
        """Whether any conflict was detected.

        Returns:
            ``True`` if there is at least one conflict.
        """
        return bool(self.conflicts)


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

    Uses :class:`packaging.specifiers.SpecifierSet` membership (when the optional
    ``packaging`` library is installed) rather than token regexes, so bounded
    ranges and ``==``/``~=`` pins are handled correctly. When ``packaging`` is
    absent the classifier conservatively returns ``(False, False)`` (no hard
    split) so it never reports a conflict it cannot verify.

    Args:
        numpy_spec: The raw numpy specifier (e.g. ``"<2"``, ``"~=1.26.0"``,
            ``"(any)"``), or ``None`` when numpy is not required.

    Returns:
        A ``(numpy1_only, numpy2_required)`` pair. ``numpy1_only`` is ``True``
        when the spec admits a 1.x but no 2.x; ``numpy2_required`` is ``True``
        when it admits a 2.x but no 1.x. An unconstrained / both-admitting /
        unparseable spec, or a missing ``packaging``, yields ``(False, False)``.
    """
    if not numpy_spec:
        return (False, False)
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import Version
    except ImportError:
        return (False, False)
    raw = "" if numpy_spec == "(any)" else numpy_spec
    try:
        spec_set = SpecifierSet(raw)
    except InvalidSpecifier:
        return (False, False)
    admits1 = any(Version(p) in spec_set for p in _NUMPY1_PROBES)
    admits2 = any(Version(p) in spec_set for p in _NUMPY2_PROBES)
    return (admits1 and not admits2, admits2 and not admits1)


def _numpy_spec_for(requires: list[str] | None) -> str | None:
    """Extract the numpy version specifier from a distribution's requirements.

    Args:
        requires: The distribution's ``Requires-Dist`` entries.

    Returns:
        The numpy specifier string, or ``None`` if numpy is not required.
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
        report.notes.append(
            "Install the optional 'packaging' library for precise numpy "
            "conflict analysis; without it, version-range conflicts are not "
            "classified."
        )

    numpy1_only: list[PluginNumpy] = []
    numpy2_required: list[PluginNumpy] = []
    for p in report.plugins:
        only1, req2 = _classify_numpy(p.numpy_spec)
        if only1:
            numpy1_only.append(p)
        if req2:
            numpy2_required.append(p)

    if numpy1_only and numpy2_required:
        report.conflicts.append(
            "numpy 1.x vs 2.x conflict: "
            + ", ".join(f"{p.distribution} ({p.numpy_spec})" for p in numpy1_only)
            + " require numpy<2 while "
            + ", ".join(f"{p.distribution} ({p.numpy_spec})" for p in numpy2_required)
            + " require numpy>=2. They cannot share one process; run the "
            "conflicting plugin out-of-process (subprocess/server isolation)."
        )

    if sys.version_info >= (3, 13) and numpy1_only:
        report.conflicts.append(
            "On Python 3.13+ there is no numpy<2 wheel: "
            + ", ".join(p.distribution for p in numpy1_only)
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
    if report.has_conflict:
        lines.append("Conflicts:")
        lines.extend(f"  ! {c}" for c in report.conflicts)
    else:
        lines.append("No dependency conflicts detected.")
    if report.notes:
        lines.append("")
        lines.extend(f"  note: {n}" for n in report.notes)
    return "\n".join(lines)


__all__ = ["DoctorReport", "PluginNumpy", "diagnose", "format_report"]
