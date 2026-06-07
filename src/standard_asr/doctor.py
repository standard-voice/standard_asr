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

_NUMPY_REQ = re.compile(r"^\s*numpy\b(?P<spec>[^;]*)", re.IGNORECASE)
_UPPER_LT_2 = re.compile(r"<\s*2(\.0)?\b")
_LOWER_GE_2 = re.compile(r">=\s*2\b")


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
    plugins: list[PluginNumpy] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    @property
    def has_conflict(self) -> bool:
        """Whether any conflict was detected.

        Returns:
            ``True`` if there is at least one conflict.
        """
        return bool(self.conflicts)


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

    numpy1_only = [p for p in report.plugins if p.numpy_spec and _UPPER_LT_2.search(p.numpy_spec)]
    numpy2_required = [
        p for p in report.plugins if p.numpy_spec and _LOWER_GE_2.search(p.numpy_spec)
    ]

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
    return "\n".join(lines)


__all__ = ["DoctorReport", "PluginNumpy", "diagnose", "format_report"]
