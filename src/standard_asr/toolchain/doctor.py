# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only diagnostics for ``standard-asr doctor``: dependencies and the Hugging Face endpoint.

Enumerates installed Standard ASR plugins, reads each plugin distribution's
declared ``numpy`` requirement, and reports conflicts that cannot coexist in a
single process -- most importantly the numpy 1.x-vs-2.x split. It
never resolves or installs anything; it only diagnoses and suggests remediation
(out-of-process isolation when a conflict is real).

Scope: doctor diagnoses ``numpy`` ONLY. numpy is the single shared native
dependency the standard itself has, and its 1.x-vs-2.x break is a clean C-ABI
split whose conflict is fully encoded in version specifiers -- so a version-range
intersection decides it. Other shared native libraries (torch CUDA build
variants; onnxruntime vs onnxruntime-gpu package identity) have fundamentally
different conflict models that version intersection cannot decide, so they are
explicitly known-uncovered today; their hard conflicts fall under the general
out-of-process isolation guidance.
See the per-library seam in :func:`_numpy_spec_for` for the rationale.

The endpoint check (``doctor --network``, :func:`check_endpoint`) is opt-in,
because it is the one diagnostic that sends a request off the machine. When
huggingface_hub is installed, the check sends one ``HEAD`` request to
huggingface_hub's endpoint with huggingface_hub's own client, so the request
uses the proxies and the certificates of a request from huggingface_hub. It
diagnoses reachability only: it downloads nothing, and it does not test the
hosts a download can be redirected to.
"""

from __future__ import annotations

import importlib
import re
import socket
import ssl
import sys
import threading
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import PackageNotFoundError, entry_points, requires
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from standard_asr.plugins.discovery import ENTRYPOINT_GROUP
from standard_asr.runtime.redaction import redact_url, url_user_information

if TYPE_CHECKING:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

# Display-only fallback for the packaging-absent path. ``packaging`` is the
# authoritative parser (it evaluates environment markers and the legacy
# parenthesized form ``numpy (>=1.26)``); this pattern is used solely to render a
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
# iff its intersection with the corresponding range is satisfiable, decided by
# ``_intersection_satisfiability``.
_NUMPY1_RANGE = ">=1.0,<2.0"
_NUMPY2_RANGE = ">=2.0,<3.0"

# A sentinel "arbitrarily large" release, used to witness that an open upper
# bound (``>=``/``>`` with no ceiling) is satisfiable. Any real numpy pin is far
# below this, so it is a safe stand-in for "+infinity" when probing emptiness.
_OPEN_UPPER_SENTINEL = "100000.0.0"

# A large component value used to construct a version that sits *just below* a
# boundary at a given release position (for example, just under ``2.1`` -> ``2.0.<big>``).
_JUST_BELOW_FILL = 99999


def _empty_plugins() -> list[PluginNumpy]:
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
        numpy_spec: The raw numpy version specifier (for example, ``"<2"``), or ``None``.
    """

    entrypoint: str
    distribution: str
    numpy_spec: str | None


#: Seconds the endpoint check waits for an answer, name resolution included.
ENDPOINT_CHECK_TIMEOUT: Final = 5.0

EndpointOutcome = Literal[
    "reachable",
    "not_applicable",
    "could_not_run",
    "offline",
    "proxy_refused",
    "server_error",
    "name_not_resolved",
    "connection_refused",
    "timed_out",
    "certificate_not_verified",
    "failed",
]
"""How the endpoint check ended; :class:`EndpointCheck` describes each value."""

#: The outcomes that leave the doctor verdict clean.
_CLEAN_OUTCOMES: Final[frozenset[str]] = frozenset({"reachable", "not_applicable"})

#: Shown with every outcome except ``could_not_run`` and ``not_applicable``:
#: what the check cannot see.
NOT_CHECKED: Final = (
    "Not checked: whether a model's weights download, which also needs access to the "
    "repository, a token if it requires one, and disk space. Also not checked: the storage "
    "hosts a file can be redirected to, and files stored with Xet, which go through hf_xet, "
    "a separate client."
)


@dataclass(frozen=True)
class EndpointCheck:
    """The result of the endpoint check (``standard-asr doctor --network``).

    The check sends one ``HEAD`` request, without following a redirect, to
    ``huggingface_hub.constants.ENDPOINT`` (huggingface_hub's reading of
    ``HF_ENDPOINT``) with ``huggingface_hub.get_session()``, the client
    huggingface_hub sends its own requests with. So the request uses the
    proxies and the certificates of a request from huggingface_hub, also
    when an application replaced the client with
    ``huggingface_hub.set_client_factory()``; such a client can send more
    than one request, for example to answer an authentication challenge.

    Every configured URL in the result has its user information, the
    ``user:password`` part before the ``@``, replaced with ``***``. The
    result names each error by its type and never quotes the error's
    message: a client's message can quote part of a URL without the URL
    itself, such as the piece of a password a parser took for a port, and a
    rule that reads only the message cannot find that piece. The one
    exception is OpenSSL's description of why a certificate did not verify,
    such as ``self-signed certificate``, without the host name the ``ssl``
    module adds to a name mismatch.

    Attributes:
        outcome: How the check ended:

            * ``reachable``: an HTTP answer arrived, from the endpoint or from
              a proxy on the way, with a status other than 407 or 5xx.
            * ``not_applicable``: huggingface_hub is not installed, so no
              engine here downloads through it.
            * ``could_not_run``: huggingface_hub failed to import or to load
              its settings, or did not give the check a client it can drive;
              the result says nothing about the network.
            * ``offline``: huggingface_hub's offline mode refused the request.
            * ``proxy_refused``: a proxy refused the request, or answered 407.
            * ``server_error``: a server on the way answered with a 5xx status.
            * ``name_not_resolved``, ``connection_refused``, ``timed_out``,
              and ``certificate_not_verified``: the failure the request ended
              with; ``timed_out`` includes a name lookup that stalled past
              the deadline.
            * ``failed``: any other failure, described in ``detail``.

        endpoint: The endpoint URL, or ``None`` when huggingface_hub could not
            be read.
        client: The library and the HTTP client the check used or tried to
            use, such as ``huggingface_hub 1.19.0 (httpx)``, or ``None`` when
            huggingface_hub could not be imported.
        status_code: The HTTP status of the answer, when one arrived.
        detail: What happened, in one or two sentences.
        remedy: What to do about a problem, or ``None``.
        proxies: The proxies Python resolves for this process, by scheme,
            from ``urllib.request.getproxies()``, which httpx and requests
            read too. The ``no`` key lists the hosts that skip the proxy, an
            entry with user information masked like a URL.
        ambiguous_urls: The names of the configured URLs with a ``/``,
            ``?``, or ``#`` in the part the result masks, between the scheme
            and the last ``@``: ``endpoint``, or a proxy's scheme in
            ``proxies``. A client reads the user name, the password, the
            host, and the port only up to the first of those characters.
            When one belongs to the user name or the password, the client
            takes part of them for the host or the port, and the result
            shows a host the client does not use.
    """

    outcome: EndpointOutcome
    endpoint: str | None
    client: str | None
    status_code: int | None
    detail: str
    remedy: str | None
    proxies: Mapping[str, str]
    ambiguous_urls: tuple[str, ...] = ()

    @property
    def is_problem(self) -> bool:
        """Whether the outcome makes the doctor verdict non-clean.

        Returns:
            ``False`` for ``reachable`` and ``not_applicable``, ``True``
            otherwise.
        """
        return self.outcome not in _CLEAN_OUTCOMES

    @property
    def plain_http(self) -> bool:
        """Whether the endpoint uses plain HTTP.

        Returns:
            ``True`` when the endpoint URL starts with ``http://``.
        """
        return self.endpoint is not None and self.endpoint.lower().startswith("http://")


@dataclass
class DoctorReport:
    """The result of a dependency diagnosis.

    Args:
        python_version: The running interpreter version (``X.Y``).
        plugins: The discovered plugins and their numpy requirements.
        core: The standard-asr core's own effective numpy requirement, as a
            conflict participant. ``None`` in four states: no plugins are
            installed (nothing to conflict with); the core distribution's
            metadata is unreadable (non-clean, ``analysis_unavailable``); the
            metadata is readable but declares no interpreter-applicable numpy
            line (disclosed via a note); or ``packaging`` is unavailable (the
            marker-blind display fallback would show the wrong floor, so no
            row is shown; ``analysis_unavailable`` covers it). Core is ALWAYS
            in the process, so its
            interpreter-conditional numpy floor MUST join the intersection: a
            plugin whose numpy range excludes the core floor can never run
            with standard-asr on this interpreter, and omitting core from the
            analysis let doctor report such an environment clean.
        conflicts: Human-readable conflict descriptions.
        notes: Supplementary remediation hints (non-verdict footer lines).
        analysis_unavailable: Whether conflict analysis could not (fully) run
            with plugins installed -- the optional ``packaging`` distribution
            is missing, the core's own distribution metadata is unreadable so
            the plugin-vs-core relation never ran, or one or more range
            relations came back :attr:`Satisfiability.UNKNOWN` (an older
            ``packaging`` without the exact ``is_unsatisfiable`` oracle,
            where witness probing found no satisfying version but cannot
            prove emptiness) -- so the environment cannot be proven
            conflict-free. This is a non-clean state distinct from "no
            conflicts detected"; the accompanying note names the specific
            gap.
        endpoint: The result of the endpoint check, or ``None`` when it was
            not requested (``diagnose(network=False)``).
    """

    python_version: str
    plugins: list[PluginNumpy] = field(default_factory=_empty_plugins)
    core: PluginNumpy | None = None
    conflicts: list[str] = field(default_factory=_empty_strs)
    notes: list[str] = field(default_factory=_empty_strs)
    analysis_unavailable: bool = False
    endpoint: EndpointCheck | None = None

    @property
    def has_conflict(self) -> bool:
        """Whether any conflict was detected.

        Returns:
            ``True`` if there is at least one conflict.
        """
        return bool(self.conflicts)

    @property
    def dependencies_clean(self) -> bool:
        """Whether the dependency analysis proved the environment conflict-free.

        Clean requires both that no conflict was detected and that conflict
        analysis ran (``analysis_unavailable`` is a non-clean state: an
        unprovable environment must not read as clean). The
        :func:`format_report` headline for the dependencies reads it.

        Returns:
            ``True`` when no conflict was detected and analysis ran.
        """
        return not self.has_conflict and not self.analysis_unavailable

    @property
    def is_clean(self) -> bool:
        """Whether every check that ran found no problem (the verdict).

        The single source of the doctor verdict, consumed by the CLI exit
        code: the dependencies are clean, and the endpoint check, when it
        ran, found no problem (:attr:`EndpointCheck.is_problem`). A future
        non-clean state is wired in here once, not re-derived in every
        consumer.

        Returns:
            ``True`` when every check that ran found no problem.
        """
        endpoint_ok = self.endpoint is None or not self.endpoint.is_problem
        return self.dependencies_clean and endpoint_ok


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


@dataclass(frozen=True)
class _NumpyMajorSplit:
    """A specifier's classification against the numpy 1.x/2.x boundary.

    Args:
        numpy1_only: The spec provably admits a 1.x and provably admits no 2.x.
        numpy2_required: The spec provably admits a 2.x and provably admits no
            1.x.
        undecided: At least one side of the classification came back
            :attr:`Satisfiability.UNKNOWN` -- the split cannot be asserted
            either way (feeds ``analysis_unavailable``, never a conflict).
    """

    numpy1_only: bool = False
    numpy2_required: bool = False
    undecided: bool = False


def _classify_numpy(numpy_spec: str | None) -> _NumpyMajorSplit:
    """Classify a numpy specifier against the 1.x/2.x major boundary.

    Whether the spec admits a 1.x (resp. 2.x) release is decided by
    intersecting it with the full major range (``[1.0, 2.0)`` / ``[2.0,
    3.0)``) via :func:`_intersection_satisfiability` -- never against a fixed
    probe grid, which misreads an exact off-grid pin such as ``==2.2.0`` as
    admitting neither major. Both "only" verdicts are EXACT-side claims: a
    side is asserted only when the admitted major is provably satisfiable AND
    the excluded major is provably empty; an :attr:`Satisfiability.UNKNOWN`
    on either side yields ``undecided`` instead (a classification that cannot
    be proven must not manufacture a conflict). When ``packaging`` is absent
    the classifier conservatively returns the all-``False`` split.

    Args:
        numpy_spec: The raw numpy specifier (for example, ``"<2"``, ``"~=1.26.0"``,
            ``"(any)"``), or ``None`` when numpy is not required.

    Returns:
        The :class:`_NumpyMajorSplit` for the spec. An unconstrained /
        both-admitting / unparseable spec, or a missing ``packaging``, yields
        the default all-``False`` split.
    """
    spec_set = _specset(numpy_spec)
    if spec_set is None:
        return _NumpyMajorSplit()
    from packaging.specifiers import SpecifierSet

    admits1 = _intersection_satisfiability([spec_set, SpecifierSet(_NUMPY1_RANGE)])
    admits2 = _intersection_satisfiability([spec_set, SpecifierSet(_NUMPY2_RANGE)])
    if Satisfiability.UNKNOWN in (admits1, admits2):
        return _NumpyMajorSplit(undecided=True)
    return _NumpyMajorSplit(
        numpy1_only=admits1 is Satisfiability.SAT and admits2 is Satisfiability.UNSAT,
        numpy2_required=admits2 is Satisfiability.SAT and admits1 is Satisfiability.UNSAT,
    )


def _specset(numpy_spec: str | None) -> SpecifierSet | None:
    """Parse a numpy specifier string into a ``SpecifierSet``.

    Args:
        numpy_spec: The effective numpy specifier (for example, ``"<2"``, ``"~=1.26.0"``,
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
    bound is recognized as satisfiable) and ``0`` (so an open ``<``/``<=`` upper
    bound is too).

    Coverage is **approximate, not total**: PEP 440 release tuples have
    unbounded length, so no finite candidate set finds a witness for every
    conceivable interval (for example, ``>2.0,<2.0.0.0.1`` needs a five-component
    witness). The candidate set covers release- and
    one-sub-release-granularity intervals -- with each edge's EPOCH preserved
    on every release-derived candidate, so an epoch interval such as
    ``>1!1.0,<1!1.1`` is witnessed by ``1!1.0.1`` -- plus same-kind
    pre/dev/post windows via :func:`_segment_neighbor_candidates`
    (``>2.0rc1,<2.0rc3`` is witnessed by ``2.0rc2``) and adjacent windows via
    each edge's ``.dev0`` (``>2.0rc1,<2.0rc2`` is witnessed by
    ``2.0rc2.dev0``) -- which covers numpy's real pins and the canonical
    1.x/2.x split. Adding candidates is one-directionally safe: witnesses can
    only prove :attr:`Satisfiability.SAT`, never emptiness -- which is
    exactly why :func:`_intersection_satisfiability` answers
    :attr:`Satisfiability.UNKNOWN` (not ``UNSAT``) when no candidate is a
    member.

    Args:
        combined: The merged specifier whose satisfiability is being probed.

    Returns:
        A set of candidate version strings. Most are final releases (carrying
        the edge's epoch where one exists); a verbatim edge string may carry a
        pre-release segment, which by PEP 440 membership only matches a
        specifier that itself admits it -- so it never manufactures a false
        witness.
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
        except Exception:  # noqa: BLE001 - skip a non-version edge (for example, a ``===`` URL)
            continue
        # Keep the edge verbatim (``str(Version(edge))`` canonicalizes but
        # preserves epoch/pre-release), so an epoch/pre-release pin witnesses
        # itself. A bare ``release`` candidate would silently drop those segments.
        candidates.add(str(parsed))
        # Neighbors WITHIN a pre/dev/post segment: a satisfiable interval
        # between two same-kind pre-releases (``>2.0rc1,<2.0rc3`` -- witness
        # ``2.0rc2``; likewise a1/a3, dev1/dev3, post1/post3) has no witness
        # among release-level candidates, because every final-release
        # neighbor sorts outside the pre/dev window. Bump the innermost
        # segment by one in each direction, rebuilding the version prefix
        # around it, so those windows are witnessed too (adding candidates
        # remains one-directionally safe).
        candidates.update(_segment_neighbor_candidates(parsed))
        # The dev-0 release immediately below the edge: ``X.dev0`` sorts just
        # under ``X`` while staying above any earlier release, so it witnesses
        # an ADJACENT window whose interior holds no same-segment neighbor
        # (``>2.0rc1,<2.0rc2`` -- witness ``2.0rc2.dev0``; PEP 440 bars the
        # rc1.post* family from ``>2.0rc1``, but a dev of the UPPER edge is
        # admitted). Skipped when the edge already carries a dev segment
        # (its own neighbors above cover that shape).
        if parsed.dev is None:
            candidates.add(f"{parsed}.dev0")
        # Every release-derived candidate below MUST carry the edge's epoch:
        # PEP 440 orders epochs before everything else, so an epoch-0 neighbor
        # of a ``1!``-epoch edge sorts below the entire epoch and can never
        # witness an interval inside it (``>1!1.0,<1!1.1`` needs ``1!1.0.1``,
        # not ``1.0.1`` -- dropping the epoch here once misreported every
        # epoch range as empty).
        epoch = f"{parsed.epoch}!" if parsed.epoch else ""
        release = parsed.release
        candidates.add(epoch + (".".join(str(r) for r in release) or "0"))
        # One level deeper than the edge: ``release + (1,)`` sits inside the open
        # interval immediately above ``release`` (for example, ``2.0`` -> ``2.0.1``),
        # witnessing strict-lower-bound / next-edge pairs like ``>2.0,<2.1``.
        deeper_above = (*release, 1)
        candidates.add(epoch + ".".join(str(r) for r in deeper_above))
        for i in range(len(release)):
            # Just above this release position: bump component i, zero the tail.
            above = (*release[:i], release[i] + 1, *((0,) * (len(release) - i - 1)))
            candidates.add(epoch + ".".join(str(r) for r in above))
            # Just below: decrement component i (when > 0), fill the tail high,
            # and append one extra high-filled component so a sub-release-width
            # upper bound (``<2.0.1`` -> witness ``2.0.0.99999``) is covered.
            if release[i] > 0:
                below = (
                    *release[:i],
                    release[i] - 1,
                    *((_JUST_BELOW_FILL,) * (len(release) - i)),
                )
                candidates.add(epoch + ".".join(str(r) for r in below))
    return candidates


def _segment_neighbor_candidates(parsed: Version) -> set[str]:
    """Derive witnesses adjacent to a version's pre/dev/post segment.

    For an edge version carrying a pre-release, development, or post-release
    segment, return the versions one step above and below within that SAME
    segment (for example, ``2.0rc1`` -> ``2.0rc0`` / ``2.0rc2``;
    ``2.0rc1.dev3`` -> ``2.0rc1.dev2`` / ``2.0rc1.dev4``), preserving the
    epoch and every outer segment. These witness sub-release-width intervals
    such as ``>2.0rc1,<2.0rc3`` that the release-position candidates cannot
    (every final-release neighbor sorts outside the pre/dev window).

    Args:
        parsed: The parsed edge version.

    Returns:
        Candidate version strings (empty for a plain final release).
    """
    out: set[str] = set()
    epoch = f"{parsed.epoch}!" if parsed.epoch else ""
    release = ".".join(str(r) for r in parsed.release)
    pre = f"{parsed.pre[0]}{parsed.pre[1]}" if parsed.pre is not None else ""
    if parsed.pre is not None:
        label, n = parsed.pre
        for m in (n - 1, n + 1):
            if m >= 0:
                out.add(f"{epoch}{release}{label}{m}")
    if parsed.post is not None:
        for m in (parsed.post - 1, parsed.post + 1):
            if m >= 0:
                out.add(f"{epoch}{release}{pre}.post{m}")
    if parsed.dev is not None:
        post = f".post{parsed.post}" if parsed.post is not None else ""
        for m in (parsed.dev - 1, parsed.dev + 1):
            if m >= 0:
                out.add(f"{epoch}{release}{pre}{post}.dev{m}")
    return out


class Satisfiability(Enum):
    """Three-state satisfiability verdict for a specifier intersection.

    A finite witness search can PROVE satisfiability (a witness exists) but
    can never prove emptiness (PEP 440 release tuples are unbounded, so no
    finite candidate set is a completeness oracle). Collapsing "no witness
    found" into "unsatisfiable" is the logic error that once misreported
    perfectly runnable environments (an epoch range such as ``>1!1.0,<1!1.1``)
    as hard conflicts -- so the verdict is three-state, and only the exact
    oracle may ever produce :attr:`UNSAT`.
    """

    SAT = "sat"
    """A satisfying version provably exists (exact oracle, or a witness)."""

    UNSAT = "unsat"
    """Provably empty -- produced ONLY by packaging's exact algebra."""

    UNKNOWN = "unknown"
    """No witness found and no exact oracle available; emptiness unproven."""


def _exact_unsatisfiable(combined: SpecifierSet) -> bool | None:
    """Consult packaging's exact emptiness oracle, when this version has one.

    ``packaging >= 26.1`` exposes ``SpecifierSet.is_unsatisfiable()``, an
    exact decision over the specifier algebra (epochs, pre-releases,
    arbitrary-equality ``===`` included). Older versions lack it; treat any
    absence, raise, or non-bool answer as "no oracle" so the caller falls
    back to witness probing -- this defensive shape is also the test seam for
    simulating legacy ``packaging``.

    Args:
        combined: The merged specifier set.

    Returns:
        The oracle's boolean, or ``None`` when no trustworthy oracle exists.
    """
    oracle = getattr(combined, "is_unsatisfiable", None)
    if oracle is None:
        return None
    try:
        verdict = oracle()
    except Exception:  # noqa: BLE001 - a broken oracle degrades to witness probing
        return None
    return verdict if isinstance(verdict, bool) else None


def _intersection_satisfiability(specs: list[SpecifierSet]) -> Satisfiability:
    """Decide whether the intersection of numpy ``SpecifierSet``s is satisfiable.

    Computes the real combined specifier (``&``) and decides three-state:

    * **Exact** (``packaging >= 26.1``): ``SpecifierSet.is_unsatisfiable()``
      gives a definitive :attr:`Satisfiability.SAT` / :attr:`~Satisfiability.UNSAT`.
    * **Fallback** (older ``packaging``): probe the boundary-derived candidate
      versions (:func:`_emptiness_candidates`). A member proves
      :attr:`~Satisfiability.SAT`; finding none proves NOTHING -- the result
      is :attr:`~Satisfiability.UNKNOWN`, never ``UNSAT`` (a finite search
      cannot certify emptiness). Each membership test is guarded: on
      ``packaging <= 25.0`` an arbitrary-equality ``===`` edge re-parses its
      non-PEP440 version inside ``__contains__`` and raises; a raising
      candidate is simply not a witness.

    Consumers MUST treat only :attr:`~Satisfiability.UNSAT` as a conflict;
    :attr:`~Satisfiability.UNKNOWN` feeds ``analysis_unavailable`` (a
    non-clean, non-conflict state) so an unprovable environment is never
    reported clean NOR falsely condemned.

    Args:
        specs: The :class:`packaging.specifiers.SpecifierSet`s to intersect
            (per-plugin specs, or a spec paired with a major range for
            classification). Must be non-empty and contain only real specifier
            sets. A single internally unsatisfiable set (for example, ``<2`` and
            ``>=2.1`` declared by one plugin) is a valid input.

    Returns:
        The three-state satisfiability verdict for the combined specifier.
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    combined = SpecifierSet()
    for spec in specs:
        combined &= spec
    exact = _exact_unsatisfiable(combined)
    if exact is not None:
        return Satisfiability.UNSAT if exact else Satisfiability.SAT
    for candidate in _emptiness_candidates(combined):
        try:
            if Version(candidate) in combined:
                return Satisfiability.SAT
        except Exception:  # noqa: BLE001 - a raising candidate is just not a witness
            continue
    return Satisfiability.UNKNOWN


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
    regular-expression extraction (no marker evaluation, no conflict classification).

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
    # python_version doctor reports and makes it overridable -- for example, a test that
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

    A single distribution that ships several presets (``plugin-entry-points.md``
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

    The core declares an interpreter-conditional numpy floor (for example,
    ``numpy>=1.26`` below Python 3.13, ``numpy>=2.1`` at 3.13+), and the core is
    in EVERY process that runs standard_asr -- including any isolated worker a
    plugin would be moved into. Its requirement therefore belongs in the same
    intersection as the plugins': without it, a plugin declaring, for example,
    ``numpy<1.26`` read as conflict-free while resolution against core is
    impossible on this interpreter.

    Returns:
        The effective numpy specifier string (marker-evaluated for the running
        interpreter, same rules as :func:`_numpy_spec_for`); ``None`` when
        ``packaging`` is unavailable, OR when the metadata is readable but
        declares no numpy requirement applicable to this interpreter (two
        states the caller distinguishes via :func:`packaging_available` -- the
        latter is a fact about the declaration, not an analysis failure). The
        packaging-absent display fallback other rows use is marker-blind and
        would show the FIRST line of core's interpreter-conditional dual
        declaration -- ``>=1.26`` on a 3.13+ interpreter whose effective floor
        is ``>=2.1`` -- a factually wrong core requirement in the very row
        added to expose it, so no core row is shown at all in that mode (the
        report is already flagged ``analysis_unavailable`` there).

    Raises:
        PackageNotFoundError: If the core distribution's metadata is
            unreadable (an anomalous install; the caller reports it as a
            non-clean analysis gap -- silently conflating it with "no numpy
            line declared" would either false-alarm on repackaged cores or
            hide a genuinely broken install).
    """
    if not packaging_available():
        return None
    core_requires = requires(_CORE_DISTRIBUTION)
    return _numpy_spec_for(core_requires)


def _chain(error: BaseException) -> list[BaseException]:
    """Return an error and its causes, outermost first, cycle-safe."""
    links: list[BaseException] = []
    node: BaseException | None = error
    while node is not None and node not in links and len(links) < 12:
        links.append(node)
        node = node.__cause__ if node.__cause__ is not None else node.__context__
    return links


def _error_types(error: BaseException) -> str:
    """Name an error and its causes by their types, outermost first.

    The types stand in for the messages, which the endpoint check never
    quotes (see :class:`EndpointCheck`).
    """
    names: list[str] = []
    for link in _chain(error):
        cls = type(link)
        module = cls.__module__
        names.append(cls.__qualname__ if module == "builtins" else f"{module}.{cls.__qualname__}")
    return ", caused by ".join(names)


def _read_message(code: str) -> str:
    """Tell how to read the message of an error the report names by type only.

    The command names the interpreter that runs the check: the ``python``
    on the shell's path can be another one, as under ``pipx`` or ``uv run``.
    """
    python = sys.executable or "python"
    if any(char.isspace() for char in python):
        python = f'"{python}"'
    return (
        f'Run {python} -c "{code}" to see the message, which the report leaves out because it '
        "can quote part of a password."
    )


def _is_ambiguous(url: str) -> bool:
    """Return whether the user information of ``url`` holds a ``/``, ``?``, or ``#``.

    A client reads the user name, the password, the host, and the port only
    up to the first of those characters (RFC 3986, section 3.2), so it
    splits such user information and takes part of it for the host or the
    port.
    """
    user_information = url_user_information(url)
    return user_information is not None and any(char in user_information for char in "/?#")


#: The two verification errors whose description the ``ssl`` module extends
#: with the host name the client connected to, a host a misread URL can take
#: from a password. The report shows OpenSSL's description alone.
_NAME_MISMATCH: Final[dict[int, str]] = {62: "Hostname mismatch", 64: "IP address mismatch"}


def _client_family(session: object) -> Literal["httpx", "requests"] | None:
    """Name the HTTP client family a session belongs to, or ``None``.

    huggingface_hub 1.x returns an ``httpx.Client``, and 0.x a
    ``requests.Session``; an application can set a subclass of either. The
    class hierarchy decides, so the check imports neither library.
    """
    for cls in type(session).__mro__:
        if cls.__name__ == "Client" and cls.__module__.split(".")[0] == "httpx":
            return "httpx"
        if cls.__name__ == "Session" and cls.__module__.split(".")[0] == "requests":
            return "requests"
    return None


#: The remedy for a certificate that did not verify, by client family.
_CERTIFICATE_REMEDY: Final = {
    "httpx": (
        "If a proxy inspects TLS, set SSL_CERT_FILE to a certificate bundle that includes the "
        "proxy's root certificate: httpx trusts the certifi bundle, not the operating "
        "system's store."
    ),
    "requests": (
        "If a proxy inspects TLS, set REQUESTS_CA_BUNDLE to a certificate bundle that includes "
        "the proxy's root certificate: requests trusts the certifi bundle, not the operating "
        "system's store."
    ),
}

_TIMEOUT_REMEDY: Final = (
    "If this network reaches the internet through a proxy, set HTTPS_PROXY to it, or "
    "HTTP_PROXY for a plain-HTTP endpoint. If the endpoint is blocked here, set HF_ENDPOINT "
    "to an endpoint you trust that this machine reaches."
)


def _classify(
    error: BaseException, family: Literal["httpx", "requests"]
) -> tuple[EndpointOutcome, str, str | None]:
    """Classify a failed request from the error and its causes.

    The causes carry the standard-library error the client wrapped, so one
    classification serves both client families.
    """
    links = _chain(error)
    names = [type(link).__name__ for link in links]
    errors = _error_types(error)
    if "OfflineModeIsEnabled" in names:
        return (
            "offline",
            "huggingface_hub's offline mode refused the request, so no download can run.",
            "Unset HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE; huggingface_hub reads both.",
        )
    certificate = next(
        (link for link in links if isinstance(link, ssl.SSLCertVerificationError)), None
    )
    if certificate is not None:
        # OpenSSL's description of the failure, such as "self-signed
        # certificate in certificate chain", tells whether the remedy fits.
        code = getattr(certificate, "verify_code", None)
        mismatch = _NAME_MISMATCH.get(code) if isinstance(code, int) else None
        reason = mismatch or str(getattr(certificate, "verify_message", None) or "")
        reason = " ".join(reason.split()).rstrip(".")
        return (
            "certificate_not_verified",
            f"The TLS certificate did not verify{f': {reason}' if reason else ''} ({errors}).",
            _CERTIFICATE_REMEDY[family],
        )
    if any(isinstance(link, socket.gaierror) for link in links):
        return (
            "name_not_resolved",
            f"A host name, the endpoint's or the proxy's, did not resolve ({errors}).",
            "Check the host name in HF_ENDPOINT and in the proxy URL, if either is set, and "
            "that this machine can resolve host names.",
        )
    if any(isinstance(link, ConnectionRefusedError) for link in links):
        return (
            "connection_refused",
            f"The endpoint or the proxy refused the connection ({errors}).",
            "Check the host and the port in HF_ENDPOINT and in the proxy URL, if either is "
            "set. If this network requires a proxy, set HTTPS_PROXY.",
        )
    if any(isinstance(link, TimeoutError) or "Timeout" in name for link, name in zip(links, names)):
        return ("timed_out", f"No answer arrived in time ({errors}).", _TIMEOUT_REMEDY)
    if any(name.endswith("ProxyError") for name in names):
        return (
            "proxy_refused",
            f"A proxy refused the request ({errors}).",
            "Check the proxy URL and its credentials in HTTPS_PROXY or HTTP_PROXY, or list the "
            "endpoint's host in NO_PROXY if requests to it must not use the proxy.",
        )
    return (
        "failed",
        f"The request failed ({errors}).",
        _read_message(
            "from huggingface_hub import constants, get_session; "
            "get_session().head(constants.ENDPOINT)"
        ),
    )


def _client(
    utils: Any,
) -> tuple[Any, Literal["httpx", "requests"]] | tuple[str, str | None]:
    """Create huggingface_hub's client, or say why it cannot be used.

    Creating the client sends nothing, so it happens before the wait starts,
    and a failure here is a check that could not run.

    Returns:
        The session and its family, or the detail and the remedy of why the
        check cannot run.
    """
    try:
        session: Any = utils.get_session()
    except Exception as error:
        return (
            f"huggingface_hub could not create its HTTP client ({_error_types(error)}).",
            "Check the variables the client reads when it is created: the certificate path in "
            "SSL_CERT_FILE or SSL_CERT_DIR, and each proxy URL. "
            + _read_message("from huggingface_hub import get_session; get_session()"),
        )
    family = _client_family(session)
    if family is None:
        return (
            f"huggingface_hub returned a {type(session).__qualname__} client; doctor can send "
            "the request only with an httpx or a requests client.",
            None,
        )
    return session, family


def _probe(
    session: Any,
    family: Literal["httpx", "requests"],
    url: str,
    timeout: float,
    result: Callable[..., EndpointCheck],
) -> EndpointCheck:
    """Send the one request and classify the answer; see :func:`check_endpoint`."""
    try:
        # One request: a redirect is an answer, and following it would test
        # another host.
        if family == "httpx":
            response: Any = session.head(url, timeout=timeout, follow_redirects=False)
        else:
            response = session.head(url, timeout=timeout, allow_redirects=False)
    except Exception as error:
        outcome, detail, remedy = _classify(error, family)
        return result(outcome, detail, remedy)
    try:
        status = int(response.status_code)
    finally:
        response.close()
    if status == 407:
        return result(
            "proxy_refused",
            "The request got an HTTP answer with status 407: a proxy wants credentials.",
            "Put the proxy's user name and password in its URL in HTTPS_PROXY or HTTP_PROXY, "
            "as in http://user:password@proxy:8080.",
            status_code=status,
        )
    if status >= 500:
        return result(
            "server_error",
            f"The request got an HTTP answer with status {status}: a server on the way, the "
            "endpoint or a proxy, answered with an error.",
            "Try again later; if the error stays, ask whoever runs the endpoint or the proxy.",
            status_code=status,
        )
    return result(
        "reachable", f"The request got an HTTP answer with status {status}.", status_code=status
    )


def check_endpoint(*, timeout: float = ENDPOINT_CHECK_TIMEOUT) -> EndpointCheck:
    """Send one request to the Hugging Face endpoint and classify what happened.

    This is the check ``standard-asr doctor --network`` runs; it is the only
    part of doctor that sends a request off the machine. See
    :class:`EndpointCheck` for what the request is and what each outcome
    means. huggingface_hub is imported only here, and only when installed.

    The request has ``timeout`` as its own timeout. The wait for its answer
    also ends after ``timeout`` seconds when a name lookup stalls, which that
    timeout does not cover. The request runs on a daemon thread: when the
    time is up, the call returns ``timed_out``, and the thread can still
    finish the request in the background. The imports and the creation of
    the client, which send nothing, happen before the wait starts.

    A remedy that gives a command to print an error's message runs the
    interpreter that runs this check, with huggingface_hub's default client.
    An application that set its own client with ``set_client_factory()``
    sends the request again in its own process to read the message.

    Args:
        timeout: Seconds to wait for an answer.

    Returns:
        The result of the check.
    """
    configured = sorted(urllib.request.getproxies().items())
    # An entry of the no-proxy list is a host, but a URL pasted there is
    # masked like any other.
    proxies = {
        scheme: ",".join(map(redact_url, value.split(","))) if scheme == "no" else redact_url(value)
        for scheme, value in configured
    }
    ambiguous = tuple(
        scheme for scheme, value in configured if scheme != "no" and _is_ambiguous(value)
    )

    def not_run(detail: str, code: str) -> EndpointCheck:
        return EndpointCheck(
            "could_not_run", None, None, None, detail, _read_message(code), proxies, ambiguous
        )

    try:
        hub: Any = importlib.import_module("huggingface_hub")
    except ModuleNotFoundError as error:
        if error.name == "huggingface_hub":
            return EndpointCheck(
                "not_applicable",
                None,
                None,
                None,
                "huggingface_hub is not installed, so the endpoint check does not apply.",
                None,
                proxies,
                ambiguous,
            )
        return not_run(
            f"huggingface_hub is installed but failed to import ({_error_types(error)}).",
            "import huggingface_hub",
        )
    except Exception as error:
        return not_run(
            f"huggingface_hub is installed but failed to import ({_error_types(error)}).",
            "import huggingface_hub",
        )
    try:
        # Submodules, imported by name: huggingface_hub loads its attributes
        # lazily, and its lazy loader prints an import failure to stdout.
        constants: Any = importlib.import_module("huggingface_hub.constants")
        utils: Any = importlib.import_module("huggingface_hub.utils")
        url = str(constants.ENDPOINT)
    except Exception as error:
        return not_run(
            "huggingface_hub cannot be used: its endpoint setting or its client module failed "
            f"to load ({_error_types(error)}).",
            "from huggingface_hub import constants, utils; constants.ENDPOINT",
        )
    endpoint = redact_url(url)
    if _is_ambiguous(url):
        ambiguous = ("endpoint", *ambiguous)
    library = f"huggingface_hub {getattr(hub, '__version__', 'of unknown version')}"
    first, second = _client(utils)
    if isinstance(first, str):
        return EndpointCheck(
            "could_not_run", endpoint, library, None, first, second, proxies, ambiguous
        )
    session, family = first, cast('Literal["httpx", "requests"]', second)
    client = f"{library} ({family})"

    def result(
        outcome: EndpointOutcome,
        detail: str,
        remedy: str | None = None,
        *,
        status_code: int | None = None,
    ) -> EndpointCheck:
        return EndpointCheck(
            outcome, endpoint, client, status_code, detail, remedy, proxies, ambiguous
        )

    results: list[EndpointCheck] = []

    def run() -> None:
        try:
            results.append(_probe(session, family, url, timeout, result))
        except Exception as error:
            results.append(result("failed", f"The check failed ({_error_types(error)})."))

    worker = threading.Thread(target=run, name="standard-asr-endpoint-check", daemon=True)
    worker.start()
    worker.join(timeout)
    if results:
        return results[0]
    return result(
        "timed_out",
        f"No answer arrived within {timeout:g} seconds; the wait includes the name lookup.",
        _TIMEOUT_REMEDY,
    )


def diagnose(*, group: str = ENTRYPOINT_GROUP, network: bool = False) -> DoctorReport:
    """Diagnose numpy compatibility across installed plugins AND the core.

    Four conflict sources are analyzed, because their remediations differ and
    conflating them misdirects the user. First, a distribution whose OWN numpy
    declaration is internally unsatisfiable gets its own conflict (fix the
    plugin), and on Python 3.13+ a plugin requiring ``numpy<2`` gets a
    no-wheel conflict (use an older-Python worker). Then the two relations
    below, in order:

    1. **Plugin vs core** (see :func:`_core_numpy_spec`): core is in every
       process -- including any isolated worker -- so a plugin whose numpy
       range has no version in common with the core floor can never run with
       standard-asr on this interpreter AT ALL. Each such distribution gets
       its own dedicated conflict, and is then EXCLUDED from the
       environment-level analysis below: it cannot run in any process layout,
       so letting it participate would pollute the isolation advice (for example, a
       plugin-vs-plugin message telling the user to isolate a plugin that is
       equally dead in the isolated worker).
    2. **Environment level** among the remaining (core-compatible) plugins,
       plus core: the 1.x-vs-2.x split and the general empty-intersection
       check, whose remediation is out-of-process isolation.

    With ``network=True``, the report also carries the endpoint check
    (:func:`check_endpoint`), the one diagnostic that sends a request off the
    machine.

    Args:
        group: The entry-point group to inspect.
        network: Whether to run the endpoint check.

    Returns:
        A :class:`DoctorReport` describing plugins, the core requirement, any
        conflicts, and the endpoint check when requested.
    """
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    report = DoctorReport(python_version=py)

    for ep in entry_points(group=group):
        dist = ep.dist
        dist_name = dist.name if dist is not None else "<unknown>"
        spec = _numpy_spec_for(dist.requires if dist is not None else None)
        report.plugins.append(PluginNumpy(ep.name, dist_name, spec))

    if report.plugins:
        try:
            core_spec = _core_numpy_spec()
        except PackageNotFoundError:
            # Core metadata unreadable (with packaging present, so marker
            # evaluation was possible): the plugin-vs-core relation -- the
            # analysis half that catches an environment no process layout can
            # fix -- cannot run at all. That is a NON-CLEAN state, exactly
            # like the packaging-absent mode below: a footer note alone would
            # leave is_clean True and the headline claiming "No dependency
            # conflicts detected" while the core-floor gap this analysis
            # exists to close silently reopened.
            core_spec = None
            report.analysis_unavailable = True
            report.notes.append(
                "standard-asr's own distribution metadata is unreadable, so the "
                "plugin-vs-core numpy analysis could not run; the environment "
                "cannot be proven conflict-free."
            )
        if core_spec is not None:
            report.core = PluginNumpy("(core)", _CORE_DISTRIBUTION, core_spec)
        elif packaging_available() and not report.analysis_unavailable:
            # Readable metadata, no numpy line applicable to this interpreter
            # (for example, a repackaged/vendored core with rewritten requirements):
            # a fact about the declaration, NOT an unreadable-metadata
            # failure -- claiming "unreadable" here false-alarmed CI gates on
            # working environments. There is no core constraint to intersect,
            # so relation 1 has nothing to check; disclose rather than stay
            # silent (explicit over implicit).
            report.notes.append(
                "standard-asr's distribution metadata declares no numpy "
                "requirement applicable to this interpreter, so the "
                "plugin-vs-core analysis has nothing to check."
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

    # Undecidable relations (fallback witness probing without packaging's
    # exact oracle): every UNKNOWN verdict is recorded here and aggregated
    # into ONE analysis_unavailable note at the end -- an unprovable relation
    # is a non-clean state, never a conflict (only exact UNSAT convicts) and
    # never silence (an unproven environment must not read as clean).
    undecided_events: set[str] = set()

    # Relation 0 -- a distribution whose OWN numpy declaration is internally
    # unsatisfiable (for example, ``<2`` AND ``>=2.1``). It gets its dedicated
    # self-contradiction conflict PER DISTRIBUTION and is excluded from every
    # later relation: no numpy exists for it anywhere, so keeping it in the
    # joint analysis co-blamed innocent plugins and advised isolation --
    # advice that cannot fix a self-contradictory pin.
    self_unsatisfiable: list[PluginNumpy] = []
    seen_self_broken: set[str] = set()
    for p in report.plugins:
        if p.distribution in seen_self_broken:
            continue
        plugin_spec_set = _specset(p.numpy_spec)
        if plugin_spec_set is None:
            continue
        self_sat = _intersection_satisfiability([plugin_spec_set])
        if self_sat is Satisfiability.UNSAT:
            seen_self_broken.add(p.distribution)
            self_unsatisfiable.append(p)
        elif self_sat is Satisfiability.UNKNOWN:
            undecided_events.add(f"{p.distribution} ({p.numpy_spec}): self-satisfiability")
    for p in self_unsatisfiable:
        report.conflicts.append(
            f"numpy version conflict: {p.distribution} ({p.numpy_spec}) declares "
            "an internally unsatisfiable numpy range (no version can satisfy "
            "it; computed exactly by packaging's specifier algebra). Fix the "
            "plugin's numpy requirement."
        )

    # Relation 1 -- plugin vs core, per distribution (self-contradictory
    # distributions already reported and excluded above).
    core_spec_set = _specset(report.core.numpy_spec) if report.core is not None else None
    core_incompatible: list[PluginNumpy] = []
    if report.core is not None and core_spec_set is not None:
        seen_core_conflict_dists: set[str] = set()
        for p in report.plugins:
            if p.distribution in seen_core_conflict_dists or p.distribution in seen_self_broken:
                continue
            plugin_spec_set = _specset(p.numpy_spec)
            if plugin_spec_set is None:
                continue
            core_sat = _intersection_satisfiability([plugin_spec_set, core_spec_set])
            if core_sat is Satisfiability.UNSAT:
                seen_core_conflict_dists.add(p.distribution)
                core_incompatible.append(p)
            elif core_sat is Satisfiability.UNKNOWN:
                undecided_events.add(f"{p.distribution} ({p.numpy_spec}): vs standard-asr core")
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
    excluded_dists = {p.distribution for p in core_incompatible} | seen_self_broken
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
        split = _classify_numpy(p.numpy_spec)
        if split.undecided:
            undecided_events.add(f"{p.distribution} ({p.numpy_spec}): 1.x/2.x classification")
        if split.numpy1_only:
            numpy1_only.append(p)
        if split.numpy2_required:
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
    elif spec_sets:
        joint_sat = _intersection_satisfiability(spec_sets)
        if joint_sat is Satisfiability.UNSAT:
            # Real-intersection conflict that the 1.x/2.x classification alone
            # misses -- for example, disjoint same-major ranges (``==2.0.*`` vs ``>=2.3``)
            # that share no satisfying numpy release. Self-contradictory
            # distributions were already reported and excluded in relation 0, so
            # every participant here is individually satisfiable and attribution
            # reduces to one question: is the PLUGIN-ONLY intersection already
            # empty? If yes, this is a plugin-vs-plugin conflict -- core must be
            # neither listed nor blamed (naming it would invert the remediation;
            # out-of-process isolation IS the fix). Core joins the listing (and
            # flips the remedy) only when the plugins alone are satisfiable and
            # adding core empties the intersection. (Both questions are answered
            # by the same exact oracle that produced this UNSAT, so the
            # attribution can never itself be a guess.)
            plugin_pairs = [(p, s) for p, s in zip(constrained, spec_sets) if p is not report.core]
            plugin_constrained = [p for p, _ in plugin_pairs]
            plugin_specs = [s for _, s in plugin_pairs]
            plugins_alone_empty = (
                bool(plugin_specs)
                and _intersection_satisfiability(plugin_specs) is Satisfiability.UNSAT
            )
            sides = plugin_constrained if plugins_alone_empty else constrained
            listing = _render_distributions(sides)
            report.conflicts.append(
                f"numpy version conflict: {listing} declare numpy ranges with no "
                "common satisfying version." + _remedy(sides)
            )
        elif joint_sat is Satisfiability.UNKNOWN:
            undecided_events.add(
                "joint environment intersection ("
                + ", ".join(_unique_distributions(constrained))
                + ")"
            )

    if sys.version_info >= (3, 13):
        # Computed over ALL plugins, not the exclusion-filtered participants:
        # a numpy<2 plugin on 3.13 conflicts with the core floor (relation 1
        # reports that), AND no numpy<2 wheel exists for this interpreter at
        # all -- an independent fact worth stating, and one isolation cannot
        # fix (the wheel is missing in every process on this interpreter).
        # (An undecided classification was already recorded above for every
        # participant; excluded distributions never reach the classifier a
        # second time undetected because numpy1_only is exact-only.)
        numpy1_declared = [p for p in report.plugins if _classify_numpy(p.numpy_spec).numpy1_only]
        if numpy1_declared:
            report.conflicts.append(
                "On Python 3.13+ there is no numpy<2 wheel: "
                + ", ".join(_unique_distributions(numpy1_declared))
                + " cannot be installed on this interpreter in any process. Run "
                "that plugin under Python <3.13 (for example, an isolated older-Python "
                "worker)."
            )

    if undecided_events:
        # UNKNOWN is a non-clean, non-conflict state: witness probing found no
        # satisfying version for these relations but (without packaging's
        # exact oracle) cannot prove emptiness either. Reporting them as
        # conflicts would condemn possibly runnable environments; staying
        # silent would let an unproven environment read as clean. One
        # aggregated note, honest about both directions.
        report.analysis_unavailable = True
        report.notes.append(
            "Some numpy range relations could not be decided exactly: "
            + "; ".join(sorted(undecided_events))
            + ". Witness probing found no satisfying version for them, but "
            "only packaging's exact algebra can prove emptiness -- upgrade "
            "the optional 'packaging' library to a version providing "
            "SpecifierSet.is_unsatisfiable() (>= 26.1) for exact verdicts."
        )

    if network:
        report.endpoint = check_endpoint()
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
    if report.dependencies_clean:
        # The clean claim is gated on the dependency verdict, so a new
        # non-clean dependency state can never read as clean here (it still
        # needs its own rendering branch below).
        lines.append("No dependency conflicts detected.")
    elif report.has_conflict:
        lines.append("Conflicts:")
        lines.extend(f"  ! {c}" for c in report.conflicts)
    else:
        # Non-clean without a classified conflict: analysis could not run.
        # Claiming "no conflicts" here would be a silent wrong result; the
        # headline must carry the non-clean state.
        lines.append(
            "Conflict analysis unavailable or incomplete; the environment "
            "cannot be proven conflict-free (see the note below for the "
            "specific gap)."
        )
    if report.notes:
        lines.append("")
        lines.extend(f"  note: {n}" for n in report.notes)
    lines.append("")
    lines.extend(_format_endpoint(report.endpoint))
    return "\n".join(lines)


def _format_endpoint(check: EndpointCheck | None) -> list[str]:
    """Render the endpoint check as the report's network section."""
    if check is None:
        return [
            "Network: not checked. Run 'standard-asr doctor --network' to send one request "
            "to the Hugging Face endpoint and see whether this machine reaches it."
        ]
    lines = ["Network (one request to the Hugging Face endpoint):"]
    if check.endpoint is not None:
        lines.append(f"  endpoint: {check.endpoint}")
    if check.client is not None:
        lines.append(f"  client: {check.client}")
    proxies = [f"{scheme}={value}" for scheme, value in check.proxies.items() if scheme != "no"]
    lines.append(f"  proxies: {', '.join(proxies) if proxies else 'none'}")
    if "no" in check.proxies:
        lines.append(f"  no proxy for: {check.proxies['no']}")
    if check.outcome == "not_applicable":
        lines.append(f"  [INFO] {check.detail}")
        return lines
    marker = "[FAIL]" if check.is_problem else "[OK]"
    lines.append(f"  {marker} {check.detail}")
    if check.remedy is not None:
        lines.append(f"  fix: {check.remedy}")
    if check.plain_http:
        lines.append(
            "  [INFO] The endpoint uses plain HTTP: requests and answers travel without "
            "encryption, so any host on the network path can read or change them."
        )
    for name in check.ambiguous_urls:
        url = "The endpoint URL" if name == "endpoint" else f"The {name} proxy URL"
        lines.append(
            f"  [INFO] {url} has a '/', '?', or '#' after the scheme and before its last '@'. "
            "A client reads the user name, password, host, and port only up to that character. "
            "If it is in the user name or the password, percent-encode it: '/' as %2F, '?' as "
            "%3F, '#' as %23."
        )
    if check.outcome != "could_not_run":
        lines.append(f"  {NOT_CHECKED}")
    return lines


__all__ = [
    "ENDPOINT_CHECK_TIMEOUT",
    "DoctorReport",
    "EndpointCheck",
    "EndpointOutcome",
    "PluginNumpy",
    "check_endpoint",
    "diagnose",
    "format_report",
]
