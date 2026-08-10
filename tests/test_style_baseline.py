# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The baseline appendix in STYLE.md must match the vendored Vale package.

STYLE.md adopts the Google developer documentation style guide as the pinned
copy under ``.vale/styles/Google``, and lists every rule in an appendix so a
contributor or an agent can read the baseline without network access. That
listing is only trustworthy if it cannot drift: a package upgrade that adds a
rule, drops one, or a ``.vale.ini`` edit that disables one -- repo-wide or for
a single glob of files -- must not leave the statute describing a baseline the
repository no longer has. Every section of the configuration is read, because
a rule switched off for named files is just as real a change to the baseline
as one switched off everywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STYLE = _REPO_ROOT / "STYLE.md"
_VALE_INI = _REPO_ROOT / ".vale.ini"
_GOOGLE_STYLES = _REPO_ROOT / ".vale" / "styles" / "Google"

_APPENDIX_HEADING = "## Appendix: the pinned baseline, rule by rule"
_ROW = re.compile(r"^\| `(?P<rule>\w+)` \| (?P<intent>[^|]+) \| (?P<status>[^|]+) \|$", re.M)


def _appendix_rows() -> dict[str, str]:
    """Parse the appendix table into ``{rule: status}``.

    Returns:
        Each listed rule mapped to its trimmed status cell.
    """
    text = _STYLE.read_text(encoding="utf-8")
    appendix = text.split(_APPENDIX_HEADING, 1)[1]
    return {m.group("rule"): m.group("status").strip() for m in _ROW.finditer(appendix)}


def _sections() -> dict[str, str]:
    """Split ``.vale.ini`` into ``{section header: body}``.

    Reading every section matters: the two generic ones are not the whole
    configuration, and a rule switched off for a glob of files is just as
    real a change to the baseline as one switched off repo-wide.

    Returns:
        Each bracketed section mapped to the text that follows it, up to the
        next section.
    """
    text = _VALE_INI.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for match in re.finditer(r"^\[(?P<name>[^\]]+)\]$(?P<body>.*?)(?=^\[|\Z)", text, re.M | re.S):
        out[f"[{match.group('name')}]"] = match.group("body")
    return out


def _disabled_google_rules(section: str) -> set[str]:
    """Collect the ``Google.X = NO`` rule names from one ``.vale.ini`` section.

    Args:
        section: The section header to read, for example ``"[*.md]"``.

    Returns:
        The disabled rule names, without the ``Google.`` prefix.
    """
    return set(re.findall(r"^Google\.(\w+) = NO", _sections()[section], re.M))


def _per_file_disabled_google_rules() -> set[str]:
    """Collect rules switched off only for specific files or globs.

    Returns:
        Rule names disabled in any section other than the two generic ones
        and the format map.
    """
    generic = {"[*.md]", "[*.py]", "[formats]"}
    found: set[str] = set()
    for name, body in _sections().items():
        if name in generic:
            continue
        found |= set(re.findall(r"^Google\.(\w+) = NO", body, re.M))
    return found


def test_appendix_lists_exactly_the_vendored_rules() -> None:
    vendored = {path.stem for path in _GOOGLE_STYLES.glob("*.yml")}
    listed = set(_appendix_rows())
    assert listed == vendored, (
        "STYLE.md's baseline appendix is out of sync with .vale/styles/Google. "
        f"Missing from the appendix: {sorted(vendored - listed)}. "
        f"Listed but not vendored: {sorted(listed - vendored)}."
    )


def test_appendix_status_matches_the_active_configuration() -> None:
    md_off = _disabled_google_rules("[*.md]")
    py_off = _disabled_google_rules("[*.py]")
    per_file_off = _per_file_disabled_google_rules()
    for rule, status in _appendix_rows().items():
        if rule in md_off:
            expected = "Off — house delta"
        elif rule in py_off:
            expected = "Off in Python only"
        else:
            expected = "Enforced"
        if rule not in md_off and rule in per_file_off:
            # Switched off for named files on top of the generic posture.
            # Reading only the two generic sections would leave the statute
            # describing a stricter baseline than the one in force.
            expected += ", minus named files"
        assert status == expected, (
            f"STYLE.md's appendix calls Google.{rule} {status!r}, but .vale.ini "
            f"makes it {expected!r}."
        )


def test_every_listed_rule_states_what_it_asks_for() -> None:
    # A rule name alone is not a baseline a reader can follow offline; the
    # point of the appendix is that the intent travels with the name.
    for rule, intent in (
        (m.group("rule"), m.group("intent").strip())
        for m in _ROW.finditer(_STYLE.read_text(encoding="utf-8").split(_APPENDIX_HEADING, 1)[1])
    ):
        assert len(intent) > 20, f"Google.{rule} has no usable intent line: {intent!r}"
