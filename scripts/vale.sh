#!/usr/bin/env bash
# Run Vale on every governed prose surface (STYLE.md, "Scope").
#
# This script is the single source of truth for WHAT gets linted; .vale.ini
# holds the rules. The --glob exemption exists because Vale cannot exempt
# whole files from inside .vale.ini (an empty "BasedOnStyles =" section does
# not clear styles). Chinese documents, historical files, working notes, and
# the pre-standard CHANGELOG entries are exempt per STYLE.md.
#
# Usage:
#   scripts/vale.sh                # human view: pretty output, all levels
#   scripts/vale.sh --gate         # CI gate: fails on ANY alert, any level
#   scripts/vale.sh --selfcheck    # prove the gate sees what it claims to see
#   VALE=/path/to/vale scripts/vale.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VALE="${VALE:-vale}"

TARGETS=(README.md CONTRIBUTING.md AGENTS.md STYLE.md TERMINOLOGY.md RELEASING.md
    docs src tests)
EXEMPT='!{docs/spec/specification.md,docs/design-notes/*,docs/research/*,docs/work_doc/*,docs/feat_plan/*,docs/legacy/*,docs/misc.md,work/*,CHANGELOG.md}'

run_vale() {
    # bash 3.2 (macOS /bin/bash) treats expanding an empty array under
    # `set -u` as an unbound-variable error, so extra args are passed
    # explicitly by each caller instead of through a shared ARGS array.
    "${VALE}" --glob="${EXEMPT}" "$@" "${TARGETS[@]}"
}

case "${1:-}" in
--gate)
    shift
    # STYLE.md, "Enforcement": the full run is kept at zero, so the gate
    # fails on ANY alert at ANY level. Vale's own exit code reports only
    # errors, hence the explicit emptiness check on line output.
    alerts="$("${VALE}" --glob="${EXEMPT}" --output=line "$@" "${TARGETS[@]}")"
    if [[ -n "${alerts}" ]]; then
        printf '%s\n' "${alerts}"
        echo "vale gate: alerts found (the full run must stay at zero; see STYLE.md, Enforcement)." >&2
        exit 1
    fi
    echo "vale gate: clean (no alerts at any level)."
    ;;
--selfcheck)
    # Negative test: prove the extraction pipeline still sees the surfaces
    # the gate claims to cover (Markdown prose, function docstrings, and
    # comments in .py files). Guards against a config or Vale upgrade
    # silently shrinking coverage to nothing while the gate stays green.
    # Known, documented gaps (STYLE.md, "Enforcement"): a module docstring
    # that follows the SPDX header, and Python string literals.
    tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' EXIT
    printf 'A test sentence with a deliberate e.g. marker.\n' > "${tmp}/fixture.md"
    cat > "${tmp}/fixture.py" <<'PYEOF'
# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Module docstring (a known extraction gap; not asserted on)."""


def example() -> None:
    """Function docstring with a deliberate e.g. marker."""
    # A comment with a deliberate e.g. marker.
PYEOF
    out_md="$("${VALE}" --output=line "${tmp}/fixture.md" || true)"
    out_py="$("${VALE}" --output=line "${tmp}/fixture.py" || true)"
    fail=0
    case "${out_md}" in
    *Google.Latin*) ;;
    *)
        echo "selfcheck FAILED: Markdown prose is not being linted." >&2
        fail=1
        ;;
    esac
    py_hits="$(printf '%s\n' "${out_py}" | grep -c 'Google.Latin' || true)"
    if [[ "${py_hits}" -lt 2 ]]; then
        echo "selfcheck FAILED: expected the function docstring AND the comment to be linted (got ${py_hits} hits)." >&2
        fail=1
    fi
    if [[ "${fail}" -ne 0 ]]; then
        exit 1
    fi
    echo "vale selfcheck: extraction covers Markdown, function docstrings, and comments."
    ;;
*)
    run_vale "$@"
    ;;
esac
