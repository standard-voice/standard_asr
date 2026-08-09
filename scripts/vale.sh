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

# Vale silently skips a path that does not exist (exit 0, no output, no
# error), so a renamed or deleted TARGETS entry would shrink the corpus
# without a trace. Refuse to run against a stale target list.
for target in "${TARGETS[@]}"; do
    if [[ ! -e "${target}" ]]; then
        echo "vale.sh: target '${target}' does not exist; update TARGETS." >&2
        exit 1
    fi
done

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
    # errors, hence the emptiness check on line output. The `|| status=$?`
    # keeps `set -e` from killing the script at the substitution when Vale
    # exits nonzero: the alerts must be printed BEFORE the gate fails, or
    # CI shows a red check over an empty log.
    status=0
    alerts="$(run_vale --output=line "$@")" || status=$?
    if [[ -n "${alerts}" ]]; then
        printf '%s\n' "${alerts}"
    fi
    if [[ "${status}" -gt 1 ]]; then
        # Exit codes above 1 are tool failures (for example, a broken
        # .vale.ini), not alert counts; relay them unchanged.
        echo "vale gate: Vale itself failed (exit ${status}); fix the tool error above." >&2
        exit "${status}"
    fi
    if [[ -n "${alerts}" || "${status}" -ne 0 ]]; then
        echo "vale gate: alerts found (the full run must stay at zero; see STYLE.md, Enforcement)." >&2
        exit 1
    fi
    echo "vale gate: clean (no alerts at any level)."
    ;;
--selfcheck)
    # Negative test: prove the exact gate composition (config, EXEMPT glob,
    # TARGETS) still sees the surfaces it claims to cover, by planting one
    # deliberately violating fixture inside each directory target and
    # requiring the gate run to flag every one. Guards against a config
    # edit, a glob typo, or a Vale upgrade silently shrinking coverage
    # while the gate stays green. Single-file targets need no fixture: the
    # existence check above covers them, and Vale lints an existing file
    # argument unconditionally. If a killed run leaves a fixture behind,
    # the next gate run flags it -- the fixtures are violations by design.
    # Known, documented gaps (STYLE.md, "Enforcement"): a module docstring
    # that follows the SPDX header, Python string literals, attribute
    # docstrings, and the text of a tight list item that owns a nested
    # sub-list.
    fixture_md="docs/vale-selfcheck-fixture-delete-me.md"
    fixture_src="src/vale_selfcheck_fixture_delete_me.py"
    fixture_tests="tests/vale_selfcheck_fixture_delete_me.py"
    cleanup() { rm -f "${fixture_md}" "${fixture_src}" "${fixture_tests}"; }
    trap cleanup EXIT
    printf 'A test sentence with a deliberate e.g. marker.\n' > "${fixture_md}"
    for py in "${fixture_src}" "${fixture_tests}"; do
        cat > "${py}" <<'PYEOF'
def example() -> None:
    """Function docstring with a deliberate e.g. marker."""
    # A comment with a deliberate e.g. marker.
PYEOF
    done
    status=0
    out="$(run_vale --output=line)" || status=$?
    if [[ "${status}" -gt 1 ]]; then
        printf '%s\n' "${out}"
        echo "vale selfcheck: Vale itself failed (exit ${status}); fix the tool error above." >&2
        exit "${status}"
    fi
    fail=0
    if ! printf '%s\n' "${out}" | grep -F "${fixture_md}" | grep -qF 'Google.Latin'; then
        echo "selfcheck FAILED: Markdown under docs/ is not being linted." >&2
        fail=1
    fi
    for py in "${fixture_src}" "${fixture_tests}"; do
        hits="$(printf '%s\n' "${out}" | grep -F "${py}" | grep -cF 'Google.Latin' || true)"
        if [[ "${hits}" -lt 2 ]]; then
            echo "selfcheck FAILED: expected the docstring AND the comment of ${py} to be linted (got ${hits} hits)." >&2
            fail=1
        fi
    done
    if [[ "${fail}" -ne 0 ]]; then
        exit 1
    fi
    echo "vale selfcheck: the gate composition sees docs/, src/, and tests/ (Markdown prose, docstrings, comments)."
    ;;
*)
    run_vale "$@"
    ;;
esac
