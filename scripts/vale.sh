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
# --selfcheck runs Vale from a temporary mirror directory, so a relative
# VALE path (CI passes .tools/vale/vale) has to be resolved here, while the
# working directory is still the repo root. A bare name keeps its PATH lookup.
case "${VALE}" in
*/*) VALE="$(cd "$(dirname "${VALE}")" && pwd)/$(basename "${VALE}")" ;;
esac

TARGETS=(README.md CONTRIBUTING.md AGENTS.md STYLE.md TERMINOLOGY.md RELEASING.md
    docs src tests)
EXEMPT='!{docs/spec/specification.md,docs/design-notes/*,docs/research/*,docs/work_doc/*,docs/feat_plan/*,docs/legacy/*,docs/misc.md,work/*,CHANGELOG.md}'

# Root Markdown that is deliberately NOT gate-governed, with the reason
# STYLE.md gives: the pre-standard CHANGELOG history the gate cannot separate
# from new entries (review owns those), and a one-line tool pointer that
# carries no prose.
UNGOVERNED_ROOT_MD=(CHANGELOG.md CLAUDE.md)

# Vale silently skips a path that does not exist (exit 0, no output, no
# error), so a renamed or deleted TARGETS entry would shrink the corpus
# without a trace. Refuse to run against a stale target list.
for target in "${TARGETS[@]}"; do
    if [[ ! -e "${target}" ]]; then
        echo "vale.sh: target '${target}' does not exist; update TARGETS." >&2
        exit 1
    fi
done

# The selfcheck derives its expectations FROM TARGETS, so it cannot notice a
# target that was dropped from the list. Close that by accounting for every
# root document against STYLE.md, "Scope": each is either governed or
# explicitly exempt. Dropping a target, or adding a root document and
# forgetting to govern it, now fails here instead of silently shrinking the
# corpus.
for md in *.md; do
    case " ${TARGETS[*]} ${UNGOVERNED_ROOT_MD[*]} " in
    *" ${md} "*) ;;
    *)
        echo "vale.sh: root document '${md}' is neither a Vale target nor listed" >&2
        echo "  as ungoverned. Add it to TARGETS (STYLE.md, \"Scope\") or to" >&2
        echo "  UNGOVERNED_ROOT_MD with the reason it is exempt." >&2
        exit 1
        ;;
    esac
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
    # TARGETS) still reaches EVERY target. It mirrors the target layout into
    # a temporary directory -- same relative paths, same .vale.ini, same
    # styles -- plants a deliberate violation in each, and runs `run_vale`
    # there, so the composition under test is the same function the gate
    # calls. Guards against a config edit, a glob typo, or a Vale upgrade
    # silently shrinking coverage while the gate stays green.
    #
    # The mirror exists because the working tree is not a safe scratch pad:
    # planting fixtures in docs/, src/, and tests/ (the previous design)
    # truncated any same-named file, raced with a concurrent run, and left
    # debris behind when the run was killed. It also covers the single-file
    # targets, which cannot hold a planted fixture at all: the --glob is
    # applied even to an explicitly passed file argument (verified against
    # the pinned Vale), so a typo in EXEMPT can silently drop README.md
    # while both the gate and the existence check above stay green.
    #
    # Known, documented gaps (STYLE.md, "Enforcement"): a module docstring
    # that follows the SPDX header, Python string literals, attribute
    # docstrings, and the text of a tight list item that owns a nested
    # sub-list.
    tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' EXIT
    cp .vale.ini "${tmp}/"
    cp -R .vale "${tmp}/"
    md_probes=()
    py_probes=()
    for target in "${TARGETS[@]}"; do
        if [[ -d "${target}" ]]; then
            mkdir -p "${tmp}/${target}"
            probe="${target}/vale-selfcheck-probe.md"
            case "${target}" in
            src | tests)
                # The .py targets also carry tier-2 and tier-3 prose, so
                # probe docstring and comment extraction there.
                py_probe="${target}/vale_selfcheck_probe.py"
                cat > "${tmp}/${py_probe}" <<'PYEOF'
def example() -> None:
    """Function docstring with a deliberate e.g. marker."""
    # A comment with a deliberate e.g. marker.
PYEOF
                py_probes+=("${py_probe}")
                ;;
            esac
        else
            probe="${target}"
        fi
        printf 'A probe sentence with a deliberate e.g. marker.\n' > "${tmp}/${probe}"
        md_probes+=("${probe}")
    done

    status=0
    out="$(cd "${tmp}" && run_vale --output=line)" || status=$?
    if [[ "${status}" -gt 1 ]]; then
        printf '%s\n' "${out}"
        echo "vale selfcheck: Vale itself failed (exit ${status}); fix the tool error above." >&2
        exit "${status}"
    fi
    fail=0
    for probe in "${md_probes[@]}"; do
        if ! printf '%s\n' "${out}" | grep -F "${probe}:" | grep -qF 'Google.Latin'; then
            echo "selfcheck FAILED: the gate composition does not reach '${probe}'." >&2
            fail=1
        fi
    done
    for probe in "${py_probes[@]}"; do
        hits="$(printf '%s\n' "${out}" | grep -F "${probe}:" | grep -cF 'Google.Latin' || true)"
        if [[ "${hits}" -lt 2 ]]; then
            echo "selfcheck FAILED: expected the docstring AND the comment of '${probe}' to be linted (got ${hits})." >&2
            fail=1
        fi
    done
    if [[ "${fail}" -ne 0 ]]; then
        exit 1
    fi
    echo "vale selfcheck: the gate composition reaches all ${#md_probes[@]} targets (Markdown prose, docstrings, comments)."
    ;;
*)
    run_vale "$@"
    ;;
esac
