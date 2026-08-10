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
# An exported CDPATH makes every `cd <relative>` echo the resolved directory
# to stdout, which corrupts the command substitutions below (the resolved
# VALE path grows a stray line and the gate dies with exit 127).
unset CDPATH
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

# The exemption, as data (STYLE.md, "Scope"): whole directories of Chinese
# documents, historical files, and working notes, plus individual exempt
# files. The --glob below is COMPOSED from these arrays, and --selfcheck
# derives its per-directory expectations from the same arrays -- so the
# arrays are the single spec, and a glob that stops matching what they say
# fails the selfcheck instead of silently shrinking the corpus.
EXEMPT_DIRS=(docs/design-notes docs/research docs/work_doc docs/feat_plan
    docs/legacy work)
EXEMPT_FILES=(docs/spec/specification.md docs/misc.md CHANGELOG.md)

compose_exempt() {
    local parts=() entry
    for entry in "${EXEMPT_FILES[@]}"; do
        parts+=("${entry}")
    done
    for entry in "${EXEMPT_DIRS[@]}"; do
        parts+=("${entry}/*")
    done
    local IFS=,
    printf '!{%s}' "${parts[*]}"
}
EXEMPT="$(compose_exempt)"

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
    # Negative test: prove the exact gate composition (config, composed
    # EXEMPT glob, TARGETS) delivers EXACTLY the partition the arrays above
    # declare. It mirrors the real directory layout of every target into a
    # temporary directory -- same relative paths, same .vale.ini, same
    # styles -- plants a deliberate violation in every directory (exempt
    # ones included) and in every single-file target, then runs `run_vale`
    # there: a probe under a governed path MUST be flagged, a probe under an
    # exempt path MUST NOT be. Guards against a config edit, a glob-escaping
    # defect, or a Vale upgrade silently shrinking (or widening) coverage
    # while the gate stays green.
    #
    # The mirror exists because the working tree is not a safe scratch pad:
    # planting fixtures in docs/, src/, and tests/ (the previous design)
    # truncated any same-named file, raced with a concurrent run, and left
    # debris behind when the run was killed. It also covers the single-file
    # targets, which cannot hold a planted fixture at all: the --glob is
    # applied even to an explicitly passed file argument (verified against
    # the pinned Vale), so a glob defect can silently drop README.md while
    # both the gate and the existence check above stay green.
    #
    # Known, documented gaps (STYLE.md, "Enforcement"): a module docstring
    # that follows the SPDX header, Python string literals, and attribute
    # docstrings.
    tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' EXIT
    cp .vale.ini "${tmp}/"
    cp -R .vale "${tmp}/"

    is_exempt_path() {
        local path="$1" prefix
        for prefix in "${EXEMPT_DIRS[@]}"; do
            case "${path}" in
            "${prefix}"/*) return 0 ;;
            esac
        done
        return 1
    }

    plant_md() {
        mkdir -p "${tmp}/$(dirname "$1")"
        printf 'A probe sentence with a deliberate e.g. marker.\n' > "${tmp}/$1"
    }

    expect_hit=()
    expect_skip=()
    py_probes=()
    for target in "${TARGETS[@]}"; do
        if [[ -d "${target}" ]]; then
            # Every real subdirectory gets a probe, so an exemption that
            # swallows a subtree (say, all of docs/spec) is caught even
            # though the target's root probe is still reached. Caches and
            # hidden directories are not prose surfaces.
            while IFS= read -r dir; do
                probe="${dir}/vale-selfcheck-probe.md"
                plant_md "${probe}"
                if is_exempt_path "${probe}"; then
                    expect_skip+=("${probe}")
                else
                    expect_hit+=("${probe}")
                    case "${dir}" in
                    src | src/* | tests | tests/*)
                        # The .py surfaces also carry tier-2/tier-3 prose:
                        # probe docstring and comment extraction per
                        # directory.
                        py_probe="${dir}/vale_selfcheck_probe.py"
                        cat > "${tmp}/${py_probe}" <<'PYEOF'
def example() -> None:
    """Function docstring with a deliberate e.g. marker."""
    # A comment with a deliberate e.g. marker.
PYEOF
                        py_probes+=("${py_probe}")
                        ;;
                    esac
                fi
            done < <(find "${target}" \( -name '__pycache__' -o -name '.*' \) -prune -o -type d -print | LC_ALL=C sort)
        else
            plant_md "${target}"
            expect_hit+=("${target}")
        fi
    done
    # Exempt files under a directory target: prove the exclusion excludes
    # them (entries outside every target are never visited; planting them is
    # harmless and keeps the loop uniform).
    for entry in "${EXEMPT_FILES[@]}"; do
        case "${entry}" in
        *.md)
            plant_md "${entry}"
            expect_skip+=("${entry}")
            ;;
        esac
    done

    status=0
    out="$(cd "${tmp}" && run_vale --output=line)" || status=$?
    if [[ "${status}" -gt 1 ]]; then
        printf '%s\n' "${out}"
        echo "vale selfcheck: Vale itself failed (exit ${status}); fix the tool error above." >&2
        exit "${status}"
    fi
    fail=0
    for probe in "${expect_hit[@]}"; do
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
    for probe in ${expect_skip[@]+"${expect_skip[@]}"}; do
        if printf '%s\n' "${out}" | grep -qF "${probe}:"; then
            echo "selfcheck FAILED: exempt path '${probe}' was linted; the exemption no longer excludes it." >&2
            fail=1
        fi
    done
    if [[ "${fail}" -ne 0 ]]; then
        exit 1
    fi
    echo "vale selfcheck: ${#expect_hit[@]} governed probes reached, ${#expect_skip[@]} exempt probes excluded (Markdown prose, docstrings, comments)."
    ;;
*)
    run_vale "$@"
    ;;
esac
