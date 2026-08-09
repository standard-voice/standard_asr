#!/usr/bin/env bash
# Run Vale on every governed prose surface (STYLE.md, "Scope").
#
# This script is the single source of truth for WHAT gets linted; .vale.ini
# holds the rules. The --glob exemption exists because Vale cannot exempt
# whole files from inside .vale.ini (an empty "BasedOnStyles =" section does
# not clear styles). Chinese documents, historical files, and working notes
# are exempt per STYLE.md.
#
# Usage:
#   scripts/vale.sh                # the backlog view: warnings + suggestions
#   scripts/vale.sh --gate         # the CI gate: errors only, exit non-zero
#   VALE=/path/to/vale scripts/vale.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VALE="${VALE:-vale}"
ARGS=()
if [[ "${1:-}" == "--gate" ]]; then
    shift
    ARGS+=(--minAlertLevel=error)
fi

exec "${VALE}" "${ARGS[@]}" \
    --glob='!{docs/spec/specification.md,docs/design-notes/*,docs/research/*,docs/work_doc/*,docs/feat_plan/*,docs/legacy/*,docs/misc.md,work/*,CHANGELOG.md}' \
    "$@" \
    README.md CONTRIBUTING.md AGENTS.md STYLE.md TERMINOLOGY.md RELEASING.md \
    docs src tests
