# Copilot instructions

The contributor rules for this repository live in
[`AGENTS.md`](../AGENTS.md). Read it first and follow it in full: it defines
the project's purpose, the stakeholders every change must serve, the coding
rules, and the prose standard.

Before you edit a docstring, a user-facing string, or a Markdown file, also
read [`STYLE.md`](../STYLE.md) and [`TERMINOLOGY.md`](../TERMINOLOGY.md). A
prose gate (`scripts/vale.sh`) enforces the mechanizable part of both in CI,
and it fails on any alert at any level.

This file exists only as a pointer. GitHub.com Copilot Chat does not read
`AGENTS.md` (the Copilot coding agent does), so without it that surface would
get no repository instructions at all. Do not copy rules here: duplicated
rules drift, and `AGENTS.md` is the single source of truth.
