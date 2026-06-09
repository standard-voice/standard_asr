# Compatibility advisories

This file is the audit trail for every place we deliberately constrain a
dependency's **upper** range. Our default is the opposite — lower-bound-only
ranges with no caps (see the dependency policy in
[`CONTRIBUTING.md`](https://github.com/standard-voice/standard_asr/blob/main/CONTRIBUTING.md)) — because a speculative cap fragments
the ecosystem. So each cap here must justify itself with:

- **what** is constrained and **where** (the `[project]` contract, or a dev-only
  `[tool.uv] constraint-dependencies` pin),
- **why** — the concrete, observed incompatibility,
- an **upstream issue / reference**, and
- a **revisit date** by which we re-check whether the cap can be dropped.

An entry with no upstream link and no revisit date is a bug, not an advisory.

## Active advisories

### `starlette < 1.0` (dev/CI only)

- **Scope:** `[tool.uv] constraint-dependencies` in `pyproject.toml`. This pins
  **only this repo's** lock/dev resolution. It does **not** appear in
  `[project.dependencies]` and does **not** narrow the downstream contract —
  applications depending on `standard-asr` are unaffected.
- **Why:** Starlette 1.x's `TestClient` requires the `httpx2` package, while our
  FastAPI server tests drive the app through the classic `httpx`-based
  `TestClient`. With Starlette 1.x unconstrained, the dev resolution pulls a
  `TestClient` that fails to import (`ModuleNotFoundError: httpx2`).
- **Reference:** Starlette `TestClient` / httpx2 transport migration (Starlette
  1.0 release notes and `encode/starlette` `TestClient` changes).
- **Resolution path:** migrate the server tests to the `httpx2`-based transport,
  then drop this constraint so dev/CI track the latest Starlette.
- **Revisit by:** 2026-09.

## Resolved advisories

_None yet._ When a cap is removed, move its entry here with the date and the
commit/PR that lifted it, so the history of why it existed is preserved.
