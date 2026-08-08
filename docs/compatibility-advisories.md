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

_None._ Every constraint we add is recorded here; when the list is empty, no
runtime or dev dependency is capped beyond its lower-bound-only default. (The
`uv-build` cap in `[build-system]` is out of scope by its own comment: it
governs the build tool, not a dependency of the package.)

## Resolved advisories

When a cap is removed, its entry moves here with the date and the commit/PR that
lifted it, so the history of why it existed is preserved.

### `starlette < 1.0` (dev/CI only) — resolved 2026-07-05, lifted by [PR #40](https://github.com/standard-voice/standard_asr/pull/40)

- **Scope (while active):** `[tool.uv] constraint-dependencies` in
  `pyproject.toml`. It pinned **only this repo's** lock/dev resolution; it never
  appeared in `[project.dependencies]` and never narrowed the downstream
  contract.
- **Why it existed:** Starlette 1.x's `TestClient` runs on the `httpx2`
  transport (`Support httpx2 in the test client`, Starlette 1.2.0), while our
  FastAPI server tests drove the app through the classic `httpx`-based
  `TestClient`. With Starlette 1.x unconstrained, the dev resolution pulled a
  `TestClient` that emitted the "Using `httpx` with `starlette.testclient`"
  deprecation. *(Correction, 2026-07-15: the active-period entry recorded the
  failure mode as `ModuleNotFoundError: httpx2`. That is wrong for
  Starlette ≥ 1.2, whose `TestClient` falls back to classic `httpx` with the
  deprecation warning above and raises only when neither client is installed;
  the correction is noted here instead of silently rewriting the history this
  section exists to preserve.)*
- **How it was resolved:** the server tests migrated to the `httpx2`-based
  `TestClient` (dev dependency `httpx>=0.28` → `httpx2>=2.0`, response
  annotations `httpx.Response` → `httpx2.Response`), the `fastapi` floor rose to
  `>=0.133` — the first FastAPI release to drop its own `starlette<1.0.0` cap
  (0.132 still pinned it) — and the `starlette<1.0` constraint plus the two
  now-obsolete `filterwarnings` ignores were removed. `starlette>=1.3.1` is now
  declared directly in the `[server]` extra as a fail-loud security floor. The
  lock re-resolved to Starlette 1.3.1, clearing all five open Dependabot
  advisories — GHSA-82w8-qh3p-5jfq (high, fixed in 1.3.1), GHSA-wqp7-x3pw-xc5r
  (high, 1.1.0), GHSA-x746-7m8f-x49c (medium, 1.1.0), GHSA-86qp-5c8j-p5mr
  (medium, 1.0.1), GHSA-jp82-jpqv-5vv3 (low, 1.3.0) — every one fixed only on
  the Starlette 1.x line (no 0.x backports); ≥ 1.3.1 is the lowest version
  clearing all five.
- **Reference:** Starlette 1.2.0 release notes (`Support httpx2 in the test
  client`, PR encode/starlette#3291) and the `httpx2` transport migration.
