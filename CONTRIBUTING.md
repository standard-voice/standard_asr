# Contributing to Standard ASR

## Setup

We use [uv](https://docs.astral.sh/uv/) for dependency, environment, and build management. After installing uv, initialize the project with all dependency groups:

```sh
uv sync --all-groups --all-extras
```

## Git hooks (prek)

We use [**prek**](https://github.com/j178/prek) — a fast, drop-in-compatible reimplementation of `pre-commit` — to run lint, format, type-check, and a GitHub Actions security audit before each commit (and the test suite before each push). The config lives in [`.pre-commit-config.yaml`](.pre-commit-config.yaml).

Install the tool and the git hooks once:

```sh
uv tool install prek      # or: pipx install prek
prek install              # installs the pre-commit AND pre-push hooks
```

Run everything manually at any time:

```sh
prek run --all-files
```

What runs **on commit** (fast):

| Hook | Tool | Notes |
|------|------|-------|
| ruff (lint, autofix) | `uv run ruff check --fix` | same ruff pinned in `uv.lock`/CI |
| ruff (format) | `uv run ruff format` | |
| **pyright (strict typecheck)** | `uv run pyright` | reads scope from `pyproject.toml` |
| zizmor | `zizmorcore/zizmor-pre-commit` | audits `.github/workflows/` |
| generic hygiene | `pre-commit/pre-commit-hooks` | trailing whitespace, EOF, YAML/TOML, etc. |

What runs **on push**: the full test suite (`uv run pytest`).

> The lint/format/typecheck hooks are **local** hooks that call `uv run`, so they execute the exact tool versions pinned in `uv.lock` — identical to CI, with no drift between a pinned hook revision and the project's tools.

## Running the checks manually

```sh
uv run ruff format --check     # formatting (CI uses --check)
uv run ruff check              # lint (incl. NPY201 for numpy 1.x/2.x safety)
uv run pyright                 # strict type check
uv run pytest                  # tests (+ coverage via the default addopts)
actionlint .github/workflows/*.yml  # GitHub Actions syntax/semantic lint
uvx zizmor .github/workflows/  # GitHub Actions security audit
uv run standard-asr doctor     # diagnose plugin numpy conflicts
```

If you run a type checker yourself, run it via **`uv run pyright`** from the repo root so it uses the project venv and the `[tool.pyright]` scope (which excludes vendored `reference/` and sample code). Running a bare `pyright`/IDE checker with a different interpreter reports spurious unresolved-import errors.

`actionlint` is not a Python dependency. CI downloads a pinned binary and verifies its checksum; for local use, install it with your system package manager or run the equivalent check in CI.

## Writing prose and docs

Prose is part of the contract. Follow the writing standard in [`STYLE.md`](STYLE.md) and the canonical terms in [`TERMINOLOGY.md`](TERMINOLOGY.md). `uv run ruff check` runs pydocstyle, so docstring structure in `src/` is checked with the rest of the lint (tests and docs sample code are exempt from the structure rules; their prose is still governed).

[Vale](https://vale.sh) lints the prose itself — Markdown plus the comments and docstrings in `src/` and `tests/` — against the vendored Google baseline and the repo's own terminology rules (`.vale.ini`, `.vale/styles/`). Run it through the wrapper, which owns the lint targets and exemptions:

```sh
scripts/vale.sh          # human view: pretty output, all levels (keep at zero)
scripts/vale.sh --gate   # what CI enforces: fails on any alert at any level
```

Like `actionlint`, Vale is not a Python dependency: CI downloads a pinned binary and verifies its checksum; for local use, install it with your system package manager (or set `VALE=/path/to/vale`).

### Adding documentation

The documentation site renders `docs/content/`, and that directory is the whole rule: every Markdown file under it is published, and nothing outside it is. Adding a page is adding a file. Give it a frontmatter `title`, link sibling pages with explicit relative paths (`./other-page.md`), and do not use raw HTML — the site build rejects it. Internal material (design notes, research, working notes) lives in `docs/internal/`, which the site never publishes.

To preview the site locally, sync the Python environment first (the API reference is generated from the source with Griffe), then run the site from `docs/site` with Node 22+ and pnpm — regular Python development never needs them:

```sh
uv sync --locked --all-extras --all-groups
cd docs/site
pnpm install
pnpm dev
```

CI builds the site on every pull request, checks its code with the Oxc toolchain (`pnpm lint` for oxlint, `pnpm format:check` for oxfmt), and link-checks the export with lychee, so a broken internal link or `#anchor` target fails the build. The site's own rules and commands live in [`docs/site/README.md`](docs/site/README.md).

## Dependency policy

Standard ASR is infrastructure others build on, so we manage dependencies around **two distinct contracts** — keep them straight and most decisions follow.

1. **`[project.dependencies]` in `pyproject.toml` — the downstream contract.** This is what every application and plugin that installs `standard-asr` must satisfy. It is intentionally permissive: each direct dependency declares a **meaningful, verified lower bound and no speculative upper cap**. A lower bound is a promise ("we use an API introduced here"); an upper cap is a promise we usually cannot keep ("no newer release can ever work") and it fragments the ecosystem by making us incompatible with everyone who moved on. Caps are allowed **only** for a known, real incompatibility, and each must link an upstream issue and a revisit date — see [`docs/compatibility-advisories.md`](docs/compatibility-advisories.md).

2. **`uv.lock` — the reproducible environment.** It is committed so every contributor and CI run gets byte-for-byte the same dev/test environment. It is **not** shipped in the wheel and does **not** affect downstream resolution — applications resolve against the ranges in (1), never our lock.

`[tool.uv] constraint-dependencies` is a third thing, often confused with (1): it constrains **only this repo's** lock/dev resolution and never narrows the downstream contract. We reserve it for dev-only pins that must not leak into `[project]`; the advisories doc records how such a pin is justified and later retired (see the resolved `starlette < 1.0` entry for a worked example).

### Adding or changing a dependency

- **New direct dependency:** add it to `[project.dependencies]` with a lower bound you have actually verified the code needs (not "whatever is latest"). No upper cap. Then run `uv lock` and commit the lock with the change.
- **New dev/test/lint/typing tool:** add it to the appropriate PEP 735 group in `[dependency-groups]`, not to `[project.dependencies]`.
- **Raising a lower bound:** only when the code genuinely starts relying on a newer feature, or a security floor forces it. The lower-bounds CI lane must stay green afterward.
- **Adding an upper cap:** don't, unless it is a real incompatibility. If it is, record it in `docs/compatibility-advisories.md` (issue link + revisit date) and prefer a `[tool.uv] constraint-dependencies` dev-only pin if the breakage is dev/test-only rather than a downstream problem.
- **Routine version bumps** are Dependabot's job — it rewrites `uv.lock`, not the contract. Don't hand-bump the lock just to chase latest.

### CI channels

Four channels keep both contracts honest. Only the first gates a PR:

| Channel | Where | Resolution | Gates merge? | Catches |
|---------|-------|-----------|--------------|---------|
| **PR CI** | `ci.yml` | committed `uv.lock` (`--locked`) | **Yes** (`checks-complete`) | regressions in the exact, reproducible env |
| **Lower bounds** | `ci.yml` (`lower-bounds` job) | `--resolution lowest-direct`, py3.10 | **Yes** (part of `checks-complete`) | a declared floor that is actually too low |
| **Dependabot** | `dependabot.yml` | newest in-range → new `uv.lock` PRs | via PR CI | staying current; security fixes |
| **Canary** | `canary.yml` (daily) | `uv lock --upgrade` (+ `--prerelease allow`) | No (opens an issue) | upstream breakage before/just-after it ships |

### When a dependency change breaks CI

- **Lower-bounds lane red:** a declared floor is too low for the code as written. Either lower the code's requirement, or raise the bound in `[project.dependencies]` to the version that actually works (and `uv lock`).
- **Canary red:** a newer (or pre-release) version broke us. Inspect the run's `lock-drift-*` artifact to see what moved, then either adapt our code (best), raise a lower bound if the old version is genuinely unsupportable, or — only for a real, tracked incompatibility — add a capped constraint with an entry in `docs/compatibility-advisories.md`. The canary's tracking issue is the place to record the decision.
- **Dependabot PR red:** treat it like any failing PR; the change is gated by the same PR CI as everything else.

## CI

GitHub Actions enforces the gates on every PR through a single workflow, [`ci.yml`](.github/workflows/ci.yml): lock-freshness (`uv lock --check`), ruff format + lint + pyright, the test suite across Python 3.10–3.14 on Linux plus macOS/Windows edges (all `--locked`), lower-bounds, the numpy-floor lane, package hygiene, actionlint, zizmor, and coverage. They roll up into one required aggregate check, **`checks-complete`** — the only status branch protection needs. A daily [`canary.yml`](.github/workflows/canary.yml) watches newest dependency resolutions outside the PR gate. All actions are pinned to commit SHAs (kept fresh by Dependabot) with least-privilege `permissions:`.

## Releasing

Maintainer release steps live in [`RELEASING.md`](RELEASING.md). The short version: release commits land on `main` with `checks-complete` green, manual workflow dispatch publishes only to TestPyPI, and production PyPI publishing is triggered only by a published GitHub Release whose tag matches `pyproject.toml`.

## Contribution licensing

By submitting a pull request to the Standard ASR project, you agree to license your contribution under the project's Apache 2.0 License. You certify that you have the right to submit this contribution and that it does not violate any third-party rights. Your contribution is attributed to you in the git history, and you become part of "The Standard ASR Authors".
