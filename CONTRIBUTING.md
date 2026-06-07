# Contributing to Standard ASR

## Setup

We use [uv](https://docs.astral.sh/uv/) for dependency, environment, and build
management. After installing uv, initialise the project with all dependency
groups:

```sh
uv sync --all-groups --all-extras
```

To also pull in the cookbook example packages:

```sh
uv sync --all-packages --all-groups --all-extras
```

## Git hooks (prek)

We use [**prek**](https://github.com/j178/prek) — a fast, drop-in-compatible
reimplementation of `pre-commit` — to run lint, format, type-check, and a
GitHub Actions security audit before each commit (and the test suite before
each push). The config lives in [`.pre-commit-config.yaml`](.pre-commit-config.yaml).

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

> The lint/format/typecheck hooks are **local** hooks that call `uv run`, so they
> execute the exact tool versions pinned in `uv.lock` — identical to CI, with no
> drift between a pinned hook revision and the project's tools.

## Running the checks manually

```sh
uv run ruff format --check     # formatting (CI uses --check)
uv run ruff check              # lint (incl. NPY201 for numpy 1.x/2.x safety)
uv run pyright                 # strict type check
uv run pytest                  # tests (+ coverage via the default addopts)
uvx zizmor .github/workflows/  # GitHub Actions security audit
standard-asr doctor            # diagnose plugin numpy conflicts
```

If you run a type checker yourself, run it via **`uv run pyright`** from the repo
root so it uses the project venv and the `[tool.pyright]` scope (which excludes
vendored `reference/` and sample code). Running a bare `pyright`/IDE checker with
a different interpreter will report spurious unresolved-import errors.

## CI

GitHub Actions enforces the same gates on every PR: ruff format+lint+pyright
(`lint.yml`), the test suite across a **numpy 1.26 vs latest-2.x** matrix with
warnings-as-errors (`pytest.yml`), a numpy-nightly canary (`numpy-nightly.yml`),
and a zizmor audit (`zizmor.yml`). All actions are pinned to commit SHAs (kept
fresh by Dependabot) with least-privilege `permissions:`.

## Contribution licensing

By submitting a pull request to the Standard ASR project, you agree to license
your contribution under the project's Apache 2.0 License. You certify that you
have the right to submit this contribution and that it does not violate any
third-party rights. Your contribution will be attributed to you in the git
history, and you will become part of "The Standard ASR Authors".
