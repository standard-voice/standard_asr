# Releasing Standard ASR

This is the maintainer runbook for publishing the **`standard-asr`** core
package. It covers one-time PyPI/TestPyPI setup, the recurring release flow, and
the safeguards built into `.github/workflows/release.yml`.

The release process follows the project mission: explicit contracts, secure
defaults, and a low-surprise developer experience. Releases are built with
Astral uv, published with PyPI Trusted Publishing, and guarded so a bad tag or a
non-green commit fails loudly before anything reaches PyPI.

## TL;DR

After the one-time setup exists, every release is:

1. Land the release commit on `main` with `checks-complete` green.
2. Bump `version` in `pyproject.toml`, update `CHANGELOG.md`, and merge
   `chore(release): vX.Y.Z`.
3. Dry-run the exact artifact path on TestPyPI: Actions -> **Release** -> **Run
   workflow** from `main`.
4. Verify the TestPyPI install.
5. Publish a GitHub Release tagged `vX.Y.Z` at the release commit. Approve the
   protected `pypi` environment.
6. Verify PyPI, attestations, and the GitHub Release.

There are no long-lived PyPI API tokens. Production PyPI publishing is only
triggered by a GitHub Release; manual dispatch publishes only to TestPyPI.

## Release Architecture

### What Publishes

This repo publishes exactly one distribution: **`standard-asr`**. Engine plugins
live in their own repositories (e.g. `std-faster-whisper`, `std-mlx-audio`),
each with its own PyPI project, trusted publisher, and release cadence. That
keeps engine dependencies and licenses isolated from the core package.

### Toolchain Choices

- **Build frontend:** `uv build --package standard-asr --no-sources --out-dir
  dist --clear`. `--no-sources` is intentional: it proves the package builds
  from standards-compliant publishable metadata instead of accidentally relying
  on workspace or local source overrides.
- **Build backend:** `uv_build`, capped to the current minor series in
  `pyproject.toml`. The cap is on the build tool, not a runtime dependency.
- **Publish action:** `pypa/gh-action-pypi-publish`, using PyPI Trusted
  Publishing and PEP 740 attestations. uv can publish packages, but the PyPA
  action is PyPI's recommended GitHub Actions path for tokenless publishing and
  attestation upload. uv remains the source of truth for building and smoke
  testing the artifacts.
- **Version source:** static `project.version` in `pyproject.toml`. The git tag
  mirrors it as `vX.Y.Z`; the workflow fails if they differ.

### Trust Boundaries

The workflow separates build and publish:

1. `build` has no OIDC publish permission. It checks the ref, builds artifacts,
   smoke-tests the wheel and sdist in isolated uv environments, and uploads a
   workflow artifact.
2. `publish-testpypi` / `publish-pypi` have `id-token: write`, but they only
   download the already-verified artifact and upload it to the configured index.

This keeps the high-privilege OIDC publish jobs small and auditable.

### Guards

The `build` job refuses to continue unless:

- For production releases, the GitHub Release tag matches `project.version`.
- Manual dry-runs are dispatched from the default branch.
- The release candidate commit is reachable from the default branch.
- The same commit already has a successful `checks-complete` check run.
- The wheel contains only package source files (`.py`, `.pyi`, `py.typed`).
- Both wheel and sdist import correctly outside the project workspace.

The build artifacts are published only to PyPI (the canonical channel for a pip
package) with PEP 740 attestations; the GitHub Release carries the notes and
GitHub's auto-generated source archives. The workflow never uploads wheels or
sdists to the Release, so it stays compatible with immutable releases.

## One-Time Setup

Do this once per index. These are external settings and cannot be committed to
the repo.

### 1. Prerequisites

- PyPI and TestPyPI accounts with 2FA enabled.
- Admin access to `standard-voice/standard_asr`.
- The workflow file committed at `.github/workflows/release.yml`.

Use pending publishers if the `standard-asr` project does not exist yet. A
pending publisher creates the project on first successful upload, but it does
not reserve the name before that first upload.

### 2. Configure PyPI Trusted Publisher

On <https://pypi.org/manage/account/publishing/> add a pending GitHub publisher:

| Field | Value |
| --- | --- |
| PyPI project name | `standard-asr` |
| Owner | `standard-voice` |
| Repository name | `standard_asr` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

If the project already exists, add the same publisher under the project's
settings instead of the account-level pending publisher page.

### 3. Configure TestPyPI Trusted Publisher

Repeat the same setup on <https://test.pypi.org/manage/account/publishing/>,
but set:

| Field | Value |
| --- | --- |
| Environment name | `testpypi` |

PyPI and TestPyPI are separate services. Configure both.

### 4. Create GitHub Environments

In GitHub: **Settings -> Environments**.

| Environment | Used by | Protection |
| --- | --- | --- |
| `pypi` | GitHub Release -> production PyPI | Required reviewers; enable "prevent self-review" when possible |
| `testpypi` | Manual dry-run -> TestPyPI | Optional reviewer gate; useful for rehearsals |

Do not add PyPI secrets. Trusted Publishing does not use them.

### 5. Protect `main`

Branch protection for `main` should require exactly the aggregate
`checks-complete` status. The release workflow uses that same status to prove a
release candidate is CI-green.

## How The Workflow Runs

### Manual Dry-Run To TestPyPI

`workflow_dispatch` runs from the default branch and publishes to TestPyPI only:

1. `build`
2. `publish-testpypi`

No production PyPI path exists for manual dispatch.

### Production Release To PyPI

`release: published` runs when a maintainer publishes a GitHub Release:

1. `build`
2. `publish-pypi`

The `pypi` environment approval happens immediately before upload to PyPI.

## Cutting A Release

### Step 1 - Pre-Flight

- `main` contains exactly what you want to ship.
- `checks-complete` is green on the release commit.
- Decide the version using the policy below.

### Step 2 - Bump Version And Changelog

1. Set `version = "X.Y.Z"` in `pyproject.toml`.
2. Move `CHANGELOG.md` `[Unreleased]` content into `[X.Y.Z] - YYYY-MM-DD`.
3. Add a fresh empty `[Unreleased]` section above it.
4. Update the compare links at the bottom of `CHANGELOG.md`.

Local sanity checks:

```bash
uv version --short
uv lock --check
uv build --package standard-asr --no-sources --out-dir dist --clear
```

The printed version must match the tag you plan to publish.

### Step 3 - Land The Release Commit

Open a PR or push a single release commit:

```bash
chore(release): vX.Y.Z
```

Wait for `checks-complete` to pass on `main`.

### Step 4 - Dry-Run To TestPyPI

The manual dispatch runs the same CI-green guard as a real release, so
`checks-complete` must already be green on the `main` commit (Step 3) -- the
build job fails fast otherwise.

1. GitHub -> **Actions** -> **Release** -> **Run workflow**.
2. Select the `main` branch and run it. There are no inputs.
3. Approve the `testpypi` environment if it is protected.
4. Watch the workflow upload to <https://test.pypi.org/project/standard-asr/>.

### Step 5 - Verify TestPyPI

TestPyPI does not mirror all dependencies, so let TestPyPI provide
`standard-asr` while real PyPI provides the dependencies. uv defaults to a
`first-index` strategy (a dependency-confusion safeguard): when a dependency such
as `numpy` also exists on TestPyPI at an incompatible version, uv will not fall
back to PyPI and resolution fails loudly. Pass `--index-strategy
unsafe-best-match` so uv considers every index -- safe here because both indexes
are trusted:

```bash
uv venv /tmp/standard-asr-testpypi
source /tmp/standard-asr-testpypi/bin/activate
uv pip install \
  --index-strategy unsafe-best-match \
  --index https://test.pypi.org/simple/ \
  --default-index https://pypi.org/simple/ \
  "standard-asr==X.Y.Z"

python -c "import importlib.metadata as m; print(m.version('standard-asr'))"
python -c "import standard_asr"
standard-asr --help
standard-asr list
```

For plain pip:

```bash
python -m venv /tmp/standard-asr-testpypi-pip
source /tmp/standard-asr-testpypi-pip/bin/activate
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "standard-asr==X.Y.Z"
```

Confirm the TestPyPI page renders the README, classifiers, project URLs, and
metadata correctly.

### Step 6 - Publish To PyPI

1. GitHub -> **Releases** -> **Draft a new release**.
2. Choose/create tag `vX.Y.Z` at the release commit on `main`.
3. Title: `vX.Y.Z`.
4. Notes: paste the matching `CHANGELOG.md` section.
5. For alpha/beta/rc releases, mark the GitHub Release as a pre-release.
6. Publish the GitHub Release.
7. Approve the `pypi` environment when the workflow pauses.

The workflow then uploads the verified sdist/wheel to PyPI with PEP 740
attestations. Artifacts are not attached to the GitHub Release -- PyPI is the
canonical distribution channel for the package.

### Step 7 - Verify PyPI

```bash
uv venv /tmp/standard-asr-pypi
source /tmp/standard-asr-pypi/bin/activate
uv pip install "standard-asr==X.Y.Z"
standard-asr --help
```

Then verify:

- <https://pypi.org/project/standard-asr/> shows `X.Y.Z`.
- PyPI shows verified provenance / attestations.
- The built wheel and sdist are on PyPI (they are not attached to the GitHub Release).
- The release notes match `CHANGELOG.md`.

## Versioning Policy

`pyproject.toml` is the single source of truth. Tags mirror it as `vX.Y.Z`.

- **Pre-1.0:** Minor releases may contain breaking changes; patch releases are
  backward-compatible fixes. Call out breaking changes clearly in the changelog.
- **Post-1.0:** Standard SemVer.
- **Pre-releases:** Use PEP 440-compatible versions such as `0.2.0a1`,
  `0.2.0b1`, or `0.2.0rc1`, tagged as `v0.2.0a1`, `v0.2.0b1`, or
  `v0.2.0rc1`.
- **Yanking:** If a release is broken, yank it on PyPI and ship a new patch.
  Never reuse a version number.

## First Release (`v0.1.0`)

The changelog already contains a drafted `0.1.0` section, but there is no
published GitHub Release or PyPI upload yet. For the first release:

1. Confirm `pyproject.toml` still has `version = "0.1.0"`.
2. Confirm `CHANGELOG.md` accurately describes the release.
3. Run the TestPyPI dry-run.
4. Publish GitHub Release `v0.1.0` and approve `pypi`.

## Troubleshooting

| Symptom | Cause / Fix |
| --- | --- |
| Tag/version guard fails | The GitHub Release tag does not match `project.version`. Fix `pyproject.toml` or recreate the tag. |
| Commit ancestry guard fails | The tag points to a commit not reachable from `main`. Re-tag a commit from mainline history. |
| CI-green guard fails | `checks-complete` did not pass on that SHA yet. Wait for CI or fix the failing checks before releasing. |
| Manual dry-run fails on ref | Run the workflow from `main`; manual dispatch is TestPyPI-only and default-branch-only. |
| Trusted publisher 403 | PyPI/TestPyPI publisher fields do not exactly match owner, repo, workflow filename, and environment. |
| Job waits before publish | The GitHub Environment requires approval. Review and approve the deployment. |
| Upload says file already exists | PyPI/TestPyPI versions are immutable. Bump to a new version or pre-release. |
| TestPyPI install cannot resolve a dependency | uv's default first-index strategy will not fall back to PyPI for a package TestPyPI also hosts (e.g. `numpy`). Add `--index-strategy unsafe-best-match` (both indexes are trusted). |
| No PyPI attestations | Confirm the upload used Trusted Publishing with `id-token: write` and the PyPA publish action. |

## Future Plugin Releases

When adding a new engine plugin, do not add it to this release workflow. Create
a separate plugin repository and copy this pattern:

- Independent PyPI project and trusted publishers.
- Independent `release.yml`, changelog, and SemVer line.
- Dependency on `standard-asr>=...` from PyPI.
- Plugin-specific compliance checks before publish.

This preserves Standard ASR's core/plugin separation while keeping every
compliant engine installable by applications without extra integration work.

## References

- uv packaging guide: <https://docs.astral.sh/uv/guides/package/>
- uv build backend: <https://docs.astral.sh/uv/concepts/build-backend/>
- uv GitHub Actions guide: <https://docs.astral.sh/uv/guides/integration/github/>
- Python Packaging User Guide, GitHub Actions publishing:
  <https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>
- PyPI Trusted Publishing: <https://docs.pypi.org/trusted-publishers/>
- PyPA publish action: <https://github.com/pypa/gh-action-pypi-publish>
- PEP 740 attestations: <https://peps.python.org/pep-0740/>
