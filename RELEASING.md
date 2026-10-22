# Releasing Standard ASR

This is the maintainer runbook for publishing the **`standard-asr`** core package with the Release workflow, [`.github/workflows/release.yml`](.github/workflows/release.yml). It covers the one-time setup, the three steps of a release, and what to do when a run fails.

## Summary

After the one-time setup exists, a release takes three actions from a maintainer:

1. **Merge the release pull request**, `chore(release): vX.Y.Z`. It sets the version, adds the `CHANGELOG.md` section, and updates `uv.lock`.
2. **Run the Release workflow from `main`.** Its jobs run in this order, each after the one before it succeeded:
   1. `prepare` runs the guards, builds the release commit, smoke-tests the wheel and the sdist, and checks what both indexes already hold.
   2. `publish-testpypi` uploads to TestPyPI, and `verify-testpypi` checks that TestPyPI holds this build, with attestations, and that it installs.
   3. `publish-pypi` waits for a maintainer to approve the `pypi` environment, then uploads to PyPI.
   4. `verify-pypi` runs the same checks against PyPI.
   5. `github-release` creates the tag `vX.Y.Z` and the GitHub Release, with the CHANGELOG section as the notes.
   6. `dev-bump` opens a pull request that moves `main` to the next development version.
3. **Approve the CI of the dev-bump pull request, then merge it.** GitHub holds the CI runs of a pull request that the workflow token opened until a person approves them: select **Approve workflows to run** in its merge box, or close and reopen it. Merge it after `checks-complete` passes.

There are no long-lived PyPI API tokens: the publish jobs authenticate with Trusted Publishing. A run that fails can run again: a second run of the same version resumes the first, with the same bytes (see "When a run fails").

## Release architecture

### What publishes

This repo publishes exactly one distribution: **`standard-asr`**. Engine plugins live in their own repositories (for example, `std-faster-whisper`, `std-mlx-audio`), each with its own PyPI project, trusted publisher, and release cadence. That keeps engine dependencies and licenses isolated from the core package.

### Toolchain choices

- **Build frontend:** `uv build --package standard-asr --no-sources --out-dir dist --clear`, run on a detached checkout of the release commit. `--no-sources` is intentional: it proves the package builds from standards-compliant publishable metadata instead of accidentally relying on workspace or local source overrides.
- **Build uv:** the uv whose version is the lower bound of the `uv-build` requirement in `[build-system].requires` at the release commit. uv builds with its own bundled copy of `uv_build` when its version satisfies that requirement ([uv docs](https://docs.astral.sh/uv/concepts/build-backend/#bundled-build-backend)). The lower bound always does, so the release commit fixes the whole recipe, and a second run builds the same bytes. `UV_VERSION` in `release.yml` names the uv for the workflow's other commands: reading the version, the install checks, and the dev-bump commit.
- **Build backend:** `uv_build`, capped to the current minor series in `pyproject.toml`. The cap is on the build tool, not a runtime dependency.
- **Publish action:** `pypa/gh-action-pypi-publish`, using PyPI Trusted Publishing and PEP 740 attestations. uv can publish packages, but the PyPA action is PyPI's recommended GitHub Actions path for tokenless publishing and attestation upload. uv remains the source of truth for building and smoke testing the distribution files.
- **Version source:** static `project.version` in `pyproject.toml`. The workflow creates the git tag `vX.Y.Z` from it, at the release commit.

### Trust boundaries

The workflow runs seven jobs, and each holds only the permissions it needs:

| Job | Permissions | What it does |
| --- | --- | --- |
| `prepare` | `contents: read`, `checks: read` | Runs the guards, builds, smoke-tests, and uploads the workflow artifact `standard-asr-release`: the wheel, the sdist, and the release notes. |
| `publish-testpypi` | `id-token: write`, `contents: read`; environment `testpypi` | Downloads the workflow artifact and uploads the wheel and the sdist to TestPyPI. |
| `verify-testpypi` | `contents: read` | Checks the digests and the attestations on TestPyPI, and installs from it. |
| `publish-pypi` | `id-token: write`, `contents: read`; environment `pypi` | Downloads the workflow artifact and uploads the wheel and the sdist to PyPI, after approval. |
| `verify-pypi` | `contents: read` | Runs the same checks against PyPI. |
| `github-release` | `contents: write` | Creates the tag and the GitHub Release, or confirms the ones that exist. |
| `dev-bump` | `contents: write`, `pull-requests: write` | Opens the dev-bump pull request. |

The two publish jobs are the only ones that can request an OIDC token, and they only download the files `prepare` checked and upload them. This keeps the high-privilege jobs small and auditable.

### Guards

The `prepare` job stops the run, before anything is public, unless each check passes. The checks run in this order:

1. **Default branch.** The run was started from `main`.
2. **Normalized version.** `pyproject.toml` writes the version in the normalized form that uv prints, because the release commit search compares the written string.
3. **Releasable version.** `scripts/release_version.py classify` refuses a development version and a version outside the versioning policy below, and decides whether the release is a pre-release.
4. **Release commit.** `scripts/release_commit.py` finds the newest commit on the first-parent history of `main` that set the version, and a successful `checks-complete` check run must exist on it.
5. **CHANGELOG section.** `CHANGELOG.md` at the release commit has a non-empty `## [X.Y.Z]` section, which becomes the release notes.
6. **Tag and release.** A tag `vX.Y.Z` or a GitHub Release that already exists matches this run (`scripts/release_github.py check`).
7. **Wheel hygiene.** The wheel contains only package source files (`.py`, `.pyi`, `py.typed`).
8. **Smoke tests.** The wheel and the sdist import in isolated environments and report the version, and `standard-asr --help` runs from the wheel.
9. **Digest guard.** Each index lacks the version or holds only files with this build's sha256 (`scripts/release_index.py guard`).

The distribution files are published only to PyPI (the canonical channel for a pip package) with PEP 740 attestations; the GitHub Release carries the notes and GitHub's auto-generated source archives. The workflow never uploads wheels or sdists to the Release, so it stays compatible with immutable releases.

The [design case](docs/internal/feat_plan/release-pipeline-redesign.md) explains why the pipeline works this way.

## One-time setup

Do this once per index. These are external settings and cannot be committed to the repo.

### 1. Prerequisites

- PyPI and TestPyPI accounts with 2FA enabled.
- Administrator access to `standard-voice/standard_asr`.
- The workflow file committed at `.github/workflows/release.yml`.

Use pending publishers if the `standard-asr` project does not exist yet. A pending publisher creates the project on first successful upload, but it does not reserve the name before that first upload.

### 2. Configure the PyPI trusted publisher

On <https://pypi.org/manage/account/publishing/> add a pending GitHub publisher:

| Field | Value |
| --- | --- |
| PyPI project name | `standard-asr` |
| Owner | `standard-voice` |
| Repository name | `standard_asr` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

If the project already exists, add the same publisher under the project's settings instead of the account-level pending publisher page.

### 3. Configure the TestPyPI trusted publisher

Repeat the same setup on <https://test.pypi.org/manage/account/publishing/>, but set:

| Field | Value |
| --- | --- |
| Environment name | `testpypi` |

PyPI and TestPyPI are separate services. Configure both.

### 4. Create the GitHub environments

In GitHub: **Settings -> Environments**.

| Environment | Used by | Protection |
| --- | --- | --- |
| `pypi` | `publish-pypi` -> PyPI | A required reviewer for the first one or two releases on this pipeline (design case, section 2.4). Turn on "Prevent self-review" once a second maintainer can approve. |
| `testpypi` | `publish-testpypi` -> TestPyPI | None: the rehearsal runs in every release without an approval. |

Do not add PyPI secrets. Trusted Publishing does not use them.

### 5. Protect `main`

Branch protection for `main` should require exactly the aggregate `checks-complete` status. The release workflow requires the same status on the release commit.

### 6. Checks for the first run on this pipeline

Before the first run:

- Check that the `release-tags` ruleset lets the GitHub Actions token create tags. The `github-release` job creates `vX.Y.Z` through `gh release create` after PyPI is live, so a ruleset that blocks it stops the run there. Give the token a bypass if the ruleset needs one (design case, sections 5 and 7).
- Check that both trusted publishers name the workflow `release.yml` and their environment, `pypi` or `testpypi`.

When the first dev-bump pull request appears:

- Check that its commit shows GitHub's "Verified" badge. The job makes the commit through the GraphQL mutation `createCommitOnBranch`, which GitHub signs "if supported" ([GraphQL reference](https://docs.github.com/en/graphql/reference/commits#mutation-createcommitonbranch)). A ruleset that requires signed commits would block the merge of an unsigned one.

## Cutting a release

### Step 1 — the release pull request

1. Decide the version using the policy below.
2. Set `version = "X.Y.Z"` in `pyproject.toml`.
3. Move the `[Unreleased]` content of `CHANGELOG.md` into a new `## [X.Y.Z] - YYYY-MM-DD` section, and add a fresh empty `## [Unreleased]` section above it. The workflow takes the release notes from this section.
4. Update the compare links at the bottom of `CHANGELOG.md`.
5. Run `uv lock`, because `uv.lock` records the project's version.

Local sanity checks:

```bash
uv version --short
uv lock --check
uv build --package standard-asr --no-sources --out-dir dist --clear
```

The printed version must be `X.Y.Z`.

Title the pull request `chore(release): vX.Y.Z`, and merge it after `checks-complete` passes, as one commit on `main`: squash it, or use a merge commit. The workflow releases the commit on `main` that set the version, and CI runs once per push, on its last commit. If the pull request's commits land on `main` one by one, an earlier commit can become the release commit, one without a `checks-complete` run or without the CHANGELOG section.

### Step 2 — run the workflow

1. GitHub -> **Actions** -> **Release** -> **Run workflow**. Select `main`; there are no inputs.
2. Read the log of the `prepare` job. It prints the release commit and the commits on `main` after it, which are not part of this release. If the release commit is not the one you expect, cancel the run: nothing is public yet.
3. When the run pauses before `publish-pypi`, look at the rehearsal on <https://test.pypi.org/project/standard-asr/>: the README, the classifiers, the project URLs, and the metadata should render correctly. Then approve the `pypi` environment.
4. The run uploads to PyPI, checks PyPI, creates the GitHub Release `vX.Y.Z` at the release commit, and opens the dev-bump pull request.

The workflow checks each index after its upload, so these checks need no manual step:

- **Files.** The index holds every built file with the same sha256, and each file has a PEP 740 attestation on the index's integrity API.
- **TestPyPI install.** `standard-asr==X.Y.Z` comes from TestPyPI without its dependencies, and the dependencies, read from the installed package's metadata, come from PyPI. TestPyPI also hosts packages named `numpy` and `pydantic`, so taking only `standard-asr` from it keeps them out of the check. Then the job runs `uv pip check`, imports the package, checks its version, and runs `standard-asr --help`.
- **PyPI install.** `standard-asr==X.Y.Z` installs from PyPI in an isolated environment, imports, reports the version, and runs `standard-asr --help`.

To check by hand after the release:

```bash
uv run --isolated --no-project --refresh-package standard-asr --with "standard-asr==X.Y.Z" standard-asr --help
```

`--refresh-package` makes uv revalidate its cached copy of the index page, which it may otherwise reuse for several minutes. The install checks in the workflow pass it on every attempt too.

### Step 3 — the dev-bump pull request

The `dev-bump` job opens `chore: bump version to <next>` from the branch `release/dev-bump-vX.Y.Z`. It sets the next development version, `X.Y.(Z+1).dev0` after a final release or `X.Y.Zrc(N+1).dev0` after `X.Y.ZrcN`, and updates `uv.lock`.

The workflow token opened the pull request, so GitHub holds its CI runs until a person with write access approves them ([GitHub docs](https://docs.github.com/en/actions/concepts/security/github_token#when-github_token-triggers-workflow-runs)). Select **Approve workflows to run** in the merge box, or close and reopen the pull request; either way, CI starts. Merge it after `checks-complete` passes.

Until it merges, `main` still declares `X.Y.Z`, and a second run of the workflow resumes the release. After it merges, the first guard stops every run until the next release pull request merges.

### Pre-releases

A version such as `X.Y.ZaN`, `X.Y.ZbN`, or `X.Y.ZrcN` rides the same pipeline. The GitHub Release is marked as a pre-release, and the dev-bump pull request sets the next pre-release number as a development version, such as `X.Y.Zrc(N+1).dev0`, so the next release pull request can choose the next pre-release or the final version. Installers skip a pre-release unless asked: pip needs `--pre` or an exact pin. A pre-release on PyPI is still public; see "Protocol generations".

## When a run fails

Read the error in the job that failed. A second run of the same version resumes the first: it finds the same release commit, builds the same bytes, skips the files the indexes already hold, and finishes what is left. To start it, run the workflow from `main` again. It waits for the `pypi` approval again, even when PyPI already holds every file.

| Step | What the log says | What is public | What to do |
| --- | --- | --- | --- |
| `prepare` | "Run the release workflow from 'main'." | Nothing new | Start the run from `main`. |
| `prepare` | "pyproject.toml writes the version as" one form, "which uv normalizes to" another. | Nothing new | Write the normalized form that the message names (`0.2.0rc1`, not `0.2.0-rc1`) in `pyproject.toml`, in a new release pull request. |
| `prepare` | The version is "a development version". | Nothing new | No release pull request has merged since the last release, which is the normal state between releases. Merge the release pull request, then run the workflow. |
| `prepare` | The version is "outside the versioning policy", or "not a PEP 440 version". | Nothing new | Set a version the policy allows in a new release pull request. |
| `prepare` | "No commit on the first-parent history" sets the version. | Nothing new | The normalized-version check runs first, so this should not happen. If it does, read the first-parent history of `pyproject.toml` on `main` (`git log --first-parent -p -- pyproject.toml`) to see which commit set the version, and how. |
| `prepare` | "The checkout is shallow". | Nothing new | The `prepare` checkout in `release.yml` must keep `fetch-depth: 0`. Restore it, then run again. |
| `prepare` | "No successful 'checks-complete' check run". | Nothing new | Wait for CI on the release commit, then run again. A push to `main` cancels the CI run still in progress for the commit before it, so if the run on the release commit was canceled, re-run it from the **Actions** tab first. |
| `prepare` | CHANGELOG.md "has no section". | Nothing new | The notes come from the release commit, so the section needs a new release commit: see "A fix after the release commit". |
| `prepare` | The tag "points at" another commit, the release "exists but its tag does not", or it has "notes that differ" or the wrong pre-release flag. | The tag or the release | Find out how the tag or the release got there before you change anything. Make it match this run (the tag at the release commit, the notes equal to the CHANGELOG section, the pre-release flag that fits the version), then run again. |
| `prepare` | The wheel "ships unexpected non-source files", or a smoke test fails. | Nothing new | Fix the source: see "A fix after the release commit". |
| `prepare` | An index "already holds" the version "with other content". | Files of `X.Y.Z` on that index | The index keeps filenames forever, and this build cannot match its files. Release a new version. |
| `publish-testpypi`, `publish-pypi` | HTTP 403 from the trusted publisher. | Nothing new on that index | The publisher fields on the index must match the owner, the repository, the workflow filename, and the environment exactly. Fix them, then run again. |
| `publish-pypi` | The job waits. | The rehearsal on TestPyPI | The `pypi` environment needs approval. Review and approve the deployment. |
| `publish-testpypi`, `publish-pypi` | The upload failed partway. | Some files of `X.Y.Z` on that index | Run again. The digest guard confirms that the files the index holds are this build, and the publish uploads the rest. |
| `verify-testpypi`, `verify-pypi` | The index "still lacks" a file, or the install failed. | `X.Y.Z` on that index | The job retries for several minutes before it fails, so an index or a dependency download lagged or failed for longer than that. Run again. If the package itself fails to install or import, its bytes are wrong. On TestPyPI, `X.Y.Z` is then spent, and the fix ships as a new version; on PyPI, also yank `X.Y.Z`. |
| `verify-testpypi`, `verify-pypi` | The index holds a file "with other content", or "has no attestation" for one. | `X.Y.Z` on that index | The index keeps those filenames, and a file uploaded without an attestation never gets one. Release a new version. |
| `github-release` | "a draft release" for the tag exists. | `X.Y.Z` on both indexes | `prepare` cannot see drafts, because GitHub lists them only to a token with write access. Publish the draft or delete it, then run again. |
| `github-release` | A tag or release check fails. | `X.Y.Z` on both indexes | As for the same check in `prepare`. |
| `github-release` | `gh release create` failed. | `X.Y.Z` on both indexes; the release may exist | Look at the release first (`gh release view vX.Y.Z`), because a timeout can leave it created. If the `release-tags` ruleset refused the tag, let the GitHub Actions token create tags. Then run again. |
| `dev-bump` | The pull request "was closed without merging". | The release is complete | Reopen the pull request, which starts its CI, and merge it after `checks-complete` passes. The job does not open a second one. |
| `dev-bump` | The pull request "was merged, yet the branch still declares" the version. | The release is complete | Something set the version back after the merge. Look at the history of `pyproject.toml` on `main`. |

### After the dev-bump pull request merges

A second run of `X.Y.Z` resumes only while `main` declares `X.Y.Z`. Once the dev-bump pull request merges, the first guard stops every run of it. If the GitHub Release is still missing then, create it by hand at the release commit, with the command the workflow uses:

```bash
gh release create vX.Y.Z --repo standard-voice/standard_asr --target <release commit> --title vX.Y.Z --notes-file release-notes.md
```

Add `--prerelease` for a pre-release. The `prepare` log of the earlier run prints the release commit, and that run's workflow artifact `standard-asr-release` holds `release-notes.md` for 90 days.

### A fix after the release commit

The release commit is the commit that set the version, so a fix merged after it is not part of the release. What to do depends on whether any file of `X.Y.Z` reached an index:

- **A file reached TestPyPI or PyPI.** `X.Y.Z` is spent, because both indexes keep filenames forever. Merge the fix and release a new version.
- **No file reached either index.** The number is still free, but the fix needs a new release commit. Merge the fix in a pull request that also sets `version` back to the development version `main` declared before the release pull request, and runs `uv lock`. Then merge the release pull request for `X.Y.Z` again: the newest commit that sets the version becomes the release commit. Releasing the next version instead also works.

## Versioning policy

`pyproject.toml` is the single source of truth. Tags mirror it as `vX.Y.Z`.

- **Pre-1.0:** Minor releases may contain breaking changes; patch releases are backward-compatible fixes. Call out breaking changes clearly in the changelog.
- **Post-1.0:** Standard SemVer.
- **Pre-releases:** Use PEP 440-compatible versions such as `0.2.0a1`, `0.2.0b1`, or `0.2.0rc1`, tagged as `v0.2.0a1`, `v0.2.0b1`, or `v0.2.0rc1`.
- **Yanking:** If a release is broken, yank it on PyPI and ship a new patch. Never reuse a version number. A number counts as used once any of its files reached PyPI or TestPyPI: TestPyPI keeps filenames as PyPI does, so a version rehearsed with wrong bytes is never reused either.

### Protocol generations

The package version above is not the protocol version: `properties.protocol_version` names the engine-protocol contract generation, an independent version line governed by spec AR.1 (`docs/content/specification/protocol.md`). These obligations land at release time:

- The first public core release carrying a new `0.MINOR` protocol generation **freezes that generation** (AR.1's generation freeze point). Public includes pre-releases: an alpha, beta, or RC artifact on PyPI freezes the generation it carries exactly as a stable release does (the TestPyPI rehearsal is not a public release). Before publishing, confirm `CURRENT_PROTOCOL_VERSION` in `src/standard_asr/contract/protocol_version.py` names the contract you intend to freeze; after the release, a breaking change to that contract requires a new minor.
- The first generation-freezing release also **establishes the protocol ledger** -- the generation-ledger appendix of `docs/content/specification/protocol.md` -- recording the frozen token, freeze date, spec snapshot commit, and compatibility classification; every later freeze appends its row. The package changelog is never the source of truth for protocol history.
- The first stable release (protocol `1.0.0`) must execute AR.1's freeze plan and record its three freeze-day decisions: the frozen `0.N` alias policy, the multi-major support policy, and the `1.0.0` promotion row in the ledger. The RC window has no mutable period: a `1.0.0rcN` package declares, and freezes, the newest `0.N` generation; a breaking change during the RC series opens a new `0.MINOR`; the final release promotes the last frozen generation verbatim. Rehearse that promotion on a branch before you run the release workflow: flip the constants and the feature floors, point the first-party plugins at `1.0.0`, and run their compliance suites against that core. The stable token cannot appear in a public artifact before the release that freezes it, so the rehearsal is the only full-system test the promotion gets. The same release revises the `/v1` projection declaration in `docs/content/specification/server-api.md`: `/v1` adds the `1.0.x` line when the wire representation is unchanged (the expected outcome), or the core opens a new wire revision. Without that step, the statement that `/v1` projects exactly `0.2` becomes false the moment engines declare `1.0.0`.
- Before shipping protocol `1.1` (the first stable additive minor), decide how an older core's `/v1` projects an engine on a newer minor -- project only the understood intersection, filter unknown requestable capabilities, or require a core floor -- and revise the projection declaration in `docs/content/specification/server-api.md` in the same release.

## Future plugin releases

When adding a new engine plugin, do not add it to this release workflow. Create a separate plugin repository and copy this pattern:

- Independent PyPI project and trusted publishers.
- Independent `release.yml`, changelog, and SemVer line.
- Dependency on `standard-asr>=...` from PyPI.
- Plugin-specific compliance checks before publish.

This preserves Standard ASR's core/plugin separation while keeping every compliant engine installable by applications without extra integration work.

## References

- uv packaging guide: <https://docs.astral.sh/uv/guides/package/>
- uv build backend: <https://docs.astral.sh/uv/concepts/build-backend/>
- uv GitHub Actions guide: <https://docs.astral.sh/uv/guides/integration/github/>
- Python Packaging User Guide, GitHub Actions publishing: <https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>
- PyPI Trusted Publishing: <https://docs.pypi.org/trusted-publishers/>
- PyPA publish action: <https://github.com/pypa/gh-action-pypi-publish>
- PEP 740 attestations: <https://peps.python.org/pep-0740/>
- PyPI JSON API, which gives the digests: <https://docs.pypi.org/api/json/>
- PyPI integrity API, which gives the attestations: <https://docs.pypi.org/api/integrity/>
- GitHub, when the workflow token triggers workflow runs: <https://docs.github.com/en/actions/concepts/security/github_token#when-github_token-triggers-workflow-runs>
- GitHub, listing releases, drafts included: <https://docs.github.com/en/rest/releases/releases#list-releases>
- GitHub, the `createCommitOnBranch` mutation and its commit signing: <https://docs.github.com/en/graphql/reference/commits#mutation-createcommitonbranch>
