# Releasing Standard ASR

This is the maintainer runbook for publishing the **`standard-asr`** core package
to PyPI. It covers first-time setup (trusted publisher + GitHub Environments),
testing a release on TestPyPI, publishing to production PyPI, and the recurring
flow for `v0.2.0` and every future release.

The process follows 2026 packaging best practice and the project's own
principles — **security by default** (no long-lived secrets), **standard-library
rigor** (reproducible, explicit, fail-loud), and **core/plugin separation**
(goal G.4.1). Standard ASR is infrastructure others will depend on for years, so
every release must be reproducible and verifiable.

---

## TL;DR — cutting a release (after one-time setup)

Once the trusted publishers and Environments exist (do that **once**, see
[One-time setup](#one-time-setup)), every release is:

1. Make sure `main` is green in CI.
2. Bump `version` in `pyproject.toml`; move the `CHANGELOG.md` `[Unreleased]`
   items into a dated `[X.Y.Z]` section. Commit (`chore(release): vX.Y.Z`) to `main`.
3. **Dry run:** Actions → **Release** → *Run workflow* → `target = testpypi`.
   Approve the `testpypi` environment. Verify the install from TestPyPI.
4. **Publish:** create a **GitHub Release** with tag `vX.Y.Z` (must equal the
   `pyproject` version), notes from the changelog. Approve the `pypi` environment.
5. Verify `pip install "standard-asr==X.Y.Z"` and that the PyPI page shows
   **verified attestations**.

There are **no API tokens** anywhere — publishing authenticates via OIDC Trusted
Publishing, gated by a human approval on a protected GitHub Environment.

---

## What we publish, and why

- **Core only.** This repo publishes exactly one distribution: `standard-asr`
  (`uv build --package standard-asr`). The cookbook packages (`std-dummy-asr`,
  `std-faster-whisper`) are in-repo examples and are **not** published from here.
  Engines are independent, separately-versioned plugin packages (goal G.4.1 —
  core/implementation separation); when an engine graduates to a real plugin it
  gets its own repo + its own release flow (see
  [Future: publishing engine plugins](#future-publishing-engine-plugins)).
- **OIDC Trusted Publishing, no tokens.** `release.yml` authenticates to PyPI
  with a short-lived OpenID Connect token minted per run. There is no PyPI API
  token to store, leak, or rotate — the supply-chain attack surface a long-lived
  secret creates simply does not exist (security by default).
- **PEP 740 attestations.** Every uploaded file carries a signed digital
  attestation of its provenance (which repo, workflow, and commit built it),
  generated automatically under Trusted Publishing. PyPI displays these as
  *verified* provenance so downstream consumers can trust where a release came
  from.
- **Human-gated.** The publish job runs inside a protected GitHub **Environment**
  that requires a manual approval — nothing reaches PyPI without a person
  clicking "approve".
- **Reproducible & guarded.** Builds pin the `uv` version and resolve against the
  committed `uv.lock`; the workflow refuses to publish if the release **tag does
  not equal `project.version`**, and re-runs the wheel-hygiene assertion (the
  wheel must ship only source files).

---

## One-time setup

Do this once. After it exists, releases are just the [runbook](#cutting-a-release).

### 0. Prerequisites

- A **PyPI** account and a **TestPyPI** account (separate accounts/sites:
  <https://pypi.org> and <https://test.pypi.org>), each with **2FA enabled**
  (mandatory on PyPI).
- Admin access to the `standard-voice/standard_asr` GitHub repository.
- The release workflow already exists at `.github/workflows/release.yml`.

> We use **pending publishers** so neither project needs to exist on PyPI yet —
> the first successful publish creates the project automatically and binds it to
> this repo.

### 1. Configure the Trusted Publisher on PyPI (production)

1. Log in to <https://pypi.org> → **Your account → Publishing** (or, if the
   project already exists, the project's **Settings → Publishing**).
2. Under **Add a new pending publisher**, choose **GitHub** and enter exactly:
   - **PyPI Project Name:** `standard-asr`
   - **Owner:** `standard-voice`
   - **Repository name:** `standard_asr`
   - **Workflow name:** `release.yml`
   - **Environment name:** `pypi`
3. Save. PyPI now trusts uploads that come from this repo's `release.yml` running
   in the `pypi` environment — and nothing else.

### 2. Configure the Trusted Publisher on TestPyPI (dry-run index)

Repeat step 1 on <https://test.pypi.org> with the **same values**, except:
- **Environment name:** `testpypi`

TestPyPI is a throwaway index for rehearsing a release end to end.

### 3. Create the protected GitHub Environments

In the repo: **Settings → Environments → New environment**. Create **two**:

| Environment | Used by | Protection |
| --- | --- | --- |
| `pypi` | real releases (and manual `target=pypi`) | **Required reviewers** (you); optionally limit to protected tags/branches |
| `testpypi` | manual `target=testpypi` dry runs | Required reviewers optional (lighter gate is fine) |

The **required reviewer** on `pypi` is the human approval gate: the publish job
pauses until someone approves the deployment. Do **not** add any secrets to these
environments — Trusted Publishing needs none.

### 4. (Recommended) Protect `main`

**Settings → Branches → Add rule** for `main`: require the `checks-complete`
status check before merging. This keeps every release commit CI-green by
construction.

---

## How the release workflow works

`.github/workflows/release.yml`:

- **Triggers** on a **published GitHub Release** (the normal path) and on manual
  **`workflow_dispatch`** with a `target` input (`testpypi` | `pypi`) for dry
  runs.
- Has top-level `permissions: contents: read`; the single publish job opts up to
  `id-token: write` **only** (for OIDC) and runs in the `pypi` or `testpypi`
  Environment selected from the trigger.
- Steps: checkout → install pinned `uv` (no cache, so a poisoned cache can't taint
  a release) → **assert the tag matches `project.version`** (release events only)
  → `uv build --package standard-asr` → assert the wheel ships only source files →
  publish with `pypa/gh-action-pypi-publish` (Trusted Publishing + PEP 740
  attestations).
- All actions are **SHA-pinned**; the file is validated by `actionlint` and
  `zizmor` in CI.
- It is **not** part of `checks-complete` — releasing is tag/Release-driven, never
  a PR gate.

---

## Cutting a release

The full runbook. (After the first time, this is all you do.)

### Step 1 — Pre-flight

- `main` is green in CI and contains everything you want to ship.
- Decide the new version per [the versioning policy](#versioning-policy-semver).

### Step 2 — Bump version + changelog

1. Edit `pyproject.toml`: set `version = "X.Y.Z"`.
2. Edit `CHANGELOG.md`:
   - Rename the `[Unreleased]` section to `[X.Y.Z] - YYYY-MM-DD` (today's date).
   - Add a fresh empty `[Unreleased]` above it.
   - Update the compare links at the bottom of the file.
3. Sanity-check locally:
   ```bash
   uv version --short        # prints X.Y.Z — must match what you'll tag
   uv build --package standard-asr   # builds cleanly
   ```

### Step 3 — Land it on `main`

Open a PR (or push) with `chore(release): vX.Y.Z`, get CI green, merge **without
squashing if it carries meaningful history** (normal release bumps are a single
commit, so squash-vs-merge doesn't matter here). The release must be cut from a
commit on `main`.

### Step 4 — Dry-run to TestPyPI

1. GitHub → **Actions → Release → Run workflow**.
2. Set **target = `testpypi`**, run from `main`.
3. Approve the `testpypi` environment when prompted.
4. Watch it build and upload to <https://test.pypi.org/project/standard-asr/>.

### Step 5 — Verify from TestPyPI

Install the just-uploaded build in a throwaway environment. TestPyPI does not
mirror real dependencies, so pull *them* from real PyPI:

```bash
# with uv
uv venv /tmp/relcheck && source /tmp/relcheck/bin/activate
uv pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "standard-asr==X.Y.Z"

# or with pip
python -m venv /tmp/relcheck && source /tmp/relcheck/bin/activate
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "standard-asr==X.Y.Z"
```

Then smoke-test:

```bash
python -c "import importlib.metadata as m; print(m.version('standard-asr'))"  # prints X.Y.Z
python -c "import standard_asr"   # imports cleanly
standard-asr --help
standard-asr models list
```

Confirm the TestPyPI project page renders the README, classifiers, and links.

### Step 6 — Publish to PyPI (the GitHub Release)

1. GitHub → **Releases → Draft a new release**.
2. **Choose a tag:** `vX.Y.Z` — create it on publish, targeting the release
   commit on `main`. The tag's version part **must equal** `project.version`
   (the workflow fails the run otherwise).
3. **Title:** `vX.Y.Z`. **Notes:** paste the new `CHANGELOG.md` section.
4. For a pre-release, tick **"Set as a pre-release"** (see policy below).
5. **Publish release.** This fires `release.yml` on the `release: published` event.
6. The publish job pauses on the **`pypi` environment** — **approve** it.
7. It builds and uploads to <https://pypi.org/project/standard-asr/> with
   attestations.

### Step 7 — Verify and wrap up

```bash
uv venv /tmp/relverify && source /tmp/relverify/bin/activate
uv pip install "standard-asr==X.Y.Z"
standard-asr --help
```

- Confirm the PyPI page shows the new version and **verified** provenance
  (attestations).
- Confirm the GitHub Release shows the built sdist + wheel as assets.
- Announce as appropriate.

---

## Testing on TestPyPI (details & gotchas)

- **Always pass `--extra-index-url https://pypi.org/simple/`** when installing
  from TestPyPI — TestPyPI does not host `numpy`, `pydantic`, etc., so the resolve
  fails without it.
- **A version can be uploaded only once per index.** PyPI and TestPyPI both reject
  re-uploading an existing version (immutability). If a TestPyPI dry run is
  broken, bump to a throwaway pre-release (e.g. `X.Y.ZrcN`) and try again — do not
  expect to overwrite.
- TestPyPI accounts and trusted publishers are **separate** from production; both
  must be configured (steps 1–3).
- The dry run exercises the *exact* build + publish path, so a green TestPyPI run
  is strong evidence the production publish will succeed.

---

## Versioning policy (SemVer)

The version in `pyproject.toml` is the single source of truth; the git tag mirrors
it (`vX.Y.Z`), and the workflow enforces the match. We follow
[Semantic Versioning](https://semver.org):

- **Pre-1.0 (`0.y.z`) — where we are now.** The public API is still stabilising.
  A **MINOR** bump (`0.1 → 0.2`) may include **breaking changes**; **PATCH**
  (`0.1.0 → 0.1.1`) is reserved for backward-compatible fixes. Always call
  breaking changes out clearly in `CHANGELOG.md`.
- **Post-1.0.** Standard SemVer: MAJOR = breaking, MINOR = backward-compatible
  features, PATCH = backward-compatible fixes. Reaching `1.0.0` is a deliberate
  signal that the interface contract is stable enough for the ecosystem to build
  on long-term (the project's mission).
- **Pre-releases.** Tag `vX.Y.ZaN` / `bN` / `rcN` (e.g. `v0.2.0rc1`) and set
  `project.version` to the same string (`0.2.0rc1`). PyPI marks these as
  pre-releases — `pip install standard-asr` skips them; only `pip install --pre`
  or an exact pin picks them up. Use an `rc` for a real release candidate; use
  TestPyPI for throwaway rehearsals.
- **Yanking.** If a released version is found broken, **yank** it on PyPI (do not
  delete): yanked versions stay installable by exact pin but are skipped by normal
  resolution. Then ship a fixed PATCH. Never reuse a version number.

---

## The `v0.2.0` and every future release

Once the one-time setup exists, **future releases never touch PyPI/GitHub
settings again** — they are purely the [runbook](#cutting-a-release). Concretely,
for `v0.2.0`:

1. Land the `v0.2.0` work on `main` (green CI).
2. `version = "0.2.0"` in `pyproject.toml`; move `CHANGELOG.md` `[Unreleased]` →
   `[0.2.0] - YYYY-MM-DD`; if `0.2.0` includes breaking changes (allowed pre-1.0),
   lead the changelog section with a **"Changed/Breaking"** list and migration
   notes. Commit `chore(release): v0.2.0` to `main`.
3. (Optional but recommended) `rc` rehearsal: set `version = "0.2.0rc1"`, tag
   `v0.2.0rc1`, publish as a pre-release, install with `--pre`, validate; then bump
   to `0.2.0` for the real release. Or just dry-run to TestPyPI.
4. Dry-run to TestPyPI (`workflow_dispatch target=testpypi`) → verify.
5. Draft GitHub Release, tag `v0.2.0`, paste changelog, publish → approve `pypi` →
   done.
6. Verify install + attestations.

The steady-state mental model: **bump → changelog → (test) → tag a Release →
approve**. Everything security-sensitive (no tokens, attestations, env gate) is
already wired into `release.yml`.

### Patch / hotfix releases

For an urgent fix to a released line (e.g. `0.2.0 → 0.2.1`): branch the fix,
land it on `main` with CI green, `version = "0.2.1"`, changelog `[0.2.1]`, then run
the normal runbook. Patch releases must be backward-compatible.

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Workflow fails at **"Assert release tag matches project version"** | The tag (`vX.Y.Z`) ≠ `project.version`. Fix `pyproject.toml` or re-tag so they match, then re-release. |
| Publish step: **"trusted publisher … not configured" / 403** | The PyPI/TestPyPI trusted publisher values don't match (owner/repo/workflow filename/environment). Re-check [step 1/2](#1-configure-the-trusted-publisher-on-pypi-production). The workflow **filename** must be `release.yml` and the **environment** must match (`pypi` / `testpypi`). |
| Job never starts the publish step | The Environment is awaiting **approval** — approve the deployment in the run's UI. |
| **"File already exists"** on upload | That version was already uploaded (indexes are immutable). Bump the version (or, on TestPyPI, use a new `rcN`). |
| TestPyPI install can't find `numpy`/`pydantic` | Add `--extra-index-url https://pypi.org/simple/`. |
| No attestations shown on PyPI | Ensure the publish ran under Trusted Publishing (OIDC) with `id-token: write`; attestations are on by default in the pinned `gh-action-pypi-publish`. |
| `uv version --short` shows the wrong number | You forgot to bump `version` in `pyproject.toml`. |

---

## Security & provenance (what consumers get)

Because releases use Trusted Publishing + PEP 740 attestations, anyone can verify
that a `standard-asr` artifact on PyPI was built by **this** repository's
`release.yml` at a specific commit — not by a leaked token or a third party. The
chain is: a human approves the `pypi` Environment → GitHub mints a short-lived
OIDC token scoped to this repo/workflow/environment → PyPI accepts the upload only
if it matches the configured trusted publisher → each file is published with a
Sigstore-backed attestation of its provenance. No secret is ever stored, and the
build is reproducible from the pinned `uv` + committed `uv.lock`.

---

## Future: publishing engine plugins

Today only the core publishes. When an engine adapter graduates from the in-repo
`cookbook/` to a real, installable plugin (e.g. `std-faster-whisper` moving to its
own repository), it follows the **same pattern**, independently:

- Its own repo + `release.yml` (this file is the template).
- Its own PyPI project + trusted publisher + protected Environment.
- Its own SemVer line and `CHANGELOG.md`; it depends on `standard-asr>=X.Y` from
  PyPI (not a workspace path).

This keeps each engine's dependencies, license, and release cadence isolated
(goals G.4.1 / G.4.2) while every plugin is instantly usable by every Standard ASR
application.

---

## References

- PyPI Trusted Publishers: <https://docs.pypi.org/trusted-publishers/>
- Publishing action: <https://github.com/pypa/gh-action-pypi-publish>
- PEP 740 (digital attestations): <https://peps.python.org/pep-0740/>
- Keep a Changelog: <https://keepachangelog.com/en/1.1.0/> · SemVer:
  <https://semver.org/>
- Design history: GitHub issue #4 ("Add a release / build / publish pipeline").
