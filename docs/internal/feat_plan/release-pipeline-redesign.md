# Release pipeline redesign (design case)

**Status:** Implemented on 2026-10-21 in pull request #125. The design was settled on 2026-07-05 and amended on 2026-10-21, before the implementation landed; section 8 records the amendment.

> Rebuilt 2026-07-05 (an earlier draft was lost, never committed), then settled the same day in maintainer discussion. Re-derived from the v0.1.0 post-mortem record and a fresh audit of `release.yml` / `ci.yml` / `RELEASING.md` as of `2718fb3`.
>
> Amended 2026-10-21, after a design review found that a second run of the same version was not safe. Sections 2 to 7 describe the amended design; section 8 says what changed and why.

Delivers the roadmap line (#27): **"Release pipeline redesign — fully rehearsable, zero-burn-risk process."** The amendment narrows "zero-burn-risk" to the promise at the end of section 2.1.

Standing constraint (maintainer decision 2026-06-16): **immutable releases and the `release-tags` deletion rule stay OFF** until the redesigned pipeline has shipped several clean releases. Re-enabling both is the *final* step of this work (Phase 3), not a prerequisite.

---

## 1. Problem statement — the v0.1.0 post-mortem

The first real release (2026-06-16, tag `v0.1.0`) failed and permanently burned its version number:

1. `release: published` fired; `build` passed all guards.
2. The then-existing `attach-release-assets` job ran `gh release upload` against the just-published release → **422**: immutable releases freeze a release at publish time and reject post-publish asset uploads.
3. Via `needs: [build, attach-release-assets]`, the failure blocked `publish-pypi` → nothing reached PyPI.
4. During recovery the release object was deleted — and GitHub's immutable releases **permanently reserve a tag name once used**. `v0.1.0` is unrecoverable; GitHub's documented remedy is "use a new version".
5. Shipped as `v0.1.1` instead (PR #22 dropped asset attachment; PR #23 bumped).

Two structural root causes:

- **R1 — The release-only path is never rehearsed.** The TestPyPI dry run exercises only `build` + `publish-testpypi`; every release-only step (the tag==version guard body, the whole `publish-pypi` job, the `pypi` environment gate, the production OIDC handshake, tag-ref concurrency — 8 items total, enumerated in the 2026-07-05 audit) ran for the first time during the first irreversible real release.
- **R2 — The irreversible act happens *before* the checks.** The trigger is `release: published`: the public tag + release object exist *first*, then the guards run. Every guard is therefore a tag-burning failure mode.

Additional audit findings (2026-07-05): the `testpypi` environment has **no protection rules** — `RELEASING.md` frames its reviewer gate as optional ("approve … *if it is protected*"), and it is not, so nothing actually gates the TestPyPI leg; post-publish verification (install checks, attestation presence, notes-match-changelog) is entirely manual eyeballing; `pyproject.toml` is the version source of truth (the tag is only asserted equal — a property the new design builds on).

One constraint no design can remove: **PyPI upload is itself irreversible** (version filenames are reserved even after deletion/yank). The goal is therefore ordering and rehearsal — by the time the pipeline reaches PyPI, everything before it has been exercised in the same run, and everything after it is safe to run again.

---

## 2. Decided design (2026-07-05, amended 2026-10-21)

**One pipeline, one human action to start it, ordered by irreversibility.** The GitHub Release moves from *trigger* to *last step*; the TestPyPI rehearsal moves from a *separate manual dry run* to an *inline stage of every release*. A second run of the same version resumes the first: it builds the same commit with the same recipe, gets the same bytes, and finishes what the first run left undone (section 8).

```text
[human] merge the release PR, chore(release): vX.Y.Z     (version + CHANGELOG
        section + uv.lock, one commit on main)
[human] run the workflow (workflow_dispatch)   ← the only trigger; no inputs;
  │                                              version read from
  │                                              pyproject.toml on main
  ├─ 1. guards (all before anything is public):
  │      • ref is refs/heads/<default branch>, not a tag of that name
  │      • pyproject.toml writes the version in uv's normalized form   (§8)
  │      • version is releasable: not .dev, inside the policy          (§2.3)
  │      • release commit = the newest first-parent commit that set
  │        the version; the commits after it are printed;
  │        checks-complete green on the release commit                (§8)
  │      • CHANGELOG.md at the release commit has a section for X.Y.Z (§2.5)
  │      • tag and GitHub Release, if present, match this run         (§2.1)
  ├─ 2. build at the release commit with the uv it names (sdist + wheel,
  │      --no-sources) + wheel hygiene + isolated smoke test
  │      + digest guard: each index lacks X.Y.Z or holds only files
  │        with this build's sha256                                   (§8)
  ├─ 3. publish → TestPyPI   (skip-existing: true)
  ├─ 4. verify TestPyPI: every built file listed with the same sha256 and
  │      an attestation; install the package from TestPyPI and its
  │      dependencies from PyPI; pip check; import; version; --help
  ├─ 5. publish → PyPI       (skip-existing: true)   ← the irreversible step;
  │                                                     everything above it ran
  │                                                     in this same run
  ├─ 6. verify PyPI: the same digest and attestation checks; isolated
  │      install; import; version; --help
  ├─ 7. GitHub Release: the tag and release checks again, then, when absent,
  │      gh release create vX.Y.Z --target <release commit>
  │      (creates tag + release together; notes = the CHANGELOG section;
  │      pre-release flag from the version)
  └─ 8. dev-bump branch from the current tip of main: next .dev version + uv lock
         (X.Y.(Z+1).dev0, or X.Y.Zrc(N+1).dev0 after rcN); the run prints
         the link, and a person opens the pull request and merges it
```

What makes a second run safe, each part explained in section 8:

- **The release commit is the commit that set the version** (`scripts/release_commit.py`), not the tip: the newest commit on the first-parent history of `main` whose parsed `project.version` is `X.Y.Z` while its first parent's is not.
- **The release commit fixes the build recipe.** The build runs on a detached checkout of the release commit, with the uv whose version is the lower bound of the `uv-build` requirement there.
- **The build is reproducible**, and CI checks that on every push.
- **A digest guard** (`scripts/release_index.py guard`) lets a second run past a version that an index already holds, as long as every file there is this build.
- **Every step after the build is safe to run twice**, the GitHub state included (`scripts/release_github.py`).

### 2.1 Why publish-PyPI-before-GitHub-Release

The GitHub Release is the only step that permanently burns a tag name (once immutable releases return). Placed last, its failure-downstream is empty: when the release step fails, PyPI is live and verified, and a second run checks the indexes, skips the uploads, and creates the release. It also makes the announcement follow the fact: no window where GitHub says "released" but `pip install` fails.

The tag and the release are checked before anything is published, not only when the release is created. GitHub ignores `--target` when the tag already exists (the [REST reference](https://docs.github.com/en/rest/releases/releases#create-a-release) says `target_commitish` is "Unused if the Git tag already exists"), so a stray tag would put the release on another commit. `scripts/release_github.py check` runs in the `prepare` job. It stops the run when the tag points at another commit, or when a release exists without its tag, with other notes, or with the wrong pre-release flag. A lookup that fails for any reason other than 404 is an error, not an absence. The release step runs the same checks with `ensure`, creates the release when it is absent, and checks again.

Drafts are a gap. GitHub lists a draft release only to a token with write access ([List releases](https://docs.github.com/en/rest/releases/releases#list-releases)), and `check` runs with read access. A stray draft therefore stops the run at the release step, after the indexes hold the files: a person publishes or deletes the draft, and a second run resumes.

**When a second run resumes.** Each row says what is public after a failure at that point, and when a second run of the same version finishes the release:

| Failure point | Public state | A second run resumes when |
|---|---|---|
| Guards, build, smoke (1–2) | nothing | the cause lies outside the release commit: CI unfinished or canceled on it, a setting, an outage. A fix to the source lands after the release commit, so it needs a new release commit (section 8, decision 5). |
| TestPyPI publish or verify (3–4) | some or all files of `X.Y.Z` on TestPyPI | every file TestPyPI holds is this build: the publish uploads the rest, and the verify step checks the whole set and its attestations. A file with another digest, a file the build did not produce, a file without an attestation at the deadline, or a bug the rehearsal finds spends `X.Y.Z`, and the fix ships as a new version. |
| PyPI publish, auth or config error (5) | nothing new on PyPI | the trusted publisher or the `pypi` environment is fixed. |
| PyPI publish, partial upload (5) | some files of `X.Y.Z` live on PyPI | every file PyPI holds is this build: `skip-existing` fills the gap, and verify (6) checks the full set. A file that reached PyPI without an attestation never gets one, so that filename is burned and a new version follows. |
| Verify PyPI (6) | `X.Y.Z` live on PyPI; no tag or release | the index or the install lagged. A file with another digest, or without an attestation at the deadline, burns that filename: release a new version. The tag and the release are intentionally not created until this step passes. |
| Release creation (7) | `X.Y.Z` live and verified on PyPI; the tag and the release absent, or left by an earlier attempt | the tag and the release are absent, or exist and match this run. A draft, a tag on another commit, other notes, or another pre-release flag stops the run until a person reconciles it. So does a workflow change merged after the release commit while the tag does not exist yet: the workflow token cannot create a tag there (decision 6), so a person creates the release with the printed command, and a second run then finds it. |
| Dev-bump branch (8) | release complete | always: the job keeps a branch that already holds its commit and prints the link again. |

A second run resumes only while the branch still declares `X.Y.Z`, that is, until the dev-bump pull request merges (section 8, decision 8).

**The promise.** The checks for each irreversible publication run before it. A run that failed after a publication can run again, and it resumes with the same bytes. A version number is lost only when what a run published turns out to be wrong — a bug in the bytes, or a file without its attestation — and then only that number. The pipeline this design replaced did the opposite: every guard ran after the release object existed.

### 2.2 Inline TestPyPI as the rehearsal mechanism

Every real release rehearses the publish machinery (the workflow artifact download, the PyPA publish action, an OIDC trusted-publishing handshake, the index's own checks of an upload) against TestPyPI **in the same run, with the same `dist/` files** that then go to production. The never-rehearsed set shrinks to the irreducible production-only deltas: the production trusted-publisher registration, the `pypi` environment, and the production upload itself — all "constant once configured correctly".

On a resume, the files an index already holds are not uploaded again, so that run rehearses only the uploads still missing. The loudness stays: the digest guard checks both indexes before the first upload, and the verify step after each publish requires every built file, with its digest and its attestation.

Accepted trade-offs:

- **TestPyPI availability** becomes a release dependency. Accepted: releases are rare and never urgent. Deliberately **no skip-TestPyPI escape flag** — an escape flag would re-create the unrehearsed path (R1).
- **`skip-existing: true` on both indexes** trades "duplicate upload fails loudly" for partial-upload recovery. Loudness is preserved where it matters: the **digest guard** stops a run whose build differs from a file an index holds, before anything is uploaded, and the **mandatory verify** after each publish fails, with an explicit message, on any built file that is missing or differs.
- **A rehearsal spends the version on TestPyPI.** TestPyPI keeps filenames as PyPI does, so a bug the rehearsal finds ships as a new version.

### 2.3 Post-release dev versioning + the accidental-dispatch guard

After each release, a pull request the pipeline opens moves `main` to the next development version (step 8), with the `uv lock` change in the same commit. The version is a placeholder that sorts after the released version and before every version the next release pull request can choose: `X.Y.(Z+1).dev0` after a final release, and `X.Y.Zrc(N+1).dev0` after `X.Y.ZrcN` (`aN` and `bN` likewise), so the branch's version never goes backwards through a pre-release series (`scripts/release_version.py next`). The next release pull request sets the real number.

Paired with the guard that refuses a development version (`scripts/release_version.py classify`, which also refuses a version outside the versioning policy), this makes main **safe-by-default**: between releases, an accidental dispatch fails in seconds at step 1 with nothing public. Only the deliberate act of merging a release pull request arms the pipeline. Regular pull requests never touch the version — exactly two moments in the lifecycle do (the release pull request and the dev-bump pull request).

The dev-bump job first moves to the current tip of `main`, because the run may have started long before, and it does nothing when the branch no longer declares `X.Y.Z`. Otherwise it commits the next development version to `release/dev-bump-vX.Y.Z`, unless the branch already holds a commit of its own (section 5 says how), and prints the link that opens the pull request.

A person opens the pull request from that link and merges it after `checks-complete` passes. Until 2026-10-22 the job opened the pull request itself; a review found two reasons against it. The workflow token may open a pull request only where the repository allows GitHub Actions to create pull requests, a setting that also lets workflows approve them, and this repository keeps it off. And GitHub holds the CI runs of a pull request the workflow token opens until a person approves them ([GITHUB_TOKEN](https://docs.github.com/en/actions/concepts/security/github_token#when-github_token-triggers-workflow-runs)), while a pull request a person opens runs CI at once. Either way a person acts once, and opening the pull request needs no extra permission. It also leaves nothing to identify: the job no longer looks up a pull request by its branch name, which a pull request from a fork can share.

### 2.4 Human gates

- The **`pypi` environment stays** (the trusted-publisher registration binds to it). Its **required reviewer stays for the first one–two releases** on the new pipeline as a circuit breaker, then is removed: with a solo maintainer, dispatcher and approver are the same person minutes apart — no information is added (decision 2026-07-05; revisit if maintainers multiply).
- The `testpypi` environment's documented-but-unconfigured approval is resolved by the redesign itself: the inline stage needs no approval, and the runbook stops claiming one.

### 2.5 Release notes

Generated from the `CHANGELOG.md` section for the version, read at the release commit, not at the tip: the lines after the `## [X.Y.Z]` heading, up to the next `## [` heading. Guard 1 stops the run when that section is missing or empty. The notes travel with the distribution files in the workflow artifact `standard-asr-release`, so the release step uses the text the guard read. When a release already exists, its notes are compared with them; line endings, trailing whitespace, and blank lines at either end do not count, and any other difference stops the run for a person to reconcile. CHANGELOG stays hand-written — including the unwrapped-lines style — so the release notes remain hand-authored in substance; the pipeline only copies them. "Notes match changelog" stops being a manual checklist item because it becomes true by construction.

---

## 3. Alternatives considered

- **rc-tags routed to TestPyPI as the rehearsal mechanism** (this document's earlier draft): superseded by the inline TestPyPI stage, which rehearses with the same files, in the same run, with zero extra ritual. **rc releases themselves remain natively supported** — a release pull request setting `0.5.0rc1` rides the same pipeline as a PEP 440 pre-release (pip ignores it without `--pre`): the GitHub Release is marked as a pre-release, and the dev-bump sets `0.5.0rc2.dev0`. Useful near the 1.0 protocol freeze, not needed now.
- **Keep `release: published` trigger + mandatory pre-flight check**: shrinks R2's window without closing it; relies on human discipline. Rejected.
- **Draft-release flow** (check a draft, then flip it to published): the flip step re-introduces a fallible step around the irreversible one; the tag still precedes the checks. Rejected.
- **release-please / python-semantic-release** (evaluated 2026-07-05): release-please's native flow is *merge release PR → it creates tag + GitHub Release → publish triggers off the release* — exactly the R2 ordering this redesign eliminates. Its actual value (auto-drafting the version bump + CHANGELOG from Conventional Commits) automates the one part of this project's process that is cheap, while taking over the part that is a deliberate design judgment for a protocol library (version semantics; the maintainer has a standing no-reflex-bumps rule). **Deferred**: revisit if release cadence or maintainer count grows; if adopted, use it in "release-PR author only" mode (no tagging, no release creation — those stay with this pipeline).
- **Reuse the first run's workflow artifact on a resume** (2026-10-21): a second run would download the distribution files the first run built instead of building again. Rejected: an artifact that expires after 90 days becomes a second source of truth for the release bytes, while rebuilding from the release commit is simpler, and CI checks on every push that a rebuild gives the same bytes.
- **A `workflow_dispatch` input naming the release commit** (2026-10-21). Rejected: a hand-typed SHA is the kind of input this design removes, and the version alone identifies the commit.

---

## 4. Documentation changes (part of implementation, not optional)

- **`RELEASING.md` — full rewrite.** The runbook shrinks to: (1) merge the release pull request (version + CHANGELOG); (2) press Run workflow; (3) open the dev-bump pull request from the link the run prints, then merge it. Plus: what each guard failure means and the recovery play per §2.1's resume table (replacing the old troubleshooting table, whose remedies assume tag surgery that immutable releases forbid). One-time setup sections (trusted publishers, environments) stay.
- **`CONTRIBUTING.md` §Releasing — update the summary.** It pointed to `RELEASING.md` with a three-line description of the *old* flow; it now describes the new one. The split is correct and stays: CONTRIBUTING for contributors (never touch the version in normal pull requests), RELEASING as the maintainer runbook.
- Both are updated **in the same pull request** as the workflow change — a runbook describing the previous pipeline is worse than none.

---

## 5. Implementation notes & pre-flight checks

- **Jobs and permissions:** seven jobs run in order — `prepare`, `publish-testpypi`, `verify-testpypi`, `publish-pypi`, `verify-pypi`, `github-release`, `dev-bump` — each after the one before it succeeded. `contents: write` only on `github-release` (step 7); `contents: write` on `dev-bump` (step 8); `id-token: write` only on the two publish jobs; `checks: read` on `prepare`; everything else stays `contents: read`.
- **Default-branch guard:** the first step compares the full ref, `GITHUB_REF`, with `refs/heads/<default branch>`, because a tag named like the branch has the same short name (`GITHUB_REF_NAME`). `tests/test_release_workflow.py` runs that step's shell from `release.yml` under bash for `refs/heads/main` (accepted), `refs/heads/feature`, and `refs/tags/main` (both refused).
- **Historical checkout:** `prepare` checks out the whole history (`fetch-depth: 0`), because the first-parent walk needs it. `release_commit.py` refuses a shallow checkout: there the walk would end at the fetch boundary and take the boundary commit for the one that set the version.
- **Frozen recipe:** after the guards, `prepare` installs the uv that the release commit names, builds on a detached checkout of the release commit, and runs the smoke tests with that uv. uv builds with its bundled copy of `uv_build` when its own version satisfies the `uv-build` requirement ([uv docs](https://docs.astral.sh/uv/concepts/build-backend/#bundled-build-backend)), and the lower bound always does, so the release commit fixes the backend too. `UV_VERSION` in `release.yml` runs the workflow's own commands: reading the version, the install checks, and the dev-bump commit. The scripts always run from the dispatched commit (the step after the build returns to it), so a release commit older than the scripts still releases.
- **State checks:** `scripts/release_github.py` reads the tag and follows an annotated tag to its commit. It looks the release up in the release list, because the by-tag endpoint answers only published releases. `check` runs in `prepare` with a read-only token; `ensure` runs in `github-release` with write access, so it also sees drafts.
- **Retries:** `fetch_json` in `scripts/release_index.py` makes up to three attempts, five seconds apart, after a server error, a rate limit (HTTP 429), or a failed connection. A 404 is an answer (the version is absent), and any other client error stops the command at once. It closes the response that an HTTP error holds before it returns, retries, or raises, so no response waits for the garbage collector; on Python 3.14 that wait raised a `ResourceWarning`, which the test suite turns into an error. `verify` and `attestations` poll until a deadline (the `--timeout` default in the same script), because an index can lag a publish. The install checks in the two verify jobs try six times, 30 seconds apart, with `--refresh-package standard-asr`, so each attempt asks the index again instead of reusing the page that uv cached on the attempt before.
- **CI reproducibility assertion:** the `build` job in `ci.yml` builds a second time from a fresh checkout of the same commit (`git worktree add` under `RUNNER_TEMP`: new file times, another path) and fails when the sha256 lists differ. It runs with the uv that `ci.yml` pins, so it checks that the build does not depend on file times, the path, or chance; the uv pinned at the release commit covers the backend version. On 2026-10-21, two fresh clones of `31612eb`, at different paths and with different file times, built with uv 0.13.4, gave identical digests for the wheel and the sdist.
- **Guard on a real build:** every build, in `release.yml` and in `ci.yml`, passes `--no-create-gitignore`. Without it, uv writes a `.gitignore` into the output directory, and the digest guard, which accepts exactly the wheel and the sdist, refuses the directory before it reads either index. CI runs the guard on a real build: `tests/test_release_build.py` builds the package with the command of the workflow's build step and runs `local_files()` and `guard()` on the result against four index states: nothing published, the wheel already on TestPyPI, the whole build on both indexes, and the wheel's filename on PyPI with other bytes, which fails.
- **Dev-bump commit:** the checkout keeps no credentials (`persist-credentials: false`), and the job writes through the API with the workflow token. `release_github.py branch-head` reads the branch; when it is absent, the job creates it at the tip of `main` through the REST refs endpoint. The compare endpoint's `ahead_by` says whether the branch holds a commit of its own. When it holds none (new, or left by a run that failed before committing), `uv version --no-sync` sets the next version and re-locks without building an environment, `uv lock` runs after it as a guard, and the GraphQL mutation `createCommitOnBranch` commits `pyproject.toml` and `uv.lock` with `expectedHeadOid` set to the branch head, so a concurrent push fails loudly instead of being lost. Otherwise the branch's own commits stay. Either way the job ends by printing the link that opens the pull request, as a notice and on the run's summary page. GitHub's reference for the mutation says commits made with it "are automatically signed by GitHub if supported" and marked as verified in its interface ([GraphQL reference](https://docs.github.com/en/graphql/reference/commits#mutation-createcommitonbranch)). The `main` ruleset does not require signed commits (checked 2026-10-21); the signature lets the pull request merge if a ruleset does later. The two writes, the ref and the mutation, replace the REST sequence of two blobs, a tree, a commit, and a ref.
- **Concurrency and timeouts:** a workflow-level concurrency group, `release`, with `cancel-in-progress: false`, so two runs never interleave and a second run waits for the first; `timeout-minutes` on every job, so a hung publish is a diagnosis, not a zombie run.
- **Digest sources:** the index JSON API gives each file's sha256 ([JSON API](https://docs.pypi.org/api/json/)), so the check downloads nothing; attestation presence comes from the PyPI integrity API ([Integrity API](https://docs.pypi.org/api/integrity/)), the same check used to verify v0.1.1 manually.
- **Removed paths:** the `release: published` trigger and the TestPyPI-only dispatch are gone. Every dispatch runs the whole pipeline, and a development version on the branch stops it at the first guard.
- **Check before the first run:** whether the `release-tags` ruleset lets the Actions token create `vX.Y.Z` tags via `gh release create` (actor bypass may need configuring). Check the exact ruleset behavior *before* the first run, not during it. `RELEASING.md` lists this with the other first-run checks. On 2026-10-21, GitHub's public rules API listed one rule for `refs/tags/v*`, `non_fast_forward`, which does not restrict creating a tag; the check stands, because only an administrator can see the bypass settings.

The four scripts and their tests:

| Script | Commands | Tests |
|---|---|---|
| `scripts/release_commit.py` | prints the release commit and its uv version | `tests/test_release_commit.py` |
| `scripts/release_version.py` | `classify`, `next` | `tests/test_release_version.py` |
| `scripts/release_index.py` | `guard`, `verify`, `attestations` | `tests/test_release_index.py` |
| `scripts/release_github.py` | `check`, `ensure`, `branch-head` | `tests/test_release_github.py` |

Three of them use only the standard library and run with the runner's `python3`; `release_commit.py` needs Python 3.11 or later, for `tomllib`. `release_version.py` needs `packaging` and runs under `uv run --no-project --with packaging`. Two more test modules check the parts together: `tests/test_release_build.py` feeds the guard a real build, and `tests/test_release_workflow.py` runs the default-branch guard's shell. The first skips where uv is missing, the second where bash is.

---

## 6. Phasing

- **Phase 1 — implement + first release:** new workflow + rewritten `RELEASING.md`/`CONTRIBUTING.md`, implemented by the pull request named in the status block; then ship the next release on it with the `pypi` reviewer gate still on. That first release is still to come.
- **Phase 2 — tighten:** remove the reviewer gate (§2.4); add release-PR checks (CHANGELOG entry + `uv lock --check` on version-touching pull requests) so guard 1 failures move even earlier, to review time. The amendment makes this more useful: a CHANGELOG section missing at the release commit needs a new release commit (section 8, decision 5).
- **Phase 3 — lock:** after several consecutive clean releases, re-enable immutable releases + the `release-tags` deletion rule (the deliberate final step). From then on, the ordering by irreversibility is what makes the protections safe to live with.

---

## 7. Remaining open items

1. **"Several clean releases" threshold for Phase 3** — maintainer judgment at the time; no number pre-committed.
2. **Ruleset and actor check** (§5) — an implementation pre-flight before the first run, not a design question.

---

## 8. Amendment (2026-10-21): a second run resumes the first

A design review by Codex (gpt-6-astra) on 2026-10-21 found that the 2026-07-05 design could not keep its promise that no failure strands a burned version number. Recovery meant dispatching the same version again, and a second run of the same version was not the same run, in three ways:

1. **Another commit.** The design built the tip of `main`. A merge between the two runs changed the build, so the digest check after the publish would find other bytes on the index, and the version would be burned.
2. **Another recipe.** The workflow named the uv that built, so a workflow change between the runs, such as a newer `UV_VERSION`, could change the bytes of the same commit.
3. **Unknown GitHub state.** The tag or the release could already exist, in a state nobody had checked: GitHub ignores `--target` for an existing tag, and an existing release could carry other notes. On top of that, the guard "version absent from PyPI & TestPyPI" stopped every second run once a file had reached an index.

The amendment makes a second run of the same version a resume of the first. The decisions:

1. **The release commit is the commit that set the version, not the tip.** `scripts/release_commit.py` walks the first-parent history of the branch and picks the newest commit whose parsed `project.version` (read with `tomllib`) equals the version on the branch while its first parent's does not. Later merges belong to the next version, and a resumed run must build the same commit. A merge of the release pull request is one commit on the first-parent line, where a `git log -S` string search would find a commit inside the pull request, or a comment that happens to contain the string. The run prints the first-parent commits after the release commit, so the person sees what the release does not contain, and it requires a successful `checks-complete` check run on the release commit. The script refuses a shallow checkout, where the walk would end early and take the wrong commit. Before the walk, the workflow checks that `pyproject.toml` writes the version in the normalized form that uv prints, because the walk compares the written string with it.
2. **The build recipe is frozen at the release commit.** The uv that builds is the lower bound of the `uv-build` requirement in `[build-system].requires` at the release commit; the workflow's `UV_VERSION` runs only the workflow's own commands. uv builds with its bundled copy of `uv_build` when its version satisfies that requirement ([uv docs](https://docs.astral.sh/uv/concepts/build-backend/#bundled-build-backend)), so the release commit fixes the backend as well as the frontend. The build runs on a detached checkout of the release commit with `uv build --package standard-asr --no-sources --out-dir dist --clear --no-create-gitignore`. A workflow change after the release commit therefore cannot change what a resumed run builds.
3. **The build is reproducible, and CI asserts it on every push.** The `build` job in `ci.yml` builds a second time from a fresh checkout of the same commit and compares the sha256 lists. A resume compares the digests of a fresh build with what the index holds, so a build that depended on file times, on the path, or on chance would make every resume look burned.
4. **A digest guard replaces the "version absent from both indexes" guard.** Before anything is published, `scripts/release_index.py guard` requires, for each index, that the version is absent or that every file the index holds has the digest of the freshly built file with the same name. A file with another digest, or a file the build did not produce, means the version is burned on that index, and the remedy is a new version. A file the index lacks is fine: the publish fills it. After each publish, `verify` polls until the index holds every built file with the same digest, and `attestations` polls the integrity API for a PEP 740 provenance record per file. A file uploaded without an attestation never gets one, so a record missing at the deadline means that filename is burned.
5. **A version whose files reached an index is spent.** TestPyPI reserves filenames as PyPI does. When the rehearsal, or anything after the TestPyPI upload, shows a bug that needs a source change, the fix ships as a new version; by decision 1, the old version's release commit cannot contain a later fix anyway. Before any file reaches an index, the number is not spent, but a fix still needs a new release commit: the branch moves off the version, and a new release pull request sets it again, because the newest commit that sets the version is the release commit. The design stops claiming that no failure strands a version number; section 2.1 states the promise it can keep.
6. **Every step after the build is safe to run twice.** Both publishes use `skip-existing: true`, and `verify` and `attestations` check the full set afterwards. `scripts/release_github.py check` reads the GitHub state in `prepare`, before anything is published: the tag, if it exists, must point at the release commit; the release, if it exists, must not be a draft, must have its tag, must carry the CHANGELOG section as its notes, and must carry the right pre-release flag. A lookup that fails for a reason other than 404 is an error, not an absence. `ensure` runs after PyPI is verified: the same checks, then `gh release create vX.Y.Z --target <release commit> --title vX.Y.Z --notes-file <notes> [--prerelease]` when the release is absent, then the checks again. GitHub lists a draft release only to a token with write access, so `check`, with read access, cannot see one, and `ensure`, with write access, stops on it. A stray draft therefore stops the run at the release step, after the indexes hold the files; the person publishes or deletes the draft and runs the workflow again, and the run resumes.

   One more case needs a person. GitHub lets a token create a tag or a release at a commit whose files under `.github/workflows/` differ from the default branch's only if the token may change workflows, and the workflow token never may ([Create a release](https://docs.github.com/en/rest/releases/releases#create-a-release)). The release commit falls behind that way whenever a workflow change merges after it, for example a Dependabot update of an action. `scripts/release_commit.py --workflows-against` compares the two; `check` warns in `prepare` when the tag does not exist yet, and `ensure` stops instead of calling `gh release create`, printing the command a person runs with their own credentials. When the tag already exists, `ensure` creates the release without `--target`, which GitHub ignores for an existing tag. A token that may change workflows, such as one from a GitHub App with the Workflows permission, would remove the manual step. The pipeline does not hold one: such a token could rewrite any workflow, and the case arises only when a workflow change merges between the release commit and the release step, which the runbook tells the maintainer to avoid by running the release before merging such a change.
7. **The dev-bump branch is idempotent and pre-release-aware.** `scripts/release_version.py classify` refuses a development version (the accidental-dispatch guard) and any version outside the policy: `X.Y.Z` with an optional `a`, `b`, or `rc` suffix, with no epoch, no local part, no post-release, and no two- or four-part release. It also reports whether the version is a pre-release, which sets the GitHub Release's pre-release flag. `next` gives the next development version: `X.Y.(Z+1).dev0` after a final release and `X.Y.Zrc(N+1).dev0` after `X.Y.ZrcN`, so the branch's version never goes backwards through a pre-release series. The job first moves to the current tip of the default branch, because a run started before a long release would otherwise read a version the branch no longer declares, and it does nothing when the branch no longer declares the released version. It commits the next development version to `release/dev-bump-vX.Y.Z`, unless the branch already holds a commit of its own, and prints the link that opens the pull request; a person opens it and merges it (section 2.3 says why the job does not open it). The commit goes through the GraphQL mutation `createCommitOnBranch` with the workflow token, so the checkout keeps no credentials (section 5).
8. **The resumption window.** A second run of the version works until the branch's version changes, that is, until the dev-bump pull request merges; after that, `classify` stops the run at the first guard. A release left incomplete after that point is rare (the GitHub Release is missing), and a person finishes it by hand with the same `gh release create` command at the release commit.

Two smaller changes came with the amendment: the release notes come from `CHANGELOG.md` at the release commit, not at the tip (section 2.5), and the TestPyPI install takes only the package from TestPyPI and its dependencies from PyPI (section 2, step 4). A resume does not reuse the first run's workflow artifact, and the workflow takes no input naming the commit (section 3).
