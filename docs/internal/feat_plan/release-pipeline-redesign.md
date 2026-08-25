# Release Pipeline Redesign (Design Case)

**Status:** Design settled 2026-07-05 (decisions recorded below). NOT yet
implemented. Implementation is a dedicated PR touching
`.github/workflows/release.yml`, `RELEASING.md`, and `CONTRIBUTING.md`.

> Rebuilt 2026-07-05 (an earlier draft was lost, never committed), then
> settled the same day in maintainer discussion. Re-derived from the v0.1.0
> post-mortem record and a fresh audit of `release.yml` / `ci.yml` /
> `RELEASING.md` as of `2718fb3`.

Delivers the roadmap line (#27): **"Release pipeline redesign — fully
rehearsable, zero-burn-risk process."**

Standing constraint (maintainer decision 2026-06-16): **immutable releases and
the `release-tags` deletion rule stay OFF** until the redesigned pipeline has
shipped several clean releases. Re-enabling both is the *final* step of this
work (Phase 3), not a prerequisite.

---

## 1. Problem statement — the v0.1.0 post-mortem

The first real release (2026-06-16, tag `v0.1.0`) failed and permanently burned
its version number:

1. `release: published` fired; `build` passed all guards.
2. The then-existing `attach-release-assets` job ran `gh release upload` against
   the just-published release → **422**: immutable releases freeze a release at
   publish time and reject post-publish asset uploads.
3. Via `needs: [build, attach-release-assets]`, the failure blocked
   `publish-pypi` → nothing reached PyPI.
4. During recovery the release object was deleted — and GitHub's immutable
   releases **permanently reserve a tag name once used**. `v0.1.0` is
   unrecoverable; GitHub's documented remedy is "use a new version".
5. Shipped as `v0.1.1` instead (PR #22 dropped asset attachment; PR #23 bumped).

Two structural root causes:

- **R1 — The release-only path is never rehearsed.** The TestPyPI dry-run
  exercises only `build` + `publish-testpypi`; every release-only step (the
  tag==version guard body, the whole `publish-pypi` job, the `pypi` environment
  gate, the production OIDC handshake, tag-ref concurrency — 8 items total,
  enumerated in the 2026-07-05 audit) ran for the first time during the first
  irreversible real release.
- **R2 — The irreversible act happens *before* validation.** The trigger is
  `release: published`: the public tag + release object exist *first*, then the
  guards run. Every guard is therefore a tag-burning failure mode.

Additional audit findings (2026-07-05): the `testpypi` environment has **no
protection rules** — `RELEASING.md` frames its reviewer gate as optional
("approve … *if it is protected*"), and it is not, so nothing actually gates
the TestPyPI leg; post-publish verification (install checks, attestation
presence, notes-match-changelog) is entirely manual eyeballing; `pyproject.toml`
is the version source of truth (the tag is only asserted equal — a property the
new design builds on).

One constraint no design can remove: **PyPI upload is itself irreversible**
(version file names are reserved even after deletion/yank). The goal is
therefore ordering and rehearsal — by the time the pipeline reaches PyPI,
everything before it has been exercised in the same run, and everything after
it is trivially retryable.

---

## 2. Decided design (2026-07-05)

**One pipeline, one human action, fully automatic, ordered by
irreversibility.** The GitHub Release is demoted from *trigger* to *final
output*; the TestPyPI rehearsal is promoted from *separate manual dry-run* to
*inline stage of every release*.

```text
[human] merge chore(release): vX.Y.Z PR      (version + CHANGELOG entry)
[human] press workflow_dispatch              ← the only trigger; version read
  │                                            from pyproject.toml on main
  ├─ 1. guards (all rehearsable, all pre-public):
  │      • ref is default branch; checks-complete green (existing)
  │      • version is NOT a .dev version          (new — see §2.3)
  │      • version absent from PyPI & TestPyPI    (new — index JSON API)
  │      • CHANGELOG.md has a section for X.Y.Z   (new)
  ├─ 2. build (sdist+wheel, --no-sources) + wheel hygiene + isolated smoke test
  ├─ 3. publish → TestPyPI   (skip-existing: true)
  ├─ 4. verify TestPyPI: sha256 of local dist/* == digests reported by the
  │      TestPyPI JSON API; isolated install from TestPyPI; import; __version__
  ├─ 5. publish → PyPI       (skip-existing: true)   ← the irreversible step;
  │                                                     everything above it ran
  │                                                     in this same run
  ├─ 6. verify PyPI: digest match + isolated install + attestation presence
  │      via the PyPI integrity API
  ├─ 7. create GitHub Release: gh release create vX.Y.Z --target <sha>
  │      (creates tag + release atomically; notes = the CHANGELOG section)
  └─ 8. open follow-up PR: bump pyproject to X.Y.(Z+1).dev0 + `uv lock`
         (maintainer merges; CI validates it like any PR)
```

### 2.1 Why publish-PyPI-before-GitHub-Release

The GitHub Release is the only step that permanently burns a tag name (once
immutable releases return). Placed last, its failure-downstream is empty:
`gh release create` failing means PyPI is live and the fix is a retry of one
idempotent-ish step — nothing is burned, nothing is inconsistent for longer
than the retry. It also makes the announcement follow the fact: no window
where GitHub says "released" but `pip install` fails.

**Failure matrix (the "zero-burn" property):**

| Failure point | Public state | Recovery cost |
|---|---|---|
| Guards / build / smoke (1–2) | nothing public | free — fix, re-dispatch |
| TestPyPI publish/verify (3–4) | nothing that counts | free (same-version re-runs: `skip-existing` + digest check catch stale artifacts loudly) |
| PyPI publish, auth/config error (5) | nothing uploaded | free — fix publisher config, re-dispatch |
| PyPI publish, partial upload (5) | some files live | re-dispatch: `skip-existing` fills the gap; digest verify (6) confirms content |
| Verify PyPI (6) | version live on PyPI | loud failure with remediation; release/tag intentionally NOT created until resolved |
| Release creation (7) | PyPI live, no tag/release | retry step 7; nothing burned |
| Dev-bump PR (8) | release fully done | open the PR by hand; cosmetic |

No failure point strands a burned version number. Compare the audited current
pipeline, where every guard failure occurs after the release object exists.

### 2.2 Inline TestPyPI as the rehearsal mechanism

Every real release rehearses the publish machinery (artifact download, the
pypa publish action, an OIDC trusted-publishing handshake, index-side
validation) against a throwaway index **in the same run, with the same
`dist/` files** that then go to production. The never-rehearsed set shrinks to
the irreducible production-only deltas: the production trusted-publisher
registration, the `pypi` environment, and the production upload itself — all
"constant once configured correctly".

Accepted trade-offs:
- **TestPyPI availability** becomes a release dependency. Accepted: releases
  are rare and never urgent. Deliberately **no skip-TestPyPI escape flag** —
  an escape flag would re-create the unrehearsed path (R1).
- **`skip-existing: true` on both indexes** trades "duplicate upload fails
  loudly" for partial-upload recovery. Loudness is preserved where it matters
  by the **mandatory digest verification** downstream of each publish: a
  re-run after a same-version rebuild that silently kept stale index files
  fails the sha256 comparison with an explicit message.

### 2.3 Post-release dev versioning + the accidental-dispatch guard

Immediately after each release, main's version becomes `X.Y.(Z+1).dev0`
(placeholder — the next release PR sets the real number, patch or minor), via
a pipeline-opened PR (step 8; the bump requires `uv lock`, which the PR
includes and CI validates).

Paired with guard "version is not a `.dev` version", this makes main
**safe-by-default**: between releases, an accidental dispatch fails in seconds
at step 1 with nothing public. Only the deliberate act of merging a release PR
arms the pipeline. Regular PRs never touch the version — exactly two moments
in the lifecycle do (the release PR and the automated dev-bump PR).

### 2.4 Human gates

- The **`pypi` environment stays** (the trusted-publisher registration binds
  to it). Its **required reviewer stays for the first one–two releases** on
  the new pipeline as a circuit breaker, then is removed: with a solo
  maintainer, dispatcher and approver are the same person minutes apart — no
  information is added (decision 2026-07-05; revisit if maintainers multiply).
- The `testpypi` environment's documented-but-unconfigured approval is
  resolved by the redesign itself: the inline stage needs no approval, and the
  runbook stops claiming one.

### 2.5 Release notes

Generated from the `CHANGELOG.md` section for the version (guard 1 ensures it
exists). CHANGELOG stays hand-written — including the unwrapped-lines style —
so the release notes remain hand-authored in substance; the pipeline only
copies them. "Notes match changelog" stops being a manual checklist item
because it becomes true by construction.

---

## 3. Alternatives considered

- **rc-tags routed to TestPyPI as the rehearsal mechanism** (this document's
  earlier draft): superseded by the inline TestPyPI stage, which rehearses
  with the same artifact, in the same run, with zero extra ritual. **rc
  releases themselves remain natively supported** — a release PR setting
  `0.5.0rc1` rides the same pipeline as a PEP 440 pre-release (pip ignores it
  without `--pre`); useful near the 1.0 protocol freeze, not needed now.
- **Keep `release: published` trigger + mandatory pre-flight check**: shrinks
  R2's window without closing it; relies on human discipline. Rejected.
- **Draft-release flow** (validate a draft, flip to published): the flip step
  re-introduces a fallible step around the irreversible one; tag still
  precedes validation. Rejected.
- **release-please / python-semantic-release** (evaluated 2026-07-05):
  release-please's native flow is *merge release PR → it creates tag + GitHub
  Release → publish triggers off the release* — exactly the R2 ordering this
  redesign eliminates. Its actual value (auto-drafting the version bump +
  CHANGELOG from Conventional Commits) automates the one part of our process
  that is cheap, while taking over the part that is a deliberate design
  judgment for a protocol library (version semantics; the maintainer has a
  standing no-reflex-bumps rule). **Deferred**: revisit if release cadence or
  maintainer count grows; if adopted, use it in "release-PR author only" mode
  (no tagging, no release creation — those stay with this pipeline).

---

## 4. Documentation deliverables (part of implementation, not optional)

- **`RELEASING.md` — full rewrite.** The runbook shrinks to: (1) merge the
  release PR (version + CHANGELOG); (2) press Run workflow; (3) merge the
  dev-bump PR the pipeline opens. Plus: what each guard failure means and the
  recovery play per §2.1's failure matrix (replacing the current
  troubleshooting table, whose remedies assume tag surgery that immutable
  releases will forbid). One-time setup sections (trusted publishers,
  environments) stay.
- **`CONTRIBUTING.md` §Releasing — update the summary.** It already points to
  `RELEASING.md` with a three-line description of the *old* flow; replace with
  the new three lines. The split is correct and stays: CONTRIBUTING for
  contributors (never touch the version in normal PRs), RELEASING as the
  maintainer runbook.
- Both must be updated **in the same PR** as the workflow change — a runbook
  describing the previous pipeline is worse than none.

---

## 5. Implementation notes & pre-flight verifications

- **Permissions:** `contents: write` only on the release-creation job (step
  7); `contents: write` + `pull-requests: write` on the dev-bump job (step 8);
  everything else stays `contents: read` (+ `id-token: write` scoped to the
  two publish jobs, as today).
- **Verify before building:** whether the `release-tags` ruleset lets the
  Actions token create `vX.Y.Z` tags via `gh release create` (actor bypass
  may need configuring). Check the exact ruleset behavior *before* the first
  run, not during it.
- **Timeouts on every job** (`timeout-minutes`) so a hung publish is a
  diagnosis, not a zombie run; keep `cancel-in-progress: false` and a
  workflow-level concurrency group so two releases cannot interleave.
- Digest sources: index JSON APIs provide per-file sha256 (no download needed
  for the digest check); attestation presence via the PyPI integrity API
  (same check used to verify v0.1.1 manually).
- The existing `workflow_dispatch`-from-branch TestPyPI dry-run becomes
  redundant (every release includes it) but the code path costs nothing to
  keep for ad-hoc testing during Phase 1 development.

---

## 6. Phasing

- **Phase 1 — implement + first release:** new workflow + rewritten
  `RELEASING.md`/`CONTRIBUTING.md`; ship the next release on it with the
  `pypi` reviewer gate still on.
- **Phase 2 — tighten:** remove the reviewer gate (§2.4); add release-PR
  checks (CHANGELOG entry + `uv lock --check` on version-touching PRs) so
  guard 1 failures move even earlier, to PR review time.
- **Phase 3 — lock:** after several consecutive clean releases, re-enable
  immutable releases + the `release-tags` deletion rule (the deliberate final
  step). From then on, the zero-burn ordering is what makes the protections
  safe to live with.

---

## 7. Remaining open items

1. **"Several clean releases" threshold for Phase 3** — maintainer judgment
   at the time; no number pre-committed.
2. **Ruleset/actor verification** (§5) — an implementation pre-flight, not a
   design question.
