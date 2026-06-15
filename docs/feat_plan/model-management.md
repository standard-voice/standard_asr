# Model Management, Locality & Network Access (Design Case)

**Status:** Design case for a *future* feature. NOT specified, NOT implemented.
This document captures the problem, stakeholders, use cases, design options, and
open questions so we can evaluate and design it in a dedicated phase. It is
deferred from the v0.1.0 CLI-UX work because it changes the **engine protocol /
spec** (new properties fields and new optional engine methods) and must be
rolled out across the core *and* every plugin at once.

Builds on and must stay consistent with:
- [`lazy-load.md`](lazy-load.md) — `STANDARD_ASR_ALLOW_DOWNLOAD`, lazy weight loading.
- `docs/spec/download-policy.md` — IC.9 download-root precedence, `prepare()` (IC.11).
- `standard_asr.runtime` — `resolve_cache_dir()`, `ensure_cache_dir()`,
  `resolve_download_root()`, `allow_downloads()`.

---

## 1. Problem statement

A user (or an app built on Standard ASR) cannot answer basic, important
questions about the models discovered on their machine:

1. **Is this model already downloaded locally, or will running it trigger a
   (potentially multi-gigabyte) download on first use?**
2. **Is this a cloud / API-backed model** (no local weights, but requires
   network, credentials, and may cost money per request) **or an on-device
   model?**
3. **How much disk does each cached model occupy, and where on disk?**
4. **How do I delete a model I no longer want** to reclaim disk?

Today the only signal is a surprise `Fetching N files` progress bar the first
time you run a HuggingFace-backed engine, and `standard-asr list` prints only
`engine=… model=…` strings. This violates two of our own principles:

- **No surprise / no-config DX** — a user should never be ambushed by a 3 GB
  download (or a billable cloud call) they did not anticipate.
- **Explicit > implicit** — locality, cost, and disk footprint are exactly the
  kind of facts the tool should surface, not hide.

This is the same mental model `ollama` nailed for local LLMs (`ollama list`
shows what's pulled and its size; `ollama pull` / `ollama rm` manage it), and
that `huggingface-cli` exposes via `scan-cache` / `delete-cache`.

---

## 2. Context — what exists today

| Capability | Exists? | Where |
|---|---|---|
| Shared cache dir resolution | ✅ | `runtime.resolve_cache_dir()` / `ensure_cache_dir()` |
| Download-root precedence (IC.9) | ✅ | `runtime.resolve_download_root()` (explicit `download_root` > `STANDARD_ASR_MODEL_DIR` > library default > shared cache) |
| Global download on/off toggle | ✅ | `STANDARD_ASR_ALLOW_DOWNLOAD` / `runtime.allow_downloads()` |
| Pre-warm / download hook | ✅ | `EngineBase.prepare()` (IC.11), surfaced as `standard-asr prepare` |
| Per-model **local/cloud** flag | ❌ | — no field on `ModelSpec` or `BaseProperties` |
| Per-model **cache presence / size** query | ❌ | — no method anywhere |
| **Delete / evict** cached weights | ❌ | — no method, no CLI |
| **Declared** download directory / backend | ⚠️ Partial | `resolve_download_root()` computes it, but the *effective* path is not introspectable and engines do not announce it |
| **Proxy / mirror** support | ⚠️ Implicit | only whatever the engine's underlying library honors (e.g. HF `HF_ENDPOINT`, `HTTPS_PROXY`); nothing standardized at the Standard ASR layer |

Weight storage itself is **delegated to each engine's library** (e.g. the
HuggingFace hub cache). The core is "the bridge, not the endpoints" — it should
not manage weights directly. The design opportunity is to define a thin
**introspection + control contract** that the bridge offers and each engine
answers, *without* the core reaching into any particular library's cache layout.

---

## 3. Stakeholders & use cases

**End user** — "What's on my disk? What will cost me a download / bandwidth /
money before I run it? Clean up what I don't need."
- `standard-asr list` shows a status column: `local ✓ (1.5 GB)` / `local (not downloaded)` / `cloud`.
- `standard-asr pull <model>` to download deliberately, with progress + the source endpoint shown.
- `standard-asr rm <model>` to reclaim disk.
- `standard-asr list --downloaded` / `--cloud` filters.

**App developer** — gate UX on availability ("Download required — 1.5 GB"),
pre-warm in the background, show progress, support an offline mode, estimate
disk before install.

**Engine author** — declare locality and where weights live, and answer
"is it cached / how big / delete it" with minimal boilerplate by reusing the
underlying library's cache (e.g. `huggingface_hub.scan_cache_dir`). Implement a
small optional surface → get the `list` status column, `pull`, and `rm` for
free, exactly like they get the CLI / server / compliance suite for free today.

---

## 4. Design dimensions & options

### 4.1 Runtime kind (local vs. cloud) — *static* metadata

- **Option A (recommended):** add an explicit field to `properties`, e.g.
  `runtime: Literal["local", "cloud"]` (naming candidates: `runtime`,
  `execution`, `locality`, `hosting`). Cheap, static, no network, read off the
  class like other properties. We already have a real mix: `mlx-audio/*` and
  `faster-whisper/*` are local; `qwen3-asr/flash` looks cloud/API-backed.
- **Option B:** derive from other signals (presence of credential config, etc.) —
  fragile and implicit; rejected.
- **Considerations:** hybrid engines (local model + cloud fallback) — do we need
  `"hybrid"`, or a richer descriptor? Does `"cloud"` imply credentials, and how
  does it relate to existing secret config?

### 4.2 Cache / availability introspection — *dynamic*

- **Option A (recommended):** an optional engine method
  `cache_status() -> CacheStatus` where `CacheStatus` carries
  `state: present | absent | partial | unknown`, `size_bytes: int | None`,
  `location: Path | None`. Default implementation returns `unknown` so engines
  that don't implement it degrade gracefully (never a false "downloaded").
  Must be **cheap**: read a directory / hub index only — no network, no weight
  load. Prefer reading from the **class** (like `show` reads ClassVar
  capabilities) so no full instantiation / credential resolution is needed; a
  cloud engine returns `state="cloud"`/`absent` with no local footprint.
- **Option B:** a registry-level helper that knows the HF hub layout — leaky,
  couples the core to one library; rejected.
- **Considerations:** `huggingface_hub.scan_cache_dir()` makes this a few lines
  for HF-backed engines; detect `partial`/corrupt; whether to cache the result;
  concurrency with an in-flight download.

### 4.3 Deletion / eviction

- **Option A:** an optional `remove_cache() -> RemovedInfo` (freed bytes,
  location); default raises `NotSupported`. CLI `rm`.
- **Considerations (this is the riskiest dimension):** destructive, so it needs
  explicit confirmation (`--yes`), a `--dry-run`, and clear reporting. HF hub
  **deduplicates blobs across models** via symlinks — deleting one model may free
  little if blobs are shared, and must never corrupt a sibling. Permissions,
  partial deletes, and "are you sure" UX all matter. Never silent.

### 4.4 Download mechanism & directory — *plugin-declared* (raised by @user)

Today `resolve_download_root()` computes the path but the **effective directory
an engine actually uses is not introspectable**, so a user cannot see "where
will this download to / where is it cached" before or after a pull.

- **Need:** each plugin should **announce** its download/cache directory and
  mechanism so the toolchain can display it, pre-create it, and report sizes.
- **Options:**
  - Expose the effective `download_root` and a `download_backend` label
    (e.g. `"huggingface_hub" | "modelscope" | "url" | "custom"`) — either on
    `properties` or via the `CacheStatus.location` from §4.2.
  - Standardize on `resolve_download_root()` as the single source of truth and
    surface it through `cache_status()` / `show`.
  - Optional `download(progress_cb)` hook for a uniform progress experience.
- **Considerations:** per-model vs. per-engine directory; making the
  `STANDARD_ASR_MODEL_DIR` override visible; offline / `local_files_only`;
  whether the core mandates a uniform on-disk layout
  (`<cache>/<engine>/<model>/`) or just *reports* whatever the engine uses
  (HF hub has its own global layout we should not fight).

### 4.5 Proxy / mirror support — China-mainland, "day 0" (raised by @user)

**Problem:** in mainland China `huggingface.co` is frequently unreachable or
very slow, so the first run of essentially every local engine
(`mlx-audio/*`, `faster-whisper/*`) can hang or fail at download. This blocks
adoption for a large user base and should be solvable from **day 0**.

**Existing levers (today, library-specific, not standardized by us):**
- **HF mirror:** `HF_ENDPOINT=https://hf-mirror.com` redirects *all* hub traffic;
  honored by `huggingface_hub` and `transformers` (so it already works for our
  HF-backed engines). `HF_HUB_ENABLE_HF_TRANSFER=1` for throughput.
- **Generic proxy:** `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` / `NO_PROXY`
  (honored by `requests` / `httpx`).
- **Alternative hub:** ModelScope hosts many of the same weights with a
  different API (relevant for some engines / the qwen3 family).

**Options:**
1. **Minimal / day-0 (recommended to land first, no spec change):** *document*
   that `HF_ENDPOINT` and `HTTPS_PROXY` pass through to HF-backed engines, and
   add a **passthrough test** proving our plugins don't strip the environment.
   This lets a mainland user `export HF_ENDPOINT=https://hf-mirror.com` today and
   have it just work — no code change, immediate relief. `standard-asr doctor`
   can detect an unreachable hub and *suggest* the mirror.
2. **Standardized (future):** a Standard ASR-level knob — e.g.
   `STANDARD_ASR_HF_ENDPOINT` / `STANDARD_ASR_PROXY` env plus an optional
   engine-config `proxy` / `endpoint` field (`SecretStr` for authenticated
   proxies) — and a documented contract that compliant plugins MUST honor it.
   This delivers the "USB-C" promise: one uniform network knob across all
   engines instead of N library-specific env vars.
3. **doctor integration:** `doctor` reports the effective endpoint and reachability;
   `list` / `pull` surface where weights will come from.

**Security considerations (hard requirements per `AGENTS.md`):** proxy
credentials are secrets → `SecretStr`, never logged; endpoint/mirror URLs are
**validated (HTTPS, no SSRF)**; a mirror sees and could tamper with traffic →
integrity matters (pin model **revision**, verify checksums where the library
supports it). A single proxy knob may not cover engines on different hubs
(HF vs. ModelScope vs. a direct URL), so the contract must allow per-backend
configuration.

> **Day-0 minimum worth doing even before the full feature:** land option 1 —
> document `HF_ENDPOINT` / `HTTPS_PROXY`, add the passthrough test, and have
> `doctor` suggest `https://hf-mirror.com` when the hub is unreachable. Everything
> else here is future design.

---

## 5. CLI surface (when built)

- `standard-asr list` → add `LOCATION` (local/cloud) + `STATUS`
  (downloaded ✓ / not downloaded / cloud / unknown) + `SIZE` columns;
  `--downloaded` / `--cloud` filters.
- `standard-asr pull <model>` → deliberate download (evolution of / alias for
  `prepare`), with progress and the source endpoint shown.
- `standard-asr rm <model>` → guarded deletion (`--yes`, `--dry-run`).
- `standard-asr show <model>` → add cache location + size + effective download
  endpoint.
- `standard-asr doctor` → proxy / mirror detection + suggestion.

---

## 6. Spec / protocol impact (why this is deferred)

- **New `properties` field(s)** (runtime kind, maybe download backend) → spec
  change + every plugin re-declares.
- **New optional engine methods** (`cache_status`, `remove_cache`, download-dir
  declaration, proxy honoring) → protocol additions + new compliance checks.
- **Coordinated rollout** across core + `std-faster-whisper` + `std-mlx-audio` +
  `std-qwen3-asr` + docs + the compliance suite. Pre-release means we *can* make
  the change without back-compat shims, but it is a deliberate, multi-repo
  design phase — not something to fold into a CLI-ergonomics pass.

---

## 7. Open questions for the design phase

1. `local | cloud` enum, or include `hybrid`, or a richer descriptor?
2. `cache_status()` as a class method (no instantiation / credentials) vs.
   instance method? How to represent a cloud engine's "no local footprint"?
3. Does the core mandate a uniform on-disk layout, or report-only what each
   engine library uses?
4. Deletion semantics for HF-shared blobs + the safety UX (`--yes`/`--dry-run`).
5. One standard proxy/endpoint knob, or per-backend (HF / ModelScope / URL)?
6. Mirror trust & integrity — revision pinning, checksum verification.
7. Exact relationship to `STANDARD_ASR_ALLOW_DOWNLOAD` (lazy-load) and IC.9
   (`resolve_download_root`) — reuse, don't duplicate.
8. Does `prepare` become `pull`, or do both coexist (warm vs. download)?
