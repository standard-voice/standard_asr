# Download & Lazy‑Loading Policy

Standard ASR enforces a strict lazy‑loading policy to avoid surprise downloads
and heavy startup costs. This is critical for server environments and CI.

## 1. Environment Toggle

`STANDARD_ASR_ALLOW_DOWNLOAD` controls whether plugins are allowed to download
model weights at runtime. The table below is the **contract**
(`standard_asr.runtime.allow_downloads()` implements it):

- `1`, `true`, `yes` → downloads allowed
- `0`, `false`, `no` → downloads disabled
- unset → **allowed by default** (recommended for local/dev)
- any other value, **including an empty string** (e.g. a `VAR=` line in
  docker-compose) → **disabled** (fail-safe: an unrecognized value must not
  silently enable downloads). The unrecognized value is **logged once** at
  `WARNING` so the cause is traceable — the engine that later raises
  `DiscoveryError` only sees the resolved boolean, not the offending text.

  (The empty-string handling here is deliberately *not* the same as the cache
  path override `STANDARD_ASR_MODEL_DIR`, where an empty value is meaningless
  and treated as unset: for this safety toggle an empty value is an
  unrecognized value and fails safe to disabled.)

## 2. Expected Engine Behavior

- **Constructor must be lazy**: do not download or load weights in `__init__`.
- **Guard downloads**: check `standard_asr.runtime.allow_downloads()` before any
  download.
- **Raise clear errors**: if downloads are disabled and weights are missing,
  raise `DiscoveryError` with a clear next action.
- **`prepare()` honours the same gate**: the optional warm-up hook (spec IC.11),
  used by `standard-asr prepare` below, materializes weights at an
  explicit, transcription-free call point. An engine that overrides it MUST
  apply the same download gate as transcription — check `allow_downloads()` and
  raise `DiscoveryError` when downloads are disabled and weights are missing —
  and MUST keep it synchronous and idempotent (never `async def`).

## 3. Cache Location

Default cache directory is resolved by `standard_asr.runtime.resolve_cache_dir()`
in this order:

1. `STANDARD_ASR_MODEL_DIR` if set (a whitespace-only value is treated as unset;
   a relative value resolves against the current working directory at call
   time, so the result is always absolute).
2. **macOS / Linux**: `$XDG_CACHE_HOME/standard-asr` when `XDG_CACHE_HOME` is set
   to an **absolute** path (a relative value is ignored per the XDG Base
   Directory spec); otherwise `~/.cache/standard-asr`. Honouring `XDG_CACHE_HOME`
   matches the wider ML cache ecosystem (HuggingFace hub, pip, uv).
3. **Windows**: `%LOCALAPPDATA%/standard-asr`. When `LOCALAPPDATA` is unset, its
   standard location `~/AppData/Local/standard-asr` is derived directly. The
   roaming `%APPDATA%` profile is **not** used — multi-gigabyte weights must not
   land in a profile that is synced across domain logins.

Use `standard_asr.runtime.ensure_cache_dir()` to create it.

## 4. Operational Guidance

- In production or CI, set `STANDARD_ASR_ALLOW_DOWNLOAD=0`.
- Use `standard-asr prepare <engine/model>` to pre‑warm caches.
- For air‑gapped environments, pre‑download weights into the cache directory.
