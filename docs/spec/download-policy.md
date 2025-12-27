# Download & Lazy‑Loading Policy

Standard ASR enforces a strict lazy‑loading policy to avoid surprise downloads
and heavy startup costs. This is critical for server environments and CI.

## 1. Environment Toggle

`STANDARD_ASR_ALLOW_DOWNLOAD` controls whether plugins are allowed to download
model weights at runtime.

- `1`, `true`, `yes` → downloads allowed
- `0`, `false`, `no` → downloads disabled
- unset → **allowed by default** (recommended for local/dev)

## 2. Expected Engine Behavior

- **Constructor must be lazy**: do not download or load weights in `__init__`.
- **Guard downloads**: check `standard_asr.runtime.allow_downloads()` before any
  download.
- **Raise clear errors**: if downloads are disabled and weights are missing,
  raise `DiscoveryError` with a clear next action.

## 3. Cache Location

Default cache directory is resolved by:

- `STANDARD_ASR_MODEL_DIR` (if set)
- otherwise `~/.cache/standard-asr` on macOS/Linux
- or `%LOCALAPPDATA%/standard-asr` on Windows

Use `standard_asr.runtime.ensure_cache_dir()` to create it.

## 4. Operational Guidance

- In production or CI, set `STANDARD_ASR_ALLOW_DOWNLOAD=0`.
- Use `standard-asr models prepare <engine/model>` to pre‑warm caches.
- For air‑gapped environments, pre‑download weights into the cache directory.
