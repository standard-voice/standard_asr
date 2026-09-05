---
title: Download Policy
---

# Download and lazy-loading policy

Standard ASR keeps engine construction lazy and makes inference-artifact acquisition observable. Downloads remain allowed by default, but an operator can disable network transfers for production, CI, or an offline deployment.

## 1. Environment toggle

`STANDARD_ASR_ALLOW_DOWNLOAD` controls whether plugins are allowed to download model weights at runtime. The list below is the **contract** (`standard_asr.runtime.downloads.allow_downloads()` implements it):

- `1`, `true`, `yes`: Downloads are allowed.
- `0`, `false`, `no`: Downloads are disabled.
- unset → **allowed by default** (recommended for local/dev)
- any other value, **including an empty string** (for example, a `VAR=` line in docker-compose) → **disabled** (fail-closed: an unrecognized value must not silently enable downloads). The unrecognized value is logged at `WARNING` so the cause is traceable. The engine only sees the resolved boolean, not the offending text.

  (The empty-string handling here is deliberately *not* the same as the cache path override `STANDARD_ASR_MODEL_DIR`, where an empty value is meaningless and treated as unset: for this safety toggle an empty value is an unrecognized value and fails closed to disabled.)

## 2. Expected engine behavior

- **Keep the constructor lazy.** Do not download or load weights in `__init__`.
- **Guard each network transfer.** Check `allow_downloads()` (import it from `standard_asr.engine`) before the transfer. The toggle does not prohibit a local copy, extraction, conversion, verification, or process-local warm-up.
- **Report status without side effects.** `artifact_status()` does not contact a source service, acquire files, load weights, initialize an accelerator, or run inference. It reports `unknown` when a cheap inspection cannot establish the state.
- **Use artifact errors.** A known unavailable dependency before recognition raises `ArtifactUnavailableError`. A failed allowed acquisition raises `ArtifactAcquisitionError`. `DiscoveryError` remains limited to plugin discovery and factory loading.
- **Keep acquisition separate from warm-up.** `acquire_artifacts()` makes persistent inference artifacts available without transcription. The optional `prepare()` hook performs process-local warm-up and remains synchronous and safe to call more than once. When `prepare()` also needs persistent artifacts, it applies the same policy and error types.

`acquire_artifacts(refresh=True)` rejects the request when any mutable source is present and downloads are disabled. This policy check runs before blocker filtering: source re-resolution is a network metadata request, even when the referenced blobs are already present. When downloads are allowed, refresh widens the set that reaches the native hook rather than narrowing it: the non-ready requirements a plain acquisition already targets, plus every unblocked mutable requirement, `ready` ones included. A purely local operation can still run while downloads are disabled.

## 3. Cache location

The core offers cache-path helpers, but an engine owns its artifact layout and status checks. `resolve_download_root()` applies this precedence:

1. An explicit engine `download_root`, if set.
2. `STANDARD_ASR_MODEL_DIR`, if set (a whitespace-only value is treated as unset; a relative value resolves against the current working directory at call time, so the result is always absolute).
3. The native library's default cache: when the engine declares `has_library_default=True`, `resolve_download_root()` returns `None` and the engine forwards it, so the library keeps using its own cache.
4. The Standard ASR cache returned by `resolve_cache_dir()`.

The Standard ASR cache uses these platform defaults:

- **macOS / Linux**: `$XDG_CACHE_HOME/standard-asr` when `XDG_CACHE_HOME` is set to an **absolute** path (a relative value is ignored per the XDG Base Directory spec); otherwise `~/.cache/standard-asr`. Honoring `XDG_CACHE_HOME` matches the wider ML cache ecosystem (HuggingFace hub, pip, uv).
- **Windows**: `%LOCALAPPDATA%/standard-asr`. When `LOCALAPPDATA` is unset, its standard location `~/AppData/Local/standard-asr` is derived directly. The roaming `%APPDATA%` profile is **not** used — multi-gigabyte weights must not land in a profile that is synced across domain logins.

Use `ensure_cache_dir()` (from `standard_asr.engine`, like the other helpers on this page) to create it.

**Nothing enforces this section.** The order above and the cache locations are rules for the engine author that nothing verifies: the toolchain cannot see where a third-party library writes its files, so `standard-asr compliance run` passes a plugin that ignores `STANDARD_ASR_MODEL_DIR` entirely. Treat the order as part of your engine's promise to its users, not as a rule the toolchain checks.

## 4. Operational guidance

- In production or CI, set `STANDARD_ASR_ALLOW_DOWNLOAD=0` after acquisition.
- Use `standard-asr status <engine/model> --require-ready` to verify readiness.
- Use `standard-asr pull <engine/model>` to acquire persistent inference artifacts before deployment.
- Use `standard-asr prepare <engine/model>` only for process-local warm-up.
- For an air-gapped environment, acquire or install artifacts before the network is removed, then verify the configured engine with `status`.

See [Inference artifacts](../reference/artifacts.md) for the Python contract and structured errors.
