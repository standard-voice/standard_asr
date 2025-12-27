# Lazy Loading & Download Control

This document defines the lazy‑loading expectations for Standard ASR plugins and
the host runtime. The goal is to keep discovery fast, avoid surprise network
traffic, and still allow automatic weight downloads when explicitly permitted.

## Scope

- Applies to all `standard_asr.models` entry points (plugins and first‑party
  engines).
- Covers discovery, instantiation, and first inference.
- Defines the environment toggle that governs whether a plugin may download
  weights or other large artifacts at runtime.

## Current vs. Target Behaviour

| Phase            | Current behaviour (Feb 2025)                                | Target lazy behaviour (this spec)                                  |
|------------------|-------------------------------------------------------------|--------------------------------------------------------------------|
| Discovery        | Pure metadata; no imports of heavy deps.                    | Keep as‑is.                                                        |
| Instantiation    | Some plugins eagerly construct models (e.g., faster‑whisper) and may trigger downloads. | Must **not** load weights or hit the network. Only capture config. |
| First inference  | Depends on plugin; may already be loaded.                   | May load weights on first use **iff** downloads are allowed.       |
| Compliance CLI   | Default run instantiates factories, which can pull models.  | Default should skip instantiation unless user opts in.             |

Until the implementation matches the target behaviour, consider this a living
specification. Keep code and docs aligned when changes land.

## Environment Toggle

`STANDARD_ASR_ALLOW_DOWNLOAD` (string, case‑insensitive)

- `"1"`, `"true"`, `"yes"` → downloads allowed.
- `"0"`, `"false"`, `"no"` → downloads forbidden.
- Unset → **suggested default**: allowed in local/dev; set to `"0"` in API
  servers or CI where surprise downloads are undesirable.

Recommended lookup helper (to be implemented in `standard_asr.runtime` or
similar):

```python
import os

def allow_downloads() -> bool:
    value = os.getenv("STANDARD_ASR_ALLOW_DOWNLOAD")
    if value is None:
        return True  # default policy; override in servers/CI
    return value.lower() in {"1", "true", "yes"}
```

Plugins must consult this flag before initiating any download and raise a clear,
actionable exception when downloads are disallowed.

## Plugin Author Checklist

- **Lazy constructor**: Do not create heavyweight model objects in `__init__`.
  Defer to a private `_ensure_model_loaded()` called from `transcribe`.
- **Respect the toggle**: Guard any download attempt with `allow_downloads()`.
- **Clear errors**: When blocked, raise a subclass of `StandardASRError`
  (e.g., `DiscoveryError`) with instructions: “Set STANDARD_ASR_ALLOW_DOWNLOAD=1
  or pre‑populate cache at <path>”.
- **Cache‑first**: Rely on the underlying library’s cache (e.g., Hugging Face
  cache) so repeated calls stay offline when weights are present.
- **No hidden side effects**: Avoid background threads that start downloads
  implicitly; keep network activity bound to `transcribe`/`transcribe_async`.

### Example Pattern (pseudo‑code)

```python
from standard_asr.runtime import allow_downloads  # proposed helper

class FasterWhisperASR(StandardASR):
    def __init__(self, config: FasterWhisperConfig) -> None:
        self.config = config
        self._model = None

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        if not allow_downloads():
            raise DiscoveryError(
                "Weights not cached and downloads are disabled. "
                "Set STANDARD_ASR_ALLOW_DOWNLOAD=1 or prefetch weights."
            )
        self._model = WhisperModel(..., download_root=self.config.download_root)

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        self._ensure_model_loaded()
        return self._model.transcribe(audio)[0]
```

## Application Developer Guidance

- In API servers or batch environments, set `STANDARD_ASR_ALLOW_DOWNLOAD=0` and
  pre‑warm caches during deployment.
- In local notebooks or demos, leave it unset (or set to `1`) for seamless
  first‑run downloads.
- Use `standard-asr compliance entrypoints --no-instantiate` to avoid download
  attempts during health checks; opt in to `--instantiate` only when caches are
  primed.

## Operational Notes

- Air‑gapped: Pre‑fetch weights into the expected cache directory; keep the env
  flag at `0` to guarantee no network calls.
- Observability: Plugins should log a one‑line info message when a download
  starts and finishes; log a warning when blocked by the env toggle.
- Backwards compatibility: Existing plugins may be eager today. When updating
  them, keep the entry point names stable and bump plugin versions following
  semver to signal the behavioural change.

## Testing Expectations

- Unit tests for plugins should:
  - Assert no network call is made during construction.
  - Stub the env toggle to `0` and verify a clear error is raised.
  - Stub to `1` and ensure a missing cache triggers a download path (mocked).
- CLI tests should keep `--no-instantiate` as the default and add a dedicated
  test for the opt‑in `--instantiate` path.

## Summary

Lazy loading keeps Standard ASR composable and “zero‑surprise” for app
developers. The env toggle gives operators control over when downloads can
happen, while preserving the convenience of automatic weight fetching for local
exploration. Keep implementations aligned with this contract as new engines and
plugins are added.
