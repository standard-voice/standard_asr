# Adapting an ASR Engine to Standard ASR

This guide is the end-to-end workflow for ASR developers who want to publish a
Standard ASR compliant engine. It complements the protocol specs in
`docs/spec/*` and the entrypoint rules in `docs/for_asr_dev/plugin_entrypoints.md`.

## Who Should Read This

- ASR engine authors building a Standard ASR plugin.
- Maintainers reviewing third-party engines for compliance.

## What You Must Get Right (Non-Negotiable)

1) Identity invariant
   - `properties.model_id` must equal the entrypoint key (`engine_id/model_name`).
2) Discovery must be side-effect free
   - Importing entrypoints must not download models or initialize GPU resources.
3) Audio contract
   - `transcribe()` receives `np.float32` arrays and must validate channels.
4) Options and results mapping
   - Options must coerce from dict and results must follow the protocol models.

These are enforced by `standard-asr compliance entrypoints`.

---

## 1) Choose Names and Presets

### Engine id (engine_id)
- Use your package name after PEP 503 normalization (lowercase, `-` instead of
  `.`/`_` runs).
- Allowed characters: `a-z`, `0-9`, `.`, `_`, `-` and must start with `[a-z0-9]`.

### Model name (model_name)
- Represents a preset or variant.
- Allowed characters: letters, digits, `.`, `_`, `+`, `%`, `:`, `-`.
- Empty name (`engine_id/`) is allowed only for an explicit default and should
  be documented clearly.

### Recommendation
Publish one entrypoint per preset:

- `my-engine/base`
- `my-engine/base-int8`
- `my-engine/multilingual`

---

## 2) Define Properties (Static Metadata)

All engines must expose a static `BaseProperties` instance.

```python
from standard_asr import BaseProperties
from standard_asr.features import FeatureFlag


class MyEngineProperties(BaseProperties):
    engine_id: str = "my-engine"
    model_name: str = "base"
    protocol_version: str = "0.2.0"
    supported_languages: list[str] = ["en", "zh-Hant"]
    supported_devices: list[str] = ["cpu", "cuda"]
    supported_sample_rates: list[int] = [16000]
    supported_channels: list[int] = [1]
    audio_dtype: str = "float32"
    features: set[FeatureFlag] = {FeatureFlag.WORD_TIMESTAMPS}
    description: str | None = "My Engine base preset."
```

Must-follow rules:
- `supported_languages` must be valid BCP 47 tags.
- `supported_devices` must be non-empty.
- `audio_dtype` must be `float32`.
- `model_id` (computed) must equal the entrypoint key.

---

## 3) Define a Config Model

Use `BaseConfig` to capture initialization parameters.

```python
from typing import Literal
from pydantic import Field
from standard_asr import BaseConfig


class MyEngineConfig(BaseConfig[Literal["my-engine"]]):
    engine: Literal["my-engine"] = "my-engine"
    model_path: str = Field("base", description="Model size or path.")
    device: str = Field("auto", description="cpu/cuda/auto.")
```

Keep config values serializable and explicit. This config is attached to each
engine instance.

---

## 4) Implement the StandardASR Engine

Minimal skeleton:

```python
from typing import ClassVar
import numpy as np
from numpy.typing import NDArray

from standard_asr import BaseTranscribeOptions, StandardASR, TranscriptionResult
from standard_asr.options import coerce_options
from standard_asr.runtime import validate_audio_input


class MyEngine(StandardASR):
    properties: ClassVar[MyEngineProperties] = MyEngineProperties()

    def __init__(self, model_path: str = "base", device: str = "auto") -> None:
        self.config = MyEngineConfig(engine="my-engine", model_path=model_path, device=device)
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        # Lazy import so discovery stays side-effect free.
        import my_engine_lib  # noqa: PLC0415
        self._model = my_engine_lib.load(self.config.model_path, device=self.config.device)

    def transcribe(
        self,
        audio: NDArray[np.float32],
        options: BaseTranscribeOptions | dict | None = None,
    ) -> TranscriptionResult:
        # Always capture the return value.
        audio = validate_audio_input(audio, self.properties)
        resolved = coerce_options(options, MyEngineOptions)
        self._load_model()
        # Run inference and map outputs to TranscriptionResult.
        return TranscriptionResult(text="...")
```

Key points:
- Capture `validate_audio_input(...)` return value.
- Keep heavy imports inside `_load_model` or `transcribe`.
- Keep `properties` as a class variable (static metadata).

---

## 5) Define Options Mapping

`BaseTranscribeOptions` provides common fields (language, timestamps, etc.).
Extend it only with options your engine supports.

```python
from pydantic import Field
from standard_asr import BaseTranscribeOptions


class MyEngineOptions(BaseTranscribeOptions):
    beam_size: int = Field(default=5, description="Beam size for decoding.")
    temperature: float | None = Field(default=None, description="Sampling temperature.")
```

Guidance:
- Accept options as dict or model; use `coerce_options()` to normalize.
- Ignore or warn on unsupported options; do not silently misinterpret.

---

## 6) Map Results to the Protocol Models

Standard ASR expects structured results:

- `TranscriptionResult.text` is required.
- `segments` and `words` are optional, but include them if available.
- `language` should be BCP 47 normalized.
- `metadata` can carry engine-specific info (confidence, tokens, timings).

```python
from standard_asr.results import Segment, Word, TranscriptionResult

segments = [
    Segment(start=0.0, end=2.3, text="hello world"),
]
words = [
    Word(start=0.0, end=0.5, text="hello", probability=0.98),
]
result = TranscriptionResult(
    text="hello world",
    language=resolved.language,
    segments=segments,
    words=words,
    metadata={"engine": "my-engine"},
)
```

---

## 7) Audio Contract and Validation

Standard ASR assumes `transcribe()` receives `np.float32` audio with:
- Shape `(n_samples,)` for mono
- Shape `(n_samples, n_channels)` for multi-channel

Your engine must:
- Validate channel count against `supported_channels`.
- Use `validate_audio_input()` and capture its return value.

If you accept raw files or bytes in your own API, decode them outside the
engine and pass the normalized array to `transcribe()`.

---

## 8) Respect Download and Cache Policies

If your engine downloads model weights:

```python
from standard_asr.runtime import allow_downloads, ensure_cache_dir

if not allow_downloads():
    raise RuntimeError("Model downloads disabled by STANDARD_ASR_ALLOW_DOWNLOAD.")

cache_dir = ensure_cache_dir()
```

Environment variables:
- `STANDARD_ASR_ALLOW_DOWNLOAD` (default: allowed if unset).
- `STANDARD_ASR_MODEL_DIR` for cache directory override.

---

## 9) Entrypoints and Factories

Declare entrypoints in `pyproject.toml`:

```toml
[project.entry-points."standard_asr.models"]
"my-engine/base" = "my_engine.entrypoint:create"
```

Factory functions should be light and should not trigger downloads at import:

```python
from typing import Any
from standard_asr import StandardASR
from .engine import MyEngine


def create(**kwargs: Any) -> StandardASR:
    return MyEngine(**kwargs)
```

---

## 10) Compliance and Testing

Validate locally:

```bash
standard-asr models list
standard-asr compliance entrypoints
```

CI-friendly check:

```python
from standard_asr import check_entrypoints

report = check_entrypoints()
if not report.passed:
    raise SystemExit(1)
```

Write tests for:
- Properties validity (engine_id, model_name, languages).
- Result mapping (segments and words present when available).
- Options coercion.
- Lazy loading (no heavy import at discovery time).

---

## 11) Optional Streaming Support

If your engine supports streaming, implement `StreamingASR` from
`standard_asr.streaming` and advertise the feature flags:

- `FeatureFlag.STREAMING_INPUT`
- `FeatureFlag.STREAMING_OUTPUT`

Keep streaming optional; non-streaming engines are fully compliant.

---

## 12) Packaging Checklist

- Provide a clean `pyproject.toml` with entrypoints and metadata.
- Keep optional heavy dependencies as extras (e.g., `my-engine[cuda]`).
- Include `py.typed` if you ship type hints.
- Ensure `python_requires` matches supported versions (3.10+).

---

## 13) Troubleshooting Common Failures

- Entrypoint validation error
  - Check `engine_id/model_name` format and ensure `properties.model_id` matches.
- AudioProcessingError
  - Validate dtype and channels; always capture `validate_audio_input(...)`.
- DiscoveryError
  - Import errors inside factory; keep heavy imports lazy.
- Compliance failures
  - Ensure `properties`, `config`, and `transcribe()` all exist and match protocol.

---

## Next Steps

- Start from the cookbook examples:
  - `cookbook/std_dummy_asr`
  - `cookbook/std_faster_whisper`
- Run `standard-asr compliance entrypoints` before you publish.
