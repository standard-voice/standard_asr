# Plugin Entry Points

## Who Should Read This?

- **Plugin authors**: Learn how to expose your models to the Standard ASR runtime.
- **Application developers**: Understand how to discover models that have been installed.
- **Standard ASR maintainers**: Ensure the ecosystem follows the naming and compliance rules.

## Quick Summary

- New to Standard ASR? Read `docs/for_asr_dev/adapting_engine.md` first.
- Entry point group: `standard_asr.models`.
- Name format: `<engine_id>/<model_name>`.
- `engine_id` should match your distribution name after [PEP 503](https://peps.python.org/pep-0503/) normalization.
- `model_name` identifies a preset within that engine. Use an empty string for a default model *only when truly necessary*.
- Entry point value: a callable (function or class) that returns a `StandardASR` implementation.
- You can test locally with the bundled **std-dummy-asr** plugin (zero extra deps).

## Naming Rules

| Component    | Allowed characters                                  | Notes |
|--------------|------------------------------------------------------|-------|
| `engine_id`  | `a-z`, `0-9`, `.`, `_`, `-`                          | Must start with `[a-z0-9]`; `/` is forbidden. Normalization hints are logged when a name is not PEP 503 compliant. |
| `model_name` | `A-Za-z0-9`, `.`, `_`, `+`, `%`, `:`, `-`            | `/` is forbidden. Empty string signals a default model and triggers a warning. |

Multiple models per engine are encouraged. Each preset—quantised variants, multilingual/monolingual builds, device specialisations—should receive its own entry point so downstream users can request the exact behaviour they need.

### Default Models

Leaving `model_name` empty (key written as `engine_id/`) denotes the engine’s canonical default. The discovery API accepts empty names and logs a warning so authors remember to document what the default does. Supporting the default keeps today’s packages working while encouraging the new, explicit naming style.

If you publish an explicit default (`engine_id/`), the factory **must** return an
instance whose `properties.model_id` is exactly `engine_id/`. This invariant is
validated by compliance checks.

## Declaring Entry Points

```toml
[project.entry-points."standard_asr.models"]
"faster-whisper/whisper" = "std_faster_whisper.entrypoint:create"
"faster-whisper/whisper-distil" = "std_faster_whisper.entrypoint:create_distil"
"faster-whisper/" = "std_faster_whisper.entrypoint:create_default"  # optional default
```

Your callable can be a function or a class constructor:

```python
# std_faster_whisper/entrypoint.py
from typing import Any

from .runtime import FasterWhisperASR


def create(**kwargs: Any) -> FasterWhisperASR:
    """Return the multilingual preset."""

    return FasterWhisperASR(model_path="large-v3", **kwargs)
```

> **Annotate the factory with your concrete engine class, not the `StandardASR`
> protocol.** Discovery reads class-level metadata (`declared_capabilities`,
> `properties`, `provider_params_type`) *without instantiating or authenticating*
> the engine, by resolving the factory's **return annotation**
> (`ModelRegistry.engine_class`). A concrete class (`-> FasterWhisperASR`) exposes
> those `ClassVar`s; the `StandardASR` protocol does not, so annotating the
> factory `-> StandardASR` breaks instantiation-free discovery. The compliance
> suite enforces this.

The dispatcher performs thorough validation:

- Invalid names raise `EntrypointValidationError` in strict mode.
- Duplicate keys can keep the first declaration or replace with the latest, depending on `on_conflict`.
- Factories are loaded lazily; heavy dependencies stay unloaded until the model is requested.

## Discovering Models Programmatically

```python
from standard_asr import discover_models

registry = discover_models()
print(registry.names())
asr = registry.create("faster-whisper/whisper", device="cuda", compute_type="float16")
text = asr.transcribe("meeting.wav").text  # or AudioArray(samples, 16000) / (samples, 16000)
```

Helper APIs:

- `parse_entrypoint_name()` splits a key into `(engine_id, model_name)`.
- `pep503_normalize()` lets authors compute the canonical engine id.
- `ModelRegistry.by_engine(engine_id)` lists all presets for a given engine.

## Required Metadata

Your factory must return a compliant engine (typically an `EngineBase`
subclass) that exposes:

- `properties`: a `BaseProperties` instance (class attribute / `ClassVar`).
- `declared_capabilities`: a `DeclaredCapabilities` instance (`ClassVar`).
- `config`: a `BaseConfig` instance (captured at initialization).
- `transcribe(audio, params)` returning `TranscriptionResult`, where `params` is
  an optional `RuntimeParams`. Subclassing `EngineBase` gives you this
  `transcribe` template for free; you implement only `_transcribe(prepared,
  params)`.
- Engine-specific knobs live in a typed `ProviderParams` subclass declared as
  `provider_params_type` — never as extra top-level `RuntimeParams` fields
  (`RuntimeParams` is closed). See
  [`adapting_engine.md`](adapting_engine.md) for the full contract.

These are validated by `standard-asr compliance entrypoints`.

## CLI Support

Install your plugin in the same environment and use the new CLI:

```bash
$ standard-asr models list
Discovered models:
 - faster-whisper/whisper         engine=faster-whisper  model=whisper

$ standard-asr models show faster-whisper/whisper
Model: faster-whisper/whisper
  Engine ID   : faster-whisper
  Model name  : whisper
  Module      : std_faster_whisper.entrypoint
  Attribute   : create
  Value       : std_faster_whisper.entrypoint:create

$ standard-asr compliance entrypoints
✅ Entry point compliance checks passed.
```

### Local testing in the uv workspace

Install the demo plugin and run the checks end‑to‑end:

```bash
uv run uv pip install -e cookbook/std_dummy_asr
uv run standard-asr models list
uv run standard-asr compliance entrypoints
uv run python cookbook/sample_client.py
```

The sample client will pick the first discovered model (``dummy/echo``) and emit a
synthetic transcript so you can see the full discovery → instantiation → usage
cycle without heavy dependencies.

Flags of interest:

- `--strict` rejects malformed entry points immediately.
- `--no-instantiate` skips smoke-instantiation (useful when a model needs mandatory credentials at runtime).
- `--on-conflict replace` helps debug when multiple packages expose the same model id.

## Compliance Testing

The `standard_asr.compliance.check_entrypoints()` helper powers our compliance tests and the CLI. It guarantees:

1. Entry points exist (no silent typos).
2. Factories load successfully.
3. Factories that can be invoked without arguments produce an object exposing `transcribe`.
4. `properties.model_id` matches the entry point key.

Plugin authors can integrate the check into their CI:

```python
from standard_asr import check_entrypoints

report = check_entrypoints()
if not report.passed:
    for issue in report.issues:
        print(issue.level, issue.model, issue.message)
    raise SystemExit(1)
```

Our own compliance suite imports this helper to keep the ecosystem predictable. As the metadata contract expands (capabilities, supported locales, etc.) the checker will grow to verify those fields while keeping the API stable for you.

## Checklist for Plugin Authors

- [ ] Choose a PEP 503–friendly engine id (ideally your package name).
- [ ] List every shipped preset as `<engine_id>/<model_name>`.
- [ ] Provide a default model only when backwards compatibility demands it.
- [ ] Ensure factories accept keyword arguments for configurable options.
- [ ] Run `standard-asr compliance entrypoints` before publishing.

Following this guide gives downstream users a consistent discovery experience and keeps the Standard ASR catalog healthy.
