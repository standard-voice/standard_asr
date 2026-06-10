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
| `engine_id`  | `a-z`, `0-9`, `.`, `_`, `-`                          | Must start with `[a-z0-9]`; `/` is forbidden. **Upper case is rejected outright**, but a non-canonical lowercase form using `.`/`_` separators (e.g. `faster_whisper`) is *accepted and folded* to its PEP 503 routing identity (`faster-whisper`), with a normalization hint logged. The asymmetry is deliberate: distribution names on PyPI are lowercase by convention, so an upper-case engine id is treated as a mistake to fix at the source rather than silently rewritten, while the `.`/`_`↔`-` separator equivalence is a pure PEP 503 routing fold. The declared form is retained on `ModelSpec.declared_engine_id` for diagnostics. |
| `model_name` | `A-Za-z0-9`, `.`, `_`, `+`, `%`, `:`, `-`            | `/` is forbidden. Empty string signals a default model and triggers a warning. |

Multiple models per engine are encouraged. Each preset—quantised variants, multilingual/monolingual builds, device specialisations—should receive its own entry point so downstream users can request the exact behaviour they need.

### Default Models

Leaving `model_name` empty (key written as `engine_id/`) denotes the engine’s canonical default. The discovery API accepts empty names and logs a warning so authors remember to document what the default does. Supporting the default keeps today’s packages working while encouraging the new, explicit naming style.

A plugin **key** must contain the `/`: only `<engine_id>/<model_name>` and the
explicit default `<engine_id>/` are valid declaration forms. A slash-less key
(e.g. `faster-whisper` instead of `faster-whisper/`) is **not** a third valid
form — it is almost always a typo that dropped `/<model_name>`. Discovery
rejects it: `discover_models(strict=True)` (and `standard-asr compliance
entrypoints --strict`) raise, while default discovery logs a warning naming the
fix and skips the key. The trailing slash is required only on the *declaration*
side; the *lookup* helpers below accept the bare engine id as a convenience
alias for its default model.

If you publish an explicit default (`engine_id/`), the factory **must** return an
instance whose `properties.model_id` is exactly `engine_id/`. This invariant is
validated by compliance checks.

## Declaring Entry Points

```toml
[project.entry-points."standard_asr.models"]
"faster-whisper/large-v3" = "std_faster_whisper.entrypoint:create"
"faster-whisper/distil-large-v3" = "std_faster_whisper.entrypoint:create_distil_large_v3"
"faster-whisper/turbo" = "std_faster_whisper.entrypoint:create_turbo"
```

Your callable can be a function or a class constructor. Each preset selects its
model by which class it instantiates — never by passing a size name through an
init `model` field (spec IC.7). The model identity lives on the engine class so
discovery can read it without instantiating:

```python
# std_faster_whisper/entrypoint.py
from typing import Any

from .std_asr_faster_whisper import DistilLargeV3ASR, FasterWhisperASR, TurboASR


def create(**kwargs: Any) -> FasterWhisperASR:
    """Return the large-v3 multilingual preset."""

    return FasterWhisperASR(**kwargs)


def create_distil_large_v3(**kwargs: Any) -> DistilLargeV3ASR:
    """Return the distil-large-v3 preset."""

    return DistilLargeV3ASR(**kwargs)


def create_turbo(**kwargs: Any) -> TurboASR:
    """Return the large-v3-turbo preset."""

    return TurboASR(**kwargs)
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
asr = registry.create("faster-whisper/large-v3", device="cuda", compute_type="float16")
text = asr.transcribe("meeting.wav").text  # or AudioArray(samples, 16000) / (samples, 16000)
```

Helper APIs:

- `parse_entrypoint_name()` splits a key into `(engine_id, model_name)`.
- `pep503_normalize()` lets authors compute the canonical engine id.
- `ModelRegistry.keys_by_engine(engine_id)` lists all presets for a given engine.

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
 - faster-whisper/large-v3         engine=faster-whisper  model=large-v3
 - faster-whisper/distil-large-v3  engine=faster-whisper  model=distil-large-v3
 - faster-whisper/turbo            engine=faster-whisper  model=large-v3-turbo

$ standard-asr models show faster-whisper/large-v3
Model: faster-whisper/large-v3
  Engine ID   : faster-whisper
  Model name  : large-v3
  Module      : std_faster_whisper.entrypoint
  Attribute   : create
  Value       : std_faster_whisper.entrypoint:create

$ standard-asr compliance entrypoints
[OK] Entry point compliance checks passed.

$ standard-asr compliance run faster-whisper/large-v3
[OK] Entry point compliance checks passed.
[INFO] Streaming event-sequence is not run here; cover it with
       standard_asr.compliance.check_event_sequence in your tests.
[OK] Compliance run passed.
```

### Local testing in the uv workspace

Install the demo plugin and run the checks end‑to‑end:

```bash
uv run uv pip install -e cookbook/std_dummy_asr
uv run standard-asr models list
uv run standard-asr compliance entrypoints
uv run python cookbook/sample_client.py
```

The sample client selects the demo model (``dummy/echo``) explicitly and emits a
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
from standard_asr.compliance import check_entrypoints

report = check_entrypoints()
if not report.passed:
    for issue in report.issues:
        print(issue.level, issue.model, issue.message)
    raise SystemExit(1)
```

Our own compliance suite imports this helper to keep the ecosystem predictable. As the metadata contract expands (capabilities, supported locales, etc.) the checker will grow to verify those fields while keeping the API stable for you.

### The full compliance surface

`check_entrypoints()` covers entry-point metadata and class-level declarations.
The standard defines **six** compliance dimensions; the remaining checks are
also importable from `standard_asr.compliance`:

| Check | What it asserts | How to run |
| --- | --- | --- |
| `check_entrypoints` | Entry-point metadata, capability declarations, the optional `prepare()` contract | `standard-asr compliance entrypoints` / `compliance run` |
| `check_provider_params_swap_safety(engine)` | An engine rejects another engine's `provider_params` rather than silently misreading them (spec Runtime R3 / §5.4) | `standard-asr compliance run` (per zero-arg engine) |
| `check_streaming_param_gating(engine)` | A streaming engine gates an unsupported standard parameter per its strict/best_effort policy | `standard-asr compliance run` (per zero-arg streaming engine) |
| `check_recommended_wire_format(engine)` | A streaming engine's `recommended_wire_format()` is internally consistent with its declared sample rate / wire encoding | `standard-asr compliance run` (per zero-arg streaming engine) |
| `check_sync_bridge(session_factory)` | The async→sync bridge terminates without deadlock or a leaked thread | `standard-asr compliance run --include-bridge` (opens a session) |
| `check_event_sequence(events)` | A recorded streaming event stream obeys the segment/event-order contract | library API only — drive it from your own tests with recorded events |

`standard-asr compliance run` orchestrates every check except
`check_event_sequence` for you: `check_provider_params_swap_safety` for each
zero-arg engine, then `check_streaming_param_gating` and
`check_recommended_wire_format` for each streaming engine (both no-billing
probes), plus `check_sync_bridge` when opted in via `--include-bridge` (it opens
a session). `check_event_sequence` needs an author-recorded event stream the CLI
cannot synthesize, so wire it into your test suite:

```python
import pytest

from standard_asr.compliance import (
    check_entrypoints,
    check_event_sequence,
    check_provider_params_swap_safety,
    check_recommended_wire_format,
    check_streaming_param_gating,
    check_sync_bridge,
)
from my_engine import create_engine  # your zero-arg factory


def test_entrypoints_compliant() -> None:
    report = check_entrypoints()
    assert report.passed, [i.message for i in report.issues]


def test_provider_params_swap_safe() -> None:
    report = check_provider_params_swap_safety(create_engine())
    assert report.passed, [i.message for i in report.issues]


def test_streaming_gating_compliant() -> None:
    report = check_streaming_param_gating(create_engine())
    assert report.passed, [i.message for i in report.issues]


def test_recommended_wire_format_consistent() -> None:
    report = check_recommended_wire_format(create_engine())
    assert report.passed, [i.message for i in report.issues]


def test_sync_bridge_no_deadlock() -> None:
    engine = create_engine()
    fmt = ...  # an AudioFormat using one of your declared wire_encodings
    report = check_sync_bridge(lambda: engine.start_transcription(audio_format=fmt))
    assert report.passed, [i.message for i in report.issues]


def test_event_sequence_contract() -> None:
    events = [...]  # a recorded list[TranscriptionEvent] from a real session
    report = check_event_sequence(events)
    assert report.passed, [i.message for i in report.issues]
```

## Checklist for Plugin Authors

- [ ] Choose a PEP 503–friendly engine id (ideally your package name).
- [ ] List every shipped preset as `<engine_id>/<model_name>`.
- [ ] Provide a default model only when backwards compatibility demands it.
- [ ] Ensure factories accept keyword arguments for configurable options.
- [ ] Run `standard-asr compliance run` before publishing (and, for a streaming
      engine, cover `check_event_sequence` in your tests — see *The full
      compliance surface* above).

Following this guide gives downstream users a consistent discovery experience and keeps the Standard ASR catalog healthy.
