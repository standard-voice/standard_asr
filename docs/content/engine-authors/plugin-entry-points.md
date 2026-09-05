---
title: Plugin Entrypoints
---

# Plugin entry points

## Who should read this?

- **Plugin authors**: Learn how to expose your models to the Standard ASR runtime.
- **Application developers**: Understand how to discover models that have been installed.
- **Standard ASR maintainers**: Ensure the ecosystem follows the naming and compliance rules.

## Quick summary

- New to Standard ASR? Read `docs/content/engine-authors/adapt-an-asr-system.md` first.
- Entry point group: `standard_asr.models`.
- Name format: `<engine_id>/<model_name>`.
- `engine_id` should match your distribution name after [PEP 503](https://peps.python.org/pep-0503/) normalization.
- `model_name` identifies a preset within that engine. Use an empty string for a default model *only when truly necessary*.
- Entry point value: a callable (function or class) that returns a `StandardASR` implementation.
- You can test locally with any installed plugin (for example, [std-faster-whisper](https://github.com/standard-voice/std-faster-whisper)).

## Naming rules

| Component    | Allowed characters                                  | Notes |
|--------------|------------------------------------------------------|-------|
| `engine_id`  | `a-z`, `0-9`, `.`, `_`, `-`                          | Must start with `[a-z0-9]`; `/` is forbidden. **Upper case is rejected outright**, but a non-canonical lowercase form using `.`/`_` separators (for example, `faster_whisper`) is *accepted and folded* to its PEP 503 routing identity (`faster-whisper`), with a normalization hint logged. The asymmetry is deliberate. Distribution names on PyPI are lowercase by convention, so an upper-case engine id is a mistake to fix at the source, not something to silently rewrite. The `.`/`_`↔`-` separator equivalence is a pure PEP 503 routing fold. The declared form is retained on `ModelSpec.declared_engine_id` for diagnostics. |
| `model_name` | `A-Za-z0-9`, `.`, `_`, `+`, `%`, `:`, `-`            | Must start with `[A-Za-z0-9]`; `/` is forbidden. Empty string signals a default model and triggers a warning. |

Multiple models per engine are encouraged. Give each preset its own entry point. Presets include quantized variants, multilingual or monolingual builds, and device specializations. A separate entry point per preset lets downstream users request the exact behavior they need.

### Default models

Leaving `model_name` empty (key written as `engine_id/`) denotes the engine’s canonical default. The discovery API accepts empty names and logs a warning so authors remember to document what the default does. An explicit default is allowed but discouraged; if you publish one, document what it selects.

A plugin **key** MUST contain the `/`: only `<engine_id>/<model_name>` and the
explicit default `<engine_id>/` are valid declaration forms. A slash-less key
(for example, `faster-whisper` instead of `faster-whisper/`) is **not** a third valid
form — it is almost always a typo that dropped `/<model_name>`. Discovery
rejects it. The library call `discover_models(strict=True)` **raises**
`EntrypointValidationError`. `standard-asr compliance entrypoints
--strict-discovery` **reports** it as an `entrypoint_invalid` compliance error
and exits non-zero (a compliance check always returns a report, never raises).
Default discovery logs a warning naming the fix and skips the key. The trailing
slash is required only on the *declaration*
side; the *lookup* helpers below accept the bare engine id as a convenience
alias for its default model.

If you publish an explicit default (`engine_id/`), the factory MUST return an
instance whose `properties.model_id` is exactly `engine_id/`. Compliance checks
this invariant (`properties_key_mismatch`).

An `engine_id` MUST be unique across installed distributions: two
distributions that provide the same id (even through PEP 503 folding, such as
`my_engine` and `my-engine`) are an identity collision, reported as the
compliance error `engine_id_collision` even on a default run.

## Declaring entry points

```toml
[project.entry-points."standard_asr.models"]
"faster-whisper/large-v3" = "std_faster_whisper.entrypoint:create_large_v3"
"faster-whisper/distil-large-v3" = "std_faster_whisper.entrypoint:create_distil_large_v3"
"faster-whisper/large-v3-turbo" = "std_faster_whisper.entrypoint:create_turbo"
```

Your callable can be a function or a class constructor. Each preset selects its
model by which class it instantiates — never by passing a size name through an
init `model` field (spec IC.7). The model identity lives on the engine class so
discovery can read it without instantiating:

```python
# std_faster_whisper/entrypoint.py
from typing import Any

from .engine import DistilLargeV3ASR, LargeV3ASR, TurboASR


def create_large_v3(**kwargs: Any) -> LargeV3ASR:
    """Return the large-v3 multilingual preset."""

    return LargeV3ASR(**kwargs)


def create_distil_large_v3(**kwargs: Any) -> DistilLargeV3ASR:
    """Return the distil-large-v3 preset."""

    return DistilLargeV3ASR(**kwargs)


def create_turbo(**kwargs: Any) -> TurboASR:
    """Return the large-v3-turbo preset."""

    return TurboASR(**kwargs)
```

> **Annotate the factory with your concrete engine class, not the `StandardASR`
> protocol.** Discovery reads class-level metadata (`declared_capabilities`,
> `declared_metadata`, `properties`, `provider_params_type`) *without instantiating or authenticating*
> the engine, by resolving the factory's **return annotation**
> (`ModelRegistry.engine_class`). A concrete class (`-> FasterWhisperASR`) exposes
> those `ClassVar`s; the `StandardASR` protocol does not, so annotating the
> factory `-> StandardASR` breaks instantiation-free discovery. This applies to
> a **factory function**; an entry point that is the engine class itself needs
> no annotation, because the class is returned directly. Compliance checks the
> outcome either way, reporting `class_metadata_unreadable` when neither form
> resolves. The factory must also return an instance of **exactly** its
> annotated class: discovery, `show`, and the server's per-model endpoints
> describe the annotated class, while the returned class is what actually
> runs, so
> `ModelRegistry.create()` refuses any other returned class
> (`EngineContractError`) and compliance reports it as
> `factory_return_class_mismatch`. A factory that picks between engine
> subclasses needs one entry point, with one annotated class, per preset.

Discovery validates each declaration:

- Invalid names raise `EntrypointValidationError` in strict mode.
- Duplicate keys can keep the first declaration or replace with the latest, depending on `on_conflict`.
- Factories are loaded lazily; heavy dependencies stay unloaded until the model is requested.

## Discovering models programmatically

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

## Required metadata

Your factory MUST return a compliant engine (typically an `EngineBase`
subclass) that exposes:

- `properties`: a `BaseProperties` instance (class attribute / `ClassVar`).
- `declared_capabilities`: a `DeclaredCapabilities` instance (`ClassVar`).
- `declared_metadata`: a `DeclaredEngineMetadata` instance (`ClassVar`). Its
  required `artifacts` section must be authored by the
  plugin class hierarchy, not inherited from the core fail-loud placeholder.
  Additional producer sections use the
  `x_<vendor>_<name>` namespace. A future protocol version can add standard
  sections without the `x_` prefix, which older readers preserve.
- `config`: a `BaseConfig` instance (captured at initialization).
- `transcribe(audio, params)` returning `TranscriptionResult`, where `params` is
  an optional `RuntimeParams`. Subclassing `EngineBase` gives you this
  `transcribe` template for free; you implement only `_transcribe(prepared,
  params)`.
- Engine-specific parameters live in a typed `ProviderParams` subclass declared as
  `provider_params_type` — never as extra top-level `RuntimeParams` fields
  (`RuntimeParams` is closed). See
  [`adapt-an-asr-system.md`](./adapt-an-asr-system.md) for the full contract.

These are validated by `standard-asr compliance entrypoints`.

## CLI support

Install your plugin in the same environment and use the CLI. The
transcript below was captured live against
[std-faster-whisper](https://github.com/standard-voice/std-faster-whisper);
nested JSON blocks are abridged with `...` — run the commands yourself for
the full output (exact values depend on the plugin version):

```bash
$ standard-asr list
Discovered models:
 - faster-whisper/base             engine=faster-whisper  model=base
 - faster-whisper/distil-large-v3  engine=faster-whisper  model=distil-large-v3
 - faster-whisper/large-v3         engine=faster-whisper  model=large-v3
 - faster-whisper/large-v3-turbo   engine=faster-whisper  model=large-v3-turbo
 - faster-whisper/medium           engine=faster-whisper  model=medium
 - faster-whisper/small            engine=faster-whisper  model=small
 - faster-whisper/tiny             engine=faster-whisper  model=tiny

$ standard-asr show faster-whisper/large-v3
Model: faster-whisper/large-v3
  Engine ID   : faster-whisper
  Model name  : large-v3
  Module      : std_faster_whisper.entrypoint
  Attribute   : create_large_v3
  Value       : std_faster_whisper.entrypoint:create_large_v3
  Capabilities:
    {
      "batch": {
        "diarization": { ... },
        "guidance": {
          "phrase_hints": {
            "constraints": {
              "max_chars_per_term": 40,
              "max_terms": 50,
              "max_words_per_term": null
            },
            "supported": true
          },
          "prompt": { ... },
          "supported": true
        },
        "language": { ... },
        "supported": true,
        "word_timestamps": {
          "granularities": [
            "word",
            "segment"
          ],
          "supported": true
        }
      },
      "self_resamples": {
        "supported": false
      },
      "streaming": { ... },
      "streaming_input": {
        "supported": true
      },
      "streaming_output": {
        "supported": true
      }
    }
  Config schema:
    {
      "additionalProperties": false,
      "description": "Init configuration for the faster-whisper engine. ...",
      "properties": {
        "compute_type": { ... },
        "default_language": { ... },
        "device": { ... },
        "hf_token": {
          "anyOf": [
            {
              "format": "password",
              "type": "string",
              "writeOnly": true
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Hugging Face access token for gated/private model repos (secret).",
          "format": "password",
          "secret": true,
          "title": "Hf Token",
          "writeOnly": true
        },
        ...
      },
      "title": "FasterWhisperConfig",
      "type": "object"
    }

$ standard-asr compliance entrypoints
[OK] Entry point compliance checks passed.

$ standard-asr compliance run faster-whisper/large-v3
[OK] Entry point compliance checks passed.
[INFO] Two checks are not run here (each needs recorded data the CLI cannot synthesize): check_event_sequence for a streaming engine's event stream, and check_transcription_result for a batch result. Cover them with standard_asr.compliance in your tests (see docs/content/engine-authors/plugin-entry-points.md).
[OK] Compliance run passed.
```

### Local testing with a plugin

Install a plugin (for example,
[std-faster-whisper](https://github.com/standard-voice/std-faster-whisper)) and
run the checks end‑to‑end:

```bash
pip install "std-faster-whisper @ git+https://github.com/standard-voice/std-faster-whisper.git"
standard-asr list
standard-asr compliance entrypoints
```

Flags of interest:

- `--strict-discovery` reports malformed entry points as `entrypoint_invalid`
  errors (non-zero exit; the report still covers the valid engines).
- `--no-instantiate` skips smoke-instantiation. A missing credential already
  downgrades to a graceful skip (`factory_requires_config`); use this flag to
  avoid instantiation cost or side effects entirely.
- `--on-conflict replace` helps debug when multiple packages expose the same model id.

## Compliance testing

The `standard_asr.compliance.check_entrypoints()` helper powers the compliance suite and the CLI. It guarantees:

1. Entry points exist (no silent typos).
2. Factories load successfully.
3. Factories that can be invoked without arguments produce an object exposing `transcribe`.
4. `properties.model_id` matches the entry point key.
5. The engine authors valid declared metadata and the required
   artifact methods without executing status or acquisition.

Plugin authors can integrate the check into their CI:

```python
from standard_asr.compliance import check_entrypoints

report = check_entrypoints()
if not report.passed:
    for issue in report.issues:
        print(issue.level, issue.model, issue.message)
    raise SystemExit(1)
```

The Standard ASR compliance suite imports this helper to keep the ecosystem predictable. The checker already verifies capability declarations alongside entry-point metadata (see the full surface below), and grows additively as the metadata contract expands (supported locales, etc.) while keeping the API stable for you.

### The full compliance surface

`check_entrypoints()` covers entry-point metadata and class-level declarations.
The standard defines **seven** compliance dimensions; the remaining checks are
also importable from `standard_asr.compliance`:

| Check | What it asserts | How to run |
| --- | --- | --- |
| `check_entrypoints` | Entry-point metadata, capability and declared-metadata declarations, the artifact methods, and the optional `prepare()` contract | `standard-asr compliance entrypoints` / `compliance run` |
| `check_provider_params_swap_safety(engine)` | An engine rejects another engine's `provider_params` rather than silently misreading them (spec Runtime R3 / §5.4) | `standard-asr compliance run` (per zero-arg engine) |
| `check_streaming_param_gating(engine)` | A streaming engine gates an unsupported standard parameter per its strict/best_effort policy | `standard-asr compliance run` (per zero-arg streaming engine) |
| `check_recommended_wire_format(engine)` | `recommended_wire_format()` returns `AudioFormat \| None` and any returned format passes the engine's own session-establishment rule — the member is unconditional (spec §3.1: Properties-pure, capability-blind), so this holds for **every** engine, batch-only included | `standard-asr compliance run` (per zero-arg engine, inside the entrypoint instance checks) |
| `check_sync_bridge(session_factory)` | The async→sync bridge terminates without deadlock or a leaked thread | `standard-asr compliance run --include-bridge` (opens a session) |
| `check_event_sequence(events)` | A recorded streaming event stream obeys the segment/event-order contract | library API only — drive it from your own tests with recorded events |
| `check_transcription_result(result, capabilities=...)` | A recorded batch result carries no speaker labels beyond the declared `batch.diarization` capability (code `result_exceeds_diarization`) | library API only — drive it from your own tests with a recorded result |

`standard-asr compliance run` orchestrates every check except
`check_event_sequence` and `check_transcription_result` for you. It runs the
entrypoint instance checks (including the wire-format round-trip) and
`check_provider_params_swap_safety` for each zero-arg engine, then
`check_streaming_param_gating` for each streaming engine. These probes are
designed to fail at the standard gate, but a noncompliant engine can enter its
real pipeline, load artifacts, connect to a service, or incur a charge. Use
staging credentials or `--no-instantiate` when that risk matters. The command
also runs `check_sync_bridge` when you opt in via `--include-bridge` (it opens a
session). `check_event_sequence` needs an author-recorded event stream, and
`check_transcription_result` an author-recorded batch result. The CLI cannot
synthesize these, so wire them into your test suite:

```python
import pytest

from standard_asr.compliance import (
    check_entrypoints,
    check_event_sequence,
    check_provider_params_swap_safety,
    check_recommended_wire_format,
    check_streaming_param_gating,
    check_sync_bridge,
    check_transcription_result,
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


def test_batch_result_within_capabilities() -> None:
    engine = create_engine()
    result = ...  # a recorded TranscriptionResult from a real transcribe() call
    report = check_transcription_result(result, capabilities=engine.declared_capabilities)
    assert report.passed, [i.message for i in report.issues]
```

## Checklist for plugin authors

- [ ] Choose a PEP 503–friendly engine id (ideally your package name).
- [ ] List every shipped preset as `<engine_id>/<model_name>`.
- [ ] Skip the explicit default (`engine_id/`) unless you need one; if you
      publish it, document what it selects.
- [ ] Ensure factories accept keyword arguments for configurable options.
- [ ] Run `standard-asr compliance run` before publishing (and cover the
      recorded-data checks in your tests: `check_event_sequence` for a
      streaming engine, `check_transcription_result` for a batch engine — see
      *The full compliance surface* above).

Following this guide gives downstream users a consistent discovery experience and keeps the Standard ASR catalog healthy.
