# Project Goals

Concrete, executable goals that deliver on the [mission](mission.md).

## G.1: Establish a Universal Interface

- **G.1.1: Standardize the core interface.** Define a near-zero-dependency core
  protocol (`StandardASR`) that normalizes `transcribe` / `transcribe_async`
  (batch) and `start_transcription` (streaming).
- **G.1.2: Audio input negotiation and constant output.** A unified audio-input
  type system (`AudioInput` discriminated union) with deterministic negotiation.
  Lossy steps surface as structured diagnostics; impossible conversions fail
  loudly. Output is a constant-schema `TranscriptionResult` whose shape never
  changes with parameters.
- **G.1.3: Properties, capabilities, and config declarations.** Engines declare
  three machine-readable metadata surfaces: *Properties* (static I/O identity),
  *Capabilities* (hierarchical feature tree), and *Config* (typed, UI-renderable
  Pydantic model with secret-field marking).
- **G.1.4: Standardize optional features.** Streaming, word timestamps,
  diarization, phrase hints, and other advanced features have standard interfaces,
  standard return formats, and a fail-closed capability query (`supports()`).
- **G.1.5: Unify streaming semantics.** A single event protocol
  (`partial` / `final` / `supersede` / `progress` / `done` / `error`) with
  segment lifecycle and explicit stability guarantees covers every real-world
  streaming behavior -- from rewriting interims to append-only token streams to
  two-pass rescoring.

## G.2: Provide a Developer Toolkit

- **G.2.1: One-command compliance.** A compliance test suite that engine authors
  run with `standard-asr compliance run` to verify their implementation.
- **G.2.2: Out-of-the-box tooling.** A CLI for discovery and quick transcription,
  a FastAPI server that exposes any engine over HTTP/WebSocket, and a dependency
  conflict doctor.
- **G.2.3: Boilerplate templates.** A reference plugin template for quick starts.

## G.3: Zero-Config Operation

- **G.3.1: Dynamic config generation.** All engine parameters are Pydantic models
  with JSON Schema output and secret-field marking, so UIs can render config
  forms without instantiating the engine.
- **G.3.2: Plugin auto-discovery.** Entry-point-based discovery: install a plugin,
  and it appears in `discover_models()` with no application-side configuration.

## G.4: Extensible Plugin Ecosystem

- **G.4.1: Core and implementation separated.** The core package is a near-zero-
  dependency protocol (`numpy` + `pydantic`). Each engine lives in its own
  independently maintained, independently licensed package.
- **G.4.2: Dependency and license isolation.** Plugin architecture keeps
  conflicting dependencies and restrictive licenses contained. For hard conflicts
  (e.g. numpy 1.x vs 2.x), `standard-asr doctor` diagnoses and process isolation
  is the escape hatch.
- **G.4.3: Plugin catalog.** A public directory of compliant engines (known
  plugins, capability summaries, licenses).

## G.5: Cross-Language Wire Protocol

- **G.5.1: Wire contract as first-class spec.** The HTTP/WebSocket contract
  evolves into an independently versioned, language-neutral specification. The
  Python FastAPI server is a reference implementation, not the definition.
- **G.5.2: Two-layer isomorphism.** The in-process Python protocol and the wire
  protocol share the same capability model, result schema, and event semantics.
  Any evolution in one layer must have a corresponding expression in the other.
