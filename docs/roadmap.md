# Roadmap

!!! warning "Pre-release"
    Standard ASR is in active development. The API may change before `v1.0.0`.
    We follow semantic versioning strictly.

## R.1: Draft Stage (MVP)

Build the working prototype and validate it end-to-end.

### Core protocol

- [x] Audio input type system (`AudioInput` discriminated union, `AudioFormat`).
- [x] Audio negotiation matrix and conversion pipeline.
- [x] Properties and sample-rate declarations.
- [x] Pydantic-based config system (`BaseConfig`, applicability mixins, `SecretStr` enforcement).
- [x] Hierarchical capability tree (`DeclaredCapabilities`, `supports()` query).
- [x] `TranscriptionResult` with constant schema.
- [x] Streaming event protocol (`partial` / `final` / `supersede` / `progress` / `done` / `error`).
- [x] Segment lifecycle and stability guarantees (`stable_until`, `final`/`closed`).
- [x] Runtime parameter gating (strict / best_effort, structured diagnostics).
- [x] Language handling (BCP-47, `auto`, candidate languages).
- [x] Structured exception hierarchy.
- [x] Compliance test suite (6 check dimensions).
- [x] PCM wire codec (`pcm_s16le`).

### Tooling

- [x] Plugin discovery via entry points (`discover_models()`).
- [x] CLI: `models list`, `models show`, `transcribe`, `compliance`, `doctor`.
- [x] FastAPI server (HTTP batch + WebSocket streaming).
- [x] SRT / VTT renderers.
- [x] Dependency-conflict doctor.

### Quality

- [x] CI: Python 3.10--3.14, numpy 1.x/2.x, macOS/Linux/Windows.
- [x] pyright strict, ruff, 100% test coverage.
- [x] Normative specification (`docs/spec/specification.md`).

### Packaging

- [x] PyPI trusted-publishing workflow.
- [ ] First PyPI release (`v0.1.0`).
- [ ] Plugin project template (cookiecutter / copier).

### Documentation

- [x] App-developer guide, engine-author guide, plugin entrypoints guide.
- [x] API reference (mkdocstrings).
- [x] Streaming deep-dive, error reference.
- [ ] Tutorial videos.

### Community

- [x] Zulip chat.
- [ ] Standard ASR Compliant badge program.
- [ ] Plugin showcase page.

## R.2: Beta Stage (ecosystem expansion)

- Stabilize and harden the CLI and server.
- Expand the official plugin set (more engines).
- Cross-language SDKs / client libraries.
- Community plugin contributions.

## R.3: Stable Release

- `v1.0.0`: stable public API with migration policy for breaking changes.
- Long-term support.
