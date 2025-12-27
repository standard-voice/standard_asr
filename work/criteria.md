# Standard ASR Acceptance Criteria (v0)

This checklist defines the minimum acceptance criteria for completing the current
implementation scope of Standard ASR. It is derived from the mission, goals, and
roadmap in `docs/mission.md`, `docs/goals.md`, `docs/misc.md`, and `docs/roadmap.md`.

## 1) Mission & Philosophy Alignment
- [ ] The interface is **application‑developer friendly**: zero‑config usage works for any
      compliant plugin and switching engines does not require code changes.
- [ ] The interface is **ASR‑developer friendly**: plugin authors can implement a compliant
      engine with minimal boilerplate and clear guidance.
- [ ] The core package stays lightweight; heavy dependencies are optional.

## 2) Standard Interface & Data Contract
- [ ] `StandardASR` defines a stable, typed interface for sync/async transcription.
- [ ] Audio input contract is documented and validated (float32, 16kHz, channel count).
- [ ] Output format is standardized via a `TranscriptionResult` model (text + optional
      segments/words + metadata + `extra`).
- [ ] Optional features (streaming, word timestamps, diarization, translation) are
      standardized via feature flags and documented.

## 3) Properties, Config, and Options
- [ ] Every engine exposes **static** `properties` with validated language tags (BCP‑47),
      device support, audio constraints, and feature flags.
- [ ] Engine initialization config is modeled with Pydantic v2 and is discoverable for UI.
- [ ] Per‑request inference options are modeled with Pydantic v2 and are discoverable for UI.
- [ ] Plugins can expose extra (non‑standard) fields without breaking the core protocol.

## 4) Plugin Discovery & Compliance
- [ ] Entrypoint discovery (`standard_asr.models`) remains stable and documented.
- [ ] Compliance helpers validate entry points **and** minimum interface metadata.
- [ ] Lazy‑loading policy is implemented and enforced via a shared helper.

## 5) Tooling & Developer Experience
- [ ] CLI supports model discovery, compliance checks, and basic transcription.
- [ ] A FastAPI server can expose any compliant plugin as a REST API (optional deps).
- [ ] Model management helpers exist (cache path, download policy, prepare/warmup).

## 6) Cookbook & Examples
- [ ] Cookbook contains a **faster‑whisper** compliant plugin that respects lazy‑load rules.
- [ ] Dummy plugin remains functional and up‑to‑date with the new protocol.
- [ ] Sample client shows discovery → create → transcribe workflow.

## 7) Documentation
- [ ] Design docs exist for: protocol, properties, config/options, results, feature flags,
      lazy‑load/download policy, CLI, and API server.
- [ ] App‑dev and ASR‑dev guides are complete and reflect current APIs.
- [ ] README and MkDocs index are updated with correct quickstart guidance.

## 8) Quality & Verification
- [ ] 100% test coverage for `standard_asr` package (pytest + coverage).
- [ ] Pyright strict passes.
- [ ] Ruff format + check pass.
- [ ] Final criteria review completed and documented.
