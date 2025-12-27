# Standard ASR Implementation Plan

This plan follows `work/criteria.md` and the project mission/goals documents. It is
intended to be executable end‑to‑end, with every item tracked in `work/todo.csv`.

## 1) Scope
Deliver a complete MVP of Standard ASR core, tooling, docs, and cookbook examples:
- Core protocol + data models
- Properties/config/options with validations
- Plugin discovery + compliance
- CLI + optional FastAPI server
- Model download policy helpers
- Cookbook (dummy + faster‑whisper)
- Comprehensive documentation
- Full tests, lint, type checking

## 2) Architecture & Modules

### 2.1 Core Protocol
- Update `StandardASR` to return a structured `TranscriptionResult`.
- Keep `transcribe_async` default via `asyncio.to_thread`.
- Add optional streaming protocol definitions (separate protocol).

### 2.2 Data Models
- `standard_asr.results`:
  - `TranscriptionResult`, `Segment`, `Word` with standard metadata + `extra`.
- `standard_asr.features`:
  - `FeatureFlag` enum and helpers.
- `standard_asr.options`:
  - `BaseTranscribeOptions` with common fields + extension mechanism.
- `standard_asr.properties`:
  - `BaseProperties` with BCP‑47 validation, audio constraints, features.

### 2.3 Runtime Helpers
- `standard_asr.runtime`:
  - `allow_downloads()` for lazy‑load policy.
  - `standard_audio_contract()` or `validate_audio()` helpers.
  - Model cache helpers (path resolution + ensure dir).

### 2.4 Discovery & Compliance
- Extend compliance to verify:
  - `properties` exists and is `BaseProperties`.
  - `config` exists and is `BaseConfig`.
  - `transcribe` callable exists.

### 2.5 CLI
- Keep existing `models` and `compliance` subcommands.
- Add:
  - `transcribe` (file or base64 input).
  - `serve` (FastAPI server).
  - `models cache` / `models prepare` for model management.

### 2.6 FastAPI Server
- Optional dependency group: `server`.
- Endpoints:
  - `GET /v1/models` for discovered models.
  - `POST /v1/transcribe` for file uploads.
  - `POST /v1/transcribe:json` for base64 JSON payloads.
- Uses `standard_asr.utils.audio_loader` for decoding.

### 2.7 Cookbook
- Update dummy plugin to new protocol.
- Implement faster‑whisper plugin with:
  - Lazy model loading.
  - Download policy guard.
  - Options mapped to faster‑whisper parameters.
  - Results mapped to Standard ASR models.

### 2.8 Documentation
- Add design specs under `docs/spec/`.
- Update app‑dev and ASR‑dev guides.
- Update README + MkDocs index.

### 2.9 Tests & Quality
- Add tests for new modules + CLI + server.
- Ensure 100% coverage for `standard_asr`.
- Run pyright + ruff + pytest before final commits.

## 3) Criteria Review (pre‑execution)
- Each section above maps 1:1 to criteria in `work/criteria.md`.
- Any item added to `work/todo.csv` must be completed and marked done.
- Non‑code items (community, forum, badge, promo) will be tracked as
  “manual/community recommended” tasks in `work/todo.csv`.
