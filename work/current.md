# Current Repository Status (since 5ccf0b240e822db2a664416bee509e0dbd8d6265)

Date: 2025-12-27
Branch: feat/bump

This document is a detailed catch-up for maintainers. It covers all changes
since commit `5ccf0b240e822db2a664416bee509e0dbd8d6265`, including the commits
already on this branch and the latest post-review fixes that are staged for
commit. It also lists how to validate the system end-to-end.

---

## 1) Baseline and Scope

- Baseline commit: `5ccf0b240e822db2a664416bee509e0dbd8d6265`.
- Scope: all commits listed in `git log 5ccf0b..HEAD`, plus the current working
  tree changes (post-review fixes listed in Section 9).
- Primary goals in scope:
  - Implement the Standard ASR protocol and tooling per docs/spec.
  - Provide discovery + compliance for entrypoints.
  - Provide a CLI + FastAPI server.
  - Provide cookbook implementations (dummy + faster-whisper).
  - Reach 100% test coverage and strict typing.

---

## 2) Repository Architecture (Current)

Key top-level structure:

- `src/standard_asr/`
  - Core library: protocol models, options, results, properties, discovery,
    compliance, CLI, server, runtime, audio loader utilities, and helpers.
- `cookbook/`
  - Example implementations: `std_dummy_asr`, `std_faster_whisper`.
- `docs/`
  - Protocol specs, developer guides, API/CLI docs.
- `tests/`
  - Full coverage tests for core, CLI, server, discovery, compliance, utils.
- `work/`
  - Acceptance criteria, plan, and todo tracking.

---

## 3) Chronological Change Log Since Baseline

Commits after `5ccf0b240e822db2a664416bee509e0dbd8d6265` (oldest first):

1) `0d8fe2d` feat(discovery): add entrypoint registry and compliance CLI
   - Added entrypoint discovery and registry (`src/standard_asr/discovery.py`).
   - Added compliance report and checks (`src/standard_asr/compliance.py`).
   - Added initial CLI wiring for compliance (`src/standard_asr/cli.py`).

2) `34d1f4a` build: add console script entry point
   - Added console script in `pyproject.toml` for `standard-asr`.

3) `37fac5f` feat(cookbook): add dummy plugin and sample client
   - Added dummy engine plugin in `cookbook/std_dummy_asr/`.
   - Added sample client in `cookbook/sample_client.py`.

4) `67d4485` docs: add entrypoint quickstart guides
   - Added entrypoint/plug-in docs for ASR devs in `docs/for_asr_dev/`.

5) `2b844d9` refactor(config): type engine discriminator
   - Strengthened `BaseConfig` typing for engine discriminator.

6) `04db934` chore: add planning notes and update ignores
   - Added planning notes and updated `.gitignore`.

7) `4dd6eca` chore: add work plans and criteria
   - Added `work/criteria.md`, `work/plan.md`, `work/todo.csv`.

8) `8d67eb5` feat: add protocol models and runtime helpers
   - Added `TranscriptionResult`, `Segment`, `Word` models.
   - Added `BaseTranscribeOptions` and option coercion helpers.
   - Added `FeatureFlag` support.
   - Added runtime helpers (download policy, audio validation, cache paths).

9) `cb64eb7` feat: update cookbook plugins for new protocol
   - Updated dummy plugin for new protocol.
   - Added faster-whisper cookbook wrapper.

10) `0939672` test: expand coverage for new protocol
    - Added tests for protocol models, options, runtime, discovery, CLI.

11) `c91f637` docs: add protocol specs and update guides
    - Added/updated `docs/spec/*` (protocol, results, options, properties,
      features, streaming, CLI, API, download policy).
    - Updated developer guides in `docs/for_asr_dev/` and `docs/for_app_dev/`.

12) `4a723c3` test: reach 100% coverage
    - Adjusted tests to hit full statement/branch coverage at the time.

13) `c1304b4` chore: refine core typing defaults
    - Tightened defaults and typing constraints in core models.

14) `74ad99b` chore: align cookbook typing
    - Updated cookbook typing annotations for consistency.

15) `ad7776d` test: tighten server and options typing
    - Improved typing for server API and options paths.

---

## 4) Protocol and Core API (Current)

### Protocol and result model
- The Standard ASR protocol is codified in `docs/spec/protocol.md` and the
  model definitions in:
  - `src/standard_asr/results.py` (`TranscriptionResult`, `Segment`, `Word`)
  - `src/standard_asr/options.py` (`BaseTranscribeOptions`, helpers)
  - `src/standard_asr/features.py` (`FeatureFlag`)
- The protocol is currently aligned with version `0.2.0` across properties
  and cookbook implementations.

### StandardASR interface
- `src/standard_asr/asr_interface.py` defines the `StandardASR` protocol.
- The primary method signature is:
  `transcribe(audio: NDArray[np.float32], options: BaseTranscribeOptions|dict|None)`
  and returns `TranscriptionResult`.

### Properties model
- `src/standard_asr/asr_properties.py` defines `BaseProperties`.
- Validation includes:
  - `engine_id` and `model_name` format validation.
  - Non-empty `supported_devices`.
  - `audio_dtype` enforced to `float32`.
  - Language tags normalized and validated as BCP 47.
- Identity invariant (now enforced): `properties.model_id == entrypoint key`.

### Runtime helpers
- `src/standard_asr/runtime.py` adds:
  - download policy (`allow_downloads`)
  - cache paths
  - audio validation (`validate_audio_input`)

---

## 5) Discovery and Compliance (Current)

### Discovery
- `src/standard_asr/discovery.py` discovers entrypoints under
  `standard_asr.models`.
- Discovery is side-effect free: it does not import plugins during discovery.
- Model identity is parsed from entrypoint name `engine_id/model_name`.
- `validate_engine_id` and `validate_model_name` are public helpers for
  enforcing naming rules.

### Compliance
- `src/standard_asr/compliance.py` validates:
  - factory callable behavior
  - properties correctness
  - config typing
  - identity invariant: returned instance `properties.model_id` must match the
    entrypoint key

---

## 6) CLI and Server (Current)

### CLI (`standard-asr`)
- Implemented in `src/standard_asr/cli.py`.
- Key commands: `models list`, `models describe`, `transcribe`, `serve`,
  `compliance`, `cache`, `prepare`.
- Error handling:
  - Known user errors return exit code `2`.
  - Runtime/transcription failures return exit code `1`.
  - `--debug` prints traceback on unexpected errors.

### FastAPI server
- Implemented in `src/standard_asr/server.py`.
- Endpoints:
  - `GET /v1/health`
  - `GET /v1/models`
  - `POST /v1/transcribe` (multipart)
  - `POST /v1/transcribe:json` (JSON payload)
- All heavy work (decode, model init, transcribe) is run in threads via
  `asyncio.to_thread` to avoid blocking the event loop.
- The multipart endpoint now accepts bytes directly, avoiding `UploadFile`
  forward-ref issues.

---

## 7) Audio Loading and Processing (Current)

Implemented in `src/standard_asr/utils/audio_loader.py`:

- Layered decoding strategy:
  1) Stdlib `wave` for basic WAV
  2) `soundfile` if available
  3) FFmpeg subprocess fallback
- `ffprobe` channel detection now:
  - Uses correct invocation (no `-i` flag).
  - Has a timeout to avoid hangs.
- If resampling is required and `scipy` is missing, the loader falls back to
  FFmpeg and emits a warning.
- Validation functions return normalized `float32` audio and enforce contract
  constraints.

---

## 8) Cookbook Implementations (Current)

### Dummy plugin
Path: `cookbook/std_dummy_asr/`

- `dummy/echo` model for deterministic output.
- `dummy/` default preset added so identity invariant holds.
- Uses `validate_audio_input` correctly (captures returned audio).

### Faster-Whisper plugin
Path: `cookbook/std_faster_whisper/`

- Implements `faster-whisper` wrapper with:
  - Lazy model initialization.
  - Configuration model (`FasterWhisperConfig`).
  - Rich options mapping (`FasterWhisperOptions`).
  - Full `TranscriptionResult` mapping with `Segment` and `Word` data.
- Respects download policy (`allow_downloads`) and local file constraints.
- Requires `faster_whisper` installed to run.

---

## 9) Post-Review Fixes (Latest Commits on This Branch)

These are the fixes applied after the review feedback and are now committed on
this branch:

- P0: Enforce audio normalization by capturing `validate_audio_input` return
  in cookbook engines:
  - `cookbook/std_dummy_asr/src/std_dummy_asr/engine.py`
  - `cookbook/std_faster_whisper/std_asr_faster_whisper.py`
- P0: Prevent FastAPI event loop blocking by threading heavy work:
  - `src/standard_asr/server.py`
- P0: Enforce model identity invariant and add a default dummy preset:
  - `cookbook/std_dummy_asr/src/std_dummy_asr/engine.py`
  - `cookbook/std_dummy_asr/src/std_dummy_asr/entrypoint.py`
  - `src/standard_asr/compliance.py`
- P1: Improve strict discovery error aggregation:
  - `src/standard_asr/discovery.py`
- P1: Strengthen `BaseProperties` validation rules:
  - `src/standard_asr/asr_properties.py`
- P1: Add FFprobe timeout + correct invocation:
  - `src/standard_asr/utils/audio_loader.py`
- P1: Add FFmpeg fallback when `scipy` is missing:
  - `src/standard_asr/utils/audio_loader.py`
- P2: Remove emoji from library logs:
  - `src/standard_asr/utils/audio_loader.py`
  - `src/standard_asr/utils/save_utils.py`
  - `src/standard_asr/discovery.py`
- P2: Remove the UploadFile global hack by switching to byte payloads:
  - `src/standard_asr/server.py`
- Docs updated to reflect audio validation return, identity invariant, and CLI
  error/exit behavior:
  - `docs/spec/protocol.md`
  - `docs/spec/properties.md`
  - `docs/spec/cli.md`
  - `docs/for_asr_dev/plugin_entrypoints.md`
  - `docs/for_asr_dev/properties.md`
- Tests added/updated for new behavior:
  - `tests/test_audio_loader_branches.py`
  - `tests/test_cli.py`
  - `tests/test_discovery.py`
  - `tests/test_properties.py`
- Work log cleanup:
  - Removed stray work file under `src/standard_asr/` (user-created `.ini`).

---

## 10) How to Validate (Acceptance and QA)

### Static checks
Run the following from repo root:

```
uv run ruff format
uv run ruff check
uv run pyright
```

### Test suite (100% coverage)
```
uv run pytest
```

### Manual CLI checks (smoke)
```
uv run standard-asr models list
uv run standard-asr compliance
uv run standard-asr transcribe dummy/echo <path-to-audio>
```

### Server checks (optional)
```
uv run standard-asr serve --host 127.0.0.1 --port 8000
```

Then:
- `GET http://127.0.0.1:8000/v1/health`
- `GET http://127.0.0.1:8000/v1/models`
- `POST http://127.0.0.1:8000/v1/transcribe`

### Faster-whisper cookbook (optional)
Install `faster-whisper` and run:
```
uv run standard-asr transcribe faster-whisper/large-v3 <path-to-audio>
```

---

## 11) Known Pending Items (Manual/Community)

These are tracked in `work/todo.csv` and remain manual:

- Community onboarding/forum setup.
- Badge/branding assets for compliant engines.
- Promo material preparation.

---

## 12) Self-Review Notes (Planned)

After the final commits for the post-review fixes, perform a focused code
review on:

- `src/standard_asr/server.py` (async handling + error behavior)
- `src/standard_asr/utils/audio_loader.py` (fallback logic and logging)
- `src/standard_asr/compliance.py` (identity invariant)
- `cookbook/std_dummy_asr` and `cookbook/std_faster_whisper`

Then update `work/todo.csv` item 38 as completed.
