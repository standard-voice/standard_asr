# Standard ASR Integration Plan (faster-whisper + user validation)

This plan follows `work/criteria.md` and the project mission/goals. All tasks
must be tracked in `work/todo.csv` and completed before final acceptance.

## 1) Scope
- Add first-class Standard ASR support inside `faster-whisper`.
- Provide entry points for `faster-whisper/whisper` and
  `faster-whisper/distil-small.en`.
- Document usage + compliance proof steps.
- Add tests that validate adapter behavior without downloads.
- Validate end-to-end transcription in `standard_asr_user` using uv.

## 2) Design & Implementation

### 2.1 Standard ASR Adapter Module (faster-whisper)
- Create a module (e.g., `faster_whisper/standard_asr.py`) that includes:
  - `FasterWhisperConfig` (Pydantic, Standard ASR config).
  - `FasterWhisperOptions` (Pydantic options mapped to `WhisperModel.transcribe`).
  - `StandardWhisperModel` (StandardASR adapter with lazy load + download policy).
  - `FasterWhisperProperties` builder to ensure `properties.model_id` matches
    entry point keys.
- Preserve lazy loading: no model download in `__init__`.
- Use Standard ASR runtime helpers for audio validation and download policy.

### 2.2 Entry Points & Packaging
- Add `standard_asr.models` entry points in `setup.py` for:
  - `faster-whisper/whisper`
  - `faster-whisper/distil-small.en`
- Add an optional dependency extra for `standard-asr`.
- Add lazy import hook in `faster_whisper/__init__.py` so
  `from faster_whisper import StandardWhisperModel` works when the extra is
  installed.

### 2.3 Documentation
- Update `faster-whisper/README.md` with a Standard ASR section:
  - Installation (`faster-whisper[standard-asr]` + `standard-asr`).
  - Discovery + usage (`standard-asr models list`, `transcribe`).
  - Compliance proof (`standard-asr compliance entrypoints`).
  - Link/mention Standard ASR compliance docs for reference.

### 2.4 Tests
- Add a unit test that monkeypatches `WhisperModel` to avoid downloads and
  validates:
  - `StandardWhisperModel.transcribe()` result mapping.
  - Entry point factory metadata (`properties.model_id`).
- Skip Standard ASR tests if `standard_asr` is not installed.

### 2.5 End-to-End Validation (standard_asr_user)
- Initialize uv project in `standard_asr_user`.
- Add editable deps for local `standard_asr` and `faster-whisper`.
- Run transcription on `harvard.wav` using
  `faster-whisper/distil-small.en` via Standard ASR CLI or script.
- Record transcript and confirm it matches expected text.

## 3) QA & Review
- Self code review for faster-whisper changes.
- Run tests/linter/type checker for affected projects.
- Update `work/standard_asr_improvements.md` if any Standard ASR gaps are found.
- Final criteria review and complete `work/todo.csv`.

## 4) Criteria Review (pre-execution)
- Criteria 1-3 map to sections 2.1 and 2.2 (adapter + entry points + optional dep).
- Criteria 4 maps to section 2.3 (docs + proof steps).
- Criteria 5 maps to sections 2.4 and 3 (tests + QA).
- Criteria 6 maps to section 2.5 (uv project + transcript validation).
- Criteria 7 maps to section 3 (traceability + final review).

If any conflicts are found between mission/goals and implementation decisions,
pause and confirm with the user.
