# Standard ASR Integration Acceptance Criteria (faster-whisper + user validation)

This checklist defines the acceptance criteria for adding Standard ASR support
inside the `faster-whisper` library and validating it end-to-end in
`standard_asr_user`. It is derived from the project mission/goals and the
current Standard ASR specification.

## 1) Mission & Philosophy Alignment
- [x] App-dev friendly: Standard ASR discovery + usage is zero-config and
      documented with clear steps.
- [x] ASR-dev friendly: the adapter is minimal, clean, and follows the Standard
      ASR contract without extra boilerplate.
- [x] Integration keeps faster-whisper lean; Standard ASR dependency remains
      optional.

## 2) Standard ASR Adapter in faster-whisper
- [x] A Standard ASR adapter (e.g., `StandardWhisperModel`) exists with
      Pydantic config/options, proper `properties`, and `transcribe` mapping to
      `TranscriptionResult`.
- [x] Lazy-loading is enforced; model weights are not downloaded in `__init__`.
- [x] Download policy respects `STANDARD_ASR_ALLOW_DOWNLOAD`.
- [x] Audio validation follows Standard ASR contract (`float32`, 16 kHz, mono).

## 3) Entry Points & Packaging
- [x] `standard_asr.models` entry points are declared for
      `faster-whisper/whisper` and `faster-whisper/distil-small.en`.
- [x] Entry point factories return instances whose `properties.model_id` matches
      the entry point key.
- [x] Optional extra is provided for Standard ASR dependencies.

## 4) Documentation & Proof
- [x] faster-whisper docs explain Standard ASR usage and installation.
- [x] Proof steps are documented, including `standard-asr models list` and
      `standard-asr compliance entrypoints`.
- [x] Standard ASR documentation reference is linked for compliance guidance.

## 5) Tests & Quality
- [x] Unit tests validate adapter behavior and entrypoint metadata without
      downloading models.
- [x] Code review completed for faster-whisper changes.
- [x] Tests, linter, and type checker are run for affected projects.

## 6) standard_asr_user End-to-End Validation
- [x] `standard_asr_user` is an initialized uv project with required deps.
- [x] `harvard.wav` is transcribed via Standard ASR using
      `faster-whisper/distil-small.en`.
- [x] Transcript matches expected text and result is recorded.

## 7) Traceability
- [x] `work/todo.csv` is complete with all tasks checked off.
- [x] Final criteria review performed and documented.
- [x] Any discovered Standard ASR improvements are captured in
      `work/standard_asr_improvements.md`.
