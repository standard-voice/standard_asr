# Standard ASR Integration Acceptance Criteria (faster-whisper + user validation)

This checklist defines the acceptance criteria for adding Standard ASR support
inside the `faster-whisper` library and validating it end-to-end in
`standard_asr_user`. It is derived from the project mission/goals and the
current Standard ASR specification.

## 1) Mission & Philosophy Alignment
- [ ] App-dev friendly: Standard ASR discovery + usage is zero-config and
      documented with clear steps.
- [ ] ASR-dev friendly: the adapter is minimal, clean, and follows the Standard
      ASR contract without extra boilerplate.
- [ ] Integration keeps faster-whisper lean; Standard ASR dependency remains
      optional.

## 2) Standard ASR Adapter in faster-whisper
- [ ] A Standard ASR adapter (e.g., `StandardWhisperModel`) exists with
      Pydantic config/options, proper `properties`, and `transcribe` mapping to
      `TranscriptionResult`.
- [ ] Lazy-loading is enforced; model weights are not downloaded in `__init__`.
- [ ] Download policy respects `STANDARD_ASR_ALLOW_DOWNLOAD`.
- [ ] Audio validation follows Standard ASR contract (`float32`, 16 kHz, mono).

## 3) Entry Points & Packaging
- [ ] `standard_asr.models` entry points are declared for
      `faster-whisper/whisper` and `faster-whisper/distil-small.en`.
- [ ] Entry point factories return instances whose `properties.model_id` matches
      the entry point key.
- [ ] Optional extra is provided for Standard ASR dependencies.

## 4) Documentation & Proof
- [ ] faster-whisper docs explain Standard ASR usage and installation.
- [ ] Proof steps are documented, including `standard-asr models list` and
      `standard-asr compliance entrypoints`.
- [ ] Standard ASR documentation reference is linked for compliance guidance.

## 5) Tests & Quality
- [ ] Unit tests validate adapter behavior and entrypoint metadata without
      downloading models.
- [ ] Code review completed for faster-whisper changes.
- [ ] Tests, linter, and type checker are run for affected projects.

## 6) standard_asr_user End-to-End Validation
- [ ] `standard_asr_user` is an initialized uv project with required deps.
- [ ] `harvard.wav` is transcribed via Standard ASR using
      `faster-whisper/distil-small.en`.
- [ ] Transcript matches expected text and result is recorded.

## 7) Traceability
- [ ] `work/todo.csv` is complete with all tasks checked off.
- [ ] Final criteria review performed and documented.
- [ ] Any discovered Standard ASR improvements are captured in
      `work/standard_asr_improvements.md`.
