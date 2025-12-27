# CLI Specification

Standard ASR provides a built-in CLI for discovery, compliance checks, and
quick transcription.

## 1. Commands

### `standard-asr models list`
List all discovered models.

### `standard-asr models show <engine/model>`
Show metadata about a specific model entry point.

### `standard-asr models cache [--ensure]`
Display (and optionally create) the Standard ASR model cache directory.

### `standard-asr models prepare <engine/model>`
Warm up a model by loading or downloading weights.

### `standard-asr compliance entrypoints`
Validate entry points and factories. Use `--no-instantiate` to avoid loading
models.

### `standard-asr transcribe <engine/model> <audio>`
Transcribe a local audio file and print text or JSON output.

### `standard-asr serve`
Launch the FastAPI server.

## 2. Output Conventions

- Human‑readable console output by default.
- JSON output for transcription with `--json`.
- Clear error messages on failure.
