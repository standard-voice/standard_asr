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
Warm up a model by loading or downloading weights. `prepare` is best-effort: for
an engine that declares no `prepare()` hook it is a reported no-op ("nothing to
warm up") and never transcribes, so a cloud engine is never billed for a stand-in
request.

### `standard-asr compliance entrypoints`
Validate entry points and factories. Use `--no-instantiate` to avoid loading
models.

### `standard-asr transcribe <engine/model> <audio>`
Transcribe an audio file and print text or JSON output. `--options` accepts a
JSON object mapping onto the portable `RuntimeParams` standard set (e.g.
`'{"language": "en"}'`).

### `standard-asr serve`
Launch the FastAPI server (requires `standard-asr[server]`).

### `standard-asr doctor`
Read-only dependency diagnostic: enumerates installed plugins and reports numpy
1.x-vs-2.x conflicts that cannot share a process (spec §DEP.5). Exit code `1` if
a conflict is found, else `0`. Does not resolve or install anything.

### Global Flags

- `--debug`: emit stack traces for unexpected errors.

## 2. Output Conventions

- Human‑readable console output by default.
- JSON output for transcription with `--json`.
- Clear error messages on failure (stderr).
- Exit codes: `0` success, `1` runtime/transcription failures, `2` usage or validation errors.
