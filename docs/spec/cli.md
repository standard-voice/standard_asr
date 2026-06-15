# CLI Specification

Standard ASR provides a built-in CLI for discovery, compliance checks, and
quick transcription.

## 1. Commands

### `standard-asr list`
List all discovered models.

Flags:
- `--strict`: fail on invalid entry points during discovery (default: keep
  going, skipping invalid ones).
- `--on-conflict {warn_keep_first,replace}`: strategy for duplicate model keys
  (default: `warn_keep_first`).

### `standard-asr show <engine/model>`
Show metadata about a specific model entry point. The declared capabilities are
rendered as **canonical JSON** — the same serialization the REST
`GET /v1/.../capabilities` endpoint returns, with a derived `supported` boolean
at every node — so CLI and wire output can be compared field-for-field (spec §C
R6; the two layers share one capability model). If an engine mis-declares its
`declared_capabilities` (e.g. as a raw dict), the capabilities line reports the
problem and the rest of the metadata still renders.

Flags:
- `--strict`: fail on invalid entry points during discovery.

### `standard-asr cache [--ensure]`
Display (and optionally create, `--ensure`) the Standard ASR model cache
directory.

### `standard-asr prepare <engine/model>`
Warm up a model by loading or downloading weights. `prepare` is best-effort and
maps onto the optional `prepare()` hook (spec IC.11): an engine that does not
override the `EngineBase` default no-op is a reported no-op ("nothing to warm
up") and never transcribes, so a cloud engine is never billed for a stand-in
request. The hook MUST be a synchronous, zero-argument method; a coroutine
`prepare` (or a non-callable `prepare` attribute) is rejected as a usage error
(it would otherwise be called but never awaited and falsely reported complete).

### `standard-asr compliance entrypoints`
Validate entry points and factories (entry-point metadata + class-level
capability declarations). Flags:
- `--strict`: fail on invalid entry points at discovery time.
- `--no-instantiate`: skip instantiation attempts (avoids loading models).
- `--quiet`: suppress warnings in the output.

### `standard-asr compliance run [engine/model ...]`
Run the full compliance suite for the named models (default: every discovered
model). It runs `compliance entrypoints` and then, for each model that
constructs without arguments and declares a streaming axis, the streaming
**parameter-gating** check — so a streaming engine that bypassed the gating
template is caught here, not just at the entry-point level (delivers G.2.1's
"one command validates compliance"). An engine that requires constructor
arguments (e.g. credentials) is reported as *skipped*, not failed. Flags:
- `--strict`: fail on invalid entry points at discovery time.
- `--quiet`: suppress warnings in the output.
- `--include-bridge`: also run the sync-bridge check. This **opens a streaming
  session** and is therefore off by default — for a cloud engine that is a
  billable connection.

The streaming **event-sequence** check needs an author-recorded event stream the
CLI cannot synthesize; it remains a library API
(`standard_asr.compliance.check_event_sequence`) and `compliance run` prints a
note naming it, so a green run is never mistaken for full coverage.

### `standard-asr transcribe <engine/model> <audio>`
Transcribe an audio file and print text or JSON output. `--options` accepts a
JSON object mapping onto the portable standard set (`WireRuntimeParams`, e.g.
`'{"language": "en"}'`). The engine-specific `provider_params` escape hatch is
not constructible from untyped JSON and is rejected as a validation error. A
validation error **never echoes the submitted value** (a mis-pasted secret is
not reflected back; credential-named fields are redacted) — the same scrub the
server applies to its 422 body. In the default text mode the transcript is
printed to stdout and any `TranscriptionResult.diagnostics` (lossy-conversion /
degradation provenance) are rendered to **stderr**, so stdout stays a clean,
pipeable transcript while a degrade is never silent. `--json` prints the full
result (diagnostics included) to stdout.

### `standard-asr serve`
Launch the FastAPI server (requires `standard-asr[server]`). Flags:
- `--host` (default `127.0.0.1`), `--port` (default `8000`): bind address.
- `--log-level` (default `info`): uvicorn log level.

### `standard-asr doctor`
Read-only dependency diagnostic: enumerates installed plugins and reports numpy
1.x-vs-2.x conflicts that cannot share a process (spec §DEP.5). Exit code `1` if
a conflict is found, or if plugins are installed but the optional `packaging`
distribution is missing — conflict analysis is then unavailable and the
environment cannot be proven conflict-free; else `0` (including when no plugins
are installed, since there is nothing to analyze). Does not resolve or install
anything.

### Global Flags

- `--debug`: emit stack traces for unexpected errors. The trace is printed for
  every error path (not only the final generic handler), so a named error (e.g.
  an engine-internal failure) is debuggable too.

## 2. Output Conventions

- Human‑readable console output by default; ASCII status markers
  (`[OK]`/`[FAIL]`/`[WARN]`/`[INFO]`) so a redirected/piped stream never crashes
  on a decorative character.
- The output streams are forced to UTF‑8 when not already UTF‑8 (e.g. a Windows
  redirect defaulting to the ANSI code page), so non‑Latin transcripts print
  losslessly rather than raising `UnicodeEncodeError`. Transcript text is never
  silently replaced.
- JSON output for transcription with `--json`.
- Clear error messages on failure (stderr).
- Exit codes: `0` success, `1` runtime/transcription failures, `2` usage or
  validation errors.
