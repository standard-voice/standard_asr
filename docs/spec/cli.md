# CLI specification

Standard ASR provides a built-in CLI for discovery, compliance checks, and
quick transcription.

## 1. Commands

### `standard-asr list`
List all discovered models.

Flags:
- `--strict-discovery`: fail on invalid plugin entry points during discovery
  (default: keep going, skipping invalid ones). Deliberately NOT named
  `--strict`: bare `strict` is the engine's strict/best_effort *parameter-gating*
  policy (an init-config field, `--set strict=...`), a different setting.
- `--on-conflict {warn_keep_first,replace}`: strategy for duplicate model keys
  (default: `warn_keep_first`).

### `standard-asr show <engine/model>`
Show metadata about a specific model entry point. The output has three sections:
identity (engine/model, module, attribute, entry-point value), the declared
**capabilities**, and the init-**config schema**.

If the model is registered but its plugin cannot be imported, `show` prints
everything it could read plus a sanitized `Capabilities: <unavailable: ...>` line
and exits **1** (the installation is broken; the caller's key was fine, so it is
neither 0 nor 2).

The declared capabilities are rendered as **canonical JSON** — the same
serialization the REST `GET /v1/capabilities/...` endpoint returns, with a
derived `supported` boolean at every node. CLI and wire output can therefore be
compared field-for-field (spec §C R6; the two layers share one capability model).
If an engine mis-declares its `declared_capabilities` (for example, as a raw dict), the
capabilities line reports the problem and the rest of the metadata still renders.

The config schema is the JSON Schema of the engine's class-level `config_type` —
the same schema the REST `GET /v1/config-schema/...` endpoint returns — read from
the engine **class** without constructing it, so an author (or a settings UI) can
see the init fields (`device`, `compute_type`, credentials) before an engine that
needs those very values is built. An engine that declares no `config_type` prints
an explicit *no init config* line rather than nothing, so an omission is never
mistaken for an empty schema. If the engine class loads but its `config_type` is
invalid — not a `BaseConfig` subclass, or un-renderable as JSON Schema — the
config-schema line reports it *unavailable* (as the capabilities line does) and
the rest of the metadata still renders. An engine class that cannot be loaded at
all is reported once, at the capabilities line, and `show` exits **1** before the
config-schema section: the schema needs that same class, so a second unavailable
line would only restate the fault already given.

Flags:
- `--strict-discovery`: fail on invalid plugin entry points during discovery.

### `standard-asr cache [--ensure]`
Display (and optionally create, `--ensure`) the Standard ASR model cache
directory.

### `standard-asr prepare <engine/model>`
Flags:
- `--strict-discovery`: fail on invalid plugin entry points during discovery
  (the same flag as `list` / `show` / `transcribe` / `compliance`; `serve`
  deliberately has no discovery flags -- the server always discovers leniently
  so one broken co-installed plugin cannot take every other engine's endpoint
  down with it). On this command the name matters doubly: `--set strict=...`
  configures the engine's parameter-gating policy on the same command line.

Warm up a model by loading or downloading weights. `prepare` is best-effort and
maps onto the optional `prepare()` hook (spec IC.11): an engine that does not
override the `EngineBase` default no-op is a reported no-op ("nothing to warm
up") and never transcribes, so a cloud engine is never billed for a stand-in
request. The hook MUST be a synchronous, zero-argument method; a coroutine
`prepare` (or a non-callable / parameter-requiring `prepare` attribute) is
rejected as an ENGINE fault — `EngineContractError`, exit 1 — because no flag
or env var the invoker controls can fix a declaration (it would otherwise be
called but never awaited and falsely reported complete). The attribute LOOKUP
is engine code too: `prepare` may be a property, so a descriptor that raises is
classified at the same seam as the call itself.

Engine **init-config** flags (also on `transcribe`):
- `--config JSON`: the engine's init configuration as a JSON object, for example,
  `--config '{"device": "cpu"}'`. This is the same configuration otherwise
  supplied via `STANDARD_ASR_<ENGINE>__<FIELD>` env vars, now discoverable from
  `--help`. Run `standard-asr show <engine/model>` to see the config schema.
- `--set KEY=VALUE` (repeatable): set one init-config field, for example,
  `--set device=cpu --set compute_type=int8`.
- **Precedence / merge:** `--config` supplies the base object; each `--set` then
  overrides or adds a field, so `--set` **wins** over `--config` for the same key.
  `--set` values are strings — the engine's pydantic config coerces them (`"5"` →
  `5`). Scalars behave exactly like the env-var path; composite fields do
  **not**: the env-var path JSON-decodes a composite value (for example,
  `'["en","zh"]'` for a `list[str]` field) before validation, while `--set`
  passes the raw string through, so a composite `--set` value fails validation
  loudly — use `--config` for composite fields. For secrets (`api_key`, tokens) prefer the
  `STANDARD_ASR_<ENGINE>__<FIELD>` env vars — command-line values are visible in
  shell history. These flags carry **init config** (how the engine is
  constructed), distinct from `--options` on `transcribe` (per-request runtime
  params).

### `standard-asr compliance entrypoints`
Validate entry points and factories: entry-point metadata, class-level
capability declarations, and — by default — instantiation of each zero-arg
factory to verify the instance surface. Instantiation includes one
**behavioral probe**: an engine declaring no streaming axis has
`start_transcription()` called once with no arguments and MUST raise
`UnsupportedFeatureError` (a compliant engine refuses at the capability gate
before constructing anything; a returned session is never entered, but a
non-compliant implementation may still run arbitrary author code in the
method body). Flags:
- `--strict-discovery`: fail on invalid plugin entry points at discovery time.
- `--no-instantiate`: skip instantiation attempts (avoids loading models —
  and skips the batch-only refusal probe with them).
- `--quiet`: suppress warnings in the output.

### `standard-asr compliance run [engine/model ...]`
Run the full compliance suite for the named models (default: every discovered
model). It runs `compliance entrypoints` and then, for each model that
constructs without arguments and declares a streaming axis, the streaming
**parameter-gating** check — so a streaming engine that bypassed the gating
template is caught here, not just at the entry-point level (delivers G.2.1's
"one command validates compliance"). An engine that requires constructor
arguments (for example, credentials) is reported as *skipped*, not failed — the same
verdict whether the requirement shows up in the factory signature or as a
`ConfigurationRequiredError` from a zero-arg factory whose credential is
absent from the environment (`BaseConfig.from_env` raises that subtype
automatically; one run never issues two contradictory verdicts for one
engine). Any **other** `ConfigError` — an invalid supplied value, an
inconsistent declaration — is a defect and **fails**: skipping it would let a
broken plugin read as green-with-warning.

**Per-model containment (normative)**: no single model's fault may deny the
others their verdict — that aggregate IS the one-command guarantee. A
factory raising anything (`RuntimeError` from an SDK that failed to
initialize, `OSError` on an unreadable model directory, a plugin's own
exception type) is reported for THAT model as
`engine_construction_failed` and the run continues; a check implementation
that itself falls over is reported as `compliance_check_crashed`, kept
distinct so an author can tell "your plugin broke" from "the suite broke
on your plugin". Both fail the run's exit code. `KeyboardInterrupt` and
`SystemExit` are explicitly NOT contained: they are the operator's own
control flow.

**Probe honesty**: the default run's behavioral checks call public engine
entry points with deliberately rejectable inputs. The provider-params
swap-safety check invokes `transcribe()` with a one-sample silent probe and a
foreign engine's `provider_params`; the streaming gating check invokes
`start_transcription()` with an unsupported parameter (the session is
constructed but never entered, so the standard layer opens no wire
connection — and a best_effort output-only engine is reported as an
inconclusive skip rather than probed through real audio); the batch-only
refusal probe (part of the entrypoint checks above) invokes a no-argument
`start_transcription()` on engines declaring no streaming axis and requires
`UnsupportedFeatureError`. A **compliant**
engine rejects these at the gate, before any decode, connection, or
inference. A **non-compliant** engine — the very thing the probes exist to
catch — may execute its real pipeline on the probe (model load; for a cloud
engine, a billable call), or run arbitrary method-body code before the
missing refusal. Run compliance against staging credentials if that
risk matters to you.

Flags:
- `--strict-discovery`: fail on invalid plugin entry points at discovery time.
- `--quiet`: suppress warnings in the output.
- `--include-bridge`: also run the sync-bridge check. This **opens a streaming
  session** and is therefore off by default — for a cloud engine that is a
  billable connection.
- `--bridge-timeout SECONDS` (default `5.0`): timeout granted to each phase
  of the sync-bridge check -- session establishment, then the bridged drive
  (open, end-of-audio, drain, close; it also caps each bridged lifecycle
  call). Only meaningful
  with `--include-bridge` -- passing it alone is a usage error (exit 2), never
  a silent no-op. The check's remediation advice ("re-run with a larger
  timeout") is actionable through this flag; library callers pass
  `check_sync_bridge(..., timeout=...)` (finite, `> 0`; validated). An engine
  that refuses session establishment as unsupported (for example, output-only, no
  `streaming_input`) is reported as a passing `sync_bridge_not_applicable`
  warning in the report set -- structured and `--quiet`-respecting; there is
  deliberately no CLI-side pre-gate, so machine consumers of the reports see
  the verdict too. The flag's value is granted to each check phase
  (establishment, then the bridged drive) independently.

Two checks stay library-only because each needs author-recorded data the CLI
cannot synthesize: the streaming **event-sequence** check
(`standard_asr.compliance.check_event_sequence`, a recorded event stream) and the
**transcription-result** check (`standard_asr.compliance.check_transcription_result`,
a recorded batch result). `compliance run` prints a note naming **both** —
whatever the engine's shape, so a batch-only engine is told about the result
check too — so a green run is never mistaken for full coverage.

### `standard-asr transcribe <engine/model> <audio>`
Flags:
- `--strict-discovery`: fail on invalid plugin entry points during discovery
  (the same flag as `list` / `show` / `prepare` / `compliance`). Deliberately
  NOT named `--strict`: this command also carries `--set strict=...` -- the
  engine's strict/best_effort parameter-gating policy -- and one name meaning
  two policies invited silent misconfiguration.

Transcribe an audio file and print text or JSON output. `--options` accepts a
JSON object mapping onto the portable standard set (`WireRuntimeParams`, for example,
`'{"language": "en"}'`). The engine-specific `provider_params` escape hatch is
not constructible from untyped JSON and is rejected as a validation error. A
validation error **never echoes the submitted value** (a mis-pasted secret is
not reflected back; credential-named fields are redacted) — the same scrub the
server applies to its 422 body.

`--config JSON` and `--set KEY=VALUE` supply the engine's **init config** (how
the engine is constructed — `device`, `compute_type`, credentials), with the same
precedence and coercion as on `prepare` above (`--set` overrides `--config`;
`--set` values are strings the config coerces). They are distinct from
`--options`, which carries per-request runtime params.

In the default text mode the transcript is printed to stdout. Any
`TranscriptionResult.diagnostics` (lossy-conversion / degradation provenance)
are rendered to **stderr**, so stdout stays a clean, pipeable transcript while a
degrade is never silent. `--json` prints the full
result (diagnostics included) to stdout.

### `standard-asr serve`
Launch the FastAPI server (requires `standard-asr[server]`). Flags:
- `--host` (default `127.0.0.1`), `--port` (default `8000`): bind address.
- `--log-level` (default `info`): uvicorn log level.

### `standard-asr doctor`
Read-only dependency diagnostic: enumerates installed plugins and reports numpy
1.x-vs-2.x conflicts that cannot share a process (spec §DEP.5). Range
satisfiability is three-state: a conflict is asserted only on an exact
`UNSAT` from `packaging`'s specifier algebra
(`SpecifierSet.is_unsatisfiable()`, `packaging >= 26.1`); on an older
`packaging` the fallback witness search can prove satisfiability but never
emptiness, so undecidable relations are disclosed as
analysis-unavailable — never convicted, never silently passed. Exit code `1`
if a conflict is found, or if analysis is unavailable/incomplete with plugins
installed (the optional `packaging` distribution is missing, or one or more
relations were undecidable) — the environment cannot then be proven
conflict-free; else `0` (including when no plugins are installed, since there
is nothing to analyze). Does not resolve or install anything.

### Global flags

- `--debug`: emit stack traces for unexpected errors. The trace is printed on
  every error path a command reaches — not only the final generic handler, and
  including the errors a subcommand catches itself — so a named error (for
  example, an engine-internal failure) is debuggable too. Two boundaries:
  argument errors are reported by the parser before any command runs, so they
  print usage rather than a trace; and when the exception chain carries a
  pydantic `ValidationError`, the CLI prints a scrubbed one-line summary
  instead of a trace, because the native traceback would re-echo the rejected
  input (`input_value=...`), which may be a mis-pasted secret.

## 2. Output conventions

- Human‑readable console output by default; ASCII status markers
  (`[OK]`/`[FAIL]`/`[WARN]`/`[INFO]`) so a redirected/piped stream never crashes
  on a decorative character.
- The output streams are forced to UTF‑8 when not already UTF‑8 (for example, a Windows
  redirect defaulting to the ANSI code page), so non‑Latin transcripts print
  losslessly rather than raising `UnicodeEncodeError`. Transcript text is never
  silently replaced.
- JSON output for transcription with `--json`.
- Clear error messages on failure (stderr).
- Exit codes: `0` success, `1` runtime/transcription failures **and engine
  faults**, `2` usage or validation errors *the invoker can fix*. The split
  follows the same fault ownership the server uses (server.md §3.7), applied
  to the CLI's own caller role -- the invoking user owns the flags AND the
  environment -- and it is classified **at the seam that knows the source**,
  never by exception class alone:
  - **Exit 2 (invoker-actionable).** A mis-typed flag; an unknown/malformed
    model key; a bad `--options` payload (validated by `_parse_options`
    before the engine runs); a strict-mode `UnsupportedFeatureError`; a bad
    audio input; and every `ConfigError` -- configuration is invoker-owned
    at the CLI *whichever seam it surfaces at*, including a factory
    rejecting a supplied value and a deferred credential check raising
    `ConfigurationRequiredError` at first transcribe. (The same errors are
    a scrubbed 500/503 on the server, whose clients cannot supply config:
    ownership follows the supplier, not the exception site.)
  - **Exit 1 (engine/deployment fault).** A registered model whose plugin
    fails to import (`FactoryLoadError` -- the server's scrubbed-500 state,
    never the caller's 404/exit-2) on **every** command that touches the
    plugin, `show` included: rendering the fault while returning 0 told a
    script the model was usable, so `show` prints the metadata it has plus a
    sanitized `Capabilities: <unavailable: ...>` line and still exits 1;
    anything in the `ValueError` family escaping the **engine execution
    seam** -- which starts at the ATTRIBUTE LOOKUP, not the call, since
    `prepare` and `declared_capabilities` may be descriptors whose bodies are
    plugin code (`transcribe()` / `prepare()` / class-level capability reads):
    a bare SDK `ValueError`, a raw pydantic `ValidationError` from an
    engine-internal model (rendered on the operator audience), an
    `InvalidProviderParamError` the CLI user cannot have caused; every
    `EngineContractError` (broken sync-call boundary, or a declaration
    defect -- a coroutine/non-callable/parameter-requiring `prepare`, a
    malformed declared language tag, a missing IC.6 `default_language`);
    and `TranscriptionError` / other runtime failures.
  The one in-band residual is documented on `ConfigError` itself: an engine
  that raises it (or lets a non-config internal `ValidationError` escape its
  factory, which `create()` wraps) for a fault that is NOT about the
  supplied configuration mis-asserts the type's ownership contract; the
  compliance suite's zero-arg construction check
  (`engine_construction_failed`) polices that, not consumer-side guessing.
