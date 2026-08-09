<!-- SPDX-FileCopyrightText: 2026 Standard Voice Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Terminology

This file is the terminology database for Standard ASR. It gives one canonical
term per concept so that prose across the code, the messages, and the docs reads
as one voice. [`STYLE.md`](STYLE.md) requires it. Every contributor and every AI
agent must use these terms.

Each row gives the canonical term, a one-line definition, the approved usage,
and the forbidden synonyms. A forbidden synonym is a word you must not use for
that concept — it does not forbid the word for its own, separate meaning
("backend" is wrong for an engine, but correct for an audio decode library).

## Spelling

Use **American English** in all prose: `normalize`, `behavior`, `initialize`,
`serialize`, `analyze`, `color`, `canceled`, `canceling`, `modeling`.

One exception: never re-spell a verbatim identifier. A prose sentence that names
the standard library exception `CancelledError` keeps that spelling, even though
the same sentence writes the verb "canceled". See `STYLE.md`, resolution 4.

## Core terms

| Term | Definition | Approved usage | Forbidden synonyms |
| --- | --- | --- | --- |
| **engine** | A speech recognizer that speaks the Standard ASR protocol: a compliant `StandardASR` object (usually an `EngineBase` instance) that carries an `engine_id`. This is what an application calls, and the thing that "speaks the protocol". An ASR library, cloud API, or SDK that has not been adapted is not yet an engine — an adapter makes it one. In marketing or overview prose, "engine" is used loosely for any ASR system; in technical and reference prose it denotes only the compliant object. | "the engine transcribes the audio", "a compliant engine", "engine author". | Do not call an un-adapted ASR library or API an "engine" in technical prose. Not a synonym for *model* (a preset), *adapter* (the integration code), *backend*, or *provider*. |
| **model** | A discoverable **preset** of an engine, addressed as `engine_id/model_name` (for example `faster-whisper/large-v3`). A model is a catalog entry, not the runtime object and not the raw weights. | "discovered models", "model key", "the `large-v3` model". | Do not call the engine object a "model". Do not call the trained weights a "model" in prose — say "weights" or "checkpoint". |
| **plugin** | The pip-installable distribution that implements one or more engines and registers their models through the `standard_asr.models` entry-point group. | "plugin package", "plugin author", "plugin discovery". | extension, driver, module (for the package). |
| **adapter** | The integration code an author writes to make a recognizer — an existing ASR library, cloud API, or SDK, or their own model code — speak the Standard ASR protocol, exposing it as an engine. It is the body of work you write, not the running object. **Runtime behavior and lifecycle name the _engine_** (it detects a disconnect, its `_open` hangs, it converts timestamps); reserve _adapter_ for the integration work and the author's obligations (write the adapter, an adapter obligation, the adapter maps native names to BCP-47). | "compliance is a thin adapter, not a rewrite", "the adapter maps native names to BCP-47". | Do not use for the running object — that is the *engine* (so "the engine's `_open`", not "the adapter's `_open`"). Not a synonym for *plugin* (the package). |
| **backend** | An audio decode or resample library (soundfile, FFmpeg, scipy, or the built-in fallback). It is never an ASR engine. | "decode backend", "resample backend". | Never use "backend" for an ASR engine. |
| **provider** | Reserved for the fixed compound `provider_params` — an engine's own escape-hatch parameters. | "provider_params", "provider-native option". | Not a free synonym for engine, plugin, or model. |
| **registry** | The in-memory index of discovered models, `ModelRegistry`, built by `discover_models()`. | "the registry", "register a model". | catalog, index (for the object). |
| **transcript** | The recognized text. It is the `text` field of a result, not the result object. | "the transcript", "segment text". | Do not use "transcript" for the whole result object. |
| **result** | The structured return object of a batch transcription: `TranscriptionResult`, or `ChannelResult` per channel. | "the result", "transcription result". | output, response (except the wire type `TranscribeResponse`). |
| **output** | Reserved for streams (stdout) and for the `streaming_output` capability. | "streaming output", "CLI output". | Do not use "output" for the result object. |
| **segment** | A time-spanned part of a transcript: the `Segment` model. Shared by batch and streaming. | "segment", "segment timestamps". | chunk (that is an input unit). |
| **word** | A word-level unit with timestamps: the `Word` model. | "word", "word timestamps". | token (unless a token is truly meant). |
| **capability** | A declared engine ability in the `DeclaredCapabilities` tree. The instance `effective_capabilities` is a subset of the declared set. | "declared capabilities", "capability dot-path". | feature (when a `*Cap` node is meant). |
| **diagnostic** | A structured, non-fatal notice: the `Diagnostic` model, with `level`, `code`, and `message`. It reports a lossy, assumed, or degraded path. | "emit a diagnostic", "diagnostic code". | warning, note (for the object). |
| **session** | A streaming lifecycle handle: `TranscriptionSession` (async) or `SyncSession` (sync). | "streaming session", "open a session". | connection, stream (for the transport). |
| **streaming** | The `mode` domain for incremental audio in and incremental results out. Also the subsystem in `runtime/streaming.py`. Note: the `streaming_input` and `streaming_output` capabilities are orthogonal to the `streaming` mode. | "streaming mode", "streaming session". | — |
| **negotiation** | The deterministic match of the application's audio form to the engine's accepted form, in `audio/negotiation.py`. A zero-conversion match is a **passthrough**. | "audio negotiation", "passthrough". | — |
| **gating** | Parameter admission control: it drops, degrades, or rejects an unsupported parameter per the capabilities and the policy, in `runtime/gating.py`. | "parameter gating". | filtering (for this step). |
| **compliance** | The conformance test surface an engine author runs: `compliance.py`, which emits a `ComplianceIssue` and a `ComplianceReport`. | "compliance suite", "compliance check". | validation (reserve for Pydantic), conformance (use "compliance"). |
| **chunk** | The **input** audio unit given to a session: `send_audio(chunk)`, or a byte chunk to `feed(...)`. | "audio chunk", "a chunk of samples". | frame (that is the transport unit). |
| **frame** | The **transport** unit on the WebSocket wire (for example the config frame or an audio frame, bounded by the frame-size cap). | "WebSocket frame", "config frame". | Not a session input unit; not a DSP sample frame in prose. |
| **strict / best_effort** | The policy axis for an unsupported input. `strict` raises; `best_effort` drops or degrades and emits a diagnostic. | "strict mode", "best_effort policy". | — |
| **fail-\* family** | The safety idioms: "fail loudly", "fail-fast", "fail-closed", "fail-open". Each names a defined behavior. | Use the established member. | Do not invent a new "fail-X" without defining it. |

## Two level scales (do not unify)

Two models carry a `level` field with **different** value sets. This is
deliberate. Do not "fix" one to match the other.

- `Diagnostic.level` is `Literal["info", "warning"]`
  (`contract/results.py`). A diagnostic never carries "error", because an error
  raises an exception instead.
- `ComplianceIssue.level` is `Literal["error", "warning"]`
  (`compliance.py`). A compliance issue never carries "info", because a finding
  is either a hard failure or a soft warning.

## Controlled code vocabularies (single source of truth)

The following vocabularies live in the code. This file points to them and never
copies them, so they cannot drift. Add or rename a member only at its source.

- **Diagnostic codes** — the 33 `DIAG_*` module constants, in
  `contract/results.py`, `contract/language.py`, `runtime/gating.py`,
  `audio/conversion.py`, and `runtime/streaming.py`. The naming pattern is
  `<subject>_<past-tense-verb>` (for example `language_fell_back`,
  `prompt_truncated`, `frozen_prefix_rewritten`).
- **Compliance codes** — the `ComplianceIssue(code=...)` string literals in
  `compliance.py`, same `snake_case` pattern.
- **Enums** — `InputKind` (`audio/input.py`), `ConversionOp`
  (`audio/negotiation.py`), `WordTimestampGranularity` (`contract/params.py`).
- **Literal sets** — `EventType` and the mode names in `runtime/streaming.py`
  and `runtime/interface.py`.
- **Exception classes** — the hierarchy rooted at `StandardASRError` in
  `contract/exceptions.py`.
- **Public API surface** — the `__all__` list in `src/standard_asr/__init__.py`
  is the authoritative set of application-developer names.
