<!-- SPDX-FileCopyrightText: 2026 Standard Voice Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Terminology

This file is the terminology database for Standard ASR. It gives one canonical
term per concept so that prose across the code, the messages, and the docs reads
as one voice. [`STYLE.md`](STYLE.md) requires it. Every contributor and every AI
agent must use these terms.

Each row gives the canonical term, a one-line definition, the approved usage,
and the forbidden synonyms. A forbidden synonym is a word you must not use for
that concept. It does not forbid the word for its own, separate meaning:
"backend" is wrong for an engine, but correct for an audio decode library.

A few words carry more than one meaning in this domain — *model*, *frame*,
*adapter*, *provider*. Their rows name **every sense in use**. Use a listed
sense, write the sentence so the sense is clear, and never introduce a sense the
table does not list. This is what `STYLE.md`'s "one word, one meaning" rule
means here: not that a word has exactly one sense, but that every sense it has
is written down.

## Spelling

Use **American English** in all prose: `normalize`, `behavior`, `initialize`,
`serialize`, `analyze`, `color`, `canceled`, `canceling`, `modeling`.

One exception: never re-spell a verbatim identifier. A prose sentence that names
the standard library exception `CancelledError` keeps that spelling, even though
the same sentence writes the verb "canceled". See `STYLE.md`, resolution 4.

## Core terms

| Term | Definition | Approved usage | Forbidden synonyms |
| --- | --- | --- | --- |
| **engine** | A speech recognizer that speaks the Standard ASR protocol: a `StandardASR` implementation carrying an `engine_id` (`properties.engine_id`). "Engine" names the type and its instances alike — say "the engine class" for the class-level metadata a caller reads without instantiating (`ModelRegistry.engine_class`), and "the engine" or "the engine instance" for the constructed object. An ASR library, cloud API, or SDK that has not been adapted is not an engine — the adapter work makes it one. | "the engine transcribes the audio", "a compliant engine", "the engine class", "engine author". | Do not call an un-adapted ASR library or API an "engine" in technical prose. Not a synonym for *model* (a preset), *backend*, or *provider*. |
| **model** | Three senses, all in use. **(1) preset** — a discoverable preset of an engine, addressed as `engine_id/model_name` (for example `faster-whisper/large-v3`); a registry entry, not the runtime object. **(2) data model** — a pydantic class (`the event model`, `the nested model class`); always say "data model" or "pydantic model" where the preset sense could be read. **(3) trust model / threat model** — the fixed compound. | "discovered models", "model key", "the `large-v3` model", "the event data model". | Do not call the engine object or the engine class a "model". Do not call the trained weights a "model" — say "weights" or "checkpoint" (`model weights` is the approved compound for the files themselves). |
| **plugin** | The pip-installable distribution that implements one or more engines and registers their models through the `standard_asr.models` entry-point group. Discovery also accepts entry points injected with no distribution (the `eps=` test path); those are not plugins. | "plugin package", "plugin author", "plugin discovery". | extension, driver, module (for the package). Say "distribution" for the installed package when the packaging identity is the point. |
| **adapter** | The integration **work**: designing and writing what makes a recognizer speak the Standard ASR protocol. Its **artifact** is an engine. Name the artifact *engine* wherever it has a class, a method, an instance, or runtime state — the engine class, the engine's `_open`, engine state, the engine converts timestamps. Use *adapter* only for the work itself, the author's obligations, or a defect in the integration. | "compliance is a thin adapter, not a rewrite", "an adapter obligation", "the adapter maps native names to BCP-47". | Do not use for anything that has a class, an instance, or runtime state — that is the *engine*. Not a synonym for *plugin* (the package). Pydantic's `TypeAdapter` is an identifier and is exempt. |
| **backend** | An audio decode or resample library: the stdlib `wave` reader, soundfile, FFmpeg, scipy, or the built-in numpy fallback. It is never an ASR engine. | "decode backend", "resample backend". | Never use "backend" for an ASR engine. |
| **provider** | The party or system on the far side of an integration. Approved in three fixed forms only: `provider_params` (an engine's own escape-hatch parameters), "provider-native option", and "provider storage URI" (a cloud vendor's object store). | "provider_params", "provider-native option", "provider storage URI". | Not a synonym for *engine*, *plugin*, or *distribution*. Say "distribution" for the pip package that registers an `engine_id`. |
| **registry** | The in-memory index of discovered models, `ModelRegistry`, built by `discover_models()`. | "the registry", "register a model". | catalog, index (for the object). |
| **transcript** | The recognized text. It is the `text` field of a result, not the result object. | "the transcript", "segment text". | Do not use "transcript" for the whole result object. |
| **result** | The structured transcription return object: `TranscriptionResult` — returned by batch `transcribe()` and by a session's `result()` — or `ChannelResult` per channel. | "the result", "transcription result". | output, response (except the wire type `TranscribeResponse`). |
| **output** | Approved for streams (stdout), the `streaming_output` capability, and a rendering or decoding output grid or buffer ("the output grid", "the millisecond output grid"). | "streaming output", "CLI output", "the output grid". | Do not use "output" for the result object. |
| **segment** | A time-spanned part of a transcript: the `Segment` model. Shared by batch and streaming. | "segment", "segment timestamps". | chunk (that is an input unit). |
| **word** | A word-level unit with timestamps: the `Word` model. | "word", "word timestamps". | token (unless a token is truly meant). |
| **capability** | A declared engine ability in the `DeclaredCapabilities` tree. The instance `effective_capabilities` MUST be a subset of the declared set; compliance checks it (`effective_widens_declared`). Gating consults the **effective** set. | "declared capabilities", "effective capabilities", "capability dot-path". | Do not say "feature" for a `*Cap` **node** — say "capability node". "Feature" stays approved for the concept an application asks about, and in the fixed compound "feature level" (the top-level `supported` flag, as against its constraints). |
| **diagnostic** | A structured, non-fatal report attached to a result or a session: the `Diagnostic` data model (`level`, `code`, `message`, and the optional `param` / `provided` / `effective` triple). It records a lossy, assumed, or degraded path, an accepted-but-noteworthy resolution, or an engine-protocol deviation the standard layer suppressed. | "emit a diagnostic", "diagnostic code". | warning, note, notice (for the object). |
| **session** | A streaming lifecycle handle: `TranscriptionSession` (async) or `SyncSession` (sync). | "streaming session", "open a session". | connection, stream (for the transport). |
| **streaming** | The `mode` domain for incremental audio in and incremental results out. Also the subsystem in `runtime/streaming.py`. Note: `streaming_input` and `streaming_output` are engine-global transport-axis flags, distinct from the per-mode `streaming` domain — but not independent of it. Either flag may be supported only when a `streaming` domain is declared (`DeclaredCapabilities._require_streaming_domain_for_streaming_flags`). | "streaming mode", "streaming session", "the transport axis". | — |
| **negotiation** | The deterministic match of the application's audio form to the engine's accepted form, in `audio/negotiation.py`. A zero-conversion match is a **passthrough**. | "audio negotiation", "passthrough". | — |
| **gating** | Parameter admission control, in `runtime/gating.py`: it drops, degrades, or rejects an unsupported parameter per the **effective** capabilities and the policy. It also enforces `provider_params` type ownership — a wrong-engine `provider_params` always raises, independent of the policy. | "parameter gating". | filtering (for this step). |
| **compliance** | The conformity test surface an engine author runs: `compliance.py`, which emits a `ComplianceIssue` and a `ComplianceReport`. | "compliance suite", "compliance check". | *conformance* as a noun (use "compliance"); *validation* / *validate* for this step (reserve those for pydantic input validation — say "check"). The verb *conform* stays approved where an implementation conforms **to** the spec. |
| **chunk** | The **input** audio unit given to a session: `send_audio(chunk)`, or a byte chunk to `feed(...)`. | "audio chunk", "a chunk of samples". | frame (that is the transport unit). |
| **frame** | Three senses, all in use. **(1) transport** — the WebSocket unit (the config frame, an audio frame, the frame-size cap). **(2) wire format** — a fixed-size PCM unit of the streaming wire format an engine declares (`AudioFormat`), as in "bare PCM frames" and "incremental PCM frame streaming". **(3) DSP** — one sample per channel, used only where a container or codec API names it so (a WAV header declares frames; a `pcm_s16le` frame length). | "WebSocket frame", "config frame", "bare PCM frames", "the WAV header declares N frames". | Do not use *frame* for the variable-size unit an application hands `send_audio()` or `feed()` — that is a **chunk**. |
| **strict / best_effort** | The policy axis for an unsupported input. `strict` raises; `best_effort` drops or degrades and emits a diagnostic. | "strict mode", "best_effort policy". | — |
| **fail-\* family** | Four defined members. **fail loudly / fail-loud** — surface the problem as an exception or a diagnostic instead of proceeding. **fail-fast** — reject before doing expensive work. **fail-closed** — an absent, unknown, or unparseable declaration means *unsupported* or *disabled*; this is the norm here. **fail-open** — the anti-pattern in which an absent declaration reads as *supported*; name it only to describe a bug or a deliberate, documented concession. Adjectival and adverbial inflections are the same member ("a fail-loud floor", "fails loudly"). | Use a defined member. | Do not say "fail-safe" — say **fail-closed**. Do not coin any other "fail-X". |

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
  `audio/conversion.py`, and `runtime/streaming.py`. Codes are `snake_case`.
  Most read `<subject>_<past-tense-verb>` (`language_fell_back`,
  `prompt_truncated`, `frozen_prefix_rewritten`); the `lifecycle_*` and
  `supersede_*` families instead name the illegal transition or the superseding
  operation (`lifecycle_partial_after_final`, `supersede_cross_speaker_merge`).
  Match the family you are adding to.
- **Compliance codes** — the `ComplianceIssue(code=...)` string literals in
  `compliance.py`, same `snake_case` pattern.
- **Enums** — `InputKind` (`audio/input.py`), `ConversionOp`
  (`audio/negotiation.py`), `WordTimestampGranularity` (`contract/params.py`),
  and `Satisfiability` (`toolchain/doctor.py`).
- **Literal sets** — `EventType` (`runtime/streaming.py`) and `ModeName`
  (`contract/capabilities.py`, re-exported as `Mode` from `runtime/gating.py`).
- **Exception classes** — the hierarchy rooted at `StandardASRError` in
  `contract/exceptions.py`.
- **Public API surface** — the `__all__` list in `src/standard_asr/__init__.py`
  is the authoritative set of application-developer names.
