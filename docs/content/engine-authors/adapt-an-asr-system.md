---
title: Adapt an ASR System
---

# Adapting an ASR system to Standard ASR (engine authors)

> Authoritative reference: [`docs/content/specification/protocol.md`](../specification/protocol.md). Entry-point rules: [`plugin-entry-points.md`](./plugin-entry-points.md).

You implement **one** class. The standard layer gives you audio-input negotiation, conversion, resampling, parameter gating, diagnostics, the CLI, the reference server, and the compliance suite — for free.

## The contract

Subclass `EngineBase` and provide:

1. `properties: ClassVar[BaseProperties]` — static identity and I/O boundaries (`accepted_input`, `native_sample_rate`, `accepted_sample_rates`, `selectable_languages`, …). Pin `protocol_version` to the protocol the engine implements, never to the installed `standard-asr` version (spec AR.1). Bump it only after the engine fully implements the newer contract. It is not a package requirement either: declare the `standard-asr` version range your code imports from in your project metadata, because the two move independently.
2. `declared_capabilities: ClassVar[DeclaredCapabilities]` — what you support, per mode (`batch` / `streaming`). Omit what you don't support (fail-closed).
3. `declared_metadata: ClassVar[DeclaredEngineMetadata]` — static metadata for the model preset. The protocol requires an authored `artifacts` section. Use `NO_ARTIFACT_LIFECYCLE` only when no supported context has an inference-artifact lifecycle.
4. `provider_params_type: ClassVar[type[ProviderParams] | None]` — your typed escape-hatch model, or `None`. Publish your own **terminal** subclass: the bare `ProviderParams` base, a non-subclass, or a model that is not closed (`extra="forbid"`) is a compliance error (`provider_params_type_is_bare_base`, `provider_params_type_not_subclass`, or `provider_params_type_not_closed`).
5. `__init__` — capture config only. **Keep it pure**: no filesystem, GPU, or network (spec IC.9). Load weights lazily, in a method of your own; the first-party plugins call theirs `_ensure_model_loaded`.
6. `_transcribe(prepared, params) -> TranscriptionResult` — run your model on already-negotiated audio (`prepared.kind` is one of your `accepted_input`).
7. (Streaming) override `_start_transcription(*, gated_params, audio_format, prepared_audio)` returning a `TranscriptionSession` subclass.
8. `config_type: ClassVar[type[BaseConfig]]` — your config class. It is read from the class, so a settings UI can render the schema without constructing the engine. Without it, `registry.config_schema()` returns `None`, `GET /v1/config-schema/{model}` returns an empty schema, `standard-asr show` reports no init config, and compliance warns (`missing_config_type`).
9. If your `properties` declare `selectable_languages`, your config MUST carry a usable `default_language` (spec IC.6). Inherit `LanguageConfigMixin` to get the field. Without it, every `transcribe()` raises `EngineContractError` and compliance reports the error `language_config_invalid`.
10. If you set `effective_capabilities`, it MUST narrow `declared_capabilities`, never widen it. Compliance reports a widening as the error `effective_widens_declared`.
11. If `declared_metadata.artifacts.applicable` is `True`, implement `_artifact_requirements()`; if `supports_explicit_acquisition` is also `True`, implement `_acquire_artifacts()`. Compliance reports a missing hook as the error `artifact_requirements_hook_missing` or `artifact_acquisition_hook_missing`. Keep this work separate from `prepare()`, which remains an optional process-local warm-up hook. See [Inference artifacts](../reference/artifacts.md).

## Minimal batch engine

```python
from typing import ClassVar, Literal
from standard_asr.engine import (
    BaseConfig,
    BaseProperties,
    BatchCapabilities,
    DeclaredCapabilities,
    DeclaredEngineMetadata,
    EngineBase,
    FlagCap,
    InputKind,
    LanguageCaps,
    LanguageConfigMixin,
    NO_ARTIFACT_LIFECYCLE,
    PreparedAudio,
    RuntimeParams,
    TranscriptionResult,
)


class MyConfig(LanguageConfigMixin, BaseConfig[Literal["my-engine"]]):
    engine: Literal["my-engine"] = "my-engine"
    default_language: str = "en"  # IC.6: required once you declare a language axis


class MyProps(BaseProperties):
    engine_id: str = "my-engine"
    model_name: str = "base"
    protocol_version: str = "0.2.0"  # AR.1: the protocol this engine implements
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] = [16000]
    selectable_languages: list[str] = ["en", "auto"]
    detectable_languages: list[str] = ["en"]


class MyEngine(EngineBase):
    properties: ClassVar[BaseProperties] = MyProps()
    config_type: ClassVar[type[BaseConfig]] = MyConfig  # schema without instantiation
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        )
    )
    declared_metadata: ClassVar[DeclaredEngineMetadata] = DeclaredEngineMetadata(
        artifacts=NO_ARTIFACT_LIFECYCLE,
    )

    def __init__(self, **kw: object) -> None:
        self.config = MyConfig(**kw)  # extra="forbid": a mistyped option fails loudly
        self._model = None

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        audio = prepared.array  # 16 kHz float32 mono, per Properties
        text = my_model_infer(audio)  # your code
        # Report what the recognizer actually decided. Never echo
        # params.language: with "auto" selectable it can be the reserved
        # "auto", which detected_language rejects.
        return TranscriptionResult(text=text, detected_language="en")
```

## Map parameters

- The standard layer gates the portable standard set against your `effective_capabilities` (which default to `declared_capabilities`) before it calls `_transcribe`: `language`, `candidate_languages`, `word_timestamps`, `diarization`, `prompt`, `phrase_hints`. Map them onto your model's native arguments. For `diarization`, presence means enable: map `params.diarization is not None` onto your native enable switch (the current `DiarizationRequest` marker carries no fields). An engine that declares `diarization.supported=True` MUST actually diarize when the request passes the gate. The standard layer cannot verify this. Silently ignoring a gated-and-passed request is the cardinal sin: a silent wrong result.
- Engine-specific parameters → a `ProviderParams` subclass set as `provider_params_type`. Wrong-engine params raise `InvalidProviderParamError`.
- The base resolves the language axis for you on both paths: the `params.language` your `_transcribe` receives is already the effective value. Only a structural engine that bypasses `EngineBase` calls `standard_asr.contract.language.effective_language(...)` itself.
- **`word_timestamps.granularities` declares what you can honestly *deliver*, not which native API switch exists.** Declare every granularity your engine can serve — including ones that come for free. If your model emits per-segment start/end on every run (most do), declare `"segment"` even when there is no separate "segment mode" switch. Otherwise the standard layer rejects the cheapest, always-satisfiable request as a false incompatibility. Then map each granularity precisely (for example, only `"word"` enables your forced-alignment pass; a `"segment"` request MUST NOT back-fill word-level data — `words=None` means "not requested").

## Audio you receive

`prepared` is already in one of your `accepted_input` shapes:

- `InputKind.ARRAY` → `prepared.array` (float32, `prepared.sample_rate`)
- `InputKind.ENCODED_FILE` → `prepared.path`
- `InputKind.ENCODED_BYTES` → `prepared.data`
- `InputKind.FETCHABLE_URL` → `prepared.url`
- `InputKind.STORAGE_URI` → `prepared.storage_uri` (a provider storage URI such as `s3://` or `gs://`, for engines that read from cloud storage)

You never write decode/resample/encode glue — declare `accepted_input` and the standard layer delivers the right shape (and attaches conversion diagnostics).

### Advanced: native multi-input batches

Keep `transcribe()` as the public single-input entry point. When a native SDK can infer several independent inputs in one call, an `EngineBase` subclass can reuse the exact standard batch pipeline for each item through its protected hooks:

```python
requests = [self._prepare_transcription_request(audio, params) for audio in inputs]
native_results = self._native_transcribe_many(
    [(request.audio, request.params) for request in requests]
)
results = [
    self._finalize_transcription_result(native, request)
    for native, request in zip(native_results, requests, strict=True)
]
```

`_prepare_transcription_request()` performs the same protocol compatibility check, provider-parameter gate, capability degradation, language resolution, audio negotiation, conversion, and pre-inference diagnostic collection as `transcribe()`. `_finalize_transcription_result()` checks that each native return is a synchronous `TranscriptionResult`, applies standard speaker synthesis, and merges that request's diagnostics. Keep the request/result pairing intact and isolate an individual native failure before finalizing the other successful items. These are protected `EngineBase` hooks for an adapter's own optimized wrapper; they do not add a second public Standard ASR operation.

## Streaming

**Declare the transport axis first.** Set `streaming_input=FlagCap(supported=True)` if you accept incremental PCM frames, `streaming_output=FlagCap(supported=True)` if you return results incrementally, or both. These are engine-global flags, and either one may be supported only when you also declare a `streaming` domain. A `streaming` domain with neither axis is a streaming engine nobody can call: every `start_transcription()` raises `UnsupportedFeatureError`, and compliance reports the error `streaming_domain_without_axis`.

Subclass `TranscriptionSession`, implement async `_produce()` (read fed audio via `self.audio_chunks()`, yield `TranscriptionEvent` objects). The base provides `feed`/`send_audio`/`end_audio`, backpressure, the done-timeout, and the sync bridge — you only write `async`. See the spec §ST for the event model (`partial`/`final`/`supersede`/`progress`/`done`/`error`) and the `stable_text` rules.

### Session establishment — the base does the gating for you

You override `_start_transcription(...)`, **not** the public `start_transcription`. The base `start_transcription` is a template method (symmetric to `transcribe` / `_transcribe`): it runs the standard streaming pipeline and then calls your hook. Before your hook runs, the base has already:

- enforced the `audio_format` / `audio` mutual exclusion (ST §3.1) via `ensure_stream_inputs_exclusive`;
- validated the language config (LANG R1 / IC.6);
- run the **fail-closed** wire-format check (`ensure_stream_format_supported`) on the encoding, the channel count, and the sample rate. It rejects an `audio_format.encoding` not in your declared `wire_encodings`, so an undeclared encoding is not misframed as PCM and silently mistranscribed. **Declare `wire_encodings`**: leaving it unset means "unconstrained", and the encoding check is then skipped, so an encoding you never declared reaches you unchecked. This is the one fail-open concession in the check; compliance warns about it (`streaming_input_without_wire_encodings`). It rejects an `audio_format.channels` other than 1: streaming wire input is mono-only, because the standard layer does not downmix incremental frames the way the batch path does. Downmix to mono before feeding. It also rejects a wire `sample_rate` you do not accept. Per spec R7's implementation note, the standard does **not** resample streaming wire frames (only the batch `transcribe` path resamples), so an unreachable wire rate is a loud error. When `required_input_sample_rate` is set, the wire rate MUST equal it — even when another rate appears in `accepted_sample_rates`. That list describes the batch path, which resamples to the required rate before your engine; unresampled wire frames at any other rate would be misread. Otherwise the standard accepts the rate when `accepted_sample_rates` is `"any"` or when it is in that concrete list. (Standard-layer streaming resampling is a deferred capability; this guard becomes a resample once it lands.)
- **gated the runtime parameters** against your `streaming` capabilities, and resolved the language axis. Gating covers `provider_params` swap-safety (Runtime R3: a wrong `provider_params` type always raises `InvalidProviderParamError`), capability gating (R2), and guidance degradation (R4). The base attaches the gating and language **diagnostics** to the returned session; they surface through `session.diagnostics()`.
- for the **whole-input** path (`audio=...`, for example, OpenAI-style streaming output), run that complete input through the **same** audio negotiation/conversion pipeline as batch `transcribe`, and hand your hook the result as `prepared_audio`. The `prepared_audio` is a `PreparedAudio` already in one of your `accepted_input` shapes, with its conversion diagnostics attached to the session. For the incremental `audio_format=...` path there is no whole input, so `prepared_audio` is `None`.

Your hook receives the **already-gated, frozen** `gated_params` (spec R5: streaming params are frozen at `start_transcription` and MUST NOT change mid-stream, except a guidance channel declared `mutable_mid_stream` — a reserved declaration in the current generation). Use them directly — do not re-gate or re-accept raw params. The signature is keyword-only: `gated_params`, `audio_format` (the wire format, or `None`), and `prepared_audio` (the negotiated whole input, or `None`).

```python
def _start_transcription(self, *, gated_params, audio_format, prepared_audio):
    # Guards + gating + (whole-input) audio prep already ran in the base.
    # gated_params is frozen (R5); prepared_audio is None for the incremental
    # audio_format path and a PreparedAudio for the whole-input audio path.
    return MySession(gated_params, ...)
```

## Credentials & environment fallback (IC.4)

Build your config with `Config.from_env(engine_id, **explicit)` instead of the bare constructor. Unset fields fall back to `STANDARD_ASR_<ENGINE>__<FIELD>` environment variables. Note the **double underscore** that separates the engine and field segments; explicit args win. Credentials are wrapped in their masking carrier by construction, never passed around as plaintext. Put secrets (`api_key`, tokens) in `SecretStr` fields — or `SecretBytes` for byte credentials — via `secret_field()`. Declare exactly one carrier per field, optionally with `None`. Keep non-secret routing (`base_url`, `region`) plain.

A structured field (a list, a mapping, a submodel, a `TypedDict`, a dataclass) takes its env value as JSON (`'["en","ja"]'`). A scalar field — including `SecretStr` and `Path` — takes the raw string, byte for byte. The field's own schema decides which of the two applies, so any shape the config guards accept is reachable through the env convention. One shape is refused at class definition: a field accepting BOTH (`str | list[str]`) has no defined reading. `"123"` is either that string or that JSON number, and either choice would disagree with the explicit constructor, which always takes the string. Declare one shape, or model the alternatives as a named submodel.

The config's serialization surface is **closed**: `public_dump()` emits your declared input fields, serialized by pydantic's own machinery, and nothing else. Definition-time guards hold that closure by **enumeration, not proof** (the accident model — see the trust model in `AGENTS.md`): they walk your declared annotations and your serialization decorators, at every nesting depth, and reject the hazards an honest author actually writes. Anything that would make `model_dump` run author code — or emit something other than your declared inputs — is rejected at class definition: `@computed_field`, `@model_serializer`, `@field_serializer`, `PlainSerializer`/`WrapSerializer` metadata, a `SerializeAsAny[...]` field (its dump follows the *runtime* object, so the declared type no longer bounds the output), an **undeclared value shape** (`Any`, `object`, an unparametrized container, `dict[str, Any]` — same duck-typing, reached from the type instead of a marker; spell a heterogeneous mapping as `dict[str, str | int | bool | None]` or a named submodel), a **nested submodel** carrying any of those, and `Field(exclude=True)`. An enumeration has a boundary: a serializer installed through a custom `__get_pydantic_core_schema__` slips past it, and the guards do not chase it — installing one means actively smuggling code past the enumeration, which is an adversary, out of scope for a trusted plugin. Keep `extra="forbid"` too (BaseConfig's default). `extra="allow"` stores undeclared caller data and dumps it verbatim past the secret mask. `extra="ignore"` silently swallows a mistyped credential key, so it reads as an absent credential rather than a loud error. The reason: `public_dump()` is documented safe for `/v1/models`, persistence, and telemetry. That holds only while nothing author-defined can rematerialize a credential inside it. Its output must also stay the declared input surface, so persisting and reloading a config round-trips.

The input surface stays closed **at every depth**, not only on the config itself. Every nested input container your schema reaches — an options submodel, a `TypedDict`, a dataclass — must forbid undeclared keys, and one that does not is rejected at class definition. pydantic's default for all three silently *drops* an unknown key. So a user's typo'd nested option (`{"decode": {"baem": 8}}`) would read as applied while your engine runs on the field's default — a silent wrong result. The rule reads the *effective* policy from the core schema, so pydantic's config propagation is honored. A bare `TypedDict` or stdlib dataclass inherits the config's `extra="forbid"` and is closed for free. A nested `BaseModel` needs `model_config = ConfigDict(extra="forbid")`. A *pydantic* dataclass (which owns its config) needs `@pydantic.dataclasses.dataclass(config=ConfigDict(extra="forbid"))`.

Use a plain `@property` for derived in-process values (an `authorization` header belongs in your engine code, not in the config dump), and keep config fields to plain typed inputs.

One boundary is yours to keep, because no schema-level guard can hold it for you: **never copy a secret out of its carrier**. The guards bound what the *schema* installs in the dump, not the contents of values your own code builds. A validator that writes `get_secret_value()` into a plain field or onto an object's display state (say, a `Path` subclass whose `__str__` embeds the token) emits that plaintext through `public_dump()` — and would through any dump mechanism. Read the credential with `reveal_dump()` at the point of use in your engine code, and let it live nowhere else.

Provider-native wire names map onto standard fields with **plain string aliases** (`Field(alias="xi-api-key")`, or an all-string `AliasChoices`). `AliasPath` — and any `AliasChoices` carrying one — is rejected at class definition. The flat env convention and the absent-vs-invalid config classifier (what makes a missing credential a compliance *skip* instead of a fail) both resolve fields by single string tokens, which a nested path alias cannot provide. If a value is genuinely nested, declare it as a submodel field (its env value arrives as JSON).

```python
def __init__(self, **kwargs):
    self.config = MyConfig.from_env("my-engine", **kwargs)  # IC.4
```

## Wire-visible values: `extra`, diagnostics

Every slot the wire can see — a `TranscriptionResult` / `Segment` / `Word` / `TranscriptionEvent` `extra`, and `emit_diagnostic`'s `provided` / `effective` — holds **JSON values only** (`JsonValue`: null, bool, int, finite float, str, and lists/str-keyed dicts of those). The Python objects and the JSON documents are the same protocol seen twice. So a value with no JSON form is rejected at construction, naming the field, instead of failing later in the transport — after the server has already committed to a response. Non-finite floats (`NaN`, `Infinity`) are excluded for the same reason: they are Python floats but not JSON, and a conforming parser rejects the whole document.

`emit_diagnostic` projects a **structured** value (a pydantic submodel) into its JSON form itself, so `provided=my_request_model` just works. For any other wire-visible slot — or to absorb the `list`-invariance complaint a type checker raises when a `list[str]` variable meets a `list[JsonValue]` parameter (a static-analysis artifact, not a real mismatch) — use `to_json_value` from the engine surface:

```python
from standard_asr.engine import to_json_value

hints: list[str] = [...]
event = TranscriptionEvent.final("s1", text, extra={"hints": to_json_value(hints)})
```

Runtime validation is unchanged either way: a value that is genuinely not JSON is rejected loudly at construction, naming the field.

If you genuinely need an arbitrary in-process object, keep it in your own engine/session state: a standard protocol object's whole contract is that both layers can express it.

## Streaming responsibilities (what the base does vs you)

The base `TranscriptionSession` owns the pump, backpressure (bounded buffers), the done-timeout/idle deadlines, the sync bridge, and lifecycle suppression. It also corrects a `stable_text` that shrinks, or ends inside a combining character sequence or right after a zero width joiner. By default the base suppresses or corrects a violation and records a diagnostic. An event that goes beyond your engine's effective capabilities is the exception: the base delivers it unchanged and records a diagnostic (see "Declare what you emit"). With `strict_lifecycle=True`, a violation ends the session instead, with a terminal `error` event whose code is `engine_error`. **You** must:

- Send the segment's whole current `text` on every `partial` and `final`, not only the words added since the last event.
- Mark text stable only when your recognizer promises it does not change. A stability score is an estimate, not a promise. When a segment has no stable text yet and you cannot guarantee that any non-empty start of a `partial`'s text keeps that promise, send `stable_text=""` on that `partial`. If your engine cannot give that guarantee on any `partial`, declare `streaming.partial_stability` unsupported. One `partial` without stable text does not require that declaration.
- Once a segment has stable text, send at least that stable text on every later `partial` of the segment. A `partial` that leaves `stable_text` out carries the default `""`, which is shorter than the earlier stable text. The base delivers that `partial` with the earlier stable text and records `stable_text_clamped`. With `strict_lifecycle=True`, the session ends instead, with a terminal `error` event whose code is `engine_error`.
- End stable text between two user-perceived characters. For example, place the boundary at a word boundary, or split `text` into user-perceived characters with a Unicode text segmentation library (UAX #29) and move the boundary back to the start of the character it cuts. The standard layer checks only part of this rule (see "The boundary check is partial" below).
- Do not mark text stable up to its last character while a mark that belongs to that character may still arrive. That happens, for example, when your model emits a Thai consonant and its tone mark as separate tokens. The base suppresses a later event that adds a combining mark right after the stable text (`stable_text_rewritten`). If the segment is still open when the session reaches `done`, it is reported as `stable_text_abandoned`.
- Before `done`, send a `final` for every segment that has stable text, unless a `supersede` already retired it. Otherwise that text is missing from the result.
- For reconnect: detect the disconnect, re-establish the connection, and replay `self.replay_buffer()`. Keep `segment_id`, timestamps, and language continuous. Then call `self.note_reconnect(gap_start, gap_end, content_lost=...)`.

The base always emits the `progress(reconnect)` event. It emits a trailing **non-terminal** `content_lost` error (`recoverable=true` — a fidelity warning; the session stays alive and events keep flowing) **only if you pass `content_lost=True`**. That is your own determination that the reconnect and replay could not cover the gap, and that unreplayable audio was permanently lost. The base does **not** infer loss from rolling-buffer eviction (a live ring is always evicting, so that would falsely claim loss on every long session); you decide, because only you know whether the replay actually bridged the gap.

**The boundary check is partial.** The standard layer checks only that stable text does not end inside a combining character sequence or right after a zero width joiner. A combining character sequence is a base character followed by the marks and joiners that belong to it (Unicode definition D56). A cut right after a zero width joiner does not split such a sequence, but the joiner binds the character after it to the one before it, as in an emoji built from several code points. The check tests three things: the character after the stable text in `text` is not a combining mark, a zero width joiner, or a zero width non-joiner, and the stable text does not end with a zero width joiner. That check passes other splits inside one character:

- before Thai SARA AM (U+0E33) or Lao AM (U+0EB3);
- after the sign that joins two consonants into one character (U+094D in Devanagari), in Devanagari, Bengali, Gujarati, Odia, Telugu, and Malayalam, and, under the ICU library's segmentation, in Khmer and Myanmar;
- between Korean letters written as separate jamo, inside a flag or a skin-tone emoji, before a tag character in an emoji tag sequence such as the flag of Scotland, before the voiced sound mark of half-width katakana, and between CR and LF.

The base does not correct these, and `check_event_sequence` does not report them. A clean compliance report, a `True` from `validate_stable_text`, and a passing `assert_stable_text_invariant` mean that no violation was found, not that your boundaries are valid. Ending stable text between user-perceived characters is your obligation ([streaming specification](../specification/protocol.md#streaming), section 4.2).

When an incremental producer knows that it has received the complete input, call `session.set_input_duration(seconds)`. The method is idempotent for the identical measurement and feeds `duration` into both a successful result and an explicit partial snapshot. Do not substitute `audio_processed_until`: a cursor only says how far the recognizer processed, and does not prove that the input ended. For whole-input streaming, `EngineBase` records the prepared audio duration itself. Override `_max_audio_duration(mode)` only when the configured engine has a narrower mode-specific limit than its static Properties; keep Properties unchanged for discovery.

Override synchronous `close()` when your engine retains process-local model handles, workers, or accelerator resources. The base implementation is a no-op. Do not implement `async def close()`: applications and pools call it after active work drains, and a coroutine return would report a false cleanup success.

**`error` events fail closed to terminal.** An `error` event with `recoverable` unset is normalized to `recoverable=false` (terminal) at construction: unknown recoverability must not leave consumers waiting on a stream that may never continue. If you emit an advisory, non-fatal error (the session keeps going), set `recoverable=True` explicitly — otherwise your event ends the session.

**Surface non-fatal notes via `emit_diagnostic`.** Call `self.emit_diagnostic(code=..., message=..., level="info"|"warning")` from `_produce` to report a best-effort degradation, an assumed parameter, or a lossy fallback through the session's `diagnostics()` channel — the streaming counterpart of the batch path's `result.diagnostics`. It is bounded (spec ST.6.4, like the guard's own diagnostics) and the server forwards it to a WS client as a mid-stream `diagnostics` frame. Keep `error` events for *fatal* conditions.

> **Security:** a diagnostic is **engine-authored, client-facing output**, like the transcript itself. Its `message`/`param`/`provided`/`effective` fields are forwarded to (possibly unauthenticated) clients **verbatim and unredacted**. Never put a credential, API key, authenticated URL, or raw exception text in a diagnostic — route sensitive operator detail to `logging` instead. (The server *does* scrub an `error` event's `extra`, because that is auto-captured exception detail — pre-summarized input-echo-free by the standard layer, but still operator-only content; a diagnostic is content you chose, so its safety is yours.)

### Sequence invariants the guard enforces for free

Beyond lifecycle transitions, the base `_LifecycleGuard` enforces three per-stream invariants on every event you yield, so a slipped engine still cannot emit a wrong transcript:

- **Monotonic audio cursor** — a decreasing `audio_processed_until` is clamped to the prior value (the cursor never moves backwards; ST §4.4), with an `audio_cursor_decreased` diagnostic.
- **Stable text does not change** — a `partial` or `final` that changes the segment's earlier `stable_text` is suppressed with a `stable_text_rewritten` diagnostic (ST §4.2). An event changes it when its `text` no longer starts with it, or puts a combining mark or a zero width joiner or non-joiner right after it. A `partial` whose `stable_text` is shorter than before, or ends inside a combining character sequence or right after a zero width joiner, is delivered with a corrected value and a `stable_text_clamped` diagnostic. The guard catches only the splits that the boundary check above catches. One exemption, from that same section: a terminal `closed` final MAY restate stable text once, to change punctuation, casing, or number formatting (for example "twenty twenty" to "2020"). The guard admits it. Use `closed` only for a change in how the text is written. A change in what was said is a `supersede`. A plain `final` is not exempt. Your recognizer may format text when it finalizes (casing, punctuation) after you marked the unformatted text stable. In that case, send the formatted text as the `closed` final, not as a plain `final`.
- **Stable text reaches the result** — the result holds finalized segments only. A segment that has stable text and is still open when the session reaches `done` gets a `stable_text_abandoned` diagnostic. The application was told that text does not change, and the result does not contain it. A segment that a `supersede` retired is not checked.

The full set of standard-layer diagnostic codes the guard can emit (read them off the session with `session.diagnostics()`):

- `stable_text_clamped` — a `partial` whose `stable_text` shrank, or ended inside a combining character sequence or right after a zero width joiner, was delivered with a corrected value. More causes reach the guard only on an event built without validation, for example with `model_copy(update=...)`. A `partial` whose `stable_text` is not a string, or is not the start of its `text`, gets the segment's earlier stable text (`""` if it had none). A `final` whose `stable_text` is not its whole `text` gets its whole `text`.
- `stable_text_abandoned` — the session reached `done` while a segment that has stable text was still open; that text is missing from the result.
- `audio_cursor_decreased` — a decreasing `audio_processed_until` was clamped.
- `stable_text_rewritten` — an event that changed a segment's stable text was suppressed.
- `locked_speaker_rewritten` — an event changing (X→Y) or retracting (X→None) the already-accepted `speaker` of a segment that has stable text was suppressed. First assignment (None→X, the delay-to-final strategy) stays legal, and a `closed` final is exempt (terminal correction).
- `lifecycle_after_terminal` — a `partial`/`final` after the segment became `closed`/`superseded` was suppressed.
- `lifecycle_partial_after_final` — a `partial` after the segment's `final` was suppressed.
- `lifecycle_final_after_final` — a second `final` for an already-final segment was suppressed.
- `lifecycle_closed_superseded` — a `supersede` retiring a `closed` segment was suppressed.
- `lifecycle_retired_resuperseded` — a `supersede` retiring an already-superseded segment was suppressed (an id retires exactly once).
- `supersede_unknown_old_id` — a `supersede` whose `old_ids` contain a never-announced segment was suppressed.
- `supersede_reintroduces_segment` — a `supersede` whose `new_ids` reuse an already-known id was suppressed.
- `supersede_noncontiguous_old_ids` — a `supersede` whose `old_ids` do not form a contiguous block of the live reading order (in reading order) was suppressed: the replacements would have no defined placement, and an untimestamped transcript's word order would silently change.
- `supersede_cross_speaker_merge` — a `supersede` that would merge segments carrying distinct non-null speakers into fewer segments was suppressed (someone's words would be silently mis-attributed).
- `stream_exceeds_emits_partials`, `stream_exceeds_partial_stability`, `stream_exceeds_re_segments`, `stream_exceeds_audio_progress`, `stream_exceeds_timestamps`, `stream_exceeds_word_timestamps`, `stream_exceeds_diarization`, and `finality_level_not_reached` — the stream did not match your engine's effective streaming capabilities. The event was delivered unchanged, and each code is recorded once per session. See "Declare what you emit" below.
- `diagnostics_truncated` — the bounded diagnostic channel overflowed, so one aggregated summary entry replaces the excess. It is rewritten in place as more overflow arrives, so a consumer must treat a later occurrence as superseding the earlier one.

### Declare what you emit (capability ⇄ stream consistency)

Six streaming capabilities each gate one event field or event type. Your declared `streaming` capabilities and the events you actually emit MUST agree — your stream may use *less* than you declare, but never *more*:

| If you emit…                     | …declare                                            |
| -------------------------------- | --------------------------------------------------- |
| a `partial` event | `streaming.emits_partials = FlagCap(supported=True)` |
| a `partial` with non-empty `stable_text` | `streaming.partial_stability = FlagCap(supported=True)` |
| a `supersede` event | `streaming.re_segments = FlagCap(supported=True)` |
| an `audio_processed_until` cursor | `streaming.audio_progress = FlagCap(supported=True)`              |
| a segment `start` or `end` | `streaming.timestamps.mode` ≠ `"none"` |
| per-word `words`                 | `streaming.word_timestamps = WordTimestampsCap(supported=True, …)` |
| a segment- or word-level `speaker` | `streaming.diarization = DiarizationCap(supported=True)` — add `always_on=FlagCap(supported=True)` if your model is architecturally unable to disable it |

One more declaration promises events instead of gating them. Declare `streaming.finality_level = FinalityCap(mode="closed")` only if every segment that reaches `final`, and that no `supersede` retires, also reaches `closed` before the session ends with `done`. Otherwise keep the default mode, `"final"`, which neither promises nor forbids a `closed` final.

The coherent **no-timestamp streaming profile** is the all-defaults combination, except that you declare `emits_partials` supported if you send `partial` events. Leave `partial_stability`, `timestamps` (mode `"none"`), `word_timestamps`, and `diarization` unsupported, and emit none of those fields (leave `stable_text` out of every `partial`, and omit `words` and `speaker`; declare `audio_progress` if you emit `audio_processed_until`). A mismatch — for example, declaring `partial_stability` unsupported while emitting a `partial` with non-empty `stable_text` — is a capability⇄stream desync a client trusting your capabilities would mishandle.

The session checks every event it forwards against your engine's effective capabilities while it runs. It judges the event as you sent it, so a `partial` whose stable text the boundary repair empties still counts as carrying stable text. An event that the session suppresses for another reason is not checked. `EngineBase.start_transcription` gives the session your capabilities. An engine that implements the protocol without deriving from `EngineBase` does not. For such an engine, the reference server calls `bind_session_capabilities(session, engine)` for every session it opens, and any other caller can do the same (`from standard_asr import bind_session_capabilities`). That function reads the engine's `effective_capabilities` the ordinary way, so an attribute your engine serves through `__getattr__` counts. If reading it raises an exception, including an `AttributeError` raised inside an `effective_capabilities` property, the exception propagates, and the reference server opens no session. For an `AttributeError` or any other exception the server has no specific handling for, the client gets an `error` frame with code `internal_error`. The function uses the engine's `declared_capabilities` only when the engine has no `effective_capabilities` attribute, or the attribute does not hold a `DeclaredCapabilities`. A session that never gets the capabilities is not checked. The session records `stream_exceeds_emits_partials`, `stream_exceeds_partial_stability`, `stream_exceeds_re_segments`, `stream_exceeds_audio_progress`, `stream_exceeds_timestamps`, `stream_exceeds_word_timestamps`, or `stream_exceeds_diarization` for a field or event type your declaration does not back. It records `finality_level_not_reached` when you declare finality mode `"closed"` and the session reaches `done` while a segment is still `final` and not `closed`. Each code is recorded once per session, in `session.diagnostics()`. Because of this check, the session never removes the field or drops the event: removing the field would hide your bug, and dropping a `supersede` would leave the retired segments in place and duplicate the replacement text. With `strict_lifecycle=True`, the first mismatch ends the session instead, with a terminal `error` event whose code is `engine_error`. The contract is still yours to keep.

To find a mismatch before you publish, record a real session and assert it with `check_event_sequence(events, capabilities=engine.declared_capabilities)`. It runs the same check and reports each code once, as an error, however many events break the declaration. Capabilities that declare no `streaming` domain are checked against the default `StreamingCapabilities()`, in which none of the capabilities in the table is supported. That call checks your declared capabilities, and the session checks your effective ones. Your effective capabilities can be narrower, so a configuration that turns a capability off can produce a session diagnostic even when the compliance report is clean.

### Testing: assert invariants, not partial counts

Partials are **lossy under backpressure**. When the consumer reads slower than you produce, the base coalesces pending partials for a segment (spec ST.6.4). So the number of `partial` events a test observes is non-deterministic: the same engine may surface five partials or none, purely by timing. A test asserting `len(partials) == N` is therefore flaky. Assert the **invariants** instead:

- the final/reduced text is correct (`session.result()`, or the `final` event);
- the stable-text rules hold — use the exported `assert_stable_text_invariant(events)` helper. It replays the events through the session's lifecycle guard and raises for the stable-text diagnostics that replay produces. Those report stable text that is rewritten or shrinks, a `partial` whose stable text ends inside a combining character sequence or right after a zero width joiner, and, when the events include `done`, a segment that has stable text and is still open there. A legal `closed` restatement is not a violation, and a `final`'s own boundary is not checked. An event that the guard rejects first for another reason, such as a `partial` after the segment's `final`, never reaches the stable-text rules, so the helper does not raise for it. Like the runtime check, it does not prove that stable text ends between user-perceived characters. It tolerates any surviving partial count, and (unlike `check_event_sequence`) does not require a terminal event, so it also applies to a mid-stream slice. It checks one session: it stops at the first terminal event. For a full check of a recorded sequence, use `check_event_sequence`.

## Publish

Register an entry point under `standard_asr.models` (see [`plugin-entry-points.md`](./plugin-entry-points.md)).

**The engine class MUST be resolvable without calling the entry point.** Capabilities and the params schema are read from class-level `ClassVar`s *without instantiating or authenticating* the engine (CLI `show`, the registry, REST `GET /v1/capabilities/{model}` and `/v1/params-schema/{model}`). Two forms satisfy that, and the compliance suite accepts either:

- **The entry point is the engine class itself.** Nothing more is needed — the class is returned directly.
- **The entry point is a factory function.** Then its return annotation MUST name your concrete engine class (`-> MyEngine`), **not** the `StandardASR` protocol: only the annotation is read, and a Protocol has no readable `ClassVar`s, so it breaks instantiation-free discovery.

What compliance actually checks is the outcome, not the form: it reports `class_metadata_unreadable` when the class cannot be resolved either way — an unannotated factory, or one annotated with the protocol.

Check with:

```bash
standard-asr compliance run
standard-asr doctor
```
