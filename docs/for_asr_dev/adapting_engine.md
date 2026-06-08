# Adapting an ASR engine to Standard ASR (engine authors)

> Authoritative reference: [`docs/spec/specification.md`](../spec/specification.md).
> Entry-point rules: [`plugin_entrypoints.md`](plugin_entrypoints.md).

You implement **one** class. The standard layer gives you audio-input
negotiation, conversion, resampling, parameter gating, diagnostics, the CLI, the
web server, and the compliance suite — for free.

## The contract

Subclass `EngineBase` and provide:

1. `properties: ClassVar[BaseProperties]` — static identity and I/O boundaries
   (`accepted_input`, `native_sample_rate`, `accepted_sample_rates`,
   `selectable_languages`, …).
2. `declared_capabilities: ClassVar[DeclaredCapabilities]` — what you support,
   per mode (`batch` / `streaming`). Omit what you don't support (fail-closed).
3. `provider_params_type: ClassVar[type[ProviderParams] | None]` — your typed
   escape-hatch model, or `None`.
4. `__init__` — capture config only. **Keep it pure**: no filesystem, GPU, or
   network (spec IC.9). Load weights lazily in `_ensure_model_loaded`.
5. `_transcribe(prepared, params) -> TranscriptionResult` — run your model on
   already-negotiated audio (`prepared.kind` is one of your `accepted_input`).
6. (Streaming) override `start_transcription(...)` returning a
   `TranscriptionSession` subclass.

## Minimal batch engine

```python
from typing import ClassVar, Literal
from standard_asr import (
    BaseConfig, BaseProperties, EngineBase, InputKind,
    PreparedAudio, RuntimeParams, TranscriptionResult,
)
from standard_asr.capabilities import (
    BatchCapabilities, DeclaredCapabilities, FlagCap, LanguageCaps,
)

class MyConfig(BaseConfig[Literal["my-engine"]]):
    engine: Literal["my-engine"] = "my-engine"

class MyProps(BaseProperties):
    engine_id: str = "my-engine"
    model_name: str = "base"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] = [16000]
    selectable_languages: list[str] = ["en", "auto"]
    detectable_languages: list[str] = ["en"]

class MyEngine(EngineBase):
    properties: ClassVar[BaseProperties] = MyProps()
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        )
    )

    def __init__(self, **kw: object) -> None:
        self.config = MyConfig()
        self._model = None

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        audio = prepared.array            # 16 kHz float32 mono, per Properties
        text = my_model_infer(audio)      # your code
        return TranscriptionResult(text=text, detected_language=params.language)
```

## Map parameters

- Portable standard set is gated for you against `declared_capabilities` before
  `_transcribe` is called: `language`, `candidate_languages`, `word_timestamps`,
  `prompt`, `phrase_hints`. Map them onto your model's native arguments.
- Engine-specific knobs → a `ProviderParams` subclass set as
  `provider_params_type`. Wrong-engine params raise `InvalidProviderParamError`.
- Resolve the language with `standard_asr.language.effective_language(...)`.

## Audio you receive

`prepared` is already in one of your `accepted_input` shapes:

- `InputKind.ARRAY` → `prepared.array` (float32, `prepared.sample_rate`)
- `InputKind.ENCODED_FILE` → `prepared.path`
- `InputKind.ENCODED_BYTES` → `prepared.data`
- `InputKind.FETCHABLE_URL` → `prepared.url`

You never write decode/resample/encode glue — declare `accepted_input` and the
standard layer delivers the right shape (and attaches conversion diagnostics).

## Streaming

Subclass `TranscriptionSession`, implement async `_produce()` (read fed audio via
`self.audio_chunks()`, yield `TranscriptionEvent` objects). The base provides
`feed`/`send_audio`/`end_audio`, backpressure, the done-timeout, and the sync
bridge — you only write `async`. See the spec §ST for the event model
(`partial`/`final`/`supersede`/`progress`/`done`/`error`) and the `stable_until`
rules.

### Session-establishment guards

Override `start_transcription(...)` and call the two shared guards first, before
opening engine resources:

- `self.ensure_stream_inputs_exclusive(audio_format, audio)` — enforces the
  `audio_format` / `audio` mutual exclusion (ST §3.1).
- `self.ensure_stream_format_supported(audio_format)` — **fail-closed** wire-format
  check on both encoding and sample rate. It raises `UnsupportedFeatureError` when
  `audio_format.encoding` is not in the engine's declared `wire_encodings`, so an
  encoding you never declared is rejected up front instead of being misframed as PCM
  and silently mistranscribed. It **also** rejects a wire `sample_rate` the engine
  does not accept: per spec R7's v1 note the standard does **not** resample streaming
  wire frames in v1 (only the batch `transcribe` path resamples), so an unreachable
  wire rate is a loud error rather than a silent mistranscription. The rate is
  accepted when `accepted_sample_rates` is `"any"`, when it is in that concrete list,
  or when it equals `required_input_sample_rate`. (Standard-layer streaming resampling
  is a deferred capability; this guard becomes a resample once it lands.)

```python
def start_transcription(self, *, audio_format=None, params=None, audio=None):
    self.ensure_stream_inputs_exclusive(audio_format, audio)
    if audio_format is not None:
        self.ensure_stream_format_supported(audio_format)   # fail-closed: encoding + rate
    return MySession(...)
```

## Credentials & environment fallback (IC.4)

Build your config with `Config.from_env(engine_id, **explicit)` instead of the
bare constructor. Unset fields fall back to
`STANDARD_ASR_<ENGINE>_<FIELD>` environment variables (explicit args win), and
credentials are wrapped in `SecretStr` by construction — never passed around as
plaintext. Put secrets (`api_key`, tokens) in `SecretStr` fields via
`secret_field()`; keep non-secret routing (`base_url`, `region`) plain.

```python
def __init__(self, **kwargs):
    self.config = MyConfig.from_env("my-engine", **kwargs)   # IC.4
```

## Streaming responsibilities (what the base does vs you)

The base `TranscriptionSession` owns the pump, backpressure (bounded buffers),
the done-timeout/idle deadlines, the sync bridge, lifecycle suppression
(`strict_lifecycle=True` to raise instead of diagnose), and `stable_until`
monotonicity clamping. **You** must: emit cumulative/replace `text`; set
`stable_until` conservatively (0 if you have no right-context); and for
reconnect, detect the disconnect, re-establish, replay `self.replay_buffer()`,
keep `segment_id`/timestamps/language continuous, and call
`self.note_reconnect(gap_start, gap_end, content_lost=...)`. The base always
emits the `progress(reconnect)` event; it emits a trailing terminal
`content_lost` error **only if you pass `content_lost=True`** — your own
determination that the reconnect + replay could not cover the gap and
unreplayable audio was permanently lost. The base does **not** infer loss from
rolling-buffer eviction (a live ring is always evicting, so that would falsely
claim loss on every long session); you decide, because only you know whether the
replay actually bridged the gap.

### Sequence invariants the guard enforces for free

Beyond lifecycle transitions and monotonic `stable_until`, the base `_LifecycleGuard`
also enforces two further per-stream invariants on every event you yield, so a
slipped engine still cannot emit a wrong transcript:

- **Monotonic audio cursor** — a decreasing `audio_processed_until` is clamped to
  the prior value (the cursor never moves backwards; ST §4.1), with an
  `audio_cursor_decreased` diagnostic (or a raise in `strict_lifecycle`).
- **Frozen-prefix immutability** — an event that rewrites a segment's
  already-frozen prefix (`text[:stable_until]` changed) is suppressed with a
  `frozen_prefix_rewritten` diagnostic (the frozen prefix is immutable; ST §4.2).

The full set of standard-layer diagnostic codes the guard can emit (read them off
the session with `session.diagnostics()`):

- `stable_until_clamped` — a decreasing or invalid `stable_until` was clamped.
- `audio_cursor_decreased` — a decreasing `audio_processed_until` was clamped.
- `frozen_prefix_rewritten` — an event rewriting a frozen prefix was suppressed.
- `lifecycle_after_terminal` — a `partial`/`final` after the segment became
  `closed`/`superseded` was suppressed.
- `lifecycle_partial_after_final` — a `partial` after the segment's `final` was
  suppressed.
- `lifecycle_closed_superseded` — a `supersede` retiring a `closed` segment was
  suppressed.

## Publish

Register an entry point under `standard_asr.models` (see
[`plugin_entrypoints.md`](plugin_entrypoints.md)).

**The entry-point factory MUST return a concrete engine class** (annotate its
return type as your engine class, e.g. `-> MyEngine`), **not** the `StandardASR`
protocol. Capabilities and the params schema are read from class-level
`ClassVar`s *without instantiating or authenticating* the engine (CLI
`models show`, the registry, REST `GET /v1/capabilities` and
`/v1/params-schema`); a Protocol return type has no readable `ClassVar`s, so it
breaks instantiation-free discovery. The compliance suite enforces this.

Validate with:

```bash
standard-asr compliance entrypoints
standard-asr doctor
```
