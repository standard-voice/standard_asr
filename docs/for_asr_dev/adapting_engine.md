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
6. (Streaming) override
   `_start_transcription(*, gated_params, audio_format, prepared_audio)`
   returning a `TranscriptionSession` subclass.

## Minimal batch engine

```python
from typing import ClassVar, Literal
from standard_asr.engine import (
    BaseConfig, BaseProperties, BatchCapabilities, DeclaredCapabilities,
    EngineBase, FlagCap, InputKind, LanguageCaps, PreparedAudio,
    RuntimeParams, TranscriptionResult,
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
- **`word_timestamps.granularities` declares what you can honestly *deliver*, not
  which native API switch exists.** Declare every granularity your engine can
  serve — including ones that come for free. If your model emits per-segment
  start/end on every run (most do), declare `"segment"` even when there is no
  separate "segment mode" knob; otherwise the standard layer rejects the cheapest,
  always-satisfiable request as a false incompatibility. Then map each granularity
  precisely (e.g. only `"word"` flips your forced-alignment pass on; a `"segment"`
  request must not back-fill word-level data — `words=None` means "not requested").

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

### Session establishment — the base does the gating for you

You override `_start_transcription(...)`, **not** the public `start_transcription`.
The base `start_transcription` is a template method (symmetric to
`transcribe` / `_transcribe`): it runs the standard streaming pipeline and then
calls your hook. Before your hook runs, the base has already:

- enforced the `audio_format` / `audio` mutual exclusion (ST §3.1) via
  `ensure_stream_inputs_exclusive`;
- validated the language config (LANG R1 / IC.6);
- run the **fail-closed** wire-format check (`ensure_stream_format_supported`) on
  both encoding and sample rate. It rejects an `audio_format.encoding` not in your
  declared `wire_encodings` (so an undeclared encoding is not misframed as PCM and
  silently mistranscribed) and a wire `sample_rate` you do not accept: per spec R7's
  v1 note the standard does **not** resample streaming wire frames in v1 (only the
  batch `transcribe` path resamples), so an unreachable wire rate is a loud error.
  When `required_input_sample_rate` is set, the wire rate MUST equal it — even when
  another rate appears in `accepted_sample_rates` (that list describes the batch
  path, which resamples to the required rate before your engine; unresampled wire
  frames at any other rate would be misread). Otherwise the rate is accepted when
  `accepted_sample_rates` is `"any"` or when it is in that concrete list.
  (Standard-layer streaming resampling is a deferred capability; this guard becomes
  a resample once it lands.)
- **gated the runtime parameters** against your `streaming` capabilities — provider
  `provider_params` swap-safety (Runtime R3: a wrong `provider_params` type always
  raises `InvalidProviderParamError`), capability gating (R2), guidance degradation
  (R4) — and resolved the language axis. The gating / language **diagnostics** are
  attached to the returned session and surface through `session.diagnostics()`.
- for the **whole-input** path (`audio=...`, e.g. OpenAI-style streaming output),
  run that complete input through the **same** audio negotiation/conversion pipeline
  as batch `transcribe` and hand your hook the result as `prepared_audio` (a
  `PreparedAudio` already in one of your `accepted_input` shapes, with its
  conversion diagnostics attached to the session). For the incremental
  `audio_format=...` path there is no whole input, so `prepared_audio` is `None`.

Your hook receives the **already-gated, frozen** `gated_params` (spec R5: streaming
params are frozen at `start_transcription` and MUST NOT change mid-stream). Use them
directly — do not re-gate or re-accept raw params. The signature is
keyword-only: `gated_params`, `audio_format` (the wire format, or `None`), and
`prepared_audio` (the negotiated whole input, or `None`).

```python
def _start_transcription(self, *, gated_params, audio_format, prepared_audio):
    # Guards + gating + (whole-input) audio prep already ran in the base.
    # gated_params is frozen (R5); prepared_audio is None for the incremental
    # audio_format path and a PreparedAudio for the whole-input audio path.
    return MySession(gated_params, ...)
```

## Credentials & environment fallback (IC.4)

Build your config with `Config.from_env(engine_id, **explicit)` instead of the
bare constructor. Unset fields fall back to
`STANDARD_ASR_<ENGINE>__<FIELD>` environment variables (note the **double
underscore** separating the engine and field segments; explicit args win), and
credentials are wrapped in `SecretStr` by construction — never passed around as
plaintext. Put secrets (`api_key`, tokens) in `SecretStr` fields via
`secret_field()`; keep non-secret routing (`base_url`, `region`) plain. A
composite field (e.g. `default_candidate_languages: list[str]`) takes its env
value as JSON (`'["en","ja"]'`).

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
emits the `progress(reconnect)` event; it emits a trailing **non-terminal**
`content_lost` error (`recoverable=true` — a fidelity warning; the session
stays alive and events keep flowing) **only if you pass `content_lost=True`** — your own
determination that the reconnect + replay could not cover the gap and
unreplayable audio was permanently lost. The base does **not** infer loss from
rolling-buffer eviction (a live ring is always evicting, so that would falsely
claim loss on every long session); you decide, because only you know whether the
replay actually bridged the gap.

**`error` events fail safe to terminal.** An `error` event with `recoverable`
unset is normalized to `recoverable=false` (terminal) at construction: unknown
recoverability must not leave consumers waiting on a stream that may never
continue. If you emit an advisory, non-fatal error (the session keeps going),
set `recoverable=True` explicitly — otherwise your event ends the session.

**Surface non-fatal notes via `emit_diagnostic`.** Call
`self.emit_diagnostic(code=..., message=..., level="info"|"warning")` from
`_produce` to report a best-effort degradation, an assumed parameter, or a lossy
fallback through the session's `diagnostics()` channel — the streaming
counterpart of the batch path's `result.diagnostics`. It is bounded (spec
ST.6.4, like the guard's own diagnostics) and the server forwards it to a WS
client as a mid-stream `diagnostics` frame. Keep `error` events for *fatal*
conditions.

> **Security:** a diagnostic is **engine-authored, client-facing output**, like
> the transcript itself. Its `message`/`param`/`provided`/`effective` fields are
> forwarded to (possibly unauthenticated) clients **verbatim and unredacted**.
> Never put a credential, API key, auth'd URL, or raw exception text in a
> diagnostic — route sensitive operator detail to `logging` instead. (The server
> *does* scrub an `error` event's `extra`, because that is auto-captured
> `str(exc)`; a diagnostic is content you chose, so its safety is yours.)

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

### Declare what you emit (capability ⇄ stream consistency)

Three streaming capabilities each gate one event field. Your declared
`streaming` capabilities and the events you actually emit **must agree** — your
stream may use *less* than you declare, but never *more*:

| If you emit…                     | …declare                                            |
| -------------------------------- | --------------------------------------------------- |
| a non-zero `stable_until`        | `streaming.word_stability = FlagCap(supported=True)` |
| an `audio_processed_until` cursor | `streaming.timestamps.mode` ≠ `"none"`              |
| per-word `words`                 | `streaming.word_timestamps = WordTimestampsCap(supported=True, …)` |

The coherent **no-timestamp streaming profile** is the all-defaults combination:
leave `word_stability`, `timestamps` (mode `"none"`), and `word_timestamps`
unsupported, and emit none of those fields (use `stable_until=0`, omit
`audio_processed_until` and `words`). A mismatch — e.g. declaring
`word_stability` unsupported while emitting `stable_until>0` — is a
capability⇄stream desync a client trusting your capabilities would mishandle.
Record a real session and assert it with
`check_event_sequence(events, capabilities=engine.declared_capabilities)`; the
cross-check fails on any field your declaration does not back (codes
`stream_exceeds_word_stability` / `stream_exceeds_timestamps` /
`stream_exceeds_word_timestamps`). The standard layer does **not** clamp these at
runtime — clamping would hide the bug; the contract is yours to keep.

### Testing: assert invariants, not partial counts

Partials are **lossy under backpressure**: when the consumer reads slower than you
produce, the base coalesces pending partials for a segment (spec ST.6.4), so the
number of `partial` events a test observes is non-deterministic — the same engine
may surface five partials or none, purely by timing. A test asserting
`len(partials) == N` is therefore flaky. Assert the **invariants** instead:

- the final/reduced text is correct (`session.result()`, or the `final` event);
- the partials form monotonic, never-rewritten prefixes — use the exported
  `assert_prefix_invariant(events)` helper, which checks exactly that (a frozen
  `text[:stable_until]` is never rewritten and `stable_until` never regresses),
  tolerates any surviving partial count, and (unlike `check_event_sequence`) does
  not require a terminal event, so it also applies to a mid-stream slice.

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
standard-asr compliance run
standard-asr doctor
```
