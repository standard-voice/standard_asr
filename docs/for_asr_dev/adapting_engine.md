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

## Publish

Register an entry point under `standard_asr.models` (see
[`plugin_entrypoints.md`](plugin_entrypoints.md)). Validate with:

```
standard-asr compliance entrypoints
standard-asr doctor
```
