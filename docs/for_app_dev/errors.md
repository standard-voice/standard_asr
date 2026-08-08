# Errors & Diagnostics

Standard ASR follows "explicit > implicit": when something goes wrong, you get a
specific exception with machine-readable context -- never silent degradation.

## Exception hierarchy

Every exception inherits from `StandardASRError`, so a single
`except StandardASRError` catches anything the framework throws:

```
StandardASRError
+-- StructuredError (adds .param / .hint / .details)
|   +-- ConfigError            invalid config (bad language, bad value, ...)
|   |   +-- ConfigurationRequiredError  required config ABSENT (e.g. credential not set)
|   +-- TranscriptionError     engine failed during transcription
|   +-- UnsupportedFeatureError  unsupported parameter in strict mode
|   +-- InvalidProviderParamError  wrong engine's provider_params passed
+-- AudioProcessingError       audio decode / size / sample-rate failure
|   +-- IncompatibleAudioInputError  no conversion path exists
|   |   +-- UnsafeAudioUrlError   AudioUrl failed the SSRF policy (non-HTTPS, private IP)
|   +-- FFmpegNotFoundError    FFmpeg needed but not on PATH
|   +-- FFprobeNotFoundError   FFprobe needed but not on PATH
+-- EngineContractError        engine broke the protocol contract (async transcribe, bad declaration)
+-- SubtitleRenderingError     to_srt/to_vtt: segments lack measured timing (choose a policy)
+-- StreamClosedError          audio delivered to a closed session
+-- InvalidSessionUseError     session driven incorrectly (e.g. mixing feed() with send_audio())
+-- DiscoveryError             plugin discovery problem
    +-- EntrypointValidationError  bad entry-point name or metadata
    +-- FactoryLoadError          entry point failed to import / not callable
```

## When each exception fires

| Exception | When | Typical cause |
| --------- | ---- | ------------- |
| `ConfigError` | `create()` or `start_transcription()` | Invalid config value — bad pydantic validation, or a `default_language` that is malformed / not selectable. Fixable by whoever supplies the config. |
| `ConfigurationRequiredError` | `create()` / `from_env()` | A required field (e.g. an API key) is absent from both explicit config and the environment — set it and retry; compliance treats this as a skip, not a failure. |
| `TranscriptionError` | `transcribe()` | Engine crashed or returned an invalid result. |
| `UnsupportedFeatureError` | `start_transcription()` or `transcribe()` (strict mode) | Requested word timestamps on an engine that does not support them. |
| `InvalidProviderParamError` | `transcribe()` or `start_transcription()` | Passed faster-whisper's `provider_params` to an OpenAI engine (swap-safety). |
| `AudioProcessingError` | `transcribe()` | Corrupt audio file, missing sample rate, unsupported format without `[audio]` extra. |
| `IncompatibleAudioInputError` | `transcribe()` | Passed a URL to an engine that only accepts arrays, and no conversion path exists. |
| `UnsafeAudioUrlError` | `transcribe()` | An `AudioUrl` failed the SSRF policy (non-HTTPS, private IP, etc.). |
| `SubtitleRenderingError` | `to_srt()` / `to_vtt()` | A segment cannot render as a visible cue — no measured `start`/`end` span, or a span that quantizes to zero milliseconds on the output grid (players silently drop `T --> T` cues) — and `on_unrenderable` is the default `"error"`. Choose the loss explicitly: `"omit"` (renderable cues only) or `"collapse"` (one whole-text cue). Carries `.unrenderable` / `.total` counts. |
| `EngineContractError` | any synchronous protocol member, or `transcribe()` / `start_transcription()` on a language-declaration defect | The engine returned an awaitable (an `async def` implementation) or a wrong-typed value from a sync member (`transcribe()`, `start_transcription()`, `supports()`, ...), declared a language axis without a `default_language` (IC.6), or declared a malformed selectable/detectable tag. An engine/plugin bug — report it to the engine's author; nothing in your code is wrong. |
| `StreamClosedError` | `session.send_audio()` | Sending audio manually after `end_audio()` or after the session delivered a terminal event. (`feed()` never raises it: a managed source's post-terminal chunks are discarded by design.) |
| `InvalidSessionUseError` | `session.feed()` / `session.send_audio()` / iterating the session | Driving a still-live session incorrectly: mixing managed `feed()` with manual `send_audio()`/`end_audio()`, calling `feed()` twice, or iterating the event stream twice. The session is NOT closed — fix the calling code; do not rebuild the session. |
| `EntrypointValidationError` | `discover_models()` (strict mode) | A plugin's entry-point name is malformed. |
| `FactoryLoadError` | `registry.engine_class()` / `registry.create()` | Plugin's entry point cannot be imported or the factory is misconfigured. |

## Structured error context

`StructuredError` subclasses carry machine-readable fields:

```python
try:
    engine.transcribe("audio.wav", RuntimeParams(word_timestamps="word"))
except UnsupportedFeatureError as exc:
    print(exc.param)   # "word_timestamps" — the offending parameter
    print(exc.mode)    # "batch" — where the rejection happened
    print(exc.hint)    # actionable guidance, or None

try:
    registry.create("acme/model")
except ConfigError as exc:
    print(exc.param)    # the offending field, if a single one is implicated
    print(exc.details)  # sanitized [{"type", "loc", "msg"}, ...] entries
```

These fields let you build programmatic error handling (e.g. fall back to another
engine when a feature is unsupported) without parsing message strings. Every
`StructuredError` also carries `.details`, populated where structured context
exists — `ConfigError`, for example, puts the sanitized pydantic validation
entries there (`UnsupportedFeatureError` leaves it `None`).

## Diagnostics (non-fatal)

Not every problem is an exception. In `best_effort` mode, unsupported parameters
are **dropped** with a structured `Diagnostic` instead of raising:

```python
result = engine.transcribe("audio.wav", RuntimeParams(word_timestamps="word"))
for diag in result.diagnostics:
    print(diag.code, diag.message)
    # unsupported_parameter_ignored Ignored unsupported parameter 'word_timestamps' in batch mode (capability 'batch.word_timestamps' not supported).
```

Diagnostics surface:
- Parameter-gating decisions (dropped features, truncated prompts).
- Audio conversion steps (lossy resampling, format changes).
- Engine-authored messages during streaming (`session.diagnostics()`).

The `code` field is a stable, machine-readable identifier; the `message` is
human-readable. Applications should key on `code` for programmatic handling.

## Further reading

- [API Reference: exceptions](../reference/exceptions.md) -- full type
  signatures and docstrings.
- [Specification](../spec/specification.md) -- the normative error contract.
