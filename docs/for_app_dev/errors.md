# Errors & Diagnostics

Standard ASR follows "explicit > implicit": when something goes wrong, you get a
specific exception with machine-readable context -- never silent degradation.

## Exception hierarchy

Every exception inherits from `StandardASRError`, so a single
`except StandardASRError` catches anything the framework throws:

```
StandardASRError
+-- StructuredError (adds .param / .detail / .engine_id)
|   +-- ConfigError            invalid config (bad language, missing credential, ...)
|   +-- TranscriptionError     engine failed during transcription
|   +-- UnsupportedFeatureError  unsupported parameter in strict mode
|   +-- InvalidProviderParamError  wrong engine's provider_params passed
+-- AudioProcessingError       audio decode / size / sample-rate failure
|   +-- IncompatibleAudioInputError  no conversion path exists
|   +-- FFmpegNotFoundError    FFmpeg needed but not on PATH
|   +-- FFprobeNotFoundError   FFprobe needed but not on PATH
+-- StreamClosedError          audio delivered to a closed session
+-- InvalidSessionUseError     session driven incorrectly (e.g. double-end)
+-- DiscoveryError             plugin discovery problem
    +-- EntrypointValidationError  bad entry-point name or metadata
    +-- FactoryLoadError          entry point failed to import / not callable
```

## When each exception fires

| Exception | When | Typical cause |
| --------- | ---- | ------------- |
| `ConfigError` | `create()` or `start_transcription()` | Missing API key, invalid language, bad pydantic validation. |
| `TranscriptionError` | `transcribe()` | Engine crashed or returned an invalid result. |
| `UnsupportedFeatureError` | `start_transcription()` or `transcribe()` (strict mode) | Requested word timestamps on an engine that does not support them. |
| `InvalidProviderParamError` | `transcribe()` or `start_transcription()` | Passed faster-whisper's `provider_params` to an OpenAI engine (swap-safety). |
| `AudioProcessingError` | `transcribe()` | Corrupt audio file, missing sample rate, unsupported format without `[audio]` extra. |
| `IncompatibleAudioInputError` | `transcribe()` | Passed a URL to an engine that only accepts arrays, and no conversion path exists. |
| `UnsafeAudioUrlError` | `transcribe()` | An `AudioUrl` failed the SSRF policy (non-HTTPS, private IP, etc.). |
| `StreamClosedError` | `session.feed()` / `session.send_audio()` | Sending audio after the session ended. |
| `EntrypointValidationError` | `discover_models()` (strict mode) | A plugin's entry-point name is malformed. |
| `FactoryLoadError` | `registry.engine_class()` / `registry.create()` | Plugin's entry point cannot be imported or the factory is misconfigured. |

## Structured error context

`StructuredError` subclasses carry machine-readable fields:

```python
try:
    engine.transcribe("audio.wav", RuntimeParams(word_timestamps="word"))
except UnsupportedFeatureError as exc:
    print(exc.param)       # "word_timestamps"
    print(exc.engine_id)   # "faster-whisper"
    print(exc.detail)      # human-readable explanation
```

These fields let you build programmatic error handling (e.g. fall back to another
engine when a feature is unsupported) without parsing message strings.

## Diagnostics (non-fatal)

Not every problem is an exception. In `best_effort` mode, unsupported parameters
are **dropped** with a structured `Diagnostic` instead of raising:

```python
result = engine.transcribe("audio.wav", RuntimeParams(word_timestamps="word"))
for diag in result.diagnostics:
    print(diag.code, diag.message)
    # "unsupported_parameter_ignored"  "word_timestamps is not supported; ignored."
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
