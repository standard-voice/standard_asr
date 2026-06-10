# Discover and use an engine (app developers)

> Authoritative reference: [`docs/spec/specification.md`](../spec/specification.md).
> This guide shows the common app-developer flow on the current API.

## 1. Discover installed engines

Engines are pip-installable plugins discovered via entry points — zero config.

```python
from standard_asr import discover_models

registry = discover_models()
for name in registry.names():
    spec = registry.spec(name)
    print(name, spec.engine_id, spec.model_name)
```

## 2. Create an engine

```python
engine = registry.create("faster-whisper/large-v3", device="cpu")
```

## 3. Pass audio — whatever you have

`transcribe` accepts a discriminated `AudioInput` union; bare values are coerced.
A bare `str` is **always** a local path (never a URL — wrap URLs explicitly).

```python
from standard_asr import AudioArray, AudioUrl

engine.transcribe("meeting.mp3")                   # -> AudioPath
engine.transcribe((samples, 16000))                # -> AudioArray(samples, sr)
engine.transcribe(AudioUrl("https://.../a.flac"))  # explicit URL (engine-fetched)
```

The standard layer negotiates and converts to whatever the engine accepts
(decode, encode-to-WAV, read-file, resample) and reports any lossy step in
`result.diagnostics`.

## 4. Per-request parameters (portable + escape hatch)

```python
from standard_asr import RuntimeParams, WordTimestampGranularity

result = engine.transcribe(
    "meeting.mp3",
    RuntimeParams(
        language="en",                              # or "auto"
        word_timestamps=WordTimestampGranularity.WORD,
        prompt="Q3 budget review.",                 # free-text guidance
        phrase_hints=["Anthropic", "Claude"],       # term boosting
    ),
)
```

Engine-specific knobs go through `provider_params` (typed, swap-safe — passing
the wrong engine's params raises `InvalidProviderParamError`).

## 5. Check capabilities before relying on a feature

```python
if engine.supports("batch.word_timestamps"):
    ...
```

Missing capabilities are **fail-closed** (`supports(...)` returns `False`).

## 6. Use the result (constant shape)

```python
print(result.text)
print(result.detected_language, result.duration)
for seg in result.segments or []:
    print(seg.start, seg.end, seg.text)

from standard_asr import to_srt, to_vtt
open("out.srt", "w").write(to_srt(result))
```

## 7. Streaming

```python
fmt = engine.recommended_wire_format()   # the engine's preferred PCM wire format
async with engine.start_transcription(audio_format=fmt) as session:
    session.feed(microphone)
    async for event in session:
        if event.type == "partial":
            show(event.segment_id, event.text)
        elif event.type == "final":
            commit(event.segment_id, event.text)
        elif event.type == "supersede":
            for old in event.old_ids:
                remove(old)
```

A synchronous bridge (`SyncSession`) is available if you can't use `async`.
