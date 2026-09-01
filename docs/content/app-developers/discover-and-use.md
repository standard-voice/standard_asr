---
title: "Discover & Use"
---

# Discover and use an engine (app developers)

> Authoritative reference: [`docs/content/specification/protocol.md`](../specification/protocol.md).
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

### Inspect and acquire inference artifacts

Engines report the persistent artifacts needed by a configured
inference path:

```python
from standard_asr import ArtifactAcquisitionError

report = engine.artifact_status()
for requirement in report.requirements:
    for action in requirement.required_actions:
        print(action.message, action.url or "")

if any(
    requirement.state != "ready" and requirement.can_acquire_now
    for requirement in report.requirements
):
    try:
        report = engine.acquire_artifacts()
    except ArtifactAcquisitionError as exc:
        # Blocked, not broken: a license to accept, a sign-in, downloads turned off.
        if exc.report is not None:
            report = exc.report
```

Status inspection is side-effect-free. Explicit acquisition never fabricates a
transcription request. A manual prerequisite, such as accepting terms or
providing files, appears as a structured `ArtifactAction`; an engine that can
only acquire during first inference reports that limitation explicitly.
`acquire_artifacts()` can still raise after completing runnable work when a
different required artifact remains blocked. The error carries the final
report and any newly discovered `required_actions`, which is why the sample
reads them from the exception instead of parsing its message.

These facts do not classify an engine as local or remote. See
[Inference artifacts](../reference/artifacts.md) for states, refresh behavior,
progress, and errors.

## 3. Pass audio — whatever you have

`transcribe` accepts a discriminated `AudioInput` union; bare values are coerced.
A bare `str` is **always** a local path (never a URL — wrap URLs explicitly).

```python
from standard_asr import AudioArray, AudioUrl

engine.transcribe("meeting.mp3")  # -> AudioPath
engine.transcribe((samples, 16000))  # -> AudioArray(samples, sr)
engine.transcribe(AudioUrl("https://.../a.flac"))  # explicit URL (engine-fetched)
```

The standard layer negotiates and converts to whatever the engine accepts
(decode, encode-to-WAV, read-file, resample) and reports any lossy step in
`result.diagnostics`.

## 4. Per-request parameters (portable + escape hatch)

```python
from standard_asr import DIARIZE, RuntimeParams, WordTimestampGranularity

result = engine.transcribe(
    "meeting.mp3",
    RuntimeParams(
        language="en",  # or "auto"
        word_timestamps=WordTimestampGranularity.WORD,
        diarization=DIARIZE,  # "who said what" (presence = enable)
        prompt="Q3 budget review.",  # free-text guidance
        phrase_hints=["Anthropic", "Claude"],  # term boosting
    ),
)
```

`diarization` is an on/off request marker: pass `DIARIZE` (or
`DiarizationRequest()`) to enable it, leave it `None` to skip it. Gate it first
with `engine.supports("batch.diarization")`.

Engine-specific options go through `provider_params` (typed, swap-safe — passing
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
    print(seg.start, seg.end, seg.speaker, seg.text)  # speaker: label when diarized, else None

from standard_asr import to_srt, to_vtt

with open("out.srt", "w", encoding="utf-8") as f:
    f.write(to_srt(result))
```

A segment renders as a *visible* cue only if it carries a measured `start`/`end`
span. Some engines omit timestamps — check `seg.timestamp_status`. The span must
also survive the output grid: a span that quantizes to zero milliseconds fails,
because players silently drop `T --> T` cues. When a segment cannot render, the
renderers raise `SubtitleRenderingError`. They do not silently drop text, hide
text, or fabricate timing. To choose the loss yourself, pass
`on_unrenderable="omit"` to keep only the renderable cues, or `"collapse"` for
one whole-text cue.

`segment.speaker` carries the speaker label when diarization was requested and
supported (`word.speaker` gives the same detail at word level). Engines whose
diarization is `always_on` may label speakers even without a request. A `None`
label means "not attributed", never "unsupported" — capabilities answer support.

## 7. Streaming

```python
fmt = engine.recommended_wire_format()  # the engine's preferred PCM wire format
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

A synchronous bridge (`SyncSession`) is available if you cannot use `async`.
