---
title: "Discover & Use"
---

# Discover and use an engine (app developers)

> Authoritative reference: [`docs/content/specification/protocol.md`](../specification/protocol.md). This guide shows the common app-developer flow on the current API.

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

`create()` also verifies the plugin and this core share a protocol generation, so a stale installation fails here with a typed `ProtocolCompatibilityError` naming the fix. If you obtain an engine some other way -- direct construction, your own loader, dependency injection -- run that check yourself once before using the object:

```python
from standard_asr import require_engine_protocol

require_engine_protocol(engine)  # raises ProtocolCompatibilityError on a mismatch
```

### Inspect and acquire inference artifacts

Engines report the persistent artifacts needed by a configured inference path:

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

Status inspection is side-effect-free. Explicit acquisition never fabricates a transcription request. A manual prerequisite, such as accepting terms or providing files, appears as a structured `ArtifactAction`; an engine that can only acquire during first inference reports that limitation explicitly. `acquire_artifacts()` can still raise after completing runnable work when a different required artifact remains blocked. The error carries the final report and any newly discovered `required_actions`, which is why the sample reads them from the exception instead of parsing its message.

These facts do not classify an engine as local or remote. See [Inference artifacts](../reference/artifacts.md) for states, refresh behavior, progress, and errors.

## 3. Pass audio — whatever you have

`transcribe` accepts a discriminated `AudioInput` union; bare values are coerced. A bare `str` is **always** a local path (never a URL — wrap URLs explicitly).

```python
from standard_asr import AudioArray, AudioUrl

engine.transcribe("meeting.mp3")  # -> AudioPath
engine.transcribe((samples, 16000))  # -> AudioArray(samples, sr)
engine.transcribe(AudioUrl("https://.../a.flac"))  # explicit URL (engine-fetched)
```

The standard layer negotiates and converts to whatever the engine accepts (decode, encode-to-WAV, read-file, resample) and reports any lossy step in `result.diagnostics`.

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

`diarization` is an on/off request marker: pass `DIARIZE` (or `DiarizationRequest()`) to enable it, leave it `None` to skip it. Gate it first with `engine.supports("batch.diarization")`.

Engine-specific options go through `provider_params` (typed, swap-safe — passing the wrong engine's params raises `InvalidProviderParamError`).

## 5. Check capabilities before relying on a feature

```python
if engine.supports("batch.word_timestamps"):
    ...
```

Missing capabilities are **fail-closed** (`supports(...)` returns `False`).

`supports()` tells you the engine has a feature, not that you can ask for it. For example, `supports("streaming.re_segments")` is `True` for an engine that may replace segments it already sent, and no parameter asks for that. The capability tree answers the second question. `can_request(path)` is `True` only for a supported feature that a request can ask for. `kind_at(path)` returns the node's kind: `"requestable"`, `"behavior"`, `"limit"` (a `constraints` submodel), or `None` for a container such as `"batch"` or a missing path. A behavior node describes a property, a behavior, or a promise of the engine that no standard `RuntimeParams` field requests. Call them on `engine.declared_capabilities`, or on `registry.engine_class(key).declared_capabilities` before you create the engine. A configured engine can support less than its class declares, so also check `engine.supports(path)` before you send the request.

## 6. Use the transcription result

### What you get

`transcribe()` returns a `TranscriptionResult`, and so does a streaming session's `result()`. The data type is the same for batch and streaming. It is also the same for every engine, whatever its capabilities. `text` is the only required field; any other field the engine cannot give is `None`, except `diagnostics` and `extra`, which default to empty.

```python
class TranscriptionResult:
    text: str  # the whole transcript, required
    detected_language: str | None = None  # BCP-47 tag, in auto mode
    language_confidence: float | None = None  # 0 to 1
    duration: float | None = None  # audio length in seconds
    segments: list[Segment] | None = None  # in reading order
    words: list[Word] | None = None  # every Word, in segment order
    channels: list[ChannelResult] | None = None  # one entry per channel
    diagnostics: list[Diagnostic] = []  # non-fatal reports
    extra: dict[str, JsonValue] = {}  # engine-specific data


class Segment:
    start: float | None  # must be given; None if not measured
    end: float | None  # must be given; None if not measured
    text: str  # includes its boundary spaces
    words: list[Word] | None = None  # this segment's own words
    speaker: str | None = None
    channel: int | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    temperature: float | None = None
    compression_ratio: float | None = None
    extra: dict[str, JsonValue] = {}


class Word:
    start: float  # seconds
    end: float  # seconds
    text: str
    probability: float | None = None  # 0 to 1
    logprob: float | None = None
    speaker: str | None = None
    channel: int | None = None
    extra: dict[str, JsonValue] = {}


class ChannelResult:
    channel: int  # channel index
    text: str
    segments: list[Segment] | None = None
    words: list[Word] | None = None


class Diagnostic:
    level: Literal["info", "warning"] = "info"
    code: str  # stable, such as "flat_words_missing"
    message: str
    param: str | None = None  # the parameter concerned
    provided: JsonValue = None  # the value the application gave
    effective: JsonValue = None  # the value that took effect
```

### Which field does what

Read `text` for the transcript. Read `segments` for subtitles and timelines; `to_srt(result)` and `to_vtt(result)` render the top-level segments. A segment renders as a visible cue only if it carries a measured `start`/`end` span. Some engines omit timestamps, so check `seg.timestamp_status`. The span must also survive the output grid: a span that rounds to zero milliseconds fails, because players silently drop `T --> T` cues. When a segment cannot render, the renderers raise `SubtitleRenderingError`. They do not silently drop text, hide text, or fabricate timing. To choose the loss yourself, pass `on_unrenderable="omit"` to keep only the renderable cues, or `"collapse"` for one whole-text cue.

```python
from standard_asr import to_srt, to_vtt

with open("out.srt", "w", encoding="utf-8") as f:
    f.write(to_srt(result))
```

`segment.speaker` carries the speaker label when diarization was requested and supported, and `word.speaker` carries a word's own label. Engines whose diarization is `always_on` may label speakers even without a request. A `None` label means "not attributed", never "unsupported" — capabilities answer support.

Read `words` for word-level timing. Use `channels` to read one audio channel at a time, such as one side of a two-channel call. The top level merges every channel, so an application that ignores `channels` loses nothing.

### What a missing field means

A field that is `None` was not requested or does not apply. A field that is `[]` was requested and is empty, as for silence. In streaming, `session.result()` has `words=None` when none of its segments carries a `words` list, even in a silent session that requested word timestamps. Whether the engine supports a feature is a question for its capabilities (section 5). Do not rebuild one field from another: make no segments from word times, and no word times from a segment's span. The standard layer does neither, because only the engine knows which words belong to a segment.

### How the fields relate

The result's own `words` and each segment's `words` hold the same `Word` values. The result's list follows segment order, not a time merge across channels. A segment may leave its `words` as `None`. A word whose `speaker` is `None` takes its segment's `speaker`, but no list fills that in. So a word's speaker is `word.speaker` when that is not `None`, and otherwise the `speaker` of the segment that holds the word. Find the word in its segment's `words`, and do not guess its segment from timestamps. If the segments carry no `words`, the result does not say which segment holds a word. Each channel's `segments` and `words` are the top-level ones of that channel, with the same values.

### What `result.diagnostics` tells you

`result.diagnostics` holds non-fatal reports: lossy conversions, dropped parameters, and problems with the result itself. Log or show the `warning` entries, and key any handling on `code`, as [Errors & diagnostics](./errors.md) describes. Two codes concern the fields above. Each appears once for the top level and once for each channel with the problem:

- `text_segments_mismatch`: the engine's `text` is not its segment texts joined. Both reach you unchanged, so `text` and the subtitles can differ.
- `flat_words_missing`: the engine left `words` as `None` beside segments that carry `words`. The standard layer filled `words` from those segments where it could, and the message says so.

For an engine that does not derive from `EngineBase`, the standard layer attaches neither diagnostic and fills nothing. Call `finalize_result()` on each result, as the reference server's batch endpoints and `standard-asr transcribe` do. A result that already went through it comes back unchanged, so you can call it on every result:

```python
from standard_asr import finalize_result

result = finalize_result(engine.transcribe("meeting.mp3"))
```

[Spec TR.1 and TR.4](../specification/protocol.md#transcription-result) state these rules, and the [API reference](../reference/results.md) documents each field.

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
