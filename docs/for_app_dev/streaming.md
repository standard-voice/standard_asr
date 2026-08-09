# Streaming

Standard ASR unifies the widely divergent streaming behaviors of 30+ ASR engines
under one event protocol. This guide covers everything an application developer
needs to build a robust streaming integration.

## Opening a session

Ask the engine for the PCM wire format it wants, then open a session:

```python
audio_format = engine.recommended_wire_format()

async with engine.start_transcription(audio_format=audio_format) as session:
    ...
```

`recommended_wire_format()` returns the engine's preferred sample rate and
encoding as an `AudioFormat`, or `None` when the engine declares no usable
positive sample rate (no bare-frame session can be opened then). If you need a
specific format (e.g. 8 kHz for telephony), construct one yourself -- the
engine will raise `UnsupportedFeatureError` if it cannot accept it. The
recommendation is derived from the engine's static Properties; whether a
bare-frame session can be opened at all is a capability question -- gate on
`engine.supports("streaming_input")` first.

> **Known pre-1.0 limitation.** The recommendation is a format the engine's
> session-establishment *validator* accepts -- for the rare self-managed-wire
> engine (an engine that manages its own wire format and opens sessions with
> a bare `start_transcription()`, taking no `audio_format` at all), that is
> not the same thing as the right way to open the session. How such engines
> declare their transport is being settled in the capability-ontology ADR
> ([#45](https://github.com/standard-voice/standard_asr/issues/45)); until
> then, follow the engine's own documentation for the no-argument open path.

For whole-input streaming (the engine streams *output* over a complete audio
file), pass `audio=` instead of `audio_format=`:

```python
async with engine.start_transcription(audio="meeting.wav") as session:
    async for event in session:
        ...
```

## Feeding audio

For live-input streaming there are two mutually exclusive input modes:

**Managed mode** -- hand the session an iterable of PCM byte chunks and let it
drive the input side for you:

```python
session.feed(microphone)    # any sync or async iterable of bytes chunks
```

`feed()` consumes the source and signals end-of-input automatically when the
iterable finishes. Do **not** call `end_audio()` yourself in this mode -- a
session is owned by exactly one input mode, and mixing them raises
`InvalidSessionUseError`.

**Manual mode** -- push chunks yourself and signal the end explicitly:

```python
await session.send_audio(chunk)   # repeat per chunk
await session.end_audio()         # signal end-of-input
```

## The event protocol

Every streaming session emits a sequence of `TranscriptionEvent` objects. The
`type` field tells you what happened:

| Type | Meaning | `text` | `segment_id` | `speaker` |
| ---- | ------- | ------ | ------------- | --------- |
| `partial` | Interim text that **may change** with the next event on this segment. | Current best guess. | The segment this partial belongs to. | Segment speaker label when diarized, else `None`. |
| `final` | This segment's text is **settled** -- it will not change. | Final text. | The segment that is now final. | Segment speaker label when diarized, else `None`. |
| `supersede` | The engine re-segmented: one or more previously-emitted segments are **replaced**. The replacement events follow immediately. | `None` | `None` (check `old_ids`). | `None` |
| `progress` | A progress heartbeat (e.g. audio position). No transcript content. | `None` | `None`, or the segment it reports on. | `None` |
| `done` | The session is complete. No more events will follow. | `None` | `None` | `None` |
| `error` | An engine error mid-stream. Machine-readable code in `event.code`; human detail in `event.extra["detail"]`; `event.recoverable` says whether the session may continue. | `None` | `None`, or the segment the error concerns. | `None` |

Request diarization when opening the session (`RuntimeParams(diarization=DIARIZE)`,
gated by `streaming.diarization`); `event.speaker` then carries the segment-level
speaker label on `partial` / `final` events. It stays `None` when diarization was
not requested or is unsupported — except on engines whose diarization is
`always_on`, which may emit speaker labels unrequested.

## The core reduce

Handle `partial`, `final`, and `supersede`, and your app is safe on every
compliant engine -- including ones that rewrite interim text or merge segments
after emitting them:

```python
order: list[str] = []       # reading order of live segment ids
texts: dict[str, str] = {}

async for event in session:
    if event.type in ("partial", "final"):
        if event.segment_id not in order:
            order.append(event.segment_id)   # first mention claims a position
        texts[event.segment_id] = event.text
    elif event.type == "supersede":
        pos = order.index(event.old_ids[0])  # the retired block's position
        for old_id in event.old_ids:
            order.remove(old_id)
            texts.pop(old_id, None)
        order[pos:pos] = event.new_ids       # replacements take its place
```

Display text is `texts` joined in `order`. The state is a reading-order list
plus a text map, not a bare map. For engines that emit no timestamps, **list
order is the reading order**. A mid-stream `supersede` must splice its
replacements into the retired block's position. A bare dict can only append,
which would silently reorder the transcript. This exact reduce ships as
`standard_asr.runtime.streaming.reduce_event`, and `StreamReducer` /
`session.result()` build a full `TranscriptionResult` the same way.

Engines that never revise or re-segment never emit `supersede`. Your code
does not need to know which engine is running.

## Stability guarantees

Some engines can tell you how much of the current text is *frozen* and will never
change. This is surfaced via `event.stable_until`:

```python
if event.type == "partial" and event.stable_until is not None:
    frozen = event.text[:event.stable_until]
    tentative = event.text[event.stable_until:]
```

Voice agents can act on `frozen` immediately (e.g. start intent recognition)
without waiting for a `final`.

## Collapsing a session into a result

After the session ends, collapse all events into a standard `TranscriptionResult`:

```python
result = session.result()
print(result.text)
print(result.segments)
```

This gives you the same constant-shape result you get from `engine.transcribe()`,
so your downstream code (subtitle rendering, search, etc.) works identically
whether the input was batch or streamed.

One honesty note: some engines omit timestamps (or one of the two bounds)
while streaming. The reducer stores the engine's measurement verbatim:
`Segment.start`/`end` are `float | None`, and `None` means "not measured"
(check `segment.timestamp_status`: `"measured"`, `"start_only"`, or
`"unavailable"`). Nothing is fabricated. A result with any unmeasured span
also carries a `segment_timestamps_unavailable` warning diagnostic as the
aggregate disclosure. The renderers read the values themselves. A result whose
every segment is *renderable* (a measured span that survives the output's
millisecond grid) renders per-segment faithfully. An unmeasured span makes
`to_srt`/`to_vtt` raise `SubtitleRenderingError` by default. A measured span
that quantizes to zero milliseconds does the same, because players silently
drop a `T --> T` cue. Rendering such a segment would mean silently dropping,
hiding, or fabricating timing, and that trade-off is yours to make. Pass
`on_unrenderable="omit"` to keep only the renderable cues (the other segments'
text stays in `result.text` but not in the file), or `"collapse"` to render one
whole-text cue with no per-segment timeline.

## Synchronous bridge

If you cannot use `async`, wrap the session in `SyncSession`:

```python
from standard_asr import SyncSession

audio_format = engine.recommended_wire_format()
sync = SyncSession(engine.start_transcription(audio_format=audio_format))

with sync:
    sync.feed(pcm_chunks)          # an iterable of bytes chunks (or one bytes chunk)
    for event in sync:
        print(event.type, event.text)
```

`SyncSession` mirrors the async session's input modes: `feed(...)` for managed
input (end-of-input is signaled automatically), or `send_audio(chunk)` +
`end_audio()` for manual input. As with the async session, the two modes must
not be mixed.

`SyncSession` runs the async session on a background thread and exposes a
blocking iterator. See the [API reference](../reference/streaming.md) for the
full interface.

## Deadlines

Application-level deadlines control how long a session waits for the engine:

```python
from standard_asr import StreamDeadlines

async with engine.start_transcription(
    audio_format=audio_format,
    deadlines=StreamDeadlines(max_idle=5.0, max_session_seconds=60.0),
) as session:
    ...
```

The three deadlines are `done_timeout` (pipeline-inactivity hang backstop),
`max_idle` (content-stall detector), and `max_session_seconds` (absolute
wall-clock cap); each accepts `None` to disable it.

When a deadline fires, the session terminates with a terminal **`error`** event
(`code` = `done_timeout` / `stream_stalled` / `session_timeout`) -- not a
`done` event -- so a deadline-killed session is never mistaken for normal
completion. Handle the `error` event's `code` to distinguish the cases.

## Diagnostics mid-stream

Engines can emit structured diagnostics during streaming via
`session.emit_diagnostic()`. These surface parameter-gating decisions (e.g. an
unsupported feature was silently dropped) without interrupting the event flow.
Read them with `session.diagnostics()`.

## Further reading

- [API Reference: streaming](../reference/streaming.md) -- full type signatures.
- [Specification](../spec/specification.md) -- the normative segment lifecycle,
  event ordering, and backpressure rules.
