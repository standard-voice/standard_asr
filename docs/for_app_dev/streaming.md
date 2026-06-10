# Streaming

Standard ASR unifies the wildly divergent streaming behaviors of 30+ ASR engines
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
encoding as an `AudioFormat`. If you need a specific format (e.g. 8 kHz for
telephony), construct one yourself -- the engine will raise
`UnsupportedFeatureError` if it cannot accept it.

For whole-input streaming (the engine streams *output* over a complete audio
file), pass `audio=` instead of `audio_format=`:

```python
async with engine.start_transcription(audio="meeting.wav") as session:
    async for event in session:
        ...
```

## Feeding audio

For live-input streaming, feed PCM byte chunks as an iterable:

```python
session.feed(microphone)    # any (async) iterable of bytes
```

When the audio source is exhausted, call `session.end_audio()` to signal
end-of-input. If you feed an iterable, the session calls `end_audio()`
automatically when the iterable finishes.

## The event protocol

Every streaming session emits a sequence of `TranscriptionEvent` objects. The
`type` field tells you what happened:

| Type | Meaning | `text` | `segment_id` |
| ---- | ------- | ------ | ------------- |
| `partial` | Interim text that **may change** with the next event on this segment. | Current best guess. | The segment this partial belongs to. |
| `final` | This segment's text is **settled** -- it will not change. | Final text. | The segment that is now final. |
| `supersede` | The engine re-segmented: one or more previously-emitted segments are **replaced**. The replacement events follow immediately. | `None` | `None` (check `old_ids`). |
| `progress` | A progress heartbeat (e.g. audio position). No transcript content. | `None` | `None` |
| `done` | The session is complete. No more events will follow. | `None` | `None` |
| `error` | An engine error mid-stream. | Error description. | `None` |

## The core reduce

Handle `partial`, `final`, and `supersede`, and your app is safe on every
compliant engine -- including ones that rewrite interim text or merge segments
after the fact:

```python
segments: dict[str, str] = {}

async for event in session:
    if event.type == "partial":
        segments[event.segment_id] = event.text
    elif event.type == "final":
        segments[event.segment_id] = event.text
    elif event.type == "supersede":
        for old_id in event.old_ids:
            del segments[old_id]
```

Engines that never revise or re-segment simply never emit `supersede`. Your code
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

## Synchronous bridge

If you cannot use `async`, wrap the session in `SyncSession`:

```python
from standard_asr import SyncSession

audio_format = engine.recommended_wire_format()
sync = SyncSession(engine.start_transcription(audio_format=audio_format))

with sync:
    sync.feed_bytes(pcm_chunk)
    sync.end_audio()
    for event in sync:
        print(event.type, event.text)
```

`SyncSession` runs the async session on a background thread and exposes a
blocking iterator. See the [API reference](../reference/streaming.md) for the
full interface.

## Deadlines

Application-level deadlines control how long a session waits for the engine:

```python
from standard_asr import StreamDeadlines

async with engine.start_transcription(
    audio_format=audio_format,
    deadlines=StreamDeadlines(max_idle_seconds=5.0, max_session_seconds=60.0),
) as session:
    ...
```

When a deadline fires, the session emits a `done` event and closes cleanly.

## Diagnostics mid-stream

Engines can emit structured diagnostics during streaming via
`session.emit_diagnostic()`. These surface parameter-gating decisions (e.g. an
unsupported feature was silently dropped) without interrupting the event flow.
Read them with `session.diagnostics()`.

## Further reading

- [API Reference: streaming](../reference/streaming.md) -- full type signatures.
- [Specification](../spec/specification.md) -- the normative segment lifecycle,
  event ordering, and backpressure rules.
