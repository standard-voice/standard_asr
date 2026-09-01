---
title: Quickstart
---

# Quickstart

Install Standard ASR and a compliant engine plugin, then discover and transcribe
in under a minute.

## Install

```bash
pip install "standard-asr[audio]"
# Each engine is its own package; experimental plugins install from their repo
# until they publish to PyPI.
pip install "std-faster-whisper @ git+https://github.com/standard-voice/std-faster-whisper.git"
```

The `[audio]` extra adds MP3/FLAC/OGG decoding and higher-quality resampling.
Resampling itself works without it, on a built-in numpy fallback. Without the
extra, only an 8/16-bit PCM WAV *file path* decodes with no extra setup — WAV
bytes go through the same path as MP3, FLAC, and OGG, which need either the
extra or the ffmpeg suite (`ffmpeg` and `ffprobe`) on your PATH.

## Discover installed engines

```bash
standard-asr list
```

Every compliant engine plugin registers itself via entry points. No configuration
needed -- install a plugin and it appears.

## Check what the model needs on disk

Most engines keep persistent inference artifacts: weights, tokenizers, compiled
graphs, installed operating-system assets. Ask before the first run instead of
finding out during one:

```bash
standard-asr status faster-whisper/large-v3   # what is on disk, what is missing
standard-asr pull faster-whisper/large-v3     # acquire what inference needs
```

`pull` never transcribes audio to warm a cache, and it prints any step only you
can take, such as accepting a license or signing in. An engine that carries no
artifacts, such as a cloud API, reports `not_applicable`. See
[Inference artifacts](./reference/artifacts.md).

## Transcribe

```python
from standard_asr import discover_models

registry = discover_models()
engine = registry.create("faster-whisper/large-v3")
result = engine.transcribe("meeting.wav")
print(result.text)
```

The **same code** works with any other compliant engine -- only the model key
changes. Results always have the same shape (`TranscriptionResult`), so your
downstream code (subtitle rendering, search indexing, etc.) never needs to adapt.

## Check capabilities

Engines differ. Instead of guessing, ask:

```python
engine.supports("batch.word_timestamps")  # True / False, fail-closed
engine.supports("streaming_input")  # can it consume live audio?
```

## Stream (real-time)

```python
audio_format = engine.recommended_wire_format()

async with engine.start_transcription(audio_format=audio_format) as session:
    session.feed(microphone)
    async for event in session:
        if event.type == "partial":
            show(event.segment_id, event.text)  # may change
        elif event.type == "final":
            commit(event.segment_id, event.text)  # settled
        elif event.type == "supersede":
            for old in event.old_ids:
                remove(old)  # engine re-segmented
```

Those three event types (`partial` / `final` / `supersede`) are the core set
every app handles. This sketch keys display by `segment_id` only; an app that
renders joined text must also keep the reading order across a `supersede` --
use `standard_asr.runtime.streaming.reduce_event`, or see the full reduce in
the [Streaming guide](./app-developers/streaming.md).

## Next steps

- [Discover & Use](./app-developers/discover-and-use.md) -- the full app-developer
  guide (parameters, audio input types, rendering).
- [Streaming](./app-developers/streaming.md) -- deep dive into the streaming event
  protocol, stability guarantees, and the sync bridge.
- [Adapt an ASR System](./engine-authors/adapt-an-asr-system.md) -- build a compliant plugin.
- [API Reference](./reference/index.md) -- the complete public surface.
