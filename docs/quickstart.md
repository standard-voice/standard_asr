# Quickstart

Install Standard ASR and a compliant engine plugin, then discover and transcribe
in under a minute.

## Install

```bash
pip install "standard-asr[audio] @ git+https://github.com/standard-voice/standard_asr.git"
pip install "std-faster-whisper @ git+https://github.com/standard-voice/std-faster-whisper.git"
```

The `[audio]` extra adds MP3/FLAC/OGG decoding and automatic resampling. Without
it, only WAV files work out of the box.

## Discover installed engines

```bash
standard-asr list
```

Every compliant engine plugin registers itself via entry points. No configuration
needed -- install a plugin and it appears.

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
engine.supports("batch.word_timestamps")          # True / False, fail-closed
engine.supports("streaming_input")                # can it consume live audio?
```

## Stream (real-time)

```python
audio_format = engine.recommended_wire_format()

async with engine.start_transcription(audio_format=audio_format) as session:
    session.feed(microphone)
    async for event in session:
        if event.type == "partial":
            show(event.segment_id, event.text)     # may change
        elif event.type == "final":
            commit(event.segment_id, event.text)   # settled
        elif event.type == "supersede":
            for old in event.old_ids:
                remove(old)                        # engine re-segmented
```

Those three branches (`partial` / `final` / `supersede`) are the complete core
reduce. Handle them and your app works on every compliant engine.

## Next steps

- [Discover & Use](for_app_dev/discover_and_use.md) -- the full app-developer
  guide (parameters, audio input types, rendering).
- [Streaming](for_app_dev/streaming.md) -- deep dive into the streaming event
  protocol, stability guarantees, and the sync bridge.
- [Adapt an Engine](for_asr_dev/adapting_engine.md) -- build a compliant plugin.
- [API Reference](reference/index.md) -- the complete public surface.
