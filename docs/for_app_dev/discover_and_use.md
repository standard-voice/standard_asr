# Discovering Standard ASR Plugins (for app developers)

This guide shows how an application can discover and use Standard ASR–compliant
models via Python entry points.

## One‑time setup in this repo (uv workspace)

```bash
uv run uv pip install -e cookbook/std_dummy_asr
```

The dummy plugin is dependency‑free and exposes two model keys: `dummy/echo` and
`dummy/` (default alias).

## Inspect available models

```bash
uv run standard-asr models list
uv run standard-asr models show dummy/echo
```

## Run compliance checks

```bash
uv run standard-asr compliance entrypoints
```

## Minimal client code

```python
from standard_asr import discover_models
import numpy as np

registry = discover_models()
asr = registry.create("dummy/echo")
audio = np.zeros(16_000, dtype=np.float32)
result = asr.transcribe(audio)
print(result.text)
```

The same snippet is available at `cookbook/sample_client.py`. Replace the model
key with any other discovered entry point to switch engines without changing
your application logic.

## Transcription Result

`StandardASR.transcribe()` returns a structured `TranscriptionResult`:

```python
result = asr.transcribe(audio)
print(result.text)
```

Use `result.segments` or `result.words` when the engine supports timestamps.

## Passing Options

```python
from standard_asr import BaseTranscribeOptions

options = BaseTranscribeOptions(language="en", word_timestamps=True)
result = asr.transcribe(audio, options=options)
```

## CLI Quick Usage

```bash
standard-asr models list
standard-asr transcribe dummy/echo path/to/audio.wav
```
