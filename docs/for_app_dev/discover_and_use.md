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
print(asr.transcribe(audio))
```

The same snippet is available at `cookbook/sample_client.py`. Replace the model
key with any other discovered entry point to switch engines without changing
your application logic.
