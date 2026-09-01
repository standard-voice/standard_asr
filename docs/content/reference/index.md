---
title: standard_asr
api_module: standard_asr
---

# standard_asr

The top-level `standard_asr` namespace is the **application-developer surface**.
Import what you need to discover engines, pass audio, read results, and stream:

```python
from standard_asr import discover_models, RuntimeParams, TranscriptionResult
```

For the engine-author surface (building a plugin), see
[`standard_asr.engine`](./engine.md).

For inference-artifact status, acquisition, progress, and errors, see
[Inference artifacts](./artifacts.md).
