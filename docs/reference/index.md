# standard_asr

The top-level `standard_asr` namespace is the **application-developer surface**.
Import what you need to discover engines, pass audio, read results, and stream:

```python
from standard_asr import discover_models, RuntimeParams, TranscriptionResult
```

For the engine-author surface (building a plugin), see
[`standard_asr.engine`](engine.md).

::: standard_asr
    options:
      show_submodules: false
