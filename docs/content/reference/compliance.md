---
title: standard_asr.compliance
api_module: standard_asr.compliance
---

# standard_asr.compliance

The compliance test suite. Engine authors run these checks to verify their plugin
before publishing; applications can also run `check_entrypoints()` at startup to
catch broken installations early.

For protocol 1.1, the default entry-point check validates the authored declared
metadata, artifact method signatures, and `EngineBase` hook obligations. It does
not call `artifact_status()` or `acquire_artifacts()`. Runtime status and real
acquisition remain deferred opt-in profiles because they can observe or change
external state.

```python
from standard_asr.compliance import check_entrypoints, check_streaming_param_gating
```
