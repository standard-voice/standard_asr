# standard_asr.compliance

The compliance test suite. Engine authors run these checks to verify their plugin
before publishing; applications can also run `check_entrypoints()` at startup to
catch broken installations early.

```python
from standard_asr.compliance import check_entrypoints, check_streaming_param_gating
```

::: standard_asr.compliance
