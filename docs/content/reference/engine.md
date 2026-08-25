---
title: standard_asr.engine
api_module: standard_asr.engine
---

# standard_asr.engine

The **engine-author facade**: everything you need to build a compliant ASR plugin,
in a single import path.

```python
from standard_asr.engine import (
    EngineBase,
    BaseConfig,
    BaseProperties,
    DeclaredCapabilities,
    BatchCapabilities,
    FlagCap,
    LanguageCaps,
    PreparedAudio,
    RuntimeParams,
    TranscriptionResult,
)
```
