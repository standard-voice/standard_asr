# standard_asr.engine

The **engine-author facade**: everything you need to build a compliant ASR plugin,
in a single import path.

```python
from standard_asr.engine import (
    EngineBase, BaseConfig, BaseProperties,
    DeclaredCapabilities, BatchCapabilities, FlagCap, LanguageCaps,
    PreparedAudio, RuntimeParams, TranscriptionResult,
)
```

::: standard_asr.engine
