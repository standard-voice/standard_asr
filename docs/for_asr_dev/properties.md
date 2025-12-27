# Properties for ASR Developers

All Standard ASR engines **must** expose static metadata via `BaseProperties`.
This is the contract that lets applications discover capabilities without
instantiating the engine.

## Required Fields

```python
class BaseProperties(BaseModel):
    engine_id: str
    model_name: str
    protocol_version: str
    supported_languages: list[str]
    supported_devices: list[str]
    supported_sample_rates: list[int]
    supported_channels: list[int]
    audio_dtype: str
    features: set[FeatureFlag]
    description: str | None
    extra: dict[str, Any]
```

### Notes
- `engine_id` should be PEP 503 normalized.
- `model_name` should match the entrypoint preset name.
- `supported_languages` **must** use BCP 47 tags.

## BCP 47 Language Tags

We use BCP 47 because it can represent scripts and regions beyond ISO 639‑1.
If your model requires a different language code, convert from BCP 47 in your
engine implementation.

Examples:
- `en`
- `zh-Hant`
- `pt-BR`

## Example Implementation

```python
class MyEngineProperties(BaseProperties):
    engine_id: str = "my-engine"
    model_name: str = "default"
    protocol_version: str = "0.2.0"
    supported_languages: list[str] = ["en", "zh-Hant"]
    supported_devices: list[str] = ["cpu", "cuda"]
    supported_sample_rates: list[int] = [16000]
    supported_channels: list[int] = [1]
    audio_dtype: str = "float32"
    features: set[FeatureFlag] = {FeatureFlag.WORD_TIMESTAMPS}
```
