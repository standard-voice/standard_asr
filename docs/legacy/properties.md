# Properties Specification

Standard ASR engines must expose static metadata via `BaseProperties`. This
metadata enables discovery, UI generation, and capability negotiation.

## 1. BaseProperties Fields

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

## 2. Validation Rules

- `supported_languages` must use **BCP 47** tags.
- `supported_sample_rates` must be positive integers.
- `supported_channels` must be positive integers.
- `engine_id` must follow the entrypoint naming rules and should be PEP 503 normalized.
- `model_name` must follow the entrypoint naming rules (empty is allowed only for defaults).
- `supported_devices` must be non-empty and contain non-empty identifiers.
- `audio_dtype` must be `float32` to satisfy the Standard ASR audio contract.

## 3. Model Identifier

The fully qualified model identifier is:

```
{engine_id}/{model_name}
```

This must match the entry point name. Compliance checks enforce this identity
invariant to keep discovery, routing, and telemetry consistent.

## 4. Feature Flags

Declare optional capabilities through `features`. This is the single source of
truth for what the engine supports.

## 5. Extra Metadata

`extra` is reserved for engine‑specific metadata, experimental features, and
information that has not yet been standardized.

## 6. Example

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
