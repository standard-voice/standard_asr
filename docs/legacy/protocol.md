# Standard ASR Protocol Specification

This document defines the core protocol for Standard ASR engines. It is the
source of truth for how application code interacts with any compliant engine.

## 1. Scope

The protocol defines:
- The interface that every engine must implement.
- The Standard ASR audio input contract.
- The standard output result model.
- The optional feature flag system for advanced capabilities.

## 2. Core Interface

Every engine **must** implement the `StandardASR` protocol:

```python
class StandardASR(Protocol):
    config: BaseConfig
    properties: ClassVar[BaseProperties]

    def transcribe(
        self,
        audio: NDArray[np.float32],
        options: BaseTranscribeOptions | None = None,
    ) -> TranscriptionResult:
        ...

    async def transcribe_async(
        self,
        audio: NDArray[np.float32],
        options: BaseTranscribeOptions | None = None,
    ) -> TranscriptionResult:
        ...
```

### Required Attributes

- `config`: An instance of `BaseConfig` (or subclass) describing initialization
  parameters for the engine.
- `properties`: A **class attribute** instance of `BaseProperties` (or subclass)
  describing static engine metadata.

## 3. Audio Input Contract

All engines **must** accept audio as a NumPy float32 waveform with the following
contract:

- dtype: `np.float32`
- Sample rate: `16,000 Hz` (unless engine declares additional supported sample
  rates)
- Value range: `[-1.0, 1.0]` (clipped)
- Shape:
  - Mono: `(n_samples,)`
  - Multi‑channel: `(n_samples, n_channels)`

Engines should validate channel count against `properties.supported_channels`.
Use `standard_asr.runtime.validate_audio_input()` to enforce the contract and
**always assign its return value**, because it normalizes dtype and shape.

## 4. Transcription Output

The output is a `TranscriptionResult` model (see `docs/spec/results.md`):

- `text`: full transcript
- `segments`: optional segment list
- `words`: optional word list
- `metadata`: standardized metadata
- `extra`: engine‑specific fields

This ensures that app developers receive a predictable structure regardless of
engine choice.

## 5. Options & Config

- **Config**: Engine initialization parameters must be modeled with Pydantic v2
  and exposed via a `BaseConfig` subclass.
- **Options**: Per‑request options must be modeled with Pydantic v2 and exposed
  via a `BaseTranscribeOptions` subclass.

Both config and options must be UI‑discoverable and include descriptions.

## 6. Optional Features

Advanced capabilities are standardized through feature flags in
`standard_asr.features.FeatureFlag` (see `docs/spec/features.md`). Engines must
declare supported features in `BaseProperties.features`.

Examples:
- Word timestamps
- Streaming input/output
- Speaker diarization
- Translation
- VAD / language detection

## 7. Error Handling

Engines should raise `StandardASRError` subclasses for predictable failures:
- `TranscriptionError` for runtime transcription failures.
- `ConfigError` for invalid configuration.
- `DiscoveryError` for plugin discovery and loading issues.

## 8. Compliance

The Standard ASR compliance tools validate that:
- Entry points are discoverable.
- Factories are callable.
- Instances expose `transcribe`, `config`, and `properties`.

See `docs/spec/cli.md` and `docs/for_asr_dev/plugin_entrypoints.md`.
