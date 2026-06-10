# Streaming Specification

Streaming is an optional capability. Engines that support it must implement the
`StreamingASR` protocol and declare the relevant feature flags.

## 1. StreamingASR Protocol

```python
class StreamingASR(Protocol):
    def transcribe_stream(
        self,
        audio_stream: Iterable[NDArray[np.float32]],
        options: BaseTranscribeOptions | None = None,
    ) -> Iterator[StreamChunk]:
        ...

    async def transcribe_stream_async(
        self,
        audio_stream: AsyncIterable[NDArray[np.float32]],
        options: BaseTranscribeOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        ...
```

## 2. StreamChunk Model

```python
class StreamChunk(BaseModel):
    text: str
    start: float | None
    end: float | None
    is_final: bool
    extra: dict[str, Any]
```

## 3. Feature Flags

Engines must declare:
- `FeatureFlag.STREAMING_INPUT`
- `FeatureFlag.STREAMING_OUTPUT` (if partial output is supported)

## 4. Behavior Requirements

- Results should be emitted as soon as they become stable.
- `is_final=True` indicates that the chunk will no longer change.
- Engine-specific streaming metadata goes into `extra`.
