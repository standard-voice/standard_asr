# Transcription Result Specification

Standard ASR defines a consistent, structured output for transcription results.
Every engine must return a `TranscriptionResult` object.

## 1. Core Result Model

```python
class TranscriptionResult(BaseModel):
    text: str
    language: str | None
    duration: float | None
    segments: list[Segment] | None
    words: list[Word] | None
    metadata: dict[str, Any]
    extra: dict[str, Any]
```

### Required Fields
- **text**: The full transcript string.

### Optional Fields
- **language**: Detected or forced language (BCP 47).
- **duration**: Audio duration in seconds.
- **segments**: A list of segments with timestamps and metadata.
- **words**: A flattened list of word timestamps.
- **metadata**: Engine‑agnostic metadata (e.g., language probability).
- **extra**: Engine‑specific metadata not yet standardized.

## 2. Segment Model

```python
class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] | None
    speaker: str | None
    temperature: float | None
    avg_logprob: float | None
    compression_ratio: float | None
    no_speech_prob: float | None
    extra: dict[str, Any]
```

Segments allow precise timestamps and advanced metadata without forcing every
engine to support word‑level timestamps.

## 3. Word Model

```python
class Word(BaseModel):
    start: float
    end: float
    text: str
    probability: float | None
    speaker: str | None
    extra: dict[str, Any]
```

## 4. Extra Fields

`extra` should be used for experimental or engine‑specific attributes. If a
feature later becomes standardized, it should be promoted into a first‑class
field and removed from `extra` in a backward‑compatible release.

## 5. Minimal Compliance

Engines that only produce plain text can return a result with:

```json
{
  "text": "hello world",
  "language": null,
  "duration": null,
  "segments": null,
  "words": null,
  "metadata": {},
  "extra": {}
}
```

## 6. Mapping Guidance

- Word timestamps → `words` (and optionally embedded inside `segments`).
- Speaker diarization → `speaker` fields.
- Engine‑specific metadata → `metadata` or `extra`.

This guarantees interoperability between applications and engines.
