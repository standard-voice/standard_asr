# Transcription Options Specification

Standard ASR uses Pydantic v2 models to describe per‑request inference options.
These options are discoverable and can be used to auto‑generate UI.

## 1. BaseTranscribeOptions

```python
class BaseTranscribeOptions(BaseModel):
    language: str | None
    task: str
    word_timestamps: bool
    speaker_diarization: bool
    extra: dict[str, Any]
```

### Meaning
- **language**: Optional BCP 47 tag to force the language.
- **task**: The task type. Common values: `transcribe`, `translate`.
- **word_timestamps**: Request word-level timestamps if supported.
- **speaker_diarization**: Request speaker labels if supported.
- **extra**: Engine-specific options that are not yet standardized.

## 2. Engine‑Specific Options

Engines should subclass `BaseTranscribeOptions` and add engine-specific fields,
while keeping the base fields intact. Example:

```python
class MyEngineOptions(BaseTranscribeOptions):
    beam_size: int = 5
    temperature: float = 0.0
```

## 3. Compatibility Rules

- Engines **must** accept `BaseTranscribeOptions | dict | None` and coerce them
  into their own options model.
- Options not supported by an engine should raise `TranscriptionError` with a
  clear, actionable message.

## 4. Extra Field

Use `extra` to carry experimental options until they are standardized.
