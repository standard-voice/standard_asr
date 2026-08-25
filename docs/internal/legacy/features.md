# Feature Flags Specification

Optional features are declared through `FeatureFlag` and advertised in
`BaseProperties.features`. This avoids ambiguity and lets applications detect
capabilities at runtime.

## 1. FeatureFlag Values

- `streaming_input`: Engine accepts streaming audio chunks.
- `streaming_output`: Engine emits incremental results.
- `word_timestamps`: Engine returns word-level timestamps.
- `speaker_diarization`: Engine returns speaker labels.
- `translation`: Engine supports translate task.
- `language_detection`: Engine can detect language automatically.
- `vad`: Engine supports voice activity detection controls.

## 2. How to Use

```python
if FeatureFlag.WORD_TIMESTAMPS in asr.properties.features:
    options.word_timestamps = True
```

## 3. Standardization Policy

New features must first appear in `extra`, then be proposed for standardization.
Once standardized, they are added to `FeatureFlag` and to the core result model.
