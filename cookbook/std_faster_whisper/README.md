std_faster_whisper is a sample implementation of a Standard ASR compliant
faster‑whisper engine. It demonstrates how to build a plugin that respects the
Standard ASR protocol, options model, and lazy‑loading rules.

## Local Usage

```bash
uv run uv pip install -e cookbook/std_faster_whisper
standard-asr list
standard-asr transcribe faster-whisper/large-v3 path/to/audio.wav
```

## Presets

Each model is a separate **entry-point preset** (spec IC.7: model selection =
entry-point preset, never an init `model` field), so `standard-asr list`,
the registry, and any settings UI can enumerate the available models:

| Entry point key                 | Model           |
|---------------------------------|-----------------|
| `faster-whisper/large-v3`       | `large-v3`      |
| `faster-whisper/distil-large-v3`| `distil-large-v3` |
| `faster-whisper/turbo`          | `large-v3-turbo`  |

faster-whisper ships ~15 sizes; this demo registers a representative few. To add
another (e.g. `small`), define a properties subclass overriding `model_name`, an
engine subclass overriding `model_size`, and a factory, then register the key in
`pyproject.toml` under `[project.entry-points."standard_asr.models"]`:

```toml
"faster-whisper/small" = "std_faster_whisper.entrypoint:create_small"
```

`model_path` is **not** a model selector — it is an optional local checkpoint
path/override (spec IC.7 weights/path). The preset chooses the model.

## Notes

- The engine loads the model lazily on first transcription.
- Downloads respect the `STANDARD_ASR_ALLOW_DOWNLOAD` policy.
