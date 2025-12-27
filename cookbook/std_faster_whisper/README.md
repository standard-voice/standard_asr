std_faster_whisper is a sample implementation of a Standard ASR compliant
faster‑whisper engine. It demonstrates how to build a plugin that respects the
Standard ASR protocol, options model, and lazy‑loading rules.

## Local Usage

```bash
uv run uv pip install -e cookbook/std_faster_whisper
standard-asr models list
standard-asr transcribe faster-whisper/whisper path/to/audio.wav
```

## Notes

- The engine loads the model lazily on first transcription.\n- Downloads respect the `STANDARD_ASR_ALLOW_DOWNLOAD` policy.
