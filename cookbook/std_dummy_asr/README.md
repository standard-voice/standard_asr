# std-dummy-asr

This package is a **zero-dependency** (aside from `standard-asr`) demo plugin that
exposes Standard ASR entry points. It is meant for local testing and examples and
ships two presets:

- `dummy/echo` – echoes the length of the provided audio array.
- `dummy/` – a default alias for `dummy/echo` to demonstrate default model keys.

Install it in editable mode from the workspace root to experiment with discovery
and compliance tools:

```bash
uv run uv pip install -e cookbook/std_dummy_asr
uv run standard-asr list
uv run standard-asr compliance entrypoints
```

The code lives in `src/std_dummy_asr` and is intentionally small so you can read
through it quickly when learning how to integrate your own engine.
