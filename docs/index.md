# Standard ASR Documentation

Standard ASR is the **USB‑C of ASR inference**: a unified interface that lets
application developers integrate any ASR engine without rewriting integration
code. It also gives ASR developers a clean, minimal standard and tooling so they
can focus on models instead of infrastructure.

## Quick Start (Local Demo)

```bash
uv run uv pip install -e cookbook/std_dummy_asr
uv run standard-asr models list
uv run standard-asr transcribe dummy/echo path/to/audio.wav
```

## Key Docs

- Protocol & data models: `docs/spec/`
- App developer guide: `docs/for_app_dev/discover_and_use.md`
- ASR developer guide: `docs/for_asr_dev/plugin_entrypoints.md`
- Cookbook examples: `cookbook/`

## Principles

- Application developer friendly (zero‑config, plug‑and‑play).
- ASR developer friendly (clear spec + toolchain).
- Optional heavy dependencies, lean core.

See `docs/mission.md` and `docs/goals.md` for full project philosophy.
