# Introduction

Standard ASR provides a **single, stable protocol** for ASR inference. This lets
applications switch engines by changing only a model key, not application code.

For ASR developers, Standard ASR is a **lightweight framework + toolchain** that
makes publishing compliant engines simple and consistent.

If you are:
- **App developer** -> start with `docs/for_app_dev/discover_and_use.md`.
- **ASR developer** -> start with `docs/for_asr_dev/adapting_engine.md`,
  then review `docs/for_asr_dev/plugin_entrypoints.md`.

The detailed protocol specifications live under `docs/spec/`.
