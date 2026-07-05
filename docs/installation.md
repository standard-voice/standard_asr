# Installation

## Core package

```bash
pip install standard-asr
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install standard-asr
# or, in a uv project:
uv add standard-asr
```

The core is intentionally light: only `numpy` and `pydantic`. Everything heavy is
an opt-in extra.

## Optional extras

| Extra      | What it adds | Pulls in |
| ---------- | ------------ | -------- |
| **(core)** | The protocol, engine discovery, audio negotiation, and the CLI. Decodes WAV with the standard library. | `numpy`, `pydantic` |
| **audio**  | Battery-included audio loading: MP3, FLAC, OGG, M4A, raw bytes, base64. Handles decoding, resampling, and channel mixing. | `soundfile`, `scipy` (+ optional system FFmpeg) |
| **server** | FastAPI server exposing any compliant engine over HTTP and WebSocket. | `fastapi`, `python-multipart`, `uvicorn`, `websockets` |

```bash
pip install "standard-asr[audio]"
pip install "standard-asr[audio,server]"
```

## Install an engine plugin

Standard ASR discovers engines automatically via entry points. Install a plugin
and it appears in `standard-asr list`. Each engine is its own package;
experimental plugins install from their repo until they publish to PyPI:

```bash
pip install "std-faster-whisper @ git+https://github.com/standard-voice/std-faster-whisper.git"
```

## Install from source

To try unreleased changes, install the core straight from the repository (this
pulls a snapshot of `main`, not an editable checkout):

```bash
pip install "standard-asr @ git+https://github.com/standard-voice/standard_asr.git"
```

To work on Standard ASR itself, clone the repository and follow
[CONTRIBUTING.md](https://github.com/standard-voice/standard_asr/blob/main/CONTRIBUTING.md).

## Verify

```bash
standard-asr list                      # discover installed engines
standard-asr compliance entrypoints    # verify plugins resolve correctly
standard-asr doctor                    # diagnose dependency conflicts
```
