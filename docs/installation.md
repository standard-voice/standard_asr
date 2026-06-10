# Installation

!!! warning "Pre-release"
    Standard ASR is not yet published to PyPI. Install directly from GitHub as
    shown below. Once published, `pip install standard-asr` will work.

## Core package

```bash
pip install "standard-asr @ git+https://github.com/standard-voice/standard_asr.git"
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install "standard-asr @ git+https://github.com/standard-voice/standard_asr.git"
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
pip install "standard-asr[audio] @ git+https://github.com/standard-voice/standard_asr.git"
pip install "standard-asr[audio,server] @ git+https://github.com/standard-voice/standard_asr.git"
```

## Install an engine plugin

Standard ASR discovers engines automatically via entry points. Install a plugin
and it appears in `standard-asr models list`:

```bash
pip install "std-faster-whisper @ git+https://github.com/standard-voice/std-faster-whisper.git"
```

## Verify

```bash
standard-asr models list               # discover installed engines
standard-asr compliance entrypoints    # verify plugins resolve correctly
standard-asr doctor                    # diagnose dependency conflicts
```
