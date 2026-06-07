<div align="center">

<img src="docs/assets/branding/icon.png" alt="Standard ASR" width="120" />

# Standard ASR

**A universal, plug-and-play protocol for speech recognition.**
*Write your app once — run it with any ASR engine, today's and tomorrow's.*

[![CI](https://github.com/standard-voice/standard_asr/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/standard-voice/standard_asr/actions/workflows/ci.yml)
[![Canary](https://github.com/standard-voice/standard_asr/actions/workflows/canary.yml/badge.svg)](https://github.com/standard-voice/standard_asr/actions/workflows/canary.yml)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Chat on Zulip](https://img.shields.io/badge/Join%20Chat-Zulip?style=flat&logo=zulip&label=Zulip&color=blue)](https://standard-voice.zulipchat.com)

</div>

> [!WARNING]
> **Standard ASR is pre-release and a work in progress** — breaking changes may land at any
> time. For production use, wait for the `v1.0.0` release, where we stabilize the public API
> and enforce a migration policy for breaking changes. We strictly follow semantic versioning.
> Try it out and tell us what you think — let's shape the future of ASR tooling together.

![Standard ASR concept](docs/assets/concept.jpg)

---

## Introduction

**Standard ASR** is a universal protocol that standardizes how applications talk to
Automatic Speech Recognition (ASR) engines.

> **Think of it as USB-C for speech recognition** — one common interface that lets any
> application work with any ASR engine, seamlessly.

Today, every ASR library has its own API, its own dependencies, its own quirks. Integrating
a new engine means writing a whole new adapter. Standard ASR removes that tax: your
application speaks one protocol, and **any compliant engine — cloud API or local model —
drops in without a single line of app code changing.**

### Why Standard ASR?

- **Write once, run with any engine.** Code against the protocol, not the vendor. Switch
  from a cloud API to a local model (or the reverse) by changing a model key.
- **Engines as true plugins.** Each engine declares its capabilities; your app adapts
  dynamically. Support the latest models on day one.
- **No vendor lock-in, no dependency hell.** Each engine is an isolated, pip-installable
  plugin, so conflicting dependencies and restrictive licenses stay contained.
- **The choice goes to the user.** End users — especially for under-served languages and
  domains — can install the engine that serves them best, without waiting for the app
  author to add support.

### Who benefits?

| You are… | You get… |
|---|---|
| **An application developer** | One integration that works with every compliant engine; zero vendor lock-in; automatic discovery of whatever the user installs. |
| **An ASR engine developer / researcher** | Focus on the model, not boilerplate. Implement one interface and get a CLI, a Web API server, and a compliance test suite **for free**. Reach the whole ecosystem instantly. |
| **An end user** | Access to cutting-edge models sooner, and the freedom to pick the engine that fits your language or domain — not whatever the app author happened to choose. |

---

## Quickstart

Standard ASR discovers compliant plugins through the `standard_asr.models` entry-point group;
each plugin exposes model presets keyed as `<engine_id>/<model_name>`. A tiny demo plugin
ships in `cookbook/std_dummy_asr` so you can try the whole workflow with **no extra
dependencies**:

```bash
pip install standard-asr
pip install -e cookbook/std_dummy_asr      # the demo plugin (echoes a synthetic transcript)

standard-asr models list                   # discover installed engines
standard-asr compliance entrypoints        # check the plugins resolve correctly
python cookbook/sample_client.py           # discover -> instantiate -> transcribe
```

Use this flow as a template when building or trying your own plugin.

---

## Python usage

Discover whatever compliant engines are installed, then transcribe:

```python
from standard_asr import discover_models

registry = discover_models()
engine = registry.create("dummy/echo")     # swap this key for any installed engine

# Pass the audio you already have — a file path, raw bytes, a base64 data URI, or a
# NumPy array. Standard ASR negotiates the right form for the chosen engine and converts
# only when needed.
result = engine.transcribe("meeting.wav")
print(result.text)
```

The **same app code** runs against any other compliant engine — only the model key changes
(e.g. swapping `dummy/echo` for a real local or cloud engine you've installed as a plugin).

**Streaming (full-duplex)** — feed audio while receiving live results that the engine may
revise before committing them as final. Requires a streaming-capable engine:

```python
session = engine.start_transcription(audio_format=audio_format)   # audio_format per the spec

async with session:
    async def pump_microphone():
        async for chunk in microphone():        # your byte source
            await session.send_audio(chunk)
        await session.end_audio()
    start_task(pump_microphone())

    async for event in session:
        if event.type == "partial":             # may still change
            show_live(event.text)
        elif event.type == "final":             # locked in
            commit(event.text)
```

> For environments that aren't async, `SyncSession` wraps a streaming session behind a
> blocking iterator. See `docs/spec/` for the full streaming contract.

---

## CLI

```bash
standard-asr models list
standard-asr transcribe dummy/echo path/to/audio.wav
standard-asr doctor                         # diagnose plugin/dependency conflicts
```

---

## Installation & optional extras

The **core package is intentionally light** — only `numpy` and `pydantic`. Everything heavy
is an **opt-in extra**, so you install exactly the capabilities you need and nothing else.
This is how Standard ASR stays a clean protocol layer instead of a dependency monster.

| Extra | Install | What it adds | Pulls in |
|---|---|---|---|
| **(core)** | `pip install standard-asr` | The protocol itself: engine discovery, capability/properties negotiation, input/output validation, and the `standard-asr` CLI. Decodes basic `.wav` with the standard library — no extra install. | `numpy`, `pydantic` |
| **audio** | `pip install "standard-asr[audio]"` | **Battery-included audio loading.** Hand over almost any audio — MP3, FLAC, OGG, M4A, raw bytes, base64 — and still drive engines that only accept NumPy arrays. Handles decoding, resampling, and channel mixing. | `soundfile`, `scipy` *(plus optional system **FFmpeg** on `PATH` for the widest format coverage)* |
| **server** | `pip install "standard-asr[server]"` | A **FastAPI server** exposing any compliant engine over HTTP (and WebSocket for streaming), so non-Python apps can use the ecosystem too. | `fastapi`, `python-multipart`, `uvicorn` |
| **docs** | `pip install "standard-asr[docs]"` | Builds the documentation site. *(For maintainers/contributors.)* | `mkdocs-material` |

> [!NOTE]
> **Why the `audio` extra matters.** Audio wrangling — formats, sample rates, channels — is one
> of the most painful parts of using ASR. Standard ASR absorbs that pain: pass what you have,
> and the framework gets it into the shape the engine needs. The canonical array format is
> `float32`, mono, **16 kHz by default** (a safe, universal target for ASR); when an engine
> wants a different rate or only accepts files, the conversion happens automatically. The heavy
> decoders stay optional — basic WAV works with zero extra installs.

### FastAPI server

```bash
pip install "standard-asr[server]"
standard-asr serve --host 0.0.0.0 --port 8000
```

See [`docs/spec/`](docs/spec/) for the protocol specification and the HTTP/WebSocket API contract.

---

## What Standard ASR is **not**

Standard ASR is **not** a library that bundles every ASR model. It defines the **protocol and
interface** and ships the surrounding tooling. Adapting a specific engine is done by that
engine's developers (or a fork) as a separate plugin package.

Many projects try to be a toolbox that crams every model in. That doesn't scale: models carry
conflicting dependencies and restrictive licenses (GPL/AGPL), and a central hub becomes a
maintenance bottleneck. Standard ASR is instead a **common language** between models and
applications — and it hands maintenance back to the engine authors. Giving the choice of model
to the end user means **no app code changes and no maintainer attention** are needed to adopt
a new engine.

---

## FAQ

> **Why support different engines? Why not just use Whisper?**

- Different languages have different state-of-the-art models; Whisper is strong in some, weak
  in others.
- GPU/hardware acceleration support varies across platforms.
- The field moves fast — today's SOTA will be replaced. Write once against Standard ASR, and
  countless engines (present and future) are supported automatically.

---

## Project status & design

Pre-release, under active redesign with standard-library rigor (Pydantic v2 models,
`pyright --strict`, 100% test coverage, multi-channel dependency CI). The authoritative
material lives in-repo:

- `docs/spec/` — the protocol specification.
- `CONTRIBUTING.md` — dev setup, the dependency policy, and the CI channel model.
- `cookbook/` — runnable example plugins (`std_dummy_asr`, `std_faster_whisper`).

## Communication

We use **Zulip** for development discussion: https://standard-voice.zulipchat.com

## Contributing

Please read [`CONTRIBUTING.md`](./CONTRIBUTING.md) before opening a pull request.

## License

Apache 2.0. See [LICENSE](./LICENSE).
</content>
</invoke>
