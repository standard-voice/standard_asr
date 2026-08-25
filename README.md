<div align="center">

<img src="https://raw.githubusercontent.com/standard-voice/standard_asr/main/docs/assets/branding/icon.png" alt="Standard ASR" width="120" />

# Standard ASR

**The open standard interface between applications and speech-recognition engines.**
_Apps integrate speech-to-text once and gain every engine. Engines implement once and reach every app._

<!-- Package & community -->
[![PyPI](https://img.shields.io/pypi/v/standard-asr?label=PyPI&logo=pypi&logoColor=white&color=blue)](https://pypi.org/project/standard-asr/)
[![Python versions](https://img.shields.io/pypi/pyversions/standard-asr)](https://pypi.org/project/standard-asr/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/standard-voice/standard_asr/blob/main/LICENSE)
[![Chat on Zulip](https://img.shields.io/badge/Join%20Chat-Zulip?style=flat&logo=zulip&label=Zulip&color=blue)](https://standard-voice.zulipchat.com)
[![CI](https://github.com/standard-voice/standard_asr/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/standard-voice/standard_asr/actions/workflows/ci.yml)
[![Canary](https://github.com/standard-voice/standard_asr/actions/workflows/canary.yml/badge.svg)](https://github.com/standard-voice/standard_asr/actions/workflows/canary.yml)
[![codecov](https://codecov.io/gh/standard-voice/standard_asr/graph/badge.svg)](https://codecov.io/gh/standard-voice/standard_asr)
[![OpenSSF Scorecard](https://img.shields.io/ossf-scorecard/github.com/standard-voice/standard_asr?label=OpenSSF%20Scorecard)](https://scorecard.dev/viewer/?uri=github.com/standard-voice/standard_asr)
[![pyright strict](https://img.shields.io/badge/pyright-strict-2ea44f)](https://microsoft.github.io/pyright/)

</div>

> [!WARNING]
> **Alpha — the core protocol works, but major pieces are still missing.** The standard
> interface is functional and exercised by real engine plugins (interface-level
> compliance; end-to-end runtime verification is still being built), but features like
> hardware metadata and model cards are not yet part of the protocol.
> Developer tooling is also incomplete. Expect breaking changes.
> For production use, wait for a stable release. We follow semantic versioning.
> See the [Roadmap](https://github.com/standard-voice/standard_asr/issues/27) for what's planned.

![Standard ASR concept](https://raw.githubusercontent.com/standard-voice/standard_asr/main/docs/assets/concept.jpg)

**A preview of the current state.**
[standard-asr-live](https://github.com/standard-voice/standard-asr-live) is an experimental
terminal app written against Standard ASR alone — it never imports a concrete engine. In the
clip below it starts with no engines installed, so it has nothing to transcribe with. One
`pip install std-mlx-audio` later, the same command offers every model that plugin ships, and
picking one streams live transcription onto the screen. Nothing about the app changed — installing
the engine _was_ the integration.

https://github.com/user-attachments/assets/528f5545-4c79-4a5b-a7fd-562cbf833938

---

## The problem

Speech recognition never got its standard interface. Every ASR library and cloud API ships
its own calling convention, its own audio-input rules, its own streaming protocol.
Integrating one engine means writing an adapter; integrating five means maintaining five.
So most applications hard-wire two or three engines. Their users then get only the languages
and domains those engines handle well, and wait for an "official support" release that usually
never comes. Meanwhile the model that would serve them best already exists.

**Standard ASR** removes that tax: one vendor-neutral interface that both sides implement.
Applications code against the protocol and gain every compliant engine, cloud API, or local
model. Engines implement it once and reach every application. Switching engines becomes a
one-line model-key change — not another adapter.

## "Nice idea — but how does a protocol with no adopters get adopted?"

**Standard ASR does not need any vendor's cooperation to be useful today.** For existing
engines, compliance is a thin adapter — not a rewrite — and the result ships as an ordinary
pip-installable plugin package that anyone can publish. An application developer gets the
payoff — one interface, swappable engines — from day one, before any engine vendor
officially adopts the protocol. If the protocol earns an ecosystem, engine authors gain an organic incentive to ship
native compliance: one interface implemented means every Standard ASR application is a
potential user, plus a CLI, an HTTP/WebSocket server, and a compliance test suite for free.
But nothing waits on that flywheel to start turning.

**"Why a protocol and plugins, and not another all-in-one package?"** Because the
all-in-one shape has been tried, repeatedly, and it structurally fails: a monolith that
bundles adapters for every engine becomes a maintenance bottleneck (new models outpace any
single team), a dependency minefield (engines pin conflicting numpy/torch versions in one
process), and a licensing trap (GPL/AGPL engines can't be bundled with permissive ones).
Model creators won't open pull requests against someone else's mega-repo. Standard ASR
inverts the structure: the core defines the protocol and toolchain; every engine lives in
its own independently maintained, independently licensed package. Maintenance stays with
the people who know each engine best, and the core never becomes the bottleneck.

## Why build on Standard ASR?

- **Write once, run with any engine.** Code against the protocol, not the vendor. Switching
  from a cloud API to a local model (or the reverse) is a one-line model-key change — your
  integration work survives every vendor decision you'll make later.
- **One streaming model for every engine.** Real-time ASR has no shared conventions: some
  engines rewrite their interim results, some never revise a token, some merge already-emitted
  segments after a second decoding pass. Standard ASR unifies all of it under one event
  protocol with explicit stability guarantees — designed against an in-repo survey of 30+
  real engine APIs ([`docs/internal/research/`](docs/internal/research/)).
- **Audio negotiation, batteries included.** Hand over what you have — a file path, raw
  bytes, a NumPy array, a URL — and the framework negotiates and converts to whatever form
  the engine accepts, loudly reporting anything lossy. No more sample-rate guesswork.
- **No dependency hell, no licensing traps.** Each engine is its own pip-installable
  plugin, so restrictive licenses and heavy dependencies stay in the packages that carry
  them. Hard dependency conflicts (for example, numpy 1.x vs 2.x) cannot share one environment —
  `standard-asr doctor` surfaces them instead of letting them hide. Process isolation is
  the escape hatch for plugin-vs-plugin conflicts; a plugin incompatible with the core's
  own numpy floor cannot run anywhere, and doctor reports that as its own conflict.
- **The choice goes to the user.** End users — especially for under-served languages and
  domains — install the engine that serves them best and use it immediately, without
  waiting for the app author to add support.

---

## Quickstart

Install Standard ASR and a compliant engine plugin, then discover and transcribe:

```bash
# Install (see Installation below for extras)
pip install standard-asr
# uv: uv add standard-asr

# Install a compliant engine plugin. Each engine is its own package; experimental
# plugins install from their repo until they publish to PyPI.
pip install "std-faster-whisper @ git+https://github.com/standard-voice/std-faster-whisper.git"

standard-asr list                          # discover installed engines
standard-asr compliance entrypoints        # verify the plugins resolve correctly
```

---

## Python usage

### Transcribe

Discover whatever compliant engines are installed, then transcribe:

```python
from standard_asr import discover_models

registry = discover_models()
engine = registry.create("faster-whisper/large-v3")  # any installed engine's model key

# Pass the audio you already have — a file path, raw bytes, a base64 data URI, or a
# NumPy array. Standard ASR negotiates the right form for the chosen engine and converts
# only when needed (every lossy step is reported as a structured diagnostic).
result = engine.transcribe("meeting.wav")
print(result.text)
```

The **same app code** runs against any other compliant engine — only the model key changes.

Results always have the **same shape** — no format flags that turn the return value into a
string, no fields that appear and disappear. Render subtitles from any engine's result:

```python
from standard_asr import to_srt, to_vtt

print(to_srt(result))  # works for every compliant engine
```

### Discover capabilities & configuration

Engines differ — that's the point. Instead of guessing, ask:

```python
engine.supports("batch.word_timestamps")  # True / False, fail-closed
engine.supports("batch.diarization")  # speaker labels ("who said what")?
engine.supports("streaming.guidance.phrase_hints")
engine.supports("streaming_input")  # can it consume live audio?

registry.config_schema("faster-whisper/large-v3")  # the engine's init-config JSON Schema —
# render a settings UI without
# instantiating (secrets are marked)
```

Unsupported parameters never degrade silently: depending on policy, they either raise
(`strict`) or are dropped with a structured diagnostic telling you exactly what was
ignored and why (`best_effort`).

### Stream

**Full-duplex streaming** — feed audio while receiving live results. Requires a
streaming-capable engine:

```python
# Ask the engine for the PCM wire format it wants (sample rate + encoding), and
# encode your microphone frames to match. A correctly-declared streaming engine
# returns a concrete format; build an `AudioFormat` yourself if you need a specific one.
audio_format = engine.recommended_wire_format()

async with engine.start_transcription(audio_format=audio_format) as session:
    session.feed(microphone())  # any (async) iterable of PCM byte chunks

    order: list[str] = []  # reading order of live segment ids
    texts: dict[str, str] = {}
    async for event in session:
        if event.type in ("partial", "final"):
            if event.segment_id not in order:
                order.append(event.segment_id)  # first mention claims a position
            texts[event.segment_id] = event.text  # partial: may change; final: settled
        elif event.type == "supersede":  # engine re-segmented (two-pass rescoring)
            pos = order.index(event.old_ids[0])
            for old_id in event.old_ids:
                order.remove(old_id)
                texts.pop(old_id, None)
            order[pos:pos] = event.new_ids  # replacements take the block's place
        render(order, texts)

print(session.result().text)  # collapse the session into a TranscriptionResult
```

Those branches (packaged as `standard_asr.runtime.streaming.reduce_event`) are the
**complete core reduce** — handle them and your app is safe on
every compliant engine, including ones that rewrite interim text or merge segments after the
fact. Engines that never do these things simply never emit those events. Voice agents can go
further and act on `event.stable_until`, the engine's guarantee of how much of the text is
frozen against further recognition (a terminal `closed` restatement may still reformat it —
see the streaming guide's "Finality" section).

> Not async? `SyncSession` wraps any streaming session behind a blocking iterator.
> See [`docs/content/specification/`](docs/content/specification/) for the full streaming contract — segment lifecycle,
> stability guarantees, reconnect semantics, and backpressure rules.

---

## Who benefits?

| You are…                                 | You get…                                                                                                                                                                     |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **An application developer**             | One integration that works with every compliant engine; zero vendor lock-in; automatic discovery of whatever the user installs.                                              |
| **An ASR engine developer / researcher** | Focus on the model, not boilerplate. Implement one interface and get a CLI, a reference server, and a compliance test suite **for free**. Reach the whole ecosystem instantly. |
| **An end user**                          | Access to cutting-edge models sooner, and the freedom to pick the engine that fits your language or domain — not whatever the app author happened to choose.                 |

---

## CLI

```bash
standard-asr list                                              # what's installed?
standard-asr show faster-whisper/large-v3                      # properties, capabilities, config schema
standard-asr transcribe faster-whisper/large-v3 audio.wav      # quick transcription
standard-asr serve                                             # expose engines over HTTP/WS
standard-asr doctor                                            # diagnose plugin dependency conflicts
```

---

## Installation

```bash
# pip
pip install standard-asr

# uv
uv pip install standard-asr
# or, in a uv project:
uv add standard-asr
```

With extras:

```bash
# pip
pip install "standard-asr[audio]"
pip install "standard-asr[server]"
pip install "standard-asr[audio,server]"

# uv
uv pip install "standard-asr[audio,server]"
# or, in a uv project:
uv add "standard-asr[audio,server]"
```

### Optional extras

The **core package is intentionally light** — only `numpy` and `pydantic`. Everything heavy
is an **opt-in extra**, so you install exactly the capabilities you need and nothing else.
This is how Standard ASR stays a clean protocol layer instead of a dependency monster.

| Extra      | What it adds                                                                                                                                                                                                       | Pulls in                                                                                          |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **(core)** | The protocol itself: engine discovery, capability discovery and gating, audio negotiation, input and result validation, and the `standard-asr` CLI. Decodes basic `.wav` with the standard library — no extra install.                  | `numpy`, `pydantic`                                                                               |
| **audio**  | **Battery-included audio loading.** Hand over almost any audio — MP3, FLAC, OGG, raw bytes, base64 — and still drive engines that only accept NumPy arrays. Handles decoding, resampling, and channel mixing. | `soundfile`, `scipy` _(plus optional system **FFmpeg** on `PATH` for M4A and the widest format coverage)_ |
| **server** | A **FastAPI server** exposing any compliant engine over HTTP (and WebSocket for streaming), so non-Python apps can use the ecosystem too.                                                                          | `fastapi`, `starlette`, `python-multipart`, `uvicorn`, `websockets`                                            |
| **docs**   | Generates the API reference data for the documentation site (`docs/site`, a Node application). _(For maintainers/contributors.)_                                                                                   | `griffelib`                                                                                 |

> [!NOTE]
> **Why the `audio` extra matters.** Audio wrangling — formats, sample rates, channels — is one
> of the most painful parts of using ASR. Standard ASR absorbs that pain: pass what you have,
> and the framework gets it into the shape the engine needs. The canonical array format is
> `float32`, mono, **16 kHz by default** (a safe, universal target for ASR); when an engine
> wants a different rate or only accepts files, the conversion happens automatically — and
> never silently: every lossy conversion is surfaced as a structured diagnostic. The heavy
> decoders stay optional — basic WAV works with zero extra installs.

### FastAPI server

```bash
# install with the server extra (see Installation above), then:
standard-asr serve --host 0.0.0.0 --port 8000
```

See [`docs/content/specification/server-api.md`](docs/content/specification/server-api.md) for the full HTTP/WebSocket API contract,
and [`docs/content/specification/`](docs/content/specification/) for the protocol specification. The WebSocket endpoint covers
the incremental-streaming path (declare an `audio_format`, push raw PCM frames, receive live
events); whole-input engines use the batch HTTP endpoints.

---

## Building an engine plugin

An engine plugin is an ordinary pip-installable package that subclasses `EngineBase`,
declares its **properties** (what audio it accepts), **capabilities** (what features it
supports), and **config** (its typed, UI-discoverable settings model), and registers a
`standard_asr.models` entry point. The standard layer handles audio negotiation, parameter
gating, language resolution, and the sync/async bridge — you implement the model call, and
the CLI, the HTTP/WebSocket server, and the compliance checks come for free.

See [`docs/content/engine-authors/`](docs/content/engine-authors/) for the plugin authoring guide, then check
your implementation with the full suite:

```bash
standard-asr compliance run
```

---

## FAQ

> **Why support different engines? Why not just use Whisper?**

- Different languages have different state-of-the-art models; Whisper is strong in some, weak
  in others.
- GPU/hardware acceleration support varies across platforms.
- The field moves fast — today's state-of-the-art model gets replaced. Write once against Standard ASR, and
  countless engines (present and future) are supported automatically.

---

## Project status & design

**Alpha.** The core protocol — engine interface, audio negotiation, capability discovery,
streaming events, plugin system — is shipped and exercised by four engine plugins; the
compliance suite checks the interface contract (runtime inference verification is
tracked separately). The toolchain (CLI, FastAPI server, compliance suite) works. What's
missing: features like hardware metadata and model cards are not yet part of the protocol.
A richer CLI and a plugin starter template are also not done yet.
See the open issues for what's planned.

Built with a normative, RFC-style specification (`docs/content/specification/`),
Pydantic v2 models, `pyright --strict`, 100% test coverage, and CI across numpy 1.x/2.x
and Python 3.10–3.14. Design decisions are grounded in an in-repo survey of 30+ real ASR
engines and APIs (`docs/internal/research/`). The authoritative material lives in-repo:

- `docs/content/specification/` — the protocol specification.
- `docs/internal/research/` — the engine surveys the design is tested against.
- `CONTRIBUTING.md` — dev setup, the dependency policy, and the CI channel model.

## Communication

We use **Zulip** for development discussion: https://standard-voice.zulipchat.com

## Contributing

Please read [`CONTRIBUTING.md`](./CONTRIBUTING.md) before opening a pull request.

## License

Apache 2.0. See [LICENSE](./LICENSE).
