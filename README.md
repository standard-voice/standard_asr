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

## The problem

Every ASR library and cloud API has its own calling convention, its own audio-input rules,
its own streaming protocol, its own dependencies. Integrating one engine means writing an
adapter; integrating five means maintaining five. So in practice most applications hard-wire
two or three engines — and their users are stuck with whatever languages and domains those
engines happen to be good at, waiting for an "official support" release that usually never
comes. Meanwhile the model that would actually serve them best already exists.

**Standard ASR** removes that tax. It is a universal protocol between applications and
speech-recognition engines:

> **Think of it as the OpenAI Chat Completions API for speech recognition** — one common
> interface, so any application can use any compliant engine, cloud API or local model,
> without a single line of app code changing.

## "Nice idea — but how does a protocol with no adopters get adopted?"

That's the right question to ask, so let's answer it up front.

**Standard ASR does not need any vendor's cooperation to be useful today.** For existing
engines, compliance is a thin adapter — not a rewrite — and adapters are ordinary
pip-installable plugin packages that anyone can publish. We maintain first-party adapter
plugins in this repo (see [`cookbook/`](cookbook/), including a real
[faster-whisper](cookbook/std_faster_whisper) adapter and a dependency-free demo engine you
can run in under a minute). An application developer gets the payoff — one interface,
swappable engines — from day one, with zero engines "officially" on board. If the protocol
earns an ecosystem, engine authors gain an organic incentive to ship native compliance:
one interface implemented means every Standard ASR application is a potential user, plus a
CLI, an HTTP/WebSocket server, and a compliance test suite for free. But nothing waits on
that flywheel to start turning.

**"Why a protocol and plugins, and not another all-in-one package?"** Because the
all-in-one shape has been tried, repeatedly, and it structurally fails: a monolith that
bundles adapters for every engine becomes a maintenance bottleneck (new models outpace any
single team), a dependency minefield (engines pin conflicting numpy/torch versions in one
process), and a licensing trap (GPL/AGPL engines can't be bundled with permissive ones).
Model creators won't open pull requests against someone else's mega-repo. Standard ASR
inverts the structure: the core defines the protocol and toolchain; every engine lives in
its own independently-maintained, independently-licensed package. Maintenance stays with
the people who know each engine best, and the core never becomes the bottleneck.

## Why build on Standard ASR?

- **Write once, run with any engine.** Code against the protocol, not the vendor. Switching
  from a cloud API to a local model (or the reverse) is a one-line model-key change — your
  integration work survives every vendor decision you'll make later.
- **One streaming model for every engine.** Real-time ASR is the wild west: some engines
  rewrite their interim results, some never revise a token, some merge already-emitted
  segments after a second decoding pass. Standard ASR unifies all of it under one event
  protocol with explicit stability guarantees — designed against an in-repo survey of 30+
  real engine APIs ([`docs/research/`](docs/research/)).
- **Audio negotiation, batteries included.** Hand over what you have — a file path, raw
  bytes, a NumPy array, a URL — and the framework negotiates and converts to whatever form
  the engine accepts, loudly reporting anything lossy. No more sample-rate guesswork.
- **No dependency hell, no licensing traps.** Each engine is an isolated, pip-installable
  plugin, so conflicting dependencies and restrictive licenses stay contained in the
  packages that carry them.
- **The choice goes to the user.** End users — especially for under-served languages and
  domains — install the engine that serves them best and use it immediately, without
  waiting for the app author to add support.

---

## Quickstart — see it work in 60 seconds

Standard ASR discovers compliant plugins through the `standard_asr.models` entry-point
group; each plugin exposes model presets keyed as `<engine_id>/<model_name>`. A tiny demo
plugin ships in `cookbook/std_dummy_asr` so you can try the whole workflow with **no extra
dependencies**. The demo plugin and sample client live in this repo, so clone it first:

```bash
git clone https://github.com/standard-voice/standard_asr.git
cd standard_asr

pip install standard-asr
pip install -e cookbook/std_dummy_asr      # the demo plugin (echoes a synthetic transcript)

standard-asr models list                   # discover installed engines
standard-asr compliance entrypoints        # check the plugins resolve correctly
python cookbook/sample_client.py           # discover -> instantiate -> transcribe
```

Use this flow as a template when building or trying your own plugin.

---

## Python usage

### Transcribe

Discover whatever compliant engines are installed, then transcribe:

```python
from standard_asr import discover_models

registry = discover_models()
engine = registry.create("dummy/echo")     # swap this key for any installed engine

# Pass the audio you already have — a file path, raw bytes, a base64 data URI, or a
# NumPy array. Standard ASR negotiates the right form for the chosen engine and converts
# only when needed (every lossy step is reported as a structured diagnostic).
result = engine.transcribe("meeting.wav")
print(result.text)
```

The **same app code** runs against any other compliant engine — only the model key changes
(e.g. swapping `dummy/echo` for a real local or cloud engine you've installed as a plugin).

Results always have the **same shape** — no format flags that turn the return value into a
string, no fields that appear and disappear. Render subtitles from any engine's result:

```python
from standard_asr import to_srt, to_vtt

print(to_srt(result))                      # works for every compliant engine
```

### Discover capabilities & configuration

Engines differ — that's the point. Instead of guessing, ask:

```python
engine.supports("batch.word_timestamps")          # True / False, fail-closed
engine.supports("streaming.guidance.phrase_hints")
engine.supports("streaming_input")                # can it consume live audio?

registry.config_schema("dummy/echo")              # the engine's init-config JSON Schema —
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
from standard_asr import AudioFormat

# Declare the wire format of the PCM frames you'll send (must match the engine's
# advertised wire_encodings).
audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=16_000)

async with engine.start_transcription(audio_format=audio_format) as session:
    session.feed(microphone())             # any (async) iterable of PCM byte chunks

    segments: dict[str, str] = {}
    async for event in session:
        if event.type in ("partial", "final"):
            segments[event.segment_id] = event.text   # partial: may change; final: settled
        elif event.type == "supersede":
            for old_id in event.old_ids:              # engine re-segmented (e.g. two-pass
                del segments[old_id]                  # rescoring); replacements follow
        render(segments)

print(session.result().text)               # collapse the session into a TranscriptionResult
```

Those three branches are the **complete core reduce** — handle them and your app is safe on
every compliant engine, including ones that rewrite interim text or merge segments after the
fact. Engines that never do these things simply never emit those events. Voice agents can go
further and act on `event.stable_until`, the engine's guarantee of how much of the text is
frozen and will never change.

> Not async? `SyncSession` wraps any streaming session behind a blocking iterator.
> See [`docs/spec/`](docs/spec/) for the full streaming contract — segment lifecycle,
> stability guarantees, reconnect semantics, and backpressure rules.

---

## Who benefits?

| You are… | You get… |
|---|---|
| **An application developer** | One integration that works with every compliant engine; zero vendor lock-in; automatic discovery of whatever the user installs. |
| **An ASR engine developer / researcher** | Focus on the model, not boilerplate. Implement one interface and get a CLI, a Web API server, and a compliance test suite **for free**. Reach the whole ecosystem instantly. |
| **An end user** | Access to cutting-edge models sooner, and the freedom to pick the engine that fits your language or domain — not whatever the app author happened to choose. |

---

## CLI

```bash
standard-asr models list                            # what's installed?
standard-asr models show dummy/echo                 # properties & capabilities
standard-asr transcribe dummy/echo audio.wav        # quick transcription
standard-asr serve                                  # expose engines over HTTP/WS
standard-asr doctor                                 # diagnose plugin dependency conflicts
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
> wants a different rate or only accepts files, the conversion happens automatically — and
> never silently: every lossy conversion is surfaced as a structured diagnostic. The heavy
> decoders stay optional — basic WAV works with zero extra installs.

### FastAPI server

```bash
pip install "standard-asr[server]"
standard-asr serve --host 0.0.0.0 --port 8000
```

See [`docs/spec/server.md`](docs/spec/server.md) for the full HTTP/WebSocket API contract,
and [`docs/spec/`](docs/spec/) for the protocol specification. The WebSocket endpoint covers
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

Start from the runnable examples in [`cookbook/`](cookbook/) (`std_dummy_asr` is a minimal
skeleton; `std_faster_whisper` wraps a real local engine), then validate with:

```bash
standard-asr compliance entrypoints
```

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

Pre-release, under active redesign with standard-library rigor: a normative, RFC-style
specification (`docs/spec/`), Pydantic v2 models, `pyright --strict`, 100% test coverage,
and CI across numpy 1.x/2.x and Python 3.10–3.14. The protocol's design decisions are
grounded in an in-repo survey of 30+ real ASR engines and APIs (`docs/research/`). The
authoritative material lives in-repo:

- `docs/spec/` — the protocol specification.
- `docs/research/` — the engine surveys the design is tested against.
- `CONTRIBUTING.md` — dev setup, the dependency policy, and the CI channel model.
- `cookbook/` — runnable example plugins (`std_dummy_asr`, `std_faster_whisper`).

## Communication

We use **Zulip** for development discussion: https://standard-voice.zulipchat.com

## Contributing

Please read [`CONTRIBUTING.md`](./CONTRIBUTING.md) before opening a pull request.

## License

Apache 2.0. See [LICENSE](./LICENSE).
