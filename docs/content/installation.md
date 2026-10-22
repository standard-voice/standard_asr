---
title: Installation
---

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

The core is intentionally light: only `numpy` and `pydantic`. Everything heavy is an opt-in extra.

## Optional extras

| Extra      | What it adds | Pulls in |
| ---------- | ------------ | -------- |
| **(core)** | The protocol, engine discovery, audio negotiation, and the CLI. Decodes basic 8/16-bit PCM WAV files with the standard library. | `numpy`, `pydantic` |
| **audio**  | Battery-included audio loading: MP3, FLAC, OGG, raw bytes, base64. Handles decoding, resampling, and channel mixing. | `soundfile`, `scipy` (+ optional system FFmpeg for M4A and more) |
| **server** | FastAPI server exposing any compliant engine over HTTP and WebSocket. | `fastapi`, `starlette`, `python-multipart`, `uvicorn`, `websockets` |

```bash
pip install "standard-asr[audio]"
pip install "standard-asr[audio,server]"
```

## Install an engine plugin

Standard ASR discovers engines automatically via entry points. Install a plugin and it appears in `standard-asr list`. Each engine is its own package; experimental plugins install from their repo until they publish to PyPI:

```bash
pip install "std-faster-whisper @ git+https://github.com/standard-voice/std-faster-whisper.git"
```

## Install from source

To try unreleased changes, install the core straight from the repository (this pulls a snapshot of `main`, not an editable checkout):

```bash
pip install "standard-asr @ git+https://github.com/standard-voice/standard_asr.git"
```

To work on Standard ASR itself, clone the repository and follow [CONTRIBUTING.md](https://github.com/standard-voice/standard_asr/blob/main/CONTRIBUTING.md).

## Verify

```bash
standard-asr list                      # discover installed engines
standard-asr compliance entrypoints    # verify plugins resolve correctly
standard-asr doctor                    # diagnose dependency conflicts
standard-asr doctor --network          # also check that this machine reaches the Hugging Face endpoint
```

## Behind a proxy or a mirror

An engine downloads model weights with its own library, and that library reads the network settings from the environment. Standard ASR passes the environment through unchanged ([download policy](./specification/download-policy.md), section 5). Set the variables in the environment that starts the process. huggingface_hub reads `HF_ENDPOINT` the first time it is used, and reads the proxy and certificate variables once, when it creates its shared client.

| Variable | What it sets |
| --- | --- |
| `HF_ENDPOINT` | The Hugging Face endpoint, such as a mirror your organization runs. Standard ASR suggests none. |
| `HTTPS_PROXY`, `HTTP_PROXY` | The proxy for HTTPS requests and for plain-HTTP requests. A user name and a password go in the URL, as in `http://user:password@proxy:8080`. |
| `NO_PROXY` | The hosts that skip the proxy, such as `localhost,127.0.0.1` for a server on this machine. |
| `SSL_CERT_FILE` | A certificate bundle to trust, for a proxy that inspects TLS. For `httpx`, it replaces the `certifi` bundle. |

What each first-party plugin reads, checked with the versions their lock files pin (huggingface_hub 1.19.0, `httpx` 0.28.1, `websockets` 16.0):

| Plugin | Sends its requests with | `HF_ENDPOINT` | Proxy variables | Certificates |
| --- | --- | --- | --- | --- |
| std-faster-whisper | huggingface_hub, which uses `httpx` | Read | `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY` | `SSL_CERT_FILE`, else `SSL_CERT_DIR`, else the `certifi` bundle |
| std-mlx-audio | huggingface_hub, directly and through transformers | Read | The same | The same |
| std-qwen3-asr, HTTP requests | `httpx`, to DashScope or a vLLM server | Does not apply: it downloads no weights | The same | The same, unless the engine config turns `verify_tls` off |
| std-qwen3-asr, vLLM realtime stream | `websockets` | Does not apply | `ws://`: `WS_PROXY`, `SOCKS_PROXY`, `HTTPS_PROXY`, then `HTTP_PROXY`. `wss://`: `WSS_PROXY`, `SOCKS_PROXY`, then `HTTPS_PROXY`. `NO_PROXY`; not `ALL_PROXY` | Python's default certificate file and directory, which `SSL_CERT_FILE` and `SSL_CERT_DIR` replace; on Windows, the system store as well. The engine config's `verify_tls` does not apply to it |

On macOS and Windows, these clients also use a proxy set in the operating system's settings, read through Python's `urllib.request.getproxies()`. They use it only while no proxy variable is set, and `NO_PROXY` counts as one.

Two cases need care:

- **Files stored with Xet.** Hugging Face stores the large files of many repositories with Xet, its storage system. huggingface_hub hands such a file to `hf_xet`, a separate client with its own network code, at 1.5.1 in the lock files. This project has not checked which settings `hf_xet` reads. If such a download fails behind a proxy, `HF_HUB_DISABLE_XET=1` makes huggingface_hub download the file itself.
- **A vLLM server on this machine or your network.** With a proxy set, list the server's host in `NO_PROXY`, or its requests go to the proxy. If your proxy comes from the operating system's settings, setting `NO_PROXY` switches it off, so set `HTTPS_PROXY` and `HTTP_PROXY` as well. For a `ws://` server URL, `websockets` takes `HTTPS_PROXY` before `HTTP_PROXY`, unlike `httpx`.

`standard-asr doctor --network` sends one request to the Hugging Face endpoint with huggingface_hub's own client, and reports whether this machine reaches it and what to do when it does not.
