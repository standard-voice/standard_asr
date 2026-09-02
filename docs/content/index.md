---
title: Overview
---

# Standard ASR

**The open standard interface between applications and speech-recognition engines.**

Standard ASR defines a vendor-neutral protocol. Applications integrate speech-to-text once and gain every compliant engine; engines implement it once and reach every application. Think USB-C for ASR inference.

> **Pre-release:** Standard ASR is a work in progress. Breaking changes may occur before `v1.0.0`. Try it out and tell us what you think.

## Who is this for?

| Who you are | What you get |
| ---------- | ---------- |
| **An application developer** | One integration that works with every compliant engine. Zero vendor lock-in. Automatic discovery of whatever the user installs. |
| **An ASR engine developer** | Focus on the model. Implement one interface and get a CLI, a reference server, and a compliance test suite for free. |
| **An end user** | Install the plugin whose engine fits your language or domain and use it immediately, without waiting for the app author to add support. |

## Start here

- **[Quickstart](./quickstart.md)** -- transcribe in under a minute.
- **[Installation](./installation.md)** -- install options and optional extras.
- **[Discover & Use](./app-developers/discover-and-use.md)** -- the full app-developer guide.
- **[Inference artifacts](./reference/artifacts.md)** -- see what a model needs on disk, and acquire it before the first request.
- **[Adapt an ASR System](./engine-authors/adapt-an-asr-system.md)** -- build a compliant plugin.
- **[API Reference](./reference/index.md)** -- the complete public surface.
