---
title: Mission
---

# Mission & philosophy

## Why Standard ASR?

Speech recognition never got its standard interface. Every ASR library and cloud API ships its own calling convention, audio-input rules, and streaming protocol. Integrating one engine means writing an adapter; integrating five means maintaining five. In practice, most applications hard-wire two or three engines, and their users are limited to whatever languages and domains those engines cover.

Meanwhile, the model that would serve them best already exists -- as an open source checkpoint, a cloud endpoint, or a research prototype. The problem is not a lack of good ASR; it is the absence of a shared protocol that lets applications and engines meet without per-pair integration work.

## Mission

**Become the standard interface for ASR inference.**

Standard ASR defines a vendor-neutral protocol for the application-to-engine boundary. Like USB-C for physical connectors, it lets both sides implement once and interoperate with everything on the other side.

- Applications code against the protocol and gain every compliant engine.
- Engines implement it once and reach every application.
- Switching engines becomes a one-line model-key change, not another adapter.

### Streaming semantics are the core value proposition

Real-time ASR is the most fragmented part of the ecosystem: some engines rewrite interim results, some never revise a token, some merge segments after a second decoding pass. Standard ASR unifies all of this under one event protocol with explicit stability guarantees -- designed against an in-repo survey of 30+ real engine APIs.

### Two layers, kept in sync

The standard has two layers that share the same capability model, result schema, and event semantics:

- **In-process Python protocol** -- the zero-copy layer for local inference and the host of the plugin ecosystem.
- **Wire protocol (HTTP / WebSocket)** -- the cross-language layer, so non-Python applications get the same capabilities via the network.

## Philosophy

### Application developer first

The application developer is the one we design for first. When the three stakeholders pull in different directions, the application developer wins. An installed engine works with nothing configured on the application's side: discovery is automatic, and an engine that needs settings declares them in a typed model an application can turn into a form. The library surprises nobody and leaves nothing ambiguous. It ships the pieces that help, such as audio loading and the SRT/VTT renderers, but heavy dependencies stay optional, behind the `[audio]` and `[server]` extras.

### Engine-author friendly

Getting an engine to users takes little work. Implement one interface, and you get the CLI, the reference server, and the compliance test suite for free. From then on the engine works with every Standard ASR application, without any application having to change.

### Explicit over implicit

Whatever happens, show it: no swallowed errors, no silent degradation, no fake success, no hidden data flows. Silent wrong results are the cardinal sin. When in doubt, fail loudly or emit a structured diagnostic. A warning says what can go wrong and what to do about it, in a tone that matches the real risk. Convenience never means silence: do the helpful thing and say what you guessed or lost. When nothing helpful can be done, fail loudly: an error the developer can fix beats a silent wrong transcript.

### Standard-library rigor

Other people build on this for years, so it is written like the standard library: complete types, sharp boundaries, explicit error paths, no implicit behavior. You should be able to learn how something behaves from its signature and its docstring, without reading the code. Every name is a design decision.

### Long-term optimum

We decide like the people who maintain this for the next ten years. Before the first stable release, we pick the design that is best for all three stakeholders over that span, even if it breaks backward compatibility. We never keep a worse design just because the better one is more work. But a design keeps its complexity forever, so that complexity counts as part of the design, and a bigger design has to justify itself against the smaller one that would also work. The optimum sets the destination, not the step size: we land the work in small, reviewable steps.

### Trust model

Plugins are trusted code; data from outside is not. The application, the library, and every installed plugin are one party. Nothing in the library can stop a plugin that means harm, so the security layer defends against accidents, not adversaries. Data from outside is untrusted: whatever the application takes from its users, and every request the reference server receives. Authentication and rate limiting belong to the deployment. The operator is trusted, but the library never reads a safety switch such as `allow_private_urls` from an environment variable: a check turned off there is a check nobody sees. Before we add a defense, we ask who it stops and whether it is ours to stop.

## Stakeholders

1. **Application developers** (primary) -- every ASR engine through one stable interface, no vendor lock-in, zero-config discovery.
2. **ASR engine authors** -- a low barrier to getting an engine to users. Implement one interface, and the engine works with every Standard ASR application. Focus on models, not plumbing.
3. **End users** -- choose the best ASR for their language, domain, and hardware. Install a plugin and use it immediately, without changing the application.
