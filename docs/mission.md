# Mission & Philosophy

## Why Standard ASR?

Speech recognition never got its standard interface. Every ASR library and cloud
API ships its own calling convention, audio-input rules, and streaming protocol.
Integrating one engine means writing an adapter; integrating five means
maintaining five. In practice, most applications hard-wire two or three engines,
and their users are stuck with whatever languages and domains those engines happen
to cover.

Meanwhile, the model that would actually serve them best already exists -- as an
open-source checkpoint, a cloud endpoint, or a research prototype. The problem is
not a lack of good ASR; it is the absence of a shared protocol that lets
applications and engines meet without per-pair integration work.

## Mission

**Become the standard interface for ASR inference.**

Standard ASR defines a vendor-neutral protocol for the application-to-engine
boundary. Like USB-C for physical connectors, it lets both sides implement once
and interoperate with everything on the other side.

- Applications code against the protocol and gain every compliant engine.
- Engines implement it once and reach every application.
- Switching engines becomes a one-line model-key change, not another adapter.

### Streaming semantics are the core value proposition

Real-time ASR is the most fragmented part of the ecosystem: some engines rewrite
interim results, some never revise a token, some merge segments after a second
decoding pass. Standard ASR unifies all of this under one event protocol with
explicit stability guarantees -- designed against an in-repo survey of 30+ real
engine APIs.

### Two layers, kept in sync

The standard has two layers that share the same capability model, result schema,
and event semantics:

- **In-process Python protocol** -- the zero-copy layer for local inference and
  the host of the plugin ecosystem.
- **Wire protocol (HTTP / WebSocket)** -- the cross-language layer, so non-Python
  applications get the same capabilities via the network.

## Philosophy

### Application-developer friendly

The primary stakeholder. Zero-config, zero-surprise, zero-ambiguity.
Battery-included where it helps (audio loading, SRT/VTT renderers), but heavy
dependencies stay optional.

### ASR-developer friendly

Low barrier to publish a compliant plugin. Implement one interface and get a CLI,
an HTTP/WebSocket server, and a compliance test suite for free.

### Explicit over implicit

Silent wrong results are the cardinal sin. When in doubt, fail loudly or emit a
structured diagnostic -- never silently degrade. When convenience and
explicitness conflict, correctness wins.

### Standard-library rigor

This is infrastructure others build on for years. Types complete, boundaries
sharp, error paths explicit, no implicit behavior.

### Security by default

Credentials use `SecretStr`. URLs are validated (HTTPS-only, no SSRF). Unsafe
options require explicit opt-in.

## Stakeholders

1. **Application developers** (primary) -- one stable interface, no vendor
   lock-in, zero-config discovery.
2. **ASR engine developers** -- focus on models, not plumbing. Implement once,
   reach the whole ecosystem.
3. **End users** -- choose the best ASR for their language or domain. Install a
   plugin, use it immediately.
