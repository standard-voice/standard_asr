---
title: Server API
---

# Server specification (HTTP / WebSocket API)

Standard ASR ships an optional FastAPI server (`standard-asr[server]`) that
exposes any discovered, compliant engine over HTTP, plus a WebSocket endpoint
for incremental streaming. This document is the authoritative contract for that
server; the implementation in `standard_asr.toolchain.server` conforms to it (any
divergence is a bug in the implementation, not the spec).

Launch with `standard-asr serve` or `standard_asr.toolchain.server.run(...)`.

## 1. Security & limits

- **No per-endpoint authentication.** v1 targets localhost / trusted-LAN use.
  Transcription is CPU/GPU-expensive and there is no quota or rate limiting.
  Before exposing beyond localhost, operators **MUST** front the server with a
  reverse proxy providing authentication and rate limiting.
- The declared-metadata, capability, and schema endpoints are deliberately
  readable without authentication. They expose declarations, not configured
  values or artifact status.
- **Validation errors never echo request input VALUES.** Every REST **422** that
  originates from a pydantic validation failure of the CLIENT's own request
  material — the global `RequestValidationError` handler **and** the
  standalone-`ValidationError` path for the `options` build — returns the
  **same** structured body and strips the offending
  `input` value (which FastAPI / pydantic echo by default) and **redacts
  credential-looking fields** (`api_key`, `token`, `secret`, `password`,
  `authorization`, …). This prevents a mis-placed secret (for example, an API key put in
  the JSON body or `options`) from being reflected back into the client, an
  intermediary proxy, or a copied bug report. **`loc` carries caller text
  too** — a rejected `extra_forbidden` key, a `dict[str, T]` field's mapping
  key, a non-string key's `repr` — so every component is filtered: one
  shaped like a field name (an identifier-ish token of ≤ 32 chars) or a
  pydantic structural marker (`[key]`) survives; anything longer or
  non-identifier-shaped — the structural signature of pasted credential
  MATERIAL (real key formats run 40+ characters or carry base64 padding) —
  becomes `[redacted-key]`. Naming a surviving key is deliberate DX on the
  **caller** surface (this 422 body, the WS `bad_request` message, CLI
  stderr): it points at a typo (`"optinos"`) or a mis-placed credential
  FIELD (`"api_key"`, whose name is not the secret; the value is stripped),
  and the sender already has whatever they sent. **Operator** surfaces
  (server/CI logs, compliance reports, `ConfigError` text) drop a rejected
  unknown key entirely even when it is field-name-shaped: the operator did
  not send it, cannot act on it, and those are the sinks IC.3 names. The body is

  ```json
  { "detail": [ { "type": "...", "loc": [ ... ], "msg": "..." }, ... ] }
  ```

  The safe structured fields (`type`, `loc`, `msg`) are preserved so the caller
  can still fix the request (and branch on the machine-readable `type`, for example,
  `extra_forbidden` for a rejected `provider_params` key). A standalone error's
  `loc` is anchored under the request field it came from (`["options", ...]`
  for the wire options). Keeping **one body shape per validation 422** means a
  cross-language client parses a single structure rather than discriminating
  string-vs-list per code. (A pydantic `ValidationError` from ENGINE
  construction or engine-side re-validation is **not** a 422 at all — the
  client cannot cause it, so §3.7 maps it to a scrubbed 500.)

  > A 422 that instead carries an engine-/standard-**authored** semantic message
  > (`UnsupportedFeatureError` — the only caller-fixable semantic rejection;
  > see §3.7, where `ConfigError` / `InvalidProviderParamError` map to a
  > scrubbed 500 because no wire request can cause them) has no pydantic
  > `loc`/`type` to expose and returns the string form
  > `{ "detail": "<authored message>" }`. These messages are written for the
  > caller (never a raw `input` echo). The two 422 forms are disjoint by cause:
  > **pydantic validation → list**, **authored semantic error → string**.
- **Request-body cap.** `DEFAULT_MAX_BODY_BYTES` = `16 * 1024 * 1024` (16 MiB),
  overridable per app via `create_app(max_body_bytes=...)`. Enforced by a
  pure-ASGI middleware in two layers, *before* the body is parsed:
  - **Declared size (early).** A non-integer `Content-Length` → **400**; a
    `Content-Length` over the cap → **413**, before any body is read.
  - **Actual size (true cap).** `Content-Length` is advisory — a chunked /
    streamed request may omit or under-state it. The middleware therefore counts
    body bytes off the ASGI receive channel and aborts with **413** the moment
    the cumulative total exceeds the cap, so an oversize body is never fully
    buffered or parsed downstream (no Content-Length-bypass gap).
  - The body-size middleware covers the **HTTP scope only**; the WebSocket
    surface (`/v1/stream`) is byte-bounded separately (see §4.4).
- **WebSocket frame / session caps.** The streaming bridge bounds bytes directly:
  - `DEFAULT_MAX_WS_FRAME_BYTES` (16 MiB) — maximum size of a single frame; it
    applies to both binary **audio** frames and the JSON **config/handshake**
    frame; overridable via `create_app(max_ws_frame_bytes=...)`.
  - `DEFAULT_MAX_WS_SESSION_BYTES` (256 MiB) — cumulative cap on total audio
    bytes ingested over one session; overridable via
    `create_app(max_ws_session_bytes=...)`.
  - Exceeding any of these closes the socket with a `payload_too_large` policy
    error frame (see §4.4) and logs the violation.
  - The WebSocket **transport** also imposes its own `ws_max_size` (uvicorn's
    default is 16 MiB), so the effective per-frame bound is
    `min(max_ws_frame_bytes, transport ws_max_size)`. `run()` passes
    `ws_max_size=max_ws_frame_bytes` so the app cap and transport cap match;
    behind another ASGI server the config-frame check still enforces the app cap.

## 2. Audio is **not** pre-decoded

The server **does not decode audio**. The upload is forwarded as an
`AudioInput` (`AudioBytes` for multipart, `AudioBase64` for JSON) directly into
the engine's own negotiation. The standard layer then decodes/resamples per the
engine's `accepted_input`, so per-engine sample-rate requirements are honored
and encoded-only / URL-only engines remain servable. The upload's true sample
rate is never silently overridden.

### 2.1 Runtime params: portable-only over the wire (D5)

Over the wire the server accepts **only** the portable standard `RuntimeParams`
set, modeled by `WireRuntimeParams` (the portable fields, `extra="forbid"`). The
portable set is exactly the fields in [protocol.md §3.1](./protocol.md)
— `language`, `candidate_languages`, `word_timestamps`, `diarization`, `prompt`,
`phrase_hints`, and the `on_unsupported` guidance-degradation policy field — so a
cross-language client can express the opt-in `on_unsupported="degrade_to_prompt"`
over the wire. `diarization` follows the three-way wire mapping of
protocol.md §RT 3.4: `{"diarization": {}}` requests diarization, `null` or
an absent key means not requested, and any nested key inside the marker object
is rejected with a 422 (REST) / `bad_request` (WS).
The engine-specific `provider_params` escape hatch is **discover-only, not
sendable**:

- It can be **discovered** — its JSON Schema is published at §3.6 for UI
  generation and tooling.
- It **cannot be sent.** It is not constructible from untyped wire JSON without
  the engine's params type, and accepting a raw object would let it reach the
  engine untyped and unvalidated. A request whose `options` (REST) or config
  `options` (WebSocket) include a `provider_params` key is therefore **rejected
  with a clear 422** (REST) / `bad_request` (WS) rather than silently dropped or
  mis-routed.

> The long-term JSON-Schema-over-wire path (validating `provider_params` against
> the discovered schema) is **deferred**; for v1 the escape hatch is in-process
> only (pass it to `transcribe(...)` / `start_transcription(...)` directly).

## 3. REST endpoints

### 3.1 `GET /v1/health`
Returns `{"status": "ok"}`.

### 3.2 `GET /v1/models`
Returns a list of `ModelInfo`:
`{"key": "<engine/model>", "engine_id": "...", "model_name": "..."}`.

This bulk endpoint reads entry-point identity only. It does not import every
installed plugin and does not include declared metadata or artifact status.

### 3.2.1 `GET /v1/metadata/{model}`

Returns the engine class's `DeclaredEngineMetadata` as canonical JSON. The endpoint
loads only the selected model's engine class and does not instantiate it. The
initial `artifacts` section exposes three independent static upper bounds:
artifact-lifecycle applicability, explicit-acquisition support, and possible
acquisition during inference.

An unknown model key returns **404**. A registered plugin that cannot load, an
invalid declaration, or a protocol line this core does not support (older or
newer) is a deployment fault and returns a scrubbed **500**. The endpoint never
fabricates `NO_ARTIFACT_LIFECYCLE` for an engine on an unsupported line.

### 3.3 `POST /v1/transcribe` (multipart form)
Transcribe an uploaded file.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `model` | form string | yes | Model key in `engine/model` format. |
| `file` | file upload | yes | Encoded audio payload (forwarded as `AudioBytes`). |
| `options` | form string | no | JSON object mapping onto the portable `WireRuntimeParams` set (§2.1). |

Returns a `TranscribeResponse`:
`{"model": "<engine/model>", "result": <TranscriptionResult>}`.

Wire note on segment timing: `result.segments[*].start`/`end` are
**number-or-null** (spec TR.2): `null` means the engine measured no such time
(`(start, null)` = start-only; `(null, null)` = unavailable; `(null, number)`
never occurs — it is construction-rejected). There is no reserved
`Segment.extra` key on the wire; the nullable values are the timing truth.

Un-parseable `options` JSON (malformed syntax) → **400**. A *semantically*
invalid `options` object — a bad value, an unknown key, or a non-portable
`provider_params` key (§2.1) — → **422**, before transcription.

### 3.4 `POST /v1/transcribe:json` (JSON body)
Transcribe a base64 / data-URI payload.

```json
{
  "model": "engine/model",
  "audio": "<base64 or data: URI>",
  "options": { "language": "en" }
}
```

- `audio` is forwarded as `AudioBase64`; decode failures surface as
  `AudioProcessingError` → **400** (see §3.7).
- `options` may be `null`. It is validated against the portable
  `WireRuntimeParams` set (§2.1); unknown keys and a non-portable
  `provider_params` key are rejected (`extra="forbid"`).
- A semantically invalid `options` object → **422** before transcription.

Returns a `TranscribeResponse` (same shape as §3.3).

### 3.5 `GET /v1/capabilities/{model}`
Returns the engine's declared capability tree as `canonical_json()` — read from
the engine **class** without instantiation. Every node carries a derived
`supported` field. **404** if the model key is unknown or declares no
capabilities; a registered model whose plugin fails to LOAD is an
engine/deployment fault → scrubbed **500** (§3.7), never a caller-blaming 404.

### 3.6 `GET /v1/params-schema/{model}`
Returns the JSON Schema of the engine's `provider_params` (read from the engine
class, for discovery / UI generation), or `{}` if the engine declares none.
**404** if the model key is unknown (a plugin that fails to load → scrubbed
**500**, §3.7). Note these params cannot yet be sent over the transcribe
endpoints (§2).

### 3.6.1 `GET /v1/config-schema/{model}`
Returns the JSON Schema of the engine's **init config** (read from the engine
class's `config_type`, without instantiation), or `{}` if the engine declares
no `config_type`. **404** if the model key is unknown (a plugin that fails
to load → scrubbed **500**, §3.7).

This is the wire-side discovery path for settings UIs (G.3.1): a client can
render an engine's configuration form **before** the engine is constructed —
construction may require the very values (credentials, `default_language`)
the form collects. Secret fields carry `format: password` / `writeOnly: true`
markers, so schema-driven UIs render them safely. The schema describes field
*shapes* only and never contains configured values, so — like capabilities and
params-schema — it is deliberately readable without authentication. Note that
the server itself does not accept engine construction over the wire in v1; the
collected config is consumed by the operator-side process that constructs the
engine (for example, `registry.create(key, **values)`).

> The `{model}` path segment matches the full `engine/model` key (it may contain
> a slash).

### 3.7 Error → HTTP status mapping

The transcribe endpoints map errors from **both** engine construction
(`model_registry.create`) and the `transcribe` call as follows:

| Condition | Status |
|---|---|
| Unknown / unparseable model key (`EntrypointValidationError` — the caller's key does not exist or is malformed) | **404** (authored detail: the caller's own key + available keys) |
| Registered model whose plugin fails to load/resolve (`FactoryLoadError` — the key resolved, a server-installed plugin is broken) | **500** (scrubbed generic detail; plugin import/annotation internals are safe-logged only — a 404 would blame the caller for a fault it cannot fix) |
| Required configuration absent from the SERVER environment (`ConfigurationRequiredError` — for example, a credential env var not set), at construction OR discovered lazily at call time | **503** (stable generic detail; the absent field names are deployment detail, safe-logged only) |
| Required inference artifacts are unavailable (`ArtifactUnavailableError`) or their acquisition failed (`ArtifactAcquisitionError`) | **503** (stable generic detail; reports, paths, actions, source URLs, and native error text are deployment detail) |
| Any other construction failure (`ConfigError`, `InvalidProviderParamError`, `ValidationError`, anything unexpected) | **500** (scrubbed) |
| Engine config/contract fault during transcription (`ConfigError` — for example, a bad `default_language` value; `EngineContractError` — for example, a malformed declared language tag or a missing IC.6 `default_language`; `InvalidProviderParamError`; a bare engine-side `ValidationError`) | **500** (scrubbed) |
| Unsupported standard feature / non-selectable language requested, strict mode (`UnsupportedFeatureError`) | **422** |
| Audio decode/processing failure (`AudioProcessingError`) | **400** |
| Un-parseable `options` JSON syntax (multipart, before transcription) | **400** |
| Semantically invalid `options` / non-portable `provider_params` (`WireRuntimeParams` build, before transcription) | **422** |
| Any other / unexpected error during transcription | **500** |

**Fault ownership (normative).** The wire surface gives the client exactly
two inputs: the model key and the portable request (`options` + audio).
Engine construction is `registry.create(model)` — **zero-arg** — and
`provider_params` never crosses the wire (`WireRuntimeParams` rejects it),
so `ConfigError` and `InvalidProviderParamError` are UNREACHABLE from a
request: wherever they surface (construction, transcription, session
establishment) they are engine/deployment faults, mirroring the compliance
suite's classification of the same states. Absent required config
(`ConfigurationRequiredError`, the state compliance *skips*) and an
inference-artifact failure are operator-side availability states → **503**,
whether hit at construction or lazily at call time. Every other
`ConfigError`/`EngineContractError`/
`InvalidProviderParamError` (the states compliance *fails*) → scrubbed
**500**. Client-caused rejections
each have their own type and status: `UnsupportedFeatureError` → 422,
request-model `ValidationError` → 422, `AudioProcessingError` → 400. Engine
faults never return field names, authored config detail, or validation
detail to the caller — those are safe-logged for the operator.

**422 body shape (§1).** The 422 rows above split by cause into two disjoint
body forms: a pydantic `ValidationError` from the client's own request
material (the `options` build, the request models) returns the **structured
list** `{ "detail": [ { "type", "loc", "msg" }, ... ] }` (with the offending
`input` stripped and credential fields redacted; `loc` anchored under
`["options"]`), while the authored `UnsupportedFeatureError` returns the
**string** form `{ "detail": "<authored message>" }`. The string-form
messages are written for the caller and carry no `input` echo. (An
engine-side `ValidationError` is never a 422: it maps to the scrubbed 500
above.)

The **500** response is non-leaking: it returns a stable generic message
(`"Internal transcription error. See server logs for details."` for the
transcribe path, `"Internal model construction error. ..."` for construction);
the raw exception text is logged server-side only, never returned (avoids
leaking internal paths or upstream/credential material).

**Metadata fault boundary (normative).** The discovery endpoints
(`/v1/metadata/{model}`, `/v1/capabilities/{model}`, `/v1/params-schema/{model}`,
`/v1/config-schema/{model}`) MUST wrap the WHOLE operation — class
resolution, the descriptor read, `canonical_json()` /
`model_json_schema()`, and the JSON projection — in ONE fault boundary,
starting from the model key. Resolution itself is third-party code:
loading the entry-point target imports the plugin, and the duck-type
checks (`_is_protocol`, `transcribe`) are metaclass/descriptor reads that
dispatch plugin machinery; then the attribute read dispatches whatever
descriptor (or metaclass property) the plugin installed, the projection
methods are the plugin's, and a custom `__get_pydantic_json_schema__` runs
inside the latter. A fault raised *during resolution* that is neither of
the mapped `EntrypointValidationError`/`FactoryLoadError` shapes
(a metaclass property carrying a `ValidationError`, say) would otherwise
escape BEFORE the boundary ever began — and any raise in the whole
stretch that escapes to the ASGI server yields an
undocumented plain 500 **and** — the reason this is a security rule —
bypasses the operator-log redaction below, letting the server's native
traceback logger render a `ValidationError`'s input echo. Mapping: an
unknown/unparseable model key → **404**; the endpoint's own deliberate
verdict (for example, "no capabilities declared") → passed through unchanged;
everything else → safe-logged scrubbed **500**. `BaseException` is not
caught. The projection MUST also be completed inside the boundary
(encoded eagerly, non-finite numbers rejected — they are not JSON), so a
plugin returning an unencodable payload is a scrubbed 500 rather than a
crash after the endpoint returned. These endpoints are unauthenticated
discovery surfaces, so "a plugin descriptor does not usually fail" is not
a boundary.

**Operator-log redaction (normative).** The server-side log is itself a
transport: it lands in CI logs, aggregators, and copy-pasted bug reports.
Generic exception TEXT is operator-log OK (an operator debugging a plugin
fault needs the real message), but a pydantic `ValidationError`'s input
echo is **NEVER** logged — not as the active exception, and not through
the `__cause__`/`__context__` chain: a plain `logger.exception` traceback
re-renders every link's raw message, `input_value=...` echo included, and
the standard layer deliberately chains the raw error under sanitized
wrappers (for example, `raise TranscriptionError(...) from exc`), so the chain is
the echo's normal road into the log.

The rule (reference implementation
`standard_asr.runtime.redaction.log_exception_safely`): walk the
`cause`/`context` chain — following `__cause__` when set, else
`__context__`, bounded and cycle-safe (`raise ... from None` only
suppresses the *display*; the suppressed error is still in `__context__`,
and its text may already be copied into the wrapper's message). A chain
with no `ValidationError` link logs the native traceback unchanged — the
operator keeps every frame. A chain carrying one logs a scrubbed one-line
summary instead (`standard_asr.runtime.redaction.safe_exception_summary`):
each link renders as `TypeName: text`, where a `ValidationError` link's
text is its sanitized loc/msg form (fields named, values never — the same
entry rules as the 422 body above) and any other link's text is its own
`str()` — except that a link whose message contains a line of a chained
`ValidationError`'s text byte-for-byte has its message withheld: a wrapper
built as `raise RuntimeError(f"engine failed: {exc}") from exc` (the
common honest mistake) copied the echo into its own message, and the
verbatim copy is caught by plain substring comparison.

The summary is bounded and line-safe: the chain walk is link-capped, each
link's text is whitespace-collapsed to a single line (a multi-line message
MUST NOT be able to forge a second record in a line-oriented log — the
collapse splits on every `str.splitlines` boundary, NEL and
`U+2028`/`U+2029` included) and length-capped, and a link whose own
`str()` raises degrades to a placeholder — the containment layer doing the
logging never crashes.

**Accepted limits (per the AGENTS.md trust model).** This is an accident
rule, not a prover: a paraphrased or re-encoded echo, a wrapper that
copies the error's text and then discards the chain, or a wrapper whose
explicit benign `__cause__` hides a `ValidationError` left in
`__context__`, can still reach the log. Those residuals are accepted —
closing them means proving properties of third-party code and
introspecting pydantic/CPython internals, which is out of budget by
design (plugins are trusted code, and the machinery that once tried
produced more defects than the residuals it closed). If a residual bites
in practice, the answer is a targeted rule in `runtime.redaction`, not a
prover. This is the operator-log half of IC.3's "plaintext credentials
MUST NEVER be logged".

## 4. WebSocket endpoint `/v1/stream/{model}`

Bridges a WebSocket to an engine streaming session (the incremental
`audio_format` path). The `{model}` segment is the full `engine/model` key.

### 4.1 Frame protocol

1. **Config frame (client → server).** After the socket is accepted, the client
   sends exactly one JSON **text** frame:
   ```json
   {
     "audio_format": { "encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1 },
     "options": { "language": "en" }
   }
   ```
   - `audio_format.encoding` MUST be one of the engine's `wire_encodings`;
     `sample_rate` is in Hz (> 0); `channels` is optional and defaults to `1`.
   - `options` maps onto the portable `RuntimeParams` set, or may be `null`.
   - The config frame is a **closed object**: `audio_format` and `options`
     are its only legal top-level keys, and an unknown key is a loud
     `bad_request` naming the offending key — never silently ignored (a
     client typo like `"optinos"` MUST NOT start the session on defaults
     while the client believes its options were applied). A key shaped
     like credential material (long / non-identifier) is named as
     `[redacted-key]` instead (§1's key-side rule).
   - Only caller-fixable handshake failures map to `bad_request`
     (non-text first frame, unparseable JSON, request validation — with
     pydantic detail sanitized); any other failure during the handshake is
     an internal fault: scrubbed `internal_error`, specifics safe-logged.
   - The config frame MUST be a **text** frame. A binary first frame is a
     malformed handshake — **binary** frames are reserved for raw audio (§4.1.3),
     so the first-frame type is the sole discriminator between config and audio —
     and is rejected with a `bad_request` policy frame (§4.2), then the socket
     closes. (A conformant server does **not** parse a binary first frame as
     config; relying on that leniency breaks against strict implementations.)

2. **Initial diagnostics frame (server → client).** Immediately after
   `start_transcription` returns — *before* the bridge enters the session, and
   *before* any audio is sent — the server forwards the
   standard-layer diagnostics attached at session establishment (best-effort
   parameter degrade, language resolution, audio conversion), so the client learns
   **why** a parameter was dropped or changed (the REST path returns these on the
   result). Sent as a single JSON **text** frame, and **only when** the session
   has establishment-time diagnostics:
   ```json
   { "type": "diagnostics", "diagnostics": [ { "level": "...", "code": "...", "message": "...", "param": "...", "provided": ..., "effective": ... } ] }
   ```
   These messages are standard-layer-authored (not raw exception text), so they
   are forwarded verbatim. A client with no degradations never receives this
   frame. Building this frame MUST sit inside a fault boundary like every
   other stage: it runs after session establishment has returned and before
   the forward loop's own catch begins, so an unprojectable diagnostic
   otherwise killed the route with an unhandled exception — bypassing the
   operator-log redaction on the way out. It fails LOUDLY (a scrubbed
   `internal_error` frame, then close) rather than dropping the frame and
   streaming on: diagnostics are the channel that tells a client something
   degraded, so silently omitting them hides exactly what the client needs.

   Further `diagnostics` frames arrive **mid-stream** as the session accrues
   them (an engine's `emit_diagnostic`, a lifecycle suppression), so the WS
   view converges on the same set the in-process/REST layers report (G.5.2).
   Entries **append**, with one exception the client MUST honor: the
   `diagnostics_truncated` overflow summary is a **singleton**. The session's
   diagnostic channel is bounded; past the cap the standard layer keeps a
   single aggregated summary and updates its per-code tally in place, so a
   later `diagnostics_truncated` entry **supersedes** the one already
   delivered rather than sitting beside it. (Without that rule the wire view
   would freeze on the counts from the first overflow while the Python view
   converged on the final ones.) The final tally is always delivered: the
   bridge takes one last delta after the terminal event.

3. **Audio frames (client → server).** Subsequent **binary** frames are raw PCM
   chunks, fed to the session via `send_audio`. **Any text frame** OR a
   disconnect signals end-of-audio (`end_audio`); after that, no further audio
   is accepted.

4. **Event frames (server → client).** The server streams each
   `TranscriptionEvent` back as a JSON text frame
   (`event.model_dump(mode="json")`) until a terminal event, then closes the
   socket. Event `type` is one of
   `"partial" | "final" | "supersede" | "progress" | "done" | "error"`. A client
   that disconnects mid-stream simply ends the session (remaining events are
   dropped).

### 4.2 Two distinct error shapes — both possible

Client authors MUST handle **both**:

- **Pre-bridge error** (before streaming starts: bad config frame, unknown
  model, an unavailable/unconstructable engine, an unsupported feature, or an
  internal fault). Sent as a single frame, then the socket closes (the frame's
  `code` is the sole verdict channel — no WS close-code semantics are
  defined):
  ```json
  { "type": "error", "code": "bad_request" | "unknown_model" | "service_unavailable" | "unsupported" | "internal_error", "message": "..." }
  ```
  - `bad_request`: malformed config frame / invalid `audio_format` / invalid
    `options` — that is, the client's own request material failed validation.
  - `unknown_model`: the caller's model key does not exist or is malformed
    (`EntrypointValidationError` only — a registered model whose plugin
    fails to LOAD is an engine/deployment fault and maps to the scrubbed
    `internal_error` frame, mirroring the REST 404/500 split).
  - `service_unavailable`: required configuration is absent from the server
    environment, or an inference-artifact failure prevents session
    establishment. Establishment covers session OPEN as well: an engine that
    materializes artifacts or checks credentials when the session opens (the
    `_open` hook) reaches the client through this same frame. On that one path
    the client may already hold the §4.1.2 diagnostics frame, because
    establishment succeeded and attached its diagnostics before the open
    failed: both facts are true and both are delivered. "Sent as a single
    frame" describes the error's own shape, not a promise that it is the first
    frame on the socket, so a client MUST dispatch on `type` rather than treat
    frame one as the verdict. Every earlier pre-bridge failure returns before
    the diagnostics frame is built, so there the error is the only frame. The
    open failure reaches the client here because no event channel exists yet.
    This is the WebSocket twin of the REST 503 (§3.7): an
    operator-side state with a stable generic `message`, never field names,
    artifact reports, paths, actions, URLs, or native error text.
  - `unsupported`: engine cannot start a streaming session for this request
    (`UnsupportedFeatureError` — the only caller-fixable establishment
    rejection; by establishment every client input is already validated, so
    a bare `ValueError` is never mapped here).
  - `internal_error`: an engine/deployment fault — any other construction
    failure, an engine config/contract fault at establishment
    (`ConfigError`, for example, a bad `default_language` value;
    `EngineContractError`, for example, a missing IC.6 `default_language`;
    `InvalidProviderParamError`, which no wire client can legally cause; an
    engine-side `ValidationError`; a bare `ValueError` surviving past
    request validation), or an unexpected establishment fault.
    The `message` is a stable generic string (the raw cause is logged
    server-side only, never sent — mirrors the REST scrubbed-500 contract,
    §3.7).

- **In-stream error** (a `TranscriptionEvent` with `type == "error"`, produced by
  the engine once streaming has begun). This shape is **different**: it has
  `code`, `recoverable`, and `retriable_after` — and **no** `message` field:
  ```json
  { "type": "error", "code": "session_timeout", "recoverable": false, "retriable_after": null, ... }
  ```
  (It carries the full `TranscriptionEvent` field set; other fields are `null`
  or defaults.)

  `artifact_unavailable` and `artifact_acquisition_failed` are distinct
  terminal codes when an inference-artifact failure occurs while the engine is
  producing events. Both set `recoverable=false`. `artifact_unavailable` has no
  retry delay; `artifact_acquisition_failed` can retain a nonnegative
  `retriable_after`, but the client must open a new session for a retry. A
  failure before the first event — construction, establishment, or session
  open — cannot use these event fields and remains the coarser
  `service_unavailable` shape above.

  > **Non-leak (mirrors the REST 500 contract, §3.7).** For `error` events the
  > server **drops the `extra` payload** before sending (it is emptied to `{}`).
  > The standard streaming layer stores a human-readable message under
  > `extra["detail"]` — for the `engine_error` catch-all this is **already the
  > input-echo-free summary** (`safe_exception_summary`), produced at the
  > producer's catch site: by the time the server sees the event the exception
  > chain no longer exists, so redaction cannot be deferred to the log layer.
  > The summarized text may still name filesystem paths or upstream hosts
  > (deliberate operator content), so it is never forwarded to the
  > (unauthenticated) client; the server logs the dropped detail **as the
  > opaque string it received** — it does not (and cannot) re-analyze a
  > chain, and it renders nothing else: a value installed *past* the
  > declared space (a mutated `extra` mapping, `model_construct`) reaches
  > the log line before the drop, so the bridge logs it only when it is an
  > exact `str` (a `ValidationError`'s `repr` echoes its input — the very
  > echo the drop protects against); anything else is withheld as such.
  > An `error` event an ENGINE constructs directly carries plugin-authored
  > `extra` content: the standard cannot vet it, which is precisely why the
  > drop-before-send applies to every `error` event regardless of origin
  > (engine authors own the safety of what they log server-side themselves).
  > The safe structured fields (`code`, `recoverable`, `retriable_after`,
  > `segment_id`, and the gap/reconnect fields) are preserved. "Drop before
  > send" is an ORDER, not an outcome: the field MUST be excluded from the
  > serialization, not overwritten after it. Serializing first made the whole
  > payload's fate depend on the very field being discarded — an `extra` value
  > with no JSON form raised inside the dump, so the client lost every safe
  > structured field above to a value that was never going to be sent.
  >
  > Non-`error` events keep their `extra` payload (it is forwarded verbatim):
  > `extra` is the engine-specific, non-portable event slot defined in the
  > in-process protocol (`protocol.md` §4.1). Only `error` events scrub it.

### 4.3 Scope limit (v1)

The WebSocket surface supports **only** the incremental `audio_format` path
(declare format, push raw PCM frames, receive live events). The
**whole-input + streaming-output** path
(`start_transcription(audio=...)`, OpenAI SSE style, spec §7.3) is **NOT**
exposed over WebSocket in v1. For those engines, use the batch REST endpoints
(`POST /v1/transcribe` or `POST /v1/transcribe:json`).

### 4.4 Frame / session byte caps (DoS bound)

The HTTP body-size guard (§1) does not cover the WebSocket scope, so the stream
bridge enforces its own per-frame and per-session byte caps (§1, configurable
via `create_app`):

- The **config/handshake frame** exceeding `max_ws_frame_bytes` (checked before
  it is parsed, so it is bounded by the app cap and not only the transport
  `ws_max_size`), **or**
- a single binary **audio** frame exceeding `max_ws_frame_bytes`, **or**
- a cumulative session total exceeding `max_ws_session_bytes`,

is rejected: a single policy frame is sent and the socket closes (the violation
is also logged server-side; for audio caps the input is ended first):

```json
{ "type": "error", "code": "payload_too_large", "message": "..." }
```

This is distinct from the §4.2 in-stream `error` event (an engine-produced
`TranscriptionEvent`); the policy frame is emitted by the **server**, not the
engine, and carries a human-readable `message` (the cap that was exceeded; it
contains no internal/engine detail). The effective per-frame bound is
`min(max_ws_frame_bytes, transport ws_max_size)`; `run()` wires
`ws_max_size=max_ws_frame_bytes` so they coincide (§1).

A failure on the audio-input pump (for example, a client protocol violation such as
sending audio after the session ended) is likewise never swallowed silently: it
is logged server-side and surfaced as a single generic, **non-leaking** frame
before teardown:

```json
{ "type": "error", "code": "stream_input_error", "message": "Audio input failed. See server logs for details." }
```
