# Server Specification (HTTP / WebSocket API)

Standard ASR ships an optional FastAPI server (`standard-asr[server]`) that
exposes any discovered, compliant engine over HTTP, plus a WebSocket endpoint
for incremental streaming. The implementation in `standard_asr.server` is the
source of truth; this document describes exactly what it does.

Launch with `standard-asr serve` or `standard_asr.server.run(...)`.

## 1. Security & Limits

- **No per-endpoint authentication.** v1 targets localhost / trusted-LAN use.
  Transcription is CPU/GPU-expensive and there is no quota or rate limiting.
  Before exposing beyond localhost, operators **MUST** front the server with a
  reverse proxy providing authentication and rate limiting.
- The capability and params-schema endpoints are deliberately readable without
  auth (spec §3.1 / §C: declared metadata is discoverable without instantiation
  or authentication).
- **Request-body cap.** `DEFAULT_MAX_BODY_BYTES` = `16 * 1024 * 1024` (16 MiB),
  overridable per app via `create_app(max_body_bytes=...)`.
  - Enforced *before* decoding, by a pure-ASGI middleware inspecting
    `Content-Length`. Oversize → **413**; non-integer `Content-Length` → **400**.
  - A chunked/streamed request with no `Content-Length` bypasses the early
    guard, but both transcribe endpoints re-check the materialised payload size
    (`len(file)` / `len(audio)`) and reject oversize with **413**.
  - The body-size middleware covers the **HTTP scope only**; the WebSocket
    surface (`/v1/stream`) is byte-bounded separately (see §4.4).
- **WebSocket audio caps.** The streaming bridge bounds audio bytes directly:
  - `DEFAULT_MAX_WS_FRAME_BYTES` (16 MiB) — maximum size of a single binary
    audio frame; overridable via `create_app(max_ws_frame_bytes=...)`.
  - `DEFAULT_MAX_WS_SESSION_BYTES` (256 MiB) — cumulative cap on total audio
    bytes ingested over one session; overridable via
    `create_app(max_ws_session_bytes=...)`.
  - Exceeding either cap closes the socket with a `payload_too_large` policy
    error frame (see §4.4) and logs the violation.

## 2. Audio is NOT pre-decoded

The server **does not decode audio**. The upload is forwarded as an
`AudioInput` (`AudioBytes` for multipart, `AudioBase64` for JSON) directly into
the engine's own negotiation. The standard layer then decodes/resamples per the
engine's `accepted_input`, so per-engine sample-rate requirements are honored
and encoded-only / URL-only engines remain servable. The upload's true sample
rate is never silently overridden.

Only the portable standard `RuntimeParams` set is accepted over the wire.
Engine `provider_params` cannot be *sent* (they are not constructible without
the engine type); they can only be *discovered* via §3.6.

## 3. REST Endpoints

### 3.1 `GET /v1/health`
Returns `{"status": "ok"}`.

### 3.2 `GET /v1/models`
Returns a list of `ModelInfo`:
`{"key": "<engine/model>", "engine_id": "...", "model_name": "..."}`.

### 3.3 `POST /v1/transcribe` (multipart form)
Transcribe an uploaded file.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `model` | form string | yes | Model key in `engine/model` format. |
| `file` | file upload | yes | Encoded audio payload (forwarded as `AudioBytes`). |
| `options` | form string | no | JSON object mapping onto `RuntimeParams`. |

Returns a `TranscribeResponse`:
`{"model": "<engine/model>", "result": <TranscriptionResult>}`.

Invalid `options` JSON or `RuntimeParams` → **400** before transcription.

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
- `options` may be `null`. Unknown top-level keys are rejected (`extra="forbid"`).
- Invalid `options` (`RuntimeParams`) → **400** before transcription.

Returns a `TranscribeResponse` (same shape as §3.3).

### 3.5 `GET /v1/capabilities/{model}`
Returns the engine's declared capability tree as `canonical_json()` — read from
the engine **class** without instantiation. Every node carries a derived
`supported` field. **404** if the model is unknown or declares no capabilities.

### 3.6 `GET /v1/params-schema/{model}`
Returns the JSON Schema of the engine's `provider_params` (read from the engine
class, for discovery / UI generation), or `{}` if the engine declares none.
**404** if the model is unknown. Note these params cannot currently be sent
over the transcribe endpoints (§2).

> The `{model}` path segment matches the full `engine/model` key (it may contain
> a slash).

### 3.7 Error → HTTP status mapping

The transcribe endpoints map errors from **both** engine construction
(`model_registry.create`) and the `transcribe` call as follows:

| Condition | Status |
|---|---|
| Unknown / unloadable model (`EntrypointValidationError` / `FactoryLoadError`) | **404** |
| Client config error during construction — bad config, missing credentials, or validation (`ConfigError`, `InvalidProviderParamError`, `ValidationError`) | **422** |
| Invalid provider param / unsupported standard feature / config error / validation error during transcription (`InvalidProviderParamError`, `UnsupportedFeatureError`, `ConfigError`, `ValidationError`) | **422** |
| Audio decode/processing failure (`AudioProcessingError`) | **400** |
| Bad `options` (parse / `RuntimeParams` build, before transcription) | **400** |
| Any other / unexpected error (construction or transcription) | **500** |

Engine construction errors are mapped the same way (`unknown model → 404`,
client config/validation → 422, unexpected → 500); they do **not** escape as a
non-spec 500.

The **500** response is non-leaking: it returns a stable generic message
(`"Internal transcription error. See server logs for details."` for the
transcribe path, `"Internal model construction error. ..."` for construction);
the raw exception text is logged server-side only, never returned (avoids
leaking internal paths or upstream/credential material).

## 4. WebSocket Endpoint `/v1/stream/{model}`

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

2. **Audio frames (client → server).** Subsequent **binary** frames are raw PCM
   chunks, fed to the session via `send_audio`. **Any text frame** OR a
   disconnect signals end-of-audio (`end_audio`); after that, no further audio
   is accepted.

3. **Event frames (server → client).** The server streams each
   `TranscriptionEvent` back as a JSON text frame
   (`event.model_dump(mode="json")`) until a terminal event, then closes the
   socket. Event `type` is one of
   `"partial" | "final" | "supersede" | "progress" | "done" | "error"`. A client
   that disconnects mid-stream simply ends the session (remaining events are
   dropped).

### 4.2 Two distinct error shapes — both possible

Client authors MUST handle **both**:

- **Pre-bridge error** (before streaming starts: bad config frame, unknown
  model, a client config error during engine construction, an unsupported
  feature, or an internal construction fault). Sent as a single frame, then the
  socket closes:
  ```json
  { "type": "error", "code": "bad_request" | "unknown_model" | "unsupported" | "internal_error", "message": "..." }
  ```
  - `bad_request`: malformed config frame / invalid `audio_format` / invalid
    `options`, **or** a client config error surfaced during engine construction
    (`ConfigError` / `InvalidProviderParamError` / `ValidationError` — bad
    config, missing credentials; mirrors the REST 422 mapping, §3.7).
  - `unknown_model`: model key does not resolve (`EntrypointValidationError` / `FactoryLoadError`).
  - `unsupported`: engine cannot start a streaming session for this request
    (`UnsupportedFeatureError` / `ValueError`).
  - `internal_error`: an unexpected fault during engine construction. The
    `message` is a stable generic string (the raw cause is logged server-side
    only, never sent — mirrors the REST scrubbed-500 contract, §3.7).

- **In-stream error** (a `TranscriptionEvent` with `type == "error"`, produced by
  the engine once streaming has begun). This shape is **different**: it has
  `code`, `recoverable`, and `retriable_after` — and **no** `message` field:
  ```json
  { "type": "error", "code": "session_timeout", "recoverable": false, "retriable_after": null, ... }
  ```
  (It carries the full `TranscriptionEvent` field set; other fields are `null`
  or defaults.)

  > **Non-leak (mirrors the REST 500 contract, §3.7).** For `error` events the
  > server **drops the `extra` payload** before sending (it is emptied to `{}`).
  > The streaming layer stores a human-readable message under `extra["detail"]`
  > — for the `engine_error` catch-all this is the raw `str(exc)`, which may
  > contain filesystem paths, upstream URLs, or credential fragments — so it is
  > never forwarded to the (unauthenticated) client. The safe structured fields
  > (`code`, `recoverable`, `retriable_after`, `segment_id`, and the
  > gap/reconnect fields) are preserved. The dropped detail is logged
  > server-side for operators.

### 4.3 Scope limit (v1)

The WebSocket surface supports **only** the incremental `audio_format` path
(declare format, push raw PCM frames, receive live events). The
**whole-input + streaming-output** path
(`start_transcription(audio=...)`, OpenAI SSE style, spec §7.3) is **NOT**
exposed over WebSocket in v1. For those engines, use the batch REST endpoints
(`POST /v1/transcribe` or `POST /v1/transcribe:json`).

### 4.4 Audio byte caps (DoS bound)

The HTTP body-size guard (§1) does not cover the WebSocket scope, so the stream
bridge enforces its own per-frame and per-session byte caps (§1, configurable
via `create_app`):

- A single binary audio frame exceeding `max_ws_frame_bytes`, **or**
- a cumulative session total exceeding `max_ws_session_bytes`,

is rejected: the input is ended, a single policy frame is sent, and the socket
closes (the violation is also logged server-side):

```json
{ "type": "error", "code": "payload_too_large", "message": "..." }
```

This is distinct from the §4.2 in-stream `error` event (an engine-produced
`TranscriptionEvent`); the policy frame is emitted by the **server**, not the
engine, and carries a human-readable `message` (the cap that was exceeded; it
contains no internal/engine detail).

A failure on the audio-input pump (e.g. a client protocol violation such as
sending audio after the session ended) is likewise never swallowed silently: it
is logged server-side and surfaced as a single generic, **non-leaking** frame
before teardown:

```json
{ "type": "error", "code": "stream_input_error", "message": "Audio input failed. See server logs for details." }
```
