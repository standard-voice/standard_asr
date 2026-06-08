# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""FastAPI server utilities for Standard ASR.

**Security note (operators MUST read).** These endpoints intentionally ship
**without authentication**: for v1 they are designed for localhost / trusted-LAN
use and for fronting by a reverse proxy. The capability and params-schema
endpoints are deliberately readable without auth (spec §3.1 / §C: declared
metadata is discoverable without instantiation or authentication). Before
exposing this server beyond localhost, operators **MUST** front it with
authentication and rate limiting -- there is no per-endpoint auth, no quota, and
transcription is CPU/GPU-expensive. A configurable request-body cap
(:data:`DEFAULT_MAX_BODY_BYTES`) guards against memory-exhaustion DoS, but it is
not a substitute for a rate limiter.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .audio_format import AudioFormat
from .audio_input import AudioBase64, AudioBytes, AudioInput
from .discovery import FactoryLoadError, ModelRegistry, discover_models
from .exceptions import (
    AudioProcessingError,
    ConfigError,
    EntrypointValidationError,
    InvalidProviderParamError,
    UnsupportedFeatureError,
)
from .results import TranscriptionResult
from .runtime_params import RuntimeParams
from .streaming import TranscriptionEvent, TranscriptionSession

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = logging.getLogger(__name__)

#: Default maximum accepted request-body size, in bytes (16 MiB). Enforced
#: *before* decoding to bound peak memory and prevent unauthenticated
#: memory-exhaustion DoS. Override per app via ``create_app(max_body_bytes=...)``.
DEFAULT_MAX_BODY_BYTES: int = 16 * 1024 * 1024

#: Default per-frame byte cap for the WebSocket audio path. Mirrors the HTTP
#: body cap: a single binary audio frame is treated like a request body and may
#: not exceed this size. Bounds peak memory for one frame and prevents an
#: unauthenticated client from exhausting memory with a few huge frames (the
#: HTTP body-size middleware does not cover the WS scope). Override per app via
#: ``create_app(max_ws_frame_bytes=...)``.
DEFAULT_MAX_WS_FRAME_BYTES: int = DEFAULT_MAX_BODY_BYTES

#: Default cumulative per-session audio byte cap for the WebSocket path. A
#: streaming session legitimately sends many small frames over time, so the
#: per-session ceiling is a larger multiple of the per-frame cap (256 MiB); it
#: bounds total ingested audio so a long-lived session cannot drive unbounded
#: memory/CPU even within the per-frame limit. Override per app via
#: ``create_app(max_ws_session_bytes=...)``.
DEFAULT_MAX_WS_SESSION_BYTES: int = 256 * 1024 * 1024


class _BodySizeLimitMiddleware:
    """Pure-ASGI middleware that rejects over-large request bodies (413).

    Implemented as raw ASGI rather than a ``BaseHTTPMiddleware`` so it inspects
    only the ``Content-Length`` header and never buffers or re-streams the body.
    A ``BaseHTTPMiddleware`` here would consume the request stream and break
    multipart ``request.form()`` parsing on starlette < 0.40 (the well-known
    BaseHTTPMiddleware body bug), which the lower-bounds CI lane caught.

    This is an early, cheap guard on the *declared* size. A chunked / streamed
    request with no ``Content-Length`` bypasses it, but **both** transcribe
    endpoints then enforce the limit on the materialised payload (the multipart
    ``len(file)`` and the JSON ``len(payload.audio)``), so oversize payloads are
    always rejected; the only residual exposure is that such a body is buffered
    before rejection.

    Args:
        app: The wrapped ASGI application.
        max_body_bytes: Maximum accepted body size in bytes.
    """

    def __init__(self, app: Any, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Reject the request with 413/400 on a bad/oversize Content-Length.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive callable.
            send: The ASGI send callable.
        """
        if scope.get("type") == "http":
            from fastapi.responses import JSONResponse

            for name, value in scope.get("headers", []):
                if name != b"content-length":
                    continue
                try:
                    declared = int(value)
                except ValueError:
                    await JSONResponse(
                        status_code=400,
                        content={"detail": "Invalid Content-Length header."},
                    )(scope, receive, send)
                    return
                if declared > self.max_body_bytes:
                    await JSONResponse(
                        status_code=413,
                        content={
                            "detail": (
                                f"Request body too large: {declared} bytes exceeds "
                                f"the {self.max_body_bytes}-byte limit."
                            )
                        },
                    )(scope, receive, send)
                    return
                break
        await self.app(scope, receive, send)


class ModelInfo(BaseModel):
    """Serializable model info for API responses.

    Args:
        key: Full model key in ``engine/model`` format.
        engine_id: Engine identifier.
        model_name: Model preset name.

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    # `model_name` is a deliberate API field; opt out of pydantic's `model_`
    # protected namespace so it does not warn (the warning fires on older
    # pydantic, e.g. the lower-bounds lane's 2.5).
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    key: str = Field(..., description="Model key in 'engine/model' format.")
    engine_id: str = Field(..., description="Engine identifier.")
    model_name: str = Field(..., description="Model preset name.")


class TranscribeJsonRequest(BaseModel):
    """JSON payload for transcription requests.

    Args:
        model: Model key in ``engine/model`` format.
        audio: Base64 data URI or raw base64 audio payload.
        options: Optional transcription options as JSON object.

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model key in 'engine/model' format.")
    audio: str = Field(..., description="Base64 data URI or raw base64-encoded audio payload.")
    options: dict[str, Any] | None = Field(
        default=None, description="Optional transcription options."
    )


class TranscribeResponse(BaseModel):
    """Standard transcription response.

    Args:
        model: Model key that handled the request.
        result: Standard ASR transcription result.

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model key that handled the request.")
    result: TranscriptionResult = Field(..., description="Standard ASR transcription result.")


def create_app(
    registry: ModelRegistry | None = None,
    *,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    max_ws_frame_bytes: int = DEFAULT_MAX_WS_FRAME_BYTES,
    max_ws_session_bytes: int = DEFAULT_MAX_WS_SESSION_BYTES,
):
    """Create a FastAPI application for Standard ASR.

    Args:
        registry: Optional pre-discovered registry.
        max_body_bytes: Maximum accepted request-body size in bytes. Requests
            exceeding this are rejected with ``413`` *before* the body is
            decoded, bounding peak memory (see :data:`DEFAULT_MAX_BODY_BYTES`).
        max_ws_frame_bytes: Maximum size of a single WebSocket binary audio
            frame in bytes. The HTTP body-size guard does not cover the WS
            scope, so the stream bridge enforces this per-frame cap directly;
            an over-cap frame closes the socket with a policy error (see
            :data:`DEFAULT_MAX_WS_FRAME_BYTES`).
        max_ws_session_bytes: Cumulative cap on total audio bytes ingested over
            one WebSocket session; exceeding it closes the socket with a policy
            error (see :data:`DEFAULT_MAX_WS_SESSION_BYTES`).

    Returns:
        FastAPI application instance.

    Raises:
        ImportError: If FastAPI dependencies are missing.
        ValueError: If any byte cap is not positive.
    """
    if max_body_bytes <= 0:
        raise ValueError("max_body_bytes must be a positive integer.")
    if max_ws_frame_bytes <= 0:
        raise ValueError("max_ws_frame_bytes must be a positive integer.")
    if max_ws_session_bytes <= 0:
        raise ValueError("max_ws_session_bytes must be a positive integer.")
    try:
        from fastapi import FastAPI, File, Form, HTTPException
        from fastapi import WebSocket as _WebSocket
    except ImportError as exc:
        raise ImportError(
            "FastAPI dependencies are missing. Install with: pip install 'standard-asr[server]'."
        ) from exc

    # Make the WebSocket type resolvable in this module's globals so FastAPI can
    # evaluate the stringified route annotation (future-annotations) while the
    # import itself stays lazy/optional.
    globals()["WebSocket"] = _WebSocket

    app = FastAPI(title="Standard ASR")
    model_registry = registry or discover_models()

    # Pure-ASGI body-size guard (see _BodySizeLimitMiddleware): rejects over-large
    # bodies via Content-Length before they are read, without buffering the body.
    app.add_middleware(_BodySizeLimitMiddleware, max_body_bytes=max_body_bytes)

    @app.get("/v1/health")
    def health() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        """Return basic service health.

        Args:
            None.

        Returns:
            Health status payload.

        Raises:
            None.
        """
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> list[ModelInfo]:  # pyright: ignore[reportUnusedFunction]
        """List discovered models.

        Args:
            None.

        Returns:
            List of model info objects.

        Raises:
            None.
        """
        infos: list[ModelInfo] = []
        for name in model_registry.names():
            spec = model_registry.spec(name)
            infos.append(ModelInfo(key=name, engine_id=spec.engine_id, model_name=spec.model_name))
        return infos

    @app.post("/v1/transcribe", response_model=TranscribeResponse)
    async def transcribe_file(  # pyright: ignore[reportUnusedFunction]
        model: str = Form(...),
        file: bytes = File(...),
        options: str | None = Form(None),
    ) -> TranscribeResponse:
        """Transcribe audio from a multipart file upload.

        Args:
            model: Model key in ``engine/model`` format.
            file: Uploaded audio payload.
            options: Optional JSON options string.

        Returns:
            Transcription response.

        Raises:
            HTTPException: If decoding or transcription fails.
        """
        if len(file) > max_body_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Uploaded file too large: {len(file)} bytes exceeds the "
                    f"{max_body_bytes}-byte limit."
                ),
            )
        try:
            params = _build_params(json.loads(options) if options else None)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Hand the encoded bytes to the engine's own negotiation rather than
        # pre-decoding here. The standard layer then converts/resamples per the
        # engine's accepted_input (so an encoded-only engine gets bytes, an
        # array engine gets an array at its accepted rate -- the upload's true
        # sample rate is never silently overridden).
        return await _run_transcription(
            model_registry, model, AudioBytes(data=file), params, HTTPException
        )

    @app.post("/v1/transcribe:json", response_model=TranscribeResponse)
    async def transcribe_json(  # pyright: ignore[reportUnusedFunction]
        payload: TranscribeJsonRequest,
    ) -> TranscribeResponse:
        """Transcribe audio from a JSON payload.

        Args:
            payload: JSON request payload.

        Returns:
            Transcription response.

        Raises:
            HTTPException: If decoding or transcription fails.
        """
        if len(payload.audio) > max_body_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Encoded audio too large: {len(payload.audio)} bytes exceeds the "
                    f"{max_body_bytes}-byte limit."
                ),
            )
        try:
            params = _build_params(payload.options)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Pass the base64/data-URI payload straight to engine negotiation, which
        # decodes and converts per the engine's accepted_input (see the
        # multipart endpoint). Decode failures surface as AudioProcessingError
        # and map to 400 in _run_transcription.
        return await _run_transcription(
            model_registry, payload.model, AudioBase64(payload.audio), params, HTTPException
        )

    @app.get("/v1/capabilities/{model:path}")
    def capabilities(model: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return an engine's declared capabilities as canonical JSON.

        Read from the engine **class** without instantiating it (spec §3.1 / §C:
        declared metadata is readable without instantiation or authentication).

        Args:
            model: Model key in ``engine/model`` format.

        Returns:
            The declared capability tree.

        Raises:
            HTTPException: If the model is unknown or has no capabilities.
        """
        engine_class = _engine_class_or_404(model_registry, model, HTTPException)
        caps = getattr(engine_class, "declared_capabilities", None)
        if caps is None:
            raise HTTPException(status_code=404, detail="No capabilities declared.")
        return caps.canonical_json()

    @app.get("/v1/params-schema/{model:path}")
    def params_schema(model: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return the JSON Schema for an engine's ``provider_params``.

        Read from the engine **class** without instantiating it (spec §3.1 / §C).
        Note that ``provider_params`` cannot currently be *sent* over the wire
        (the JSON/multipart transcribe endpoints accept only the portable
        standard set); this schema is published for discovery and UI generation.

        Args:
            model: Model key in ``engine/model`` format.

        Returns:
            The provider-params JSON Schema, or ``{}`` if the engine has none.

        Raises:
            HTTPException: If the model is unknown.
        """
        engine_class = _engine_class_or_404(model_registry, model, HTTPException)
        params_type = getattr(engine_class, "provider_params_type", None)
        if params_type is None:
            return {}
        return params_type.model_json_schema()

    @app.websocket("/v1/stream/{model:path}")
    async def stream(  # pyright: ignore[reportUnusedFunction]
        websocket: WebSocket, model: str
    ) -> None:
        """Bridge a WebSocket to an engine streaming session (mission G.2.2).

        Protocol: the client first sends a JSON text frame
        ``{"audio_format": {"encoding", "sample_rate", "channels"}, "options": {...}}``,
        then binary audio frames, then any text frame to signal end-of-audio (or
        simply disconnects). The server streams each
        :class:`~standard_asr.streaming.TranscriptionEvent` back as a JSON text
        frame. Errors before the bridge are reported as a single
        ``{"type": "error", "code", "message"}`` frame, then the socket closes.

        Args:
            websocket: The client WebSocket connection.
            model: Model key in ``engine/model`` format.
        """
        await websocket.accept()
        try:
            config = await websocket.receive_json()
            audio_format = AudioFormat(**config["audio_format"])
            params = _build_params(config.get("options"))
        except Exception as exc:  # noqa: BLE001
            await websocket.send_json({"type": "error", "code": "bad_request", "message": str(exc)})
            await websocket.close()
            return

        try:
            asr = await asyncio.to_thread(model_registry.create, model)
        except (EntrypointValidationError, FactoryLoadError) as exc:
            await websocket.send_json(
                {"type": "error", "code": "unknown_model", "message": str(exc)}
            )
            await websocket.close()
            return

        try:
            # start_transcription is part of the structural StandardASR protocol
            # (a batch-only engine raises UnsupportedFeatureError, handled below).
            session = asr.start_transcription(audio_format=audio_format, params=params)
        except (UnsupportedFeatureError, ValueError) as exc:
            await websocket.send_json({"type": "error", "code": "unsupported", "message": str(exc)})
            await websocket.close()
            return

        await _bridge_stream(
            websocket,
            session,
            max_frame_bytes=max_ws_frame_bytes,
            max_session_bytes=max_ws_session_bytes,
        )
        await websocket.close()

    return app


def _scrub_event_for_client(event: TranscriptionEvent) -> dict[str, Any]:
    """Serialize an event to JSON, stripping internal detail from errors.

    The streaming layer stores a human-readable message (which for the
    ``engine_error`` catch-all is ``str(exc)`` and may contain filesystem
    paths, upstream URLs, or credential fragments) under ``extra["detail"]`` of
    an ``error`` event. Forwarding it verbatim to an unauthenticated WebSocket
    client would contradict the REST 500 non-leak contract (server.md §3.7), so
    for ``error`` events the ``extra`` payload is dropped before it leaves the
    server. The safe structured fields (``code``, ``recoverable``,
    ``retriable_after``, ``segment_id``, and the gap/reconnect fields) are
    preserved; operators keep the dropped detail via the caller's logging.

    Non-error events are serialized unchanged.

    Args:
        event: The event produced by the session.

    Returns:
        The JSON-serializable payload to send to the client.
    """
    payload = event.model_dump(mode="json")
    if event.type == "error":
        # Drop any internal detail (e.g. extra["detail"]); keep only the safe
        # structured fields that the client protocol documents (server.md §4.2).
        payload["extra"] = {}
    return payload


async def _bridge_stream(
    websocket: WebSocket,
    session: TranscriptionSession,
    *,
    max_frame_bytes: int,
    max_session_bytes: int,
) -> None:
    """Pump client audio into ``session`` while streaming its events back.

    Reads binary frames as audio and any text frame (or a disconnect) as
    end-of-audio, feeding the session from a background task; concurrently
    forwards each produced event to the client as JSON. ``error`` events are
    scrubbed of internal detail before sending (see
    :func:`_scrub_event_for_client`); the raw detail is logged server-side for
    operators. A client that vanishes mid-stream simply ends the session (its
    remaining events are dropped).

    The WS audio path is byte-bounded (the HTTP body-size middleware does not
    cover the WS scope): a single frame exceeding ``max_frame_bytes`` or a
    cumulative session total exceeding ``max_session_bytes`` is rejected with a
    ``{"type": "error", "code": "payload_too_large"}`` policy frame, the input
    is ended, and the socket is closed (and the violation is logged). This
    bounds peak/total memory against an unauthenticated client feeding a few
    huge frames.

    Args:
        websocket: The accepted client WebSocket.
        session: The engine's :class:`~standard_asr.streaming.TranscriptionSession`.
        max_frame_bytes: Maximum size of a single binary audio frame in bytes.
        max_session_bytes: Cumulative cap on total ingested audio bytes.
    """
    # Out-of-band terminal frames the pump asks the forward loop to deliver. A
    # byte-cap ``violation`` (``payload_too_large``) or a swallowed pump
    # ``failure`` (``stream_input_error``) stops the loop forwarding engine
    # events and sends a single, non-leaking policy frame instead.
    violation: dict[str, str] = {}
    pump_failed = False

    async def _pump_audio() -> None:
        nonlocal pump_failed
        total = 0
        try:
            while True:
                message = await websocket.receive()
                chunk = message.get("bytes")
                if chunk is not None:
                    frame_len = len(chunk)
                    if frame_len > max_frame_bytes:
                        violation["message"] = (
                            f"Audio frame too large: {frame_len} bytes exceeds the "
                            f"{max_frame_bytes}-byte per-frame limit."
                        )
                        break
                    total += frame_len
                    if total > max_session_bytes:
                        violation["message"] = (
                            f"Session audio too large: {total} bytes exceeds the "
                            f"{max_session_bytes}-byte per-session limit."
                        )
                        break
                    await session.send_audio(chunk)
                else:
                    # A text frame signals end-of-audio; a disconnect message has
                    # neither bytes nor text. Either way, stop feeding.
                    break
            await session.end_audio()
        except Exception:
            # A client protocol violation (e.g. send_audio after the session
            # ended -> StreamClosedError) or any feed failure MUST NOT be
            # silently swallowed by the gather's return_exceptions (spec:
            # explicit > implicit / fail-loud). Log the full detail server-side
            # and flag the forward loop to emit a single generic, non-leaking
            # error frame. (CancelledError derives from BaseException on the
            # teardown path, so it is not caught here and propagates as required.)
            logger.exception("WebSocket audio pump failed")
            pump_failed = True
            # Best-effort end the input so the session drains and the forward
            # loop wakes to deliver the generic error frame (rather than blocking
            # on a session that will never produce a terminal event). end_audio
            # is idempotent and does not raise on the StreamClosedError path, so
            # no further guard is needed here.
            await session.end_audio()

    async with session:
        pump = asyncio.create_task(_pump_audio())
        try:
            async for event in session:
                if violation or pump_failed:
                    # A byte-cap violation / pump failure occurred: stop
                    # forwarding engine events; the policy frame is sent below.
                    break
                if event.type == "error":
                    # Keep the (potentially sensitive) detail server-side only;
                    # the client receives the scrubbed event below.
                    logger.error(
                        "Stream error event for client: code=%r detail=%r",
                        event.code,
                        event.extra.get("detail"),
                    )
                await websocket.send_json(_scrub_event_for_client(event))
            if violation:
                logger.warning("WebSocket audio cap exceeded: %s", violation["message"])
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "payload_too_large",
                        "message": violation["message"],
                    }
                )
            elif pump_failed:
                # Generic, non-leaking signal: the raw cause is already logged.
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "stream_input_error",
                        "message": "Audio input failed. See server logs for details.",
                    }
                )
        except Exception:  # noqa: BLE001
            # The client went away mid-stream; stop forwarding and tear down.
            pass
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)


async def _run_transcription(
    registry: ModelRegistry,
    model: str,
    audio: AudioInput,
    params: RuntimeParams | None,
    http_exception: type[Exception],
) -> TranscribeResponse:
    """Instantiate the engine, transcribe, and map errors to HTTP status codes.

    The audio is passed as an :data:`~standard_asr.audio_input.AudioInput` (not a
    pre-decoded array) so the engine's standard negotiation owns decoding and
    resampling. Client-caused errors map to 4xx; everything else to a generic
    500 (the raw exception text is logged server-side, never returned, to avoid
    leaking internal paths or upstream/credential material).

    Args:
        registry: The model registry.
        model: Model key in ``engine/model`` format.
        audio: The audio input to negotiate and transcribe.
        params: Parsed runtime parameters, or ``None``.
        http_exception: The ``HTTPException`` class to raise.

    Returns:
        The transcription response.

    Raises:
        Exception: ``http_exception`` with an appropriate status code.
    """
    try:
        asr = await asyncio.to_thread(registry.create, model)
    except (EntrypointValidationError, FactoryLoadError) as exc:
        raise http_exception(status_code=404, detail=str(exc)) from exc  # type: ignore[call-arg]

    try:
        result = await asyncio.to_thread(asr.transcribe, audio, params)
    except (
        InvalidProviderParamError,
        UnsupportedFeatureError,
        ConfigError,
        ValidationError,
    ) as exc:
        # Client-caused: bad params / unsupported standard feature / invalid config.
        raise http_exception(status_code=422, detail=str(exc)) from exc  # type: ignore[call-arg]
    except AudioProcessingError as exc:
        raise http_exception(status_code=400, detail=str(exc)) from exc  # type: ignore[call-arg]
    except Exception as exc:  # noqa: BLE001
        # Internal/unexpected: log details, return a stable generic message so we
        # never leak internal paths or upstream/credential text to the client.
        logger.exception("Transcription failed for model %r", model)
        detail = "Internal transcription error. See server logs for details."
        raise http_exception(status_code=500, detail=detail) from exc  # type: ignore[call-arg]

    return TranscribeResponse(model=model, result=result)


def _build_params(options: dict[str, Any] | None) -> RuntimeParams | None:
    """Build :class:`RuntimeParams` from a JSON options object.

    Only the portable standard set is supported over the wire (engine
    ``provider_params`` are not constructible without the engine type).

    Args:
        options: A JSON options object, or ``None``.

    Returns:
        Parsed runtime parameters, or ``None``.
    """
    if options is None:
        return None
    return RuntimeParams.model_validate(options)


def _engine_class_or_404(
    registry: ModelRegistry, model: str, http_exception: type[Exception]
) -> Any:
    """Resolve an engine class (without instantiation) or raise a 404.

    Args:
        registry: The model registry.
        model: Model key in ``engine/model`` format.
        http_exception: The ``HTTPException`` class to raise.

    Returns:
        The engine class.

    Raises:
        Exception: ``http_exception`` with status 404 if the model is unknown or
            its class cannot be resolved.
    """
    try:
        return registry.engine_class(model)
    except (EntrypointValidationError, FactoryLoadError) as exc:
        raise http_exception(status_code=404, detail=str(exc)) from exc  # type: ignore[call-arg]


def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info",
) -> None:
    """Run the FastAPI server using Uvicorn.

    Args:
        host: Bind host.
        port: Bind port.
        reload: Enable auto-reload.
        log_level: Uvicorn log level.

    Returns:
        None.

    Raises:
        ImportError: If Uvicorn is not installed.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "Uvicorn is required to run the server. Install with: "
            "pip install 'standard-asr[server]'."
        ) from exc

    app = create_app()
    uvicorn.run(app, host=host, port=port, reload=reload, log_level=log_level)


__all__ = [
    "ModelInfo",
    "TranscribeJsonRequest",
    "TranscribeResponse",
    "create_app",
    "run",
]
