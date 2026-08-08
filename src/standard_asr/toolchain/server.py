# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""FastAPI server utilities for Standard ASR.

**Security note (operators MUST read).** These endpoints intentionally ship
**without authentication**: for v1 they are designed for localhost / trusted-LAN
use and for fronting by a reverse proxy. The capability and params-schema
endpoints are deliberately readable without auth (declared metadata is
discoverable without instantiation or authentication). Before
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
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from standard_asr.audio.format import AudioFormat
from standard_asr.audio.input import AudioBase64, AudioBytes, AudioInput
from standard_asr.contract.exceptions import (
    AudioProcessingError,
    ConfigError,
    ConfigurationRequiredError,
    EntrypointValidationError,
    InvalidProviderParamError,
    UnsupportedFeatureError,
)
from standard_asr.contract.params import RuntimeParams, WireRuntimeParams
from standard_asr.contract.results import TranscriptionResult
from standard_asr.plugins.discovery import FactoryLoadError, ModelRegistry, discover_models
from standard_asr.runtime.protocol_boundary import require_sync_result
from standard_asr.runtime.redaction import (
    log_exception_safely,
    sanitize_validation_errors,
    sanitized_validation_message,
)
from standard_asr.runtime.streaming import (
    DIAGNOSTICS_TRUNCATED_CODE as _OVERFLOW_CODE,
)
from standard_asr.runtime.streaming import TranscriptionEvent, TranscriptionSession

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

#: Stable client-facing detail for a model whose engine requires server-side
#: configuration that is absent from this deployment
#: (``ConfigurationRequiredError`` -- at zero-arg construction, or discovered
#: lazily at transcription/session establishment). Deliberately
#: generic: the absent FIELD NAMES are deployment detail (safe-logged for the
#: operator), never sent to an unauthenticated caller. Shared verbatim by the
#: REST 503 body and the WS ``service_unavailable`` frame.
_ENGINE_CONFIG_ABSENT_DETAIL: str = (
    "The engine for this model requires server-side configuration that is not "
    "present on this deployment (for example a credential environment "
    "variable). This is an operator-side state, not a request error."
)


def _internal_error_message(stage: str) -> str:
    """Build the scrubbed internal-error message for a failed server stage.

    The message names the stage and points the caller to the server logs. It
    carries no internal detail, so it is safe to send to an unauthenticated
    client. Every stage shares one format, so the client-facing wording cannot
    drift between the REST and WebSocket paths.

    Args:
        stage: The stage that failed (for example ``"model construction"``).

    Returns:
        A client-safe message of the form ``"Internal <stage> error. See
        server logs for details."``.
    """
    return f"Internal {stage} error. See server logs for details."


# The credential-scrubbing of pydantic validation errors is shared with the CLI
# (and any other transport that surfaces an `options` validation error) so the
# two cannot drift on the "never echo the request input" rule. The single owner
# is :mod:`standard_asr.runtime.redaction`; these aliases preserve the historical
# `server._sanitize_validation_errors` / `server._sanitized_validation_message`
# names used by call sites and tests.
_sanitize_validation_errors = sanitize_validation_errors
_sanitized_validation_message = sanitized_validation_message


def _sanitized_validation_detail(
    exc: ValidationError, *, loc_prefix: Sequence[object]
) -> list[dict[str, Any]]:
    """Build the structured, input-free 422 ``detail`` for a pydantic error.

    Every REST ``422`` -- the global ``RequestValidationError`` handler *and*
    the standalone-``ValidationError`` path for the client's own request
    material (the ``options`` build) -- returns
    the **same** machine-readable shape: a list of ``{type, loc, msg}`` entries
    (the input is never echoed back). Keeping one body shape per status code means a
    cross-language client parses a single structure (and can branch on ``type``,
    e.g. ``extra_forbidden`` for a rejected ``provider_params`` key) rather than
    discriminating string-vs-list per code. The ``loc_prefix`` anchors a
    standalone error's model-relative ``loc`` under the request field it came
    from (e.g. ``["options"]`` / ``["config"]``), replacing the prose label the
    old flat-string form used. Never echoes the submitted value and redacts
    credential-like fields (mirrors the shared ``sanitize_validation_errors``).

    Args:
        exc: The pydantic validation error.
        loc_prefix: Path components naming where the error originated (e.g.
            ``["options"]`` for the wire-options build, ``["config"]`` for
            engine construction).

    Returns:
        A list of sanitized ``{type, loc, msg}`` entries safe to return.
    """
    return _sanitize_validation_errors(exc.errors(), loc_prefix=loc_prefix)


class _BodySizeLimitMiddleware:
    """Pure-ASGI middleware that rejects over-large request bodies (413).

    Implemented as raw ASGI rather than a ``BaseHTTPMiddleware`` so it never has
    to buffer the whole body itself: a ``BaseHTTPMiddleware`` here would consume
    the request stream and break multipart ``request.form()`` parsing on
    starlette < 0.40 (the well-known BaseHTTPMiddleware body bug), which the
    lower-bounds CI lane caught.

    Enforcement is two-layered:

    1. **Declared size (cheap, early).** A bad ``Content-Length`` header → 400;
       a ``Content-Length`` over the cap → 413, before any body is read.
    2. **Actual size (true cap).** ``Content-Length`` is advisory: a chunked /
       streamed request may omit it or under-state it, slipping past layer 1 and
       being parsed by FastAPI / pydantic first. So the ``receive`` channel is
       wrapped to count body bytes as they arrive and abort with 413 the moment
       the cumulative total exceeds the cap -- bounding peak memory regardless of
       the declared length. (The WS scope has its own per-frame / per-session
       caps and is passed straight through.)

    Args:
        app: The wrapped ASGI application.
        max_body_bytes: Maximum accepted body size in bytes.
    """

    def __init__(self, app: Any, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Reject the request with 413/400 on an oversize declared or actual body.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive callable.
            send: The ASGI send callable.
        """
        if scope.get("type") != "http":
            # WebSocket / lifespan: no HTTP body to bound here (the WS surface
            # enforces its own per-frame / per-session caps).
            await self.app(scope, receive, send)
            return

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

        # Enforce the *actual* cap by counting bytes off the receive channel.
        # ``Content-Length`` is advisory; this catches a chunked / under-stated
        # body before it is fully buffered/parsed downstream.
        state = {"received": 0, "rejected": False}

        async def receive_capped() -> Any:
            message = await receive()
            if message.get("type") != "http.request":
                return message
            state["received"] += len(message.get("body", b""))
            if state["received"] <= self.max_body_bytes:
                return message
            # Cap breached. Emit the 413 directly (once), then hand the app a
            # disconnect so its body read unwinds promptly. ``send_capped`` drops
            # the app's subsequent (now-moot) response so it cannot clobber ours.
            if not state["rejected"]:
                state["rejected"] = True
                await JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            f"Request body too large: exceeds the {self.max_body_bytes}-byte limit."
                        )
                    },
                )(scope, receive, send)
            return {"type": "http.disconnect"}

        async def send_capped(message: Any) -> None:
            # Suppress the app's response once we've committed our own 413 (its
            # body read raised on the injected disconnect).
            if state["rejected"]:
                return
            await send(message)

        await self.app(scope, receive_capped, send_capped)


class ModelInfo(BaseModel):
    """Serializable model info for API responses.

    Attributes:
        key: Full model key in ``engine/model`` format.
        engine_id: Engine identifier.
        model_name: Model preset name.

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

    Attributes:
        model: Model key in ``engine/model`` format.
        audio: Base64 data URI or raw base64 audio payload.
        options: Optional transcription options as JSON object.

    Raises:
        ValueError: If validation fails.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model key in 'engine/model' format.")
    audio: str = Field(..., description="Base64 data URI or raw base64-encoded audio payload.")
    options: dict[str, Any] | None = Field(
        default=None, description="Optional transcription options."
    )


class StreamConfigRequest(BaseModel):
    """The WebSocket config/handshake frame -- a CLOSED request model.

    ``extra="forbid"`` is the load-bearing part: the old ad-hoc parse
    (``config["audio_format"]`` + ``config.get("options")``) silently ignored
    unknown top-level keys, so a client typo (``"optinos"``) started the
    session with defaults while the client believed its options were applied
    -- a silent wrong result, the exact failure mode every other request
    model on the wire surface already forbids.

    Attributes:
        audio_format: The negotiated raw-audio wire format for the session.
        options: Optional transcription options as a JSON object (validated
            against the portable ``WireRuntimeParams`` set by the caller).

    Raises:
        ValueError: If validation fails (unknown keys included).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    audio_format: AudioFormat = Field(
        ..., description="Raw-audio wire format for the session's binary frames."
    )
    options: dict[str, Any] | None = Field(
        default=None, description="Optional transcription options."
    )


class TranscribeResponse(BaseModel):
    """Standard transcription response.

    Attributes:
        model: Model key that handled the request.
        result: Standard ASR transcription result.

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
        registry: Pre-discovered registry to expose. When ``None`` (the
            default), plugins are auto-discovered via ``discover_models()``. An
            explicitly-passed registry is used as-is **even when empty** (an
            empty ``ModelRegistry({})`` exposes zero models; it does *not* fall
            back to discovery).
        max_body_bytes: Maximum accepted request-body size in bytes. Requests
            exceeding this are rejected with ``413`` *before* the body is
            decoded, bounding peak memory (see :data:`DEFAULT_MAX_BODY_BYTES`).
        max_ws_frame_bytes: Maximum size of a single WebSocket frame in bytes
            (audio frames *and* the config/handshake frame). The HTTP body-size
            guard does not cover the WS scope, so the stream bridge enforces this
            per-frame cap directly; an over-cap frame closes the socket with a
            policy error (see :data:`DEFAULT_MAX_WS_FRAME_BYTES`). The transport
            also imposes its own ``ws_max_size`` (uvicorn's default is 16 MiB),
            so the effective bound is ``min(max_ws_frame_bytes, transport
            ws_max_size)``; :func:`run` passes ``ws_max_size=max_ws_frame_bytes``
            so the two match.
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
        from fastapi import FastAPI, File, Form, HTTPException, Request
        from fastapi import WebSocket as _WebSocket
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse, Response
    except ImportError as exc:
        raise ImportError(
            "FastAPI dependencies are missing. Install with: pip install 'standard-asr[server]'."
        ) from exc

    # Make the WebSocket type resolvable in this module's globals so FastAPI can
    # evaluate the stringified route annotation (future-annotations) while the
    # import itself stays lazy/optional.
    globals()["WebSocket"] = _WebSocket

    app = FastAPI(title="Standard ASR")
    # Use the caller's registry when one is given -- even an empty one. A bare
    # ``registry or discover_models()`` would treat an explicitly-passed empty
    # ``ModelRegistry({})`` as falsey (it is len 0) and silently fall back to
    # full plugin discovery, so an operator who wants to expose ZERO models would
    # instead expose every installed plugin. ``is not None`` honors the intent.
    model_registry = registry if registry is not None else discover_models()

    # Pure-ASGI body-size guard (see _BodySizeLimitMiddleware): rejects over-large
    # bodies via Content-Length before they are read, without buffering the body.
    app.add_middleware(_BodySizeLimitMiddleware, max_body_bytes=max_body_bytes)

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(  # pyright: ignore[reportUnusedFunction]
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return a 422 that never echoes the offending request ``input``.

        FastAPI's default handler reflects each error's ``input`` value verbatim.
        A caller who mis-places a secret (e.g. an ``api_key`` in the JSON body or
        ``options``) would have it bounced back into the client / any proxy / a
        copied bug report. We strip the ``input`` echo (and the ``url``) and
        redact credential-like fields (see :func:`_sanitize_validation_errors`),
        preserving the safe structured fields so the caller can still fix the
        request.

        Args:
            _request: The incoming request (unused).
            exc: The raised request-validation error.

        Returns:
            A ``422`` JSON response with sanitized error detail.
        """
        return JSONResponse(
            status_code=422,
            content={"detail": _sanitize_validation_errors(exc.errors())},
        )

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
    ) -> Response:
        """Transcribe audio from a multipart file upload.

        Args:
            model: Model key in ``engine/model`` format.
            file: Uploaded audio payload.
            options: Optional JSON options string.

        Returns:
            The transcription response, pre-encoded by
            :func:`_run_transcription` (the single wire projection;
            ``response_model`` stays for the OpenAPI schema).

        Raises:
            HTTPException: If decoding or transcription fails.
        """
        # The request-body cap is enforced at the ASGI boundary by
        # _BodySizeLimitMiddleware (Content-Length *and* actual bytes), so the
        # uploaded ``file`` is already bounded by the time it materialises here.
        try:
            parsed_options = json.loads(options) if options else None
        except Exception as exc:  # noqa: BLE001
            # Malformed options *syntax* (un-parseable JSON) is a bad request.
            raise HTTPException(status_code=400, detail=f"Invalid options JSON: {exc}") from exc
        try:
            params = _build_params(parsed_options)
        except ValidationError as exc:
            # A semantically invalid options object (bad value, unknown key, or a
            # non-portable provider_params key) is an unprocessable entity.
            # Return the structured, sanitized detail (same shape as the global
            # RequestValidationError handler, anchored under ["options"]):
            # pydantic's raw detail echoes the offending input value, so a
            # mis-placed secret would otherwise be reflected.
            raise HTTPException(
                status_code=422,
                detail=_sanitized_validation_detail(exc, loc_prefix=["options"]),
            ) from exc

        # Hand the encoded bytes to the engine's own negotiation rather than
        # pre-decoding here. The standard layer then converts/resamples per the
        # engine's accepted_input (so an encoded-only engine gets bytes, an
        # array engine gets an array at its accepted rate -- the upload's true
        # sample rate is never silently overridden).
        body = await _run_transcription(
            model_registry, model, AudioBytes(data=file), params, HTTPException
        )
        return Response(content=body, media_type="application/json")

    @app.post("/v1/transcribe:json", response_model=TranscribeResponse)
    async def transcribe_json(  # pyright: ignore[reportUnusedFunction]
        payload: TranscribeJsonRequest,
    ) -> Response:
        """Transcribe audio from a JSON payload.

        Args:
            payload: JSON request payload.

        Returns:
            The transcription response, pre-encoded by
            :func:`_run_transcription` (the single wire projection;
            ``response_model`` stays for the OpenAPI schema).

        Raises:
            HTTPException: If decoding or transcription fails.
        """
        # The request-body cap is enforced at the ASGI boundary by
        # _BodySizeLimitMiddleware (Content-Length *and* actual bytes), so the
        # encoded ``audio`` is already bounded by the time it materialises here.
        try:
            # `payload.options` is already a parsed object, so the only failure
            # here is params validation (bad value, unknown key, or a non-portable
            # provider_params key) -> 422. pydantic's raw detail echoes the
            # offending input value (a mis-placed secret would be reflected), so
            # return the structured, sanitized detail instead.
            params = _build_params(payload.options)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=_sanitized_validation_detail(exc, loc_prefix=["options"]),
            ) from exc

        # Pass the base64/data-URI payload straight to engine negotiation, which
        # decodes and converts per the engine's accepted_input (see the
        # multipart endpoint). Decode failures surface as AudioProcessingError
        # and map to 400 in _run_transcription.
        body = await _run_transcription(
            model_registry, payload.model, AudioBase64(payload.audio), params, HTTPException
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/capabilities/{model:path}")
    def capabilities(model: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return an engine's declared capabilities as canonical JSON.

        Read from the engine **class** without instantiating it (declared
        metadata is readable without instantiation or authentication).

        Args:
            model: Model key in ``engine/model`` format.

        Returns:
            The declared capability tree.

        Raises:
            HTTPException: 404 if the model key is unknown or the engine
                declares no capabilities; scrubbed 500 if the registered
                model's plugin fails to load, its capability descriptor
                raises, or its canonical JSON is unencodable (see
                :func:`_metadata_or_http_error`).
        """

        def _project(engine_class: Any) -> dict[str, Any]:
            caps = getattr(engine_class, "declared_capabilities", None)
            if caps is None:
                raise HTTPException(status_code=404, detail="No capabilities declared.")
            return caps.canonical_json()

        return _metadata_or_http_error(model_registry, model, HTTPException, project=_project)

    @app.get("/v1/params-schema/{model:path}")
    def params_schema(model: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return the JSON Schema for an engine's ``provider_params``.

        Read from the engine **class** without instantiating it.
        Note that ``provider_params`` cannot currently be *sent* over the wire
        (the JSON/multipart transcribe endpoints accept only the portable
        standard set); this schema is published for discovery and UI generation.

        Args:
            model: Model key in ``engine/model`` format.

        Returns:
            The provider-params JSON Schema, or ``{}`` if the engine has none.

        Raises:
            HTTPException: 404 if the model key is unknown; scrubbed 500
                if the registered model's plugin fails to load, its
                descriptor raises, or its schema is unencodable (see
                :func:`_metadata_or_http_error`).
        """

        def _project(engine_class: Any) -> dict[str, Any]:
            params_type = getattr(engine_class, "provider_params_type", None)
            if params_type is None:
                return {}
            return params_type.model_json_schema()

        return _metadata_or_http_error(model_registry, model, HTTPException, project=_project)

    @app.get("/v1/config-schema/{model:path}")
    def config_schema(model: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return the JSON Schema for an engine's init config (``config_type``).

        Read from the engine **class** without instantiating it. This is the
        wire-side discovery path for settings UIs: a
        non-Python client can render the engine's configuration form (secret
        fields carry ``format: password`` / ``writeOnly`` markers) before the
        engine is ever constructed. The schema describes field *shapes* only --
        it never contains configured values, so it is safe to expose without
        authentication alongside the capabilities endpoint.

        Args:
            model: Model key in ``engine/model`` format.

        Returns:
            The init-config JSON Schema, or ``{}`` if the engine does not
            declare a class-level ``config_type``.

        Raises:
            HTTPException: 404 if the model key is unknown; scrubbed 500
                if the registered model's plugin fails to load, its
                descriptor raises, or its schema is unencodable (see
                :func:`_metadata_or_http_error`).
        """

        def _project(engine_class: Any) -> dict[str, Any]:
            config_type = getattr(engine_class, "config_type", None)
            if config_type is None:
                return {}
            return config_type.model_json_schema()

        return _metadata_or_http_error(model_registry, model, HTTPException, project=_project)

    @app.websocket("/v1/stream/{model:path}")
    async def stream(  # pyright: ignore[reportUnusedFunction]
        websocket: WebSocket, model: str
    ) -> None:
        """Bridge a WebSocket to an engine streaming session.

        Protocol: the client first sends a JSON text frame
        ``{"audio_format": {"encoding", "sample_rate", "channels"}, "options": {...}}``,
        then binary audio frames, then any text frame to signal end-of-audio (or
        simply disconnects). The server streams each
        :class:`~standard_asr.runtime.streaming.TranscriptionEvent` back as a JSON text
        frame. Errors before the bridge are reported as a single
        ``{"type": "error", "code", "message"}`` frame, then the socket closes.

        Args:
            websocket: The client WebSocket connection.
            model: Model key in ``engine/model`` format.
        """
        await websocket.accept()
        try:
            raw_config = await _receive_config_frame(websocket, max_ws_frame_bytes)
            request = StreamConfigRequest.model_validate(raw_config)
            audio_format = request.audio_format
            params = _build_params(request.options)
        except _ConfigFrameTooLarge as exc:
            # The config/handshake frame is bounded by the app cap too (not just
            # the transport ws_max_size), so the documented DoS bound holds
            # regardless of the ASGI server in front. Reported like the audio
            # caps.
            await websocket.send_json(
                {"type": "error", "code": "payload_too_large", "message": str(exc)}
            )
            await websocket.close()
            return
        except _ConfigFrameNotText as exc:
            # A malformed handshake (binary first frame): the caller's mistake,
            # reported with the standard-authored message.
            await websocket.send_json({"type": "error", "code": "bad_request", "message": str(exc)})
            await websocket.close()
            return
        except json.JSONDecodeError as exc:
            # Unparseable JSON text: caller-fixable. JSONDecodeError's str()
            # is positional ("Expecting value: line 1 column 2"), never the
            # document text, so it is safe and actionable to send.
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "bad_request",
                    "message": f"Config frame is not valid JSON: {exc}.",
                }
            )
            await websocket.close()
            return
        except ValidationError as exc:
            # The CLOSED StreamConfigRequest model rejects unknown top-level
            # keys (a client typo like "optinos" must fail loudly, never start
            # the session on silent defaults), a non-object frame, a bad
            # audio_format, and invalid options -- all caller-fixable.
            # Sanitize: pydantic's str(exc) echoes the offending input value,
            # so a mis-placed secret in options would be reflected back.
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "bad_request",
                    "message": _sanitized_validation_message(exc),
                }
            )
            await websocket.close()
            return
        except Exception:  # noqa: BLE001
            # Anything else here is OUR fault (the receive machinery, an
            # internal bug), not the caller's: the old catch-all mapped it to
            # `bad_request` AND sent raw str(exc) -- misattributing the fault
            # and leaking internal text to an unauthenticated client. Scrubbed
            # internal_error, specifics safe-logged.
            log_exception_safely(logger, "WS handshake failed internally for model %r", model)
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": _internal_error_message("handshake"),
                }
            )
            await websocket.close()
            return

        try:
            asr = await asyncio.to_thread(model_registry.create, model)
        except EntrypointValidationError as exc:
            # The caller's model key does not exist or cannot be parsed --
            # genuinely caller-fixable; the authored message names only the
            # caller's own key and the available keys.
            await websocket.send_json(
                {"type": "error", "code": "unknown_model", "message": str(exc)}
            )
            await websocket.close()
            return
        except FactoryLoadError:
            # The key RESOLVED; a server-installed plugin failed to
            # import/resolve/validate. That is an engine/deployment fault the
            # caller can neither see nor fix -- calling it "unknown model"
            # misattributed it, and the message carries plugin-internal
            # import/annotation text that must not cross the trust boundary.
            # Scrubbed internal_error, specifics safe-logged (§3.7 twin).
            log_exception_safely(logger, "Registered model %r failed to load for streaming", model)
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": _internal_error_message("model construction"),
                }
            )
            await websocket.close()
            return
        except ConfigurationRequiredError:
            # MUST precede any broader arm (subclasses ConfigError). Zero-arg
            # construction means every config input is the SERVER's: required
            # config absent from this deployment is the operator's to fix --
            # the WS twin of the REST 503, with the same stable generic detail
            # (the absent field names are deployment detail, safe-logged only).
            log_exception_safely(
                logger,
                "Engine %r requires configuration absent from the server environment",
                model,
            )
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "service_unavailable",
                    "message": _ENGINE_CONFIG_ABSENT_DETAIL,
                }
            )
            await websocket.close()
            return
        except Exception:  # noqa: BLE001
            # Internal/unexpected construction fault (incl. ConfigError /
            # InvalidProviderParamError / ValidationError from the zero-arg
            # factory -- a deployment or plugin defect, never the caller's;
            # the old bad_request arm blamed the caller for faults it cannot
            # see or fix): never crash the route or leak detail. Log
            # server-side; send a single generic, non-leaking frame (mirrors
            # the REST scrubbed-500 contract).
            log_exception_safely(logger, "Engine construction failed for streaming model %r", model)
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": _internal_error_message("model construction"),
                }
            )
            await websocket.close()
            return

        try:
            # start_transcription is part of the structural StandardASR protocol
            # (a batch-only engine raises UnsupportedFeatureError, handled below).
            session = asr.start_transcription(audio_format=audio_format, params=params)
            # Sync-call boundary: an `async def` start_transcription (or a sync
            # wrapper delegating to one) hands back a coroutine that the
            # consumers below (`session.diagnostics()`, the bridge) would hit
            # OUTSIDE this try as a secondary AttributeError while the
            # coroutine leaked never-awaited. EngineContractError is not a
            # ValueError, so it lands on the generic engine-fault arm below
            # (scrubbed internal_error frame).
            require_sync_result(
                session, "start_transcription()", expected_type=TranscriptionSession
            )
        except ConfigurationRequiredError:
            # Required config absent, discovered lazily at establishment (an
            # engine deferring its credential check past construction): the WS
            # twin of the REST 503 -- operator-side, stable generic message,
            # never the absent field names. MUST precede the ConfigError arm
            # (subclass), which MUST precede the ValueError arm below.
            log_exception_safely(
                logger,
                "Engine %r requires configuration absent from the server environment",
                model,
            )
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "service_unavailable",
                    "message": _ENGINE_CONFIG_ABSENT_DETAIL,
                }
            )
            await websocket.close()
            return
        except (ConfigError, InvalidProviderParamError):
            # An ENGINE fault, not a request error: the WS surface gives the
            # client no way to cause either -- engine init config never
            # crosses the wire, `provider_params` is rejected by
            # WireRuntimeParams in the config frame, and client-fixable
            # rejections have their own types (UnsupportedFeatureError ->
            # `unsupported`; frame/options ValidationError -> `bad_request`
            # at parse time). A ConfigError here is an engine
            # declaration/config defect (e.g. a missing `default_language`)
            # whose authored message may carry server-side config detail --
            # the old `bad_request` frame blamed the caller AND surfaced that
            # text. Scrubbed internal_error, specifics safe-logged. NOTE:
            # both subclass ValueError -- callers must not rely on a broader
            # ValueError arm existing (there deliberately is none).
            log_exception_safely(
                logger,
                "Engine-side configuration/contract fault during establishment for %r",
                model,
            )
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": _internal_error_message("stream establishment"),
                }
            )
            await websocket.close()
            return
        except ValidationError:
            # By session establishment the client's params are already
            # validated -- a bare ValidationError here is an ENGINE fault (a
            # structural engine's internals; EngineBase wraps this seam as
            # TranscriptionError, but the server cannot assume the base
            # class). Labelling it "unsupported" misattributes the fault, and
            # ``str(exc)`` echoes pydantic's offending input_value -- a
            # mis-placed engine-side secret would cross the trust boundary.
            # Scrubbed internal_error, logged server-side. A dedicated arm
            # (not the generic one below) so the log names the seam.
            log_exception_safely(
                logger, "Engine-side validation failure during establishment for %r", model
            )
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": _internal_error_message("stream establishment"),
                }
            )
            await websocket.close()
            return
        except UnsupportedFeatureError as exc:
            # The ONLY caller-fixable establishment rejection: the engine
            # cannot serve this (validated) request's feature set. A bare
            # ValueError deliberately does NOT belong here: by establishment
            # every client input is already validated (config frame ->
            # pydantic; language/candidates/unknown keys -> WireRuntimeParams;
            # audio_format -> its model), and a compliant engine signals
            # unsupported features with UnsupportedFeatureError -- so a
            # surviving ValueError is an engine/adapter/SDK fault. Mapping it
            # to `unsupported` blamed the caller AND sent str(exc) -- an
            # engine-internal message, possibly credential-bearing -- to an
            # unauthenticated client; it now falls to the scrubbed
            # internal_error arm below (REST's fault-ownership twin, §3.7).
            await websocket.send_json({"type": "error", "code": "unsupported", "message": str(exc)})
            await websocket.close()
            return
        except Exception:  # noqa: BLE001
            # Internal/unexpected session-establishment fault (e.g. a fault in the
            # engine's own _start_transcription hook, or a bare ValueError from
            # engine internals): never crash the route or leak detail. Log
            # server-side; send a single generic, non-leaking frame (mirrors
            # the construction scrubbed-frame contract).
            log_exception_safely(logger, "Stream session establishment failed for model %r", model)
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": _internal_error_message("stream establishment"),
                }
            )
            await websocket.close()
            return

        # Forward the standard-layer diagnostics (best-effort parameter degrade,
        # language resolution, audio conversion) attached at session
        # establishment, so a WS client can see WHY word_timestamps / prompt /
        # language were dropped or changed -- the REST path returns these on the
        # result, and the WS surface must not silently hide them.
        try:
            diagnostics_frame = _initial_diagnostics_frame(session)
        except Exception:  # noqa: BLE001 - projecting is the server's job, not the route's death
            # This projection ran OUTSIDE every boundary: the establishment
            # try/except has already returned by here, and `_bridge_stream`'s
            # forward-loop catch has not started. A diagnostic carrying a
            # value with no JSON form (only reachable past the JsonValue
            # declaration -- `model_construct`, or mutation after
            # construction) therefore killed the route with an unhandled
            # exception, bypassing the operator-log redaction on the way out.
            # Failing loudly is deliberate: diagnostics are the channel that
            # tells a client something degraded, so dropping the frame and
            # streaming on would hide exactly what the client needs.
            log_exception_safely(logger, "Stream diagnostics projection failed for model %r", model)
            # No session teardown on this return, deliberately: the session is
            # CONSTRUCTED but never entered (start_transcription does not open;
            # the producer/feed tasks and the adapter's _open run inside
            # _bridge_stream's `async with`), so the standard layer holds no
            # live resources here, and __aexit__ would call the adapter's
            # _close without a matching _open -- an unspecified state. Same
            # stance as the compliance gating probe's constructed-not-entered
            # abandonment.
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": _internal_error_message("stream diagnostics"),
                }
            )
            await websocket.close()
            return
        if diagnostics_frame is not None:
            await websocket.send_json(diagnostics_frame)

        await _bridge_stream(
            websocket,
            session,
            max_frame_bytes=max_ws_frame_bytes,
            max_session_bytes=max_ws_session_bytes,
        )
        await websocket.close()

    return app


class _ConfigFrameTooLarge(Exception):
    """The WebSocket config/handshake frame exceeded the app per-frame cap."""


class _ConfigFrameNotText(Exception):
    """The first WebSocket frame was not a JSON text frame.

    Binary frames are reserved for raw audio; a binary first frame is a
    malformed handshake and is surfaced to the client as ``bad_request``.
    """


async def _receive_config_frame(websocket: WebSocket, max_frame_bytes: int) -> Any:
    """Receive and parse the WS config frame, bounded by the app per-frame cap.

    The audio frames are byte-bounded by :func:`_bridge_stream`, but the very
    first config/handshake frame is read before the bridge and would otherwise be
    covered **only** by the transport's ``ws_max_size`` (uvicorn's default is 16
    MiB) -- so a smaller app cap (``max_ws_frame_bytes``) would not actually bound
    it, and the documented vs. enforced DoS bound could diverge. Reading the raw
    frame and checking its length against the app cap *before* parsing closes that
    gap independently of the ASGI server in front (the effective bound is
    ``min(app cap, transport ws_max_size)``).

    Args:
        websocket: The accepted client WebSocket.
        max_frame_bytes: The app per-frame byte cap.

    Returns:
        The parsed JSON value, unvalidated -- the caller validates it against
        the CLOSED :class:`StreamConfigRequest` model (which also rejects a
        non-object frame).

    Raises:
        _ConfigFrameTooLarge: If the raw config frame exceeds ``max_frame_bytes``.
        _ConfigFrameNotText: If the first frame is not a text frame (e.g. a
            binary frame); surfaced as ``bad_request`` by the caller.
        json.JSONDecodeError: If the text frame is not parseable JSON
            (surfaced as ``bad_request`` by the caller -- its message is
            positional, never the document text).
    """
    message = await websocket.receive()
    raw = message.get("text")
    # The wire contract requires the config/handshake frame to
    # be a JSON **text** frame; **binary** frames are reserved for raw audio.
    # Accepting a binary first frame as config would make the two frame classes
    # distinguishable only by arrival order, not by WebSocket frame type, and
    # would bake an undefined leniency into the reference implementation that any
    # strict third-party server (which treats the first binary frame as audio)
    # would not share -- a cross-implementation compatibility hazard for the
    # versioned wire protocol. Reject a non-text first frame explicitly.
    if not isinstance(raw, str):
        raise _ConfigFrameNotText(
            "Config frame must be a JSON text frame (binary frames are reserved for audio)."
        )
    payload: bytes = raw.encode()
    if len(payload) > max_frame_bytes:
        raise _ConfigFrameTooLarge(
            f"Config frame too large: {len(payload)} bytes exceeds the "
            f"{max_frame_bytes}-byte per-frame limit."
        )
    return json.loads(payload)


#: What the WS bridge has already delivered on the diagnostics channel:
#: how many entries, and the rendered text of the overflow summary among
#: them (``None`` when the channel has not overflowed). The summary is
#: rewritten IN PLACE by the guard, so a count alone cannot detect a change
#: -- see :func:`_diagnostics_delta_frame`.
_DiagnosticsCursor = tuple[int, str | None]


def _initial_diagnostics_frame(session: TranscriptionSession) -> dict[str, Any] | None:
    """Build the standard-layer diagnostics frame for a freshly-started session.

    The base ``start_transcription`` template attaches the parameter-gating and
    language-axis diagnostics (best-effort degrade, language resolution, audio
    conversion) to the session before handing it back, so they are available via
    :meth:`~standard_asr.runtime.streaming.TranscriptionSession.diagnostics` immediately.
    The REST path returns these on the result; the WS surface forwards them as a
    single ``diagnostics`` frame up front so the client learns WHY a parameter
    was dropped or changed before audio flows.

    Unlike the ``engine_error`` detail (pre-summarized exception text,
    dropped by :func:`_scrub_event_for_client` before it leaves the server),
    these messages are standard-layer-authored (not exception text), so they
    are forwarded verbatim -- exactly as REST returns them.

    Args:
        session: The just-established streaming session.

    Returns:
        A ``{"type": "diagnostics", "diagnostics": [...]}`` frame, or ``None``
        when the session exposes no diagnostics.
    """
    diagnostics = session.diagnostics()
    if not diagnostics:
        return None
    return {
        "type": "diagnostics",
        "diagnostics": [diag.model_dump(mode="json") for diag in diagnostics],
    }


def _diagnostics_delta_frame(
    session: TranscriptionSession, already_sent: _DiagnosticsCursor
) -> tuple[dict[str, Any] | None, _DiagnosticsCursor]:
    """Build a ``diagnostics`` frame for entries accrued since ``already_sent``.

    The session's bounded diagnostics channel grows mid-stream -- an engine's
    :meth:`~standard_asr.runtime.streaming.TranscriptionSession.emit_diagnostic` call, or a
    guard suppression -- but :func:`_initial_diagnostics_frame` is sent only once,
    at establishment. This returns the NEW diagnostics (those past ``already_sent``)
    as a frame plus the updated cursor, so the bridge can forward them as they
    appear. Without it a WS client never sees a diagnostic emitted after
    establishment, while the REST path returns all of them on the result -- a
    two-layer drift the WS surface must not introduce.

    LENGTH IS NOT ENOUGH. Once the channel hits its cap the guard stops
    appending and instead REWRITES its trailing ``diagnostics_truncated``
    summary in place, updating the per-code tally. The list then stops
    growing while its content keeps changing, so a length-only cursor
    reported "nothing new" forever: the WS client kept the counts from the
    FIRST overflow while the in-process and REST views converged on the
    final ones -- exactly the two-layer drift G.5.2 forbids. The cursor
    therefore carries the summary's rendered text as well, and a changed
    summary is re-sent.

    That makes the overflow summary a SINGLETON on the wire: a
    ``diagnostics_truncated`` entry supersedes any previously delivered one
    rather than appending beside it (documented in the server spec's WS
    vocabulary). Bounded by construction -- the guard keeps exactly one such
    entry -- and the final tally is delivered because the bridge takes a
    delta after the terminal event, which the guard's own finalize precedes.

    Args:
        session: The streaming session.
        already_sent: Cursor from the previous call; the caller seeds it
            from what :func:`_initial_diagnostics_frame` delivered.

    Returns:
        ``(frame, cursor)``; ``frame`` is ``None`` when nothing new has accrued.
    """
    diagnostics = session.diagnostics()
    count, last_summary = already_sent
    fresh = [diag.model_dump(mode="json") for diag in diagnostics[count:]]
    summary = next(
        (diag.message for diag in reversed(diagnostics) if diag.code == _OVERFLOW_CODE), None
    )
    if not fresh and summary == last_summary:
        return None, already_sent
    if not fresh and summary is not None:
        # The list did not grow; only the in-place summary changed. Re-send
        # it alone -- the client replaces the entry it already has.
        fresh = [
            diag.model_dump(mode="json") for diag in diagnostics if diag.code == _OVERFLOW_CODE
        ]
    return {"type": "diagnostics", "diagnostics": fresh}, (len(diagnostics), summary)


def _scrub_event_for_client(event: TranscriptionEvent) -> dict[str, Any]:
    """Serialize an event to JSON, stripping internal detail from errors.

    The standard streaming layer stores a human-readable message under
    ``extra["detail"]`` of an ``error`` event -- for the ``engine_error``
    catch-all this is already the input-echo-free summary
    (``safe_exception_summary``, applied at the producer's catch site while
    the exception chain still exists). That summarized text may still name
    filesystem paths or upstream hosts (deliberate operator content), and an
    engine-constructed ``error`` event's ``extra`` is plugin-authored data
    the standard cannot vet at all -- so for EVERY ``error`` event the
    ``extra`` payload is dropped before it leaves the server (forwarding it
    verbatim to an unauthenticated WebSocket client would contradict the
    REST 500 non-leak contract). The safe structured fields (``code``,
    ``recoverable``, ``retriable_after``, ``segment_id``, and the
    gap/reconnect fields) are preserved; operators keep the dropped detail
    via the caller's logging (the server logs it as the opaque string it
    received -- the chain is gone, so it cannot re-analyze one). To surface
    a **non-sensitive** note to the client, an engine should use the
    structured diagnostics channel (:meth:`~standard_asr.runtime.streaming.\
TranscriptionSession.emit_diagnostic`), forwarded as a ``diagnostics`` frame. That
    channel is engine-authored and is **NOT** scrubbed (it is forwarded verbatim,
    like the transcript), so it is for non-sensitive notes only -- *sensitive*
    operator detail belongs in server-side ``logging``, never in a client-facing
    event or diagnostic. The asymmetry is deliberate: an ``error`` event's
    ``extra`` is auto-captured, so the server drops it here; a diagnostic is
    content the engine chose, so the engine owns its safety.

    Non-error events are serialized unchanged.

    Args:
        event: The event produced by the session.

    Returns:
        The JSON-serializable payload to send to the client.
    """
    if event.type == "error":
        # DROP, then dump -- not dump, then overwrite. The rule is
        # drop-before-send, and serializing first made the payload's fate
        # depend on the very field being discarded: an `extra` value with no
        # JSON form (an engine's exception object, a response handle) raised
        # inside the dump, so the client lost `code`, `recoverable`,
        # `retriable_after` and the gap/reconnect fields -- every safe
        # structured field the protocol documents -- to a value that was
        # never going to be sent.
        payload = event.model_dump(mode="json", exclude={"extra"})
        payload["extra"] = {}
        return payload
    return event.model_dump(mode="json")


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
        session: The engine's :class:`~standard_asr.runtime.streaming.TranscriptionSession`.
        max_frame_bytes: Maximum size of a single binary audio frame in bytes.
        max_session_bytes: Cumulative cap on total ingested audio bytes.
    """
    # Deferred import, mirroring create_app: fastapi is an optional extra, and
    # this function is only ever reached from the fastapi WS route.
    from fastapi import WebSocketDisconnect

    # Out-of-band terminal frames the pump asks the forward loop to deliver. A
    # byte-cap ``violation`` (``payload_too_large``) or a swallowed pump
    # ``failure`` (``stream_input_error``) stops the loop forwarding engine
    # events and sends a single, non-leaking policy frame instead.
    violation: dict[str, str] = {}
    pump_failed = False
    # Diagnostics already delivered to the client. The caller sent the
    # establishment-time set via _initial_diagnostics_frame; the producer has not
    # run yet, so this cursor matches exactly what the client has, and any change
    # below (emit_diagnostic / guard suppression / an updated overflow summary)
    # is forwarded as a delta.
    established = session.diagnostics()
    sent_diagnostics: _DiagnosticsCursor = (
        len(established),
        next((d.message for d in reversed(established) if d.code == _OVERFLOW_CODE), None),
    )

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
            # silently swallowed by the gather's return_exceptions
            # (explicit > implicit / fail-loud). Log the full detail server-side
            # and flag the forward loop to emit a single generic, non-leaking
            # error frame. (CancelledError derives from BaseException on the
            # teardown path, so it is not caught here and propagates as required.)
            log_exception_safely(logger, "WebSocket audio pump failed")
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
                    # the client receives the scrubbed event below. Log ONLY
                    # values in the event's declared space: ``%r``/``%s`` on an
                    # arbitrary object dispatches its ``__repr__``/``__str__``,
                    # and a value installed PAST validation (a mutated
                    # ``extra`` dict, ``model_construct``) reaches this line
                    # before the scrub below -- a held ``ValidationError``'s
                    # ``repr`` echoes its input into the very operator log the
                    # redaction contract closes. Exact ``str``/``None``
                    # dispatch no author code, so they log; anything else is
                    # withheld by SHAPE (``type(...) is str``, never
                    # ``isinstance`` -- a subclass can override rendering).
                    # Logged with ``%r``, never ``%s``: an exact str's builtin
                    # ``repr`` escapes every ``splitlines`` boundary (\\n, \\r,
                    # NEL, U+2028/29 -- all non-printable), so a multi-line
                    # engine-authored detail stays ONE log record instead of
                    # forging lines a parser attributes to other requests.
                    code = event.code
                    detail = event.extra.get("detail")
                    if (code is None or type(code) is str) and (
                        detail is None or type(detail) is str
                    ):
                        logger.error(
                            "Stream error event for client: code=%r detail=%r",
                            code,
                            detail,
                        )
                    else:
                        # Keep whichever part IS exact str/None (the usual
                        # case: a valid code beside a smuggled detail).
                        safe_code: object = (
                            code if (code is None or type(code) is str) else "<withheld>"
                        )
                        logger.error(
                            "Stream error event for client: code=%r "
                            "detail=<withheld: value installed past validation, "
                            "its rendering dispatches author code>",
                            safe_code,
                        )
                await websocket.send_json(_scrub_event_for_client(event))
                # Forward any diagnostics the producer accrued while emitting this
                # event (emit_diagnostic / a guard suppression), so the WS client
                # sees them as they happen rather than never.
                diag_frame, sent_diagnostics = _diagnostics_delta_frame(session, sent_diagnostics)
                if diag_frame is not None:
                    await websocket.send_json(diag_frame)
            if violation or pump_failed:
                # The break above skipped the per-event delta take, so forward
                # diagnostics accrued since the last delivered event (an
                # engine note, a guard suppression, an updated overflow
                # summary) BEFORE the terminal policy frame -- otherwise the
                # capped/failed session ends with the client never learning
                # e.g. that a parameter was degraded, while the REST path
                # returns every diagnostic on the result (the two-layer drift
                # the delta forwarder exists to prevent). Best-effort by
                # construction: the abandoned iteration is not drained, so a
                # diagnostic the producer records after this take is gone --
                # the cap is a resource bound, and cutting the violator off
                # promptly outranks completing its session.
                diag_frame, sent_diagnostics = _diagnostics_delta_frame(session, sent_diagnostics)
                if diag_frame is not None:
                    await websocket.send_json(diag_frame)
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
        except WebSocketDisconnect:
            # The client went away mid-stream; stop forwarding and tear down.
            pass
        except Exception:  # noqa: BLE001
            # Any other forward-loop fault (session iteration, event
            # serialization) MUST NOT vanish silently (explicit > implicit /
            # fail-loud). Log the full detail server-side and best-effort send
            # one generic, non-leaking frame (mirrors the _pump_audio contract).
            log_exception_safely(logger, "WebSocket event forwarding failed")
            try:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "internal_error",
                        "message": _internal_error_message("streaming"),
                    }
                )
            except Exception:  # noqa: BLE001
                # The socket may be unusable too; the fault is already logged.
                pass
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)


async def _create_engine_or_http_error(
    registry: ModelRegistry,
    model: str,
    http_exception: type[Exception],
) -> Any:
    """Instantiate the engine, mapping construction errors to HTTP status codes.

    Construction is ``registry.create(model)`` -- ZERO-ARG: the client chooses
    the model key and nothing else; every configuration input (credentials,
    endpoints, engine settings) comes from the server's own environment. Fault
    ownership follows from that, mirroring the compliance suite's
    classification of the same states:

    - an unknown or malformed model KEY (``EntrypointValidationError``) is a
      routing problem the caller can fix -> ``404``;
    - a key that RESOLVED to a registered model whose server-installed plugin
      then failed to load (``FactoryLoadError``) is a deployment fault, not a
      routing one: the caller's key was right and nothing they can send will
      change the outcome -> scrubbed ``500`` (``str(exc)`` also carries
      plugin import internals, which §3.7 keeps off client surfaces);
    - required configuration ABSENT from the server environment
      (``ConfigurationRequiredError`` -- e.g. a credential env var not set) is
      the OPERATOR's to fix, not the caller's: the model exists but is not
      available on this deployment -> ``503`` with a stable generic detail
      (never the field names; the specifics are safe-logged for the
      operator);
    - anything else -- including a plain ``ConfigError`` /
      ``InvalidProviderParamError`` / ``ValidationError``, which from a
      zero-arg factory is a broken deployment or plugin, exactly the state
      compliance fails as ``engine_construction_failed`` -- is an internal
      fault -> a generic, scrubbed ``500`` (same non-leak contract as
      :func:`_run_transcription`). The old ``422`` mapping blamed the caller
      for faults the caller cannot see, reach, or fix.

    Args:
        registry: The model registry.
        model: Model key in ``engine/model`` format.
        http_exception: The ``HTTPException`` class to raise.

    Returns:
        The instantiated engine.

    Raises:
        Exception: ``http_exception`` with an appropriate status code.
    """
    try:
        return await asyncio.to_thread(registry.create, model)
    except EntrypointValidationError as exc:
        # Unknown / unparseable model key: caller-fixable 404 with the
        # authored message (it names only the caller's key + available keys).
        raise http_exception(status_code=404, detail=str(exc)) from exc  # type: ignore[call-arg]
    except FactoryLoadError as exc:
        # The key resolved; the server-installed plugin failed to load. An
        # engine/deployment fault: a 404 blamed the caller for a fault it
        # cannot fix, and str(exc) carries plugin import/annotation
        # internals. Scrubbed 500, specifics safe-logged (§3.7).
        log_exception_safely(logger, "Registered model %r failed to load", model)
        detail = _internal_error_message("model construction")
        raise http_exception(status_code=500, detail=detail) from exc  # type: ignore[call-arg]
    except ConfigurationRequiredError as exc:
        # MUST precede any broader arm: ConfigurationRequiredError subclasses
        # ConfigError. The absent-config message names the missing field(s) --
        # deployment detail an unauthenticated caller has no business seeing.
        log_exception_safely(
            logger, "Engine %r requires configuration absent from the server environment", model
        )
        detail = _ENGINE_CONFIG_ABSENT_DETAIL
        raise http_exception(status_code=503, detail=detail) from exc  # type: ignore[call-arg]
    except Exception as exc:  # noqa: BLE001
        # Internal/unexpected construction fault (incl. ConfigError /
        # ValidationError from the zero-arg factory: a deployment or plugin
        # defect, never the caller's): log details, return a stable generic
        # message so we never leak internal paths or credential text.
        log_exception_safely(logger, "Engine construction failed for model %r", model)
        detail = _internal_error_message("model construction")
        raise http_exception(status_code=500, detail=detail) from exc  # type: ignore[call-arg]


async def _run_transcription(
    registry: ModelRegistry,
    model: str,
    audio: AudioInput,
    params: RuntimeParams | None,
    http_exception: type[Exception],
) -> str:
    """Instantiate the engine, transcribe, and map errors to HTTP status codes.

    The audio is passed as an :data:`~standard_asr.audio.input.AudioInput` (not a
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
        The :class:`TranscribeResponse` document, JSON-encoded -- the exact
        response body the routes send verbatim. Encoded inside the
        fault-mapping region so the wire projection is proven (and paid for)
        exactly once.

    Raises:
        Exception: ``http_exception`` with an appropriate status code.
    """
    asr = await _create_engine_or_http_error(registry, model, http_exception)

    try:
        result = await asyncio.to_thread(asr.transcribe, audio, params)
        # asyncio.to_thread returns transcribe()'s RAW value: an `async def`
        # implementation (or a sync wrapper delegating to one) hands back a
        # coroutine object that to_thread never drives. Enforce the sync-call
        # boundary here, and build the response INSIDE the fault-mapping
        # region -- a malformed result would otherwise raise a bare pydantic
        # ValidationError out of TranscribeResponse(...) past every arm below
        # (echoing engine input text past the scrubbed-500 contract).
        require_sync_result(result, "transcribe()", expected_type=TranscriptionResult)
        response = TranscribeResponse(model=model, result=result)
        # Finish the wire projection INSIDE the fault-mapping region -- and
        # ONCE. The dump is encoded here (allow_nan=False: NaN/Infinity are
        # Python floats but not JSON), so a result carrying a value with no
        # JSON form (reachable past the JsonValue declarations by
        # `model_construct` or by mutating an `extra` dict after
        # construction) fails HERE, in its true fault class -- the scrubbed
        # 500 below, safe-logged -- never in the ASGI encoder after the
        # endpoint returned. The encoded document IS the body the routes
        # send verbatim: encoding as a proof and then letting FastAPI
        # validate and serialize the same object AGAIN roughly doubled
        # response-serialization CPU and peak memory on every successful
        # request (tens of MB of words[] for a long recording).
        # Encoded the way the wire had it before this function owned the
        # projection (starlette's JSONResponse kwargs): ensure_ascii=False
        # ships a CJK/Cyrillic/Arabic transcript as UTF-8 instead of 6-byte
        # \uXXXX escapes (~1.35-3x smaller bodies on non-ASCII text), and
        # compact separators drop the per-element padding. allow_nan=False
        # stays: NaN/Infinity are Python floats but not JSON.
        return json.dumps(
            response.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except ValidationError as exc:
        # An ENGINE fault, not a request error: by this point the client's
        # options were already validated (WireRuntimeParams at the route) and
        # promoted to a typed RuntimeParams, so a bare pydantic
        # ValidationError escaping transcribe() can only come from the
        # engine's own internals (e.g. a structural engine constructing an
        # invalid TranscriptionResult -- EngineBase wraps that seam as
        # TranscriptionError, but the server cannot assume the base class).
        # Mapping it to 422 blamed the client's options for a plugin bug;
        # fault ownership must not depend on whether the engine inherits
        # EngineBase. Scrubbed 500 (pydantic's message echoes input values) --
        # and the LOG record is scrubbed the same way (the echo must not land
        # in operator/CI logs either).
        log_exception_safely(logger, "Engine-side validation failure for model %r", model)
        detail = _internal_error_message("transcription")
        raise http_exception(status_code=500, detail=detail) from exc  # type: ignore[call-arg]
    except ConfigurationRequiredError as exc:
        # Required config absent, discovered lazily at CALL time (an engine
        # that defers its credential check past construction): the same
        # operator-side availability state as the construction 503 -- the
        # caller cannot fix it and must not be told the field names. MUST
        # precede the ConfigError arm (subclass).
        log_exception_safely(
            logger, "Engine %r requires configuration absent from the server environment", model
        )
        detail = _ENGINE_CONFIG_ABSENT_DETAIL
        raise http_exception(status_code=503, detail=detail) from exc  # type: ignore[call-arg]
    except (ConfigError, InvalidProviderParamError) as exc:
        # An ENGINE fault, not a request error: the wire surface gives the
        # client no way to cause either -- engine init config never crosses
        # the wire (construction is zero-arg), `provider_params` is rejected
        # by WireRuntimeParams before transcription, and every client-fixable
        # rejection has its own type (UnsupportedFeatureError below; request
        # ValidationError at the route). A ConfigError here is an engine
        # declaration/config defect (e.g. a bad `default_language`) whose
        # authored message may carry server-side config detail. Scrubbed 500,
        # specifics safe-logged for the operator.
        log_exception_safely(logger, "Engine-side configuration/contract fault for model %r", model)
        detail = _internal_error_message("transcription")
        raise http_exception(status_code=500, detail=detail) from exc  # type: ignore[call-arg]
    except UnsupportedFeatureError as exc:
        # Client-caused: the request asked for a feature/language the engine
        # does not support (strict mode). The authored message is written for
        # the caller.
        raise http_exception(status_code=422, detail=str(exc)) from exc  # type: ignore[call-arg]
    except AudioProcessingError as exc:
        raise http_exception(status_code=400, detail=str(exc)) from exc  # type: ignore[call-arg]
    except Exception as exc:  # noqa: BLE001
        # Internal/unexpected (including EngineContractError from the sync-call
        # boundary above): log details, return a stable generic message so we
        # never leak internal paths or upstream/credential text to the client.
        log_exception_safely(logger, "Transcription failed for model %r", model)
        detail = _internal_error_message("transcription")
        raise http_exception(status_code=500, detail=detail) from exc  # type: ignore[call-arg]


def _build_params(options: dict[str, Any] | None) -> RuntimeParams | None:
    """Build :class:`RuntimeParams` from an untyped JSON options object.

    Validation goes through :class:`WireRuntimeParams`, the **portable-only** wire
    view, so a request that includes the engine-specific ``provider_params``
    escape hatch is rejected with a clear validation error (``provider_params``
    cannot be sent -- it is discover-only via the params-schema endpoint and is
    not constructible from untyped wire JSON). The validated portable params are
    then promoted to the internal :class:`RuntimeParams`.

    Args:
        options: A JSON options object, or ``None``.

    Returns:
        Parsed runtime parameters, or ``None``.

    Raises:
        ValidationError: If ``options`` is not a valid portable params object
            (including when it carries a ``provider_params`` key).
    """
    if options is None:
        return None
    return WireRuntimeParams.model_validate(options).to_runtime_params()


def _engine_class_or_http_error(
    registry: ModelRegistry, model: str, http_exception: type[Exception]
) -> Any:
    """Resolve an engine class (without instantiation), or map the failure to HTTP.

    Args:
        registry: The model registry.
        model: Model key in ``engine/model`` format.
        http_exception: The ``HTTPException`` class to raise.

    Returns:
        The engine class.

    Raises:
        Exception: ``http_exception`` -- 404 when the model key is unknown or
            unparseable (caller-fixable), scrubbed 500 when the key resolved
            but the plugin's class failed to load/resolve (an
            engine/deployment fault, safe-logged).
    """
    try:
        return registry.engine_class(model)
    except EntrypointValidationError as exc:
        raise http_exception(status_code=404, detail=str(exc)) from exc  # type: ignore[call-arg]
    except FactoryLoadError as exc:
        # Same fault-ownership split as engine construction: the metadata
        # endpoints must not blame the caller for a broken plugin nor leak
        # its import/annotation internals.
        log_exception_safely(logger, "Registered model %r failed to load its class", model)
        detail = _internal_error_message("model metadata")
        raise http_exception(status_code=500, detail=detail) from exc  # type: ignore[call-arg]


def _prove_json_projectable(payload: object) -> None:
    """Prove a payload can be sent, before the response is committed to.

    The Python and JSON layers are meant to be the same protocol seen twice
    (G5.2), and the wire-visible fields are declared in
    :data:`~pydantic.JsonValue` so that holds by construction. This is the
    enforcement point for what a declaration cannot reach: a value installed
    past validation (``model_construct``, or mutating an ``extra`` dict after
    construction). Without it the failure surfaced in the ASGI encoder --
    after the endpoint had already returned, so the documented scrubbed
    response and the safe-logging boundary were both bypassed.

    ``allow_nan=False`` because ``NaN``/``Infinity`` are Python floats but
    not JSON: the permissive default emits a document no conforming parser
    accepts, which is a wrong result sent silently rather than a fault.

    Args:
        payload: The already-dumped payload.

    Raises:
        Exception: Whatever encoding raises (``TypeError``/``ValueError``),
            for the caller's fault boundary to map.
    """
    json.dumps(payload, allow_nan=False)


def _metadata_or_http_error(
    registry: ModelRegistry,
    model: str,
    http_exception: type[Exception],
    *,
    project: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """Run a whole metadata read behind ONE fault boundary, or map it to HTTP.

    The read spans engine-class RESOLUTION too
    (:func:`_engine_class_or_http_error`): everything from the model key
    onward is third-party code -- loading the entry-point target, the
    ``_is_protocol`` / ``transcribe`` duck-type reads (metaclass/descriptor
    dispatch), then reading ``declared_capabilities`` /
    ``provider_params_type`` / ``config_type`` (whatever descriptor the
    plugin put there), ``canonical_json()`` and ``model_json_schema()``
    (plugin methods, and a custom ``__get_pydantic_json_schema__`` runs
    inside the latter). A raise anywhere in that stretch that is not one of
    the mapped failures otherwise left the endpoint through Starlette's
    unhandled-exception path: the client got an
    undocumented plain 500 instead of the documented scrubbed body, and --
    the reason this is a security fault and not a tidiness one --
    :func:`log_exception_safely` never ran, so the ASGI server's own
    traceback logger rendered the exception chain natively. A
    ``ValidationError`` anywhere in it then printed its input echo into the
    operator log, which is exactly what the redaction contract forbids.
    These endpoints are unauthenticated discovery surfaces, so "a plugin
    descriptor does not usually fail" is not a boundary.

    The projection is also finished HERE rather than left to the ASGI
    encoder: the value is JSON-encoded eagerly (rejecting non-finite
    numbers, which are not JSON), so a plugin returning something
    unencodable is a scrubbed 500 from inside the boundary instead of a
    crash after the endpoint returned.

    Args:
        registry: The model registry.
        model: Model key in ``engine/model`` format.
        http_exception: The ``HTTPException`` class to raise.
        project: Reads the metadata off the engine class and returns the
            JSON-ready payload. It may raise ``http_exception`` itself for
            a DELIBERATE verdict (e.g. 404 "no capabilities declared"),
            which passes through unchanged.

    Returns:
        The payload, proven JSON-encodable.

    Raises:
        Exception: ``http_exception`` -- 404 for an unknown/unparseable
            model key or the projection's own deliberate verdict, scrubbed
            500 for every engine/deployment fault (safe-logged).
    """
    try:
        # Resolution is INSIDE the boundary too: it dispatches the plugin's
        # own machinery (loading the entry-point target, and
        # ``_ensure_engine_class``'s ``_is_protocol`` / ``transcribe`` reads
        # -- both metaclass/descriptor dispatch). A fault raised THERE (a
        # metaclass property carrying a ``ValidationError``, say) is not an
        # ``EntrypointValidationError``/``FactoryLoadError``; resolved
        # outside this try it left the endpoint through Starlette's
        # unhandled path -- the undocumented plain 500, and the ASGI
        # server's native traceback logging of the raw chain, echo included.
        engine_class = _engine_class_or_http_error(registry, model, http_exception)
        payload = project(engine_class)
        _prove_json_projectable(payload)
    except http_exception:
        # The resolution/projection's own deliberate verdicts (404/500
        # mapped by the helper, or the projection's 404 "no capabilities").
        raise
    except Exception as exc:  # noqa: BLE001 - a plugin fault is a scrubbed 500
        log_exception_safely(logger, "Model %r failed to produce its metadata", model)
        detail = _internal_error_message("model metadata")
        raise http_exception(status_code=500, detail=detail) from exc  # type: ignore[call-arg]
    return payload


def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "info",
    *,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    max_ws_frame_bytes: int = DEFAULT_MAX_WS_FRAME_BYTES,
    max_ws_session_bytes: int = DEFAULT_MAX_WS_SESSION_BYTES,
) -> None:
    """Run the FastAPI server using Uvicorn.

    Auto-reload is intentionally **not** offered here. Uvicorn's reload (and
    multi-worker) modes require the app to be passed as an import string, but this
    function builds and passes a configured ``FastAPI`` instance (so the byte
    caps are honored); uvicorn rejects ``reload`` for a non-import-string app by
    exiting the process. Rather than ship a parameter that can only fail, the
    server library does not own dev-reload: run uvicorn directly against an import
    string for that workflow.

    The WebSocket per-frame cap is wired to uvicorn's transport ``ws_max_size``
    so the app-level bound (``max_ws_frame_bytes``) and the transport bound are
    the **same** honest value -- uvicorn's default ``ws_max_size`` is 16 MiB, so
    without this a smaller app cap would not actually bound a frame at the
    transport (and a larger one would be silently clamped by the transport). The
    config/handshake frame is additionally bounded by the app cap in
    :func:`_receive_config_frame`, so the effective per-frame bound is
    ``min(app cap, transport ws_max_size)`` regardless of deployment.

    Args:
        host: Bind host.
        port: Bind port.
        log_level: Uvicorn log level.
        max_body_bytes: HTTP request-body cap (see :func:`create_app`).
        max_ws_frame_bytes: WebSocket per-frame cap; also passed to uvicorn as
            ``ws_max_size`` (see :func:`create_app`).
        max_ws_session_bytes: WebSocket per-session cap (see :func:`create_app`).

    Returns:
        None.

    Raises:
        ImportError: If Uvicorn is not installed.
        ValueError: If any byte cap is not positive (via :func:`create_app`).
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "Uvicorn is required to run the server. Install with: "
            "pip install 'standard-asr[server]'."
        ) from exc

    app = create_app(
        max_body_bytes=max_body_bytes,
        max_ws_frame_bytes=max_ws_frame_bytes,
        max_ws_session_bytes=max_ws_session_bytes,
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        ws_max_size=max_ws_frame_bytes,
    )


__all__ = [
    "ModelInfo",
    "TranscribeJsonRequest",
    "TranscribeResponse",
    "create_app",
    "run",
]
