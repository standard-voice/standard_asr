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
from collections.abc import Sequence
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
from .runtime_params import RuntimeParams, WireRuntimeParams
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

#: Substring tokens that mark a field name as credential-like. A request field
#: whose name contains any of these (case-insensitive) has its value redacted
#: from validation-error detail, so a mis-placed secret (e.g. an ``api_key`` put
#: in the JSON body / ``options``) is never reflected back to the
#: client / proxy / bug-report logs.
_CREDENTIAL_FIELD_TOKENS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "authorization",
    "auth",
    "credential",
    "private_key",
    "access_key",
    "session_key",
    "bearer",
)

#: Placeholder substituted for a redacted credential value in error detail.
_REDACTED: str = "[redacted]"


def _loc_to_list(loc: object) -> list[object]:
    """Normalize a pydantic error ``loc`` into a plain list.

    Args:
        loc: The ``loc`` value from a single pydantic error entry (a tuple/list
            of path components, or a scalar).

    Returns:
        The components as a list (a scalar is wrapped in a single-element list).
    """
    if isinstance(loc, (list, tuple)):
        return list(loc)  # pyright: ignore[reportUnknownArgumentType]
    return [loc]


def _loc_is_credential(loc: list[object]) -> bool:
    """Return whether a pydantic error ``loc`` names a credential-like field.

    Args:
        loc: The normalized ``loc`` path components.

    Returns:
        ``True`` if any string component of ``loc`` contains a credential token.
    """
    for part in loc:
        if isinstance(part, str):
            lowered = part.lower()
            if any(token in lowered for token in _CREDENTIAL_FIELD_TOKENS):
                return True
    return False


def _sanitize_validation_errors(errors: Sequence[Any]) -> list[dict[str, Any]]:
    """Strip the echoed ``input`` (and ``url``) from pydantic error entries.

    FastAPI / pydantic's default error detail echoes the offending ``input``
    value verbatim (and may repeat it under ``ctx``). When a caller mis-places a
    secret (e.g. an ``api_key`` in the JSON body or ``options``), that value is
    reflected back into the client / any intermediary proxy / a copied bug report
    -- a credential leak. This rebuilds each entry from only the safe structured
    fields (``type``, ``loc``, ``msg``), thereby dropping the ``input`` echo, the
    ``ctx``, and the ``url`` entirely, and additionally redacts the ``msg`` of any
    entry whose ``loc`` names a credential-like field (whose validator message
    could itself contain the value).

    Args:
        errors: The raw ``ValidationError.errors()`` / ``RequestValidationError``
            error list.

    Returns:
        A new list of sanitized error entries safe to return to a client.
    """
    sanitized: list[dict[str, Any]] = []
    for raw in errors:
        error: dict[str, Any] = dict(raw)
        loc = _loc_to_list(error.get("loc", ()))
        is_credential = _loc_is_credential(loc)
        entry: dict[str, Any] = {
            "type": error.get("type"),
            "loc": loc,
            # Redact the message for credential fields (it can echo the value);
            # otherwise keep the validator's own message (never the raw input).
            "msg": _REDACTED if is_credential else error.get("msg"),
        }
        sanitized.append(entry)
    return sanitized


def _sanitized_validation_message(exc: ValidationError) -> str:
    """Build a safe, input-free summary string from a pydantic error.

    Used where a single ``detail`` string is expected (the ``options`` build
    path). Mirrors :func:`_sanitize_validation_errors`: it names the offending
    field(s) and the validator message but never echoes the submitted value, and
    redacts credential-like fields entirely.

    Args:
        exc: The pydantic validation error.

    Returns:
        A human-readable, secret-free error string.
    """
    parts: list[str] = []
    for entry in _sanitize_validation_errors(exc.errors()):
        loc = ".".join(str(p) for p in entry["loc"]) or "(root)"
        parts.append(f"{loc}: {entry['msg']}")
    joined = "; ".join(parts) or "invalid options"
    return f"Invalid options: {joined}"


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
        from fastapi.responses import JSONResponse
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
            # non-portable provider_params key, D5) is an unprocessable entity.
            # Surface a sanitized message: pydantic's str(exc) echoes the
            # offending input value (a mis-placed secret would be reflected).
            raise HTTPException(status_code=422, detail=_sanitized_validation_message(exc)) from exc

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
        # The request-body cap is enforced at the ASGI boundary by
        # _BodySizeLimitMiddleware (Content-Length *and* actual bytes), so the
        # encoded ``audio`` is already bounded by the time it materialises here.
        try:
            # `payload.options` is already a parsed object, so the only failure
            # here is params validation (bad value, unknown key, or a non-portable
            # provider_params key, D5) -> 422. pydantic's str(exc) echoes the
            # offending input value (a mis-placed secret would be reflected), so
            # surface a sanitized message instead.
            params = _build_params(payload.options)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=_sanitized_validation_message(exc)) from exc

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
            config = await _receive_config_frame(websocket, max_ws_frame_bytes)
            audio_format = AudioFormat(**config["audio_format"])
            params = _build_params(config.get("options"))
        except _ConfigFrameTooLarge as exc:
            # The config/handshake frame is bounded by the app cap too (not just
            # the transport ws_max_size), so the documented DoS bound holds
            # regardless of the ASGI server in front. Reported like the audio
            # caps (server.md §4.4).
            await websocket.send_json(
                {"type": "error", "code": "payload_too_large", "message": str(exc)}
            )
            await websocket.close()
            return
        except ValidationError as exc:
            # Sanitize: pydantic's str(exc) echoes the offending input value, so
            # a mis-placed secret in options would be reflected to the client.
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "bad_request",
                    "message": _sanitized_validation_message(exc),
                }
            )
            await websocket.close()
            return
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
        except (InvalidProviderParamError, ConfigError, ValidationError) as exc:
            # Client-caused construction failure (bad config / missing
            # credentials / invalid options) -- the caller can fix it, so it is a
            # bad_request, mirroring the REST 422 mapping.
            await websocket.send_json({"type": "error", "code": "bad_request", "message": str(exc)})
            await websocket.close()
            return
        except Exception:  # noqa: BLE001
            # Internal/unexpected construction fault: never crash the route or
            # leak detail. Log server-side; send a single generic, non-leaking
            # frame (mirrors the REST scrubbed-500 contract, §3.7).
            logger.exception("Engine construction failed for streaming model %r", model)
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": "Internal model construction error. See server logs for details.",
                }
            )
            await websocket.close()
            return

        try:
            # start_transcription is part of the structural StandardASR protocol
            # (a batch-only engine raises UnsupportedFeatureError, handled below).
            session = asr.start_transcription(audio_format=audio_format, params=params)
        except (ConfigError, InvalidProviderParamError) as exc:
            # The streaming template now runs the full gating pipeline at session
            # establishment, so a language misconfig (ConfigError) or a
            # swapped-engine provider_params mismatch (InvalidProviderParamError)
            # surfaces HERE, not only at engine construction. These are
            # client-fixable -> bad_request, mirroring the construction mapping
            # above and the REST 422 (server.md §4.2). NOTE: both subclass
            # ValueError, so this clause MUST precede the UnsupportedFeatureError /
            # ValueError clause below, which would otherwise mislabel them
            # "unsupported".
            await websocket.send_json({"type": "error", "code": "bad_request", "message": str(exc)})
            await websocket.close()
            return
        except (UnsupportedFeatureError, ValueError) as exc:
            await websocket.send_json({"type": "error", "code": "unsupported", "message": str(exc)})
            await websocket.close()
            return
        except Exception:  # noqa: BLE001
            # Internal/unexpected session-establishment fault (e.g. a fault in the
            # engine's own _start_transcription hook): never crash the route or
            # leak detail. Log server-side; send a single generic, non-leaking
            # frame (mirrors the construction scrubbed-frame contract, §3.7).
            logger.exception("Stream session establishment failed for model %r", model)
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": "Internal stream establishment error. See server logs for details.",
                }
            )
            await websocket.close()
            return

        # Forward the standard-layer diagnostics (best-effort parameter degrade,
        # language resolution, audio conversion) attached at session
        # establishment, so a WS client can see WHY word_timestamps / prompt /
        # language were dropped or changed -- the REST path returns these on the
        # result, and the WS surface must not silently hide them.
        diagnostics_frame = _initial_diagnostics_frame(session)
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


async def _receive_config_frame(websocket: WebSocket, max_frame_bytes: int) -> dict[str, Any]:
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
        The parsed config object.

    Raises:
        _ConfigFrameTooLarge: If the raw config frame exceeds ``max_frame_bytes``.
        Exception: If the frame is not a JSON object (surfaced as ``bad_request``
            by the caller).
    """
    message = await websocket.receive()
    raw = message.get("text")
    payload: bytes = raw.encode() if isinstance(raw, str) else (message.get("bytes") or b"")
    if len(payload) > max_frame_bytes:
        raise _ConfigFrameTooLarge(
            f"Config frame too large: {len(payload)} bytes exceeds the "
            f"{max_frame_bytes}-byte per-frame limit."
        )
    return json.loads(payload)


def _initial_diagnostics_frame(session: TranscriptionSession) -> dict[str, Any] | None:
    """Build the standard-layer diagnostics frame for a freshly-started session.

    The base ``start_transcription`` template attaches the parameter-gating and
    language-axis diagnostics (best-effort degrade, language resolution, audio
    conversion) to the session before handing it back, so they are available via
    :meth:`~standard_asr.streaming.TranscriptionSession.diagnostics` immediately.
    The REST path returns these on the result; the WS surface forwards them as a
    single ``diagnostics`` frame up front so the client learns WHY a parameter
    was dropped or changed before audio flows.

    Unlike the ``engine_error`` detail (raw ``str(exc)``, scrubbed by
    :func:`_scrub_event_for_client`), these messages are standard-layer-authored
    (not raw exception text), so they are forwarded verbatim -- exactly as REST
    returns them.

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


async def _create_engine_or_http_error(
    registry: ModelRegistry,
    model: str,
    http_exception: type[Exception],
) -> Any:
    """Instantiate the engine, mapping construction errors to HTTP status codes.

    Engine construction (factory load + ``__init__``) can fail for distinct
    reasons that must NOT all collapse to a non-spec ``500``:

    - an unknown / unloadable model (``EntrypointValidationError`` /
      ``FactoryLoadError``) is a routing problem -> ``404`` (server.md §3.7);
    - a client-supplied config problem surfaced during construction -- bad
      config, missing credentials, or a ``pydantic`` validation error
      (``ConfigError`` / ``InvalidProviderParamError`` / ``ValidationError``) --
      is the caller's to fix -> ``422``;
    - anything else is an internal fault -> a generic, scrubbed ``500`` whose
      raw text is logged server-side only (same non-leak contract as
      :func:`_run_transcription`).

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
    except (EntrypointValidationError, FactoryLoadError) as exc:
        raise http_exception(status_code=404, detail=str(exc)) from exc  # type: ignore[call-arg]
    except (InvalidProviderParamError, ConfigError, ValidationError) as exc:
        # Client-caused construction failure (bad config / missing credentials /
        # invalid options) -- the caller can fix it, so it is a 422, not a 500.
        raise http_exception(status_code=422, detail=str(exc)) from exc  # type: ignore[call-arg]
    except Exception as exc:  # noqa: BLE001
        # Internal/unexpected construction fault: log details, return a stable
        # generic message so we never leak internal paths or credential text.
        logger.exception("Engine construction failed for model %r", model)
        detail = "Internal model construction error. See server logs for details."
        raise http_exception(status_code=500, detail=detail) from exc  # type: ignore[call-arg]


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
    asr = await _create_engine_or_http_error(registry, model, http_exception)

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
    """Build :class:`RuntimeParams` from an untyped JSON options object (D5).

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
    *,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    max_ws_frame_bytes: int = DEFAULT_MAX_WS_FRAME_BYTES,
    max_ws_session_bytes: int = DEFAULT_MAX_WS_SESSION_BYTES,
) -> None:
    """Run the FastAPI server using Uvicorn.

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
        reload: Enable auto-reload.
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
        reload=reload,
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
