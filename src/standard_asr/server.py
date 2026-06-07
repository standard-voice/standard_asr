"""FastAPI server utilities for Standard ASR."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from .discovery import ModelRegistry, discover_models
from .results import TranscriptionResult
from .runtime_params import RuntimeParams
from .utils.audio_loader import load_audio, load_audio_from_bytes


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

    model_config = ConfigDict(frozen=True, extra="forbid")

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
    audio: str = Field(
        ..., description="Base64 data URI or raw base64-encoded audio payload."
    )
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
    result: TranscriptionResult = Field(
        ..., description="Standard ASR transcription result."
    )


def create_app(registry: ModelRegistry | None = None):
    """Create a FastAPI application for Standard ASR.

    Args:
        registry: Optional pre-discovered registry.

    Returns:
        FastAPI application instance.

    Raises:
        ImportError: If FastAPI dependencies are missing.
    """
    try:
        from fastapi import FastAPI, File, Form, HTTPException
    except ImportError as exc:
        raise ImportError(
            "FastAPI dependencies are missing. Install with: "
            "pip install 'standard-asr[server]'."
        ) from exc

    app = FastAPI(title="Standard ASR")
    model_registry = registry or discover_models()

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
        return [
            ModelInfo(
                key=name,
                engine_id=model_registry.spec(name).engine_id,
                model_name=model_registry.spec(name).model_name,
            )
            for name in model_registry.names()
        ]

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
        try:
            audio = await asyncio.to_thread(load_audio_from_bytes, file)
            params = _build_params(json.loads(options) if options else None)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            asr = await asyncio.to_thread(model_registry.create, model)
            result = await asyncio.to_thread(asr.transcribe, audio, params)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return TranscribeResponse(model=model, result=result)

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
        try:
            audio = await asyncio.to_thread(_decode_audio_payload, payload.audio)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            params = _build_params(payload.options)
            asr = await asyncio.to_thread(model_registry.create, payload.model)
            result = await asyncio.to_thread(asr.transcribe, audio, params)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return TranscribeResponse(model=payload.model, result=result)

    @app.get("/v1/capabilities/{model:path}")
    def capabilities(model: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return an engine's declared capabilities as canonical JSON.

        Args:
            model: Model key in ``engine/model`` format.

        Returns:
            The declared capability tree.

        Raises:
            HTTPException: If the model is unknown or has no capabilities.
        """
        engine = _create_or_404(model_registry, model, HTTPException)
        caps = getattr(engine, "declared_capabilities", None)
        if caps is None:
            raise HTTPException(status_code=404, detail="No capabilities declared.")
        return caps.model_dump(mode="json")

    @app.get("/v1/params-schema/{model:path}")
    def params_schema(model: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return the JSON Schema for an engine's ``provider_params``.

        Args:
            model: Model key in ``engine/model`` format.

        Returns:
            The provider-params JSON Schema, or ``{}`` if the engine has none.

        Raises:
            HTTPException: If the model is unknown.
        """
        engine = _create_or_404(model_registry, model, HTTPException)
        params_type = getattr(engine, "provider_params_type", None)
        if params_type is None:
            return {}
        return params_type.model_json_schema()

    return app


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


def _create_or_404(
    registry: ModelRegistry, model: str, http_exception: type[Exception]
) -> Any:
    """Create an engine instance or raise a 404.

    Args:
        registry: The model registry.
        model: Model key in ``engine/model`` format.
        http_exception: The ``HTTPException`` class to raise.

    Returns:
        The engine instance.

    Raises:
        Exception: ``http_exception`` with status 404 if the model is unknown.
    """
    try:
        return registry.create(model)
    except Exception as exc:  # noqa: BLE001
        raise http_exception(status_code=404, detail=str(exc)) from exc  # type: ignore[call-arg]


def _decode_audio_payload(payload: str) -> NDArray[np.float32]:
    """Decode a base64 payload into a normalized audio array.

    Args:
        payload: Base64 data URI or raw base64 string.

    Returns:
        Normalized audio array.

    Raises:
        ValueError: If decoding fails.
    """
    if payload.strip().lower().startswith("data:") and ";base64," in payload:
        return load_audio(payload)
    try:
        decoded = base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid base64 payload for audio.") from exc
    return load_audio_from_bytes(decoded)


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
