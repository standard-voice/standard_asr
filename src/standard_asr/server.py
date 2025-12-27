"""FastAPI server utilities for Standard ASR."""

from __future__ import annotations

import base64
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .discovery import ModelRegistry, discover_models
from .results import TranscriptionResult
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
        None, description="Optional transcription options."
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
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    except ImportError as exc:
        raise ImportError(
            "FastAPI dependencies are missing. Install with: "
            "pip install 'standard-asr[server]'."
        ) from exc

    app = FastAPI(title="Standard ASR")
    model_registry = registry or discover_models()

    @app.get("/v1/health")
    def health() -> dict[str, str]:
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
    def list_models() -> list[ModelInfo]:
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
    async def transcribe_file(
        model: str = Form(...),
        file: UploadFile = File(...),
        options: str | None = Form(None),
    ) -> TranscribeResponse:
        """Transcribe audio from a multipart file upload.

        Args:
            model: Model key in ``engine/model`` format.
            file: Uploaded audio file.
            options: Optional JSON options string.

        Returns:
            Transcription response.

        Raises:
            HTTPException: If decoding or transcription fails.
        """
        try:
            raw = await file.read()
            audio = load_audio_from_bytes(raw)
            options_payload = json.loads(options) if options else None
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            asr = model_registry.create(model)
            result = asr.transcribe(audio, options=options_payload)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return TranscribeResponse(model=model, result=result)

    @app.post("/v1/transcribe:json", response_model=TranscribeResponse)
    def transcribe_json(payload: TranscribeJsonRequest) -> TranscribeResponse:
        """Transcribe audio from a JSON payload.

        Args:
            payload: JSON request payload.

        Returns:
            Transcription response.

        Raises:
            HTTPException: If decoding or transcription fails.
        """
        try:
            audio = _decode_audio_payload(payload.audio)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            asr = model_registry.create(payload.model)
            result = asr.transcribe(audio, options=payload.options)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return TranscribeResponse(model=payload.model, result=result)

    return app


def _decode_audio_payload(payload: str) -> Any:
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


__all__ = ["ModelInfo", "TranscribeJsonRequest", "TranscribeResponse", "create_app", "run"]
