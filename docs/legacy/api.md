# FastAPI Server Specification

Standard ASR ships an optional FastAPI server to expose any compliant engine as
an HTTP API. Install with:

```bash
pip install "standard-asr[server]"
```

## 1. Endpoints

### `GET /v1/health`
Simple health check.

### `GET /v1/models`
List discovered models.

Response example:

```json
[
  {
    "key": "faster-whisper/whisper",
    "engine_id": "faster-whisper",
    "model_name": "whisper"
  }
]
```

### `POST /v1/transcribe`
Multipart file upload for transcription.

**Form fields**:
- `model`: model key (`engine/model`)
- `file`: audio file
- `options`: JSON string (optional)

### `POST /v1/transcribe:json`
JSON API for transcription.

Request schema:

```json
{
  "model": "faster-whisper/whisper",
  "audio": "data:audio/wav;base64,..." ,
  "options": {"language": "en"}
}
```

Response schema:

```json
{
  "model": "faster-whisper/whisper",
  "result": { "text": "...", "segments": [] }
}
```

## 2. Error Handling

- 400 for invalid input or decode errors.
- 500 for transcription failures.

## 3. Deployment

The server respects `STANDARD_ASR_ALLOW_DOWNLOAD`. For production, set it to
`0` and warm up models using `standard-asr models prepare`.
