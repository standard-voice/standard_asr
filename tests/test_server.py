# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FastAPI server helpers.

The transcription endpoints deliberately do **not** pre-decode uploads: they
hand the encoded payload to the engine's own standard negotiation. The tests
below therefore exercise real :class:`EngineBase` engines so that decoding,
resampling and encoded-passthrough are proven end-to-end (a bare stub that
ignored the audio would mask the very contract the server must honour).
"""

from __future__ import annotations

import base64
import builtins
import io
import json
import logging
import wave
from collections.abc import AsyncIterator
from importlib.metadata import EntryPoint
from typing import Any, ClassVar, Literal, cast

import httpx2
import numpy as np
import pytest

from standard_asr import (
    DiarizationRequest,
    RuntimeParams,
    TranscriptionResult,
)
from standard_asr.contract.capabilities import (
    BatchCapabilities,
    CandidateLanguagesCap,
    CandidateLanguagesConstraints,
    DeclaredCapabilities,
    DiarizationCap,
    FlagCap,
    LanguageCaps,
    StreamingCapabilities,
)
from standard_asr.contract.params import ProviderParams
from standard_asr.engine import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    InputKind,
    PreparedAudio,
    SampleRateRange,
)
from standard_asr.plugins.discovery import ModelRegistry, discover_models
from standard_asr.runtime.config import LanguageConfigMixin
from standard_asr.runtime.streaming import TranscriptionEvent, TranscriptionSession
from standard_asr.toolchain import server as server_module


class _DummyConfig(BaseConfig[str]):
    engine: str = "dummy"


class _DummyProperties(BaseProperties):
    engine_id: str = "dummy"
    model_name: str = "echo"
    protocol_version: str = "0.2.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    selectable_languages: list[str] = ["en"]


class _DummyParams(ProviderParams):
    beam: int = 1


_DUMMY_CAPS = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
    )
)


class _DummyASR:
    """Bare structural engine (not EngineBase): ignores audio, returns a fixed
    transcript. Used for the error-mapping, capabilities and params-schema
    tests, none of which depend on audio negotiation."""

    properties: ClassVar[_DummyProperties] = _DummyProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _DUMMY_CAPS
    provider_params_type: ClassVar[type[ProviderParams] | None] = _DummyParams

    def __init__(self) -> None:
        self.config = _DummyConfig(engine="dummy")

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="dummy")


def _dummy_factory() -> _DummyASR:  # pyright: ignore[reportUnusedFunction]
    return _DummyASR()


class _FailASR(_DummyASR):
    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        raise RuntimeError("boom: /secret/internal/path leaked")


class _CjkASR(_DummyASR):
    """Returns a non-ASCII transcript (wire-encoding test)."""

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        return TranscriptionResult(text="中文轉錄測試")


def _cjk_factory() -> _CjkASR:  # pyright: ignore[reportUnusedFunction]
    return _CjkASR()


def _fail_factory() -> _FailASR:  # pyright: ignore[reportUnusedFunction]
    return _FailASR()


class _AsyncTranscribeASR(_DummyASR):
    """Sync ``transcribe`` delegating to an ``async def`` (a coroutine leaks out).

    The declaration is synchronous (``iscoroutinefunction`` is ``False``), so
    only the RETURNED value betrays the defect -- the exact shape the runtime
    sync-call boundary exists to catch.
    """

    async def _impl(self) -> TranscriptionResult:
        """Return a result from the async implementation (never driven).

        Returns:
            A fixed result (unreachable in these tests).
        """
        return TranscriptionResult(text="never")  # pragma: no cover - never awaited

    def transcribe(self, audio: Any, options: Any = None) -> Any:
        return self._impl()


def _async_transcribe_factory() -> _AsyncTranscribeASR:  # pyright: ignore[reportUnusedFunction]
    return _AsyncTranscribeASR()


class _WrongResultTypeASR(_DummyASR):
    """``transcribe`` returns a plain dict instead of a ``TranscriptionResult``."""

    def transcribe(self, audio: Any, options: Any = None) -> Any:
        return {"text": "dict-not-result"}


def _wrong_result_type_factory() -> _WrongResultTypeASR:  # pyright: ignore[reportUnusedFunction]
    return _WrongResultTypeASR()


class _AsyncStartASR(_DummyASR):
    """Sync ``start_transcription`` delegating to an ``async def``."""

    async def _establish(self) -> TranscriptionSession:
        """Establish the session from the async implementation (never driven).

        Returns:
            Nothing usable (unreachable in these tests).

        Raises:
            RuntimeError: Always, if ever driven.
        """
        raise RuntimeError("never driven")  # pragma: no cover - never awaited

    def start_transcription(self, *, audio_format: Any = None, params: Any = None) -> Any:
        return self._establish()


def _async_start_factory() -> _AsyncStartASR:  # pyright: ignore[reportUnusedFunction]
    return _AsyncStartASR()


class _ClientErrorASR(_DummyASR):
    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        from standard_asr.contract.exceptions import UnsupportedFeatureError

        raise UnsupportedFeatureError("word_timestamps not supported")


def _client_error_factory() -> _ClientErrorASR:  # pyright: ignore[reportUnusedFunction]
    return _ClientErrorASR()


class _NoInstantiateASR(_DummyASR):
    def __init__(self) -> None:
        raise RuntimeError("instantiation forbidden (would resolve credentials)")


def _no_instantiate_factory() -> _NoInstantiateASR:  # pyright: ignore[reportUnusedFunction]
    return _NoInstantiateASR()


class _WithConfigTypeASR(_DummyASR):
    """Engine declaring ``config_type`` so its config schema is discoverable."""

    config_type: ClassVar[type[BaseConfig[str]] | None] = _DummyConfig


def _with_config_type_factory() -> _WithConfigTypeASR:  # pyright: ignore[reportUnusedFunction]
    return _WithConfigTypeASR()


class _NoInstantiateConfigTypeASR(_NoInstantiateASR):
    """Credentialed engine: construction raises, but config schema must serve."""

    config_type: ClassVar[type[BaseConfig[str]] | None] = _DummyConfig


def _no_instantiate_config_type_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _NoInstantiateConfigTypeASR
):
    return _NoInstantiateConfigTypeASR()


class _NotAConfigType:
    """Deliberately not a BaseConfig subclass."""


class _BrokenConfigTypeASR(_DummyASR):
    """Engine whose config_type declaration is broken (not a BaseConfig)."""

    config_type: ClassVar[Any] = _NotAConfigType


def _broken_config_type_factory() -> _BrokenConfigTypeASR:  # pyright: ignore[reportUnusedFunction]
    return _BrokenConfigTypeASR()


class _ConfigErrorOnConstructASR(_DummyASR):
    """Construction raises a client-config error (e.g. missing credential)."""

    def __init__(self) -> None:
        from standard_asr.contract.exceptions import ConfigError

        raise ConfigError("missing API key for /secret/internal/path")


def _config_error_construct_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _ConfigErrorOnConstructASR
):
    return _ConfigErrorOnConstructASR()


class _ConfigurationRequiredASR(_DummyASR):
    """Construction raises the narrow absence subtype (credential not set).

    The message deliberately names the absent field and its env var -- exactly
    the deployment detail that must reach the OPERATOR log but never the
    client response/frame.
    """

    def __init__(self) -> None:
        from standard_asr.contract.exceptions import ConfigurationRequiredError

        raise ConfigurationRequiredError(
            "api_key is required; set STANDARD_ASR_DUMMY__API_KEY or pass it explicitly"
        )


def _configuration_required_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _ConfigurationRequiredASR
):
    return _ConfigurationRequiredASR()


class _ValidationErrorOnConstructASR(_DummyASR):
    """Construction raises a pydantic ValidationError whose echoed ``input``
    is SERVER-side material (a malformed credential from config/env)."""

    def __init__(self) -> None:
        from pydantic import BaseModel

        class _EnvConfig(BaseModel):
            api_key: int

        _EnvConfig.model_validate({"api_key": "sk-SERVER-ENV-SECRET"})


def _validation_error_construct_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _ValidationErrorOnConstructASR
):
    return _ValidationErrorOnConstructASR()


class _ValidationErrorTranscribeASR(_DummyASR):
    """``transcribe`` raises a pydantic ValidationError echoing a mis-placed
    secret (engine-side params re-validation)."""

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        from pydantic import BaseModel

        class _EngineParams(BaseModel):
            beam: int

        _EngineParams.model_validate({"beam": "sk-ENGINE-SIDE-SECRET"})
        return TranscriptionResult(text="unreachable")  # pragma: no cover


def _validation_error_transcribe_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _ValidationErrorTranscribeASR
):
    return _ValidationErrorTranscribeASR()


class _ValidationErrorStartASR(_DummyASR):
    """``start_transcription`` raises a pydantic ValidationError echoing an
    engine-side secret (a structural engine's internal model failing)."""

    def start_transcription(self, **kwargs: Any) -> Any:
        from pydantic import BaseModel

        class _EngineSessionParams(BaseModel):
            beam: int

        _EngineSessionParams.model_validate({"beam": "sk-ENGINE-SIDE-SECRET"})
        raise AssertionError("unreachable")  # pragma: no cover


def _validation_error_start_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _ValidationErrorStartASR
):
    return _ValidationErrorStartASR()


class _BareValueErrorStartASR(_DummyASR):
    """``start_transcription`` raises a bare ``ValueError`` carrying internal text.

    Models a structural engine (or adapter/SDK) bug surfacing past request
    validation -- NOT a caller-fixable rejection (a compliant engine signals
    those with ``UnsupportedFeatureError``).
    """

    def start_transcription(self, **kwargs: Any) -> Any:
        """Raise a bare engine-internal ``ValueError``.

        Args:
            **kwargs: Ignored.

        Returns:
            Never returns.

        Raises:
            ValueError: Always, with engine-internal (secret-bearing) text.
        """
        raise ValueError("internal adapter state invalid: token sk-WS-INTERNAL-SECRET")


def _bare_value_error_start_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _BareValueErrorStartASR
):
    return _BareValueErrorStartASR()


class _StaleResultProperties(BaseProperties):
    engine_id: str = "dummy"
    model_name: str = "echo"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = "any"
    selectable_languages: list[str] = []


class _StaleResultEngine(EngineBase):
    """A real ``EngineBase`` whose ``_transcribe`` builds an invalid result.

    Models a plugin built against an older core: it still passes the removed
    blanket ``metadata`` field, which the ``extra="forbid"`` result model
    rejects with a pydantic ``ValidationError``.
    """

    properties: ClassVar[BaseProperties] = _StaleResultProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        batch=BatchCapabilities()
    )

    def __init__(self) -> None:
        self.config = _DummyConfig(engine="dummy")

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        return TranscriptionResult.model_validate({"text": "x", "metadata": {"cost": 1}})


def _stale_result_factory() -> _StaleResultEngine:  # pyright: ignore[reportUnusedFunction]
    return _StaleResultEngine()


# --- Real EngineBase engines that record what negotiation hands them ----------

#: Set by the recording engines' ``_transcribe`` so tests can assert on the
#: shape/rate/bytes the standard negotiation actually produced.
_RECORDED: dict[str, Any] = {}

_REC_CAPS = DeclaredCapabilities(batch=BatchCapabilities())


def _wav_bytes(rate: int, samples: int = 1600) -> bytes:
    """Return a minimal mono 16-bit PCM WAV at ``rate`` Hz."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(np.zeros(samples, dtype=np.int16).tobytes())
    return buf.getvalue()


class _Array8kProperties(BaseProperties):
    engine_id: str = "rec"
    model_name: str = "array8k"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 8000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [8000]
    selectable_languages: list[str] = []


class _RecordingArray8kASR(EngineBase):
    """8 kHz-native engine: an 8 kHz upload must reach it at 8 kHz, never
    silently up-sampled to 16 kHz."""

    properties: ClassVar[BaseProperties] = _Array8kProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _REC_CAPS

    def __init__(self) -> None:
        self.config = _DummyConfig(engine="rec")

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        _RECORDED["kind"] = prepared.kind
        _RECORDED["sample_rate"] = prepared.sample_rate
        _RECORDED["array_len"] = int(prepared.array.size) if prepared.array is not None else None
        return TranscriptionResult(text="array8k")


def _recording_array8k_factory() -> _RecordingArray8kASR:  # pyright: ignore[reportUnusedFunction]
    return _RecordingArray8kASR()


class _EncodedProperties(BaseProperties):
    engine_id: str = "rec"
    model_name: str = "bytes"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ENCODED_BYTES}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = "any"
    selectable_languages: list[str] = []


class _RecordingEncodedASR(EngineBase):
    """Encoded-only engine: must be servable at all and must
    receive the original encoded bytes byte-for-byte (passthrough)."""

    properties: ClassVar[BaseProperties] = _EncodedProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _REC_CAPS

    def __init__(self) -> None:
        self.config = _DummyConfig(engine="rec")

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        _RECORDED["kind"] = prepared.kind
        _RECORDED["data"] = prepared.data
        return TranscriptionResult(text="bytes")


def _recording_encoded_factory() -> _RecordingEncodedASR:  # pyright: ignore[reportUnusedFunction]
    return _RecordingEncodedASR()


class _AutoLangProperties(BaseProperties):
    """Auto-detect engine with a BOUNDED detectable set (the 422 regression)."""

    engine_id: str = "rec"
    model_name: str = "autolang"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    selectable_languages: list[str] = ["auto"]
    detectable_languages: list[str] = ["en", "ja", "ko"]


class _AutoLangConfig(LanguageConfigMixin, BaseConfig[str]):
    engine: str = "rec"
    default_language: str | None = "auto"


_AUTO_LANG_CAPS = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(
            runtime_override=FlagCap(supported=True),
            candidate_languages=CandidateLanguagesCap(
                supported=True,
                constraints=CandidateLanguagesConstraints(max=2),
            ),
        )
    )
)


class _AutoLangASR(EngineBase):
    """Strict engine (config default) whose detectable set is bounded.

    A request naming a candidate outside ``detectable_languages`` is a CLIENT
    error the standard layer rejects in strict mode; it must reach the wire as a
    422, never a 500.
    """

    properties: ClassVar[BaseProperties] = _AutoLangProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _AUTO_LANG_CAPS

    def __init__(self) -> None:
        self.config = _AutoLangConfig(engine="rec")

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        _RECORDED["candidate_languages"] = params.candidate_languages
        return TranscriptionResult(text="autolang")


def _auto_lang_factory() -> _AutoLangASR:  # pyright: ignore[reportUnusedFunction]
    return _AutoLangASR()


class _RecordingOptionsASR(_DummyASR):
    """Records the params object the server built from the wire ``options``."""

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        _RECORDED["options"] = options
        return TranscriptionResult(text="dummy")


def _recording_options_factory() -> _RecordingOptionsASR:  # pyright: ignore[reportUnusedFunction]
    return _RecordingOptionsASR()


def _registry():
    eps = [
        EntryPoint(
            name="dummy/echo",
            value="tests.test_server:_dummy_factory",
            group="standard_asr.models",
        )
    ]
    return discover_models(eps=eps, strict=True)


def _registry_for(factory: str):
    eps = [
        EntryPoint(
            name="dummy/echo",
            value=f"tests.test_server:{factory}",
            group="standard_asr.models",
        )
    ]
    return discover_models(eps=eps, strict=True)


def test_create_app_missing_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "fastapi":
            raise ImportError("fastapi not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        server_module.create_app()


def test_create_app_empty_registry_exposes_no_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # An explicitly-passed empty ModelRegistry must expose ZERO models and MUST
    # NOT fall back to plugin discovery (a bare `registry or discover_models()`
    # would treat the len-0 registry as falsey and expose every installed
    # plugin instead -- the opposite of the operator's intent).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    def _boom() -> ModelRegistry:
        raise AssertionError("discover_models() must not be called for an explicit registry")

    monkeypatch.setattr(server_module, "discover_models", _boom)

    app = server_module.create_app(registry=ModelRegistry({}))
    client = TestClient(app)
    resp: httpx2.Response = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_app_endpoints() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    response: httpx2.Response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    models: httpx2.Response = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()[0]["key"] == "dummy/echo"

    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode("utf-8"),
    }
    transcribe: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert transcribe.status_code == 200
    assert transcribe.json()["result"]["text"] == "dummy"


def test_server_array_engine_keeps_native_rate_through_negotiation() -> None:
    # An 8 kHz upload to an 8 kHz-native engine must arrive as an ARRAY at
    # 8000 Hz -- proving the server routes through negotiation and never forces
    # the old unconditional 16 kHz resample.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _RECORDED.clear()
    app = server_module.create_app(registry=_registry_for("_recording_array8k_factory"))
    client = TestClient(app)

    files = {"file": ("audio.wav", _wav_bytes(rate=8000), "audio/wav")}
    resp: httpx2.Response = client.post("/v1/transcribe", data={"model": "dummy/echo"}, files=files)
    assert resp.status_code == 200
    assert _RECORDED["kind"] is InputKind.ARRAY
    assert _RECORDED["sample_rate"] == 8000


def test_server_encoded_engine_receives_original_bytes_multipart() -> None:
    # An encoded-only engine must be servable and receive the
    # uploaded bytes verbatim (passthrough, no lossy decode/re-encode).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _RECORDED.clear()
    wav = _wav_bytes(rate=16000)
    app = server_module.create_app(registry=_registry_for("_recording_encoded_factory"))
    client = TestClient(app)

    files = {"file": ("audio.wav", wav, "audio/wav")}
    resp: httpx2.Response = client.post("/v1/transcribe", data={"model": "dummy/echo"}, files=files)
    assert resp.status_code == 200
    assert _RECORDED["kind"] is InputKind.ENCODED_BYTES
    assert _RECORDED["data"] == wav


def test_server_encoded_engine_receives_original_bytes_json() -> None:
    # The JSON (base64) endpoint feeds the same negotiation path.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _RECORDED.clear()
    wav = _wav_bytes(rate=16000)
    app = server_module.create_app(registry=_registry_for("_recording_encoded_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(wav).decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 200
    assert _RECORDED["kind"] is InputKind.ENCODED_BYTES
    assert _RECORDED["data"] == wav


def test_transcribe_json_decode_error_maps_to_400() -> None:
    # Invalid base64 reaching a real engine fails inside negotiation and maps
    # to 400 (no pre-decode in the endpoint any more).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_recording_array8k_factory"))
    client = TestClient(app)

    resp: httpx2.Response = client.post(
        "/v1/transcribe:json", json={"model": "dummy/echo", "audio": "not-valid-base64!!!"}
    )
    assert resp.status_code == 400


def test_transcribe_file_decode_error_maps_to_400() -> None:
    # Undecodable upload bytes fail in negotiation -> 400.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_recording_array8k_factory"))
    client = TestClient(app)

    files = {"file": ("audio.wav", b"this is not audio", "audio/wav")}
    resp: httpx2.Response = client.post("/v1/transcribe", data={"model": "dummy/echo"}, files=files)
    assert resp.status_code == 400


def test_transcribe_json_internal_error_maps_to_500() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_fail_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500


def test_transcribe_file_success_and_internal_error() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    data = {"model": "dummy/echo"}
    response: httpx2.Response = client.post("/v1/transcribe", data=data, files=files)
    assert response.status_code == 200
    assert response.json()["result"]["text"] == "dummy"

    app_fail = server_module.create_app(registry=_registry_for("_fail_factory"))
    client_fail = TestClient(app_fail)
    response = client_fail.post("/v1/transcribe", data=data, files=files)
    assert response.status_code == 500


def test_transcribe_body_is_the_single_wire_projection_verbatim() -> None:
    # The endpoint encodes the wire document ONCE, inside the fault-mapping
    # region, and the route sends that body verbatim. The previous shape --
    # a full dump+encode as a projectability PROOF whose output was
    # discarded, followed by FastAPI validating and serializing the same
    # response again -- roughly doubled response-serialization CPU and peak
    # memory on every successful request (tens of MB of words[] for a long
    # recording). Byte-equality against _run_transcription's own output is
    # the pin: a second serialization layer produces its own encoding, not
    # this one.
    import asyncio

    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    from standard_asr.audio.input import AudioBytes

    registry = _registry()
    expected_body = asyncio.run(
        server_module._run_transcription(  # pyright: ignore[reportPrivateUsage]
            registry, "dummy/echo", AudioBytes(data=b"fake"), None, HTTPException
        )
    )
    assert isinstance(expected_body, str)
    assert json.loads(expected_body)["result"]["text"] == "dummy"

    app = server_module.create_app(registry=registry)
    client = TestClient(app)
    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    response: httpx2.Response = client.post(
        "/v1/transcribe", data={"model": "dummy/echo"}, files=files
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.content == expected_body.encode()

    # The wire encoding matches what the framework serializer shipped before
    # the endpoint owned the projection: compact separators, and non-ASCII
    # transcript text as UTF-8 -- not 6-byte \uXXXX escapes, which inflated
    # CJK/Cyrillic/Arabic bodies ~1.35-3x.
    assert b'": ' not in response.content  # compact separators
    cjk_body = asyncio.run(
        server_module._run_transcription(  # pyright: ignore[reportPrivateUsage]
            _registry_for("_cjk_factory"),
            "dummy/echo",
            AudioBytes(data=b"fake"),
            None,
            HTTPException,
        )
    )
    assert "中文轉錄測試" in cjk_body  # UTF-8 passthrough...
    assert "\\u4e2d" not in cjk_body  # ...never an ASCII escape


def test_run_handles_missing_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "uvicorn":
            raise ImportError("uvicorn not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        server_module.run()


def test_run_calls_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    uvicorn_stub = types.ModuleType("uvicorn")
    setattr(uvicorn_stub, "called", False)
    setattr(uvicorn_stub, "kwargs", {})

    def _run(app: Any, **kwargs: Any) -> None:
        setattr(uvicorn_stub, "called", True)
        setattr(uvicorn_stub, "kwargs", kwargs)

    uvicorn_stub.run = _run  # type: ignore[attr-defined]

    monkeypatch.setitem(__import__("sys").modules, "uvicorn", uvicorn_stub)

    create_app_kwargs: dict[str, Any] = {}

    def _create_app(**kwargs: Any) -> str:
        create_app_kwargs.update(kwargs)
        return "app"

    monkeypatch.setattr(server_module, "create_app", _create_app)

    server_module.run(
        host="127.0.0.1",
        port=9999,
        log_level="warning",
        max_ws_frame_bytes=4096,
    )

    assert getattr(uvicorn_stub, "called") is True
    kwargs = getattr(uvicorn_stub, "kwargs")
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9999
    # The WS per-frame cap is wired to uvicorn's transport ws_max_size so the
    # app-level bound and the transport bound match.
    assert kwargs["ws_max_size"] == 4096
    # The same cap is propagated to the app it builds.
    assert create_app_kwargs["max_ws_frame_bytes"] == 4096


def test_capabilities_endpoint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    resp: httpx2.Response = client.get("/v1/capabilities/dummy/echo")
    assert resp.status_code == 200
    body = resp.json()
    assert body["batch"]["language"]["runtime_override"]["supported"] is True


def test_capabilities_endpoint_unknown_model() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    resp: httpx2.Response = client.get("/v1/capabilities/nope/missing")
    assert resp.status_code == 404


def test_params_schema_endpoint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    resp: httpx2.Response = client.get("/v1/params-schema/dummy/echo")
    assert resp.status_code == 200
    schema = resp.json()
    assert "beam" in schema.get("properties", {})


def test_transcribe_client_error_maps_to_422() -> None:
    """UnsupportedFeatureError (client-caused) must map to 422, not 500."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_client_error_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422
    assert "word_timestamps" in resp.json()["detail"]


def test_transcribe_unknown_model_maps_to_404() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    payload = {"model": "nope/missing", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 404


def test_transcribe_500_does_not_leak_internal_detail() -> None:
    """Unexpected errors return a generic message; raw text stays server-side."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_fail_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "/secret/internal/path" not in detail
    assert "See server logs" in detail


def test_transcribe_construction_config_error_maps_to_500_scrubbed() -> None:
    """A plain ConfigError at ZERO-ARG construction is a deployment fault -> 500.

    The client cannot see, reach, or fix the server's engine configuration
    (construction takes nothing from the request but the model key), and
    compliance classifies this exact state as `engine_construction_failed`.
    The old 422 blamed the caller AND surfaced server-side config detail.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_config_error_construct_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "missing API key" not in detail
    assert "/secret/internal/path" not in resp.text
    assert "See server logs" in detail


def test_transcribe_construction_missing_config_maps_to_503(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``ConfigurationRequiredError`` at construction -> 503, field names withheld.

    Absent required config is an operator-side availability state (the state
    compliance SKIPS): the model exists but is not usable on this deployment.
    The stable generic detail never names the absent field -- deployment
    detail belongs in the operator log, which gets it via safe logging.

        Args:
            caplog: Pytest fixture capturing log records.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_configuration_required_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    with caplog.at_level(logging.ERROR, logger="standard_asr.toolchain.server"):
        resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "server-side configuration" in detail
    assert "api_key" not in resp.text
    assert "STANDARD_ASR_DUMMY__API_KEY" not in resp.text
    # The operator DOES get the specifics, in the log.
    assert any("requires configuration absent" in r.getMessage() for r in caplog.records)


def test_transcribe_construction_unexpected_error_maps_to_500_no_leak() -> None:
    """An unexpected construction fault -> generic 500 with no internal detail."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # _NoInstantiateASR.__init__ raises a RuntimeError carrying internal text.
    app = server_module.create_app(registry=_registry_for("_no_instantiate_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "instantiation forbidden" not in detail
    assert "See server logs" in detail


def test_transcribe_file_construction_config_error_maps_to_500() -> None:
    """The multipart endpoint maps construction config errors identically."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_config_error_construct_factory"))
    client = TestClient(app)

    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    resp: httpx2.Response = client.post("/v1/transcribe", data={"model": "dummy/echo"}, files=files)
    assert resp.status_code == 500


def test_transcribe_construction_validation_error_maps_to_500_scrubbed() -> None:
    """A pydantic ValidationError at construction -> scrubbed 500, nothing echoed.

    At construction time both the echoed ``input_value`` AND the failing field
    names are server-side config/env material; a zero-arg construction fault
    is a deployment/plugin defect the caller cannot fix, so nothing about it
    belongs in the response.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_validation_error_construct_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500
    assert "sk-SERVER-ENV-SECRET" not in resp.text
    assert "api_key" not in resp.text
    assert "See server logs" in resp.json()["detail"]


class _LazyCredTranscribeASR(_DummyASR):
    """Defers its credential check to transcribe() (lazy absence)."""

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        from standard_asr.contract.exceptions import ConfigurationRequiredError

        raise ConfigurationRequiredError(
            "api_key is required; set STANDARD_ASR_DUMMY__API_KEY or pass it explicitly"
        )


def _lazy_cred_transcribe_factory() -> _LazyCredTranscribeASR:  # pyright: ignore[reportUnusedFunction]
    return _LazyCredTranscribeASR()


class _ConfigErrorTranscribeASR(_DummyASR):
    """Raises a declaration ConfigError at transcribe() (engine defect)."""

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        from standard_asr.contract.exceptions import ConfigError

        raise ConfigError("default_language 'xx' is not in selectable_languages ['en']")


def _config_error_transcribe_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _ConfigErrorTranscribeASR
):
    return _ConfigErrorTranscribeASR()


class _IPPETranscribeASR(_DummyASR):
    """Raises InvalidProviderParamError at transcribe() (engine contract bug)."""

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        from standard_asr.contract.exceptions import InvalidProviderParamError

        raise InvalidProviderParamError("foreign provider_params: beam='sk-PROVIDER-SECRET'")


def _ippe_transcribe_factory() -> _IPPETranscribeASR:  # pyright: ignore[reportUnusedFunction]
    return _IPPETranscribeASR()


def test_transcribe_lazy_missing_config_maps_to_503() -> None:
    """A lazy ``ConfigurationRequiredError`` at transcribe() is the 503 state.

    Absence is operator-side wherever it surfaces; the caller gets the same
    stable generic 503 as at construction, never the field names.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_lazy_cred_transcribe_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 503
    assert "server-side configuration" in resp.json()["detail"]
    assert "api_key" not in resp.text
    assert "STANDARD_ASR_DUMMY__API_KEY" not in resp.text


def test_transcribe_engine_config_error_maps_to_500_scrubbed() -> None:
    """A transcription-time ``ConfigError`` is an ENGINE fault: scrubbed 500.

    The wire client cannot supply engine config, so a `ConfigError` here is a
    declaration defect whose authored message may carry server-side config
    detail. The old 422 blamed the request and surfaced that text; client-
    fixable language rejections have their own type (`UnsupportedFeatureError`,
    still 422 -- pinned by ``test_transcribe_client_error_maps_to_422``).
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_config_error_transcribe_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500
    assert "default_language" not in resp.text
    assert "See server logs" in resp.json()["detail"]


def test_transcribe_provider_param_error_maps_to_500_scrubbed() -> None:
    """``InvalidProviderParamError`` from the wire path is an engine fault.

    ``WireRuntimeParams`` rejects ``provider_params`` before transcription,
    so no wire client can legally trigger this error -- it escaping
    transcribe() means the server/engine contract broke. Scrubbed 500.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_ippe_transcribe_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500
    assert "sk-PROVIDER-SECRET" not in resp.text
    assert "See server logs" in resp.json()["detail"]


def test_transcribe_engine_validation_error_maps_to_500_scrubbed() -> None:
    """A bare ValidationError escaping a structural engine's transcribe() is a 500.

    By the time the engine runs, the client's options were already validated
    (WireRuntimeParams at the route) and promoted to typed RuntimeParams -- a
    pydantic ValidationError escaping transcribe() can only come from the
    engine's own internals (here: the fixture's engine-side model fed an
    engine-side value the client never sent). The old mapping returned 422
    anchored under ["options"], blaming the client's request for a plugin
    fault -- and made fault ownership depend on whether the engine inherits
    EngineBase (whose template seam wraps the same fault into a 500). Same
    fault domain, same verdict: scrubbed 500, no field detail, no echo.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_validation_error_transcribe_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal transcription error. See server logs for details."
    assert "sk-ENGINE-SIDE-SECRET" not in resp.text
    # The client's options must not be blamed for an engine-internal fault.
    assert "options" not in resp.text
    assert "beam" not in resp.text


def test_transcribe_engine_validation_error_never_reaches_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The operator log is a transport too: the input echo must not land there.

    The client response was already scrubbed; pre-fix ``logger.exception``
    still wrote the raw ``ValidationError`` -- ``input_value=`` echo included
    -- into the server log (and thus CI logs / aggregators). The safe-logging
    path keeps the fault structure but never the value.

        Args:
            caplog: Pytest fixture capturing log records.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_validation_error_transcribe_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    with caplog.at_level(logging.ERROR, logger="standard_asr.toolchain.server"):
        resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)

    assert resp.status_code == 500
    assert caplog.records  # the fault IS logged for operators...
    assert "sk-ENGINE-SIDE-SECRET" not in caplog.text  # ...without the echoed input
    assert "input_value" not in caplog.text
    assert any("ValidationError" in record.getMessage() for record in caplog.records)


def test_ws_engine_validation_error_never_reaches_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The WS establishment path's operator log is scrubbed the same way.

    Args:
        caplog: Pytest fixture capturing log records.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_validation_error_start_factory"))
    client = TestClient(app)
    with caplog.at_level(logging.ERROR, logger="standard_asr.toolchain.server"):
        with client.websocket_connect("/v1/stream/dummy/echo") as ws:
            ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
            err = ws.receive_json()

    assert err["code"] == "internal_error"
    assert caplog.records
    assert "sk-ENGINE-SIDE-SECRET" not in caplog.text
    assert "input_value" not in caplog.text


def test_transcribe_engine_built_invalid_result_maps_to_500_not_422() -> None:
    """An invalid result built INSIDE ``_transcribe`` is a 500, never a 422.

    Regression: a pydantic ``ValidationError`` escaping ``_transcribe`` used to
    reach the server's ``ValidationError`` clause and return 422 with a detail
    anchored under ``["options"]`` -- telling the client its own request was
    malformed when the fault was entirely the engine's (here: a plugin still
    sending the removed ``metadata`` field). ``EngineBase.transcribe`` now wraps
    it as a ``TranscriptionError``, so it lands on the generic internal path:
    HTTP 500 with the stable non-leaking message.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stale_result_factory"))
    client = TestClient(app)

    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(_wav_bytes(16000)).decode(),
    }
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal transcription error. See server logs for details."
    # The client's `options` must not be blamed, and no pydantic field detail leaks.
    assert "options" not in resp.text
    assert "metadata" not in resp.text


def test_transcribe_async_engine_maps_to_500_scrubbed() -> None:
    """A sync-wrapper ``transcribe`` handing back a coroutine is a scrubbed 500.

    ``asyncio.to_thread`` returns the RAW value, so an ``async def``
    implementation (or a sync wrapper over one) yields a coroutine object the
    route would previously have fed into ``TranscribeResponse`` OUTSIDE the
    fault-mapping try -- a raw pydantic ``ValidationError`` escaping the
    route. The sync-call boundary must classify it as an engine fault (the
    generic scrubbed 500) with the stray coroutine closed, not leaked.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_async_transcribe_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal transcription error. See server logs for details."
    assert "coroutine" not in resp.text  # no raw pydantic detail escapes


def test_transcribe_wrong_result_type_maps_to_500_scrubbed() -> None:
    """A non-``TranscriptionResult`` return is an engine fault: scrubbed 500.

    Pre-fix, the dict reached ``TranscribeResponse(...)`` outside the try and
    escaped as a raw ``ValidationError`` (500 with pydantic's input echo via
    the framework's default handler, or worse). The boundary pins the type
    inside the fault-mapping region.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_wrong_result_type_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal transcription error. See server logs for details."
    assert "dict-not-result" not in resp.text  # the engine's value never echoes


def test_ws_stream_async_start_maps_to_internal_error_frame() -> None:
    """A coroutine from ``start_transcription`` is a scrubbed internal_error.

    Pre-fix the coroutine was treated as a session: the route crashed on
    ``session.diagnostics()`` AFTER the establishment error-mapping block
    (an ``AttributeError`` escaping the route) while the coroutine leaked
    never-awaited. The boundary must classify it inside the block -- an
    engine fault, scrubbed frame, coroutine closed.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_async_start_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "internal_error"
    assert "See server logs" in err["message"]
    assert "coroutine" not in err["message"]


def test_ws_stream_construction_missing_config_reports_service_unavailable() -> None:
    """``ConfigurationRequiredError`` at construction: the WS twin of the REST 503.

    An operator-side availability state -- a stable generic frame, never the
    absent field names or env vars.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_configuration_required_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "service_unavailable"
    assert "server-side configuration" in err["message"]
    assert "api_key" not in err["message"]
    assert "STANDARD_ASR_DUMMY__API_KEY" not in err["message"]


def test_ws_stream_construction_config_error_reports_internal_error() -> None:
    """A plain ConfigError at zero-arg construction is a deployment fault.

    The WS twin of the REST construction 500: the old ``bad_request`` frame
    blamed the caller (and surfaced server-side config text) for a fault the
    caller cannot see or fix.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_config_error_construct_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "internal_error"
    assert "missing API key" not in err["message"]
    assert "/secret/internal/path" not in err["message"]
    assert "See server logs" in err["message"]


def test_ws_stream_construction_validation_error_reports_internal_error() -> None:
    """A construction-time pydantic ValidationError: scrubbed internal_error.

    Both the echoed input AND the field names are server-side config/env
    material; nothing about a zero-arg construction fault belongs in a client
    frame.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_validation_error_construct_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "internal_error"
    assert "sk-SERVER-ENV-SECRET" not in err["message"]
    assert "api_key" not in err["message"]
    assert "See server logs" in err["message"]


def test_ws_stream_engine_validation_error_is_internal_and_never_echoes() -> None:
    """A structural engine's ValidationError at establishment: scrubbed internal_error.

    pydantic ``ValidationError`` subclasses ``ValueError``, so without a
    dedicated clause it fell into the ``(UnsupportedFeatureError, ValueError)``
    handler -- labelled ``unsupported`` (misattributing an engine fault to the
    client's request) and echoed via ``str(exc)``, whose text includes
    pydantic's offending ``input_value`` (here an engine-side secret crossing
    the trust boundary). The dedicated clause must classify it as a scrubbed
    ``internal_error``, mirroring the EngineBase template seam's wrap.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_validation_error_start_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "internal_error"
    assert err["code"] != "unsupported"
    assert "sk-ENGINE-SIDE-SECRET" not in err["message"]
    assert "beam" not in err["message"]
    assert "See server logs" in err["message"]


def test_ws_stream_bare_value_error_is_internal_and_never_echoes() -> None:
    """A bare engine ``ValueError`` at establishment: scrubbed ``internal_error``.

    By establishment every client input is already validated (config frame,
    ``WireRuntimeParams``, ``AudioFormat``), and a compliant engine signals
    unsupported features with ``UnsupportedFeatureError`` -- so a surviving
    bare ``ValueError`` is an engine/adapter fault. The old
    ``(UnsupportedFeatureError, ValueError)`` arm labelled it ``unsupported``
    (blaming the caller) and sent ``str(exc)`` -- engine-internal, possibly
    credential-bearing text -- to an unauthenticated client.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_bare_value_error_start_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "internal_error"
    assert err["code"] != "unsupported"
    assert "sk-WS-INTERNAL-SECRET" not in err["message"]
    assert "internal adapter state" not in err["message"]
    assert "See server logs" in err["message"]


def test_ws_stream_unsupported_feature_error_keeps_authored_message() -> None:
    """``UnsupportedFeatureError`` stays the caller-facing ``unsupported`` frame.

    Narrowing the arm to the dedicated type must not lose the one genuinely
    caller-fixable establishment rejection or its authored message.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # _RecordingArray8kASR is batch-only: EngineBase's start_transcription
    # raises UnsupportedFeatureError with an authored refusal message.
    app = server_module.create_app(registry=_registry_for("_recording_array8k_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "unsupported"
    assert err["message"]  # the authored refusal text survives


def test_ws_stream_construction_unexpected_error_reports_internal_no_leak() -> None:
    # An unexpected construction fault must not crash the route or leak detail:
    # a single generic internal_error frame is sent instead.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_no_instantiate_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "internal_error"
    assert "instantiation forbidden" not in err["message"]
    assert "See server logs" in err["message"]


def test_body_size_limit_returns_413() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry(), max_body_bytes=64)
    client = TestClient(app)

    big = base64.b64encode(b"x" * 1024).decode()
    payload = {"model": "dummy/echo", "audio": big}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 413


def test_create_app_rejects_nonpositive_max_body() -> None:
    pytest.importorskip("fastapi")
    with pytest.raises(ValueError):
        server_module.create_app(registry=_registry(), max_body_bytes=0)


class _AudioErrorASR(_DummyASR):
    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        from standard_asr.contract.exceptions import AudioProcessingError

        raise AudioProcessingError("bad audio frames")


def _audio_error_factory() -> _AudioErrorASR:  # pyright: ignore[reportUnusedFunction]
    return _AudioErrorASR()


class _NoCapsASR(_DummyASR):
    declared_capabilities: ClassVar[DeclaredCapabilities | None] = None  # type: ignore[assignment]
    provider_params_type: ClassVar[type[ProviderParams] | None] = None


#: Sentinel that must never reach the operator log through a metadata read.
_METADATA_SECRET = "sk-METADATA-SENTINEL"  # noqa: S105 - test fixture


class _HostileCapsMeta(type):
    """A metaclass property: the descriptor the endpoints dispatch on read."""

    @property
    def declared_capabilities(cls) -> Any:
        """Fail the way a plugin's lazily-built capability tree can.

        Returns:
            Never returns.

        Raises:
            ValidationError: Always -- carrying an input echo.
        """
        raise _capability_validation_error()


def _capability_validation_error() -> Any:
    """Build a ValidationError whose rendering echoes the sentinel.

    Returns:
        The pydantic error.
    """
    from pydantic import BaseModel, ValidationError

    class _Strict(BaseModel):
        api_key: int

    try:
        _Strict(api_key=_METADATA_SECRET)  # pyright: ignore[reportArgumentType]
    except ValidationError as exc:
        return exc
    raise AssertionError("the model was expected to reject the input")


class _HostileCapsASR(_DummyASR, metaclass=_HostileCapsMeta):
    pass


def _hostile_caps_factory() -> _HostileCapsASR:  # pyright: ignore[reportUnusedFunction]
    return _HostileCapsASR()


class _RaisingSchemaParams(ProviderParams):
    """A params type whose JSON-schema generation raises."""

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Fail the way a custom __get_pydantic_json_schema__ hook can.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            Never returns.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("schema generation exploded")


class _RaisingSchemaASR(_DummyASR):
    provider_params_type: ClassVar[type[ProviderParams] | None] = _RaisingSchemaParams


def _raising_schema_factory() -> _RaisingSchemaASR:  # pyright: ignore[reportUnusedFunction]
    return _RaisingSchemaASR()


class _UnencodableCaps:
    """Declared capabilities whose canonical JSON is not JSON at all."""

    def canonical_json(self) -> dict[str, Any]:
        """Return a payload no conforming JSON parser accepts.

        Returns:
            A dict carrying a non-finite number.
        """
        return {"limit": float("inf")}


class _UnencodableCapsASR(_DummyASR):
    declared_capabilities: ClassVar[Any] = _UnencodableCaps()


def _unencodable_caps_factory() -> _UnencodableCapsASR:  # pyright: ignore[reportUnusedFunction]
    return _UnencodableCapsASR()


class _HostileProtocolMeta(type):
    """Metaclass whose ``_is_protocol`` read fails during CLASS RESOLUTION."""

    @property
    def _is_protocol(cls) -> Any:  # pyright: ignore[reportGeneralTypeIssues]
        """Fail the way a plugin's resolution-time descriptor can.

        Returns:
            Never returns.

        Raises:
            ValidationError: Always -- carrying an input echo.
        """
        raise _capability_validation_error()


class _HostileTranscribeMeta(type):
    """Metaclass whose ``transcribe`` read fails during CLASS RESOLUTION."""

    @property
    def transcribe(cls) -> Any:  # pyright: ignore[reportGeneralTypeIssues]
        """Fail the duck-type read the resolution performs.

        Returns:
            Never returns.

        Raises:
            ValidationError: Always -- carrying an input echo.
        """
        raise _capability_validation_error()


class _HostileProtocolASR(_DummyASR, metaclass=_HostileProtocolMeta):
    pass


class _HostileTranscribeASR(_DummyASR, metaclass=_HostileTranscribeMeta):
    pass


def _hostile_protocol_factory() -> _HostileProtocolASR:  # pyright: ignore[reportUnusedFunction]
    return _HostileProtocolASR()


def _hostile_transcribe_factory() -> _HostileTranscribeASR:  # pyright: ignore[reportUnusedFunction]
    return _HostileTranscribeASR()


def _no_caps_factory() -> _NoCapsASR:  # pyright: ignore[reportUnusedFunction]
    return _NoCapsASR()


def test_transcribe_audio_error_maps_to_400() -> None:
    # AudioProcessingError raised inside transcribe maps to 400.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_audio_error_factory"))
    client = TestClient(app)

    payload = {"model": "dummy/echo", "audio": base64.b64encode(b"fake").decode()}
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 400
    assert "bad audio frames" in resp.json()["detail"]


def test_transcribe_json_with_options_builds_params() -> None:
    # A non-null options object is parsed into RuntimeParams (the _build_params
    # validate path).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode(),
        "options": {"language": "en"},
    }
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 200


def test_transcribe_options_accept_diarization_marker() -> None:
    # {"diarization": {}} on the wire reaches the engine as the empty frozen
    # marker (the three-way mapping: {} -> DiarizationRequest()).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _RECORDED.clear()
    app = server_module.create_app(registry=_registry_for("_recording_options_factory"))
    client = TestClient(app)
    payload: dict[str, Any] = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode(),
        "options": {"diarization": {}},
    }
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 200
    options = _RECORDED["options"]
    assert isinstance(options, RuntimeParams)
    assert options.diarization == DiarizationRequest()


def test_transcribe_options_diarization_unknown_key_maps_to_422() -> None:
    # The marker is a closed type (extra="forbid"): an unknown nested key --
    # e.g. a not-yet-graduated num_speakers -- is rejected, never silently
    # dropped (it would otherwise read as a satisfied request).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_recording_options_factory"))
    client = TestClient(app)
    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode(),
        "options": {"diarization": {"num_speakers": 2}},
    }
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422


def test_strict_non_detectable_candidate_language_maps_to_422_not_500() -> None:
    """REGRESSION: a strict engine asked for a well-formed candidate language it
    cannot detect is a CLIENT error. The standard layer used to raise a bare
    ValueError here, which fell through the server's client-error clauses into
    the catch-all and answered 500 "Internal transcription error" -- blaming
    the server for the caller's request and hiding the fixable reason. It is
    now the standard strict-gate UnsupportedFeatureError -> 422 with the
    offending parameter named in the detail.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_auto_lang_factory"))
    client = TestClient(app)

    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(_wav_bytes(rate=16000)).decode(),
        "options": {"language": "auto", "candidate_languages": ["zz"]},
    }
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)

    assert resp.status_code == 422
    detail = str(resp.json()["detail"])
    # The detail names the offending tag and the reason, so the caller can fix it.
    assert "'zz'" in detail
    assert "detectable" in detail
    # The generic internal-error text must NOT appear: this is the caller's bug.
    assert "Internal transcription error" not in detail


def test_strict_over_max_candidate_languages_maps_to_422() -> None:
    """The sibling strict rejection (more candidates than the declared max=2)
    travels the same path to 422.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_auto_lang_factory"))
    client = TestClient(app)

    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(_wav_bytes(rate=16000)).decode(),
        "options": {"language": "auto", "candidate_languages": ["en", "ja", "ko"]},
    }
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)

    assert resp.status_code == 422
    detail = str(resp.json()["detail"])
    assert "3 entries" in detail and "max is 2" in detail
    assert "Internal transcription error" not in detail


def test_detectable_candidate_languages_still_transcribe() -> None:
    """The mirror case: candidates the engine CAN detect are accepted and reach
    the engine, so the 422 above is a real rejection and not a blanket block.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _RECORDED.clear()
    app = server_module.create_app(registry=_registry_for("_auto_lang_factory"))
    client = TestClient(app)

    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(_wav_bytes(rate=16000)).decode(),
        "options": {"language": "auto", "candidate_languages": ["en", "ja"]},
    }
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)

    assert resp.status_code == 200
    assert resp.json()["result"]["text"] == "autolang"
    assert _RECORDED["candidate_languages"] == ["en", "ja"]


def test_transcribe_json_with_bad_options_maps_to_422() -> None:
    # A semantically invalid options object in the JSON body (a malformed
    # language tag) is an unprocessable entity (422), not a malformed-syntax 400.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode(),
        "options": {"language": "english"},
    }
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422


def test_transcribe_file_with_bad_options_maps_to_400() -> None:
    # A malformed options JSON string in the multipart form is a client error.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    data = {"model": "dummy/echo", "options": "{not json}"}
    resp: httpx2.Response = client.post("/v1/transcribe", data=data, files=files)
    assert resp.status_code == 400


def test_body_validation_error_does_not_echo_input() -> None:
    # A body-validation failure (here: wrong type for `audio`) must NOT reflect
    # the offending submitted value -- FastAPI's default handler would echo it.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    # `audio` must be a string; send a recognisable sentinel as the wrong type.
    resp: httpx2.Response = client.post(
        "/v1/transcribe:json", json={"model": "dummy/echo", "audio": 1234567890}
    )
    assert resp.status_code == 422
    assert "1234567890" not in resp.text


def test_body_validation_error_redacts_credential_field_value() -> None:
    # A mis-placed secret (an `api_key` put at the top level of the JSON body)
    # is rejected by extra="forbid"; its value must be redacted, never bounced
    # back to the client / proxy / a copied bug report.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    secret = "sk-LEAKED-SECRET-VALUE"
    resp: httpx2.Response = client.post(
        "/v1/transcribe:json",
        json={"model": "dummy/echo", "audio": "Zm9v", "api_key": secret},
    )
    assert resp.status_code == 422
    assert secret not in resp.text
    detail = resp.json()["detail"]
    # The credential entry is present (so the caller knows what to fix) but its
    # message is redacted.
    assert any("api_key" in entry["loc"] for entry in detail)
    assert any(entry["msg"] == "[redacted]" for entry in detail)


def test_options_validation_error_does_not_echo_secret() -> None:
    # A secret mis-placed inside `options` reaches _build_params, whose pydantic
    # str(exc) would otherwise echo input_value=. The sanitized message must not
    # contain it (and the offending field is named).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    secret = "sk-OPTIONS-SECRET"
    payload = {
        "model": "dummy/echo",
        "audio": "Zm9v",
        "options": {"api_key": secret},
    }
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422
    assert secret not in resp.text
    # Structured list; the offending field is named in a loc entry
    # (anchored under ["options"]) so the caller can still fix the request.
    detail = resp.json()["detail"]
    assert any("api_key" in entry["loc"] for entry in detail)


def test_options_validation_error_message_omits_input_value() -> None:
    # A malformed language tag in options must surface a useful message but never
    # the raw value: neither pydantic's input_value echo nor the validator's own
    # message may reflect it (a secret mis-pasted as `language` is not
    # credential-NAMED, so field-name redaction alone would not catch it).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    sentinel = "my secret passphrase here"
    payload = {
        "model": "dummy/echo",
        "audio": "Zm9v",
        "options": {"language": sentinel},
    }
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422
    assert sentinel not in resp.text
    # Structured list; the offending field is named (under
    # ["options"]) but neither the raw value nor pydantic's ``input``/``ctx``
    # echo (which would carry it) survive.
    detail = resp.json()["detail"]
    assert any("language" in entry["loc"] for entry in detail)
    assert "input_value" not in resp.text
    assert "input" not in {k for entry in detail for k in entry}


def test_transcribe_file_options_validation_error_is_sanitized() -> None:
    # The multipart endpoint's options (valid JSON, invalid RuntimeParams) take
    # the sanitized ValidationError branch (a mis-placed secret is not echoed).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    secret = "sk-MULTIPART-SECRET"
    data = {"model": "dummy/echo", "options": json.dumps({"api_key": secret})}
    resp: httpx2.Response = client.post("/v1/transcribe", data=data, files=files)
    assert resp.status_code == 422
    assert secret not in resp.text
    # Structured list (same shape as the JSON endpoint).
    assert any("api_key" in entry["loc"] for entry in resp.json()["detail"])


def test_transcribe_json_rejects_provider_params_over_wire() -> None:
    # provider_params is discover-only, never sendable. A request whose
    # options carry it must be rejected with a clear 422 (not silently dropped
    # or mis-routed into the internal model).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    payload = {
        "model": "dummy/echo",
        "audio": "Zm9v",
        "options": {"language": "en", "provider_params": {"beam": 5}},
    }
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422
    # The rejected key is named in a loc entry; cross-language clients
    # can also branch on the machine-readable type (``extra_forbidden``).
    detail = resp.json()["detail"]
    assert any("provider_params" in entry["loc"] for entry in detail)
    assert any(entry["type"] == "extra_forbidden" for entry in detail)


def test_transcribe_file_rejects_provider_params_over_wire() -> None:
    # The multipart endpoint enforces the same portable-only wire contract.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    files = {"file": ("audio.wav", b"fake", "audio/wav")}
    data = {"model": "dummy/echo", "options": json.dumps({"provider_params": {"beam": 5}})}
    resp: httpx2.Response = client.post("/v1/transcribe", data=data, files=files)
    assert resp.status_code == 422
    # Structured list (same shape as the JSON endpoint).
    assert any("provider_params" in entry["loc"] for entry in resp.json()["detail"])


def test_transcribe_json_portable_params_still_work() -> None:
    # A request carrying only portable params validates and transcribes normally.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    payload = {
        "model": "dummy/echo",
        "audio": base64.b64encode(b"fake").decode(),
        "options": {
            "language": "en",
            "word_timestamps": "word",
            "prompt": "hello",
            "phrase_hints": ["foo"],
            "on_unsupported": "degrade_to_prompt",
        },
    }
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 200
    assert resp.json()["result"]["text"] == "dummy"


def test_ws_rejects_provider_params_over_wire() -> None:
    # The WS config-frame path shares _build_params; provider_params in its
    # options must be rejected (bad_request), never reach the session.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json(
            {
                "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                "options": {"provider_params": {"beam": 5}},
            }
        )
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "bad_request"
    assert "provider_params" in err["message"]


def test_validation_error_with_non_string_loc_index_is_handled() -> None:
    # A bad element inside a list field yields a loc with an int index
    # (e.g. ["candidate_languages", 0]); the redaction scan must skip the
    # non-string component without error.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)

    payload = {
        "model": "dummy/echo",
        "audio": "Zm9v",
        "options": {"candidate_languages": [123]},
    }
    resp: httpx2.Response = client.post("/v1/transcribe:json", json=payload)
    assert resp.status_code == 422
    # Structured list; the loc carries the int index
    # (e.g. ["options", "candidate_languages", 0]) without error.
    detail = resp.json()["detail"]
    assert any("candidate_languages" in entry["loc"] for entry in detail)


def test_ws_options_validation_error_does_not_echo_secret() -> None:
    # The WS config-frame path shares _build_params; a mis-placed secret in
    # options must not be echoed in the bad_request frame.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    secret = "sk-WS-OPTIONS-SECRET"
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json(
            {
                "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                "options": {"api_key": secret},
            }
        )
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "bad_request"
    assert secret not in json.dumps(err)
    assert "api_key" in err["message"]


def test_ws_config_frame_unknown_key_is_a_loud_bad_request() -> None:
    """A typo'd top-level key must fail the handshake, never start the session.

    The old ad-hoc parse (``config.get("options")``) silently ignored an
    unknown key: ``{"optinos": {...}}`` started the session on DEFAULTS
    while the client believed its options were applied -- a silent wrong
    result. The closed ``StreamConfigRequest`` model rejects it loudly,
    naming the offending key.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json(
            {
                "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                "optinos": {"language": "ja"},
            }
        )
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "bad_request"
    assert "optinos" in err["message"]  # the offending key is named


def test_ws_config_frame_non_object_json_is_bad_request() -> None:
    """A JSON array/scalar config frame is a caller mistake, not a crash."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_text("[1, 2, 3]")
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "bad_request"


def test_ws_config_frame_invalid_json_is_bad_request_without_document_echo() -> None:
    """Unparseable JSON maps to bad_request with positional detail only."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_text('{"audio_format": sk-NOT-JSON-SECRET}')
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "bad_request"
    assert "not valid JSON" in err["message"]
    assert "sk-NOT-JSON-SECRET" not in err["message"]  # positional detail only


def test_ws_handshake_internal_fault_is_internal_error_not_bad_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server-side handshake bug is OUR fault: scrubbed internal_error.

    The old catch-all mapped ANY exception to ``bad_request`` and sent raw
    ``str(exc)`` -- blaming the caller for internal faults and leaking
    internal text to an unauthenticated client.

        Args:
            monkeypatch: Pytest fixture for attribute patching.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    def _boom(options: object) -> object:
        raise RuntimeError("internal state at /secret/path leaked")

    monkeypatch.setattr(server_module, "_build_params", _boom)
    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "internal_error"
    assert err["code"] != "bad_request"
    assert "/secret/path" not in json.dumps(err)
    assert "See server logs" in err["message"]


def test_capabilities_endpoint_none_caps_returns_404() -> None:
    # An engine class with declared_capabilities=None has no caps to serve.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_no_caps_factory"))
    client = TestClient(app)
    resp: httpx2.Response = client.get("/v1/capabilities/dummy/echo")
    assert resp.status_code == 404
    assert "No capabilities" in resp.json()["detail"]


def test_params_schema_endpoint_none_returns_empty() -> None:
    # An engine with no provider_params_type publishes an empty schema.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_no_caps_factory"))
    client = TestClient(app)
    resp: httpx2.Response = client.get("/v1/params-schema/dummy/echo")
    assert resp.status_code == 200
    assert resp.json() == {}


def test_config_schema_endpoint() -> None:
    # The init-config JSON Schema is served from the class-level config_type.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_with_config_type_factory"))
    client = TestClient(app)
    resp: httpx2.Response = client.get("/v1/config-schema/dummy/echo")
    assert resp.status_code == 200
    schema = resp.json()
    assert "engine" in schema.get("properties", {})
    assert "strict" in schema.get("properties", {})


def test_config_schema_endpoint_none_returns_empty() -> None:
    # An engine without a declared config_type publishes an empty schema.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    resp: httpx2.Response = client.get("/v1/config-schema/dummy/echo")
    assert resp.status_code == 200
    assert resp.json() == {}


def test_config_schema_endpoint_unknown_model() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    resp: httpx2.Response = client.get("/v1/config-schema/nope/missing")
    assert resp.status_code == 404


def test_config_schema_endpoint_broken_config_type_is_scrubbed_500(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken ``config_type`` declaration is a scrubbed 500, never a crash.

    Reading ``config_type`` and rendering its schema is plugin code, so it sits
    inside the shared metadata fault boundary: the endpoint answers with the
    documented scrubbed body instead of leaving through Starlette's unhandled
    path, and the specifics reach the operator log via safe logging. The CLI's
    ``show`` degrades on the same fault through
    :meth:`ModelRegistry.config_schema`, which raises ``FactoryLoadError``.

        Args:
            caplog: Pytest fixture capturing log records.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_broken_config_type_factory"))
    client = TestClient(app)
    with caplog.at_level(logging.ERROR, logger="standard_asr.toolchain.server"):
        resp: httpx.Response = client.get("/v1/config-schema/dummy/echo")

    assert resp.status_code == 500
    # The response stays generic; the internal type name never reaches the wire.
    assert "_NotAConfigType" not in resp.text
    assert "Traceback" not in resp.text
    # The operator DOES get the specifics, in the log.
    assert any("failed to produce its metadata" in r.getMessage() for r in caplog.records)


def test_config_schema_no_instantiation() -> None:
    """config-schema must NOT instantiate the engine (it exists precisely so a
    credentialed engine's config form can be rendered before construction)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_no_instantiate_config_type_factory"))
    client = TestClient(app)
    resp: httpx2.Response = client.get("/v1/config-schema/dummy/echo")
    assert resp.status_code == 200
    assert "engine" in resp.json().get("properties", {})


def test_invalid_content_length_returns_400() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    # A non-integer Content-Length must be rejected by the body-size middleware.
    resp: httpx2.Response = client.post(
        "/v1/transcribe:json",
        content=b"{}",
        headers={"Content-Length": "not-a-number", "Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert "Invalid Content-Length" in resp.json()["detail"]


def test_body_size_middleware_passes_non_http_scope() -> None:
    # Non-HTTP scopes (websocket / lifespan) must pass straight through to the
    # wrapped app without the Content-Length inspection.
    import asyncio

    forwarded: list[str] = []

    async def _inner(scope: Any, receive: Any, send: Any) -> None:
        forwarded.append(scope["type"])

    mw = server_module._BodySizeLimitMiddleware(_inner, max_body_bytes=10)  # pyright: ignore[reportPrivateUsage]

    async def _noop() -> dict[str, Any]:
        return {}

    async def _send(_: Any) -> None:
        return None

    asyncio.run(mw({"type": "lifespan", "headers": []}, _noop, _send))
    assert forwarded == ["lifespan"]


def test_body_size_middleware_counts_streamed_bytes_and_suppresses_app_response() -> None:
    # The true-cap layer: an oversize body delivered as multiple chunks (no
    # honest Content-Length) is rejected with 413 the moment the cumulative count
    # exceeds the cap; the app keeps reading past the breach (covering the
    # already-rejected branch and the disconnect passthrough) and its own late
    # response is suppressed so it cannot clobber the 413.
    import asyncio

    # Two over-cap body chunks (the cap is 4): the first breaches and triggers
    # the 413; a (deliberately stubborn) app keeps reading, so the wrapper is
    # re-entered on a second over-cap chunk and must NOT emit a second 413 (the
    # already-rejected branch), then yields a disconnect.
    incoming: list[dict[str, Any]] = [
        {"type": "http.request", "body": b"aaaaa", "more_body": True},
        {"type": "http.request", "body": b"bbbbb", "more_body": False},
    ]

    async def _receive() -> dict[str, Any]:
        if incoming:
            return incoming.pop(0)
        return {"type": "http.disconnect"}

    sent: list[dict[str, Any]] = []

    async def _send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def _app(scope: Any, receive: Any, send: Any) -> None:
        # A stubborn body-reading app: pull a couple of frames even past a
        # disconnect (forcing the wrapper to re-enter after rejection), then try
        # to respond (this late response must be suppressed).
        for _ in range(3):
            await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"late"})

    mw = server_module._BodySizeLimitMiddleware(_app, max_body_bytes=4)  # pyright: ignore[reportPrivateUsage]
    asyncio.run(mw({"type": "http", "headers": []}, _receive, _send))

    # Exactly one response was emitted: the middleware's 413 (the app's 200 +
    # body were suppressed).
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1
    assert starts[0]["status"] == 413
    assert not any(m.get("body") == b"late" for m in sent)


def test_body_size_middleware_within_cap_streamed_passes_through() -> None:
    # A streamed body within the cap passes through untouched: the app reads all
    # frames and its own response is delivered (no 413, no suppression).
    import asyncio

    incoming: list[dict[str, Any]] = [
        {"type": "http.request", "body": b"ab", "more_body": True},
        {"type": "http.request", "body": b"c", "more_body": False},
    ]

    async def _receive() -> dict[str, Any]:
        if incoming:
            return incoming.pop(0)
        return {"type": "http.disconnect"}

    sent: list[dict[str, Any]] = []

    async def _send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def _app(scope: Any, receive: Any, send: Any) -> None:
        total = 0
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                break
            total += len(message.get("body", b""))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": str(total).encode()})

    mw = server_module._BodySizeLimitMiddleware(_app, max_body_bytes=4)  # pyright: ignore[reportPrivateUsage]
    asyncio.run(mw({"type": "http", "headers": []}, _receive, _send))

    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1 and starts[0]["status"] == 200
    # The app saw the full 3-byte body.
    assert any(m.get("body") == b"3" for m in sent)


def test_transcribe_file_over_limit_without_content_length() -> None:
    # A chunked upload (no Content-Length) bypasses the early middleware guard;
    # the handler still rejects the materialised oversize body with 413.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry(), max_body_bytes=8)
    client = TestClient(app)

    # Build a multipart body by hand and stream it via an iterator so httpx2 omits
    # Content-Length (Transfer-Encoding: chunked), defeating the early guard.
    boundary = "----stdasrboundary"
    big_file = b"x" * 64
    parts = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="model"\r\n\r\n'
            "dummy/echo\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + big_file
        + f"\r\n--{boundary}--\r\n".encode()
    )

    def _gen() -> Any:
        yield parts

    resp: httpx2.Response = client.post(
        "/v1/transcribe",
        content=_gen(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"]


def test_transcribe_json_over_limit_without_content_length() -> None:
    # The JSON endpoint must reject an over-limit encoded payload too, even when
    # a chunked request (no Content-Length) slips past the early middleware guard.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry(), max_body_bytes=8)
    client = TestClient(app)

    body = json.dumps({"model": "dummy/echo", "audio": "x" * 64}).encode()

    def _gen() -> Any:
        yield body

    resp: httpx2.Response = client.post(
        "/v1/transcribe:json",
        content=_gen(),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"]


def test_capabilities_no_instantiation() -> None:
    """capabilities/params-schema must NOT instantiate the engine (DoS / auth)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # _NoInstantiateASR.__init__ raises; reading ClassVars must still work.
    app = server_module.create_app(registry=_registry_for("_no_instantiate_factory"))
    client = TestClient(app)

    caps: httpx2.Response = client.get("/v1/capabilities/dummy/echo")
    assert caps.status_code == 200
    assert caps.json()["batch"]["language"]["runtime_override"]["supported"] is True

    schema: httpx2.Response = client.get("/v1/params-schema/dummy/echo")
    assert schema.status_code == 200
    assert "beam" in schema.json().get("properties", {})


# --- WebSocket streaming surface ---------------------------------------------


class _StreamProperties(BaseProperties):
    engine_id: str = "stream"
    model_name: str = "echo"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    selectable_languages: list[str] = []
    wire_encodings: list[str] | None = ["pcm_s16le"]


class _StreamEchoSession(TranscriptionSession):
    """Emits one final per fed chunk (its decoded text), then the base done."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        index = 0
        async for chunk in self.audio_chunks():
            yield TranscriptionEvent.final(
                f"seg-{index}",
                chunk.decode("utf-8", "replace"),
                start=float(index),
                end=float(index + 1),
            )
            index += 1


class _StreamEchoEngine(EngineBase):
    properties: ClassVar[BaseProperties] = _StreamProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
    )

    def __init__(self) -> None:
        self.config = _DummyConfig(engine="stream")

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        return TranscriptionResult(text="")  # batch path unused by these tests

    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        return _StreamEchoSession()


def _stream_echo_factory() -> _StreamEchoEngine:  # pyright: ignore[reportUnusedFunction]
    return _StreamEchoEngine()


class _UnprojectableDiagSession(_StreamEchoSession):
    """Carries a diagnostic whose value has no JSON form.

    Reaching this state needs `model_construct`: `emit_diagnostic` is typed
    JsonValue and rejects it loudly at the emission site, which is the other
    half of the fix. What is pinned here is that the TRANSPORT contains it.
    """

    def __init__(self) -> None:
        super().__init__()
        from standard_asr.contract.results import Diagnostic

        self._guard.record_diagnostic(  # pyright: ignore[reportPrivateUsage]
            Diagnostic.model_construct(code="x", message="m", provided=object())
        )


class _UnprojectableDiagEngine(_StreamEchoEngine):
    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        return _UnprojectableDiagSession()


def _unprojectable_diag_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _UnprojectableDiagEngine
):
    return _UnprojectableDiagEngine()


class _UnprojectableResultASR(_DummyASR):
    """Returns a result whose `extra` was mutated past validation."""

    def transcribe(self, audio: Any, options: Any = None) -> TranscriptionResult:
        result = TranscriptionResult(text="ok")
        result.extra["opaque"] = object()  # pyright: ignore[reportArgumentType]
        return result


def _unprojectable_result_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _UnprojectableResultASR
):
    return _UnprojectableResultASR()


class _IterationFaultSession(_StreamEchoSession):
    """Faults the bridge's forward loop itself: iteration raises directly
    instead of being funneled into an ``engine_error`` event by the base."""

    def __aiter__(self) -> AsyncIterator[TranscriptionEvent]:
        raise RuntimeError("forward-loop fault: /secret/internal/path")


class _IterationFaultEngine(_StreamEchoEngine):
    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        return _IterationFaultSession()


def _iteration_fault_factory() -> _IterationFaultEngine:  # pyright: ignore[reportUnusedFunction]
    return _IterationFaultEngine()


class _StreamBestEffortEngine(_StreamEchoEngine):
    """A best_effort streaming engine: an unsupported standard param requested at
    session start is dropped + diagnosed (rather than raising), so the session
    carries a standard-layer diagnostic to forward to the WS client."""

    def __init__(self) -> None:
        # best_effort: unsupported params are dropped with a diagnostic.
        self.config = _DummyConfig(engine="stream", strict=False)


def _stream_best_effort_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _StreamBestEffortEngine
):
    return _StreamBestEffortEngine()


class _SpeakerStreamSession(TranscriptionSession):
    """Emits one speaker-labelled final per fed chunk."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        index = 0
        async for chunk in self.audio_chunks():
            yield TranscriptionEvent.final(
                f"seg-{index}", chunk.decode("utf-8", "replace"), speaker="A"
            )
            index += 1


class _SpeakerStreamEngine(_StreamEchoEngine):
    """Streaming engine that declares diarization and labels its finals."""

    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities(
        streaming=StreamingCapabilities(diarization=DiarizationCap(supported=True)),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
    )

    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        return _SpeakerStreamSession()


def _speaker_stream_factory() -> _SpeakerStreamEngine:  # pyright: ignore[reportUnusedFunction]
    return _SpeakerStreamEngine()


def test_ws_stream_happy_path() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json(
            {"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}, "options": None}
        )
        ws.send_bytes(b"abc")
        ws.send_bytes(b"de")
        ws.send_text("end")  # any text frame signals end-of-audio
        events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "done":
                break
    finals = {e["text"] for e in events if e["type"] == "final"}
    assert finals == {"abc", "de"}


def test_ws_event_carries_speaker() -> None:
    # End-to-end diarization wire pin: {"diarization": {}} in the WS start
    # options passes gating (declared supported) and the event's speaker rides
    # the JSON payload; the constant event shape serializes speaker=null on
    # unlabelled events (here: done).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_speaker_stream_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json(
            {
                "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                "options": {"diarization": {}},
            }
        )
        ws.send_bytes(b"abc")
        ws.send_text("end")
        events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "done":
                break
    finals = [e for e in events if e["type"] == "final"]
    assert finals and finals[0]["speaker"] == "A"
    assert events[-1]["type"] == "done" and events[-1]["speaker"] is None


def test_ws_stream_forwards_degrade_diagnostics() -> None:
    # A best_effort session whose params are degraded (an unsupported
    # word_timestamps request is dropped) must deliver the standard-layer
    # diagnostic to the WS client -- up front, before audio events -- mirroring
    # how REST returns diagnostics on the result.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_best_effort_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json(
            {
                "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                "options": {"word_timestamps": "word"},
            }
        )
        # The diagnostics frame arrives before any audio is sent.
        first = ws.receive_json()
        ws.send_text("end")
        rest: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            rest.append(event)
            if event["type"] == "done":
                break
    assert first["type"] == "diagnostics"
    codes = {d["code"] for d in first["diagnostics"]}
    assert "unsupported_parameter_ignored" in codes
    assert any(d.get("param") == "word_timestamps" for d in first["diagnostics"])
    # The diagnostics frame is distinct from the event stream.
    assert all(e["type"] != "diagnostics" for e in rest)


def test_ws_stream_no_diagnostics_frame_when_none() -> None:
    # A session with no standard-layer diagnostics must NOT emit a diagnostics
    # frame: the client sees only events (the happy path is unchanged).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json(
            {"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}, "options": None}
        )
        ws.send_bytes(b"abc")
        ws.send_text("end")
        events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "done":
                break
    assert all(e["type"] != "diagnostics" for e in events)
    assert {e["text"] for e in events if e["type"] == "final"} == {"abc"}


class _StreamDiagSession(_StreamEchoSession):
    """Emits a final per chunk AND a mid-stream diagnostic via emit_diagnostic."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        index = 0
        async for chunk in self.audio_chunks():
            self.emit_diagnostic(code="vad_fallback", message="used energy VAD", level="warning")
            yield TranscriptionEvent.final(
                f"seg-{index}",
                chunk.decode("utf-8", "replace"),
                start=float(index),
                end=float(index + 1),
            )
            index += 1


class _StreamDiagEngine(_StreamEchoEngine):
    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        return _StreamDiagSession()


def _stream_diag_factory() -> _StreamDiagEngine:  # pyright: ignore[reportUnusedFunction]
    return _StreamDiagEngine()


def test_ws_stream_forwards_mid_stream_diagnostics() -> None:
    # A diagnostic emitted by _produce mid-stream (emit_diagnostic) reaches
    # the client as a `diagnostics` frame -- not only the establishment-time set.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_diag_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json(
            {"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}, "options": None}
        )
        ws.send_bytes(b"abc")
        ws.send_text("end")
        events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "done":
                break
    types = [e["type"] for e in events]
    diag_frames = [e for e in events if e["type"] == "diagnostics"]
    assert diag_frames, types
    assert any(d["code"] == "vad_fallback" for f in diag_frames for d in f["diagnostics"])
    # It is a MID-stream frame: it follows the first audio event (not only up front).
    assert types.index("diagnostics") > types.index("final")


class _StreamOverflowSession(_StreamEchoSession):
    """Overflows the bounded diagnostics channel mid-stream.

    The guard keeps at most ``max_diagnostics`` entries; past the cap it
    stops appending and REWRITES its trailing ``diagnostics_truncated``
    summary in place, so the list stops growing while its per-code tally
    keeps changing.
    """

    def __init__(self) -> None:
        super().__init__(max_guard_diagnostics=2)

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        index = 0
        async for chunk in self.audio_chunks():
            for _ in range(4):
                self.emit_diagnostic(code="vad_fallback", message="noisy", level="warning")
            yield TranscriptionEvent.final(
                f"seg-{index}",
                chunk.decode("utf-8", "replace"),
                start=float(index),
                end=float(index + 1),
            )
            index += 1


class _StreamOverflowEngine(_StreamEchoEngine):
    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        return _StreamOverflowSession()


def _stream_overflow_factory() -> _StreamOverflowEngine:  # pyright: ignore[reportUnusedFunction]
    return _StreamOverflowEngine()


def test_ws_stream_delivers_the_updated_overflow_summary() -> None:
    # G.5.2: the WS client's final view of the diagnostics channel must match
    # the in-process/REST one. The delta cursor counted ENTRIES, but the
    # overflow summary is rewritten IN PLACE -- the list stops growing while
    # its counts keep changing -- so the client kept the tally from the FIRST
    # overflow forever while the session converged on the final one.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_overflow_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json(
            {"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}, "options": None}
        )
        ws.send_bytes(b"ab")
        ws.send_bytes(b"cd")
        ws.send_text("end")
        events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "done":
                break

    delivered = [d for e in events if e["type"] == "diagnostics" for d in e["diagnostics"]]
    summaries = [d for d in delivered if d["code"] == "diagnostics_truncated"]
    assert summaries, [e["type"] for e in events]
    # The LAST summary the client received is the final tally, not the first.
    assert "'vad_fallback': 7" in summaries[-1]["message"]
    # It is a singleton on the wire: a later occurrence supersedes the earlier
    # one rather than accumulating a new kind of entry.
    assert len({s["message"] for s in summaries}) == len(summaries)


def test_diagnostics_delta_frame_resends_only_the_changed_summary() -> None:
    # The re-send carries the summary ALONE: the entries the client already
    # holds are not repeated just because the summary moved.
    from standard_asr.contract.results import Diagnostic
    from standard_asr.runtime import streaming as streaming_module
    from standard_asr.runtime.streaming import DIAGNOSTICS_TRUNCATED_CODE

    guard = streaming_module._LifecycleGuard(max_diagnostics=3)  # pyright: ignore[reportPrivateUsage]

    class _Channel:
        def diagnostics(self) -> list[Any]:
            return list(guard.diagnostics)

    channel = cast("Any", _Channel())
    cursor: Any = (0, None)
    frames: list[Any] = []
    for _ in range(6):
        guard.record_diagnostic(Diagnostic(level="warning", code="noisy", message="n"))
        frame, cursor = server_module._diagnostics_delta_frame(channel, cursor)  # pyright: ignore[reportPrivateUsage]
        if frame is not None:
            frames.append(frame)

    # Every frame after the cap carries exactly the summary, once.
    tail = frames[-1]["diagnostics"]
    assert len(tail) == 1
    assert tail[0]["code"] == DIAGNOSTICS_TRUNCATED_CODE
    # An unchanged channel yields no frame at all.
    assert server_module._diagnostics_delta_frame(channel, cursor)[0] is None  # pyright: ignore[reportPrivateUsage]


def test_initial_diagnostics_frame_none_when_empty() -> None:
    # A session reporting no diagnostics yields no frame (the caller then sends
    # nothing, leaving the happy path untouched).
    class _EmptyChannel:
        def diagnostics(self) -> list[Any]:
            return []

    assert server_module._initial_diagnostics_frame(_EmptyChannel()) is None  # pyright: ignore[reportPrivateUsage, reportArgumentType]


def test_ws_stream_bad_config_reports_error() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"no_audio_format": True})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "bad_request"


def test_ws_stream_unknown_model_reports_error() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/nope/missing") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "unknown_model"


def test_ws_stream_non_streaming_engine_reports_unsupported() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # _RecordingArray8kASR is a batch-only EngineBase: start_transcription raises.
    app = server_module.create_app(registry=_registry_for("_recording_array8k_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "unsupported"


class _StreamLangAxisProperties(_StreamProperties):
    # A language axis is present (selectable_languages non-empty) but the engine's
    # _DummyConfig sets no default_language, so _validate_language_config raises
    # ConfigError at SESSION ESTABLISHMENT (not construction) -- language-config
    # validation is total.
    selectable_languages: list[str] = ["en"]


class _StreamConfigErrorEngine(_StreamEchoEngine):
    properties: ClassVar[BaseProperties] = _StreamLangAxisProperties()


def _stream_config_error_factory() -> (  # pyright: ignore[reportUnusedFunction]
    _StreamConfigErrorEngine
):
    return _StreamConfigErrorEngine()


class _StreamRaisingEngine(_StreamEchoEngine):
    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        # An unexpected adapter fault during establishment, carrying internal
        # detail that MUST NOT leak to the client.
        raise RuntimeError("boom: internal detail /secret/path")


def _stream_raising_factory() -> _StreamRaisingEngine:  # pyright: ignore[reportUnusedFunction]
    return _StreamRaisingEngine()


def test_ws_stream_establishment_config_error_is_internal_not_bad_request() -> None:
    """An establishment-time ``ConfigError`` is an ENGINE fault: scrubbed frame.

    The fixture's fault is a missing ``default_language`` on a language-axis
    engine -- an engine DECLARATION defect no WS client can see, reach, or
    fix (engine config never crosses the wire). The old ``bad_request`` frame
    blamed the caller and surfaced the authored config text. Must map to the
    scrubbed ``internal_error`` -- and never the misleading ``unsupported``
    (ConfigError subclasses ValueError, so clause order matters).
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_config_error_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "internal_error"
    assert "default_language" not in err["message"]
    assert "See server logs" in err["message"]


class _StreamLazyCredEngine(_StreamEchoEngine):
    """Defers its credential check to session establishment (lazy absence)."""

    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        from standard_asr.contract.exceptions import ConfigurationRequiredError

        raise ConfigurationRequiredError(
            "api_key is required; set STANDARD_ASR_STREAM__API_KEY or pass it explicitly"
        )


def _stream_lazy_cred_factory() -> _StreamLazyCredEngine:  # pyright: ignore[reportUnusedFunction]
    return _StreamLazyCredEngine()


def test_ws_stream_establishment_missing_config_reports_service_unavailable() -> None:
    """Lazy ``ConfigurationRequiredError`` at establishment: the WS 503 twin.

    Absence is an operator-side availability state wherever it surfaces --
    construction OR a deferred credential check at establishment. Field names
    and env vars stay out of the frame.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_lazy_cred_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "service_unavailable"
    assert "server-side configuration" in err["message"]
    assert "api_key" not in err["message"]
    assert "STANDARD_ASR_STREAM__API_KEY" not in err["message"]


class _StreamIPPEEngine(_StreamEchoEngine):
    """Raises ``InvalidProviderParamError`` at establishment (engine contract bug)."""

    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        from standard_asr.contract.exceptions import InvalidProviderParamError

        raise InvalidProviderParamError("foreign provider_params: beam='sk-PROVIDER-SECRET'")


def _stream_ippe_factory() -> _StreamIPPEEngine:  # pyright: ignore[reportUnusedFunction]
    return _StreamIPPEEngine()


def test_ws_stream_establishment_provider_param_error_is_internal() -> None:
    """``InvalidProviderParamError`` at establishment is an engine fault.

    ``WireRuntimeParams`` rejects ``provider_params`` in the config frame, so
    no WS client can legally cause this error -- it reaching the route means
    the server/engine contract broke. Scrubbed ``internal_error``; the
    authored message (which may embed provider payload text) never crosses.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_ippe_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "internal_error"
    assert "sk-PROVIDER-SECRET" not in err["message"]


def test_ws_stream_establishment_unexpected_error_reports_internal_no_leak() -> None:
    # An unexpected fault in the engine's _start_transcription hook must not crash
    # the route or leak internal detail: a single generic internal_error frame is
    # sent instead.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_raising_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "internal_error"
    assert "boom" not in err["message"]
    assert "secret" not in err["message"]


def test_ws_stream_forward_loop_fault_reports_internal_no_leak() -> None:
    """A fault in the forward loop itself is logged and reported, not swallowed.

    Engine faults are normally funneled into ``engine_error`` events by the
    session base; this exercises the bridge's own failure path (session
    iteration / event serialization), which must emit one generic, non-leaking
    ``internal_error`` frame instead of silently dropping the stream.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_iteration_fault_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "internal_error"
    assert "/secret/internal/path" not in err["message"]
    assert "See server logs" in err["message"]


def test_ws_stream_client_disconnect_is_handled() -> None:
    # A client that leaves mid-stream must not crash the server: the bridge ends
    # the session and stops forwarding (covers the disconnect + send-failure path).
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        ws.send_bytes(b"abc")
        first = ws.receive_json()
        assert first["type"] == "final"
    # Exiting the context closes the socket without an end frame.


class _StreamErrorSession(TranscriptionSession):
    """Raises a detail-bearing exception so the base synthesizes an
    ``engine_error`` event whose ``extra['detail']`` carries the raw text."""

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        async for _chunk in self.audio_chunks():
            raise RuntimeError("boom: /secret/internal/path leaked")
        yield TranscriptionEvent.done()  # pragma: no cover - never reached


class _StreamErrorEngine(_StreamEchoEngine):
    def _start_transcription(
        self,
        *,
        gated_params: Any = None,
        audio_format: Any = None,
        prepared_audio: PreparedAudio | None = None,
    ) -> TranscriptionSession:
        return _StreamErrorSession()


def _stream_error_factory() -> _StreamErrorEngine:  # pyright: ignore[reportUnusedFunction]
    return _StreamErrorEngine()


def test_ws_stream_error_event_does_not_leak_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An engine that raises mid-stream surfaces an `error` event. Its raw
    # exception text MUST stay server-side (logged), never reach the client --
    # matching the REST 500 non-leak contract.
    import logging

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_error_factory"))
    client = TestClient(app)
    with caplog.at_level(logging.ERROR, logger="standard_asr.toolchain.server"):
        with client.websocket_connect("/v1/stream/dummy/echo") as ws:
            ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
            ws.send_bytes(b"abc")
            ws.send_text("end")
            events: list[dict[str, Any]] = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] == "error":
                    break

    error = next(e for e in events if e["type"] == "error")
    assert error["code"] == "engine_error"
    # The structured fields survive; the raw detail is scrubbed from the frame.
    assert error["recoverable"] is False
    assert error["extra"] == {}
    assert "/secret/internal/path" not in json.dumps(error)
    # The dropped detail is logged server-side for operators.
    assert any("/secret/internal/path" in rec.getMessage() for rec in caplog.records)


def test_ws_stream_oversize_frame_rejected() -> None:
    # A single binary AUDIO frame larger than the per-frame cap is rejected with
    # a policy error and the session is torn down (the HTTP body guard does not
    # cover the WS scope, so the bridge must cap frames itself). The cap is set
    # above the small config frame so this exercises the audio-frame path.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(
        registry=_registry_for("_stream_echo_factory"), max_ws_frame_bytes=128
    )
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        ws.send_bytes(b"x" * 200)  # 200 bytes > 128-byte per-frame cap
        events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "error":
                break
    err = events[-1]
    assert err["code"] == "payload_too_large"
    assert "per-frame limit" in err["message"]


def test_ws_stream_cumulative_cap_rejected() -> None:
    # Each audio frame is within the per-frame cap, but their cumulative total
    # exceeds the per-session cap: the bridge rejects with the policy error. The
    # per-frame cap is set above the config frame so it is not pre-empted.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(
        registry=_registry_for("_stream_echo_factory"),
        max_ws_frame_bytes=128,
        max_ws_session_bytes=5,
    )
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        ws.send_bytes(b"abc")  # 3 bytes (ok)
        ws.send_bytes(b"def")  # cumulative 6 > 5-byte session cap
        events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "error":
                break
    err = events[-1]
    assert err["code"] == "payload_too_large"
    assert "per-session limit" in err["message"]


def test_ws_stream_oversize_config_frame_rejected() -> None:
    # The config/handshake frame is bounded by the app per-frame cap too (not
    # only the transport ws_max_size), so an oversize config frame is rejected
    # before it is parsed -- the documented DoS bound holds regardless of the
    # ASGI server in front.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(
        registry=_registry_for("_stream_echo_factory"), max_ws_frame_bytes=16
    )
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        # A valid-but-large config frame (well over the 16-byte cap).
        ws.send_json(
            {
                "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                "options": {"prompt": "x" * 64},
            }
        )
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "payload_too_large"
    assert "Config frame too large" in err["message"]


def test_ws_stream_config_frame_as_bytes_is_rejected() -> None:
    # The config/handshake frame MUST be a JSON *text* frame;
    # *binary* frames are reserved for raw audio. A binary
    # first frame is a malformed handshake and is rejected with a `bad_request`
    # policy frame (not parsed as config). Accepting it would bake an undefined
    # leniency into the reference implementation that strict third-party servers
    # would not share -- a cross-implementation hazard for the versioned wire
    # protocol.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_stream_echo_factory"))
    client = TestClient(app)
    config = json.dumps(
        {"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}, "options": None}
    ).encode()
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_bytes(config)  # config (wrongly) delivered as a binary frame
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "bad_request"
    assert "text frame" in err["message"]


def test_ws_stream_within_caps_still_works() -> None:
    # Within both caps, audio still flows and the session completes normally.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(
        registry=_registry_for("_stream_echo_factory"),
        max_ws_frame_bytes=128,
        max_ws_session_bytes=64,
    )
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        ws.send_bytes(b"abc")
        ws.send_text("end")
        events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "done":
                break
    assert {e["text"] for e in events if e["type"] == "final"} == {"abc"}


@pytest.mark.parametrize("kwargs", [{"max_ws_frame_bytes": 0}, {"max_ws_session_bytes": 0}])
def test_create_app_rejects_nonpositive_ws_caps(kwargs: dict[str, int]) -> None:
    pytest.importorskip("fastapi")
    with pytest.raises(ValueError):
        server_module.create_app(registry=_registry(), **kwargs)


def test_bridge_stream_pump_failure_is_logged_and_signalled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A protocol violation on the pump side (send_audio raising, e.g.
    # StreamClosedError) must NOT be silently swallowed: it is logged
    # server-side and surfaced to the client as a single generic, non-leaking
    # error frame.
    import asyncio
    import logging

    pytest.importorskip("fastapi")
    from standard_asr.contract.exceptions import StreamClosedError

    class _FakeWS:
        def __init__(self) -> None:
            self._frames: list[dict[str, Any]] = [{"type": "websocket.receive", "bytes": b"abc"}]
            self.sent: list[Any] = []

        async def receive(self) -> dict[str, Any]:
            if self._frames:
                return self._frames.pop(0)
            return {"type": "websocket.disconnect"}

        async def send_json(self, data: Any) -> None:
            self.sent.append(data)

    class _FakeSession:
        def __init__(self) -> None:
            # Set when input ends so the producer terminates (mirrors a real
            # session: it does not emit `done` until input is ended).
            self._ended = asyncio.Event()

        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def send_audio(self, chunk: bytes) -> None:
            raise StreamClosedError("session already ended: /secret/path")

        async def end_audio(self) -> None:
            self._ended.set()

        def diagnostics(self) -> list[Any]:
            return []

        def __aiter__(self) -> AsyncIterator[TranscriptionEvent]:
            async def _gen() -> AsyncIterator[TranscriptionEvent]:
                await self._ended.wait()
                yield TranscriptionEvent.done()

            return _gen()

    websocket = _FakeWS()
    with caplog.at_level(logging.ERROR, logger="standard_asr.toolchain.server"):
        asyncio.run(
            server_module._bridge_stream(  # pyright: ignore[reportPrivateUsage]
                websocket,  # pyright: ignore[reportArgumentType]
                _FakeSession(),  # pyright: ignore[reportArgumentType]
                max_frame_bytes=1024,
                max_session_bytes=1024,
            )
        )
    # The failure was logged (with detail) and a generic error frame was sent.
    assert any("audio pump failed" in rec.getMessage().lower() for rec in caplog.records)
    error_frames = [f for f in websocket.sent if f.get("type") == "error"]
    assert error_frames and error_frames[-1]["code"] == "stream_input_error"
    assert "/secret/path" not in json.dumps(websocket.sent)


def test_bridge_stream_tolerates_send_failure() -> None:
    # If the client vanishes mid-stream, the send raises WebSocketDisconnect
    # (starlette maps the transport OSError to it): the bridge treats it as the
    # quiet teardown path -- no propagation, input ended, session torn down.
    import asyncio

    pytest.importorskip("fastapi")
    from fastapi import WebSocketDisconnect

    class _FakeWS:
        def __init__(self) -> None:
            self.send_attempted = False

        async def receive(self) -> dict[str, Any]:
            return {"type": "websocket.disconnect"}

        async def send_json(self, data: Any) -> None:
            self.send_attempted = True
            raise WebSocketDisconnect(code=1006)

    class _FakeSession:
        def __init__(self) -> None:
            self.ended = False

        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def send_audio(self, chunk: bytes) -> None:  # pragma: no cover - unused
            return None

        async def end_audio(self) -> None:
            self.ended = True

        def diagnostics(self) -> list[Any]:
            return []

        def __aiter__(self) -> AsyncIterator[TranscriptionEvent]:
            async def _gen() -> AsyncIterator[TranscriptionEvent]:
                yield TranscriptionEvent.done()

            return _gen()

    websocket = _FakeWS()
    asyncio.run(
        server_module._bridge_stream(  # pyright: ignore[reportPrivateUsage]
            websocket,  # pyright: ignore[reportArgumentType]
            _FakeSession(),  # pyright: ignore[reportArgumentType]
            max_frame_bytes=1024,
            max_session_bytes=1024,
        )
    )
    # The send was attempted and its failure was swallowed (the run completed).
    assert websocket.send_attempted is True


def test_bridge_stream_unexpected_send_failure_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A NON-disconnect send failure must not be silently swallowed like a
    # vanished client: it is logged server-side, and the best-effort generic
    # frame is attempted (here it fails too, which must still not propagate).
    import asyncio
    import logging

    pytest.importorskip("fastapi")

    class _FakeWS:
        def __init__(self) -> None:
            self.send_attempts = 0

        async def receive(self) -> dict[str, Any]:
            return {"type": "websocket.disconnect"}

        async def send_json(self, data: Any) -> None:
            self.send_attempts += 1
            raise RuntimeError("serialization fault: /secret/path")

    class _FakeSession:
        def __init__(self) -> None:
            self.ended = False

        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def send_audio(self, chunk: bytes) -> None:  # pragma: no cover - unused
            return None

        async def end_audio(self) -> None:
            self.ended = True

        def diagnostics(self) -> list[Any]:
            return []

        def __aiter__(self) -> AsyncIterator[TranscriptionEvent]:
            async def _gen() -> AsyncIterator[TranscriptionEvent]:
                yield TranscriptionEvent.done()

            return _gen()

    websocket = _FakeWS()
    with caplog.at_level(logging.ERROR, logger="standard_asr.toolchain.server"):
        asyncio.run(
            server_module._bridge_stream(  # pyright: ignore[reportPrivateUsage]
                websocket,  # pyright: ignore[reportArgumentType]
                _FakeSession(),  # pyright: ignore[reportArgumentType]
                max_frame_bytes=1024,
                max_session_bytes=1024,
            )
        )
    # The fault was logged (with detail, server-side only) and the bridge also
    # attempted the generic internal_error frame before tearing down.
    assert any("event forwarding failed" in rec.getMessage().lower() for rec in caplog.records)
    assert websocket.send_attempts == 2


def test_server_extra_declares_a_websocket_library() -> None:
    # Drift guard: server.md promises a WebSocket
    # streaming endpoint, but bare uvicorn ships no WS protocol implementation.
    # The documented `pip install standard-asr[server]` must therefore pull one
    # in, or /v1/stream answers 404 on upgrade in every user install while the
    # in-process TestClient suite stays green.
    import re
    from pathlib import Path

    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^server\s*=\s*\[(?P<deps>[^\]]*)\]", pyproject, re.MULTILINE)
    assert match is not None, "pyproject.toml must declare the [server] extra"
    deps = match.group("deps")
    assert "websockets" in deps or "wsproto" in deps, (
        "The [server] extra must include a WebSocket protocol library "
        "(websockets or wsproto); bare uvicorn cannot serve /v1/stream."
    )


# --------------------------------------------------------------------------- #
# Round-5 H1: a registered model whose plugin fails to LOAD is an engine
# fault (scrubbed 500 / internal_error), never an "unknown model" 404.
# --------------------------------------------------------------------------- #


def _broken_load_registry() -> ModelRegistry:
    """Build a registry whose one model's entry point cannot be loaded.

    The spec EXISTS (discovery validates names, not loadability), so the
    model key resolves; `load_factory` then raises `FactoryLoadError` whose
    message embeds the import failure (the module attribute named below is
    the sentinel that must never reach a client).

    Returns:
        A registry with one registered-but-unloadable model.
    """
    eps = [
        EntryPoint(
            name="dummy/echo",
            value="tests.test_server:_missing_SENTINEL_target",
            group="standard_asr.models",
        )
    ]
    return discover_models(eps=eps, strict=True)


def test_rest_registered_model_load_failure_is_scrubbed_500_not_404() -> None:
    """REST transcribe: a broken plugin is OUR fault, not the caller's.

    The old joint `(EntrypointValidationError, FactoryLoadError)` arm
    returned 404 "unknown model" with the raw plugin-fault text: the caller
    cannot fix a broken server-side plugin, and the message carries
    import/annotation internals.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_broken_load_registry())
    client = TestClient(app)
    resp: httpx2.Response = client.post(
        "/v1/transcribe:json", json={"model": "dummy/echo", "audio": "Zm9v"}
    )
    assert resp.status_code == 500
    assert resp.status_code != 404
    assert "_missing_SENTINEL_target" not in resp.text
    assert "tests.test_server" not in resp.text
    assert "See server logs" in resp.text


def test_ws_registered_model_load_failure_is_internal_error_not_unknown_model() -> None:
    """WS: the same fault-ownership split as REST."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_broken_load_registry())
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "internal_error"
    assert err["code"] != "unknown_model"
    assert "_missing_SENTINEL_target" not in json.dumps(err)


def test_metadata_endpoints_load_failure_is_scrubbed_500_not_404() -> None:
    """capabilities / params-schema / config-schema: same split."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_broken_load_registry())
    client = TestClient(app)
    for path in (
        "/v1/capabilities/dummy/echo",
        "/v1/params-schema/dummy/echo",
        "/v1/config-schema/dummy/echo",
    ):
        resp: httpx2.Response = client.get(path)
        assert resp.status_code == 500, path
        assert "_missing_SENTINEL_target" not in resp.text, path
        assert "tests.test_server" not in resp.text, path


def test_truly_unknown_model_key_is_still_404() -> None:
    """The caller-fixable half keeps its authored 404 (nothing regressed)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_broken_load_registry())
    client = TestClient(app)
    resp: httpx2.Response = client.post(
        "/v1/transcribe:json", json={"model": "nope/nothing", "audio": "Zm9v"}
    )
    assert resp.status_code == 404
    assert "not found" in resp.text


def test_metadata_endpoints_contain_plugin_faults_and_never_leak(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Resolving the class was mapped, but EVERYTHING after it -- the
    # descriptor read, canonical_json(), model_json_schema() -- ran outside
    # any boundary and left the endpoint through Starlette's unhandled path:
    # an undocumented plain 500, and log_exception_safely never ran, so the
    # ASGI server's own traceback logger rendered the chain natively. A
    # ValidationError in it printed its input echo into the operator log.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    caplog.set_level(logging.ERROR)

    # (1) a metaclass property (the shape a lazily-built capability tree takes)
    app = server_module.create_app(registry=_registry_for("_hostile_caps_factory"))
    resp: httpx2.Response = TestClient(app).get("/v1/capabilities/dummy/echo")
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal model metadata error. See server logs for details."
    assert _METADATA_SECRET not in resp.text

    # (2) schema generation raising (a custom __get_pydantic_json_schema__)
    app = server_module.create_app(registry=_registry_for("_raising_schema_factory"))
    resp = TestClient(app).get("/v1/params-schema/dummy/echo")
    assert resp.status_code == 500
    assert "See server logs" in resp.json()["detail"]

    # (3) a payload that is not JSON at all: the projection is finished
    # inside the boundary, not left to the ASGI encoder.
    app = server_module.create_app(registry=_registry_for("_unencodable_caps_factory"))
    resp = TestClient(app).get("/v1/capabilities/dummy/echo")
    assert resp.status_code == 500
    assert "See server logs" in resp.json()["detail"]

    # The operator keeps a safe-logged record of all three, echo-free.
    assert "failed to produce its metadata" in caplog.text
    assert _METADATA_SECRET not in caplog.text


def test_metadata_boundary_starts_at_the_model_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Class RESOLUTION is third-party code too: ``_ensure_engine_class``
    # reads ``_is_protocol`` and ``transcribe`` -- both metaclass/descriptor
    # dispatch. Resolved OUTSIDE the boundary, a fault there left the
    # endpoint through Starlette's unhandled path: an undocumented plain
    # 500, and the ASGI server's native traceback logger rendering the raw
    # chain (input echo included) because log_exception_safely never ran.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    for factory in ("_hostile_protocol_factory", "_hostile_transcribe_factory"):
        for route in ("capabilities", "params-schema", "config-schema"):
            with caplog.at_level(logging.ERROR, logger="standard_asr.toolchain.server"):
                app = server_module.create_app(registry=_registry_for(factory))
                resp: httpx2.Response = TestClient(app).get(f"/v1/{route}/dummy/echo")
            assert resp.status_code == 500
            assert (
                resp.json()["detail"]
                == "Internal model metadata error. See server logs for details."
            )
            assert _METADATA_SECRET not in resp.text
    # The operator keeps the scrubbed record (sanitized, never the echo).
    assert "failed to produce its metadata" in caplog.text
    assert "ValidationError" in caplog.text
    assert _METADATA_SECRET not in caplog.text


def test_metadata_boundary_keeps_deliberate_verdicts_and_healthy_reads() -> None:
    # The boundary must not swallow the endpoints' OWN verdicts (the 404 for
    # an engine that declares no capabilities), nor change a healthy read.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = server_module.create_app(registry=_registry_for("_no_caps_factory"))
    client = TestClient(app)
    resp: httpx2.Response = client.get("/v1/capabilities/dummy/echo")
    assert resp.status_code == 404
    assert "No capabilities" in resp.json()["detail"]
    assert client.get("/v1/params-schema/dummy/echo").json() == {}

    app = server_module.create_app(registry=_registry())
    client = TestClient(app)
    assert client.get("/v1/capabilities/dummy/echo").status_code == 200
    assert client.get("/v1/params-schema/dummy/echo").status_code == 200
    # An unknown key is still the caller's 404, not a server fault.
    assert client.get("/v1/capabilities/nope/nope").status_code == 404


def test_bridge_forwards_final_diagnostics_delta_on_cap_violation() -> None:
    # A byte-cap violation breaks out of the forward loop BEFORE the
    # per-event delta take, so diagnostics accrued since the last delivered
    # event (an engine note, a guard suppression) were silently dropped: the
    # capped client ended the session never learning e.g. that a parameter
    # was degraded, while the REST path returns every diagnostic on the
    # result. The bridge must take one final delta before the terminal
    # policy frame.
    import asyncio

    pytest.importorskip("fastapi")
    from standard_asr.contract.results import Diagnostic

    class _FakeWS:
        def __init__(self) -> None:
            self.sent: list[Any] = []

        async def receive(self) -> dict[str, Any]:
            # One oversized audio frame: trips the per-frame cap immediately.
            return {"type": "websocket.receive", "bytes": b"x" * 64}

        async def send_json(self, data: Any) -> None:
            self.sent.append(data)

    class _FakeSession:
        def __init__(self) -> None:
            self.ended = asyncio.Event()

        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def send_audio(self, chunk: bytes) -> None:  # pragma: no cover
            return None

        async def end_audio(self) -> None:
            self.ended.set()

        def diagnostics(self) -> list[Diagnostic]:
            # Empty at establishment; the note "accrues" while the last
            # event is produced -- after the cap already tripped.
            if self.ended.is_set():
                return [
                    Diagnostic(
                        level="warning",
                        code="unsupported_parameter_ignored",
                        message="word_timestamps degraded",
                    )
                ]
            return []

        def __aiter__(self) -> AsyncIterator[TranscriptionEvent]:
            async def _gen() -> AsyncIterator[TranscriptionEvent]:
                # Yield only once the violation is certain, so the forward
                # loop's next iteration takes the break path deterministically.
                await self.ended.wait()
                yield TranscriptionEvent.done()

            return _gen()

    websocket = _FakeWS()
    asyncio.run(
        server_module._bridge_stream(  # pyright: ignore[reportPrivateUsage]
            websocket,  # pyright: ignore[reportArgumentType]
            _FakeSession(),  # pyright: ignore[reportArgumentType]
            max_frame_bytes=8,
            max_session_bytes=1024,
        )
    )
    types = [frame.get("type") for frame in websocket.sent]
    # The delta arrives, and BEFORE the terminal policy frame.
    assert "diagnostics" in types
    assert types.index("diagnostics") < types.index("error")
    diag_frame = next(f for f in websocket.sent if f.get("type") == "diagnostics")
    assert [d["code"] for d in diag_frame["diagnostics"]] == ["unsupported_parameter_ignored"]
    error = next(f for f in websocket.sent if f.get("type") == "error")
    assert error["code"] == "payload_too_large"


def test_bridge_logs_a_multiline_detail_as_one_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # ``extra["detail"]`` is engine-authored and an exact multi-line ``str``
    # passes the shape gate, so ``detail=%s`` interpolated it raw: a provider
    # error body carrying newlines forged additional log lines that a
    # line-oriented parser attributes to other requests (the same
    # line-forging class redaction's one-line summary contract closes). ``%r``
    # on an exact str dispatches only builtin code AND escapes every
    # ``splitlines`` boundary, so the record stays one line.
    import asyncio

    pytest.importorskip("fastapi")

    forged = "upstream failed\nWARNING fake line injected token=sk-forged"
    event = TranscriptionEvent.make_error(
        code="engine_error", recoverable=False, extra={"detail": forged}
    )

    class _FakeWS:
        def __init__(self) -> None:
            self.sent: list[Any] = []

        async def receive(self) -> dict[str, Any]:
            return {"type": "websocket.disconnect"}

        async def send_json(self, data: Any) -> None:
            self.sent.append(data)

    class _FakeSession:
        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def send_audio(self, chunk: bytes) -> None:  # pragma: no cover
            return None

        async def end_audio(self) -> None:
            return None

        def diagnostics(self) -> list[Any]:
            return []

        def __aiter__(self) -> AsyncIterator[TranscriptionEvent]:
            async def _gen() -> AsyncIterator[TranscriptionEvent]:
                yield event

            return _gen()

    with caplog.at_level(logging.ERROR, logger="standard_asr.toolchain.server"):
        asyncio.run(
            server_module._bridge_stream(  # pyright: ignore[reportPrivateUsage]
                _FakeWS(),  # pyright: ignore[reportArgumentType]
                _FakeSession(),  # pyright: ignore[reportArgumentType]
                max_frame_bytes=1024,
                max_session_bytes=1024,
            )
        )
    record = next(r for r in caplog.records if "Stream error event" in r.getMessage())
    message = record.getMessage()
    # One record, one line: every splitlines boundary arrives escaped.
    assert len(message.splitlines()) == 1
    assert "WARNING fake line injected" in message  # content kept, boundary escaped


def test_bridge_never_logs_a_detail_installed_past_validation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The bridge logs an error event's detail BEFORE the scrub -- the raw
    # detail is deliberately operator-side. But ``%r`` on the value
    # dispatches its ``__repr__``, and a value installed PAST the JsonValue
    # gate (a mutated ``extra`` dict is plain Python; ``frozen`` guards the
    # field, not the dict's contents) reaches that line first: a held
    # ``ValidationError``'s repr echoes its input into the operator log the
    # scrub exists to close. The bridge logs only exact-str / None.
    import asyncio

    pytest.importorskip("fastapi")
    from pydantic import BaseModel, ValidationError

    secret = "sk-WS-BRIDGE-DETAIL"  # noqa: S105 - test fixture

    class _StrictModel(BaseModel):
        api_key: int

    try:
        _StrictModel(api_key=secret)  # pyright: ignore[reportArgumentType]
    except ValidationError as exc:
        smuggled_error = exc
    else:  # pragma: no cover - construction must fail
        raise AssertionError("the model was expected to reject the input")

    event = TranscriptionEvent.make_error(
        code="engine_error", recoverable=False, extra={"detail": "safe text"}
    )
    event.extra["detail"] = smuggled_error  # pyright: ignore[reportArgumentType] -- past-validation mutation

    class _FakeWS:
        def __init__(self) -> None:
            self.sent: list[Any] = []

        async def receive(self) -> dict[str, Any]:
            return {"type": "websocket.disconnect"}

        async def send_json(self, data: Any) -> None:
            self.sent.append(data)

    class _FakeSession:
        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def send_audio(self, chunk: bytes) -> None:  # pragma: no cover
            return None

        async def end_audio(self) -> None:
            return None

        def diagnostics(self) -> list[Any]:
            return []

        def __aiter__(self) -> AsyncIterator[TranscriptionEvent]:
            async def _gen() -> AsyncIterator[TranscriptionEvent]:
                yield event

            return _gen()

    websocket = _FakeWS()
    with caplog.at_level(logging.ERROR, logger="standard_asr.toolchain.server"):
        asyncio.run(
            server_module._bridge_stream(  # pyright: ignore[reportPrivateUsage]
                websocket,  # pyright: ignore[reportArgumentType]
                _FakeSession(),  # pyright: ignore[reportArgumentType]
                max_frame_bytes=1024,
                max_session_bytes=1024,
            )
        )
    assert secret not in caplog.text
    assert any(
        "withheld: value installed past validation" in rec.getMessage() for rec in caplog.records
    )
    # The valid code survives the gate (only the smuggled part is withheld).
    assert any("engine_error" in rec.getMessage() for rec in caplog.records)
    # The client still gets the scrubbed event with the structured fields.
    error_frames = [f for f in websocket.sent if f.get("type") == "error"]
    assert error_frames and error_frames[-1]["code"] == "engine_error"
    assert error_frames[-1]["extra"] == {}
    assert secret not in json.dumps(websocket.sent)


def test_error_event_drops_extra_before_serializing() -> None:
    # The rule is drop-BEFORE-send. Serializing first made the payload's fate
    # depend on the very field being discarded: an `extra` value with no JSON
    # form raised inside the dump, so the client lost `code`, `recoverable`,
    # `retriable_after` and the gap/reconnect fields -- every safe structured
    # field the protocol documents -- to a value that was never going to be
    # sent. (Reaching this state needs model_construct: the JsonValue
    # declaration rejects it at construction, which is the other half of the
    # fix.)
    event = TranscriptionEvent.make_error(code="engine_error", recoverable=True)
    smuggled = event.model_copy(update={"extra": {"detail": object()}})

    payload = server_module._scrub_event_for_client(  # pyright: ignore[reportPrivateUsage]
        smuggled
    )
    assert payload["extra"] == {}
    assert payload["code"] == "engine_error"
    assert payload["recoverable"] is True
    assert payload["type"] == "error"


def test_wire_visible_slots_reject_values_with_no_json_form() -> None:
    # G5.2: the Python objects and the JSON documents are the same protocol
    # seen twice, so a wire-visible slot may only hold what a JSON document
    # can hold. Declared `Any`, these constructed happily and then failed the
    # projection AFTER an endpoint had committed to a response.
    from pydantic import JsonValue, ValidationError

    from standard_asr.contract.params import DIARIZE
    from standard_asr.contract.results import Diagnostic, Segment, Word, to_json_value

    for kwargs in (
        {"code": "x", "message": "m", "provided": object()},
        {"code": "x", "message": "m", "effective": object()},
    ):
        with pytest.raises(ValidationError):
            Diagnostic(**kwargs)  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValidationError):
        TranscriptionResult(text="x", extra={"opaque": object()})  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError):
        TranscriptionEvent.make_error(code="engine_error", extra={"o": object()})
    with pytest.raises(ValidationError):
        Word(start=0.0, end=1.0, text="w", extra={"o": object()})  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError):
        Segment(start=0.0, end=1.0, text="s", extra={"o": object()})  # pyright: ignore[reportArgumentType]

    # Non-finite floats are Python floats but NOT JSON: admitting them would
    # emit a document no conforming parser accepts.
    with pytest.raises(ValidationError):
        Diagnostic(code="x", message="m", provided=float("nan"))
    with pytest.raises(ValidationError):
        TranscriptionResult(text="x", extra={"ratio": float("inf")})

    # Everything a JSON document CAN hold still passes, nested arbitrarily.
    nested: JsonValue = {"a": [1, 2.5, "s", True, None, {"b": []}]}
    assert TranscriptionResult(text="x", extra={"a": nested}).extra == {"a": nested}
    assert Diagnostic(code="x", message="m", provided=nested).provided == nested

    # A TYPED container is JSON data but `list` is invariant, so a checker
    # rejects it where list[JsonValue] is expected. `to_json_value` absorbs
    # that once -- an engine author hands it a list[str] instead of writing a
    # cast at every call site -- and a structured value is dumped by the same
    # helper. Runtime validation is unaffected either way.
    hints: list[str] = ["Anthropic", "Claude"]
    assert Diagnostic(code="x", message="m", provided=to_json_value(hints)).provided == hints
    assert to_json_value(DIARIZE) == DIARIZE.model_dump(mode="json")


def test_rest_projection_failure_is_a_scrubbed_500_not_an_asgi_crash(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The response object is built inside the fault-mapping region, but
    # FastAPI serializes it AFTER the endpoint returns -- so a result whose
    # projection fails escaped every arm: an undocumented plain 500, and
    # log_exception_safely never ran.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    caplog.set_level(logging.ERROR)
    app = server_module.create_app(registry=_registry_for("_unprojectable_result_factory"))
    client = TestClient(app, raise_server_exceptions=False)
    resp: httpx2.Response = client.post(
        "/v1/transcribe:json",
        json={"model": "dummy/echo", "audio": base64.b64encode(_wav_bytes(rate=16000)).decode()},
    )
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal transcription error. See server logs for details."
    assert "Transcription failed for model" in caplog.text


def test_ws_initial_diagnostics_projection_failure_is_contained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The initial diagnostics frame was built between two boundaries: the
    # establishment try/except had already returned, and the forward loop's
    # catch had not started. A diagnostic with no JSON form killed the route
    # with an unhandled exception, bypassing the operator-log redaction.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    caplog.set_level(logging.ERROR)
    app = server_module.create_app(registry=_registry_for("_unprojectable_diag_factory"))
    client = TestClient(app)
    with client.websocket_connect("/v1/stream/dummy/echo") as ws:
        ws.send_json({"audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000}})
        frame = ws.receive_json()
    assert frame["type"] == "error"
    assert frame["code"] == "internal_error"
    assert "See server logs" in frame["message"]
    assert "diagnostics projection failed" in caplog.text
