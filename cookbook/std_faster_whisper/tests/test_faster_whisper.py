# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the faster-whisper Standard ASR cookbook adapter.

Every test runs against the injected ``FakeWhisperModel`` (see conftest); the
real faster-whisper model is never instantiated and no weights are downloaded.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from std_faster_whisper import (
    FasterWhisperASR,
    FasterWhisperConfig,
    FasterWhisperParams,
    FasterWhisperProperties,
    create,
)
from std_faster_whisper.entrypoint import create as entrypoint_create
from std_faster_whisper.main import main
from std_faster_whisper.std_asr_faster_whisper import (
    _convert_segments,  # pyright: ignore[reportPrivateUsage]
    _provider_kwargs,  # pyright: ignore[reportPrivateUsage]
    _safe_metadata,  # pyright: ignore[reportPrivateUsage]
)

from standard_asr import RuntimeParams, StandardASR
from standard_asr.audio_input import AudioArray, AudioPath
from standard_asr.capabilities import DeclaredCapabilities
from standard_asr.exceptions import DiscoveryError
from standard_asr.runtime_params import WordTimestampGranularity

from .conftest import FakeInfo, FakeSegment, FakeWhisperModel, FakeWord


def _audio(n: int = 16000) -> AudioArray:
    return AudioArray(np.zeros(n, dtype=np.float32), 16000)


# --------------------------------------------------------------------------- #
# Static metadata / config / params
# --------------------------------------------------------------------------- #
def test_engine_is_standard_asr() -> None:
    assert isinstance(FasterWhisperASR(), StandardASR)


def test_class_level_metadata() -> None:
    assert isinstance(FasterWhisperASR.properties, FasterWhisperProperties)
    assert isinstance(FasterWhisperASR.declared_capabilities, DeclaredCapabilities)
    assert FasterWhisperASR.provider_params_type is FasterWhisperParams
    assert FasterWhisperASR.properties.model_id == "faster-whisper/whisper"


def test_config_defaults() -> None:
    config = FasterWhisperConfig()
    assert config.engine == "faster-whisper"
    assert config.model_path == "large-v3"
    assert config.default_language == "auto"
    assert config.local_files_only is False


def test_provider_params_defaults() -> None:
    params = FasterWhisperParams()
    assert params.task == "transcribe"
    assert params.beam_size == 5
    assert params.temperature is None


# --------------------------------------------------------------------------- #
# Model loading (lazy)
# --------------------------------------------------------------------------- #
def test_ensure_model_loaded_missing_library(monkeypatch: pytest.MonkeyPatch) -> None:
    # No faster_whisper module installed -> DiscoveryError with install hint.
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)

    import builtins

    real_import = builtins.__import__

    def _import(name: str, *a: object, **k: object) -> object:
        if name == "faster_whisper" or name.startswith("faster_whisper."):
            raise ImportError("no faster_whisper")
        return real_import(name, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(DiscoveryError, match="not installed"):
        FasterWhisperASR().prepare()


def test_ensure_model_loaded_init_failure(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # The library imports, but constructing the model fails (e.g. download
    # disabled) -> DiscoveryError with a remediation hint.
    fake_faster_whisper.raise_on_init = RuntimeError("weights missing")
    with pytest.raises(DiscoveryError, match="Failed to load"):
        FasterWhisperASR().prepare()


def test_prepare_loads_model_once(fake_faster_whisper: type[FakeWhisperModel]) -> None:
    engine = FasterWhisperASR(model_path="tiny", device="cpu")
    engine.prepare()
    assert engine._model is not None  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    first = engine._model  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    # A second prepare() is a no-op (does not rebuild the model).
    engine.prepare()
    assert engine._model is first  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert fake_faster_whisper.last_init_kwargs["model_size_or_path"] == "tiny"


def test_init_passes_download_root_and_local_only(
    fake_faster_whisper: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    # When downloads are globally disabled, local_files_only is forced True even
    # if the config left it False; download_root is stringified when set.
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    engine = FasterWhisperASR(model_path="tiny", download_root="/tmp/models")
    engine.prepare()
    assert fake_faster_whisper.last_init_kwargs["local_files_only"] is True
    assert fake_faster_whisper.last_init_kwargs["download_root"] == "/tmp/models"


def test_init_download_root_none_when_unset(
    fake_faster_whisper: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "1")
    FasterWhisperASR(model_path="tiny").prepare()
    assert fake_faster_whisper.last_init_kwargs["download_root"] is None
    assert fake_faster_whisper.last_init_kwargs["local_files_only"] is False


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
def test_transcribe_array_basic(fake_faster_whisper: type[FakeWhisperModel]) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "Hello world.")]
    fake_faster_whisper.info = FakeInfo(language="en")

    result = FasterWhisperASR(model_path="tiny").transcribe(_audio(), RuntimeParams(language="en"))
    assert result.text == "Hello world."
    assert result.detected_language == "en"
    assert result.language_confidence == pytest.approx(0.97)
    assert result.duration == pytest.approx(1.23)
    # An explicit language is forwarded (primary subtag only) and word_timestamps
    # defaults to False.
    kwargs = fake_faster_whisper.last_transcribe_kwargs
    assert kwargs["language"] == "en"
    assert kwargs["word_timestamps"] is False


def test_transcribe_auto_language_sends_none(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "hi")]
    # default_language is "auto" -> no language forced (auto-detect).
    FasterWhisperASR(model_path="tiny").transcribe(_audio())
    assert fake_faster_whisper.last_transcribe_kwargs["language"] is None


def test_transcribe_with_word_timestamps(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    words = [FakeWord(0.0, 0.5, "Hi", 0.9), FakeWord(0.5, 1.0, "there", 0.8)]
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "Hi there", words=words)]

    result = FasterWhisperASR(model_path="tiny").transcribe(
        _audio(), RuntimeParams(language="en", word_timestamps=WordTimestampGranularity.WORD)
    )
    assert fake_faster_whisper.last_transcribe_kwargs["word_timestamps"] is True
    assert result.words is not None
    assert [w.text for w in result.words] == ["Hi", "there"]
    assert result.segments is not None
    assert result.segments[0].words is not None


def test_transcribe_with_prompt_and_phrase_hints(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    FasterWhisperASR(model_path="tiny").transcribe(
        _audio(),
        RuntimeParams(language="en", prompt="context", phrase_hints=["Anthropic", "Claude"]),
    )
    kwargs = fake_faster_whisper.last_transcribe_kwargs
    assert kwargs["initial_prompt"] == "context"
    assert kwargs["hotwords"] == "Anthropic Claude"


def test_transcribe_provider_params_forwarded(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    params = RuntimeParams(
        language="en",
        provider_params=FasterWhisperParams(task="translate", beam_size=3, temperature=[0.0, 0.2]),
    )
    FasterWhisperASR(model_path="tiny").transcribe(_audio(), params)
    kwargs = fake_faster_whisper.last_transcribe_kwargs
    assert kwargs["task"] == "translate"
    assert kwargs["beam_size"] == 3
    assert kwargs["temperature"] == [0.0, 0.2]


def test_transcribe_from_file_path(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: object
) -> None:
    import wave
    from pathlib import Path

    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "from file")]
    assert isinstance(tmp_path, Path)
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(16, dtype=np.int16).tobytes())

    result = FasterWhisperASR(model_path="tiny").transcribe(
        AudioPath(wav), RuntimeParams(language="en")
    )
    assert result.text == "from file"
    # encoded_file is accepted, so the path is passed straight through (no array).
    assert fake_faster_whisper.last_transcribe_kwargs["source"] == str(wav)


def test_transcribe_detected_language_none_when_unknown(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    fake_faster_whisper.info = FakeInfo(language=None)
    result = FasterWhisperASR(model_path="tiny").transcribe(_audio(), RuntimeParams(language="en"))
    assert result.detected_language is None


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_convert_segments_with_and_without_words() -> None:
    segs = [
        FakeSegment(0.0, 1.0, "a", words=[FakeWord(0.0, 0.5, "a", 0.9)]),
        FakeSegment(1.0, 2.0, "b", words=None),
    ]
    segments, words = _convert_segments(segs)
    assert len(segments) == 2
    assert [w.text for w in words] == ["a"]
    assert segments[0].words is not None
    assert segments[1].words is None


def test_provider_kwargs_none_returns_empty() -> None:
    assert _provider_kwargs(None) == {}


def test_provider_kwargs_omits_temperature_when_none() -> None:
    kwargs = _provider_kwargs(FasterWhisperParams(temperature=None))
    assert "temperature" not in kwargs
    assert kwargs["beam_size"] == 5


def test_provider_kwargs_includes_temperature_when_set() -> None:
    kwargs = _provider_kwargs(FasterWhisperParams(temperature=0.4))
    assert kwargs["temperature"] == 0.4


def test_safe_metadata_whitelists_options() -> None:
    meta = _safe_metadata(FakeInfo(duration_after_vad=0.9))
    opts = meta["transcription_options"]
    assert opts["task"] == "transcribe"
    assert "initial_prompt" not in opts  # never echoed back
    assert meta["duration_after_vad"] == pytest.approx(0.9)


def test_safe_metadata_without_options_or_vad() -> None:
    meta = _safe_metadata(FakeInfo(with_options=False, duration_after_vad=None))
    assert meta["transcription_options"] == {}
    assert "duration_after_vad" not in meta


def test_safe_metadata_skips_absent_whitelisted_fields() -> None:
    # An older faster-whisper whose options lack some whitelisted fields: the
    # hasattr guard skips them rather than raising (the False branch).
    class _PartialOptions:
        beam_size = 7  # only one of the whitelisted fields present

    class _Info:
        transcription_options = _PartialOptions()
        duration_after_vad = None

    meta = _safe_metadata(_Info())
    assert meta["transcription_options"] == {"beam_size": 7}


# --------------------------------------------------------------------------- #
# Entry point + console main
# --------------------------------------------------------------------------- #
def test_create_returns_engine() -> None:
    assert isinstance(create(), FasterWhisperASR)
    assert isinstance(entrypoint_create(model_path="tiny"), FasterWhisperASR)


def test_main_prints_greeting(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    assert "std-faster-whisper" in capsys.readouterr().out


def test_main_module_guard_executes(capsys: pytest.CaptureFixture[str]) -> None:
    # Run main.py with run_name="__main__" so the ``if __name__ == "__main__"``
    # guard executes main(); runpy runs the actual file, so coverage counts it.
    import importlib.util
    import runpy

    spec = importlib.util.find_spec("std_faster_whisper.main")
    assert spec is not None and spec.origin is not None
    runpy.run_path(spec.origin, run_name="__main__")
    assert "std-faster-whisper" in capsys.readouterr().out
