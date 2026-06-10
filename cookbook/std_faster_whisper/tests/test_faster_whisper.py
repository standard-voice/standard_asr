# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the faster-whisper Standard ASR cookbook adapter.

Every test runs against the injected ``FakeWhisperModel`` (see conftest); the
real faster-whisper model is never instantiated and no weights are downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from std_faster_whisper import (
    DistilLargeV3ASR,
    FasterWhisperASR,
    FasterWhisperConfig,
    FasterWhisperParams,
    FasterWhisperProperties,
    TurboASR,
    create,
    create_distil_large_v3,
    create_turbo,
)
from std_faster_whisper.entrypoint import create as entrypoint_create
from std_faster_whisper.std_asr_faster_whisper import (
    _convert_segments,  # pyright: ignore[reportPrivateUsage]
    _provider_kwargs,  # pyright: ignore[reportPrivateUsage]
    _safe_extra,  # pyright: ignore[reportPrivateUsage]
)

from standard_asr import RuntimeParams, StandardASR
from standard_asr.audio_input import AudioArray, AudioPath
from standard_asr.capabilities import (
    DeclaredCapabilities,
    PhraseHintsCap,
    PromptCap,
    WordTimestampsCap,
)
from standard_asr.exceptions import DiscoveryError, UnsupportedFeatureError
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
    # The base engine is the canonical large-v3 preset; its model_id
    # equals the entry-point key (compliance-enforced).
    assert FasterWhisperASR.properties.model_id == "faster-whisper/large-v3"


def test_config_defaults() -> None:
    config = FasterWhisperConfig()
    assert config.engine == "faster-whisper"
    # Model_path is a local override (None by default), not the
    # model selector -- the preset picks the model.
    assert config.model_path is None
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
    # An explicit model_path is a local override and wins over the
    # preset's model_size.
    assert fake_faster_whisper.last_init_kwargs["model_size_or_path"] == "tiny"


def test_preset_loads_its_model_size_by_default(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # With no model_path override, each preset loads the model its
    # class selects (model_size) -- the model is chosen by the preset, not a
    # config field.
    FasterWhisperASR().prepare()
    assert fake_faster_whisper.last_init_kwargs["model_size_or_path"] == "large-v3"
    DistilLargeV3ASR().prepare()
    assert fake_faster_whisper.last_init_kwargs["model_size_or_path"] == "distil-large-v3"
    TurboASR().prepare()
    assert fake_faster_whisper.last_init_kwargs["model_size_or_path"] == "large-v3-turbo"


def test_init_passes_download_root_and_local_only(
    fake_faster_whisper: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # When downloads are globally disabled, local_files_only is forced True even
    # if the config left it False; the explicit download_root wins over the
    # STANDARD_ASR_MODEL_DIR tier (spec IC.9) and is stringified.
    env_dir = tmp_path / "ignored-env-models"
    explicit_dir = tmp_path / "models"
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    monkeypatch.setenv("STANDARD_ASR_MODEL_DIR", str(env_dir))
    engine = FasterWhisperASR(model_path="tiny", download_root=str(explicit_dir))
    engine.prepare()
    assert fake_faster_whisper.last_init_kwargs["local_files_only"] is True
    assert fake_faster_whisper.last_init_kwargs["download_root"] == str(explicit_dir)


def test_init_download_root_honors_standard_model_dir_env(
    fake_faster_whisper: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Spec IC.9 second tier: with no explicit download_root,
    # STANDARD_ASR_MODEL_DIR governs where model artifacts land.
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("STANDARD_ASR_MODEL_DIR", str(tmp_path))
    FasterWhisperASR(model_path="tiny").prepare()
    assert fake_faster_whisper.last_init_kwargs["download_root"] == str(tmp_path)


def test_init_download_root_defers_to_library_default(
    fake_faster_whisper: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Spec IC.9 third tier: no explicit download_root, no
    # STANDARD_ASR_MODEL_DIR -> defer to faster-whisper's OWN default cache by
    # passing download_root=None through (WhisperModel resolves it via the
    # HuggingFace hub cache). Forcing a concrete directory here would break
    # offline loads of hub-cached models and silently re-download them.
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "1")
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
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


def test_transcribe_wraps_engine_failure_as_transcription_error(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # spec Runtime R7: a native faster-whisper failure during
    # inference MUST surface as a portable TranscriptionError, preserving the
    # original exception as __cause__, so applications catch one type across
    # engines instead of faster-whisper's native error.
    from standard_asr.exceptions import TranscriptionError

    boom = RuntimeError("CUDA out of memory")
    fake_faster_whisper.raise_on_transcribe = boom

    with pytest.raises(TranscriptionError) as exc_info:
        FasterWhisperASR(model_path="tiny").transcribe(_audio(), RuntimeParams(language="en"))
    assert exc_info.value.__cause__ is boom


def test_transcribe_region_tagged_language_uses_primary_subtag(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # faster-whisper wants a primary subtag, so a region-tagged BCP-47 language
    # (en-US) must be reduced to its primary subtag (en) before forwarding.
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "hi")]
    FasterWhisperASR(model_path="tiny").transcribe(_audio(), RuntimeParams(language="en-US"))
    assert fake_faster_whisper.last_transcribe_kwargs["language"] == "en"


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


def test_segment_granularity_is_declared() -> None:
    # Faster-whisper emits per-segment start/end for free, so it MUST
    # declare the "segment" granularity (alongside "word"); omitting it would make
    # the cheapest, always-satisfiable request a false incompatibility.
    node = FasterWhisperASR.declared_capabilities.node_at("batch.word_timestamps")
    assert isinstance(node, WordTimestampsCap)
    assert set(node.granularities) == {"word", "segment"}


def test_transcribe_segment_granularity_does_not_request_words(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # A SEGMENT request must (a) NOT be rejected (it is now offered),
    # (b) leave upstream word_timestamps=False (no extra forced-alignment pass),
    # and (c) NOT back-fill word-level data the caller did not ask for
    # (words=None), while segment timestamps remain available.
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "Hi there")]
    result = FasterWhisperASR(model_path="tiny").transcribe(
        _audio(), RuntimeParams(language="en", word_timestamps=WordTimestampGranularity.SEGMENT)
    )
    assert fake_faster_whisper.last_transcribe_kwargs["word_timestamps"] is False
    assert result.words is None
    assert result.segments is not None
    assert result.segments[0].start == pytest.approx(0.0)
    assert result.segments[0].end == pytest.approx(1.0)


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


def test_guidance_constraints_are_declared() -> None:
    # Faster-whisper silently truncates over-budget guidance at ~223
    # tokens. The adapter MUST declare conservative limits so the standard layer
    # can gate BEFORE the engine eats the overflow.
    node = FasterWhisperASR.declared_capabilities.node_at("batch.guidance.prompt")
    assert isinstance(node, PromptCap)
    assert node.constraints.max_tokens == 200
    hints = FasterWhisperASR.declared_capabilities.node_at("batch.guidance.phrase_hints")
    assert isinstance(hints, PhraseHintsCap)
    assert hints.constraints.max_terms == 50
    assert hints.constraints.max_chars_per_term == 40


def test_over_budget_prompt_fails_loud_in_strict_mode(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # A prompt over the declared max_tokens is rejected at the gate
    # (strict default) instead of being silently truncated by faster-whisper.
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    long_prompt = " ".join(["word"] * 201)  # 201 > max_tokens=200
    with pytest.raises(UnsupportedFeatureError, match="prompt"):
        FasterWhisperASR(model_path="tiny").transcribe(
            _audio(), RuntimeParams(language="en", prompt=long_prompt)
        )


def test_over_budget_prompt_truncated_with_diagnostic_in_best_effort(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # In best_effort the standard layer truncates to the declared
    # budget and emits a diagnostic, and the engine receives the truncated prompt
    # (never the silently-cut original).
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    long_prompt = " ".join(["word"] * 201)
    result = FasterWhisperASR(model_path="tiny", strict=False).transcribe(
        _audio(), RuntimeParams(language="en", prompt=long_prompt)
    )
    forwarded = fake_faster_whisper.last_transcribe_kwargs["initial_prompt"]
    assert len(forwarded.split()) == 200
    assert any(d.code == "prompt_truncated" for d in result.diagnostics)


def test_over_limit_phrase_hints_fail_loud_in_strict_mode(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # More than max_terms phrase hints is rejected at the gate
    # (strict default) rather than silently truncated by the engine.
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    too_many = [f"term{i}" for i in range(51)]  # 51 > max_terms=50
    with pytest.raises(UnsupportedFeatureError, match="phrase_hints"):
        FasterWhisperASR(model_path="tiny").transcribe(
            _audio(), RuntimeParams(language="en", phrase_hints=too_many)
        )


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


def test_transcribe_from_bytes_uses_binary_file_like(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    import io
    import wave

    from standard_asr.audio_input import AudioBytes

    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "from bytes")]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(16, dtype=np.int16).tobytes())

    result = FasterWhisperASR(model_path="tiny").transcribe(
        AudioBytes(buf.getvalue()), RuntimeParams(language="en")
    )
    assert result.text == "from bytes"
    # encoded_bytes is accepted, so the bytes pass through as a binary file-like.
    assert isinstance(fake_faster_whisper.last_transcribe_kwargs["source"], io.BytesIO)


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


def test_safe_extra_whitelists_options() -> None:
    extra = _safe_extra(FakeInfo(duration_after_vad=0.9))
    opts = extra["transcription_options"]
    assert opts["task"] == "transcribe"
    assert "initial_prompt" not in opts  # never echoed back
    assert extra["duration_after_vad"] == pytest.approx(0.9)


def test_safe_extra_without_options_or_vad() -> None:
    extra = _safe_extra(FakeInfo(with_options=False, duration_after_vad=None))
    assert extra["transcription_options"] == {}
    assert "duration_after_vad" not in extra


def test_safe_extra_skips_absent_whitelisted_fields() -> None:
    # An older faster-whisper whose options lack some whitelisted fields: the
    # hasattr guard skips them rather than raising (the False branch).
    class _PartialOptions:
        beam_size = 7  # only one of the whitelisted fields present

    class _Info:
        transcription_options = _PartialOptions()
        duration_after_vad = None

    extra = _safe_extra(_Info())
    assert extra["transcription_options"] == {"beam_size": 7}


def test_transcribe_engine_specific_data_goes_to_extra_not_metadata(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # Faster-whisper's private decoding options / duration_after_vad
    # are engine-specific, so they land in result.extra; metadata (reserved for
    # standardized engine-agnostic keys, spec TR.1) stays empty.
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    fake_faster_whisper.info = FakeInfo(language="en", duration_after_vad=0.8)
    result = FasterWhisperASR(model_path="tiny").transcribe(_audio(), RuntimeParams(language="en"))
    assert result.metadata == {}
    assert result.extra["transcription_options"]["task"] == "transcribe"
    assert result.extra["duration_after_vad"] == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Entry point / presets
# --------------------------------------------------------------------------- #
def test_create_returns_engine() -> None:
    assert isinstance(create(), FasterWhisperASR)
    assert isinstance(entrypoint_create(model_path="tiny"), FasterWhisperASR)


def test_preset_factories_return_their_classes() -> None:
    # Each preset has its own factory returning its concrete class
    # so the registry can resolve the class (and its model_id) without
    # instantiation.
    assert type(create()) is FasterWhisperASR
    assert type(create_distil_large_v3()) is DistilLargeV3ASR
    assert type(create_turbo()) is TurboASR


def test_preset_model_ids_match_entry_point_keys() -> None:
    # + compliance (model_id == key): every preset's properties.model_id
    # equals the registered entry-point key so discovery can enumerate models.
    assert FasterWhisperASR.properties.model_id == "faster-whisper/large-v3"
    assert DistilLargeV3ASR.properties.model_id == "faster-whisper/distil-large-v3"
    assert TurboASR.properties.model_id == "faster-whisper/turbo"
