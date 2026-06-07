# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard ASR wrapper for faster-whisper.

A realistic local-inference adapter on the new interface: engine-specific
decoding knobs live in a :class:`ProviderParams` subclass, while the portable
standard set (``language`` / ``word_timestamps`` / ``prompt`` / ``phrase_hints``)
maps onto faster-whisper's native arguments. faster-whisper accepts both decoded
arrays and file paths, so it declares ``{array, encoded_file}`` and the standard
layer passes through whichever the application provided.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterable, Literal, Sequence, cast

from pydantic import Field

from standard_asr import (
    BaseConfig,
    BaseProperties,
    DeviceConfigMixin,
    DownloadConfigMixin,
    EngineBase,
    InputKind,
    PreparedAudio,
    RuntimeParams,
    TranscriptionResult,
)
from standard_asr.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    FlagCap,
    GuidanceCaps,
    LanguageCaps,
    PhraseHintsCap,
    PromptCap,
    WordTimestampsCap,
)
from standard_asr.exceptions import DiscoveryError, TranscriptionError
from standard_asr.language import effective_language, normalize_bcp47
from standard_asr.results import Segment, Word
from standard_asr.runtime import allow_downloads
from standard_asr.runtime_params import ProviderParams

# A representative subset of Whisper's languages (it supports ~99).
_LANGUAGES = ["en", "zh", "es", "fr", "de", "ja", "ko", "ru", "pt", "it"]


class FasterWhisperConfig(
    DeviceConfigMixin, DownloadConfigMixin, BaseConfig[Literal["faster-whisper"]]
):
    """Init configuration for the faster-whisper engine.

    Args:
        engine: Discriminator value for the engine.
        model_path: Model size/name or local path.
        device: Compute device (cpu, cuda, auto).
        device_index: Device index or list of indices.
        compute_type: Quantization/precision type.
        cpu_threads: Number of CPU threads to use.
        num_workers: Number of worker threads for parallel inference.
        default_language: Default language (``"auto"`` for detection).
        local_files_only: Disable downloads when ``True``.
        revision: Optional Hugging Face revision.
    """

    engine: Literal["faster-whisper"] = "faster-whisper"
    model_path: str = Field(default="large-v3", description="Model size/name or local path.")
    device: str | None = Field(default="auto", description="Compute device (cpu, cuda, auto).")
    device_index: int | list[int] = Field(default=0, description="Device index/indices.")
    compute_type: str = Field(default="default", description="Quantization/precision type.")
    cpu_threads: int = Field(default=0, description="CPU threads (0 = runtime default).")
    num_workers: int = Field(default=1, description="Worker threads for parallel inference.")
    default_language: str | None = Field(default="auto", description="Default language or 'auto'.")
    local_files_only: bool = Field(default=False, description="Disable downloads when True.")
    revision: str | None = Field(default=None, description="Optional HF model revision.")


class FasterWhisperParams(ProviderParams):
    """Engine-specific decoding knobs for faster-whisper (non-portable).

    Args:
        task: ``"transcribe"`` (default) or ``"translate"`` (translate speech to
            English). Whisper-native; not a portable standard-set parameter, so
            it lives here. Without it, translation would be impossible.
        beam_size: Beam size for decoding.
        best_of: Candidates sampled when temperature > 0.
        patience: Beam search patience.
        length_penalty: Length penalty for decoding.
        temperature: Sampling temperature(s).
        compression_ratio_threshold: Compression ratio threshold.
        log_prob_threshold: Log-probability threshold.
        no_speech_threshold: No-speech probability threshold.
        condition_on_previous_text: Use previous text as prompt.
        vad_filter: Enable VAD filtering.
        vad_parameters: Optional VAD configuration dict.
    """

    task: Literal["transcribe", "translate"] = "transcribe"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    temperature: float | Sequence[float] | None = None
    compression_ratio_threshold: float | None = 2.4
    log_prob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6
    condition_on_previous_text: bool = True
    vad_filter: bool = False
    vad_parameters: dict[str, Any] | None = None


class FasterWhisperProperties(BaseProperties):
    """Static metadata describing the faster-whisper engine."""

    engine_id: str = "faster-whisper"
    model_name: str = "whisper"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY, InputKind.ENCODED_FILE}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | Literal["any"] = [16000]
    selectable_languages: list[str] = ["auto", *_LANGUAGES]
    detectable_languages: list[str] = list(_LANGUAGES)
    description: str | None = "Standard ASR wrapper for faster-whisper."


_CAPABILITIES = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        # faster-whisper only produces word-level timestamps (when
        # word_timestamps=True); segment start/end always exist but there is no
        # distinct "segment" granularity mode, and no char-level support.
        word_timestamps=WordTimestampsCap(supported=True, granularities=["word"]),
        guidance=GuidanceCaps(
            prompt=PromptCap(supported=True),
            phrase_hints=PhraseHintsCap(supported=True),
        ),
    )
)


class FasterWhisperASR(EngineBase):
    """Standard ASR adapter for faster-whisper.

    Args:
        **kwargs: Configuration overrides for :class:`FasterWhisperConfig`.
    """

    properties: ClassVar[BaseProperties] = FasterWhisperProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _CAPABILITIES
    provider_params_type: ClassVar[type[ProviderParams] | None] = FasterWhisperParams

    def __init__(self, **kwargs: Any) -> None:
        """Capture configuration (pure; weights load lazily).

        Config is built via ``from_env``: unset fields fall back to
        ``STANDARD_ASR_FASTER_WHISPER_*`` environment variables (spec IC.4) and
        explicit ``kwargs`` win. Credentials (none here, but e.g. an HF token)
        are wrapped in ``SecretStr`` by construction, never passed as plaintext.

        Args:
            **kwargs: Configuration overrides.
        """
        self.config = FasterWhisperConfig.from_env("faster-whisper", **kwargs)
        self._model: object | None = None

    def _ensure_model_loaded(self) -> None:
        """Load the faster-whisper model lazily.

        Raises:
            DiscoveryError: If the library is missing or weights cannot load.
        """
        if self._model is not None:
            return
        try:
            from faster_whisper import (  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]
                WhisperModel,  # pyright: ignore[reportUnknownVariableType]
            )
        except Exception as exc:  # noqa: BLE001
            raise DiscoveryError(
                "faster-whisper is not installed. Install 'faster-whisper'."
            ) from exc

        config = cast(FasterWhisperConfig, self.config)
        local_only = config.local_files_only or not allow_downloads()
        try:
            self._model = WhisperModel(
                model_size_or_path=config.model_path,
                device=config.device or "auto",
                device_index=config.device_index,
                compute_type=config.compute_type,
                cpu_threads=config.cpu_threads,
                num_workers=config.num_workers,
                download_root=str(config.download_root) if config.download_root else None,
                local_files_only=local_only,
                revision=config.revision,
            )
        except Exception as exc:  # noqa: BLE001
            raise DiscoveryError(
                "Failed to load faster-whisper model. If downloads are disabled, "
                "set STANDARD_ASR_ALLOW_DOWNLOAD=1 or pre-download the model."
            ) from exc

    def prepare(self) -> None:
        """Preload model weights without transcribing.

        Raises:
            DiscoveryError: If weights cannot be loaded.
        """
        self._ensure_model_loaded()

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        """Transcribe negotiated audio with faster-whisper.

        Args:
            prepared: Engine-ready audio (an array or a file path).
            params: Gated runtime parameters.

        Returns:
            A Standard ASR transcription result.

        Raises:
            TranscriptionError: If the model fails to load.
        """
        self._ensure_model_loaded()
        if self._model is None:  # pragma: no cover - defensive
            raise TranscriptionError("Model failed to load.")

        config = cast(FasterWhisperConfig, self.config)
        resolved = effective_language(
            params.language,
            config.default_language,
            has_language_axis=True,
            runtime_override_supported=True,
        )
        language = None
        if resolved and resolved != "auto":
            language = normalize_bcp47(resolved).split("-", maxsplit=1)[0]

        if prepared.array is not None:
            # We declare accepted_sample_rates=[16000]; the standard layer
            # negotiates to it, but assert defensively -- feeding faster-whisper
            # an off-rate array silently produces wrong timings/text.
            assert prepared.sample_rate == 16000, (
                f"faster-whisper requires 16 kHz audio; got "
                f"{prepared.sample_rate} Hz (audio negotiation should have "
                "resampled to 16000)."
            )
        source: Any = prepared.array if prepared.array is not None else prepared.path
        model = cast(Any, self._model)
        segments, info = model.transcribe(
            source,
            language=language,
            word_timestamps=params.word_timestamps is not None,
            initial_prompt=params.prompt,
            hotwords=" ".join(params.phrase_hints) if params.phrase_hints else None,
            **_provider_kwargs(params.provider_params),
        )

        segment_list, word_list = _convert_segments(segments)
        text = "".join(seg.text for seg in segment_list)
        detected = normalize_bcp47(info.language) if info.language else None
        return TranscriptionResult(
            text=text,
            detected_language=detected,
            language_confidence=getattr(info, "language_probability", None),
            duration=info.duration,
            segments=segment_list or None,
            words=word_list if params.word_timestamps else None,
            metadata=_safe_metadata(info),
        )


def _convert_segments(segments: Iterable[Any]) -> tuple[list[Segment], list[Word]]:
    """Convert faster-whisper segments into Standard ASR models.

    Args:
        segments: faster-whisper segment iterator.

    Returns:
        A ``(segments, flattened_words)`` pair.
    """
    segment_list: list[Segment] = []
    word_list: list[Word] = []
    for segment in segments:
        words = None
        if segment.words:
            words = [
                Word(
                    start=word.start,
                    end=word.end,
                    text=word.word,
                    probability=word.probability,
                )
                for word in segment.words
            ]
            word_list.extend(words)
        segment_list.append(
            Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=words,
                temperature=segment.temperature,
                avg_logprob=segment.avg_logprob,
                compression_ratio=segment.compression_ratio,
                no_speech_prob=segment.no_speech_prob,
            )
        )
    return segment_list, word_list


#: Fields from faster-whisper's ``TranscriptionInfo`` / ``transcription_options``
#: that are safe to surface as result metadata. We deliberately exclude
#: ``initial_prompt`` / ``prefix`` / ``hotwords`` / ``suppress_tokens`` and other
#: large or sensitive inputs so the prompt text is not echoed back (privacy) and
#: the metadata stays small enough to carry over REST.
_SAFE_OPTION_FIELDS: tuple[str, ...] = (
    "task",
    "beam_size",
    "best_of",
    "patience",
    "length_penalty",
    "temperatures",
    "compression_ratio_threshold",
    "log_prob_threshold",
    "no_speech_threshold",
    "condition_on_previous_text",
    "word_timestamps",
)


def _safe_metadata(info: Any) -> dict[str, Any]:
    """Build whitelisted result metadata from a ``TranscriptionInfo``.

    Only small, non-sensitive decoding options are included -- never the prompt,
    hotwords, or other free-text inputs, which could leak user content or bloat
    the result when serialized over REST.

    Args:
        info: faster-whisper's ``TranscriptionInfo``.

    Returns:
        A JSON-friendly metadata mapping.
    """
    options = getattr(info, "transcription_options", None)
    safe: dict[str, Any] = {}
    if options is not None:
        for name in _SAFE_OPTION_FIELDS:
            if hasattr(options, name):
                safe[name] = getattr(options, name)
    metadata: dict[str, Any] = {"transcription_options": safe}
    vad = getattr(info, "duration_after_vad", None)
    if vad is not None:
        metadata["duration_after_vad"] = vad
    return metadata


def _provider_kwargs(params: ProviderParams | None) -> dict[str, Any]:
    """Convert provider params into faster-whisper keyword arguments.

    Args:
        params: The engine-specific parameters, if any.

    Returns:
        Keyword arguments for ``WhisperModel.transcribe``.
    """
    if params is None:
        return {}
    fw = cast(FasterWhisperParams, params)
    kwargs: dict[str, Any] = {
        "task": fw.task,
        "beam_size": fw.beam_size,
        "best_of": fw.best_of,
        "patience": fw.patience,
        "length_penalty": fw.length_penalty,
        "compression_ratio_threshold": fw.compression_ratio_threshold,
        "log_prob_threshold": fw.log_prob_threshold,
        "no_speech_threshold": fw.no_speech_threshold,
        "condition_on_previous_text": fw.condition_on_previous_text,
        "vad_filter": fw.vad_filter,
        "vad_parameters": fw.vad_parameters,
    }
    if fw.temperature is not None:
        kwargs["temperature"] = fw.temperature
    return kwargs
