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

import io
from typing import Any, ClassVar, Iterable, Literal, Sequence, cast

from pydantic import Field

from standard_asr.engine import (
    BaseConfig,
    BaseProperties,
    BatchCapabilities,
    DeclaredCapabilities,
    DeviceConfigMixin,
    DownloadConfigMixin,
    EngineBase,
    FlagCap,
    GuidanceCaps,
    InputKind,
    LanguageCaps,
    PhraseHintsCap,
    PhraseHintsConstraints,
    PreparedAudio,
    PromptCap,
    PromptConstraints,
    ProviderParams,
    RuntimeParams,
    SampleRateRange,
    Segment,
    TranscriptionResult,
    Word,
    WordTimestampGranularity,
    WordTimestampsCap,
    allow_downloads,
    effective_language,
    normalize_bcp47,
    resolve_download_root,
)
from standard_asr.exceptions import DiscoveryError, TranscriptionError

# A representative subset of Whisper's languages (it supports ~99).
_LANGUAGES = ["en", "zh", "es", "fr", "de", "ja", "ko", "ru", "pt", "it"]


class FasterWhisperConfig(
    DeviceConfigMixin, DownloadConfigMixin, BaseConfig[Literal["faster-whisper"]]
):
    """Init configuration for the faster-whisper engine.

    Args:
        engine: Discriminator value for the engine.
        model_path: Optional path to a LOCAL checkpoint directory that overrides
            the preset's model. This is an init weights/path override (spec IC.7),
            NOT the model selector -- the model is chosen by which entry-point
            preset you instantiate (``faster-whisper/large-v3``,
            ``faster-whisper/distil-large-v3``, ...), never by passing a size name
            here. ``None`` (default) loads the preset's model from the Hub.
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
    model_path: str | None = Field(
        default=None,
        description=(
            "Optional local checkpoint directory overriding the preset's model "
            "(an init weights/path override, spec IC.7). The model is selected by "
            "the entry-point preset, not by this field; None loads the preset's model."
        ),
    )
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
    """Static metadata describing the faster-whisper engine.

    ``model_name`` MUST equal the entry-point preset key's model component so
    ``properties.model_id`` matches the registered key (compliance-enforced; see
    :mod:`standard_asr.compliance`). The base properties describe the canonical
    ``faster-whisper/large-v3`` preset; other presets subclass this and override
    ``model_name`` only.
    """

    engine_id: str = "faster-whisper"
    model_name: str = "large-v3"
    protocol_version: str = "1.0.0"
    # faster-whisper's transcribe() takes a path, a decoded array, or a binary
    # file-like, so the engine accepts all three. Declaring ENCODED_BYTES (not
    # just ENCODED_FILE) lets the Web API / cloud-style integrations hand it
    # in-memory uploads without a temp file.
    accepted_input: set[InputKind] = {
        InputKind.ARRAY,
        InputKind.ENCODED_FILE,
        InputKind.ENCODED_BYTES,
    }
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    selectable_languages: list[str] = ["auto", *_LANGUAGES]
    detectable_languages: list[str] = list(_LANGUAGES)
    description: str | None = "Standard ASR wrapper for faster-whisper."


_CAPABILITIES = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        # ``granularities`` declares which timestamp granularities the engine can
        # HONESTLY DELIVER -- not which upstream API switches exist. faster-whisper
        # emits per-segment start/end on EVERY transcription at zero cost (its
        # default is word_timestamps=False), so "segment" is always satisfiable
        # and MUST be declared; omitting it would make the standard layer reject a
        # legal, cheapest-cost request (spec §3.1, granularity is an enum value) on
        # an engine that can serve it -- a false incompatibility. "word" needs the
        # upstream word_timestamps=True pass; "char" is unsupported.
        word_timestamps=WordTimestampsCap(supported=True, granularities=["word", "segment"]),
        # faster-whisper SILENTLY truncates over-budget guidance: get_prompt caps
        # both the initial_prompt (previous_tokens[-(max_length//2-1):]) and the
        # encoded hotwords at ~223 tokens (max_length//2 - 1) with no signal.
        # Declaring constraints lets the standard layer fail-loud (strict) or
        # truncate+diagnose (best_effort) BEFORE the engine eats the overflow,
        # which is exactly the silent-degradation the guidance contract forbids
        # (spec §3.3 "never silently degrade"). The standard counts tokens with a conservative,
        # script-aware approximation (not Whisper's BPE), so these limits sit BELOW
        # the ~223 hard cap with headroom (spec §3.3 max_tokens guidance): a long
        # Latin word / URL is 1 unit here but several BPE tokens upstream.
        guidance=GuidanceCaps(
            prompt=PromptCap(supported=True, constraints=PromptConstraints(max_tokens=200)),
            phrase_hints=PhraseHintsCap(
                supported=True,
                # Hotwords are joined into one string that shares the ~223-token
                # budget; keep terms and per-term length conservative so the
                # combined hint set stays well under it.
                constraints=PhraseHintsConstraints(max_terms=50, max_chars_per_term=40),
            ),
        ),
    )
)


class FasterWhisperASR(EngineBase):
    """Standard ASR adapter for the ``faster-whisper/large-v3`` preset.

    This is the canonical large-v3 multilingual preset. Each Whisper variant is a
    SEPARATE preset with its own entry point (spec IC.7: model selection =
    entry-point preset, never an init ``model`` field), so the discovery layer
    (``models list`` / registry / UI) can enumerate the available models. Other
    variants subclass this and override :attr:`model_size` + :attr:`properties`
    only -- see :class:`DistilLargeV3ASR` / :class:`TurboASR`.

    Args:
        **kwargs: Configuration overrides for :class:`FasterWhisperConfig`.
    """

    #: The faster-whisper model id this preset loads (passed upstream as
    #: ``model_size_or_path``). Overridden per preset; a local ``model_path``
    #: config override (spec IC.7 weights/path) still wins when set.
    model_size: ClassVar[str] = "large-v3"

    properties: ClassVar[BaseProperties] = FasterWhisperProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _CAPABILITIES
    provider_params_type: ClassVar[type[ProviderParams] | None] = FasterWhisperParams
    config_type: ClassVar[type[BaseConfig[str]] | None] = FasterWhisperConfig

    def __init__(self, **kwargs: Any) -> None:
        """Capture configuration (pure; weights load lazily).

        Config is built via ``from_env``: unset fields fall back to
        ``STANDARD_ASR_FASTER_WHISPER__*`` environment variables (spec IC.4; the
        engine and field segments are joined by a double underscore) and explicit
        ``kwargs`` win. Credentials (none here, but e.g. an HF token) are wrapped
        in ``SecretStr`` by construction, never passed as plaintext.

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
        # Spec IC.9 precedence: explicit download_root > STANDARD_ASR_MODEL_DIR >
        # the library's own default cache > the shared standard cache.
        # faster-whisper HAS a library default: WhisperModel(download_root=None)
        # resolves via the HuggingFace hub cache, so the resolver's None
        # passthrough is forwarded unchanged. Forcing a concrete directory on an
        # unconfigured install would break offline loads of hub-cached models
        # and silently re-download them into a second cache.
        download_root = resolve_download_root(config.download_root, has_library_default=True)
        # Model selection is by preset (the class's model_size, spec IC.7); a
        # local model_path is an optional weights/path override that wins when set.
        model_source = config.model_path or type(self).model_size
        try:
            self._model = WhisperModel(
                model_size_or_path=model_source,
                device=config.device or "auto",
                device_index=config.device_index,
                compute_type=config.compute_type,
                cpu_threads=config.cpu_threads,
                num_workers=config.num_workers,
                download_root=None if download_root is None else str(download_root),
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
            TranscriptionError: If the model fails to load, or if faster-whisper
                raises during inference. The standard batch error contract (spec
                Runtime R7) requires an engine-execution failure to surface as a
                portable ``TranscriptionError`` (with the native exception kept as
                ``__cause__``) so applications can catch one type across engines.
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
        # Dispatch on the negotiated shape: array, file path, or in-memory bytes
        # (wrapped as a binary file-like, which faster-whisper accepts directly).
        source: Any
        if prepared.array is not None:
            source = prepared.array
        elif prepared.data is not None:
            source = io.BytesIO(prepared.data)
        else:
            source = prepared.path
        # Map the requested granularity ONTO faster-whisper's native switch:
        # only WORD requires the upstream word_timestamps=True pass (which runs
        # the extra forced-alignment). A SEGMENT request is served by the
        # always-present per-segment start/end, so it leaves word_timestamps=False
        # and never back-fills word-level data the caller did not ask for (spec
        # §3.1 null-semantics: words=None means "not requested").
        want_word_ts = params.word_timestamps == WordTimestampGranularity.WORD
        model = cast(Any, self._model)
        # faster-whisper returns a LAZY segment generator, so decode/inference
        # runs while _convert_segments consumes it -- both the transcribe() call
        # and the consumption are inside the wrap. Spec Runtime R7: a native
        # engine failure MUST surface as a portable TranscriptionError, preserving
        # the original exception as __cause__ (raise ... from exc).
        try:
            segments, info = model.transcribe(
                source,
                language=language,
                word_timestamps=want_word_ts,
                initial_prompt=params.prompt,
                hotwords=" ".join(params.phrase_hints) if params.phrase_hints else None,
                **_provider_kwargs(params.provider_params),
            )
            segment_list, word_list = _convert_segments(segments)
        except Exception as exc:  # noqa: BLE001 - normalized to the standard contract
            raise TranscriptionError(
                f"faster-whisper transcription failed: {type(exc).__name__}."
            ) from exc

        text = "".join(seg.text for seg in segment_list)
        detected = normalize_bcp47(info.language) if info.language else None
        return TranscriptionResult(
            text=text,
            detected_language=detected,
            language_confidence=getattr(info, "language_probability", None),
            duration=info.duration,
            segments=segment_list or None,
            words=word_list if want_word_ts else None,
            extra=_safe_extra(info),
        )


# --------------------------------------------------------------------------- #
# Additional presets. Each Whisper variant is its own entry point (spec IC.7),
# so the discovery layer can list every available model. A preset overrides only
# the model_name (for the matching properties.model_id) and the model_size (the
# upstream weights id); everything else -- config, params, capabilities, the
# transcribe pipeline -- is inherited unchanged. faster-whisper ships ~15 sizes;
# we register a representative few and document how to add more in the README.
# --------------------------------------------------------------------------- #
class DistilLargeV3Properties(FasterWhisperProperties):
    """Static metadata for the ``faster-whisper/distil-large-v3`` preset."""

    model_name: str = "distil-large-v3"
    description: str | None = "faster-whisper distil-large-v3 (faster, English-leaning)."


class DistilLargeV3ASR(FasterWhisperASR):
    """The ``faster-whisper/distil-large-v3`` preset (distilled, lower latency)."""

    model_size: ClassVar[str] = "distil-large-v3"
    properties: ClassVar[BaseProperties] = DistilLargeV3Properties()


class TurboProperties(FasterWhisperProperties):
    """Static metadata for the ``faster-whisper/turbo`` preset."""

    model_name: str = "turbo"
    description: str | None = "faster-whisper large-v3-turbo (fastest large preset)."


class TurboASR(FasterWhisperASR):
    """The ``faster-whisper/turbo`` preset (large-v3-turbo)."""

    model_size: ClassVar[str] = "large-v3-turbo"
    properties: ClassVar[BaseProperties] = TurboProperties()


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
#: that are safe to surface in the result's ``extra``. We deliberately exclude
#: ``initial_prompt`` / ``prefix`` / ``hotwords`` / ``suppress_tokens`` and other
#: large or sensitive inputs so the prompt text is not echoed back (privacy) and
#: the payload stays small enough to carry over REST.
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


def _safe_extra(info: Any) -> dict[str, Any]:
    """Build the whitelisted engine-specific ``extra`` from a ``TranscriptionInfo``.

    These are faster-whisper-private values -- the decoding knobs the run used
    (``transcription_options``) and the post-VAD duration -- not standardized
    cross-engine metadata, so per spec TR.1 they belong in ``result.extra``, the
    engine-specific / experimental channel, never in ``result.metadata`` (which is
    reserved for engine-agnostic standardized keys). Only small, non-sensitive
    decoding options are included -- never the prompt, hotwords, or other
    free-text inputs, which could leak user content or bloat the result when
    serialized over REST.

    Args:
        info: faster-whisper's ``TranscriptionInfo``.

    Returns:
        A JSON-friendly mapping for ``TranscriptionResult.extra``.
    """
    options = getattr(info, "transcription_options", None)
    safe: dict[str, Any] = {}
    if options is not None:
        for name in _SAFE_OPTION_FIELDS:
            if hasattr(options, name):
                safe[name] = getattr(options, name)
    extra: dict[str, Any] = {"transcription_options": safe}
    vad = getattr(info, "duration_after_vad", None)
    if vad is not None:
        extra["duration_after_vad"] = vad
    return extra


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
