# Copyright 2025 The Standard ASR Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standard ASR wrapper for faster-whisper."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, ClassVar, Iterable, Literal, Sequence, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from standard_asr import BaseConfig, BaseTranscribeOptions, StandardASR, TranscriptionResult
from standard_asr.asr_properties import BaseProperties
from standard_asr.exceptions import DiscoveryError, TranscriptionError
from standard_asr.features import FeatureFlag
from standard_asr.language import normalize_bcp47
from standard_asr.options import coerce_options
from standard_asr.results import Segment, Word
from standard_asr.runtime import allow_downloads, validate_audio_input

if TYPE_CHECKING:  # pragma: no cover
    from faster_whisper import WhisperModel


class FasterWhisperConfig(BaseConfig[Literal["faster-whisper"]]):
    """Configuration for the faster-whisper engine.

    Args:
        engine: Discriminator value for the engine.
        model_path: Model size/name or local path.
        device: Compute device (cpu, cuda, auto).
        device_index: Device index or list of indices.
        compute_type: Quantization/precision type.
        cpu_threads: Number of CPU threads to use.
        num_workers: Number of worker threads for parallel inference.
        download_root: Optional download/cache directory.
        local_files_only: Disable downloads when ``True``.
        revision: Optional Hugging Face revision.
        use_auth_token: Optional Hugging Face auth token.

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    engine: Literal["faster-whisper"] = "faster-whisper"

    model_path: str = Field(
        "large-v3",
        description="Model size/name or local path for faster-whisper.",
    )
    device: str = Field(
        "auto", description="Compute device (cpu, cuda, auto)."
    )
    device_index: int | list[int] = Field(
        0, description="Device index or list of indices."
    )
    compute_type: str = Field(
        "default", description="Quantization/precision type."
    )
    cpu_threads: int = Field(
        0,
        description="CPU threads for inference (0 uses runtime defaults).",
    )
    num_workers: int = Field(
        1, description="Number of worker threads for parallel inference."
    )
    download_root: str | None = Field(
        None, description="Optional download/cache directory."
    )
    local_files_only: bool = Field(
        False, description="Disable downloads when True."
    )
    revision: str | None = Field(
        None, description="Optional Hugging Face model revision."
    )
    use_auth_token: str | bool | None = Field(
        None, description="Optional Hugging Face authentication token."
    )


class FasterWhisperOptions(BaseTranscribeOptions):
    """Transcription options for faster-whisper.

    Args:
        beam_size: Beam size for decoding.
        best_of: Candidates sampled when temperature > 0.
        patience: Beam search patience.
        length_penalty: Length penalty for decoding.
        repetition_penalty: Repetition penalty (>1 discourages repeats).
        no_repeat_ngram_size: N-gram repetition size.
        temperature: Sampling temperature(s).
        compression_ratio_threshold: Compression ratio threshold.
        log_prob_threshold: Log-probability threshold.
        no_speech_threshold: No-speech probability threshold.
        condition_on_previous_text: Use previous text as prompt.
        prompt_reset_on_temperature: Reset prompt if temperature exceeds this value.
        initial_prompt: Optional text or token IDs prompt.
        prefix: Optional prefix text.
        suppress_blank: Suppress blank outputs at start.
        suppress_tokens: Tokens to suppress.
        without_timestamps: Disable timestamps entirely.
        max_initial_timestamp: Maximum initial timestamp.
        word_timestamps: Enable word-level timestamps.
        prepend_punctuations: Punctuation to prepend when word timestamps enabled.
        append_punctuations: Punctuation to append when word timestamps enabled.
        multilingual: Enable per-segment language detection.
        vad_filter: Enable VAD filtering.
        vad_parameters: Optional VAD configuration dict.
        max_new_tokens: Max new tokens per chunk.
        chunk_length: Override chunk length (seconds).
        clip_timestamps: Timestamp ranges to process.
        hallucination_silence_threshold: Hallucination silence threshold.
        hotwords: Hotwords/prompt phrases.
        language_detection_threshold: Language detection confidence threshold.
        language_detection_segments: Segments to consider for language detection.

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    beam_size: int = Field(5, description="Beam size for decoding.")
    best_of: int = Field(5, description="Candidates sampled when temperature > 0.")
    patience: float = Field(1.0, description="Beam search patience.")
    length_penalty: float = Field(1.0, description="Length penalty for decoding.")
    repetition_penalty: float = Field(1.0, description="Repetition penalty.")
    no_repeat_ngram_size: int = Field(0, description="N-gram repetition size.")
    temperature: float | Sequence[float] | None = Field(
        None, description="Sampling temperature(s)."
    )
    compression_ratio_threshold: float | None = Field(
        2.4, description="Compression ratio threshold."
    )
    log_prob_threshold: float | None = Field(
        -1.0, description="Log probability threshold."
    )
    no_speech_threshold: float | None = Field(
        0.6, description="No speech probability threshold."
    )
    condition_on_previous_text: bool = Field(
        True, description="Use previous text as prompt."
    )
    prompt_reset_on_temperature: float = Field(
        0.5, description="Prompt reset threshold for temperature."
    )
    initial_prompt: str | Iterable[int] | None = Field(
        None, description="Optional prompt text or token IDs."
    )
    prefix: str | None = Field(None, description="Optional prefix text.")
    suppress_blank: bool = Field(True, description="Suppress blank outputs.")
    suppress_tokens: list[int] | None = Field(
        default_factory=lambda: [-1],
        description="Token IDs to suppress.",
    )
    without_timestamps: bool = Field(False, description="Disable timestamps.")
    max_initial_timestamp: float = Field(1.0, description="Max initial timestamp.")
    word_timestamps: bool = Field(False, description="Enable word timestamps.")
    prepend_punctuations: str = Field(
        "\"'“¿([{-", description="Punctuations to prepend."
    )
    append_punctuations: str = Field(
        "\"'.。,，!！?？:：”)]}、", description="Punctuations to append."
    )
    multilingual: bool = Field(False, description="Enable per-segment language detection.")
    vad_filter: bool = Field(False, description="Enable VAD filtering.")
    vad_parameters: dict[str, Any] | None = Field(
        None, description="Optional VAD parameters dict."
    )
    max_new_tokens: int | None = Field(
        None, description="Max new tokens per chunk."
    )
    chunk_length: float | None = Field(
        None, description="Override chunk length (seconds)."
    )
    clip_timestamps: str | list[float] = Field(
        "0", description="Clip timestamps to process."
    )
    hallucination_silence_threshold: float | None = Field(
        None, description="Hallucination silence threshold."
    )
    hotwords: str | None = Field(None, description="Hotwords/hint phrases.")
    language_detection_threshold: float = Field(
        0.5, description="Language detection confidence threshold."
    )
    language_detection_segments: int = Field(
        1, description="Segments to consider for language detection."
    )


class FasterWhisperProperties(BaseProperties):
    """Static metadata describing the faster-whisper engine.

    Args:
        None.

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    engine_id: str = "faster-whisper"
    model_name: str = "whisper"
    protocol_version: str = "0.2.0"
    supported_languages: list[str] = ["und"]
    supported_devices: list[str] = ["cpu", "cuda"]
    supported_sample_rates: list[int] = [16000]
    supported_channels: list[int] = [1]
    audio_dtype: str = "float32"
    features: set[FeatureFlag] = {
        FeatureFlag.WORD_TIMESTAMPS,
        FeatureFlag.LANGUAGE_DETECTION,
        FeatureFlag.TRANSLATION,
        FeatureFlag.VAD,
    }
    description: str | None = "Standard ASR wrapper for faster-whisper."


class FasterWhisperASR(StandardASR):
    """Standard ASR adapter for faster-whisper.

    Args:
        config: Engine configuration instance.

    Returns:
        None.

    Raises:
        ValueError: If configuration validation fails.
    """

    config: FasterWhisperConfig
    properties: ClassVar[FasterWhisperProperties] = FasterWhisperProperties()

    def __init__(self, **kwargs: Any) -> None:
        self.config = FasterWhisperConfig(engine="faster-whisper", **kwargs)
        self._model: WhisperModel | None = None

    def _ensure_model_loaded(self) -> None:
        """Load the faster-whisper model lazily.

        Args:
            None.

        Returns:
            None.

        Raises:
            DiscoveryError: If model weights cannot be loaded.
        """
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            raise DiscoveryError(
                "faster-whisper is not installed. Install with 'pip install faster-whisper'."
            ) from exc

        local_only = self.config.local_files_only or not allow_downloads()
        try:
            self._model = WhisperModel(
                model_size_or_path=self.config.model_path,
                device=self.config.device,
                device_index=self.config.device_index,
                compute_type=self.config.compute_type,
                cpu_threads=self.config.cpu_threads,
                num_workers=self.config.num_workers,
                download_root=self.config.download_root,
                local_files_only=local_only,
                revision=self.config.revision,
                use_auth_token=self.config.use_auth_token,
            )
        except Exception as exc:  # noqa: BLE001
            raise DiscoveryError(
                "Failed to load faster-whisper model. If downloads are disabled, "
                "set STANDARD_ASR_ALLOW_DOWNLOAD=1 or pre-download the model."
            ) from exc

    def prepare(self) -> None:
        """Preload model weights without running transcription.

        Args:
            None.

        Returns:
            None.

        Raises:
            DiscoveryError: If model weights cannot be loaded.
        """
        self._ensure_model_loaded()

    def transcribe(
        self,
        audio: NDArray[np.float32],
        options: BaseTranscribeOptions | dict[str, object] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio using faster-whisper.

        Args:
            audio: Audio waveform array.
            options: Optional transcription options (model or dict).

        Returns:
            Standard ASR transcription result.

        Raises:
            TranscriptionError: If transcription fails or unsupported options are used.
        """
        validate_audio_input(audio, self.properties)
        opts = coerce_options(options, FasterWhisperOptions)

        if opts.speaker_diarization:
            raise TranscriptionError("Speaker diarization is not supported by faster-whisper.")

        self._ensure_model_loaded()
        if self._model is None:
            raise TranscriptionError("Model failed to load.")

        kwargs = _options_to_kwargs(opts)

        language = None
        if opts.language:
            normalized = normalize_bcp47(opts.language)
            language = normalized.split("-", maxsplit=1)[0]

        segments, info = self._model.transcribe(
            audio,
            language=language,
            task=opts.task,
            **kwargs,
        )

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
                    extra={"tokens": segment.tokens, "seek": segment.seek},
                )
            )

        text = "".join(seg.text for seg in segment_list)
        language = normalize_bcp47(info.language) if info.language else None

        metadata = {
            "language_probability": info.language_probability,
            "duration": info.duration,
            "duration_after_vad": info.duration_after_vad,
            "all_language_probs": info.all_language_probs,
            "transcription_options": asdict(info.transcription_options),
        }

        return TranscriptionResult(
            text=text,
            language=language,
            duration=info.duration,
            segments=segment_list,
            words=word_list if opts.word_timestamps else None,
            metadata=metadata,
        )


def _options_to_kwargs(options: FasterWhisperOptions) -> dict[str, Any]:
    """Convert options into faster-whisper keyword arguments.

    Args:
        options: Parsed faster-whisper options.

    Returns:
        Keyword arguments for ``WhisperModel.transcribe``.

    Raises:
        None.
    """
    kwargs: dict[str, Any] = {
        "beam_size": options.beam_size,
        "best_of": options.best_of,
        "patience": options.patience,
        "length_penalty": options.length_penalty,
        "repetition_penalty": options.repetition_penalty,
        "no_repeat_ngram_size": options.no_repeat_ngram_size,
        "compression_ratio_threshold": options.compression_ratio_threshold,
        "log_prob_threshold": options.log_prob_threshold,
        "no_speech_threshold": options.no_speech_threshold,
        "condition_on_previous_text": options.condition_on_previous_text,
        "prompt_reset_on_temperature": options.prompt_reset_on_temperature,
        "initial_prompt": options.initial_prompt,
        "prefix": options.prefix,
        "suppress_blank": options.suppress_blank,
        "suppress_tokens": options.suppress_tokens,
        "without_timestamps": options.without_timestamps,
        "max_initial_timestamp": options.max_initial_timestamp,
        "word_timestamps": options.word_timestamps,
        "prepend_punctuations": options.prepend_punctuations,
        "append_punctuations": options.append_punctuations,
        "multilingual": options.multilingual,
        "vad_filter": options.vad_filter,
        "vad_parameters": options.vad_parameters,
        "max_new_tokens": options.max_new_tokens,
        "chunk_length": options.chunk_length,
        "clip_timestamps": options.clip_timestamps,
        "hallucination_silence_threshold": options.hallucination_silence_threshold,
        "hotwords": options.hotwords,
        "language_detection_threshold": options.language_detection_threshold,
        "language_detection_segments": options.language_detection_segments,
    }

    if options.temperature is not None:
        kwargs["temperature"] = options.temperature

    return kwargs
