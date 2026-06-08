# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The Standard ASR engine interface (Protocol + ABC).

This module assembles the scattered pieces (config, properties, capabilities,
runtime params, result model, audio negotiation) into the single authoritative
engine contract (closing NEW-GAP-01). Two complementary forms are provided:

* :class:`StandardASR` -- a structural :class:`typing.Protocol` describing the
  public surface every engine exposes. Use it for typing and ``isinstance``
  checks against any compliant engine, however implemented.
* :class:`EngineBase` -- an abstract base class that implements the public
  ``transcribe`` as a *template method*: it coerces the input, negotiates and
  executes the audio conversion, gates parameters against capabilities, calls
  the engine's :meth:`EngineBase._transcribe`, and attaches diagnostics. ASR
  authors subclass it and implement only the model-specific bits.

The negotiation / conversion / gating pipeline runs in the standard layer
(:class:`EngineBase`), so authors get consistent, correct behaviour for free.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from .asr_config import BaseConfig
from .asr_properties import BaseProperties
from .audio_conversion import PreparedAudio, execute_plan
from .audio_input import AudioInput, AudioInputLike, coerce_audio_input
from .audio_negotiation import negotiate_or_raise
from .capabilities import DeclaredCapabilities
from .exceptions import ConfigError, UnsupportedFeatureError
from .language import effective_candidate_languages, effective_language
from .param_gating import Mode, gate_params
from .results import Diagnostic, TranscriptionResult
from .runtime_params import ProviderParams, RuntimeParams

if TYPE_CHECKING:
    from .audio_format import AudioFormat
    from .streaming import TranscriptionSession


@runtime_checkable
class StandardASR(Protocol):
    """Structural protocol for a Standard ASR engine.

    Any object exposing these members is a compliant engine, regardless of how
    it is implemented.
    """

    config: BaseConfig[str]
    properties: ClassVar[BaseProperties]
    declared_capabilities: ClassVar[DeclaredCapabilities]

    def transcribe(
        self, audio: AudioInputLike, params: RuntimeParams | None = None
    ) -> TranscriptionResult:
        """Transcribe a complete audio input.

        Args:
            audio: The audio to transcribe (any :data:`AudioInput` variant or a
                coercible bare value).
            params: Per-request runtime parameters.

        Returns:
            The transcription result.
        """
        ...

    def supports(self, dot_path: str) -> bool:
        """Return whether the capability at ``dot_path`` is supported.

        Args:
            dot_path: A capability dot-path.

        Returns:
            ``True`` if supported.
        """
        ...


class EngineBase(ABC):
    """Abstract base implementing the standard transcribe pipeline.

    Subclasses MUST set :attr:`properties` and :attr:`declared_capabilities` as
    class attributes, assign :attr:`config` in ``__init__`` (which MUST stay
    pure -- no filesystem, GPU, or network access; spec IC.9), and implement
    :meth:`_transcribe`. Streaming engines additionally override
    :meth:`start_transcription`.
    """

    properties: ClassVar[BaseProperties]
    declared_capabilities: ClassVar[DeclaredCapabilities]
    #: The engine's expected ``provider_params`` type, or ``None``.
    provider_params_type: ClassVar[type[ProviderParams] | None] = None

    config: BaseConfig[str]

    @property
    def effective_capabilities(self) -> DeclaredCapabilities:
        """Runtime-effective capabilities (default: the declared set).

        Engines that narrow capabilities based on configuration override this;
        the result MUST satisfy ``effective ⊆ declared``.

        Returns:
            The effective capability tree.
        """
        return self.declared_capabilities

    def supports(self, dot_path: str) -> bool:
        """Return whether a capability is supported at runtime (fail-closed).

        Args:
            dot_path: A capability dot-path.

        Returns:
            ``True`` if supported by the effective capabilities.
        """
        return self.effective_capabilities.supports(dot_path)

    @property
    def _strict(self) -> bool:
        """Whether the unsupported-parameter policy is strict.

        Returns:
            ``True`` for strict, ``False`` for best_effort.
        """
        return bool(getattr(self.config, "strict", True))

    def transcribe(
        self, audio: AudioInputLike, params: RuntimeParams | None = None
    ) -> TranscriptionResult:
        """Transcribe a complete audio input (template method).

        Runs the standard pipeline, *fail-fast first*: validate the language
        config -> gate parameters (provider_params + capability gating, which
        needs no audio) -> resolve & validate the effective language axis ->
        coerce -> negotiate -> convert/resample -> call the engine -> attach
        diagnostics.

        Parameter validation runs *before* the (potentially expensive) audio
        decode/resample so a swapped-engine ``provider_params`` bug or an
        unsupported parameter is rejected before any audio is touched (spec
        Runtime R3: "先 provider_params 快失败").

        Args:
            audio: The audio to transcribe.
            params: Per-request runtime parameters.

        Returns:
            The transcription result with gating / language / conversion
            diagnostics attached.

        Raises:
            ConfigError: If the engine exposes a language axis but its
                ``default_language`` is unset or not in ``selectable_languages``.
            IncompatibleAudioInputError: If no conversion path exists.
            UnsupportedFeatureError: In strict mode, on an unsupported parameter.
            InvalidProviderParamError: On wrong provider params.
            ValueError: On an invalid candidate-language list in strict mode.
        """
        request = params or RuntimeParams()
        # Fail fast: validate config + params (no audio needed) before decode.
        self._validate_language_config()
        gated, gate_diags = gate_params(
            request,
            self.effective_capabilities,
            "batch",
            strict=self._strict,
            expected_provider_type=self.provider_params_type,
        )
        lang_diags = self._resolve_language_axis(gated, "batch")
        # Audio decode/resample only after parameters are known-good.
        provided: AudioInput = coerce_audio_input(audio)
        plan = negotiate_or_raise(provided, set(self.properties.accepted_input))
        prepared = execute_plan(
            provided,
            plan,
            accepted_sample_rates=self.properties.accepted_sample_rates,
            native_sample_rate=self.properties.native_sample_rate,
            required_input_sample_rate=self.properties.required_input_sample_rate,
            max_file_size=self.properties.max_file_size,
            max_audio_duration=self.properties.max_audio_duration,
            strict=self._strict,
        )
        result = self._transcribe(prepared, gated)
        merged = [
            *gate_diags,
            *lang_diags,
            *prepared.diagnostics,
            *result.diagnostics,
        ]
        return result.model_copy(update={"diagnostics": merged})

    def _validate_language_config(self) -> None:
        """Enforce the ``default_language`` totality invariant (IC.6 / LANG R1).

        When the engine exposes a language axis, ``default_language`` MUST be set
        and MUST be a member of ``selectable_languages``; otherwise R2 step 2
        (fall back to ``default_language``) would yield an undefined result. This
        runs in the standard layer so a forgetful adapter fails loudly instead of
        silently transcribing in the wrong language.

        Raises:
            ConfigError: If the language axis is exposed but ``default_language``
                is unset or not in ``selectable_languages``.
        """
        if not self.properties.has_language_axis:
            return
        default = getattr(self.config, "default_language", None)
        if default is None:
            raise ConfigError(
                f"Engine {self.properties.engine_id!r} exposes a language axis "
                "(selectable_languages is non-empty) so its config MUST set "
                "default_language (spec IC.6 / LANG R1)."
            )
        if default not in self.properties.selectable_languages:
            raise ConfigError(
                f"default_language {default!r} is not in selectable_languages "
                f"{self.properties.selectable_languages!r} "
                f"(engine {self.properties.engine_id!r}, spec LANG R1)."
            )

    def _resolve_language_axis(self, params: RuntimeParams, mode: Mode) -> list[Diagnostic]:
        """Validate the effective language / candidate-language axis (LANG R2/R3).

        Runs the standard resolution so candidate-language violations are caught
        once, in the standard layer, regardless of whether the engine remembers
        to call :func:`~standard_asr.language.effective_candidate_languages`.
        The engine still reads ``params`` (and MAY call
        :func:`~standard_asr.language.effective_language` for the final value);
        this method only validates and emits diagnostics, so :meth:`_transcribe`
        keeps its existing signature.

        Args:
            params: The gated runtime parameters.
            mode: ``"batch"`` or ``"streaming"``.

        Returns:
            Diagnostics produced during candidate-language resolution.

        Raises:
            ValueError: In strict mode, on an invalid candidate-language list.
        """
        if not self.properties.has_language_axis:
            return []
        caps = self.effective_capabilities
        eff_lang = effective_language(
            params.language,
            getattr(self.config, "default_language", None),
            has_language_axis=True,
            runtime_override_supported=caps.supports(f"{mode}.language.runtime_override"),
        )
        constraints = self._candidate_max(mode)
        _, diagnostics = effective_candidate_languages(
            eff_lang,
            params.candidate_languages,
            getattr(self.config, "default_candidate_languages", None),
            candidate_supported=caps.supports(f"{mode}.language.candidate_languages"),
            detectable_languages=self.properties.detectable_languages,
            max_count=constraints,
            strict=self._strict,
        )
        return diagnostics

    def _candidate_max(self, mode: Mode) -> int | None:
        """Return the candidate-languages ``max`` constraint for ``mode``.

        Args:
            mode: ``"batch"`` or ``"streaming"``.

        Returns:
            The declared maximum candidate count, or ``None`` if unconstrained
            or the mode is unsupported.
        """
        domain = getattr(self.effective_capabilities, mode, None)
        if domain is None:
            return None
        cap = domain.language.candidate_languages
        return cap.constraints.max if cap.constraints is not None else None

    async def transcribe_async(
        self, audio: AudioInputLike, params: RuntimeParams | None = None
    ) -> TranscriptionResult:
        """Asynchronously transcribe (default: run :meth:`transcribe` in a thread).

        Args:
            audio: The audio to transcribe.
            params: Per-request runtime parameters.

        Returns:
            The transcription result.
        """
        return await asyncio.to_thread(self.transcribe, audio, params)

    @abstractmethod
    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        """Run the engine on already-negotiated audio.

        The audio in ``prepared`` is guaranteed to be in one of the engine's
        accepted shapes. Implementations should dispatch on ``prepared.kind``.

        Args:
            prepared: Engine-ready audio (in an accepted shape).
            params: Gated runtime parameters.

        Returns:
            The transcription result (diagnostics added here are merged with the
            standard layer's).
        """
        raise NotImplementedError  # pragma: no cover

    @staticmethod
    def ensure_stream_inputs_exclusive(
        audio_format: AudioFormat | None, audio: AudioInputLike | None
    ) -> None:
        """Enforce the ``audio_format`` / ``audio`` mutual-exclusion (ST §3.1).

        ``audio_format`` (incremental PCM feeding) and ``audio`` (whole-input
        streaming output) are mutually exclusive; passing both MUST raise. This
        shared guard lets every streaming engine enforce the rule with one call
        instead of reimplementing it; the base :meth:`start_transcription`
        invokes it before raising the unsupported-streaming error.

        Args:
            audio_format: The wire format for incremental frames, if any.
            audio: A complete audio input for whole-input streaming, if any.

        Raises:
            ValueError: If both ``audio_format`` and ``audio`` are provided.
        """
        if audio_format is not None and audio is not None:
            raise ValueError(
                "start_transcription: 'audio_format' (incremental feeding) and "
                "'audio' (whole-input streaming) are mutually exclusive; pass "
                "exactly one (spec Streaming §3.1)."
            )

    def ensure_stream_format_supported(self, audio_format: AudioFormat) -> None:
        """Validate a declared streaming wire format at session establishment.

        Shared session-establishment guard for streaming engines: call it first
        (like :meth:`ensure_stream_inputs_exclusive`) when opening a
        ``audio_format=...`` session. It is **fail-closed** on both the wire
        encoding and the wire sample rate.

        Wire **encoding**: an encoding the engine never declared in
        ``wire_encodings`` is rejected up front rather than misframed as PCM and
        silently mistranscribed.

        Wire **sample rate**: spec R7's v1 implementation note is explicit that
        v1 does **NOT** resample streaming bare frames in the standard layer
        (unlike the batch ``transcribe`` path, which resamples). Therefore, until
        standard-layer streaming resampling lands, a wire ``sample_rate`` that the
        engine does not accept MUST be rejected here rather than forwarded as
        frames the engine never declared -- a loud error beats a silent
        mistranscription. The rate is accepted when ``accepted_sample_rates`` is
        ``"any"``, when it is in that concrete list, or when it equals the
        engine's ``required_input_sample_rate``.

        Args:
            audio_format: The wire format the session declared.

        Raises:
            UnsupportedFeatureError: If ``wire_encodings`` is declared and the
                requested encoding is not among them, or if the wire sample rate
                is not reachable for the engine (fail-closed; v1 does not resample
                streaming wire frames).
        """
        props = self.properties
        wire = props.wire_encodings
        if wire is not None and audio_format.encoding not in wire:
            raise UnsupportedFeatureError(
                f"Streaming wire encoding {audio_format.encoding!r} is not supported; "
                f"engine {props.engine_id!r} declares wire_encodings={wire}."
            )

        accepted = props.accepted_sample_rates
        if isinstance(accepted, list):
            rate = audio_format.sample_rate
            if rate not in accepted and rate != props.required_input_sample_rate:
                raise UnsupportedFeatureError(
                    f"Streaming wire sample_rate {rate} Hz is not accepted by engine "
                    f"{props.engine_id!r} (accepted_sample_rates={accepted}). v1 does "
                    "not resample streaming wire frames, so an unreachable rate is "
                    "rejected at session establishment rather than silently "
                    "mistranscribed. Open the session at an accepted rate."
                )

    def start_transcription(
        self,
        *,
        audio_format: AudioFormat | None = None,
        params: RuntimeParams | None = None,
        audio: AudioInputLike | None = None,
    ) -> TranscriptionSession:
        """Open a streaming transcription session.

        The default enforces the input mutual-exclusion guard and then raises;
        streaming engines override this (and SHOULD call
        :meth:`ensure_stream_inputs_exclusive` first).

        Args:
            audio_format: Wire format for incremental PCM frames.
            params: Per-request runtime parameters.
            audio: A complete audio input for whole-input streaming output.

        Returns:
            A streaming session.

        Raises:
            ValueError: If both ``audio_format`` and ``audio`` are provided.
            UnsupportedFeatureError: When streaming is unsupported (the default).
        """
        self.ensure_stream_inputs_exclusive(audio_format, audio)
        raise UnsupportedFeatureError("This engine does not support streaming.")


__all__ = ["EngineBase", "StandardASR"]
