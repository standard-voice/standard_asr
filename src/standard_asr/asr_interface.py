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
from .exceptions import UnsupportedFeatureError
from .param_gating import gate_params
from .results import TranscriptionResult
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

        Runs the standard pipeline: coerce -> negotiate -> convert/resample ->
        gate parameters -> call the engine -> attach diagnostics.

        Args:
            audio: The audio to transcribe.
            params: Per-request runtime parameters.

        Returns:
            The transcription result with conversion / gating diagnostics
            attached.

        Raises:
            IncompatibleAudioInputError: If no conversion path exists.
            UnsupportedFeatureError: In strict mode, on an unsupported parameter.
            InvalidProviderParamError: On wrong provider params.
        """
        request = params or RuntimeParams()
        provided: AudioInput = coerce_audio_input(audio)
        plan = negotiate_or_raise(provided, set(self.properties.accepted_input))
        prepared = execute_plan(
            provided,
            plan,
            accepted_sample_rates=self.properties.accepted_sample_rates,
            native_sample_rate=self.properties.native_sample_rate,
            required_input_sample_rate=self.properties.required_input_sample_rate,
            max_file_size=self.properties.max_file_size,
            strict=self._strict,
        )
        gated, gate_diags = gate_params(
            request,
            self.effective_capabilities,
            "batch",
            strict=self._strict,
            expected_provider_type=self.provider_params_type,
        )
        result = self._transcribe(prepared, gated)
        merged = [*prepared.diagnostics, *gate_diags, *result.diagnostics]
        return result.model_copy(update={"diagnostics": merged})

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
    def _transcribe(
        self, prepared: PreparedAudio, params: RuntimeParams
    ) -> TranscriptionResult:
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

    def start_transcription(
        self,
        *,
        audio_format: AudioFormat | None = None,
        params: RuntimeParams | None = None,
        audio: AudioInputLike | None = None,
    ) -> TranscriptionSession:
        """Open a streaming transcription session.

        The default raises; streaming engines override this.

        Args:
            audio_format: Wire format for incremental PCM frames.
            params: Per-request runtime parameters.
            audio: A complete audio input for whole-input streaming output.

        Returns:
            A streaming session.

        Raises:
            UnsupportedFeatureError: Always, unless overridden.
        """
        raise UnsupportedFeatureError("This engine does not support streaming.")


__all__ = ["EngineBase", "StandardASR"]
