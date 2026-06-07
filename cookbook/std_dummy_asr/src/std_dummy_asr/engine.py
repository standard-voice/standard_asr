# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Lightweight Standard ASR implementation for demos and tests.

This is the canonical minimal adapter: it subclasses :class:`EngineBase`,
declares its :class:`BaseProperties` and :class:`DeclaredCapabilities`, and
implements only :meth:`_transcribe`. The standard layer handles audio
negotiation, conversion, and parameter gating.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, cast

from pydantic import Field

from standard_asr import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    InputKind,
    LanguageConfigMixin,
    PreparedAudio,
    RuntimeParams,
    TranscriptionResult,
)
from standard_asr.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    FlagCap,
    LanguageCaps,
)
from standard_asr.language import effective_language


class DummyASRConfig(LanguageConfigMixin, BaseConfig[Literal["dummy"]]):
    """Configuration model for the dummy ASR engine.

    Args:
        engine: Discriminator identifying this engine (always ``"dummy"``).
        message: Text prefix inserted into the transcript.
        default_language: Default language for the engine.
    """

    engine: Literal["dummy"] = "dummy"
    default_language: str | None = "en"
    message: str = Field(
        "echo",
        description="Text prefix included in the emitted transcript for demos.",
    )


class DummyASRProperties(BaseProperties):
    """Static metadata describing the dummy ASR engine."""

    engine_id: str = "dummy"
    model_name: str = "echo"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | Literal["any"] = [16000]
    selectable_languages: list[str] = ["en", "auto"]
    detectable_languages: list[str] = ["en"]
    description: str | None = "Dummy echo engine for testing and demos."


class DummyDefaultProperties(DummyASRProperties):
    """Static metadata describing the default dummy ASR preset."""

    model_name: str = ""
    description: str | None = "Default dummy engine preset."


_CAPABILITIES = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
    )
)


class DummyASR(EngineBase):
    """Trivial ASR implementation that reports the input shape.

    Args:
        message: Text prefix for the transcript.
    """

    properties: ClassVar[BaseProperties] = DummyASRProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _CAPABILITIES

    def __init__(self, message: str | None = None) -> None:
        """Initialize the dummy engine.

        Construct config via ``from_env`` so unset fields fall back to
        ``STANDARD_ASR_DUMMY_*`` environment variables (spec IC.4); an explicit
        ``message`` wins over the environment. ``message`` defaults to ``None``
        (not ``"echo"``) so that omitting it lets the env var take effect rather
        than passing a default that would always override it.

        Args:
            message: Text prefix for the transcript, or ``None`` to use the
                environment / config default.
        """
        explicit: dict[str, Any] = {} if message is None else {"message": message}
        self.config = DummyASRConfig.from_env("dummy", **explicit)

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        """Return a short description of the provided audio buffer.

        Args:
            prepared: Engine-ready audio (an array, per ``accepted_input``).
            params: Gated runtime parameters.

        Returns:
            A Standard ASR transcription result.
        """
        config = cast(DummyASRConfig, self.config)
        samples = int(prepared.array.size) if prepared.array is not None else 0
        lang = effective_language(
            params.language,
            config.default_language,
            has_language_axis=self.properties.has_language_axis,
            runtime_override_supported=self.supports("batch.language.runtime_override"),
        )
        return TranscriptionResult(
            text=f"{config.message}: {samples} samples",
            detected_language=lang,
            duration=samples / self.properties.native_sample_rate,
            metadata={"samples": samples},
        )


class DummyDefaultASR(DummyASR):
    """Default dummy preset whose model_id matches ``dummy/``."""

    properties: ClassVar[BaseProperties] = DummyDefaultProperties()
