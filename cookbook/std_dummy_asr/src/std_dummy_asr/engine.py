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

from standard_asr.engine import (
    AUTO,
    BaseConfig,
    BaseProperties,
    BatchCapabilities,
    DeclaredCapabilities,
    EngineBase,
    FlagCap,
    InputKind,
    LanguageCaps,
    LanguageConfigMixin,
    PreparedAudio,
    RuntimeParams,
    SampleRateRange,
    TranscriptionResult,
    effective_language,
)


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
        default="echo",
        description="Text prefix included in the emitted transcript for demos.",
    )


class DummyASRProperties(BaseProperties):
    """Static metadata describing the dummy ASR engine."""

    engine_id: str = "dummy"
    model_name: str = "echo"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
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
    config_type: ClassVar[type[BaseConfig[str]] | None] = DummyASRConfig

    def __init__(self, **kwargs: Any) -> None:
        """Capture configuration (pure; nothing is loaded).

        Config is built via ``from_env``, so **every field the published
        ``config_type`` (``DummyASRConfig``) advertises** -- ``message``,
        ``default_language``, ``default_candidate_languages`` -- can be supplied
        as a keyword argument. A caller that follows the config contract (e.g.
        ``registry.create("dummy/echo", default_language="fr")``) is therefore
        honoured rather than rejected with a ``TypeError``. Explicit ``kwargs``
        win over the ``STANDARD_ASR_DUMMY__*`` environment (spec IC.4; the engine
        and field segments are joined by a double underscore), and an unknown
        field is rejected by the closed config model rather than silently
        ignored. The ``engine`` discriminator is fixed, not a caller argument.

        Args:
            **kwargs: Configuration overrides (any ``DummyASRConfig`` field).
        """
        self.config = DummyASRConfig.from_env("dummy", **kwargs)

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
        # ``effective_language`` returns the reserved ``"auto"`` directive verbatim
        # when the request (or the default) asks for detection. ``detected_language``
        # is the language the engine RESOLVED to, so it MUST be a concrete BCP-47
        # tag, never ``"auto"`` -- ``validate_detected_language`` rejects the reserved
        # word (spec TR.1). Every adapter must therefore translate ``"auto"`` into a
        # real detection result. This dummy only "detects" English (its sole
        # ``detectable_languages`` entry), so ``"auto"`` resolves to ``"en"``.
        detected = "en" if lang == AUTO else lang
        return TranscriptionResult(
            text=f"{config.message}: {samples} samples",
            detected_language=detected,
            duration=samples / self.properties.native_sample_rate,
            # ``samples`` is an engine-specific diagnostic, not standardized
            # cross-engine metadata, so it belongs in ``extra`` -- ``metadata`` is
            # reserved for engine-agnostic standardized keys (spec TR.1). Putting
            # engine-private data here would let it collide with future standard
            # metadata keys and re-teach the slot's meaning to the ecosystem.
            extra={"samples": samples},
        )


class DummyDefaultASR(DummyASR):
    """Default dummy preset whose model_id matches ``dummy/``."""

    properties: ClassVar[BaseProperties] = DummyDefaultProperties()
