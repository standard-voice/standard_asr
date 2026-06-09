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
from typing import TYPE_CHECKING, ClassVar, Protocol, cast, runtime_checkable

from .asr_config import BaseConfig
from .asr_properties import BaseProperties
from .audio_conversion import PreparedAudio, execute_plan
from .audio_input import AudioInput, AudioInputLike, coerce_audio_input
from .audio_negotiation import negotiate_or_raise
from .capabilities import DeclaredCapabilities
from .exceptions import ConfigError, UnsupportedFeatureError
from .language import AUTO, effective_candidate_languages, effective_language, normalize_bcp47
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
    it is implemented. The protocol describes the *full* public surface every
    engine exposes -- batch (:meth:`transcribe` / :meth:`transcribe_async`) and
    the streaming entry point (:meth:`start_transcription`). ``start_transcription``
    is always present; streaming support itself is optional, so a batch-only
    engine raises :class:`~standard_asr.exceptions.UnsupportedFeatureError` from
    it. Because the surface is complete, callers (e.g. the server) can type an
    engine as ``StandardASR`` and call the streaming entry point without a cast.
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

    async def transcribe_async(
        self, audio: AudioInputLike, params: RuntimeParams | None = None
    ) -> TranscriptionResult:
        """Asynchronously transcribe a complete audio input.

        Args:
            audio: The audio to transcribe (any :data:`AudioInput` variant or a
                coercible bare value).
            params: Per-request runtime parameters.

        Returns:
            The transcription result.
        """
        ...

    def start_transcription(
        self,
        *,
        audio_format: AudioFormat | None = None,
        params: RuntimeParams | None = None,
        audio: AudioInputLike | None = None,
    ) -> TranscriptionSession:
        """Open a streaming transcription session.

        Always present on a compliant engine, but streaming itself is optional:
        a batch-only engine raises
        :class:`~standard_asr.exceptions.UnsupportedFeatureError` here. Callers
        that need streaming should gate on
        ``supports("streaming_input")`` / ``supports("streaming_output")`` (or be
        ready to handle the unsupported-streaming error).

        Args:
            audio_format: Wire format for incremental PCM frames.
            params: Per-request runtime parameters.
            audio: A complete audio input for whole-input streaming output.

        Returns:
            A streaming session.
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


def _canonical_language(tag: str) -> str:
    """Canonicalize a BCP-47 tag for case-insensitive matching, preserving AUTO.

    ``selectable_languages`` and ``default_language`` may be declared as
    non-canonical class-level defaults -- pydantic does not run the field
    validators on defaults -- so language membership tests must canonicalize BOTH
    sides here rather than trusting either to be pre-normalized. The reserved
    ``auto`` directive is not a BCP-47 tag, so it is matched verbatim.

    Args:
        tag: A BCP-47 tag or the reserved ``auto`` token.

    Returns:
        The canonical form (``auto`` returned unchanged).

    Raises:
        ValueError: If ``tag`` is empty/whitespace (a malformed declaration).
    """
    return tag if tag == AUTO else normalize_bcp47(tag)


def _language_is_selectable(tag: str, selectable: set[str]) -> bool:
    """Whether a canonical BCP-47 tag is selectable, via RFC 4647 lookup matching.

    A request is selectable if its canonical form -- or any prefix obtained by
    progressively dropping trailing subtags -- is in the (canonical) selectable
    set. This lets an engine declare a primary language subtag (``en``) and still
    accept a region/script refinement of it (``en-US``, ``zh-Hant``), which the
    engine reduces internally, without enumerating every variant. Genuinely
    unrelated languages (``fr`` against an ``en`` engine) still do not match, and
    the reserved ``auto`` token has no subtags so it only ever matches verbatim.

    Args:
        tag: The canonical requested tag (or ``auto``).
        selectable: The canonical selectable set.

    Returns:
        ``True`` if ``tag`` or one of its prefixes is selectable.
    """
    parts = tag.split("-")
    return any("-".join(parts[:i]) in selectable for i in range(len(parts), 0, -1))


class EngineBase(ABC):
    """Abstract base implementing the standard transcribe pipeline.

    Subclasses MUST set :attr:`properties` and :attr:`declared_capabilities` as
    class attributes, assign :attr:`config` in ``__init__`` (which MUST stay
    pure -- no filesystem, GPU, or network access; spec IC.9), and implement
    :meth:`_transcribe`. Streaming engines additionally override
    :meth:`_start_transcription` (the streaming template hook); the public
    :meth:`start_transcription` runs the standard gating pipeline for them.
    """

    properties: ClassVar[BaseProperties]
    declared_capabilities: ClassVar[DeclaredCapabilities]
    #: The engine's expected ``provider_params`` type, or ``None``.
    provider_params_type: ClassVar[type[ProviderParams] | None] = None
    #: The engine's init-config model type, or ``None`` when not declared.
    #: Declaring it makes the config JSON Schema (including the secret-field
    #: markers from :func:`~standard_asr.asr_config.secret_field`) readable
    #: *without instantiation* (spec §3.1 / G.3.1) -- the discovery path for
    #: settings UIs. Without it, an app cannot render a config form for a
    #: credentialed engine, because constructing the engine requires the very
    #: credentials the form is meant to collect. Engines SHOULD declare it;
    #: the compliance suite reports a warning when it is missing.
    config_type: ClassVar[type[BaseConfig[str]] | None] = None

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
        gated, lang_diags = self._resolve_language_axis(gated, "batch")
        # Audio decode/resample only after parameters are known-good.
        prepared = self._prepare_audio(audio)
        result = self._transcribe(prepared, gated)
        merged = [
            *gate_diags,
            *lang_diags,
            *prepared.diagnostics,
            *result.diagnostics,
        ]
        return result.model_copy(update={"diagnostics": merged})

    def _prepare_audio(self, audio: AudioInputLike) -> PreparedAudio:
        """Decode, negotiate, and resample an audio input (shared pipeline).

        The single owner of the audio-conversion arguments threaded into
        :func:`~standard_asr.audio_conversion.execute_plan`, shared by
        :meth:`transcribe` and the whole-input :meth:`start_transcription` path so
        both honor identical negotiation against the engine's declared audio
        properties. A new conversion parameter is then wired in exactly one place
        and can never silently diverge between the batch and streaming paths.

        Args:
            audio: The caller's audio input (path, bytes, URL, array, ...).

        Returns:
            The prepared audio (decoded / resampled per the engine's properties),
            carrying any conversion diagnostics.

        Raises:
            IncompatibleAudioInputError: If no conversion path exists.
        """
        provided: AudioInput = coerce_audio_input(audio)
        plan = negotiate_or_raise(provided, set(self.properties.accepted_input))
        return execute_plan(
            provided,
            plan,
            accepted_sample_rates=self.properties.accepted_sample_rates,
            native_sample_rate=self.properties.native_sample_rate,
            required_input_sample_rate=self.properties.required_input_sample_rate,
            max_file_size=self.properties.max_file_size,
            max_audio_duration=self.properties.max_audio_duration,
            strict=self._strict,
        )

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
        # Canonicalize BOTH sides: BCP-47 membership is case-insensitive, and
        # either default_language or selectable_languages may be a non-canonical
        # class-level default, so a raw "en-us" must still match a canonical
        # "en-US" instead of spuriously failing LANG R1 and blocking the engine.
        if _canonical_language(default) not in {
            _canonical_language(tag) for tag in self.properties.selectable_languages
        }:
            raise ConfigError(
                f"default_language {default!r} is not in selectable_languages "
                f"{self.properties.selectable_languages!r} "
                f"(engine {self.properties.engine_id!r}, spec LANG R1)."
            )

    def _resolve_language_axis(
        self, params: RuntimeParams, mode: Mode
    ) -> tuple[RuntimeParams, list[Diagnostic]]:
        """Resolve and validate the effective language axis (LANG R2/R3).

        Runs standard resolution so the engine receives the same effective
        ``language`` and ``candidate_languages`` values that the standard layer
        validated and diagnosed.

        Args:
            params: The gated runtime parameters.
            mode: ``"batch"`` or ``"streaming"``.

        Returns:
            A ``(params, diagnostics)`` pair containing the effective runtime
            parameters plus diagnostics produced during language resolution.

        Raises:
            UnsupportedFeatureError: In strict mode, if the resolved language is
                not selectable by this engine.
            ValueError: In strict mode, on an invalid candidate-language list.
        """
        if not self.properties.has_language_axis:
            return params, []
        caps = self.effective_capabilities
        # default_language is non-None here: _validate_language_config (always run
        # before this) enforces it whenever has_language_axis is True. Canonicalize
        # it up front so the best-effort fallback below (and the diagnostic it
        # emits) carry a canonical tag, never a raw class-level default ("en-us").
        default_language = _canonical_language(
            cast("str", getattr(self.config, "default_language", None))
        )
        default_candidates = cast(
            "list[str] | None", getattr(self.config, "default_candidate_languages", None)
        )
        eff_lang = effective_language(
            params.language,
            default_language,
            has_language_axis=True,
            runtime_override_supported=caps.supports(f"{mode}.language.runtime_override"),
        )
        if eff_lang is not None and eff_lang != AUTO:
            eff_lang = normalize_bcp47(eff_lang)

        diagnostics: list[Diagnostic] = []
        # Canonicalize the selectable set too (it may be a non-canonical
        # class-level default), so a canonical eff_lang matches case-insensitively.
        # Membership uses RFC 4647 lookup so a region/script refinement of a
        # selectable primary subtag (e.g. "en-US" against "en") is accepted and
        # handed to the engine to reduce -- engines need not enumerate variants.
        selectable = {_canonical_language(tag) for tag in self.properties.selectable_languages}
        if eff_lang is not None and not _language_is_selectable(eff_lang, selectable):
            if self._strict:
                raise UnsupportedFeatureError(
                    f"language {eff_lang!r} is not selectable in {mode} mode "
                    f"for engine {self.properties.engine_id!r} "
                    f"(selectable_languages={self.properties.selectable_languages!r}).",
                    param="language",
                    mode=mode,
                    hint=(
                        "Request one of the engine's selectable_languages, or use "
                        "best_effort to fall back to default_language."
                    ),
                )
            diagnostics.append(
                Diagnostic(
                    level="warning",
                    code="language_not_selectable",
                    message=(
                        f"Fell back from non-selectable language {eff_lang!r} to "
                        f"default_language {default_language!r} in {mode} mode."
                    ),
                    param="language",
                    provided=eff_lang,
                    effective=default_language,
                )
            )
            eff_lang = default_language

        constraints = self._candidate_max(mode)
        eff_candidates, candidate_diags = effective_candidate_languages(
            eff_lang,
            params.candidate_languages,
            default_candidates,
            candidate_supported=caps.supports(f"{mode}.language.candidate_languages"),
            detectable_languages=self.properties.detectable_languages,
            max_count=constraints,
            strict=self._strict,
        )
        diagnostics.extend(candidate_diags)
        effective_params = params.model_copy(
            update={"language": eff_lang, "candidate_languages": eff_candidates}
        )
        return effective_params, diagnostics

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
                requested encoding is not among them, if the wire ``channels`` is
                not ``1`` (v1 streaming wire is mono-only), or if the wire sample
                rate is not reachable for the engine (fail-closed; v1 does not
                resample streaming wire frames).
        """
        props = self.properties
        wire = props.wire_encodings
        if wire is not None and audio_format.encoding not in wire:
            raise UnsupportedFeatureError(
                f"Streaming wire encoding {audio_format.encoding!r} is not supported; "
                f"engine {props.engine_id!r} declares wire_encodings={wire}.",
                param="audio_format.encoding",
                mode="streaming",
                hint=f"Open the session with one of the declared wire_encodings={wire}.",
            )

        if audio_format.channels != 1:
            raise UnsupportedFeatureError(
                f"Streaming wire format declares channels={audio_format.channels}; v1 "
                "streaming wire input is mono-only. The standard layer does not process "
                "incremental wire frames, so it cannot downmix multi-channel frames the "
                "way the batch path does. Downmix to mono before feeding.",
                param="audio_format.channels",
                mode="streaming",
                hint="Open the session with AudioFormat(..., channels=1) and downmix client-side.",
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
                    "mistranscribed. Open the session at an accepted rate.",
                    param="audio_format.sample_rate",
                    mode="streaming",
                    hint=f"Open the session at an accepted_sample_rates value: {accepted}.",
                )

    def _overrides_streaming(self) -> bool:
        """Return whether this engine implements the streaming hook.

        A streaming engine implements :meth:`_start_transcription`; a batch-only
        engine inherits the base no-op. The base :meth:`start_transcription`
        template uses this to raise the "does not support streaming" error
        *before* any parameter gating runs, so a non-streaming engine never
        surfaces a confusing wire-encoding / parameter error instead of the
        clear unsupported-streaming one.

        Returns:
            ``True`` if the concrete class overrides :meth:`_start_transcription`.
        """
        return type(self)._start_transcription is not EngineBase._start_transcription

    def start_transcription(
        self,
        *,
        audio_format: AudioFormat | None = None,
        params: RuntimeParams | None = None,
        audio: AudioInputLike | None = None,
    ) -> TranscriptionSession:
        """Open a streaming transcription session (template method).

        Symmetric to :meth:`transcribe`: the base runs the standard streaming
        pipeline and delegates only the engine-specific session construction to
        :meth:`_start_transcription`. The pipeline enforces input
        mutual-exclusion, validates the language config, validates the wire
        format, gates parameters against the ``streaming`` capabilities,
        resolves the language axis, prepares whole-input audio through the
        standard audio pipeline, and attaches the resulting diagnostics to the
        session.

        Because gating now runs here, spec Runtime R3 ``provider_params``
        swap-safety is enforced on the streaming path too: a swapped-engine
        ``provider_params`` type-mismatch always raises
        :class:`~standard_asr.exceptions.InvalidProviderParamError` (no longer
        undefined behaviour), and an unsupported standard parameter is rejected
        (strict) or dropped + diagnosed (best_effort) exactly as for batch.

        The streaming input/output capability axis is checked before the hook
        override defense, so an engine that implements the hook but does not
        declare the requested session mode fails on the missing capability
        rather than reaching parameter or audio gating. The hook override defense
        still runs before parameter gating, so a batch-only engine reports "does
        not support streaming" rather than a confusing parameter error -- while
        still running the input mutual-exclusion guard first, exactly as before.

        Spec Runtime R5 (streaming param freeze): the already-gated, frozen
        :class:`~standard_asr.runtime_params.RuntimeParams` is handed to the hook
        as ``gated_params``; the engine uses that for the whole session and MUST
        NOT re-accept raw params mid-stream.

        Args:
            audio_format: Wire format for incremental PCM frames.
            params: Per-request runtime parameters.
            audio: A complete audio input for whole-input streaming output.

        Returns:
            A streaming session with gating / language diagnostics attached.

        Raises:
            ValueError: If both ``audio_format`` and ``audio`` are provided.
            ConfigError: If the engine exposes a language axis but its
                ``default_language`` is unset or not in ``selectable_languages``.
            UnsupportedFeatureError: When the requested streaming input/output
                axis is unsupported, when streaming is unsupported, when the wire
                format is unreachable, or, in strict mode, on an unsupported
                parameter.
            IncompatibleAudioInputError: If no conversion path exists for a
                whole-input streaming ``audio`` value.
            InvalidProviderParamError: On wrong ``provider_params`` (swap-safety).
            ValueError: On an invalid candidate-language list in strict mode.
        """
        self.ensure_stream_inputs_exclusive(audio_format, audio)
        if audio_format is not None and not self.effective_capabilities.supports("streaming_input"):
            raise UnsupportedFeatureError(
                "start_transcription(audio_format=...) uses incremental PCM frame "
                "streaming mode and requires the streaming-input capability "
                "('streaming_input'); this engine does not declare streaming-input "
                "support.",
                param="audio_format",
                mode="streaming",
                hint=(
                    "Use an engine that declares 'streaming_input', or use "
                    "audio=... with an engine that declares 'streaming_output'."
                ),
            )
        if audio is not None and not self.effective_capabilities.supports("streaming_output"):
            raise UnsupportedFeatureError(
                "start_transcription(audio=...) uses whole-input streaming mode "
                "and requires the streaming-output capability ('streaming_output'); "
                "this engine does not declare streaming-output support.",
                param="audio",
                mode="streaming",
                hint=(
                    "Use an engine that declares 'streaming_output', or open an "
                    "audio_format=... session with an engine that declares "
                    "'streaming_input'."
                ),
            )
        if not self._overrides_streaming():
            raise UnsupportedFeatureError("This engine does not support streaming.")
        request = params or RuntimeParams()
        self._validate_language_config()
        if audio_format is not None:
            self.ensure_stream_format_supported(audio_format)
        gated, gate_diags = gate_params(
            request,
            self.effective_capabilities,
            "streaming",
            strict=self._strict,
            expected_provider_type=self.provider_params_type,
        )
        gated, lang_diags = self._resolve_language_axis(gated, "streaming")
        prepared: PreparedAudio | None = None
        if audio is not None:
            prepared = self._prepare_audio(audio)
        session = self._start_transcription(
            gated_params=gated, audio_format=audio_format, prepared_audio=prepared
        )
        # Friend API: the base engine seeds the session's standard-layer
        # diagnostics so they surface through the session's own diagnostics().
        session._attach_initial_diagnostics(  # pyright: ignore[reportPrivateUsage]
            [
                *gate_diags,
                *lang_diags,
                *(prepared.diagnostics if prepared is not None else []),
            ]
        )
        return session

    def _start_transcription(
        self,
        *,
        gated_params: RuntimeParams,
        audio_format: AudioFormat | None,
        prepared_audio: PreparedAudio | None,
    ) -> TranscriptionSession:
        """Construct the engine's streaming session (override point).

        Streaming engines override this to build and return their
        :class:`~standard_asr.streaming.TranscriptionSession`. It is invoked by
        the :meth:`start_transcription` template *after* the standard streaming
        pipeline (input exclusion, language config, wire-format validation,
        parameter gating, language resolution) has run, so the engine receives
        already-gated, frozen parameters and need not reimplement any gating.

        This is intentionally *not* abstract: batch-only engines inherit the
        default, which raises so a stray streaming call fails loudly.

        Args:
            gated_params: The gated, frozen runtime parameters (spec R5: these
                are frozen for the whole session).
            audio_format: Wire format for incremental PCM frames, if any.
            prepared_audio: Already-negotiated/resampled audio with conversion
                diagnostics for whole-input streaming, or ``None`` for the
                incremental ``audio_format`` path.

        Returns:
            The engine's streaming session.

        Raises:
            UnsupportedFeatureError: Always, in the base (streaming unsupported).
        """
        raise UnsupportedFeatureError("This engine does not support streaming.")


__all__ = ["EngineBase", "StandardASR"]
