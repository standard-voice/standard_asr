# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The Standard ASR engine interface (Protocol + ABC).

This module assembles the scattered pieces (config, properties, capabilities,
runtime params, result model, audio negotiation) into the single authoritative
engine contract. Two complementary forms are provided:

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
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, ClassVar, Protocol, cast, runtime_checkable

from .asr_config import BaseConfig
from .asr_properties import BaseProperties, sample_rate_accepted
from .audio_conversion import PreparedAudio, execute_plan
from .audio_format import AudioFormat
from .audio_input import AudioInput, AudioInputLike, coerce_audio_input
from .audio_negotiation import negotiate_or_raise
from .capabilities import DeclaredCapabilities
from .exceptions import ConfigError, UnsupportedFeatureError
from .language import AUTO, effective_candidate_languages, effective_language, normalize_bcp47
from .param_gating import Mode, gate_params
from .results import Diagnostic, TranscriptionResult
from .runtime_params import ProviderParams, RuntimeParams
from .wire import CANONICAL_WIRE_ENCODING

if TYPE_CHECKING:
    from .streaming import StreamDeadlines, TranscriptionSession


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

        Raises:
            ConfigError: On an invalid language configuration (spec LANG R1).
            IncompatibleAudioInputError: If no conversion path exists.
            UnsafeAudioUrlError: If an ``AudioUrl`` fails the R5 SSRF policy.
            AudioProcessingError: On a decode / size / missing-sample-rate
                failure in the conversion pipeline.
            UnsupportedFeatureError: In strict mode, on an unsupported parameter
                or a non-selectable ``language``.
            InvalidProviderParamError: On wrong ``provider_params`` (swap-safety).
            ValueError: On an invalid candidate-language list (always for a
                malformed/``auto`` candidate; strict-only for non-detectable /
                over-``max``; spec LANG R3).
            TranscriptionError: On an engine-execution failure (spec Runtime R7).
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

        Raises:
            Exception: The same exception set as :meth:`transcribe`.
        """
        ...

    def start_transcription(
        self,
        *,
        audio_format: AudioFormat | None = None,
        params: RuntimeParams | None = None,
        audio: AudioInputLike | None = None,
        deadlines: StreamDeadlines | None = None,
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
            deadlines: Application overrides for the session's termination
                deadlines; explicitly-set fields win over the adapter's
                construction-time choices.

        Returns:
            A streaming session.

        Raises:
            ValueError: If both ``audio_format`` and ``audio`` are provided.
            ConfigError: On an invalid language configuration (spec LANG R1).
            UnsupportedFeatureError: When streaming (or the requested streaming
                input/output axis) is unsupported, when the wire format is
                unreachable, or, in strict mode, on an unsupported parameter.
            IncompatibleAudioInputError: If no conversion path exists for a
                whole-input streaming ``audio`` value.
            UnsafeAudioUrlError: If a whole-input ``AudioUrl`` fails the R5 SSRF
                policy.
            AudioProcessingError: On a decode / size / missing-sample-rate
                failure for a whole-input ``audio`` value.
            InvalidProviderParamError: On wrong ``provider_params`` (swap-safety).
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

    ``selectable_languages`` is normalized at declaration time
    (``BaseProperties`` validates class-level defaults too), but
    ``default_language`` lives on ``BaseConfig`` (no normalization validator)
    and a third-party ``StandardASR`` implementation may not inherit
    ``BaseProperties`` at all, so membership tests canonicalize BOTH sides here
    as defense in depth rather than trusting either to be pre-normalized. The
    reserved ``auto`` directive is not a BCP-47 tag, so it is matched verbatim.

    Args:
        tag: A BCP-47 tag or the reserved ``auto`` token.

    Returns:
        The canonical form (``auto`` returned unchanged).

    Raises:
        ValueError: If ``tag`` is empty/whitespace (a malformed declaration).
    """
    return tag if tag == AUTO else normalize_bcp47(tag)


def _selectable_match(tag: str, selectable: AbstractSet[str]) -> str | None:
    """Return the selectable tag matching ``tag`` via RFC 4647 lookup, or ``None``.

    Implements the "Lookup" fallback of RFC 4647 §3.4 (normative for the runtime
    ``language`` axis, spec LANG R4): ``tag`` matches if its canonical form -- or
    any prefix obtained by progressively dropping trailing subtags -- is in the
    (canonical) selectable set. This lets an engine declare a primary language
    subtag (``en``) and still accept a region/script refinement of it (``en-US``,
    ``zh-Hant``), which the engine reduces internally, without enumerating every
    variant. Genuinely unrelated languages (``fr`` against an ``en`` engine) still
    do not match, and the reserved ``auto`` token has no subtags so it only ever
    matches verbatim.

    Per RFC 4647 §3.4, a single-character (singleton) subtag is removed **together
    with** the subtag that precedes it, so a private-use / extension sequence such
    as ``zh-x-foo`` truncates straight to ``zh`` (never the meaningless ``zh-x``).

    Args:
        tag: The canonical requested tag (or ``auto``).
        selectable: The canonical selectable set.

    Returns:
        The selectable tag that matched (equal to ``tag`` on an exact match, or a
        shorter prefix on a refinement match), or ``None`` if nothing matched.
    """
    parts = tag.split("-")
    i = len(parts)
    while i > 0:
        candidate = "-".join(parts[:i])
        if candidate in selectable:
            return candidate
        i -= 1
        # RFC 4647 §3.4: drop a singleton subtag together with the one before it.
        if i > 0 and len(parts[i - 1]) == 1:
            i -= 1
    return None


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

    #: Per-instance cache for :meth:`_canonical_language_sets` (the declared
    #: sets are class-level and immutable for the instance's lifetime).
    _language_sets_cache: tuple[frozenset[str], frozenset[str]] | None = None

    def _canonical_language_sets(self) -> tuple[frozenset[str], frozenset[str]]:
        """Canonicalize the declared language sets, once per engine instance.

        The single owner of the declared-side canonicalization shared by
        :meth:`_validate_language_config` and :meth:`_resolve_language_axis`:
        ``selectable_languages`` / ``detectable_languages`` may be declared as
        non-canonical class-level defaults (pydantic does not run field
        validators on defaults), and BCP-47 membership is case-insensitive, so
        every membership test must canonicalize both sides through the same
        rule. Centralizing it also gives a malformed declared tag ONE
        contract: the engine-naming :class:`ConfigError` (HTTP 422 through the
        server) on every path -- previously the detectable set was
        canonicalized per request inside
        :func:`~standard_asr.language.effective_candidate_languages`, where an
        empty class-default tag surfaced as an uncontracted bare
        ``ValueError`` (an opaque HTTP 500) instead.

        Returns:
            A ``(selectable, detectable)`` pair of canonical tag sets.

        Raises:
            ConfigError: If a declared selectable or detectable tag is
                malformed (empty or whitespace-only), naming the engine.
        """
        if self._language_sets_cache is not None:
            return self._language_sets_cache
        try:
            selectable = frozenset(
                _canonical_language(tag) for tag in self.properties.selectable_languages
            )
        except ValueError as exc:
            raise ConfigError(
                f"selectable_languages {self.properties.selectable_languages!r} declared "
                f"by engine {self.properties.engine_id!r} contains a malformed tag: {exc}"
            ) from exc
        try:
            detectable = frozenset(
                _canonical_language(tag) for tag in self.properties.detectable_languages
            )
        except ValueError as exc:
            raise ConfigError(
                f"detectable_languages {self.properties.detectable_languages!r} declared "
                f"by engine {self.properties.engine_id!r} contains a malformed tag: {exc}"
            ) from exc
        self._language_sets_cache = (selectable, detectable)
        return self._language_sets_cache

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

    def prepare(self) -> None:
        """Warm up the engine (download / load weights) without transcribing.

        The optional, **synchronous, idempotent** pre-warm hook (spec IC.11),
        invoked by ``standard-asr models prepare`` and by production / CI
        pre-warming (download-policy §4) to move the IC.9 lazy side effects
        (weight download / model load) off the first transcription to one
        billing-free, transcription-free call. The base implementation is a
        no-op: an engine with nothing to warm up inherits it unchanged and the
        toolchain reports a no-op rather than failing.

        Engines that load weights MUST override this to materialize them (e.g.
        call ``_ensure_model_loaded``), and that path MUST honour the same
        download gate as transcription: check
        :func:`~standard_asr.runtime.allow_downloads` and raise
        :class:`~standard_asr.exceptions.DiscoveryError` when downloads are
        disabled and weights are missing. An override MUST remain a zero-argument
        synchronous method -- never an ``async def`` (a coroutine function would
        be called but never awaited, silently reporting a false success); the
        compliance suite and the CLI reject a coroutine ``prepare``.

        Returns:
            None.

        Raises:
            DiscoveryError: An override SHOULD raise this when downloads are
                disabled and the weights are not already present (the base no-op
                never raises).
        """

    @property
    def _strict(self) -> bool:
        """Whether the unsupported-parameter policy is strict.

        Returns:
            ``True`` for strict, ``False`` for best_effort.
        """
        return bool(getattr(self.config, "strict", True))

    @property
    def _allow_private_urls(self) -> bool:
        """Whether the R5 SSRF policy is relaxed for private-address ``AudioUrl``.

        Sourced from the engine's init config (``BaseConfig.allow_private_urls``;
        spec R5). It is an init-level deployment switch -- a trust decision about
        the deployment's network, not a per-request parameter -- so it lives on
        the config, never on ``RuntimeParams``, and (like ``strict``) is excluded
        from environment fallback so the environment cannot silently relax the
        SSRF guard. Read defensively (mirroring :attr:`_strict`) so a structural
        engine whose config omits the field stays fail-closed (default ``False``).

        Returns:
            ``True`` to allow private/loopback/link-local URL targets (HTTPS is
            still required), ``False`` to keep the default SSRF rejection.
        """
        return bool(getattr(self.config, "allow_private_urls", False))

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
        Runtime R3: "fail fast on provider_params first").

        Args:
            audio: The audio to transcribe.
            params: Per-request runtime parameters.

        Returns:
            The transcription result with gating / language / conversion
            diagnostics attached.

        Raises:
            ConfigError: If the engine exposes a language axis but its
                ``default_language`` is unset, malformed, or not in
                ``selectable_languages``; or if a declared selectable/detectable
                tag is itself malformed (an engine-declaration bug).
            IncompatibleAudioInputError: If no conversion path exists.
            UnsafeAudioUrlError: If an ``AudioUrl`` fails the R5 SSRF policy
                (non-HTTPS, or a private/loopback/link-local target).
            AudioProcessingError: On an audio failure surfaced by the conversion
                pipeline -- a decode failure, an over-``max_file_size`` payload,
                or (in strict mode) a bare array with no sample rate.
            UnsupportedFeatureError: In strict mode, on an unsupported parameter
                or a requested ``language`` not selectable by the engine.
            InvalidProviderParamError: On wrong provider params.
            ValueError: On an invalid candidate-language list -- a malformed
                candidate tag or one containing ``auto`` raises **always**
                (independent of strict/best_effort, spec LANG R3); a
                non-detectable or over-``max`` candidate raises only in strict
                mode.
            TranscriptionError: On an engine-execution failure inside
                :meth:`_transcribe` (spec Runtime R7).
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
        gated, lang_diags = self._resolve_language_axis(
            gated, "batch", requested_language=request.language
        )
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
            allow_private_addresses=self._allow_private_urls,
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
                is unset, malformed (empty/whitespace), or not in
                ``selectable_languages``; or if a declared selectable or
                detectable tag is itself malformed.
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
        # either default_language or the declared sets may be a non-canonical
        # class-level default, so a raw "en-us" must still match a canonical
        # "en-US" instead of spuriously failing LANG R1 and blocking the engine.
        # Canonicalization raises ValueError on an empty/whitespace tag; this
        # method promises ConfigError, so wrap it naming the malformed value (a
        # language tag is not a secret -- echoing it is safe and actionable).
        try:
            canonical_default = _canonical_language(default)
        except ValueError as exc:
            raise ConfigError(
                f"default_language {default!r} is malformed for engine "
                f"{self.properties.engine_id!r}: {exc} (spec LANG R1)."
            ) from exc
        selectable, _ = self._canonical_language_sets()
        if canonical_default not in selectable:
            raise ConfigError(
                f"default_language {default!r} is not in selectable_languages "
                f"{self.properties.selectable_languages!r} "
                f"(engine {self.properties.engine_id!r}, spec LANG R1)."
            )

    def _resolve_language_axis(
        self, params: RuntimeParams, mode: Mode, *, requested_language: str | None = None
    ) -> tuple[RuntimeParams, list[Diagnostic]]:
        """Resolve and validate the effective language axis (LANG R2/R3).

        Runs standard resolution so the engine receives the same effective
        ``language`` and ``candidate_languages`` values that the standard layer
        validated and diagnosed.

        Args:
            params: The gated runtime parameters (``params.language`` is already
                ``None`` if the gate dropped an unsupported override).
            mode: ``"batch"`` or ``"streaming"``.
            requested_language: The language the caller *originally* requested,
                before gating. Used only to report the true effective value when
                the gate dropped a per-request language because the engine does
                not support ``language.runtime_override``: the gate's
                ``unsupported_parameter_ignored`` diagnostic records
                ``effective=None``, but the engine actually transcribes with its
                ``default_language``. Pass the un-gated ``RuntimeParams.language``;
                defaults to ``None`` for direct callers that never gated.

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
        runtime_override_supported = caps.supports(f"{mode}.language.runtime_override")
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
            runtime_override_supported=runtime_override_supported,
        )
        if eff_lang is not None and eff_lang != AUTO:
            eff_lang = normalize_bcp47(eff_lang)

        diagnostics: list[Diagnostic] = []
        # Complete the gate's best_effort story for a dropped per-request language.
        # When the engine lacks ``language.runtime_override`` and the caller DID
        # request a language, the gate dropped it (effective=None in its
        # diagnostic) and ``effective_language`` fell back to default_language
        # here -- so the request is actually transcribed in default_language, a
        # "final value" the caller could not otherwise see (the gate has no access
        # to default_language; the spec requires the best_effort diagnostics to
        # surface the final value). Emit it explicitly. (Strict mode never reaches
        # here for this case: the gate raises on the unsupported language first.)
        if requested_language is not None and not runtime_override_supported:
            diagnostics.append(
                Diagnostic(
                    level="warning",
                    code="language_fell_back",
                    message=(
                        f"Per-request language was dropped (engine does not support "
                        f"language.runtime_override in {mode} mode); transcribing with "
                        f"default_language {eff_lang!r}."
                    ),
                    param="language",
                    provided=requested_language,
                    effective=eff_lang,
                )
            )
        # Both declared sets come canonical (and ConfigError-checked) from the
        # shared per-engine canonicalization, so a canonical eff_lang matches
        # case-insensitively. Membership uses RFC 4647 lookup (spec LANG R4) so a
        # region/script refinement of a selectable primary subtag (e.g. "en-US"
        # against "en") is accepted and handed to the engine to reduce -- engines
        # need not enumerate variants.
        selectable, detectable = self._canonical_language_sets()
        # eff_lang is non-None here: this method only runs when has_language_axis
        # is True, and effective_language then returns default_language (R1
        # guarantees it is set, enforced by _validate_language_config above), or
        # the request override -- both non-None. `auto` has no subtags, so
        # _selectable_match treats it as an exact membership test (matched ==
        # "auto" when selectable, else None), preserving the prior
        # auto-selectability behavior; the refinement branch below applies only to
        # real BCP-47 tags.
        assert eff_lang is not None
        matched = _selectable_match(eff_lang, selectable)
        if matched is None:
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
        elif matched != eff_lang:
            # Accepted as an RFC 4647 refinement of a selectable primary subtag
            # (spec LANG R4): the engine receives the full requested tag and
            # reduces it internally. Surface an informational diagnostic so the
            # caller can see the tag was matched by reduction rather than exact
            # membership (no value is changed).
            diagnostics.append(
                Diagnostic(
                    level="info",
                    code="language_refinement_accepted",
                    message=(
                        f"language {eff_lang!r} accepted in {mode} mode as a "
                        f"refinement of selectable {matched!r} (RFC 4647 lookup); "
                        "the engine reduces it internally."
                    ),
                    param="language",
                    provided=eff_lang,
                    effective=eff_lang,
                )
            )

        constraints = self._candidate_max(mode)
        eff_candidates, candidate_diags = effective_candidate_languages(
            eff_lang,
            params.candidate_languages,
            default_candidates,
            candidate_supported=caps.supports(f"{mode}.language.candidate_languages"),
            detectable_languages=detectable,
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

        Raises:
            Exception: The same exception set as :meth:`transcribe` (it runs that
                method): ``ConfigError``, ``IncompatibleAudioInputError``,
                ``UnsafeAudioUrlError``, ``AudioProcessingError``,
                ``UnsupportedFeatureError``, ``InvalidProviderParamError``,
                ``ValueError``, and ``TranscriptionError``.
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

        Raises:
            TranscriptionError: On any engine-execution failure (model inference,
                network call, or SDK error). Implementations MUST wrap the native
                exception as ``raise TranscriptionError(...) from exc`` so an
                application can catch one portable type across every engine
                instead of each engine's native exception (spec Runtime R7). This
                is the batch counterpart of the streaming ``engine_error`` event
                (spec ST §6.2).
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
        ``audio_format=...`` session. It is **fail-closed** on the wire sample
        rate and the channel count unconditionally, and on the wire encoding
        **when ``wire_encodings`` is declared**.

        Wire **encoding**: when the engine declares ``wire_encodings``, an
        encoding not among them is rejected up front rather than misframed as PCM
        and silently mistranscribed. When ``wire_encodings`` is ``None``
        ("unconstrained", spec §AI) the encoding cannot be validated and the
        check is skipped -- the engine is then trusted to accept any encoding
        (typically a self-managed-wire-format adapter). The compliance suite
        emits a warning for a ``streaming_input`` engine that leaves
        ``wire_encodings`` unset, since that skip is where a forgotten
        declaration would let a non-PCM frame be misframed.

        Wire **sample rate**: spec R7's v1 implementation note is explicit that
        v1 does **NOT** resample streaming bare frames in the standard layer
        (unlike the batch ``transcribe`` path, which resamples). Therefore, until
        standard-layer streaming resampling lands, a wire ``sample_rate`` that the
        engine does not accept MUST be rejected here rather than forwarded as
        frames the engine never declared -- a loud error beats a silent
        mistranscription. When ``required_input_sample_rate`` is set, the wire
        rate MUST equal it -- even when ``accepted_sample_rates`` is ``"any"``
        (that combination is constructible; the declaration-time reachability
        validator only checks concrete lists). Otherwise the rate is accepted
        when ``accepted_sample_rates`` is ``"any"`` or when it is in that
        concrete list.

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

        rate = audio_format.sample_rate
        required = props.required_input_sample_rate
        # A hard-required wire rate binds regardless of accepted_sample_rates:
        # "any" + required_input_sample_rate is constructible (the declaration
        # reachability validator only checks concrete lists), and v1 does not
        # resample streaming wire frames, so a differing rate fails closed here.
        if required is not None and rate != required:
            raise UnsupportedFeatureError(
                f"Streaming wire sample_rate {rate} Hz does not match the "
                f"required_input_sample_rate={required} Hz that engine "
                f"{props.engine_id!r} hard-requires. v1 does not resample streaming "
                "wire frames, so the required rate is enforced at session "
                "establishment even when accepted_sample_rates is 'any'.",
                param="audio_format.sample_rate",
                mode="streaming",
                hint=f"Open the session at sample_rate={required}.",
            )
        accepted = props.accepted_sample_rates
        if accepted != "any" and not sample_rate_accepted(accepted, rate):
            raise UnsupportedFeatureError(
                f"Streaming wire sample_rate {rate} Hz is not accepted by engine "
                f"{props.engine_id!r} (accepted_sample_rates={accepted!r}). v1 does "
                "not resample streaming wire frames, so an unreachable rate is "
                "rejected at session establishment rather than silently "
                "mistranscribed. Open the session at an accepted rate.",
                param="audio_format.sample_rate",
                mode="streaming",
                hint=f"Open the session at an accepted_sample_rates value: {accepted!r}.",
            )

    def recommended_wire_format(self) -> AudioFormat | None:
        """Return a minimal wire :class:`AudioFormat` to open a streaming session.

        Single source of truth for the legal bare-frame wire format the standard
        layer uses when it must open a ``streaming_input`` session but has no
        application-chosen format -- the CLI sync-bridge runner and the streaming
        gating probe both rely on it. They previously derived one independently and
        disagreed (which sample-rate source to use, and what to do with no declared
        ``wire_encodings``); this unifies them (AW-2). The format is built from the
        engine's own Properties so :meth:`ensure_stream_format_supported` accepts
        it (the compliance suite asserts that round-trip):

        * ``sample_rate`` = ``required_input_sample_rate`` when the engine
          hard-requires one, else ``native_sample_rate`` (the R7 reachability
          invariant guarantees the native rate is accepted).
        * ``encoding`` = the first declared ``wire_encodings`` entry, else the
          canonical ``pcm_s16le`` (used only when ``wire_encodings`` is
          unconstrained, where the engine accepts any encoding).
        * ``channels`` = 1 (v1 streaming wire is mono-only).

        Returns:
            A wire format the engine's session-establishment guard accepts, or
            ``None`` when the engine declares no usable (positive) sample rate, so
            no bare-frame streaming format can be recommended.
        """
        props = self.properties
        # required_input_sample_rate (int | None) wins when set; else the native
        # rate. ``or`` also treats a 0 required-rate as unset. The result is typed
        # ``int``; a non-positive rate (a malformed declaration) yields no
        # recommendation -- no bare-frame session can open without a positive rate.
        sample_rate = props.required_input_sample_rate or props.native_sample_rate
        if sample_rate <= 0:
            return None
        wire = props.wire_encodings
        encoding = wire[0] if wire else CANONICAL_WIRE_ENCODING
        return AudioFormat(encoding=encoding, sample_rate=sample_rate, channels=1)

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
        deadlines: StreamDeadlines | None = None,
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
            deadlines: Application overrides for the session's termination
                deadlines (spec ST.6.1). Applied by this template *after* the
                engine hook constructed the session, so explicitly-set fields
                always win over the adapter's construction-time choices --
                precedence: application explicit > adapter choice > standard
                default. Unset fields are left untouched.

        Returns:
            A streaming session with gating / language diagnostics attached.

        Raises:
            ValueError: If both ``audio_format`` and ``audio`` are provided, or
                on an invalid candidate-language list (always for a
                malformed/``auto`` candidate; strict-only for non-detectable /
                over-``max``; spec LANG R3).
            ConfigError: If the engine exposes a language axis but its
                ``default_language`` is unset, malformed, or not in
                ``selectable_languages``.
            UnsupportedFeatureError: When the requested streaming input/output
                axis is unsupported, when streaming is unsupported, when the wire
                format is unreachable, or, in strict mode, on an unsupported
                parameter.
            IncompatibleAudioInputError: If no conversion path exists for a
                whole-input streaming ``audio`` value.
            UnsafeAudioUrlError: If a whole-input ``AudioUrl`` fails the R5 SSRF
                policy.
            AudioProcessingError: On a decode / size / missing-sample-rate
                failure for a whole-input ``audio`` value.
            InvalidProviderParamError: On wrong ``provider_params`` (swap-safety).
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
            raise UnsupportedFeatureError(
                f"Engine {self.properties.engine_id!r} does not support streaming.",
                mode="streaming",
                hint="Use an engine that declares 'streaming_input' or 'streaming_output'.",
            )
        # A bare call (neither audio_format nor audio) opens an INCREMENTAL session
        # for an engine that self-manages its wire format (spec ST §3.1), which is
        # the streaming-input axis. Gate it on the same 'streaming_input' capability
        # as the audio_format path; otherwise a streaming_output-only engine that
        # implements the hook would hand back an incremental session it cannot feed
        # (audio_format=None, prepared_audio=None) -- undefined behaviour instead of
        # the fail-closed UnsupportedFeatureError (R1). Placed AFTER the hook-override
        # defense so a batch-only engine still reports the clearer "does not support
        # streaming" rather than this capability-specific message.
        if (
            audio_format is None
            and audio is None
            and not self.effective_capabilities.supports("streaming_input")
        ):
            raise UnsupportedFeatureError(
                "start_transcription() with no audio_format/audio opens an "
                "incremental (self-managed wire format) session, which requires "
                "the streaming-input capability ('streaming_input'); this engine "
                "does not declare streaming-input support.",
                param="audio_format",
                mode="streaming",
                hint=(
                    "Use an engine that declares 'streaming_input', or pass "
                    "audio=... with an engine that declares 'streaming_output'."
                ),
            )
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
        gated, lang_diags = self._resolve_language_axis(
            gated, "streaming", requested_language=request.language
        )
        prepared: PreparedAudio | None = None
        if audio is not None:
            prepared = self._prepare_audio(audio)
        session = self._start_transcription(
            gated_params=gated, audio_format=audio_format, prepared_audio=prepared
        )
        # Friend API: validate the reserved-attribute guard now, before the base
        # seeds diagnostics / applies deadline overrides below -- so the check sees
        # the pristine post-__init__ snapshot and a subclass that clobbered base
        # state (e.g. its own self._buffer) fails loudly here (SF-1), not as a
        # cryptic crash deep in the producer.
        session._ensure_reserved_attrs_checked()  # pyright: ignore[reportPrivateUsage]
        # Friend API: the base engine seeds the session's standard-layer
        # diagnostics so they surface through the session's own diagnostics().
        session._attach_initial_diagnostics(  # pyright: ignore[reportPrivateUsage]
            [
                *gate_diags,
                *lang_diags,
                *(prepared.diagnostics if prepared is not None else []),
            ]
        )
        if deadlines is not None:
            session._apply_deadline_overrides(deadlines)  # pyright: ignore[reportPrivateUsage]
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
        raise UnsupportedFeatureError(
            f"Engine {self.properties.engine_id!r} does not support streaming.",
            mode="streaming",
            hint="Use an engine that declares 'streaming_input' or 'streaming_output'.",
        )


__all__ = ["EngineBase", "StandardASR"]
