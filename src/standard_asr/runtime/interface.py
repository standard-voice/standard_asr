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
(:class:`EngineBase`), so authors get consistent, correct behavior for free.
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, cast, final, runtime_checkable

from pydantic import ValidationError

from standard_asr.audio.conversion import PreparedAudio, execute_plan
from standard_asr.audio.format import AudioFormat
from standard_asr.audio.input import AudioInput, AudioInputLike, coerce_audio_input
from standard_asr.audio.negotiation import negotiate_or_raise
from standard_asr.audio.wire import CANONICAL_WIRE_ENCODING
from standard_asr.contract.artifacts import (
    ARTIFACT_BLOCKER_ACTION_REQUIRED,
    ARTIFACT_BLOCKER_DOWNLOADS_DISABLED,
    ARTIFACT_PROGRESS_FINALIZING,
    ARTIFACT_PROGRESS_RESOLVING,
    ARTIFACT_READY,
    ArtifactAction,
    ArtifactContext,
    ArtifactProgress,
    ArtifactProgressCallback,
    ArtifactReport,
    ArtifactRequirement,
)
from standard_asr.contract.capabilities import DeclaredCapabilities
from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactProgressCallbackError,
    ArtifactStatusError,
    ConfigError,
    EngineContractError,
    InvalidProviderParamError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from standard_asr.contract.language import (
    AUTO,
    DIAG_LANGUAGE_FELL_BACK,
    DIAG_LANGUAGE_NOT_SELECTABLE,
    DIAG_LANGUAGE_REFINEMENT_ACCEPTED,
    effective_candidate_languages,
    effective_language,
    normalize_bcp47,
)
from standard_asr.contract.metadata import ArtifactDeclaration, DeclaredEngineMetadata
from standard_asr.contract.params import ProviderParams, RuntimeParams
from standard_asr.contract.properties import BaseProperties, sample_rate_accepted
from standard_asr.contract.protocol_version import require_supported_protocol
from standard_asr.contract.results import (
    ChannelResult,
    Diagnostic,
    Segment,
    TranscriptionResult,
    synthesize_segment_speaker,
)
from standard_asr.runtime.config import BaseConfig
from standard_asr.runtime.downloads import allow_downloads
from standard_asr.runtime.gating import Mode, gate_params
from standard_asr.runtime.protocol_boundary import require_sync_result, safe_type_name
from standard_asr.runtime.streaming import TranscriptionSession

if TYPE_CHECKING:
    from standard_asr.runtime.streaming import StreamDeadlines


@runtime_checkable
class StandardASR(Protocol):
    """Structural protocol for a Standard ASR engine.

    Any object exposing these members is a compliant engine, regardless of how
    it is implemented. The protocol describes the *full* public surface every
    engine exposes -- batch (:meth:`transcribe` / :meth:`transcribe_async`) and
    the streaming entry point (:meth:`start_transcription`). ``start_transcription``
    is always present; streaming support itself is optional, so a batch-only
    engine raises :class:`~standard_asr.contract.exceptions.UnsupportedFeatureError` from
    it. Because the surface is complete, callers (for example, the server) can type an
    engine as ``StandardASR`` and call the streaming entry point without a cast.
    """

    properties: ClassVar[BaseProperties]
    declared_capabilities: ClassVar[DeclaredCapabilities]
    declared_metadata: ClassVar[DeclaredEngineMetadata]

    @property
    def config(self) -> BaseConfig[str]:
        """The engine's runtime configuration.

        Declared as a READ-ONLY property, not a mutable attribute: a mutable
        protocol member is invariant under strict typing, so a real engine
        annotating its own subtype (``config: WhisperConfig``) would not be
        structurally assignable to ``StandardASR`` without a cast -- defeating
        the protocol's own no-cast promise above. Read-only makes the member
        covariant (any engine's narrower config satisfies it), and matches
        intent: config is constructor-injected; callers never reassign it
        through the protocol. Implementations satisfy this with a plain
        (even mutable) instance attribute -- no ``@property`` required.

        Returns:
            The engine's config instance.
        """
        ...

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
            ProtocolCompatibilityError: If the engine's declared protocol
                line is not supported by this core (checked before any work).
            ConfigError: On an invalid language configuration VALUE
                (``default_language`` malformed or not selectable) -- fixable
                by whoever supplies the config.
            EngineContractError: On an engine-declaration defect -- a
                declared language axis with no ``default_language`` (IC.6),
                or a malformed declared selectable/detectable tag.
            IncompatibleAudioInputError: If no conversion path exists.
            UnsafeAudioUrlError: If an ``AudioUrl`` fails the SSRF policy.
            AudioProcessingError: On a decode / size / missing-sample-rate
                failure in the conversion pipeline.
            UnsupportedFeatureError: In strict mode, on an unsupported parameter,
                a non-selectable ``language``, or a valid-but-unreachable
                candidate list (non-detectable candidate / over-``max``).
            InvalidProviderParamError: On wrong ``provider_params`` (swap-safety).
            ValueError: On a malformed or ``"auto"`` candidate-language entry
                (a caller code bug; raises independent of strict/best_effort).
            ArtifactUnavailableError: When required inference artifacts cannot
                support recognition under the current policy.
            ArtifactAcquisitionError: When an allowed implicit acquisition
                attempt fails before recognition.
            TranscriptionError: On an engine-execution failure.
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
        :class:`~standard_asr.contract.exceptions.UnsupportedFeatureError` here. Callers
        that need streaming should gate on
        ``supports("streaming_input")`` / ``supports("streaming_output")`` (or be
        ready to handle the unsupported-streaming error).

        Args:
            audio_format: Wire format for incremental PCM frames.
            params: Per-request runtime parameters.
            audio: A complete audio input for whole-input streaming output.
            deadlines: Application overrides for the session's termination
                deadlines; explicitly set fields win over the engine's
                construction-time choices.

        Returns:
            A streaming session.

        Raises:
            ProtocolCompatibilityError: If the engine's declared protocol
                line is not supported by this core (checked before any work).
            ValueError: If both ``audio_format`` and ``audio`` are provided, or
                on a malformed/``auto`` candidate-language entry (a caller code
                bug; always raises, independent of strict/best_effort).
            ConfigError: On an invalid language configuration VALUE
                (``default_language`` malformed or not selectable).
            EngineContractError: On an engine-declaration defect -- a
                declared language axis with no ``default_language`` (IC.6),
                or a malformed declared selectable/detectable tag.
            UnsupportedFeatureError: When streaming (or the requested streaming
                input/output axis) is unsupported, when the wire format is
                unreachable, or, in strict mode, on an unsupported parameter or
                a valid-but-unreachable candidate list (non-detectable /
                over-``max``).
            IncompatibleAudioInputError: If no conversion path exists for a
                whole-input streaming ``audio`` value.
            UnsafeAudioUrlError: If a whole-input ``AudioUrl`` fails the SSRF
                policy.
            AudioProcessingError: On a decode / size / missing-sample-rate
                failure for a whole-input ``audio`` value.
            InvalidProviderParamError: On wrong ``provider_params`` (swap-safety).
            ArtifactUnavailableError: When required inference artifacts cannot
                support session establishment under the current policy.
            ArtifactAcquisitionError: When an allowed implicit acquisition
                attempt fails during session establishment.
            TranscriptionError: When a pydantic ``ValidationError`` escapes the
                engine's session-construction hook (an invalid model
                construction is an engine fault, never a request error).
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

    def artifact_status(
        self,
        context: ArtifactContext | None = None,
    ) -> ArtifactReport:
        """Inspect inference-artifact readiness without acquiring anything.

        Args:
            context: Optional request context. ``None`` resolves an engine mode
                and uses default runtime parameters.

        Returns:
            A point-in-time artifact report for the resolved context.

        Raises:
            ProtocolCompatibilityError: If the engine's declared protocol
                line is not supported by this core.
            InvalidProviderParamError: If provider params belong to another
                engine.
            ValueError: If request language data is malformed or the explicit
                mode is not supported by the engine.
            ConfigError: If engine configuration is invalid.
            EngineContractError: If engine declarations or hook results violate
                the protocol.
            ArtifactStatusError: If native status inspection fails.
        """
        ...

    def acquire_artifacts(
        self,
        context: ArtifactContext | None = None,
        *,
        refresh: bool = False,
        progress: ArtifactProgressCallback | None = None,
    ) -> ArtifactReport:
        """Acquire inference artifacts explicitly and return fresh status.

        Args:
            context: Optional request context.
            refresh: Whether to re-resolve unblocked mutable source references.
            progress: Optional synchronous progress observer.

        Returns:
            A newly inspected artifact report.

        Raises:
            ProtocolCompatibilityError: If the engine's declared protocol
                line is not supported by this core.
            ArtifactStatusError: If preflight or final status inspection fails.
            ArtifactAcquisitionError: If acquisition is blocked or fails.
            ArtifactProgressCallbackError: After successful acquisition and
                status inspection, if the observer failed.
            EngineContractError: If the engine violates the operation contract.
            InvalidProviderParamError: If provider params belong to another
                engine.
            ConfigError: If engine configuration is invalid.
            ValueError: If request language data is malformed or the explicit
                mode is not supported by the engine.
        """
        ...

    def recommended_wire_format(self) -> AudioFormat | None:
        """Return a wire :class:`AudioFormat` this engine accepts for streaming.

        Part of the protocol because it is the documented first step of the
        streaming journey (README / quickstart / streaming guide all start with
        it) and the toolchain's sync-bridge runner and gating probes rely on it
        -- a member every caller is taught to invoke MUST be part of the
        contract, or a structural engine (and every ``StandardASR``-typed
        variable) breaks on the standard's own 80% path. The value is purely
        derivable from the engine's static Properties (see
        :meth:`EngineBase.recommended_wire_format` for the derivation
        ``EngineBase`` provides for free) -- deliberately capability-blind:
        whether a bare-frame session can be OPENED is the
        ``streaming_input`` capability gate's job inside
        :meth:`start_transcription`, so a batch-only or output-only engine
        still derives a format here (callers gate on
        ``supports("streaming_input")`` first, per the streaming guide).

        Returns:
            A wire format the engine's session-establishment guard accepts, or
            ``None`` when no bare-frame streaming format can be recommended.
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
    ``language`` axis): ``tag`` matches if its canonical form -- or
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


def _synthesize_segment_list(
    segments: list[Segment] | None,
) -> tuple[list[Segment] | None, bool]:
    """Fill missing segment speakers from word speakers in one segment list.

    Applies the pinned segment-speaker synthesis rule
    (:func:`~standard_asr.contract.results.synthesize_segment_speaker`) to every segment
    whose ``speaker`` is ``None`` while its ``words`` carry speakers. Segments
    are frozen models, so a changed segment is rebuilt via ``model_copy``
    (which skips validators -- safe, because the synthesized label is one of
    the already-validated ``Word.speaker`` values); unchanged segments are
    reused as-is so an untouched list keeps object identity for its members.

    Args:
        segments: The segment list, or ``None``.

    Returns:
        A ``(segments, changed)`` pair: the original list object when nothing
        changed, otherwise a new list with the synthesized segments swapped in.
    """
    if segments is None:
        return None, False
    changed = False
    out: list[Segment] = []
    for segment in segments:
        if segment.speaker is None and segment.words:
            label = synthesize_segment_speaker(segment.words)
            if label is not None:
                out.append(segment.model_copy(update={"speaker": label}))
                changed = True
                continue
        out.append(segment)
    return (out, True) if changed else (segments, False)


def _synthesize_result_speakers(result: TranscriptionResult) -> TranscriptionResult:
    """Synthesize missing segment speakers across a whole batch result.

    Runs over the top-level ``segments`` AND every ``channels[i].segments``:
    the result contract promises the two views agree, so synthesizing only the
    top level would silently desync them. Runs regardless of whether diarization
    was requested -- an always-on engine's unrequested-but-legal word speakers
    (a named exemption) benefit identically. When nothing changes the
    input is returned unchanged (identity fast path: no churn, and untouched
    results keep object equality for callers).

    Args:
        result: The engine's raw transcription result.

    Returns:
        The result with segment speakers synthesized where derivable, or
        ``result`` itself when no segment needed one.
    """
    segments, segments_changed = _synthesize_segment_list(result.segments)
    channels: list[ChannelResult] | None = result.channels
    channels_changed = False
    if channels is not None:
        rebuilt: list[ChannelResult] = []
        for entry in channels:
            entry_segments, entry_changed = _synthesize_segment_list(entry.segments)
            if entry_changed:
                rebuilt.append(entry.model_copy(update={"segments": entry_segments}))
                channels_changed = True
            else:
                rebuilt.append(entry)
        if channels_changed:
            channels = rebuilt
    if not segments_changed and not channels_changed:
        return result
    update: dict[str, object] = {}
    if segments_changed:
        update["segments"] = segments
    if channels_changed:
        update["channels"] = channels
    return result.model_copy(update=update)


def ensure_wire_format_supported(properties: BaseProperties, audio_format: AudioFormat) -> None:
    """Validate a streaming wire format against an engine's declared Properties.

    The standard's session-establishment format rule as a PURE function of
    ``(Properties, AudioFormat)`` -- the single owner shared by
    :meth:`EngineBase.ensure_stream_format_supported` (the template's
    establishment guard) and the compliance suite's
    ``check_recommended_wire_format`` round-trip. Compliance validating through
    this function instead of the ``EngineBase`` method keeps the check honest
    for structural (non-``EngineBase``) engines: the guard method is NOT a
    ``StandardASR`` protocol member, so calling it on a structural engine
    raised ``AttributeError`` and mis-reported a fully compliant engine as
    self-inconsistent.

    See :meth:`EngineBase.ensure_stream_format_supported` for the full
    normative semantics (fail-closed on sample rate and channels; fail-closed
    on encoding only when ``wire_encodings`` is declared).

    Args:
        properties: The engine's declared static Properties.
        audio_format: The wire format the session declared.

    Raises:
        UnsupportedFeatureError: If ``wire_encodings`` is declared and the
            requested encoding is not among them, if the wire ``channels`` is
            not ``1`` (v1 streaming wire is mono-only), or if the wire sample
            rate is not reachable for the engine (fail-closed; v1 does not
            resample streaming wire frames).
    """
    props = properties
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


def require_engine_protocol(engine: object) -> BaseProperties:
    """Require an engine or engine class to be on the supported protocol line.

    The one fail-closed gate every selected-engine surface shares: registry
    creation, the ``EngineBase`` inference and artifact entries, the CLI's
    ``show`` and transcribe preflight, and the per-model metadata endpoints
    all call it before interpreting anything the engine declares. Consumers
    that look up ``artifact_status`` or ``acquire_artifacts`` call it FIRST,
    so an engine on an unsupported line gets a typed compatibility error
    instead of a missing-attribute error -- and an unsupported line MUST NOT
    be read as ``NO_ARTIFACT_LIFECYCLE``. Missing or untyped ``properties``
    fails closed too: an engine whose protocol line cannot be established is
    less knowable than one on a wrong line, never more trustworthy. The check
    is the general line gate
    (:func:`~standard_asr.contract.protocol_version.require_supported_protocol`),
    not a feature-table lookup: every supported line carries the artifact
    lifecycle, so no per-feature minimum survives to gate it -- which also
    keeps the first stable release's feature-table rewrite from breaking
    this guard.

    Args:
        engine: Engine instance or engine class whose declared protocol
            version is inspected.

    Returns:
        The validated ``BaseProperties`` declaration.

    Raises:
        EngineContractError: If the engine does not declare ``properties`` as
            a ``BaseProperties`` instance.
        ProtocolCompatibilityError: If its declared protocol line is not
            supported by this core.
    """
    properties = getattr(engine, "properties", None)
    if not isinstance(properties, BaseProperties):
        raise EngineContractError(
            "Engine protocol compatibility cannot be established: 'properties' "
            "is missing or is not a BaseProperties instance."
        )
    require_supported_protocol(properties.protocol_version)
    return properties


class _ArtifactProgressObserver:
    """Serialize progress delivery and retain the first observer failure."""

    def __init__(self, callback: ArtifactProgressCallback) -> None:
        self._callback = callback
        self._lock = threading.Lock()
        self.callback_error: BaseException | None = None
        self.contract_error: EngineContractError | None = None

    def __call__(self, event: ArtifactProgress) -> None:
        """Validate and deliver one event without canceling acquisition.

        Args:
            event: Engine-authored progress event.

        Returns:
            None.
        """
        with self._lock:
            if self.callback_error is not None or self.contract_error is not None:
                return
            if type(event) is not ArtifactProgress:
                self.contract_error = EngineContractError(
                    "_acquire_artifacts progress callback received a value that "
                    f"is not ArtifactProgress (got {safe_type_name(event)})."
                )
                return
            try:
                validated = ArtifactProgress.model_validate(event.model_dump(mode="python"))
            except Exception as exc:  # noqa: BLE001 - retained as a contract-error cause
                self.contract_error = EngineContractError(
                    "_acquire_artifacts progress callback received an invalid "
                    "ArtifactProgress value."
                )
                self.contract_error.__cause__ = exc
                return
            try:
                result = self._callback(validated)
                require_sync_result(
                    result,
                    "artifact progress callback",
                    expected_type=type(None),
                )
            except asyncio.CancelledError as exc:
                self.callback_error = exc
            except Exception as exc:  # noqa: BLE001 - reported after acquisition succeeds
                self.callback_error = exc


def _artifact_blocker_reason(
    requirements: tuple[ArtifactRequirement, ...],
) -> Literal["downloads_disabled", "action_required", "unsupported"]:
    """Project blockers onto the portable acquisition-error reason.

    Args:
        requirements: Blocked requirements.

    Returns:
        The portable reason, using the contract precedence. Unknown blocker
        tokens conservatively project to ``"unsupported"``.
    """
    blockers = {requirement.acquisition_blocker for requirement in requirements}
    if ARTIFACT_BLOCKER_ACTION_REQUIRED in blockers:
        return "action_required"
    if ARTIFACT_BLOCKER_DOWNLOADS_DISABLED in blockers:
        return "downloads_disabled"
    return "unsupported"


def _artifact_required_actions(
    requirements: tuple[ArtifactRequirement, ...],
) -> tuple[ArtifactAction, ...]:
    """Flatten actions from requirements while preserving report order.

    Args:
        requirements: Requirements whose actions are collected.

    Returns:
        The ordered action tuple.
    """
    return tuple(action for requirement in requirements for action in requirement.required_actions)


class EngineBase(ABC):
    """Abstract base implementing the standard transcribe pipeline.

    Subclasses MUST set :attr:`properties`,
    :attr:`declared_capabilities`, and :attr:`declared_metadata` as class
    attributes, assign :attr:`config` in ``__init__`` (which MUST stay pure --
    no filesystem, GPU, or network access), and implement :meth:`_transcribe`.
    Streaming engines additionally override
    :meth:`_start_transcription` (the streaming template hook); the public
    :meth:`start_transcription` runs the standard gating pipeline for them.
    """

    properties: ClassVar[BaseProperties]
    declared_capabilities: ClassVar[DeclaredCapabilities]
    #: Fail-loud placeholder, not a usable default. Compliance requires a
    #: plugin-owned :class:`DeclaredEngineMetadata` value; an engine that
    #: inherits this placeholder gets the typed declaration error from the
    #: artifact methods instead of a bare ``AttributeError``.
    declared_metadata: ClassVar[DeclaredEngineMetadata] = cast("DeclaredEngineMetadata", None)
    #: The engine's expected ``provider_params`` type, or ``None``.
    provider_params_type: ClassVar[type[ProviderParams] | None] = None
    #: The engine's init-config model type, or ``None`` when not declared.
    #: Declaring it makes the config JSON Schema (including the secret-field
    #: markers from :func:`~standard_asr.runtime.config.secret_field`) readable
    #: *without instantiation* -- the discovery path for
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
        contract: the engine-naming :class:`EngineContractError` on every
        path -- previously the detectable set was
        canonicalized per request inside
        :func:`~standard_asr.contract.language.effective_candidate_languages`, where an
        empty class-default tag surfaced as an uncontracted bare
        ``ValueError`` (an opaque HTTP 500) instead. It is a DECLARATION
        defect -- the engine author's spec violation, nothing any caller
        can fix -- so it is typed as an engine fault, not as
        :class:`ConfigError` (whose invalid-configuration meaning maps to
        caller-actionable surfaces such as the CLI's usage exit).

        Returns:
            A ``(selectable, detectable)`` pair of canonical tag sets.

        Raises:
            EngineContractError: If a declared selectable or detectable tag
                is malformed (empty or whitespace-only), naming the engine.
        """
        if self._language_sets_cache is not None:
            return self._language_sets_cache
        try:
            selectable = frozenset(
                _canonical_language(tag) for tag in self.properties.selectable_languages
            )
        except ValueError as exc:
            raise EngineContractError(
                f"selectable_languages {self.properties.selectable_languages!r} declared "
                f"by engine {self.properties.engine_id!r} contains a malformed tag: {exc}"
            ) from exc
        try:
            detectable = frozenset(
                _canonical_language(tag) for tag in self.properties.detectable_languages
            )
        except ValueError as exc:
            raise EngineContractError(
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
        """Warm up process-local engine state without transcribing.

        This optional, synchronous, idempotent hook moves process-local loading
        or initialization off the first transcription. Persistent
        inference-artifact acquisition belongs to :meth:`acquire_artifacts`.
        An engine can call that operation before continuing its warm-up when the
        native library cannot separate the two steps. The base implementation is
        a no-op, which the toolchain reports without fabricating an inference
        request.

        An override must remain a zero-argument synchronous method, never an
        ``async def``. A coroutine function would be called but never awaited
        and would silently report a false success. The compliance suite and the
        CLI reject that declaration.

        Returns:
            None.

        Raises:
            ArtifactUnavailableError: If required inference artifacts cannot
                support warm-up under the current policy.
            ArtifactAcquisitionError: If an allowed acquisition attempt fails.
        """

    def _artifact_declaration(self) -> ArtifactDeclaration:
        """Return the engine's authored artifact declaration.

        Returns:
            The engine's artifact declaration.

        Raises:
            EngineContractError: If the required metadata is absent or invalid.
        """
        metadata = cast("object", type(self).declared_metadata)
        if not isinstance(metadata, DeclaredEngineMetadata):
            raise EngineContractError(
                "Engine metadata must be a DeclaredEngineMetadata "
                "instance with an authored artifacts section."
            )
        # The outer isinstance proves the CLASS, not the section's type. The
        # section can still hold a raw mapping -- ``model_copy(update=...)``
        # stores whatever it is given -- and every caller below reads
        # ``.applicable`` off the result. Without this check that is
        # a bare AttributeError escaping a public lifecycle method, where this
        # method's own contract (and AR.2) says an invalid declaration is an
        # EngineContractError.
        declaration = cast("object", metadata.artifacts)
        if not isinstance(declaration, ArtifactDeclaration):
            raise EngineContractError(
                "declared_metadata.artifacts must be an "
                f"ArtifactDeclaration (got {safe_type_name(declaration)})."
            )
        # The isinstance proves the section's class, not its values: a
        # ``model_copy(update=...)`` declaration stores whatever it was given,
        # and every consumer of the result gates on plain truthiness -- a
        # truthy non-bool such as the string "false" would enable an
        # acquisition path here while ``canonical_json()`` publishes ``false``
        # for the same metadata (two verdicts for one declaration).
        # Re-validation applies the model's own construction semantics
        # (``revalidate_instances``): a value the constructor would coerce
        # behaves exactly as if it had been authored there, and a value it
        # would reject becomes the EngineContractError this method promises.
        try:
            return ArtifactDeclaration.model_validate(declaration)
        except ValidationError as exc:
            raise EngineContractError("declared_metadata.artifacts fails re-validation.") from exc

    def _resolve_artifact_mode(self, requested: Mode | None, *, applicable: bool) -> Mode:
        """Resolve an omitted artifact mode without inventing engine support.

        Args:
            requested: Caller-selected mode, or ``None``.
            applicable: Whether artifact acquisition can apply to this model.

        Returns:
            A concrete mode for gating and the report.

        Raises:
            ConfigError: If several non-batch modes would make an omitted mode
                ambiguous.
            ValueError: If an explicit mode has no inference domain on the
                engine.
        """
        capabilities = self.effective_capabilities
        if requested is not None:
            if getattr(capabilities, requested) is None:
                raise ValueError(
                    f"ArtifactContext.mode {requested!r} is not supported by this engine."
                )
            return requested
        if capabilities.batch is not None:
            return "batch"
        available: list[Mode] = []
        if capabilities.streaming is not None:
            available.append("streaming")
        if len(available) == 1:
            return available[0]
        if not available and not applicable:
            return "batch"
        raise ConfigError(
            "ArtifactContext.mode is required because this engine has no "
            "unambiguous default inference mode."
        )

    def _resolve_artifact_context(
        self,
        context: ArtifactContext | None,
        *,
        applicable: bool,
    ) -> tuple[ArtifactContext, list[Diagnostic]]:
        """Resolve mode, gate params, and resolve language for introspection.

        Args:
            context: Caller context, or ``None`` for defaults.
            applicable: Whether artifact acquisition can apply to this model.

        Returns:
            The resolved context and ordered standard-layer diagnostics.

        Raises:
            ConfigError: On invalid engine configuration or ambiguous mode.
            EngineContractError: On malformed declarations.
            InvalidProviderParamError: On wrong-engine provider params.
            ValueError: On malformed request language data or an explicit mode
                the engine does not support.
        """
        requested_context = context or ArtifactContext()
        mode = self._resolve_artifact_mode(requested_context.mode, applicable=applicable)
        request = requested_context.params
        self._validate_language_config()
        gated, gate_diagnostics = gate_params(
            request,
            self.effective_capabilities,
            mode,
            strict=False,
            expected_provider_type=self.provider_params_type,
        )
        gated, language_diagnostics = self._resolve_language_axis(
            gated,
            mode,
            requested_language=request.language,
            strict=False,
        )
        return (
            ArtifactContext(mode=mode, params=gated),
            [*gate_diagnostics, *language_diagnostics],
        )

    @staticmethod
    def _artifact_hook_output(
        value: object,
    ) -> tuple[bool, tuple[ArtifactRequirement, ...], tuple[Diagnostic, ...]]:
        """Validate the protected status hook's exact return shape.

        Args:
            value: Hook return value.

        Returns:
            The typed applicability, requirements, and diagnostics tuple, with
            the diagnostics re-validated.

        Raises:
            EngineContractError: If any part has the wrong type or a
                diagnostic fails re-validation.
        """
        require_sync_result(value, "_artifact_requirements()", expected_type=tuple)
        if type(value) is not tuple:
            raise EngineContractError(
                "_artifact_requirements() must return a three-item tuple: "
                "(applicable, requirements, diagnostics)."
            )
        items = cast("tuple[object, ...]", value)
        if len(items) != 3:
            raise EngineContractError(
                "_artifact_requirements() must return a three-item tuple: "
                "(applicable, requirements, diagnostics)."
            )
        applicable, requirements, diagnostics = items
        if type(applicable) is not bool:
            raise EngineContractError("_artifact_requirements() applicable must be an exact bool.")
        if type(requirements) is not tuple:
            raise EngineContractError(
                "_artifact_requirements() requirements must be a tuple of "
                "ArtifactRequirement values."
            )
        requirement_items = cast("tuple[object, ...]", requirements)
        if any(type(requirement) is not ArtifactRequirement for requirement in requirement_items):
            raise EngineContractError(
                "_artifact_requirements() requirements must be a tuple of "
                "ArtifactRequirement values."
            )
        if type(diagnostics) is not tuple:
            raise EngineContractError(
                "_artifact_requirements() diagnostics must be a tuple of Diagnostic values."
            )
        diagnostic_items = cast("tuple[object, ...]", diagnostics)
        if any(type(diagnostic) is not Diagnostic for diagnostic in diagnostic_items):
            raise EngineContractError(
                "_artifact_requirements() diagnostics must be a tuple of Diagnostic values."
            )
        # The exact-class checks above prove the SHAPE, not the values. The
        # requirements need no more: they re-validate when the report nests
        # them (``revalidate_instances``). Plain ``Diagnostic`` deliberately
        # keeps the framework-wide 1.0 config, so a ``model_copy(update=...)``
        # diagnostic would cross the report boundary verbatim: an invalid
        # level token would publish through every projection, a NaN
        # ``provided`` would silently become JSON ``null``, and a value with
        # no JSON form would crash the wire projection long after this seam --
        # where AR.2 makes invalid hook output an EngineContractError.
        # Same dump/validate round-trip as the progress observer;
        # ``warnings=False`` because the python-mode dump of a smuggled
        # non-JSON value warns before re-validation rejects it.
        validated_diagnostics: list[Diagnostic] = []
        for diagnostic in cast("tuple[Diagnostic, ...]", diagnostics):
            try:
                validated_diagnostics.append(
                    Diagnostic.model_validate(diagnostic.model_dump(mode="python", warnings=False))
                )
            except ValidationError as exc:
                raise EngineContractError(
                    "_artifact_requirements() produced an invalid Diagnostic."
                ) from exc
        return (
            applicable,
            cast("tuple[ArtifactRequirement, ...]", requirements),
            tuple(validated_diagnostics),
        )

    @staticmethod
    def _artifact_report_matches_declaration(
        report: ArtifactReport,
        declaration: ArtifactDeclaration,
    ) -> None:
        """Enforce that dynamic status only narrows static artifact metadata.

        Args:
            report: Validated dynamic report.
            declaration: Static artifact declaration.

        Returns:
            None.

        Raises:
            EngineContractError: If the report widens a static fact.
        """
        if report.applicable and not declaration.applicable:
            raise EngineContractError(
                "Artifact status reports applicability that declared metadata does not allow."
            )
        if any(requirement.can_acquire_now for requirement in report.requirements) and not (
            declaration.supports_explicit_acquisition
        ):
            raise EngineContractError(
                "Artifact status reports explicit acquisition that declared metadata "
                "does not allow."
            )
        if (
            any(requirement.may_acquire_during_inference for requirement in report.requirements)
            and not declaration.may_acquire_during_inference
        ):
            raise EngineContractError(
                "Artifact status reports inference acquisition that declared metadata "
                "does not allow."
            )

    @final
    def artifact_status(
        self,
        context: ArtifactContext | None = None,
    ) -> ArtifactReport:
        """Inspect inference-artifact readiness without side effects.

        Args:
            context: Optional request context.

        Returns:
            A point-in-time artifact report.

        Raises:
            ProtocolCompatibilityError: If the engine declares an unsupported protocol line.
            InvalidProviderParamError: On wrong-engine provider params.
            ValueError: On malformed request language data or an explicit mode
                the engine does not support.
            ConfigError: On invalid configuration or ambiguous mode.
            EngineContractError: On invalid declarations or hook results.
            ArtifactStatusError: On unexpected native inspection failure.
        """
        require_engine_protocol(self)
        declaration = self._artifact_declaration()
        resolved_context, standard_diagnostics = self._resolve_artifact_context(
            context,
            applicable=declaration.applicable,
        )
        mode = cast("Mode", resolved_context.mode)
        if not declaration.applicable:
            return ArtifactReport.from_requirements(
                mode=mode,
                applicable=False,
                diagnostics=standard_diagnostics,
            )
        if type(self)._artifact_requirements is EngineBase._artifact_requirements:
            raise EngineContractError(
                "An engine declaring an applicable artifact lifecycle must "
                "override _artifact_requirements()."
            )
        try:
            raw = self._artifact_requirements(resolved_context)
        except (ArtifactStatusError, ConfigError, EngineContractError, InvalidProviderParamError):
            raise
        except Exception as exc:
            raise ArtifactStatusError(
                f"Artifact status inspection failed inside the engine hook ({safe_type_name(exc)})."
            ) from exc
        applicable, requirements, engine_diagnostics = self._artifact_hook_output(raw)
        try:
            report = ArtifactReport.from_requirements(
                mode=mode,
                applicable=applicable,
                requirements=requirements,
                diagnostics=[*standard_diagnostics, *engine_diagnostics],
            )
        except ValidationError as exc:
            raise EngineContractError(
                "_artifact_requirements() produced an invalid artifact report."
            ) from exc
        self._artifact_report_matches_declaration(report, declaration)
        return report

    def _artifact_requirements(
        self,
        context: ArtifactContext,
    ) -> tuple[bool, tuple[ArtifactRequirement, ...], tuple[Diagnostic, ...]]:
        """Inspect native artifact requirements (override point).

        An engine whose declaration says acquisition applies MUST override this;
        the base body never runs for one, because :meth:`artifact_status`
        refuses an applicable declaration that left the hook in place.

        Args:
            context: Resolved, best-effort-gated request context.

        Returns:
            ``(applicable, requirements, diagnostics)``. ``applicable`` is this
            configured instance's dynamic answer to "does an inference-artifact
            lifecycle apply here", and it MUST only narrow the static
            declaration -- a report cannot claim applicability the class-level
            metadata withholds. ``requirements`` is the logical dependency
            closure for the resolved context, one entry per requirement, and
            ``diagnostics`` carries the non-fatal notes the inspection made.
            Aggregate readiness is not returned: the core derives it. The base
            implementation returns ``(False, (), ())``, the no-artifact shape.
        """
        return False, (), ()

    @staticmethod
    def _raise_artifact_blockers(
        requirements: tuple[ArtifactRequirement, ...],
        report: ArtifactReport,
    ) -> None:
        """Raise the portable error for blocked required requirements.

        Args:
            requirements: Blocked required requirements.
            report: Latest report retained on the error.

        Raises:
            ArtifactAcquisitionError: Always when ``requirements`` is nonempty.
        """
        if not requirements:
            return
        reason = _artifact_blocker_reason(requirements)
        raise ArtifactAcquisitionError(
            "Required inference artifacts cannot be acquired under the current policy.",
            reason=reason,
            report=report,
            required_actions=_artifact_required_actions(requirements),
        )

    @final
    def acquire_artifacts(
        self,
        context: ArtifactContext | None = None,
        *,
        refresh: bool = False,
        progress: ArtifactProgressCallback | None = None,
    ) -> ArtifactReport:
        """Acquire inference artifacts explicitly and return fresh status.

        Args:
            context: Optional request context.
            refresh: Whether to re-resolve unblocked mutable source references.
            progress: Optional synchronous progress observer.

        Returns:
            A newly inspected artifact report.

        Raises:
            ProtocolCompatibilityError: If the engine declares an unsupported protocol line.
            ArtifactStatusError: If preflight or final status inspection fails.
            ArtifactAcquisitionError: If acquisition is blocked or fails.
            ArtifactProgressCallbackError: After successful acquisition and
                final status, if the observer failed.
            EngineContractError: On invalid declarations, progress, or hook
                behavior.
            InvalidProviderParamError: On wrong-engine provider params.
            ConfigError: On invalid engine configuration.
            ValueError: On malformed request language data.
        """
        # Independent line gate: the preflight below dispatches through the
        # virtual `self.artifact_status`, so an override of that PUBLIC
        # member would otherwise leave this entry without the gate.
        require_engine_protocol(self)
        preflight = self.artifact_status(context)
        declaration = self._artifact_declaration()
        if (
            declaration.supports_explicit_acquisition
            and type(self)._acquire_artifacts is EngineBase._acquire_artifacts
        ):
            raise EngineContractError(
                "An engine declaring explicit artifact acquisition must override "
                "_acquire_artifacts()."
            )
        if not preflight.applicable:
            return preflight
        resolved_context, _ = self._resolve_artifact_context(
            context,
            applicable=declaration.applicable,
        )
        mutable = tuple(
            requirement for requirement in preflight.requirements if requirement.source_is_mutable
        )
        if refresh and mutable and not allow_downloads():
            raise ArtifactAcquisitionError(
                "Mutable artifact sources cannot be refreshed while downloads are disabled.",
                reason="downloads_disabled",
                report=preflight,
                required_actions=_artifact_required_actions(mutable),
            )

        runnable = tuple(
            requirement
            for requirement in preflight.requirements
            if requirement.state != ARTIFACT_READY and requirement.can_acquire_now
        )
        refresh_targets = (
            tuple(requirement for requirement in mutable if requirement.acquisition_blocker is None)
            if refresh
            else ()
        )
        targets_by_id = {requirement.artifact_id: requirement for requirement in runnable}
        for requirement in refresh_targets:
            targets_by_id.setdefault(requirement.artifact_id, requirement)
        targets = tuple(targets_by_id.values())

        blocked_required = tuple(
            requirement
            for requirement in preflight.requirements
            if requirement.required_for_inference
            and requirement.state != ARTIFACT_READY
            and not requirement.can_acquire_now
        )
        if not targets:
            self._raise_artifact_blockers(blocked_required, preflight)
            return preflight
        if not declaration.supports_explicit_acquisition:
            raise ArtifactAcquisitionError(
                "This configured engine cannot explicitly acquire or refresh artifacts.",
                reason="unsupported",
                report=preflight,
            )
        observer = _ArtifactProgressObserver(progress) if progress is not None else None
        hook_progress: ArtifactProgressCallback | None = observer
        if observer is not None:
            observer(ArtifactProgress(phase=ARTIFACT_PROGRESS_RESOLVING))
        try:
            result = self._acquire_artifacts(
                resolved_context,
                targets,
                refresh,
                hook_progress,
            )
            require_sync_result(result, "_acquire_artifacts()", expected_type=type(None))
        except ArtifactAcquisitionError as exc:
            if exc.report is None:
                # AR.5: the error preserves the full report. A hook-level
                # helper may raise without one (no preflight in its scope);
                # the template has it, so backfill rather than surface a
                # structured error stripped of its status context
                # (round-16 review).
                exc.report = preflight
            raise
        except EngineContractError:
            raise
        except Exception as exc:
            raise ArtifactAcquisitionError(
                f"Artifact acquisition failed inside the engine hook ({safe_type_name(exc)}).",
                reason="failed",
                report=preflight,
            ) from exc
        if observer is not None:
            observer(ArtifactProgress(phase=ARTIFACT_PROGRESS_FINALIZING))
        try:
            final_report = self.artifact_status(context)
        except ArtifactStatusError as exc:
            raise ArtifactStatusError(
                "Artifact acquisition completed, but final status inspection failed."
            ) from exc

        final_by_id = {
            requirement.artifact_id: requirement for requirement in final_report.requirements
        }
        attempted_target_ids = {requirement.artifact_id for requirement in targets}
        missing_target_ids = tuple(
            artifact_id for artifact_id in attempted_target_ids if artifact_id not in final_by_id
        )
        if missing_target_ids:
            raise EngineContractError(
                "Final artifact status omitted a target from the configured artifact lifecycle."
            )
        attempted_failures = tuple(
            artifact_id
            for artifact_id in attempted_target_ids
            if final_by_id[artifact_id].state != ARTIFACT_READY
        )
        if attempted_failures:
            raise ArtifactAcquisitionError(
                "Artifact acquisition returned without making an attempted requirement ready.",
                reason="failed",
                report=final_report,
            )
        final_blocked_required = tuple(
            requirement
            for requirement in final_report.requirements
            if requirement.required_for_inference
            and requirement.state != ARTIFACT_READY
            and not requirement.can_acquire_now
        )
        self._raise_artifact_blockers(final_blocked_required, final_report)
        if any(
            requirement.required_for_inference and requirement.state != ARTIFACT_READY
            for requirement in final_report.requirements
        ):
            raise ArtifactAcquisitionError(
                "Required inference artifacts remain unavailable after acquisition.",
                reason="failed",
                report=final_report,
            )
        if observer is not None and observer.contract_error is not None:
            raise observer.contract_error
        if observer is not None and observer.callback_error is not None:
            raise ArtifactProgressCallbackError(
                "Artifact acquisition succeeded, but its progress callback failed.",
                report=final_report,
            ) from observer.callback_error
        return final_report

    def _acquire_artifacts(
        self,
        context: ArtifactContext,
        requirements: tuple[ArtifactRequirement, ...],
        refresh: bool,
        progress: ArtifactProgressCallback | None,
    ) -> None:
        """Perform native artifact acquisition (override point).

        Args:
            context: Resolved artifact context.
            requirements: Runnable acquisition and refresh targets.
            refresh: Whether mutable targets must be re-resolved.
            progress: Validating serialized progress observer, if requested.

        Returns:
            None.
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
        """Whether the SSRF policy is relaxed for private-address ``AudioUrl``.

        Sourced from the engine's init config (``BaseConfig.allow_private_urls``).
        It is an init-level deployment switch -- a trust decision about
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

    @final
    def transcribe(
        self, audio: AudioInputLike, params: RuntimeParams | None = None
    ) -> TranscriptionResult:
        """Transcribe a complete audio input (template method).

        Runs the standard pipeline, *fail-fast first*: validate the language
        config -> gate parameters (provider_params + capability gating, which
        needs no audio) -> resolve & validate the effective language axis ->
        coerce -> negotiate -> convert/resample -> call the engine ->
        synthesize missing segment speakers from word speakers (the pinned
        synthesis rule) -> attach diagnostics.

        Parameter validation runs *before* the (potentially expensive) audio
        decode/resample so a swapped-engine ``provider_params`` bug or an
        unsupported parameter is rejected before any audio is touched (fail fast
        on provider_params first).

        Args:
            audio: The audio to transcribe.
            params: Per-request runtime parameters.

        Returns:
            The transcription result with gating / language / conversion
            diagnostics attached.

        Raises:
            ProtocolCompatibilityError: If the engine's declared protocol
                line is not supported by this core (checked before any work).
            ConfigError: If the engine's ``default_language`` VALUE is
                malformed or not in ``selectable_languages`` -- fixable by
                whoever supplies the config.
            EngineContractError: If ``properties`` is missing or untyped
                (the shared gate fails closed before any work), or on an
                engine-declaration defect -- a
                declared language axis with no ``default_language`` (IC.6),
                or a malformed declared selectable/detectable tag.
            IncompatibleAudioInputError: If no conversion path exists.
            UnsafeAudioUrlError: If an ``AudioUrl`` fails the SSRF policy
                (non-HTTPS, or a private/loopback/link-local target).
            AudioProcessingError: On an audio failure surfaced by the conversion
                pipeline -- a decode failure, an over-``max_file_size`` payload,
                or (in strict mode) a bare array with no sample rate.
            UnsupportedFeatureError: In strict mode, on an unsupported parameter,
                a requested ``language`` not selectable by the engine, or a
                valid-but-unreachable candidate list (a non-detectable candidate
                or one over the declared ``max``).
            InvalidProviderParamError: On wrong provider params.
            ValueError: On a malformed candidate tag or one containing ``auto``
                -- a caller code bug, raised **always** (independent of
                strict/best_effort).
            ArtifactUnavailableError: When required inference artifacts cannot
                support recognition under the current policy.
            ArtifactAcquisitionError: When an allowed implicit acquisition
                attempt fails before recognition.
            TranscriptionError: On an engine-execution failure inside
                :meth:`_transcribe` -- including a pydantic ``ValidationError``
                escaping it (an invalid result construction is an engine
                fault; the template wraps it here so it can never masquerade
                as a client-input validation error).
        """
        # The full engine gate runs before any work: AR.1 makes each 0.MINOR
        # generation potentially contract-breaking, so running a
        # mismatched-line engine could return a structurally valid but
        # semantically drifted transcript -- a silent wrong result -- and an
        # engine whose properties cannot even be typed is less knowable
        # still. A parse and tuple comparison is negligible next to
        # inference.
        require_engine_protocol(self)
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
        try:
            result = self._transcribe(prepared, gated)
            # The boundary belongs HERE, at the author hook, not only on the
            # public method a consumer sees: the template consumes this value
            # immediately (speaker synthesis, then .diagnostics), so an
            # `async def` _transcribe surfaced as a secondary AttributeError
            # on a coroutine -- plus a never-awaited warning -- long before
            # any consumer's own check could classify it. A public
            # `transcribe()` being synchronous says nothing about the hook it
            # delegates to, so no surface-level modality check can see this.
            require_sync_result(result, "_transcribe()", expected_type=TranscriptionResult)
        except ValidationError as exc:
            # A pydantic ValidationError escaping _transcribe is an ENGINE
            # fault (params were validated before this point; the usual cause
            # is the engine constructing a TranscriptionResult/Segment the
            # model rejects -- for example, a field removed from the contract, or an
            # invalid timestamp). Without this wrap it masquerades as a
            # client-input validation error: the server's ValidationError
            # clause turned it into a 422 blaming the request's options.
            # Wrapping enforces the spec's portable batch error contract
            # (engine-execution failure -> TranscriptionError, original
            # exception preserved as __cause__) at the one template seam that
            # can see it.
            raise TranscriptionError(
                "Engine produced an invalid result (or raised an unwrapped "
                "validation error) inside _transcribe -- an engine/plugin "
                "fault, not a request error. See the chained ValidationError "
                "for the offending fields (for example, a field the result model no "
                "longer accepts).",
                hint=(
                    "Report this to the engine plugin's author; a core/plugin "
                    "version mismatch (the plugin building a result with "
                    "removed or invalid fields) is the usual cause."
                ),
            ) from exc
        # Standard-layer diarization synthesis: the streaming
        # reducer applies the same shared rule, so batch and streaming yield
        # the same Segment.speaker for the same engine output.
        result = _synthesize_result_speakers(result)
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
        :func:`~standard_asr.audio.conversion.execute_plan`, shared by
        :meth:`transcribe` and the whole-input :meth:`start_transcription` path so
        both honor identical negotiation against the engine's declared audio
        properties. A new conversion parameter is then wired in exactly one place
        and can never silently diverge between the batch and streaming paths.

        Args:
            audio: The caller's audio input (path, bytes, URL, array, and so on).

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
        """Enforce the ``default_language`` totality invariant.

        When the engine exposes a language axis, ``default_language`` MUST be set
        and MUST be a member of ``selectable_languages``; otherwise the
        fall-back-to-``default_language`` resolution step would yield an
        undefined result. This
        runs in the standard layer so a forgetful engine fails loudly instead of
        silently transcribing in the wrong language.

        Raises:
            ConfigError: If ``default_language`` carries an invalid VALUE --
                malformed (empty/whitespace) or not in
                ``selectable_languages``. The value came from the
                configuration, so whoever supplies the config can fix it
                (the CLI's usage exit; the reference server's clients
                cannot supply config, so it scrubs to 500 there).
            EngineContractError: If the language axis is exposed but
                ``default_language`` is unset -- IC.6 obliges the ENGINE to
                make it required or defaulted when it declares the axis, so
                a ``None`` here is the engine's declaration inconsistency,
                not a fixable configuration value -- or if a declared
                selectable/detectable tag is itself malformed.
        """
        if not self.properties.has_language_axis:
            return
        default = getattr(self.config, "default_language", None)
        if default is None:
            raise EngineContractError(
                f"Engine {self.properties.engine_id!r} exposes a language axis "
                "(selectable_languages is non-empty) so its config MUST set "
                "default_language (IC.6: required or defaulted when the axis "
                "is declared -- an engine-declaration defect, not a value the "
                "caller can supply)."
            )
        # Canonicalize BOTH sides: BCP-47 membership is case-insensitive, and
        # either default_language or the declared sets may be a non-canonical
        # class-level default, so a raw ``en-us`` must still match a canonical
        # ``en-US`` instead of spuriously failing the totality check and blocking
        # the engine.
        # Canonicalization raises ValueError on an empty/whitespace tag; this
        # method promises ConfigError, so wrap it naming the malformed value (a
        # language tag is not a secret -- echoing it is safe and actionable).
        try:
            canonical_default = _canonical_language(default)
        except ValueError as exc:
            raise ConfigError(
                f"default_language {default!r} is malformed for engine "
                f"{self.properties.engine_id!r}: {exc}."
            ) from exc
        selectable, _ = self._canonical_language_sets()
        if canonical_default not in selectable:
            raise ConfigError(
                f"default_language {default!r} is not in selectable_languages "
                f"{self.properties.selectable_languages!r} "
                f"(engine {self.properties.engine_id!r})."
            )

    def _resolve_language_axis(
        self,
        params: RuntimeParams,
        mode: Mode,
        *,
        requested_language: str | None = None,
        strict: bool | None = None,
    ) -> tuple[RuntimeParams, list[Diagnostic]]:
        """Resolve and validate the effective language axis.

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
            strict: Unsupported-language policy override. ``None`` uses the
                engine config. Artifact introspection passes ``False`` so the
                entire read-only path converges through diagnostics.

        Returns:
            A ``(params, diagnostics)`` pair containing the effective runtime
            parameters plus diagnostics produced during language resolution.

        Raises:
            UnsupportedFeatureError: In strict mode, if the resolved language is
                not selectable by this engine, or on a valid-but-unreachable
                candidate list (non-detectable / over-``max``).
            ValueError: On a malformed or ``"auto"`` candidate entry (always,
                independent of strict/best_effort).
        """
        if not self.properties.has_language_axis:
            return params, []
        strict_policy = self._strict if strict is None else strict
        caps = self.effective_capabilities
        runtime_override_supported = caps.supports(f"{mode}.language.runtime_override")
        # default_language is non-None here: _validate_language_config (always run
        # before this) enforces it whenever has_language_axis is True. Canonicalize
        # it up front so the best-effort fallback below (and the diagnostic it
        # emits) carry a canonical tag, never a raw class-level default (``en-us``).
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
                    code=DIAG_LANGUAGE_FELL_BACK,
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
        # case-insensitively. Membership uses RFC 4647 lookup so a
        # region/script refinement of a selectable primary subtag (for example, ``en-US``
        # against ``en``) is accepted and handed to the engine to reduce -- engines
        # need not enumerate variants.
        selectable, detectable = self._canonical_language_sets()
        # eff_lang is non-None here: this method only runs when has_language_axis
        # is True, and effective_language then returns default_language (the
        # totality invariant guarantees it is set, enforced by
        # _validate_language_config above), or
        # the request override -- both non-None. `auto` has no subtags, so
        # _selectable_match treats it as an exact membership test (matched ==
        # "auto" when selectable, else None), preserving the prior
        # auto-selectability behavior; the refinement branch below applies only to
        # real BCP-47 tags.
        assert eff_lang is not None
        matched = _selectable_match(eff_lang, selectable)
        if matched is None:
            if strict_policy:
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
                    code=DIAG_LANGUAGE_NOT_SELECTABLE,
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
            # Accepted as an RFC 4647 refinement of a selectable primary subtag:
            # the engine receives the full requested tag and
            # reduces it internally. Surface an informational diagnostic so the
            # caller can see the tag was matched by reduction rather than exact
            # membership (no value is changed).
            diagnostics.append(
                Diagnostic(
                    level="info",
                    code=DIAG_LANGUAGE_REFINEMENT_ACCEPTED,
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
            strict=strict_policy,
            mode=mode,
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

    @final
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
            EngineContractError: If :meth:`transcribe` (overridden by the
                subclass) violated the synchronous protocol contract --
                returned an awaitable (``async def``) or a value that is not
                a :class:`~standard_asr.contract.results.TranscriptionResult`
                -- or propagated a declaration defect from :meth:`transcribe`
                (a missing IC.6 default, a malformed declared tag).
            Exception: The same exception set as :meth:`transcribe` because it
                runs that method. This includes artifact availability and
                acquisition errors without wrapping them as
                ``TranscriptionError``.
        """
        # Same line gate as transcribe(): `self.transcribe` below is virtual
        # dispatch, so a subclass that overrode the PUBLIC method would
        # otherwise carry a mismatched line straight past the template.
        require_engine_protocol(self)
        result = await asyncio.to_thread(self.transcribe, audio, params)
        # This bridge is a real CONSUMER of the synchronous protocol member,
        # exactly like the CLI and the server's REST path: `self.transcribe`
        # is virtual dispatch into subclass code, and the template's own
        # `_transcribe` boundary above says nothing about a subclass that
        # overrode the PUBLIC method (an `async def transcribe` hands
        # to_thread a coroutine it never drives; a wrong-typed override
        # passes straight through). Unchecked, the caller received that
        # coroutine as the "result" -- the wrong type AND a never-awaited
        # leak -- so the one shared boundary runs here too.
        require_sync_result(result, "transcribe()", expected_type=TranscriptionResult)
        return result

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
            ArtifactUnavailableError: When the engine knows before recognition
                that required inference artifacts cannot resolve under the
                current policy. Do not wrap this error.
            ArtifactAcquisitionError: When an allowed implicit acquisition was
                attempted and failed before recognition. Do not wrap this error.
            TranscriptionError: On every other engine-execution failure, such as
                model inference, a network inference call, or an SDK error.
                Implementations must preserve the native exception as the cause
                so applications can catch one portable execution-error type.
        """
        raise NotImplementedError  # pragma: no cover

    @staticmethod
    def ensure_stream_inputs_exclusive(
        audio_format: AudioFormat | None, audio: AudioInputLike | None
    ) -> None:
        """Enforce the ``audio_format`` / ``audio`` mutual-exclusion.

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
                "exactly one."
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
        ("unconstrained") the encoding cannot be validated and the
        check is skipped -- the engine is then trusted to accept any encoding
        (typically a self-managed-wire-format engine). The compliance suite
        emits a warning for a ``streaming_input`` engine that leaves
        ``wire_encodings`` unset, since that skip is where a forgotten
        declaration would let a non-PCM frame be misframed.

        Wire **sample rate**: the standard's v1 implementation note is explicit
        that v1 does **NOT** resample streaming bare frames in the standard layer
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
        # Delegates to the module-level pure rule so the compliance suite can
        # validate the identical semantics for structural (non-EngineBase)
        # engines without reaching for this method.
        ensure_wire_format_supported(self.properties, audio_format)

    def recommended_wire_format(self) -> AudioFormat | None:
        """Return a minimal wire :class:`AudioFormat` to open a streaming session.

        Single source of truth for the legal bare-frame wire format the standard
        layer uses when it must open a ``streaming_input`` session but has no
        application-chosen format -- the CLI sync-bridge runner and the streaming
        gating probe both rely on it. They previously derived one independently and
        disagreed (which sample-rate source to use, and what to do with no declared
        ``wire_encodings``); this unifies them. The format is built from the
        engine's own Properties so :meth:`ensure_stream_format_supported` accepts
        it (the compliance suite asserts that round-trip):

        * ``sample_rate`` = ``required_input_sample_rate`` when the engine
          hard-requires one, else ``native_sample_rate`` (the reachability
          invariant guarantees the native rate is accepted).
        * ``encoding`` = the first declared ``wire_encodings`` entry, else the
          canonical ``pcm_s16le`` (used only when ``wire_encodings`` is
          unconstrained, where the engine accepts any encoding).
        * ``channels`` = 1 (v1 streaming wire is mono-only).

        The derivation is deliberately capability-blind (Properties only):
        whether a bare-frame session can be opened at all is decided by the
        ``streaming_input`` gate in :meth:`start_transcription`, not here --
        so the recommendation stays a pure, class-level static fact and the
        compliance round-trip (format ⊆ ``ensure_stream_format_supported``)
        holds for every engine, streaming or not.

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

    @final
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

        Because gating now runs here, ``provider_params``
        swap-safety is enforced on the streaming path too: a swapped-engine
        ``provider_params`` type-mismatch always raises
        :class:`~standard_asr.contract.exceptions.InvalidProviderParamError` (no longer
        undefined behavior), and an unsupported standard parameter is rejected
        (strict) or dropped + diagnosed (best_effort) exactly as for batch.

        The streaming input/output capability axis is checked before the hook
        override defense, so an engine that implements the hook but does not
        declare the requested session mode fails on the missing capability
        rather than reaching parameter or audio gating. The hook override defense
        still runs before parameter gating, so a batch-only engine reports "does
        not support streaming" rather than a confusing parameter error -- while
        still running the input mutual-exclusion guard first, exactly as before.

        Streaming param freeze: the already-gated, frozen
        :class:`~standard_asr.contract.params.RuntimeParams` is handed to the hook
        as ``gated_params``; the engine uses that for the whole session and MUST
        NOT re-accept raw params mid-stream.

        Args:
            audio_format: Wire format for incremental PCM frames.
            params: Per-request runtime parameters.
            audio: A complete audio input for whole-input streaming output.
            deadlines: Application overrides for the session's termination
                deadlines. Applied by this template *after* the
                engine hook constructed the session, so explicitly set fields
                always win over the engine's construction-time choices --
                precedence: application explicit > engine choice > standard
                default. Unset fields are left untouched.

        Returns:
            A streaming session with gating / language diagnostics attached.

        Raises:
            ProtocolCompatibilityError: If the engine's declared protocol
                line is not supported by this core (checked before any work).
            ValueError: If both ``audio_format`` and ``audio`` are provided, or
                on a malformed/``auto`` candidate-language entry (a caller code
                bug; always raises, independent of strict/best_effort).
            ConfigError: If the engine's ``default_language`` VALUE is
                malformed or not in ``selectable_languages``.
            EngineContractError: If ``properties`` is missing or untyped
                (the shared gate fails closed before any work), or on an
                engine-declaration defect -- a
                declared language axis with no ``default_language`` (IC.6),
                or a malformed declared selectable/detectable tag.
            UnsupportedFeatureError: When the requested streaming input/output
                axis is unsupported, when streaming is unsupported, when the wire
                format is unreachable, or, in strict mode, on an unsupported
                parameter or a valid-but-unreachable candidate list
                (non-detectable / over-``max``).
            IncompatibleAudioInputError: If no conversion path exists for a
                whole-input streaming ``audio`` value.
            UnsafeAudioUrlError: If a whole-input ``AudioUrl`` fails the SSRF
                policy.
            AudioProcessingError: On a decode / size / missing-sample-rate
                failure for a whole-input ``audio`` value.
            InvalidProviderParamError: On wrong ``provider_params`` (swap-safety).
            ArtifactUnavailableError: When required inference artifacts cannot
                support session establishment under the current policy.
            ArtifactAcquisitionError: When an allowed implicit acquisition
                attempt fails during session establishment.
            TranscriptionError: When a pydantic ``ValidationError`` escapes the
                engine's ``_start_transcription`` hook (an invalid model
                construction is an engine fault; wrapped here so it can never
                masquerade as a client-input validation error).
        """
        # Same full engine gate as transcribe(), for the same reason: a
        # mismatched-line engine's streaming semantics are unknowable.
        require_engine_protocol(self)
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
        # for an engine that self-manages its wire format, which is
        # the streaming-input axis. Gate it on the same 'streaming_input' capability
        # as the audio_format path; otherwise a streaming_output-only engine that
        # implements the hook would hand back an incremental session it cannot feed
        # (audio_format=None, prepared_audio=None) -- undefined behavior instead of
        # the fail-closed UnsupportedFeatureError. Placed AFTER the hook-override
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
        try:
            session = self._start_transcription(
                gated_params=gated, audio_format=audio_format, prepared_audio=prepared
            )
            # Same seam as the batch hook: the template calls private methods
            # on this value on the very next lines, so a coroutine or a
            # wrong-typed object had to be classified here or not at all.
            require_sync_result(
                session, "_start_transcription()", expected_type=TranscriptionSession
            )
        except ValidationError as exc:
            # Same engine-fault seam as the batch wrap on _transcribe: params
            # were validated before this point, so a pydantic ValidationError
            # escaping the hook is the ENGINE constructing an invalid model.
            # Unwrapped it is a ValueError subclass and every transport
            # misattributes it as a client mistake (WS "unsupported" echoing
            # unsanitized pydantic detail; CLI usage-error exit 2).
            raise TranscriptionError(
                "Engine raised an unwrapped validation error inside its "
                "_start_transcription hook -- an engine/plugin fault, not a "
                "request error. See the chained ValidationError for the "
                "offending fields.",
                hint=(
                    "Report this to the engine plugin's author; a core/plugin "
                    "version mismatch is the usual cause."
                ),
            ) from exc
        # Friend API: validate the reserved-attribute guard now, before the base
        # seeds diagnostics / applies deadline overrides below -- so the check sees
        # the pristine post-__init__ snapshot and a subclass that clobbered base
        # state (for example, its own self._buffer) fails loudly here, not as a
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
        :class:`~standard_asr.runtime.streaming.TranscriptionSession`. It is invoked by
        the :meth:`start_transcription` template *after* the standard streaming
        pipeline (input exclusion, language config, wire-format validation,
        parameter gating, language resolution) has run, so the engine receives
        already-gated, frozen parameters and need not reimplement any gating.

        This is intentionally *not* abstract: batch-only engines inherit the
        default, which raises so a stray streaming call fails loudly.

        Args:
            gated_params: The gated, frozen runtime parameters (frozen for the
                whole session).
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


__all__ = [
    "EngineBase",
    "StandardASR",
    "ensure_wire_format_supported",
    "require_engine_protocol",
]
