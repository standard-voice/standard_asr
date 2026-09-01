# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The Standard ASR exception hierarchy (the error half of the public contract).

Every exception an application may catch when invoking a compliant engine lives
here and is re-exported from the package top level (``standard_asr``), so the
error contract is reachable from the same public surface as the types it
accompanies -- ``except standard_asr.UnsupportedFeatureError`` works without
reaching into this submodule. The hierarchy roots at :class:`StandardASRError`;
the more specific classes let an application distinguish a recoverable user
mistake (bad params, unsupported feature) from an engine/runtime fault.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Final, Literal

if TYPE_CHECKING:
    from standard_asr.contract.artifacts import ArtifactAction, ArtifactReport


class StandardASRError(Exception):
    """Base class for every domain error the Standard ASR runtime raises.

    It does not cover data-model construction, which is pydantic's: a malformed
    field on :class:`~standard_asr.contract.params.RuntimeParams`, on a result
    model, or on a config raises ``pydantic.ValidationError``. That is a
    ``ValueError``, not a ``StandardASRError``. Plain caller misuse likewise
    raises a built-in: ``ValueError`` for a bad value (two mutually exclusive
    arguments), ``TypeError`` for a wrong type (an unsupported input type).

    ``except (StandardASRError, ValueError)`` catches the domain errors and the
    value mistakes. A ``TypeError`` stays outside both on purpose -- a wrong
    input type is a code bug to fix, not a state to handle.
    """

    pass


class StructuredError(StandardASRError):
    """Base for errors that carry machine-readable context beside the message.

    Gives the error half of the contract the same "don't make me parse the
    message" property the diagnostics have: an application can read ``.param``
    (the offending field/parameter), ``.hint`` (actionable guidance), and
    ``.details`` (structured context -- for example, the sanitized pydantic error entries
    behind a wrapped config failure) without scraping ``str(exc)``. Every context
    field is optional and keyword-only, so ``Error("message")`` keeps working
    while ``Error("message", param="base_url", hint="use https")`` is now valid
    too -- removing the asymmetry where only some exceptions accepted structured
    fields (spec: explicit > implicit; structured over stringly typed).

    Args:
        message: Human-readable description of the error.
        param: The offending field / parameter name, if applicable.
        hint: Actionable guidance for resolving the error, if any.
        details: Optional machine-readable context (for example, the sanitized
            validation-error entries from a wrapped pydantic ``ValidationError``).
    """

    def __init__(
        self,
        message: str = "",
        *,
        param: str | None = None,
        hint: str | None = None,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        self.param = param
        self.hint = hint
        self.details = details
        super().__init__(message)


class ProtocolCompatibilityError(StandardASRError):
    """Raised when an engine protocol cannot provide a requested feature.

    The error reports a version compatibility boundary, not a missing Python
    attribute. Generic consumers raise it before they inspect a feature that an
    older engine protocol does not define.

    Args:
        message: Human-readable description of the incompatibility.
        protocol_version: Protocol version declared by the engine.
        feature: Stable feature identifier requested by the consumer.
        required_protocol_version: Earliest protocol version that defines the
            feature, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        protocol_version: str,
        feature: str,
        required_protocol_version: str | None = None,
    ) -> None:
        self.protocol_version = protocol_version
        self.feature = feature
        self.required_protocol_version = required_protocol_version
        super().__init__(message)


class ConfigError(StructuredError, ValueError):
    """Raised when the CONFIGURATION -- supplied or ambient -- is invalid.

    The type asserts fault ownership: the configuration's SUPPLIER can fix
    it (a bad init-config field, a ``default_language`` not in
    ``selectable_languages``, a malformed ``--config`` / ``--set``). Who
    that supplier is depends on the surface, and each surface maps the SAME
    error accordingly:

    * **CLI**: the invoking user owns the config -- the flags AND the env
      vars -- so every ``ConfigError`` is caller-actionable there: usage
      exit 2, with the sanitized message naming the field to fix.
    * **Reference server**: a wire client cannot supply engine config at
      all (construction is zero-arg; options are the portable
      ``WireRuntimeParams``), so a ``ConfigError`` reaching the server --
      at construction, transcription, or session establishment -- is a
      deployment-side fault and maps to a scrubbed 500 (WS
      ``internal_error``); the client-fixable rejections have their own
      types (:class:`UnsupportedFeatureError` -> 422, request validation ->
      422). See :class:`ConfigurationRequiredError` for the absent-config
      503 state.

    Engine-DECLARATION defects (a malformed declared language tag, an
    unsatisfiable ``prepare`` shape, an IC.6 violation) are NOT this type:
    they raise :class:`EngineContractError`, because no configuration value
    fixes them. An engine that raises ``ConfigError`` (or lets a
    construction-time ``ValidationError``, which
    ``ModelRegistry.create`` wraps into one, escape its factory) for a
    fault that is NOT about the supplied configuration mis-asserts this
    ownership contract -- the compliance suite's zero-arg construction
    check fails such engines (``engine_construction_failed``); consumers do
    not second-guess the type. The ``ValueError`` mixin serves IN-PROCESS
    callers, who genuinely can pass a bad config value to a constructor.

    The one machine-distinguishable sub-state is ABSENT required
    configuration: raise (or catch) :class:`ConfigurationRequiredError` for
    that -- consumers such as the compliance suite treat "config missing from
    this environment" (skip) differently from "config invalid" (fail).
    """

    pass


class ConfigurationRequiredError(ConfigError):
    """Raised when required runtime configuration is ABSENT, not invalid.

    The narrow, machine-distinguishable subtype of :class:`ConfigError` for
    the one state that is a fact about the ENVIRONMENT rather than about any
    code or declaration: a required config field (typically a credential) was
    neither passed explicitly nor found in the environment. Consumers use the
    distinction to keep two very different verdicts apart:

    * the compliance suite SKIPS instantiation-level checks on this error (a
      credentialed engine on a clean CI is behaving correctly; the verdict
      must not depend on the runtime's credential state), while
    * any other :class:`ConfigError` -- an invalid supplied value, an
      internally inconsistent declaration, a factory contract bug -- stays a
      compliance FAILURE (skipping those would let a broken plugin read as
      green-with-warning).

    :meth:`~standard_asr.runtime.config.BaseConfig.from_env` raises this
    automatically when construction failed solely because required fields are
    missing, so engines following the documented ``explicit > env > raise``
    pattern get the classification for free. An engine building its config
    another way should raise this type itself for the missing-credential
    state.

    Transport mapping: the reference server maps this state to HTTP **503**
    (REST) / a ``service_unavailable`` frame (WS) with a stable generic
    detail -- whether it surfaces at zero-arg engine construction or lazily
    at transcription/session establishment (an engine deferring its
    credential check past ``__init__``). An operator-side availability
    state, never the caller's 422, and never the absent field names (those
    are deployment detail, safe-logged for the operator only).
    """

    pass


class TranscriptionError(StructuredError):
    """Raised when an engine fails during batch transcription (the cardinal sin guard).

    This is the **portable batch error contract**: when an
    engine's model inference, network call, or SDK fails inside ``_transcribe``,
    the failure MUST surface as a ``TranscriptionError`` (with the original
    exception preserved as ``__cause__`` via ``raise ... from``) so an
    application can catch one type across every engine instead of each engine's
    native exception (``RuntimeError``, an SDK error, ``requests.HTTPError``, and so on).
    It is the batch counterpart of the streaming ``error`` event's
    ``engine_error`` code. It denotes an engine/runtime fault, not
    a caller mistake (those raise :class:`ConfigError` /
    :class:`UnsupportedFeatureError` / :class:`InvalidProviderParamError` /
    :class:`AudioProcessingError`), so the server maps it to a generic 5xx.

    Carries the :class:`StructuredError` fields plus ``.retriable``: when an
    engine knows a failure is transient (a 503 / timeout / rate-limit) it MAY
    pass ``retriable=True`` so an application can decide whether to retry.
    ``None`` (the default) means "unknown" -- the safe reading is *do not assume
    it is safe to retry*.

    Args:
        message: Human-readable description of the failure.
        param: The offending field / parameter name, if applicable.
        hint: Actionable guidance, if any.
        details: Optional machine-readable context.
        retriable: ``True`` / ``False`` if the engine knows whether a retry may
            succeed; ``None`` when unknown.
    """

    def __init__(
        self,
        message: str = "",
        *,
        param: str | None = None,
        hint: str | None = None,
        details: list[dict[str, Any]] | None = None,
        retriable: bool | None = None,
    ) -> None:
        self.retriable = retriable
        super().__init__(message, param=param, hint=hint, details=details)


class ArtifactStatusError(StructuredError):
    """Raised when artifact status inspection fails unexpectedly."""

    pass


#: Closed availability vocabulary; MUST mirror the ``reason`` ``Literal`` on
#: :class:`ArtifactUnavailableError`. Enforced at construction because the
#: annotation alone does not run: consumers branch on the token (the CLI exit
#: split, wire mappings), so a typo'd or wrong-typed value from an engine's
#: ``raise`` would misclassify the failure silently -- and a value that cannot
#: be hashed crashes the consumer's membership test instead of the engine hook
#: that authored it.
_ARTIFACT_UNAVAILABLE_REASONS: Final = frozenset(
    {"missing", "incomplete", "corrupt", "unknown", "action_required", "downloads_disabled"}
)

#: Closed acquisition vocabulary; MUST mirror the ``reason`` ``Literal`` on
#: :class:`ArtifactAcquisitionError` (protocol.md AR.5 keeps it portable).
_ARTIFACT_ACQUISITION_REASONS: Final = frozenset(
    {"downloads_disabled", "action_required", "unsupported", "busy", "failed"}
)


def _check_artifact_reason(reason: object, allowed: frozenset[str]) -> None:
    """Require a closed-vocabulary artifact ``reason`` token at construction.

    Args:
        reason: Engine-supplied reason token.
        allowed: The closed vocabulary the owning exception declares.

    Returns:
        None.

    Raises:
        TypeError: If ``reason`` is not a string.
        ValueError: If ``reason`` is not in the closed vocabulary.
    """
    if not isinstance(reason, str):
        raise TypeError(f"reason must be one of {sorted(allowed)}, not {type(reason).__name__}.")
    if reason not in allowed:
        raise ValueError(f"reason {reason!r} is not one of {sorted(allowed)}.")


def _checked_artifact_report(report: object) -> ArtifactReport:
    """Re-validate an artifact ``report`` payload at construction.

    Consumers dereference the report structurally (the CLI renders
    ``report.requirements``, and each requirement's actions, inside its
    exception handler), so a wrong-typed payload would crash the reporting
    boundary instead of the code that attached it. An ``isinstance`` check
    proves the class, not the contents: ``model_copy(update=...)`` skips
    validation by design, so an engine that adjusts one field of a status
    report can attach an ``ArtifactReport`` whose ``requirements`` never met a
    validator. The report is therefore rebuilt from a plain-data dump of its
    fields. A bare ``model_validate(report)`` would not do: pydantic reruns
    only the model-level validator on an existing instance, and that validator
    itself reads the requirements.

    Args:
        report: Engine-supplied report payload.

    Returns:
        The re-validated report.

    Raises:
        TypeError: If the payload is not an ``ArtifactReport``.
        ValueError: If the report fails re-validation (pydantic's
            ``ValidationError``).
    """
    # Imported here, not at module top: the exception hierarchy is the bottom
    # layer of the contract package and must stay importable without pulling
    # in the artifact data models (which is also why the top-level import is
    # TYPE_CHECKING-only). The exception constructors are cold paths.
    from standard_asr.contract.artifacts import ArtifactReport

    if report is None:
        raise TypeError("report must be an ArtifactReport, not None.")
    if not isinstance(report, ArtifactReport):
        raise TypeError(f"report must be an ArtifactReport, not {type(report).__name__}.")
    # ``warnings=False`` on the intermediate dump only, for the reason
    # DeclaredEngineMetadata.canonical_json() gives: serializing a field whose
    # value is not its declared type emits a pydantic serializer warning, and
    # the ValidationError below carries everything that warning would.
    return ArtifactReport.model_validate(report.model_dump(mode="python", warnings=False))


def _checked_required_actions(actions: Iterable[object]) -> tuple[ArtifactAction, ...]:
    """Re-validate directly attached artifact actions at construction.

    ``ArtifactAction`` re-validates itself when nested in a report, but an
    action attached straight to an exception crosses no model boundary: a
    ``model_copy(update=...)`` value that skipped the HTTPS/user-information
    validators would otherwise ride the error into the CLI's action rendering
    (protocol.md AR.8: an artifact error's actions must never carry
    credentials).

    Args:
        actions: Engine-supplied actions.

    Returns:
        The re-validated actions as a tuple.

    Raises:
        TypeError: If ``actions`` is not iterable or an item is not an
            ``ArtifactAction``.
        ValueError: If an action fails re-validation (pydantic's
            ``ValidationError``).
    """
    # See _checked_artifact_report for why this import is function-local.
    from standard_asr.contract.artifacts import ArtifactAction

    validated: list[ArtifactAction] = []
    for action in tuple(actions):
        if not isinstance(action, ArtifactAction):
            raise TypeError(
                f"required_actions items must be ArtifactAction, not {type(action).__name__}."
            )
        validated.append(ArtifactAction.model_validate(action))
    return tuple(validated)


class ArtifactUnavailableError(StructuredError):
    """Raised when required artifacts cannot support inference or warm-up.

    Args:
        message: Human-readable description of the unavailable artifacts.
        reason: Machine-readable availability reason.
        report: Status report that established the unavailable state.
        hint: Actionable guidance, if any.

    Raises:
        TypeError: If ``reason`` is not a string or ``report`` is not an
            ``ArtifactReport``.
        ValueError: If ``reason`` is not in the closed availability
            vocabulary, or ``report`` fails re-validation.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: Literal[
            "missing",
            "incomplete",
            "corrupt",
            "unknown",
            "action_required",
            "downloads_disabled",
        ],
        report: ArtifactReport,
        hint: str | None = None,
    ) -> None:
        _check_artifact_reason(reason, _ARTIFACT_UNAVAILABLE_REASONS)
        self.reason = reason
        self.report = _checked_artifact_report(report)
        super().__init__(message, hint=hint)


class ArtifactAcquisitionError(StructuredError):
    """Raised when an explicit or implicit artifact acquisition fails.

    Args:
        message: Human-readable description of the acquisition failure.
        reason: Machine-readable acquisition reason.
        report: Latest available artifact status report, if any.
        required_actions: External actions discovered by the failed attempt.
        retriable_after: Suggested nonnegative delay before another attempt.
        hint: Actionable guidance, if any.

    Raises:
        TypeError: If ``reason`` is not a string, ``report`` is not an
            ``ArtifactReport`` or ``None``, or a required action is not an
            ``ArtifactAction``.
        ValueError: If ``reason`` is not in the closed acquisition
            vocabulary, ``retriable_after`` is negative or non-finite, or the
            report or a required action fails re-validation.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: Literal[
            "downloads_disabled",
            "action_required",
            "unsupported",
            "busy",
            "failed",
        ],
        report: ArtifactReport | None = None,
        required_actions: tuple[ArtifactAction, ...] = (),
        retriable_after: float | None = None,
        hint: str | None = None,
    ) -> None:
        if retriable_after is not None and (
            not math.isfinite(retriable_after) or retriable_after < 0
        ):
            raise ValueError("retriable_after must be finite and nonnegative.")
        _check_artifact_reason(reason, _ARTIFACT_ACQUISITION_REASONS)
        self.reason = reason
        self.report = None if report is None else _checked_artifact_report(report)
        self.required_actions = _checked_required_actions(required_actions)
        self.retriable_after = retriable_after
        super().__init__(message, hint=hint)


class ArtifactProgressCallbackError(StandardASRError):
    """Raised after acquisition succeeds but its progress callback fails.

    Args:
        message: Human-readable description of the callback failure.
        report: Final artifact report produced by the successful operation.
    """

    def __init__(self, message: str, *, report: ArtifactReport) -> None:
        self.report = report
        super().__init__(message)


class AudioProcessingError(StandardASRError):
    """Raised when an error occurs during audio loading or processing.

    The audio loading and conversion functions in :mod:`standard_asr.audio`
    raise this.
    """

    pass


class IncompatibleAudioInputError(AudioProcessingError):
    """Raised when no viable conversion path exists for the provided audio.

    This happens when the shape an application provides cannot be negotiated
    into any shape the engine accepts (for example, a local array given to an engine
    that only accepts a server-fetchable URL).

    Args:
        provided: Human-readable description of the provided input shape.
        accepted: The engine's accepted input kinds.
        hint: Actionable guidance for resolving the mismatch.
    """

    def __init__(self, provided: str, accepted: object, hint: str) -> None:
        self.provided = provided
        self.accepted = accepted
        self.hint = hint
        super().__init__(f"Cannot deliver {provided} to an engine that accepts {accepted}. {hint}")


class UnsupportedFeatureError(StructuredError):
    """Raised in strict mode when a requested standard feature is unsupported.

    In best_effort mode the unsupported parameter is ignored and a structured
    diagnostic is returned instead of raising. The strict path carries the same
    structured context as that diagnostic so callers can inspect *which* feature
    was rejected without parsing the message.

    Args:
        message: Human-readable description of the rejection.
        param: The offending standard parameter name, if applicable.
        mode: The mode (``"batch"`` / ``"streaming"``) the rejection occurred in,
            if applicable.
        hint: Actionable guidance for resolving the rejection, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        mode: str | None = None,
        hint: str | None = None,
    ) -> None:
        self.mode = mode
        super().__init__(message, param=param, hint=hint)


class InvalidProviderParamError(StructuredError, ValueError):
    """Raised when ``provider_params`` are invalid for the target engine.

    Unlike standard-set parameters, ``provider_params`` errors are always raised
    regardless of strict / best_effort -- they indicate a code-level bug (such
    as passing one engine's params model to another after a swap).
    """

    pass


class EngineContractError(StandardASRError):
    """Raised when a constructed engine breaks the protocol contract.

    The runtime counterpart of a compliance failure, in two shapes:

    * **Runtime behavior**: a synchronous ``StandardASR`` member
      (``transcribe`` / ``start_transcription`` / ``supports`` /
      ``recommended_wire_format`` / ``artifact_status`` /
      ``acquire_artifacts``), or the optional ``prepare`` engine method,
      returned an awaitable or a value outside its pinned return type. Raised by
      :func:`standard_asr.runtime.protocol_boundary.require_sync_result` at
      the consumer call sites (CLI, reference server) so the defect is loud
      at the boundary instead of surfacing as a confusing secondary
      ``AttributeError`` (or a silent misreading) deep inside another
      subsystem.
    * **Declaration shape**: the engine DECLARED something the contract
      forbids -- a ``prepare`` that is a coroutine function, non-callable,
      or parameter-requiring; a malformed ``selectable_languages`` /
      ``detectable_languages`` tag; a language axis without the IC.6
      ``default_language`` obligation. No caller-side value can fix these,
      which is what separates them from :class:`ConfigError` (an invalid
      configuration VALUE, fixable by whoever supplies the config).

    An **engine/plugin fault, never a caller mistake** -- deliberately NOT a
    :class:`ValueError`: transports and the CLI map the ``ValueError`` family
    to caller-fixable surfaces (HTTP 422 / usage exit 2), while this must
    land on the engine-fault surfaces (scrubbed HTTP 500 / ``internal_error``
    frame / CLI exit 1). If you hit it as an application developer, report it
    to the engine's author. Messages carry type names only, never the
    offending value.
    """

    pass


class SubtitleRenderingError(StandardASRError, ValueError):
    """Raised by ``to_srt`` / ``to_vtt`` when segments cannot render as visible cues.

    A subtitle cue is an interval claim -- "this text occurs at this time" --
    and it must survive the output's millisecond grid to be seen at all. A
    segment is therefore UNRENDERABLE in either of two ways: it lacks a
    measured span (``Segment.timestamp_status`` is ``"start_only"`` or
    ``"unavailable"``), or its measured span quantizes to zero milliseconds
    on the output grid (``end`` and ``start`` format to the same timestamp
    -- players silently drop such cues, so emitting one silently hides the
    text while the render call reports success). Neither dropping the text
    nor fabricating timing is the renderer's to choose silently: under the
    default policy (``on_unrenderable="error"``) it raises this error, and
    the caller picks the loss explicitly (``"omit"`` drops the unrenderable
    segments' text from the timed cues; ``"collapse"`` renders one
    whole-text cue with no real timeline). Mixes in :class:`ValueError`:
    the caller can fix the call -- choose a policy, or supply renderable
    segments.

    Args:
        message: Human-readable description of the rejection.
        unrenderable: How many segments cannot render as visible cues, if
            known.
        total: How many segments the result carries, if known.
    """

    def __init__(
        self,
        message: str = "",
        *,
        unrenderable: int | None = None,
        total: int | None = None,
    ) -> None:
        self.unrenderable = unrenderable
        self.total = total
        super().__init__(message)


class StreamClosedError(StandardASRError):
    """Raised when audio is delivered to a streaming session that is closed.

    Strictly a **lifecycle-close** breach: the input side is over,
    so the audio can no longer be consumed. Covers ``send_audio`` after
    ``end_audio()`` and ``send_audio`` after the session already delivered a
    terminal event (the audio queue has no consumer anymore). It does NOT cover
    *usage* mistakes against a still-live session (mixing ``feed`` with manual
    input, calling ``feed`` twice, or iterating the event stream twice) -- those
    raise :class:`InvalidSessionUseError`, so an application can tell "the
    session ended" apart from "my code drove the session incorrectly".
    """

    pass


class InvalidSessionUseError(StandardASRError, ValueError):
    """Raised when a streaming session is driven incorrectly while still live.

    A caller-side **programming error** against an open session -- distinct from
    :class:`StreamClosedError` (the stream genuinely ended). It covers the
    still-live-session usage breaches that are NOT lifecycle-close:

    * mixing managed ``feed()`` with manual ``send_audio`` / ``end_audio`` (only
      one input mode may own a session);
    * calling ``feed()`` more than once (a session owns at most one fed source);
    * iterating the event stream more than once (single-consumer contract).

    The session is not closed in any of these cases -- the mistake is in how the
    application used it. Catching :class:`StreamClosedError` here would lead an
    application to wrongly conclude the session terminated and rebuild it.
    Mixes in :class:`ValueError` (like :class:`ConfigError` /
    :class:`InvalidProviderParamError`): it is a bad-call programming error.
    (It has no HTTP mapping: it fires only against an in-process session object,
    and the server drives its own sessions correctly by construction.)
    """

    pass


class FFmpegNotFoundError(AudioProcessingError, FileNotFoundError):
    """Raised when FFmpeg is required but not found in the system `PATH`."""

    pass


class FFprobeNotFoundError(AudioProcessingError, FileNotFoundError):
    """Raised when FFprobe is required but not found in the system `PATH`."""

    pass


class DiscoveryError(StandardASRError):
    """Base class for discovery and plugin-related errors."""

    pass


class EntrypointValidationError(DiscoveryError, ValueError):
    """Raised when an entry point name or metadata is invalid."""

    pass


class FactoryLoadError(DiscoveryError, ImportError):
    """Raised when an entry point target cannot be imported or is not callable."""

    pass


__all__ = [
    "AudioProcessingError",
    "ArtifactAcquisitionError",
    "ArtifactProgressCallbackError",
    "ArtifactStatusError",
    "ArtifactUnavailableError",
    "ConfigError",
    "ConfigurationRequiredError",
    "DiscoveryError",
    "EngineContractError",
    "EntrypointValidationError",
    "FFmpegNotFoundError",
    "FFprobeNotFoundError",
    "FactoryLoadError",
    "IncompatibleAudioInputError",
    "InvalidProviderParamError",
    "InvalidSessionUseError",
    "ProtocolCompatibilityError",
    "StandardASRError",
    "StreamClosedError",
    "StructuredError",
    "SubtitleRenderingError",
    "TranscriptionError",
    "UnsupportedFeatureError",
]
