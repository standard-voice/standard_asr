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

from typing import Any


class StandardASRError(Exception):
    """Base exception class for all errors raised by the standard_asr library."""

    pass


class StructuredError(StandardASRError):
    """Base for errors that carry machine-readable context beside the message.

    Gives the error half of the contract the same "don't make me parse the
    message" property the diagnostics have: an application can read ``.param``
    (the offending field/parameter), ``.hint`` (actionable guidance), and
    ``.details`` (structured context -- e.g. the sanitized pydantic error entries
    behind a wrapped config failure) without scraping ``str(exc)``. Every context
    field is optional and keyword-only, so ``Error("message")`` keeps working
    while ``Error("message", param="base_url", hint="use https")`` is now valid
    too -- removing the asymmetry where only some exceptions accepted structured
    fields (spec: explicit > implicit; structured over stringly-typed).

    Args:
        message: Human-readable description of the error.
        param: The offending field / parameter name, if applicable.
        hint: Actionable guidance for resolving the error, if any.
        details: Optional machine-readable context (e.g. the sanitized
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


class ConfigError(StructuredError, ValueError):
    """Raised when a configuration is invalid -- user-provided or engine-declared.

    Two fault domains share this type: a value the **caller** can fix (a bad
    init-config field, a ``default_language`` not in ``selectable_languages``)
    and a declaration mistake the **engine author** must fix (a malformed
    ``selectable_languages`` / ``detectable_languages`` tag, surfaced by the
    standard layer at first transcribe). The latter is a plugin bug, not a
    request error -- if it reaches you through an installed engine, report it to
    the plugin author rather than changing your own configuration. The server
    maps this to HTTP 422 (``ValueError`` subclass).
    """

    pass


class TranscriptionError(StructuredError):
    """Raised when an engine fails during batch transcription (the cardinal sin guard).

    This is the **portable batch error contract** (spec Runtime R7): when an
    engine's model inference, network call, or SDK fails inside ``_transcribe``,
    the failure MUST surface as a ``TranscriptionError`` (with the original
    exception preserved as ``__cause__`` via ``raise ... from``) so an
    application can catch one type across every engine instead of each engine's
    native exception (``RuntimeError``, an SDK error, ``requests.HTTPError`` ...).
    It is the batch counterpart of the streaming ``error`` event's
    ``engine_error`` code (spec ST §6.2). It denotes an engine/runtime fault, not
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


class AudioProcessingError(StandardASRError):
    """
    Raised when an error occurs during audio loading or processing.
    This is typically raised by functions in the audio_loader module.
    """

    pass


class IncompatibleAudioInputError(AudioProcessingError):
    """Raised when no viable conversion path exists for the provided audio.

    This happens when the shape an application provides cannot be negotiated
    into any shape the engine accepts (e.g. a local array given to an engine
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


class StreamClosedError(StandardASRError):
    """Raised when audio is delivered to a streaming session that is closed.

    Strictly a **lifecycle-close** breach (spec ST §6.1): the input side is over,
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
    :class:`StreamClosedError` (the stream genuinely ended). It covers the spec
    ST §3.3 / §6.1 usage breaches that are NOT lifecycle-close:

    * mixing managed ``feed()`` with manual ``send_audio`` / ``end_audio`` (only
      one input mode may own a session);
    * calling ``feed()`` more than once (a session owns at most one fed source);
    * iterating the event stream more than once (single-consumer contract).

    The session is not closed in any of these cases -- the mistake is in how the
    application used it. Catching :class:`StreamClosedError` here would lead an
    application to wrongly conclude the session terminated and rebuild it.
    Mixes in :class:`ValueError` (like :class:`ConfigError` /
    :class:`InvalidProviderParamError`): it is a bad-call programming error, and
    the server maps it to HTTP 422.
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
    "ConfigError",
    "DiscoveryError",
    "EntrypointValidationError",
    "FFmpegNotFoundError",
    "FFprobeNotFoundError",
    "FactoryLoadError",
    "IncompatibleAudioInputError",
    "InvalidProviderParamError",
    "InvalidSessionUseError",
    "StandardASRError",
    "StreamClosedError",
    "StructuredError",
    "TranscriptionError",
    "UnsupportedFeatureError",
]
