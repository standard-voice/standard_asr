# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-request runtime parameters (closed portable set + escape hatch).

:class:`RuntimeParams` is the **closed** container of per-request settings
(spec, section "Runtime Parameters"). It carries the v1 portable standard set
(``language``, ``candidate_languages``, ``word_timestamps``, ``diarization``,
and the ``guidance`` family ``prompt`` / ``phrase_hints``) plus a single typed
escape hatch, ``provider_params``, for engine-specific knobs. ASR authors MUST NOT add
top-level fields (``extra="forbid"``); engine-specific knobs go through a
:class:`ProviderParams` subclass.

The ``guidance`` family uses **flat fields** directly on ``RuntimeParams`` (no
nested sub-object) so that IDE completion surfaces every channel. Each channel
is optional, best-effort, non-binding, positive-polarity, capability-negotiated,
and never silently degraded. Degradation to ``prompt`` is opt-in and one-way via
``on_unsupported="degrade_to_prompt"``.
"""

from __future__ import annotations

from enum import Enum
from typing import Final, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

from standard_asr.contract.capabilities import WordTimestampGranularityName
from standard_asr.contract.language import AUTO, is_valid_bcp47, normalize_bcp47


class WordTimestampGranularity(str, Enum):
    """Granularity for requested word timestamps.

    The member *values* are the single source of truth shared with the
    declaration-side capability vocabulary
    :data:`~standard_asr.contract.capabilities.WordTimestampGranularityName` (a
    ``Literal``). A module-level assertion (below) and a drift test bind the
    two sets so an additive change to one cannot silently desync the other.

    Attributes:
        WORD: Word-level timestamps.
        SEGMENT: Segment-level timestamps.
        CHAR: Character-level timestamps (reserved, additive).
    """

    WORD = "word"
    SEGMENT = "segment"
    CHAR = "char"


# Enforce the single-source-of-truth link at import time. The request
# enum (`WordTimestampGranularity`) and the capability `Literal`
# (`WordTimestampGranularityName`) historically defined the same vocabulary in
# two places with no link; an additive change to one could silently desync the
# other. This invariant makes such a drift a hard import-time failure (a drift
# test asserts the same), so the two are guaranteed to stay identical.
assert {g.value for g in WordTimestampGranularity} == set(get_args(WordTimestampGranularityName)), (
    "WordTimestampGranularity desynced from capabilities.WordTimestampGranularityName"
)


class ProviderParams(BaseModel):
    """Base class for an engine's typed, non-portable parameter model.

    Engines publish a subclass (e.g. ``OpenAIParams``) and declare it as their
    expected ``provider_params`` type. Passing one engine's params model to a
    different engine is a validation error (swap-safe), raised as
    :class:`~standard_asr.contract.exceptions.InvalidProviderParamError` by the engine
    layer regardless of the strict / best_effort policy.

    The swap-safety match is **exact** (``type(provided) is <EngineParams>``),
    not ``isinstance``: every engine MUST publish a distinct *terminal* params
    type, because honoring a subclass would let one engine silently accept
    another's params and drop the extra knobs. Inheritance is
    therefore not a way to declare cross-engine compatibility. This bare base is
    never a valid concrete params model -- declaring it as an engine's
    ``provider_params`` type, or passing a bare instance, is rejected (the latter
    at :class:`RuntimeParams` construction).
    """

    # Engine params subclasses may carry `model_*` knobs; opt out of pydantic's
    # `model_` protected namespace so they do not warn (the warning fires on
    # older pydantic, e.g. the lower-bounds 2.5).
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class DiarizationRequest(BaseModel):
    """Request marker for speaker diarization ("who said what").

    Presence enables diarization: ``RuntimeParams(diarization=DiarizationRequest())``
    (or the :data:`DIARIZE` convenience constant) requests speaker labels;
    ``diarization=None`` (the default) means not requested. There is **no**
    ``[]``-analogue -- a "requested-but-empty" state is meaningless for an
    on/off feature, so ``None`` vs an instance is the whole state space. On the
    wire the marker maps three ways: ``"diarization": {}``
    -> ``DiarizationRequest()``; ``"diarization": null`` -> ``None``; key
    absent -> ``None``.

    The model is a deliberately **empty** frozen marker in v1: tuning knobs
    (``num_speakers`` / ``min``/``max`` hints, granularity selection) are
    deferred because today's engine landscape cannot honor them portably --
    they graduate additively onto this model once support is broad enough.
    ``extra="forbid"`` keeps that
    evolution honest: an unknown key (e.g. a client guessing
    ``num_speakers``) fails loudly (a 422 on the wire) instead of being
    silently ignored. An import-time assert in
    :mod:`standard_asr.runtime.gating` additionally forces sub-gating to be
    written the moment a field is added here.

    Raises:
        ValueError: If an unknown field is supplied (``extra="forbid"``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


#: Convenience constant for enabling diarization:
#: ``RuntimeParams(diarization=DIARIZE)``. Parallels :data:`~standard_asr.contract.language.AUTO`
#: for the ``language`` field -- the marker carries no state (frozen, no
#: fields), so one shared instance reads better at call sites than an inline
#: ``DiarizationRequest()``.
DIARIZE: Final[DiarizationRequest] = DiarizationRequest()


class RuntimeParams(BaseModel):
    """Closed per-request parameter container.

    Attributes:
        language: Per-request language (BCP-47 or ``"auto"``) overriding the
            engine default. Gated by ``<mode>.language.runtime_override``.
        candidate_languages: Candidate languages, meaningful only in ``auto``
            mode. Gated by ``<mode>.language.candidate_languages``.
        word_timestamps: Requested word-timestamp granularity. Gated by
            ``<mode>.word_timestamps``.
        diarization: Speaker-diarization request marker; presence = enable
            (``None`` = not requested). Gated by ``<mode>.diarization``.
        prompt: Free-text guidance prompt. Gated by ``<mode>.guidance.prompt``.
        phrase_hints: Phrase-hint boost terms. Gated by
            ``<mode>.guidance.phrase_hints``.
        on_unsupported: Guidance degradation policy. ``"fail"`` (default) keeps
            the fail-closed contract; ``"degrade_to_prompt"`` opts into the
            one-way rich->prompt fallback (a diagnostic is emitted on degrade).
        provider_params: Engine-specific typed knobs, or ``None``.

    Raises:
        ValueError: If field validation fails.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    language: str | None = Field(
        default=None, description="Per-request language (BCP-47 or 'auto')."
    )
    candidate_languages: list[str] | None = Field(
        default=None, description="Candidate languages (auto mode only)."
    )
    word_timestamps: WordTimestampGranularity | None = Field(
        default=None, description="Requested word-timestamp granularity."
    )
    diarization: DiarizationRequest | None = Field(
        default=None,
        description="Speaker-diarization request marker (presence = enable).",
    )
    prompt: str | None = Field(default=None, description="Free-text guidance prompt.")
    phrase_hints: list[str] | None = Field(default=None, description="Phrase-hint boost terms.")
    on_unsupported: Literal["fail", "degrade_to_prompt"] = Field(
        default="fail",
        description=(
            "Guidance degradation policy (opt-in one-way to prompt). This is a "
            "policy directive, not a capability-gated channel. 'fail' means do "
            "NOT degrade -- the unsupported channel then follows the standard "
            "strict/best_effort gate (strict raises, best_effort drops with a "
            "diagnostic); it does NOT force the whole request to fail. "
            "'degrade_to_prompt' opts into the one-way rich->prompt fallback."
        ),
    )
    provider_params: ProviderParams | None = Field(
        default=None, description="Engine-specific typed knobs."
    )

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str | None) -> str | None:
        """Reject a malformed language tag at construction (fail-fast).

        A malformed tag is an invalid *value*, not an unsupported feature, so --
        like ``provider_params`` errors -- it always raises,
        independent of the strict / best_effort policy. This keeps a common
        mistake (passing ``"english"`` instead of ``"en"``) from silently
        reaching the engine. ``"auto"`` (auto-detect) and ``None`` are permitted;
        membership against an engine's languages is enforced separately.

        Args:
            value: The provided language tag, ``"auto"``, or ``None``.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If ``value`` is a non-empty, non-``"auto"`` string that
                is not a well-formed BCP-47 tag.
        """
        return _validate_language_tag(value)

    @field_validator("candidate_languages")
    @classmethod
    def _validate_candidate_languages(cls, value: list[str] | None) -> list[str] | None:
        """Reject a malformed / ``"auto"`` candidate at construction (fail-fast).

        See :func:`_validate_candidate_language_list`: this gives each
        candidate the same always-raise well-formedness contract the scalar
        ``language`` field already has, so a code bug (a native name, ``"auto"``
        in the candidate list) fails loudly at construction instead of lying
        dormant until the engine/mode happens to reach candidate-language
        resolution in auto mode.

        Args:
            value: The provided candidate-language list, ``[]``, or ``None``.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If any item is malformed or is the reserved ``"auto"``.
        """
        return _validate_candidate_language_list(value)

    @field_validator("phrase_hints")
    @classmethod
    def _validate_phrase_hints(cls, value: list[str] | None) -> list[str] | None:
        """Reject empty / whitespace-only phrase-hint terms at construction.

        See :func:`_validate_phrase_hints_list`: a blank term carries no boost
        signal, would fool the degrade survival check (``"" in x`` is always
        true), and could be handed to the engine as an empty string. ``None`` and
        ``[]`` (the requested-but-empty sentinel) pass through.

        Args:
            value: The provided phrase-hint list, ``[]``, or ``None``.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If any term is empty or whitespace-only.
        """
        return _validate_phrase_hints_list(value)

    @field_validator("provider_params")
    @classmethod
    def _reject_bare_provider_params(cls, value: ProviderParams | None) -> ProviderParams | None:
        """Reject the bare :class:`ProviderParams` base at construction (fail-fast).

        ``provider_params`` is typed ``ProviderParams | None``, so pydantic would
        otherwise coerce an empty mapping (``provider_params={}``) into a bare
        ``ProviderParams()`` instance -- which then reaches the gate and trips the
        swap-safety check with a misleading ``"(swapped engine?)"`` message about
        a wrong-engine model, when the real mistake was passing a dict (or the
        base class) instead of the engine's concrete params subclass. The bare
        base carries no fields and is never a valid concrete params model, so it
        is refused here with a message that names the real fix. A concrete
        subclass instance passes through unchanged.

        Args:
            value: The provided provider params, ``None``, or (coerced) a bare
                base instance.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If ``value`` is exactly a bare ``ProviderParams`` instance
                (including one coerced from a mapping) rather than ``None`` or a
                concrete subclass.
        """
        if value is not None and type(value) is ProviderParams:
            raise ValueError(
                "provider_params must be the engine's concrete ProviderParams "
                "subclass, not the bare ProviderParams base (or a mapping coerced "
                "into it). Pass an instance of the engine's published params type."
            )
        return value


def _validate_language_tag(value: str | None) -> str | None:
    """Validate a per-request language tag (shared by the params models).

    ``"auto"`` and ``None`` are permitted; any other value must be a well-formed
    BCP-47 tag (membership against an engine's languages is enforced separately).

    Args:
        value: The provided language tag, ``"auto"``, or ``None``.

    Returns:
        The validated value unchanged.

    Raises:
        ValueError: If ``value`` is a non-empty, non-``"auto"`` string that is
            not a well-formed BCP-47 tag.
    """
    if value is None or value == "auto":
        return value
    if not is_valid_bcp47(value):
        # The raw value MUST NOT be embedded in the message: this error is
        # surfaced verbatim by every transport (CLI, logs, the server's
        # unauthenticated 422 body, where validation errors never echo the
        # request input), and a mis-pasted secret sent as `language`
        # would otherwise be reflected back.
        raise ValueError(
            "language tag is not a well-formed BCP-47 language tag "
            "(e.g. 'en', 'en-US', 'zh-Hans') or 'auto'."
        )
    return value


def _validate_candidate_language_list(value: list[str] | None) -> list[str] | None:
    """Validate per-request candidate-language *well-formedness* at construction.

    Mirrors the scalar ``language`` validator's fail-fast contract for each list
    item: a malformed BCP-47 tag or the reserved ``"auto"`` token is an invalid
    *value* (a caller code bug), so it ALWAYS raises here -- independent of the
    strict / best_effort policy (a malformed value MUST always raise, never be
    dropped). Without this, a malformed candidate slipped silently past
    construction and -- because candidate resolution short-circuits an
    *unsupported* candidate axis to ``None`` + a diagnostic BEFORE the per-item
    well-formedness check -- could lie dormant until the caller later switched to
    ``language="auto"`` on a
    supporting engine, defeating fail-fast.

    Only well-formedness and the ``"auto"`` ban are enforced here (the same two
    checks that already always-raise inside
    :func:`~standard_asr.contract.language.effective_candidate_languages`). Membership
    against an engine's ``detectable_languages`` and the ``max`` limit remain
    owned by ``language.py`` (they need engine capabilities, unavailable at
    construction). ``None`` and the empty list ``[]`` (the requested-but-empty
    sentinel) carry no item to validate and pass through.

    Args:
        value: The provided candidate-language list, ``[]``, or ``None``.

    Returns:
        The validated value unchanged.

    Raises:
        ValueError: If any item is not a well-formed BCP-47 tag, or is the
            reserved ``"auto"`` token.
    """
    if not value:  # None or [] -- nothing to validate.
        return value
    for tag in value:
        # is_valid_bcp47 is checked before the 'auto' comparison so an
        # empty/whitespace entry is attributed as malformed (not tripping the
        # normalizer's empty-tag error); 'auto' is well-formed by that guard, so
        # it survives to the reserved-word check below. This matches the ordering
        # in language.effective_candidate_languages exactly.
        if not is_valid_bcp47(tag):
            # The raw value MUST NOT be embedded in the message (same reasoning as
            # the scalar `language` validator: it is echoed verbatim by the
            # server's unauthenticated 422 body and logs).
            raise ValueError(
                "candidate_languages contains an entry that is not a well-formed "
                "BCP-47 language tag (e.g. 'en', 'en-US', 'zh-Hans')."
            )
        if normalize_bcp47(tag) == AUTO:
            raise ValueError(
                "candidate_languages MUST NOT contain 'auto' (it is a directive, "
                "not a candidate language)."
            )
    return value


def _validate_phrase_hints_list(value: list[str] | None) -> list[str] | None:
    """Reject empty / whitespace-only phrase-hint terms at construction.

    A blank term (``""`` or all whitespace) carries no boost signal and is a
    caller mistake (e.g. a stray trailing element from a form). Worse, it breaks
    downstream invariants: the opt-in ``phrase_hints -> prompt`` degrade checks
    whether any term *survived* truncation with ``term in surviving_frame``, and
    ``"" in anything`` is always ``True`` -- so a single ``""`` term would make
    the standard layer falsely report a successful degrade even when every real
    term was truncated away (never silently degrade). The
    per-term character/word limits could also shrink a term to ``""`` and hand a
    blank string to the engine (undefined input). Refusing blank terms here, at
    construction, removes the whole class loudly.

    ``None`` and the empty list ``[]`` (the requested-but-empty sentinel) carry
    no term and pass through unchanged; a non-empty list with only real terms
    passes through unchanged.

    Args:
        value: The provided phrase-hint list, ``[]``, or ``None``.

    Returns:
        The validated value unchanged.

    Raises:
        ValueError: If any term is empty or whitespace-only.
    """
    if not value:  # None or [] -- nothing to validate.
        return value
    if any(not term.strip() for term in value):
        # The raw values are not echoed (a phrase hint could carry sensitive
        # text); the message names the rule, not the offending entry.
        raise ValueError(
            "phrase_hints must not contain empty or whitespace-only terms "
            "(use [] to request no hints)."
        )
    return value


class WireRuntimeParams(BaseModel):
    """The portable runtime params accepted over an untyped wire (D5).

    The server (and any other transport that accepts JSON it did not type) MUST
    accept **only** the portable standard set. The engine-specific
    ``provider_params`` escape hatch on :class:`RuntimeParams` is **discover-only**
    -- its JSON Schema is published for discovery / UI generation, but it cannot
    be *constructed* from untyped wire JSON without the engine's params type, and
    accepting a raw ``provider_params`` object would let it reach the engine
    ambiguously (untyped, unvalidated). This model therefore carries exactly the
    portable fields and **rejects** a ``provider_params`` key via
    ``extra="forbid"`` (it has no such field), so a request that sends one fails
    loudly with a clear validation error instead of silently dropping or
    mis-routing it.

    A module-level drift assertion (below) binds this model's field set to
    ``RuntimeParams`` minus ``provider_params`` so an additive change to the
    portable set cannot silently desync the wire view.

    Attributes:
        language: See :class:`RuntimeParams`.
        candidate_languages: See :class:`RuntimeParams`.
        word_timestamps: See :class:`RuntimeParams`.
        diarization: See :class:`RuntimeParams`. On the wire, ``{}`` maps to
            :class:`DiarizationRequest` (enable), ``null`` / an absent key map
            to ``None``, and unknown keys inside the object are rejected.
        prompt: See :class:`RuntimeParams`.
        phrase_hints: See :class:`RuntimeParams`.
        on_unsupported: See :class:`RuntimeParams`.

    Raises:
        ValueError: If field validation fails, or a non-portable key (e.g.
            ``provider_params``) is supplied.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    language: str | None = Field(
        default=None, description="Per-request language (BCP-47 or 'auto')."
    )
    candidate_languages: list[str] | None = Field(
        default=None, description="Candidate languages (auto mode only)."
    )
    word_timestamps: WordTimestampGranularity | None = Field(
        default=None, description="Requested word-timestamp granularity."
    )
    diarization: DiarizationRequest | None = Field(
        default=None,
        description="Speaker-diarization request marker (presence = enable).",
    )
    prompt: str | None = Field(default=None, description="Free-text guidance prompt.")
    phrase_hints: list[str] | None = Field(default=None, description="Phrase-hint boost terms.")
    on_unsupported: Literal["fail", "degrade_to_prompt"] = Field(
        default="fail",
        description=(
            "Guidance degradation policy (opt-in one-way to prompt). This is a "
            "policy directive, not a capability-gated channel. 'fail' means do "
            "NOT degrade -- the unsupported channel then follows the standard "
            "strict/best_effort gate (strict raises, best_effort drops with a "
            "diagnostic); it does NOT force the whole request to fail. "
            "'degrade_to_prompt' opts into the one-way rich->prompt fallback."
        ),
    )

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str | None) -> str | None:
        """Reject a malformed language tag at construction (fail-fast).

        Args:
            value: The provided language tag, ``"auto"``, or ``None``.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If the tag is malformed (see :func:`_validate_language_tag`).
        """
        return _validate_language_tag(value)

    @field_validator("candidate_languages")
    @classmethod
    def _validate_candidate_languages(cls, value: list[str] | None) -> list[str] | None:
        """Reject a malformed / ``"auto"`` candidate at construction (fail-fast).

        Mirrors :class:`RuntimeParams` so the wire view rejects a malformed or
        ``"auto"`` candidate with a **422** at request validation time, instead of
        carrying it silently into a session where it would only surface on a later
        ``auto``-mode request (see :func:`_validate_candidate_language_list`).

        Args:
            value: The provided candidate-language list, ``[]``, or ``None``.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If any item is malformed or is the reserved ``"auto"``.
        """
        return _validate_candidate_language_list(value)

    @field_validator("phrase_hints")
    @classmethod
    def _validate_phrase_hints(cls, value: list[str] | None) -> list[str] | None:
        """Reject empty / whitespace-only phrase-hint terms at construction.

        Mirrors :class:`RuntimeParams` so a wire request carrying a blank term is
        rejected with a **422** (see :func:`_validate_phrase_hints_list`).

        Args:
            value: The provided phrase-hint list, ``[]``, or ``None``.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If any term is empty or whitespace-only.
        """
        return _validate_phrase_hints_list(value)

    def to_runtime_params(self) -> RuntimeParams:
        """Build the internal :class:`RuntimeParams` from the validated wire set.

        ``provider_params`` is necessarily ``None`` (it cannot be sent), so the
        resulting params carry only the portable, already-validated fields.

        Returns:
            The equivalent internal :class:`RuntimeParams`.
        """
        return RuntimeParams.model_validate(self.model_dump())


# D5 drift guard: the wire view is exactly the portable set, i.e. RuntimeParams
# minus the discover-only ``provider_params`` escape hatch. Defining the two
# field sets independently risks them desyncing as the portable set evolves; this
# import-time invariant (and a drift test) makes such a desync a hard failure.
assert set(WireRuntimeParams.model_fields) == (
    set(RuntimeParams.model_fields) - {"provider_params"}
), "WireRuntimeParams desynced from the portable RuntimeParams field set"


__all__ = [
    "DIARIZE",
    "DiarizationRequest",
    "ProviderParams",
    "RuntimeParams",
    "WireRuntimeParams",
    "WordTimestampGranularity",
]
