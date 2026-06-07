# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-request runtime parameters (closed portable set + escape hatch).

:class:`RuntimeParams` is the **closed** container of per-request settings
(spec, section "Runtime Parameters"). It carries the v1 portable standard set
(``language``, ``candidate_languages``, ``word_timestamps``, and the
``guidance`` family ``prompt`` / ``phrase_hints``) plus a single typed escape
hatch, ``provider_params``, for engine-specific knobs. ASR authors MUST NOT add
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
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WordTimestampGranularity(str, Enum):
    """Granularity for requested word timestamps.

    Attributes:
        WORD: Word-level timestamps.
        SEGMENT: Segment-level timestamps.
        CHAR: Character-level timestamps (reserved, additive).
    """

    WORD = "word"
    SEGMENT = "segment"
    CHAR = "char"


class ProviderParams(BaseModel):
    """Base class for an engine's typed, non-portable parameter model.

    Engines publish a subclass (e.g. ``OpenAIParams``) and declare it as their
    expected ``provider_params`` type. Passing one engine's params model to a
    different engine is a validation error (swap-safe), raised as
    :class:`~standard_asr.exceptions.InvalidProviderParamError` by the engine
    layer regardless of the strict / best_effort policy.
    """

    # Engine params subclasses may carry `model_*` knobs; opt out of pydantic's
    # `model_` protected namespace so they do not warn (the warning fires on
    # older pydantic, e.g. the lower-bounds 2.5).
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class RuntimeParams(BaseModel):
    """Closed per-request parameter container.

    Args:
        language: Per-request language (BCP-47 or ``"auto"``) overriding the
            engine default. Gated by ``<mode>.language.runtime_override``.
        candidate_languages: Candidate languages, meaningful only in ``auto``
            mode. Gated by ``<mode>.language.candidate_languages``.
        word_timestamps: Requested word-timestamp granularity. Gated by
            ``<mode>.word_timestamps``.
        prompt: Free-text guidance prompt. Gated by ``<mode>.guidance.prompt``.
        phrase_hints: Phrase-hint boost terms. Gated by
            ``<mode>.guidance.phrase_hints``.
        on_unsupported: Guidance degradation policy. ``"fail"`` (default) keeps
            the fail-closed contract; ``"degrade_to_prompt"`` opts into the
            one-way rich->prompt fallback (a diagnostic is emitted on degrade).
        provider_params: Engine-specific typed knobs, or ``None``.

    Returns:
        None.

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
    prompt: str | None = Field(default=None, description="Free-text guidance prompt.")
    phrase_hints: list[str] | None = Field(default=None, description="Phrase-hint boost terms.")
    on_unsupported: Literal["fail", "degrade_to_prompt"] = Field(
        default="fail",
        description="Guidance degradation policy (opt-in one-way to prompt).",
    )
    provider_params: ProviderParams | None = Field(
        default=None, description="Engine-specific typed knobs."
    )


__all__ = [
    "ProviderParams",
    "RuntimeParams",
    "WordTimestampGranularity",
]
