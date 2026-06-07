# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical capability system for Standard ASR engines.

Engines declare what they support with a single hierarchical tree grouped by
mode domain (``batch`` / ``streaming``), plus engine-global orthogonal flags
(``streaming_input`` / ``streaming_output``). This module implements the
normative capability model (spec, section "Capabilities").

Two layers exist:

* :class:`DeclaredCapabilities` -- the static, class-level (``ClassVar``) full
  capability set, discoverable without instantiating or authenticating the
  engine. Used by ``models show``, the registry, UI generation and REST.
* ``effective_capabilities`` -- an instance-level subset that may *narrow* the
  declared set based on runtime configuration. The invariant
  ``effective ⊆ declared`` is enforced by compliance tests (see
  :meth:`DeclaredCapabilities.covers`).

Every leaf node is one of three archetypes -- **flag**, **bounded**, or
**enum/mode** -- and all expose a uniform ``is_supported`` boolean so that
strict / best_effort gating is consistent across the tree. Applications query
capabilities exclusively through :meth:`DeclaredCapabilities.supports` with a
dot-path; missing keys are *fail-closed* (return ``False``).
"""

from __future__ import annotations

from typing import Any, Iterator, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

WordTimestampGranularityName = Literal["word", "segment", "char"]

#: Mode values that count as "not supported" for enum/mode archetype nodes.
_UNSUPPORTED_MODES = frozenset({"none", "unsupported"})


class _CapNode(BaseModel):
    """Base class for all capability leaf nodes.

    Subclasses MUST expose an ``is_supported`` boolean property derived from
    their archetype (flag/bounded -> ``supported``; enum/mode -> ``mode``).
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    @property
    def is_supported(self) -> bool:  # pragma: no cover - overridden
        """Whether this capability is supported.

        Returns:
            ``True`` if the engine supports the capability.
        """
        raise NotImplementedError


class _FlagLikeNode(_CapNode):
    """Archetype base for flag and bounded nodes (carry a ``supported`` bool)."""

    supported: bool = False

    @property
    def is_supported(self) -> bool:
        """Whether this capability is supported (the ``supported`` field).

        Returns:
            The value of ``supported``.
        """
        return self.supported


def _mode_supported(mode: str) -> bool:
    """Derive ``is_supported`` for an enum/mode node from its ``mode``.

    Args:
        mode: The node's mode value.

    Returns:
        ``True`` unless ``mode`` is ``"none"`` or ``"unsupported"``.
    """
    return mode not in _UNSUPPORTED_MODES


# --------------------------------------------------------------------------- #
# Constraint submodels (machine-checkable limits, live with their feature).
# --------------------------------------------------------------------------- #
class CandidateLanguagesConstraints(BaseModel):
    """Constraints for the candidate-languages capability.

    Args:
        max: Maximum number of candidate languages accepted.
    """

    model_config = ConfigDict(frozen=True, extra="allow")
    max: int = Field(..., gt=0, description="Maximum number of candidate languages.")


class PromptConstraints(BaseModel):
    """Constraints for the prompt guidance channel.

    Args:
        max_tokens: Optional maximum prompt length in tokens.
    """

    model_config = ConfigDict(frozen=True, extra="allow")
    max_tokens: int | None = Field(default=None, description="Maximum prompt tokens.")


class PhraseHintsConstraints(BaseModel):
    """Constraints for the phrase-hints guidance channel.

    Args:
        max_terms: Optional maximum number of phrase-hint terms.
        max_chars_per_term: Optional maximum characters per term.
        max_words_per_term: Optional maximum words per term.
    """

    model_config = ConfigDict(frozen=True, extra="allow")
    max_terms: int | None = Field(default=None, description="Maximum hint terms.")
    max_chars_per_term: int | None = Field(
        default=None, description="Maximum characters per term."
    )
    max_words_per_term: int | None = Field(
        default=None, description="Maximum words per term."
    )


class DiarizationConstraints(BaseModel):
    """Constraints for the diarization capability.

    Args:
        max_speakers: Optional maximum number of speakers.
    """

    model_config = ConfigDict(frozen=True, extra="allow")
    max_speakers: int | None = Field(default=None, description="Maximum speakers.")


# --------------------------------------------------------------------------- #
# Leaf capability nodes.
# --------------------------------------------------------------------------- #
class FlagCap(_FlagLikeNode):
    """A simple supported / not-supported flag."""


class CandidateLanguagesCap(_FlagLikeNode):
    """Bounded capability for candidate languages.

    Args:
        supported: Whether candidate languages are supported.
        constraints: Limits (e.g. ``max``) when supported.
    """

    constraints: CandidateLanguagesConstraints | None = None


class WordTimestampsCap(_FlagLikeNode):
    """Capability for word-level timestamps.

    Args:
        supported: Whether word timestamps are supported.
        granularities: Supported granularities (``word``/``segment``/``char``).
    """

    granularities: list[WordTimestampGranularityName] = Field(
        default_factory=lambda: cast("list[WordTimestampGranularityName]", [])
    )


class PromptCap(_FlagLikeNode):
    """Guidance channel: free-text prompt.

    Args:
        supported: Whether prompt guidance is supported.
        constraints: Limits when supported.
    """

    constraints: PromptConstraints = Field(default_factory=PromptConstraints)


class PhraseHintsCap(_FlagLikeNode):
    """Guidance channel: phrase-hint term boosting.

    Args:
        supported: Whether phrase hints are supported.
        constraints: Limits when supported.
    """

    constraints: PhraseHintsConstraints = Field(default_factory=PhraseHintsConstraints)


class DiarizationCap(_FlagLikeNode):
    """Capability for speaker diarization (request path deferred in v1).

    Args:
        supported: Whether diarization is supported.
        constraints: Limits when supported.
    """

    constraints: DiarizationConstraints = Field(default_factory=DiarizationConstraints)


class ReconnectCap(_CapNode):
    """Streaming reconnect capability.

    Args:
        mode: ``seamless`` / ``lossy`` / ``unsupported``.
    """

    mode: Literal["seamless", "lossy", "unsupported"] = "unsupported"

    @property
    def is_supported(self) -> bool:
        """Whether reconnect is supported.

        Returns:
            ``True`` unless ``mode`` is ``"unsupported"``.
        """
        return _mode_supported(self.mode)


class FinalityCap(_CapNode):
    """Streaming finality level the engine can guarantee.

    Args:
        mode: ``final`` (may still be revised by post-processing) or ``closed``.
    """

    mode: Literal["final", "closed"] = "final"

    @property
    def is_supported(self) -> bool:
        """Whether a finality level is guaranteed (always ``True`` here).

        Returns:
            ``True`` (both ``final`` and ``closed`` are supported levels).
        """
        return _mode_supported(self.mode)


class StreamTimestampsCap(_CapNode):
    """Source of streaming timestamps.

    Args:
        mode: ``native_frame_aligned`` / ``post_align`` / ``none``.
    """

    mode: Literal["native_frame_aligned", "post_align", "none"] = "none"

    @property
    def is_supported(self) -> bool:
        """Whether streaming timestamps are provided.

        Returns:
            ``True`` unless ``mode`` is ``"none"``.
        """
        return _mode_supported(self.mode)


# --------------------------------------------------------------------------- #
# Container nodes (group leaves; not capabilities themselves).
# --------------------------------------------------------------------------- #
class _Container(BaseModel):
    """Base for grouping containers; tolerant of unknown / ``x_*`` keys."""

    model_config = ConfigDict(frozen=True, extra="allow")


class LanguageCaps(_Container):
    """Language capabilities for one mode.

    Args:
        runtime_override: Whether per-request language override is allowed.
        candidate_languages: Candidate-language support and limits.
    """

    runtime_override: FlagCap = Field(default_factory=FlagCap)
    candidate_languages: CandidateLanguagesCap = Field(
        default_factory=CandidateLanguagesCap
    )


class GuidanceCaps(_Container):
    """Guidance-family capabilities for one mode.

    Args:
        prompt: Free-text prompt channel.
        phrase_hints: Phrase-hint channel.
    """

    prompt: PromptCap = Field(default_factory=PromptCap)
    phrase_hints: PhraseHintsCap = Field(default_factory=PhraseHintsCap)


class BatchCapabilities(_Container):
    """Capability tree for the ``batch`` mode domain.

    Args:
        language: Language capabilities.
        word_timestamps: Word-timestamp capability.
        guidance: Guidance-family capabilities.
        diarization: Diarization capability.
    """

    language: LanguageCaps = Field(default_factory=LanguageCaps)
    word_timestamps: WordTimestampsCap = Field(default_factory=WordTimestampsCap)
    guidance: GuidanceCaps = Field(default_factory=GuidanceCaps)
    diarization: DiarizationCap = Field(default_factory=DiarizationCap)


class StreamingCapabilities(_Container):
    """Capability tree for the ``streaming`` mode domain.

    Args:
        language: Language capabilities (MAY differ from batch).
        word_timestamps: Word-timestamp capability.
        guidance: Guidance-family capabilities (MAY differ from batch).
        emits_partials: Whether partial events are emitted.
        re_segments: Whether supersede events may occur.
        word_stability: Whether a meaningful ``stable_until`` is provided.
        reconnect: Reconnect capability mode.
        finality_level: Finality level guaranteed.
        timestamps: Source of streaming timestamps.
    """

    language: LanguageCaps = Field(default_factory=LanguageCaps)
    word_timestamps: WordTimestampsCap = Field(default_factory=WordTimestampsCap)
    guidance: GuidanceCaps = Field(default_factory=GuidanceCaps)
    emits_partials: FlagCap = Field(default_factory=FlagCap)
    re_segments: FlagCap = Field(default_factory=FlagCap)
    word_stability: FlagCap = Field(default_factory=FlagCap)
    reconnect: ReconnectCap = Field(default_factory=ReconnectCap)
    finality_level: FinalityCap = Field(default_factory=FinalityCap)
    timestamps: StreamTimestampsCap = Field(default_factory=StreamTimestampsCap)


class DeclaredCapabilities(_Container):
    """The full capability tree declared by an engine.

    Mode domains are optional: omitting a domain means the mode is not supported
    (fail-closed). Engine-global orthogonal flags live at the top level.

    Args:
        batch: Batch-mode capabilities, or ``None`` if batch is unsupported.
        streaming: Streaming-mode capabilities, or ``None`` if unsupported.
        streaming_input: Whether the engine accepts incremental audio.
        streaming_output: Whether the engine returns results incrementally.
    """

    batch: BatchCapabilities | None = None
    streaming: StreamingCapabilities | None = None
    streaming_input: FlagCap = Field(default_factory=FlagCap)
    streaming_output: FlagCap = Field(default_factory=FlagCap)

    def supports(self, dot_path: str) -> bool:
        """Return whether the capability at ``dot_path`` is supported.

        The only standard way to query capabilities. Walks the tree segment by
        segment; any missing segment returns ``False`` (fail-closed). Resolving
        a present mode-domain or container also returns ``True``.

        Args:
            dot_path: Dotted capability path without the ``capabilities.``
                prefix (e.g. ``"batch.word_timestamps"``,
                ``"streaming.guidance.phrase_hints"``, ``"streaming_input"``).

        Returns:
            ``True`` if supported, otherwise ``False``.
        """
        node: object = self
        for part in dot_path.split("."):
            node = _get_child(node, part)
            if node is None:
                return False
        return _derive_supported(node)

    def node_at(self, dot_path: str) -> _CapNode | None:
        """Return the typed capability *node* at ``dot_path``, or ``None``.

        Unlike :meth:`supports` (which returns a bool), this returns the leaf
        node object itself so callers can inspect its constraints / enums (e.g.
        a ``WordTimestampsCap`` to validate a requested granularity against
        :attr:`WordTimestampsCap.granularities`). Returns ``None`` if the path
        is absent or does not resolve to a capability leaf node.

        Args:
            dot_path: Dotted capability path without the ``capabilities.``
                prefix (e.g. ``"batch.word_timestamps"``).

        Returns:
            The capability leaf node, or ``None``.
        """
        node = self._resolve(dot_path)
        return node if isinstance(node, _CapNode) else None

    def iter_supported_paths(self) -> Iterator[str]:
        """Yield every dot-path in the tree whose node is supported.

        Only the children of a *supported* node are descended into, so an
        unsupported feature's constraint sub-containers (which are always
        present, never ``None``) do not appear. Used to verify the
        ``effective ⊆ declared`` invariant.

        Yields:
            Dot-paths of supported capability nodes and present containers.
        """
        yield from _iter_paths(self, prefix="")

    def covers(self, other: DeclaredCapabilities) -> bool:
        """Return whether ``other`` is a valid narrowing of this tree.

        Enforces the normative ``effective ⊆ declared`` invariant (spec §C):
        the effective set may only *close* declared capabilities, never widen
        them. This checks two things:

        * **Set containment** -- every supported path in ``other`` is also
          supported here (no feature is enabled that this tree did not declare).
        * **Constraint narrowing** -- where both trees support a bounded or
          enum/mode node, ``other``'s limits MUST be no looser than this tree's
          (e.g. a smaller-or-equal ``max``, a subset of ``granularities``, a
          ``mode`` that is the same or a reduction). A widening (declared
          ``max=2`` -> effective ``max=999``) is rejected.

        Args:
            other: A (typically narrowed, effective) capability tree.

        Returns:
            ``True`` if ``other`` is a subset narrowing of this tree.
        """
        mine = set(self.iter_supported_paths())
        for path in other.iter_supported_paths():
            if path not in mine:
                return False
        # Where both support a node, the effective node must not be looser.
        for path in other.iter_supported_paths():
            declared_node = self._resolve(path)
            effective_node = other._resolve(path)
            if declared_node is None or effective_node is None:
                continue
            if not _node_narrows(declared_node, effective_node):
                return False
        return True

    def _resolve(self, dot_path: str) -> object:
        """Resolve a dot-path to its node object (not its ``supported`` bool).

        Args:
            dot_path: Dotted capability path.

        Returns:
            The resolved node, or ``None`` if any segment is absent.
        """
        node: object = self
        for part in dot_path.split("."):
            node = _get_child(node, part)
            if node is None:
                return None
        return node


def _get_child(node: object, part: str) -> object:
    """Resolve a single path segment on a model or dict node.

    Args:
        node: A pydantic model or dict to descend into.
        part: The path segment.

    Returns:
        The child node, or ``None`` if absent.
    """
    if isinstance(node, BaseModel):
        if part in type(node).model_fields:
            return getattr(node, part)
        extra: dict[str, Any] = node.model_extra or {}
        return extra.get(part)
    if isinstance(node, dict):
        return cast("dict[str, object]", node).get(part)
    return None


def _derive_supported(node: object) -> bool:
    """Derive the ``is_supported`` boolean for a resolved node.

    Args:
        node: A capability leaf, container, or raw dict/value.

    Returns:
        ``True`` if the node represents a supported capability or a present
        container.
    """
    if isinstance(node, _CapNode):
        return node.is_supported
    if isinstance(node, BaseModel):
        # A present container (mode domain or grouping) counts as supported.
        return True
    if isinstance(node, dict):
        mapping = cast("dict[str, object]", node)
        if "mode" in mapping:
            return mapping["mode"] not in _UNSUPPORTED_MODES
        if "supported" in mapping:
            return bool(mapping["supported"])
        return True  # present container dict
    return False


def _iter_paths(node: object, prefix: str) -> Iterator[str]:
    """Recursively yield supported dot-paths under ``node``.

    A node's children are only descended into when the node itself is
    supported. This prevents an *unsupported* leaf's constraint sub-containers
    (which are always-present default-factory models, never ``None``, and thus
    would otherwise read as "supported present containers") from polluting the
    set used for the ``effective ⊆ declared`` comparison.

    Args:
        node: A pydantic model or dict to walk.
        prefix: The accumulated dot-path prefix.

    Yields:
        Supported dot-paths.
    """
    for name, child in _children(node):
        if child is None:
            continue
        path = f"{prefix}.{name}" if prefix else name
        supported = _derive_supported(child)
        if supported:
            yield path
        # Only descend into a supported node. An unsupported leaf has no
        # meaningful supported children (its constraints are inert).
        if supported:
            yield from _iter_paths(child, path)


def _children(node: object) -> list[tuple[str, object]]:
    """Return ``(name, child)`` pairs for a model or dict node.

    Args:
        node: A pydantic model or dict.

    Returns:
        A list of named children (declared fields plus extras).
    """
    if isinstance(node, BaseModel):
        items: list[tuple[str, object]] = [
            (name, getattr(node, name)) for name in type(node).model_fields
        ]
        extra: dict[str, Any] = node.model_extra or {}
        items.extend(extra.items())
        return items
    if isinstance(node, dict):
        return list(cast("dict[str, object]", node).items())
    return []


#: Constraint fields whose semantics are an *upper bound* (effective ≤ declared).
_MAX_CONSTRAINT_FIELDS = frozenset(
    {"max", "max_tokens", "max_terms", "max_chars_per_term", "max_words_per_term", "max_speakers"}
)

#: enum/mode reductions: declared mode -> the set of modes that are no looser.
#: A mapping value is the set of effective modes accepted for that declared mode
#: (always includes the declared mode itself plus any strictly-weaker mode).
_MODE_REDUCTIONS: dict[str, frozenset[str]] = {
    # reconnect: seamless is strongest; lossy is weaker; unsupported is off.
    "seamless": frozenset({"seamless", "lossy", "unsupported"}),
    "lossy": frozenset({"lossy", "unsupported"}),
    "unsupported": frozenset({"unsupported"}),
    # timestamps: native_frame_aligned strongest; post_align weaker; none off.
    "native_frame_aligned": frozenset({"native_frame_aligned", "post_align", "none"}),
    "post_align": frozenset({"post_align", "none"}),
    "none": frozenset({"none"}),
    # finality_level: closed is the stronger guarantee; final is weaker.
    "closed": frozenset({"closed", "final"}),
    "final": frozenset({"final"}),
}


def _node_narrows(declared: object, effective: object) -> bool:
    """Return whether ``effective`` is no looser than ``declared`` for one node.

    Implements the per-node half of the ``effective ⊆ declared`` invariant for
    bounded (``constraints``) and enum/mode nodes. Flag-only nodes always pass
    (set containment already covered them).

    Args:
        declared: The declared node (or sub-value).
        effective: The corresponding effective node (or sub-value).

    Returns:
        ``True`` if ``effective`` does not widen ``declared``.
    """
    # enum/mode nodes: the effective mode must be a reduction of the declared.
    declared_mode = _read_attr(declared, "mode")
    effective_mode = _read_attr(effective, "mode")
    if isinstance(declared_mode, str) and isinstance(effective_mode, str):
        allowed = _MODE_REDUCTIONS.get(declared_mode)
        if allowed is not None and effective_mode not in allowed:
            return False

    # granularities: effective set MUST be a subset of declared set.
    declared_grans = _read_attr(declared, "granularities")
    effective_grans = _read_attr(effective, "granularities")
    if isinstance(declared_grans, list) and isinstance(effective_grans, list):
        if not set(cast("list[object]", effective_grans)).issubset(
            set(cast("list[object]", declared_grans))
        ):
            return False

    # bounded constraints: each numeric upper-bound MUST NOT increase.
    declared_c = _read_attr(declared, "constraints")
    effective_c = _read_attr(effective, "constraints")
    if declared_c is not None and effective_c is not None:
        for field in _MAX_CONSTRAINT_FIELDS:
            d_val = _read_attr(declared_c, field)
            e_val = _read_attr(effective_c, field)
            # A declared finite bound must not be loosened. An effective bound
            # may not appear where the declared one was unbounded (None) and
            # then claim a value -- that is also a widening of an open bound.
            if isinstance(d_val, int) and isinstance(e_val, int):
                if e_val > d_val:
                    return False
            elif d_val is not None and e_val is None:
                # Declared bounded, effective claims unbounded -> widening.
                return False
    return True


def _read_attr(node: object, name: str) -> object:
    """Read ``name`` from a model (field or extra) or dict, else ``None``.

    Args:
        node: A pydantic model, dict, or other value.
        name: The attribute / key name.

    Returns:
        The value, or ``None`` if absent.
    """
    if isinstance(node, BaseModel):
        if name in type(node).model_fields:
            return getattr(node, name)
        extra: dict[str, Any] = node.model_extra or {}
        return extra.get(name)
    if isinstance(node, dict):
        return cast("dict[str, object]", node).get(name)
    return None


__all__ = [
    "BatchCapabilities",
    "CandidateLanguagesCap",
    "CandidateLanguagesConstraints",
    "DeclaredCapabilities",
    "DiarizationCap",
    "DiarizationConstraints",
    "FinalityCap",
    "FlagCap",
    "GuidanceCaps",
    "LanguageCaps",
    "PhraseHintsCap",
    "PhraseHintsConstraints",
    "PromptCap",
    "PromptConstraints",
    "ReconnectCap",
    "StreamTimestampsCap",
    "StreamingCapabilities",
    "WordTimestampGranularityName",
    "WordTimestampsCap",
]
