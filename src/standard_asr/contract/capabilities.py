# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical capability system for Standard ASR engines.

Engines declare what they support with a single hierarchical tree grouped by
mode domain (``batch`` / ``streaming``), plus engine-global orthogonal flags
(``streaming_input`` / ``streaming_output``). This module implements the
normative capability model (spec, section "Capabilities").

:class:`Capabilities` is the capability tree, and the same class types both
layers:

* ``declared_capabilities`` -- the class-level (``ClassVar``) instance: the
  full capability set, discoverable without instantiating or authenticating
  the engine. Used by ``show``, the registry, UI generation and REST.
* ``effective_capabilities`` -- the configured engine instance's tree, a
  subset that may *narrow* the declared set based on runtime configuration
  (``EngineBase`` returns the declared set unless the engine overrides it).
  The invariant ``effective ⊆ declared`` is enforced by compliance tests
  (see :meth:`Capabilities.covers`).

The tree holds three sorts of object. A *capability node*
(``_CapabilityNode``) states ``supported`` or ``mode``, and it may
still carry children such as ``diarization.always_on`` or ``constraints``.
A ``constraints`` *submodel* (``_ConstraintNode``) holds the
machine-checkable bounds of one requestable feature; it is not a capability
node. A *container* (``_Container``) only groups nodes: the root, the
mode domains ``batch`` and ``streaming``, and groups such as ``language``.

Every capability node has one of three shapes -- **flag**, **bounded**, or
**enum/mode** -- and all expose a uniform ``is_supported`` boolean so that
strict / best_effort gating is consistent across the tree.

Every typed capability node, one built from a class of this module, also
states its role (:data:`CapabilityRole`) in a ``role`` field whose value
its class fixes; an engine does not set it. The role says what the node is
for, and it is a separate axis from the shape. ``requestable`` means an
application can ask for the feature in a request through one
``RuntimeParams`` field, such as ``diarization`` or ``word_timestamps``.
``trait`` means the node states a property, a behavior, or a promise of
the engine that no standard ``RuntimeParams`` field requests, such as
``streaming_input`` (the engine accepts incremental audio),
``self_resamples`` (it resamples audio internally),
``streaming.emits_partials`` (it sends interim results),
``streaming.finality_level`` (a promise about ``final`` events), or
``diarization.always_on`` (it applies diarization to every request).

Each ``constraints`` submodel states the role ``constraints``. Under each
mode, the standard bounds are ``language.candidate_languages.constraints.max``
(how many candidate languages a request may list);
``guidance.prompt.constraints.max_tokens`` (the most tokens a prompt may
have, checked against an estimated count); ``max_terms``,
``max_chars_per_term``, and ``max_words_per_term`` in
``guidance.phrase_hints.constraints``; and
``diarization.constraints.max_speakers``. A container states no role. A raw
``x_*`` node, a plain dict under a vendor's ``x_<vendor>_<feature>`` key,
states a role only when its declaration writes one.

Applications query the tree with a dot-path, and a missing key is
*fail-closed*. :meth:`Capabilities.supports` answers whether the
engine has a feature. :meth:`Capabilities.can_request` answers
whether a request can ask for it, which is ``True`` only for a supported
``requestable`` node. :meth:`Capabilities.role_at` returns a node's
role.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from standard_asr.contract._json_extra import JsonExtraModel

WordTimestampGranularityName = Literal["word", "segment", "char"]

#: The closed set of mode-domain names (spec: the capability tree's top-level
#: partitions). Homed here -- the module that DEFINES the mode domains -- so
#: contract-layer signatures (for example, ``effective_candidate_languages``'s
#: ``mode``) can use the precise type without importing upward from
#: ``runtime.gating`` (which re-exports it as ``Mode``).
ModeName = Literal["batch", "streaming"]

#: Mode values that count as "not supported" for nodes of the enum/mode shape.
_UNSUPPORTED_MODES = frozenset({"none", "unsupported"})

#: What a node is for: the ``role`` field of every typed capability node and
#: constraint submodel. A container has none, and a raw ``x_*`` node has one
#: only when its declaration writes it. The role is a separate axis from a
#: capability node's shape (flag, bounded, or enum/mode).
#: ``requestable``: an application can ask for the feature in a request; a
#: standard node is requested through one ``RuntimeParams`` field, as
#: ``diarization`` and ``word_timestamps`` are.
#: ``trait``: a property, a behavior, or a promise of the engine that no
#: standard ``RuntimeParams`` field requests, such as ``streaming_input`` or
#: ``streaming.finality_level``. A supported trait node states a fact and
#: grants nothing a request can ask for.
#: ``constraints``: the machine-checkable bounds of one requestable feature,
#: held in its ``constraints`` submodel, such as
#: ``guidance.phrase_hints.constraints.max_terms``.
#: There is no ``observation`` role: a declaration is static, so a fact that
#: changes with the host machine lives outside the capability tree. Each
#: node class fixes one value; this alias states the domain only.
CapabilityRole = Literal["requestable", "trait", "constraints"]

#: Reserved prefix for experimental extension capabilities
#: (``x_<vendor>_<feature>``). An *extra* (non-field) key on a typed standard node
#: is a recognized capability only when it carries this prefix.
_EXTENSION_PREFIX = "x_"


def _is_extension_key(key: object) -> bool:
    """Return whether an extra key on a typed node is an ``x_*`` extension.

    Typed capability containers parse with ``extra="allow"`` so an unknown key
    (a future standard field, or a typo) does not fail validation -- forward
    compatibility ("tolerate unknown keys"). But only the reserved
    ``x_<vendor>_<feature>`` namespace is a real, queryable
    capability. Every other unknown key MUST be fail-closed when probed via
    :meth:`Capabilities.supports` / excluded from
    :meth:`Capabilities.iter_supported_paths`, so a typo'd path segment
    (for example, ``"word_timestmaps"``) never reads as a supported capability and
    weakens the gating contract. Serialization (:meth:`to_protocol_dict`) is
    separate and still round-trips every extra.

    Args:
        key: A model-extra key.

    Returns:
        ``True`` if ``key`` is a string in the ``x_`` extension namespace.
    """
    return isinstance(key, str) and key.startswith(_EXTENSION_PREFIX)


def _reject_separator_keys_on_node_surface(extras: Mapping[str, object]) -> None:
    """Reject a queryable-surface key that embeds the dot-path separator.

    The dot-path grammar is the protocol's ONE query surface:
    :meth:`Capabilities.supports`, ``iter_queryable_paths`` /
    ``iter_supported_paths``, ``covers`` and the compliance sweep all join
    and split node names on ``"."``. A node key containing the separator
    breaks that bijection both ways: the joined path can never resolve
    (``supports`` splits it into segments the tree does not have, so the
    sweep would fail a compliant plugin whose hand-written ``supports``
    honestly answers ``True`` for its own declared vendor capability), and
    two DISTINCT trees (``{"a.b": node}`` vs ``{"a": {"b": node}}``) join
    to the SAME path string, letting ``covers``'s set containment conflate
    them. A dotted key is legal JSON, so it is rejected HERE, loudly,
    naming the key -- never silently mis-resolved later.

    Only the node-traversal surface is constrained -- exactly the keys
    :func:`_children` / :func:`_get_child` walk: ``x_*`` extension keys and
    every key of a dict reachable by dict nesting from one. Dicts inside
    lists are field internals (never nodes), and non-extension extras are
    not queryable; their keys stay free, so value data keyed by a string such
    as ``"v1.2"`` remains representable outside the path space.

    Args:
        extras: A model's extra keys (values already key-vetted as exact
            ``str`` at every depth by ``require_json_string_keys``).

    Raises:
        ValueError: If a queryable-surface key contains ``"."``.
    """
    stack: list[tuple[str, object]] = [
        (key, value) for key, value in extras.items() if _is_extension_key(key)
    ]
    while stack:
        key, value = stack.pop()
        if "." in key:
            raise ValueError(
                f"capability key {key!r} contains '.', the dot-path separator: "
                "its path could never resolve via supports() and would collide "
                "with a genuinely nested spelling. Rename the key (for example, use "
                "'_'), or move dotted-name data into a list or scalar value, "
                "which is outside the queryable path space."
            )
        if isinstance(value, dict):
            stack.extend(cast("dict[str, object]", value).items())


def granularity_offers_all(granularities: Sequence[str]) -> bool:
    """Return whether a ``granularities`` list means every granularity.

    On a bounded node, an empty enumeration means that the engine did not
    enumerate, so the node offers every granularity. It does not mean that
    the node offers none. :meth:`Capabilities.covers` reads ``granularities``
    by the same rule, on the declared node and on the effective node, and
    also reads an omitted or ``None`` field as every granularity.

    The validator of :class:`WordTimestampsCapability` rejects building a
    supported node with an empty list. The list stays mutable after the node
    is built, though, and :meth:`Capabilities.covers` reads an emptied list
    as every granularity too. A raw ``x_*`` node has no validator and can
    state an empty list directly.

    Runtime parameter gating does not use this rule.
    ``runtime.gating._gate_granularity`` accepts a requested granularity
    only when a :class:`WordTimestampsCapability` lists it, so an emptied
    list accepts none.

    Args:
        granularities: A ``granularities`` list, possibly empty.

    Returns:
        ``True`` if the list is empty, which means every granularity.
    """
    return not granularities


class _JsonExtraModel(JsonExtraModel):
    """Apply capability dot-path rules to the shared JSON-extra base."""

    # ``to_protocol_dict()`` writes every field, defaults included, so the
    # serialization-mode JSON Schema can list every field as required and
    # show ``role`` as a required ``const``. A validation-mode schema leaves
    # a defaulted field optional. The schema covers fields only. The
    # ``supported`` key that ``to_protocol_dict()`` adds to every present
    # container and mode domain, and to every enum/mode node such as
    # ``FinalityCapability``, is not a field of any class, so the schema lacks it.
    # A complete schema of the capability endpoint is item 29 of the
    # roadmap (issue #27).
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    @classmethod
    def _validate_extra_keys(cls, extras: Mapping[str, object]) -> None:
        """Reject separators in queryable capability-extension keys.

        Args:
            extras: Unknown capability fields with validated string keys.

        Returns:
            None.
        """
        _reject_separator_keys_on_node_surface(extras)


class _CapabilityNode(_JsonExtraModel):
    """Base class for all capability nodes.

    A capability node may still carry children, as ``diarization`` carries
    ``always_on`` and ``constraints``. Subclasses MUST expose an
    ``is_supported`` boolean property derived from their shape
    (flag/bounded -> ``supported``; enum/mode -> ``mode``). Each concrete
    subclass also fixes ``role`` to ``requestable`` or ``trait`` with a
    single-value ``Literal`` and a default. That makes the wrong role a
    validation error, and it renders as a ``const`` in the generated schema.
    The shared :data:`CapabilityRole` type would accept a valid but wrong
    value, such as a trait node that declares ``requestable``.
    """

    # The shared type lets code that holds any node read ``node.role``.
    # Each subclass narrows it to one value. Pyright reports that narrowing
    # as an incompatible override, because it cannot see that the models are
    # frozen (``frozen=True`` in ``JsonExtraModel``) and so never reassigned.
    # Every narrowed ``role`` field therefore carries an ignore for that one
    # rule.
    role: CapabilityRole

    @property
    def is_supported(self) -> bool:  # pragma: no cover - overridden
        """Whether this capability is supported.

        Returns:
            ``True`` if the engine supports the capability.
        """
        raise NotImplementedError


class _FlagLikeNode(_CapabilityNode):
    """Base for the flag and bounded shapes: nodes that carry a ``supported`` bool."""

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
# Constraint submodels (machine-checkable bounds, live with their feature).
# --------------------------------------------------------------------------- #
class _ConstraintNode(_JsonExtraModel):
    """Base class for the constraint submodels; their ``role`` is ``constraints``.

    A constraint submodel holds the machine-checkable bounds of one
    requestable feature, such as the maximum number of phrase-hint terms. It
    is not a capability node: it has no ``supported``, and
    :meth:`Capabilities.supports` answers ``False`` for its path.
    :meth:`Capabilities.role_at` answers ``"constraints"`` for it.
    """

    role: Literal["constraints"] = "constraints"


class CandidateLanguagesConstraints(_ConstraintNode):
    """Constraints for the candidate-languages capability.

    Attributes:
        max: Maximum number of candidate languages accepted.
    """

    max: int = Field(..., gt=0, description="Maximum number of candidate languages.")


class PromptConstraints(_ConstraintNode):
    """Constraints for the prompt guidance channel.

    Attributes:
        max_tokens: Optional maximum prompt length in tokens. The standard layer
            has no engine tokenizer, so it enforces this bound against a
            **conservative, script-aware approximation** -- whitespace-delimited
            words plus one unit per space-less (CJK, kana, Hangul, Thai, and so on)
            codepoint -- not the engine's exact token count. Honest scope of the
            guarantee: the approximation never under-counts relative to that
            whitespace + no-space-script tokenization, but it MAY under-count an
            engine's subword (BPE) tokenization of long Latin words / URLs /
            digit runs (counted as 1 here, often 6-17 BPE tokens), so such a
            prompt can exceed the engine's true budget despite passing the gate.
            Declare ``max_tokens`` with headroom below the engine's hard limit
            rather than at it; the standard never exceeds the declared value.
    """

    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum prompt tokens. Enforced by the standard against a "
            "conservative, script-aware approximation (whitespace words + "
            "space-less codepoints), not the engine's exact tokenizer."
        ),
    )


class PhraseHintsConstraints(_ConstraintNode):
    """Constraints for the phrase-hints guidance channel.

    Attributes:
        max_terms: Optional maximum number of phrase-hint terms.
        max_chars_per_term: Optional maximum characters per term.
        max_words_per_term: Optional maximum words per term.
    """

    max_terms: int | None = Field(default=None, gt=0, description="Maximum hint terms.")
    max_chars_per_term: int | None = Field(
        default=None, gt=0, description="Maximum characters per term."
    )
    max_words_per_term: int | None = Field(
        default=None, gt=0, description="Maximum words per term."
    )


class DiarizationConstraints(_ConstraintNode):
    """Constraints for the diarization capability.

    Attributes:
        max_speakers: Optional maximum number of speakers.
    """

    max_speakers: int | None = Field(default=None, gt=0, description="Maximum speakers.")


# --------------------------------------------------------------------------- #
# Capability nodes.
# --------------------------------------------------------------------------- #
class TraitFlagCapability(_FlagLikeNode):
    """A supported / not-supported flag for a property, a behavior, or a promise of the engine.

    Its ``role`` is ``trait``: a supported flag tells an application a
    fact about the engine and grants nothing it can request, as with
    ``streaming.emits_partials`` or ``diarization.always_on``. A requestable
    flag has its own class, such as :class:`LanguageOverrideCapability`.
    """

    role: Literal["trait"] = "trait"  # pyright: ignore[reportIncompatibleVariableOverride]


class LanguageOverrideCapability(_FlagLikeNode):
    """Whether a request's ``language`` may override the configured ``default_language``.

    The node at ``<mode>.language.runtime_override``. Its ``role`` is
    ``requestable``, because ``RuntimeParams.language`` is gated on it. It
    has its own class because :class:`TraitFlagCapability` fixes ``trait``, so a
    ``TraitFlagCapability`` in this field fails validation.
    """

    role: Literal["requestable"] = "requestable"  # pyright: ignore[reportIncompatibleVariableOverride]


class CandidateLanguagesCapability(_FlagLikeNode):
    """Bounded capability for candidate languages.

    Attributes:
        supported: Whether candidate languages are supported.
        constraints: Limits (for example, ``max``) when supported.
    """

    role: Literal["requestable"] = "requestable"  # pyright: ignore[reportIncompatibleVariableOverride]
    constraints: CandidateLanguagesConstraints | None = None


class WordTimestampsCapability(_FlagLikeNode):
    """Capability for word-level timestamps.

    A supported word-timestamp capability MUST enumerate at least one
    granularity: an engine that declares ``supported=True`` but lists no
    granularities is ambiguous -- gating could not tell whether a
    requested granularity is offered, and silently honoring an unlisted one is
    the cardinal sin. The validator therefore rejects building a node in the
    "supported but unenumerated" state, so gating validates a requested
    granularity against the listed set. When ``supported=False`` the list is
    ignored (it defaults to empty and carries no meaning).

    Attributes:
        supported: Whether word timestamps are supported.
        granularities: Supported granularities (``word``/``segment``/``char``);
            MUST be non-empty when ``supported`` is ``True``.
    """

    role: Literal["requestable"] = "requestable"  # pyright: ignore[reportIncompatibleVariableOverride]
    granularities: list[WordTimestampGranularityName] = Field(
        default_factory=lambda: cast("list[WordTimestampGranularityName]", [])
    )

    @model_validator(mode="after")
    def _validate_granularities(self) -> WordTimestampsCapability:
        """Reject an unenumerated supported capability or duplicate granularities.

        Returns:
            The validated capability.

        Raises:
            ValueError: If ``supported`` is ``True`` but ``granularities`` is
                empty, or if ``granularities`` contains a duplicate.
        """
        if self.supported and not self.granularities:
            raise ValueError(
                "WordTimestampsCapability.granularities MUST be non-empty when supported=True "
                "(enumerate at least one of 'word'/'segment'/'char')."
            )
        # granularities is an enumerated SET of offered levels; a repeat is a
        # declaration error, rejected like the duplicate guards on the language
        # lists / accepted_sample_rates / wire_encodings.
        if len(set(self.granularities)) != len(self.granularities):
            raise ValueError(
                "WordTimestampsCapability.granularities has duplicate entries: "
                f"{self.granularities}."
            )
        return self


class PromptCapability(_FlagLikeNode):
    """Guidance channel: free-text prompt.

    Attributes:
        supported: Whether prompt guidance is supported.
        constraints: Limits when supported.
    """

    role: Literal["requestable"] = "requestable"  # pyright: ignore[reportIncompatibleVariableOverride]
    constraints: PromptConstraints = Field(default_factory=PromptConstraints)


class PhraseHintsCapability(_FlagLikeNode):
    """Guidance channel: phrase-hint term boosting.

    Attributes:
        supported: Whether phrase hints are supported.
        constraints: Limits when supported.
    """

    role: Literal["requestable"] = "requestable"  # pyright: ignore[reportIncompatibleVariableOverride]
    constraints: PhraseHintsConstraints = Field(default_factory=PhraseHintsConstraints)


class DiarizationCapability(_FlagLikeNode):
    """Capability for speaker diarization (requested via ``RuntimeParams.diarization``).

    The node's ``role`` is ``requestable``: an application asks for speaker
    labels with ``RuntimeParams.diarization``. The ``always_on`` child is a
    :class:`TraitFlagCapability`, whose ``role`` is ``trait``, like ``self_resamples``
    and the streaming flags ``emits_partials``, ``re_segments``, and
    ``partial_stability``. A supported ``always_on`` means the engine
    applies diarization to every request, because its architecture cannot
    DISABLE diarization. Speaker labels may then appear even when the
    application did not ask for them (the exemption to request-gated
    diarization that spec TR.5 names). ``always_on`` grants nothing an application
    can request.

    ``always_on`` is a **regular queryable flag node**, uniform with
    ``self_resamples``: ``supports("<mode>.diarization.always_on")``
    works, it appears in :meth:`~Capabilities.iter_supported_paths` when
    supported, :meth:`~Capabilities.to_protocol_dict` injects a uniform
    ``supported`` boolean for it, and :meth:`~Capabilities.covers` treats
    a declared-unsupported -> effective-supported change as a rejected widening
    (declaration drift) by plain set containment.

    It is NOT placed inside ``constraints`` (constraints are machine-checkable
    request bounds, and ``always_on`` is a trait, not a bound). It is
    reserved for architecturally non-disableable engines: an engine that CAN
    disable diarization MUST disable it when diarization is not requested
    (can-disable-must-disable), and MUST NOT declare ``always_on`` for adapter
    convenience.

    Attributes:
        supported: Whether diarization is supported.
        always_on: Whether diarization is architecturally non-disableable
            (labels may appear unrequested). May only be supported when
            ``supported`` is ``True``.
        constraints: Limits when supported.
    """

    role: Literal["requestable"] = "requestable"  # pyright: ignore[reportIncompatibleVariableOverride]
    always_on: TraitFlagCapability = Field(
        default_factory=TraitFlagCapability,
        description=(
            "Trait: whether diarization is architecturally "
            "non-disableable. When supported, speaker labels may appear even "
            "when diarization is not requested."
        ),
    )
    constraints: DiarizationConstraints = Field(default_factory=DiarizationConstraints)

    # As a TraitFlagCapability child, ``always_on`` participates in covers() by standard set
    # containment: a declared=false -> effective=true change is auto-rejected as
    # a widening, so no special _node_narrows branch is needed.

    @model_validator(mode="after")
    def _always_on_requires_supported(self) -> DiarizationCapability:
        """Reject the contradictory ``always_on`` supported + ``supported=False`` shape.

        An engine that cannot disable diarization necessarily supports it;
        declaring the pair is a declaration bug, made unrepresentable here so
        the two signals can never disagree.

        Returns:
            The validated capability.

        Raises:
            ValueError: If ``always_on`` is supported while ``supported`` is
                ``False``.
        """
        if self.always_on.is_supported and not self.supported:
            raise ValueError(
                "DiarizationCapability.always_on is declared supported while supported=False "
                "(an engine that cannot disable diarization necessarily supports it)."
            )
        return self


class ReconnectCapability(_CapabilityNode):
    """Streaming reconnect capability.

    Attributes:
        mode: ``seamless`` / ``lossy`` / ``unsupported``.
    """

    role: Literal["trait"] = "trait"  # pyright: ignore[reportIncompatibleVariableOverride]
    mode: Literal["seamless", "lossy", "unsupported"] = "unsupported"

    @property
    def is_supported(self) -> bool:
        """Whether reconnect is supported.

        Returns:
            ``True`` unless ``mode`` is ``"unsupported"``.
        """
        return _mode_supported(self.mode)


class FinalityCapability(_CapabilityNode):
    """Streaming finality level the engine promises.

    The rule lives in the streaming section of the protocol specification,
    section 4.2, under the list of what may still change at each stage.

    Attributes:
        mode: ``final`` or ``closed``. With ``final``, the engine does not
            promise a ``closed`` final for each segment, and it may still
            send one: a post-processing restatement that changes how the
            text is written. With ``closed``, the engine MUST bring every
            segment that reached ``final``, and that no ``supersede``
            retired, to ``closed`` before the session ends with ``done``. A
            session that has the engine's capabilities checks this at
            ``done``, and so does ``check_event_sequence`` when it is given
            them; both report a segment left ``final`` as
            ``finality_level_not_reached``.
    """

    role: Literal["trait"] = "trait"  # pyright: ignore[reportIncompatibleVariableOverride]
    mode: Literal["final", "closed"] = "final"

    @property
    def is_supported(self) -> bool:
        """Whether a finality level is guaranteed (always ``True`` here).

        Returns:
            ``True`` (both ``final`` and ``closed`` are supported levels).
        """
        return _mode_supported(self.mode)


class StreamTimestampsCapability(_CapabilityNode):
    """Source of streaming timestamps.

    Attributes:
        mode: ``native_frame_aligned`` / ``post_align`` / ``none``.
    """

    role: Literal["trait"] = "trait"  # pyright: ignore[reportIncompatibleVariableOverride]
    mode: Literal["native_frame_aligned", "post_align", "none"] = "none"

    @property
    def is_supported(self) -> bool:
        """Whether streaming timestamps are provided.

        Returns:
            ``True`` unless ``mode`` is ``"none"``.
        """
        return _mode_supported(self.mode)


# --------------------------------------------------------------------------- #
# Containers (they group nodes and are not capabilities themselves).
# --------------------------------------------------------------------------- #
class _Container(_JsonExtraModel):
    """Base for grouping containers; tolerant of unknown / ``x_*`` keys, except ``role``."""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse a container subclass that declares a ``role`` field.

        A container groups nodes and states no role (design D4 in
        ``docs/internal/feat_plan/capability-kinds.md``), so ``role_at()``
        answers ``None`` for its path. ``to_protocol_dict()``
        writes every declared field, so a ``role`` field on a subclass would
        publish a role that Python does not report.
        :meth:`_validate_extra_keys` cannot catch the field, because it sees
        only keys that no field declares. This check runs when the class is
        defined, so it catches the field even when it has a default and no
        input ever passes it. Pydantic calls this hook once ``model_fields``
        is complete; ``__init_subclass__`` runs before the fields exist.

        Args:
            kwargs: Keyword arguments from the class definition that
                pydantic does not use itself.

        Raises:
            TypeError: If the subclass declares a ``role`` field.
        """
        super().__pydantic_init_subclass__(**kwargs)
        if "role" in cls.model_fields:
            raise TypeError(
                f"{cls.__name__} declares a 'role' field, but a container groups nodes "
                "and states no role; only a capability node or a constraints "
                "submodel carries role. Remove the field."
            )

    @classmethod
    def _validate_extra_keys(cls, extras: Mapping[str, object]) -> None:
        """Reject an extra ``role`` key on a container, then apply the node-surface rules.

        A container groups nodes and states no role (design D4). This check
        sees only the extras of the input, the keys that no field of the
        class declares. An extra ``role`` would be published by
        ``to_protocol_dict()`` while ``role_at()`` answers ``None`` for the
        same path, so it is rejected at construction. A ``role`` field that
        a subclass declares is not an extra:
        :meth:`__pydantic_init_subclass__` refuses that class when it is
        defined.

        Args:
            extras: Unknown capability fields with validated string keys.

        Raises:
            ValueError: If ``extras`` carries a ``role`` key.
        """
        if "role" in extras:
            raise ValueError(
                f"{cls.__name__} carries a 'role' key, but a container groups nodes "
                "and states no role; only a capability node or a constraints "
                "submodel carries role. Remove the key."
            )
        super()._validate_extra_keys(extras)


class LanguageCapabilities(_Container):
    """Language capabilities for one mode.

    Attributes:
        runtime_override: Whether a request's ``language`` may override the
            configured ``default_language``; requestable (see
            :class:`LanguageOverrideCapability`).
        candidate_languages: Candidate-language support and limits;
            requestable, through ``RuntimeParams.candidate_languages``.
    """

    runtime_override: LanguageOverrideCapability = Field(default_factory=LanguageOverrideCapability)
    candidate_languages: CandidateLanguagesCapability = Field(
        default_factory=CandidateLanguagesCapability
    )


class GuidanceCapabilities(_Container):
    """Guidance-family capabilities for one mode.

    Attributes:
        prompt: Free-text prompt channel.
        phrase_hints: Phrase-hint channel.
    """

    prompt: PromptCapability = Field(default_factory=PromptCapability)
    phrase_hints: PhraseHintsCapability = Field(default_factory=PhraseHintsCapability)


class StreamingGuidanceCapabilities(GuidanceCapabilities):
    """Streaming guidance-family capabilities (adds mid-stream mutability).

    Identical to :class:`GuidanceCapabilities` plus ``mutable_mid_stream`` -- the
    declaration site for the "guidance may change mid-stream" flag. It lives
    only on the *streaming* guidance family because mid-stream
    mutability is meaningless for batch (a single shot); batch guidance keeps the
    plain :class:`GuidanceCapabilities`.

    A supported ``mutable_mid_stream`` means the engine MAY accept updated guidance
    after ``start_transcription`` (otherwise ``RuntimeParams`` is frozen for the
    whole session). The current generation reserves the flag as the standard query path
    (``supports("streaming.guidance.mutable_mid_stream")``) and does NOT promise an
    ``update_guidance()`` method; default ``supported=False`` coincides with the
    fail-closed "session-locked" semantics, so the compliance suite requires no
    behavior for it. Modeled as a :class:`TraitFlagCapability` (not a bare ``bool``) so it
    derives a uniform ``supported`` and ``covers()`` set-containment auto-rejects a
    ``declared=false -> effective=true`` widening.

    Attributes:
        prompt: Free-text prompt channel.
        phrase_hints: Phrase-hint channel.
        mutable_mid_stream: Whether guidance may be updated mid-session.
    """

    mutable_mid_stream: TraitFlagCapability = Field(default_factory=TraitFlagCapability)


class BatchCapabilities(_Container):
    """Capability tree for the ``batch`` mode domain.

    Attributes:
        language: Language capabilities.
        word_timestamps: Word-timestamp capability.
        guidance: Guidance-family capabilities.
        diarization: Diarization capability.
    """

    language: LanguageCapabilities = Field(default_factory=LanguageCapabilities)
    word_timestamps: WordTimestampsCapability = Field(default_factory=WordTimestampsCapability)
    guidance: GuidanceCapabilities = Field(default_factory=GuidanceCapabilities)
    diarization: DiarizationCapability = Field(default_factory=DiarizationCapability)


class StreamingCapabilities(_Container):
    """Capability tree for the ``streaming`` mode domain.

    Attributes:
        language: Language capabilities (MAY differ from batch).
        word_timestamps: Word-timestamp capability.
        diarization: Diarization capability (MAY differ from batch).
        guidance: Guidance-family capabilities (MAY differ from batch); the
            streaming variant additionally exposes ``mutable_mid_stream``.
        emits_partials: Whether partial events are emitted.
        re_segments: Whether supersede events may occur.
        partial_stability: Whether a ``partial`` event may carry non-empty
            ``stable_text``. A ``final`` is outside this capability: it marks
            its whole text stable on every engine.
        reconnect: Reconnect capability mode.
        finality_level: Whether the engine promises a ``closed`` final for
            each finalized segment (see :class:`FinalityCapability`).
        timestamps: Source of streaming timestamps.
    """

    language: LanguageCapabilities = Field(default_factory=LanguageCapabilities)
    word_timestamps: WordTimestampsCapability = Field(default_factory=WordTimestampsCapability)
    diarization: DiarizationCapability = Field(default_factory=DiarizationCapability)
    # Typed as the base GuidanceCapabilities so an engine that declares a plain
    # GuidanceCapabilities still validates (backward tolerance), but ``_coerce_streaming_
    # guidance`` below normalizes every provided value to the StreamingGuidanceCapabilities
    # subtype so the queryable ``mutable_mid_stream`` flag is a real typed node on
    # EVERY construction path (default, typed instance, plain-base instance, and
    # dict / model_validate / JSON input). Without that, a value supplied as a dict or a
    # plain GuidanceCapabilities would land the flag in ``model_extra``, vanishing from
    # supports()/covers() while still advertised by to_protocol_dict() -- a two-layer
    # desync and a covers() declared=false -> effective=true
    # widening bypass.
    guidance: GuidanceCapabilities = Field(default_factory=StreamingGuidanceCapabilities)
    emits_partials: TraitFlagCapability = Field(default_factory=TraitFlagCapability)
    re_segments: TraitFlagCapability = Field(default_factory=TraitFlagCapability)
    partial_stability: TraitFlagCapability = Field(default_factory=TraitFlagCapability)
    reconnect: ReconnectCapability = Field(default_factory=ReconnectCapability)
    finality_level: FinalityCapability = Field(default_factory=FinalityCapability)
    timestamps: StreamTimestampsCapability = Field(default_factory=StreamTimestampsCapability)

    @field_validator("guidance", mode="before")
    @classmethod
    def _coerce_streaming_guidance(cls, value: object) -> object:
        """Normalize the streaming guidance node to :class:`StreamingGuidanceCapabilities`.

        The field is annotated as the base :class:`GuidanceCapabilities` for backward
        tolerance, but the runtime node MUST be the streaming subtype on every
        construction path so that ``mutable_mid_stream`` is a real typed field
        rather than an invisible ``model_extra``. Without this, a value supplied
        as a dict (the ``model_validate`` path, which JSON input takes) or as a
        plain :class:`GuidanceCapabilities` instance would be coerced to the base type and
        the flag would vanish from ``supports()`` / ``covers()`` while still being
        advertised by ``to_protocol_dict()`` -- a two-layer desync
        and a ``covers()`` ``declared=false -> effective=true`` widening bypass.

        Args:
            value: The raw guidance input (subtype instance, base instance, dict,
                or anything else).

        Returns:
            A :class:`StreamingGuidanceCapabilities` for any recognized input -- a plain
            :class:`GuidanceCapabilities` is re-homed via its dump so its fields and
            ``x_*`` extras are preserved and ``mutable_mid_stream`` defaults to
            fail-closed. Any other value is returned unchanged for pydantic to
            validate (and reject) natively.
        """
        if isinstance(value, StreamingGuidanceCapabilities):
            return value
        if isinstance(value, GuidanceCapabilities):
            return StreamingGuidanceCapabilities.model_validate(value.model_dump())
        if isinstance(value, dict):
            return StreamingGuidanceCapabilities.model_validate(value)
        return value


class Capabilities(_Container):
    """The full capability tree declared by an engine.

    Mode domains are optional: omitting a domain means the mode is not supported
    (fail-closed). Engine-global orthogonal flags live at the top level.

    Attributes:
        batch: Batch-mode capabilities, or ``None`` if batch is unsupported.
        streaming: Streaming-mode capabilities, or ``None`` if unsupported.
        streaming_input: Whether the engine accepts incremental audio. May only
            be supported when a ``streaming`` domain is declared.
        streaming_output: Whether the engine returns results incrementally. May
            only be supported when a ``streaming`` domain is declared.
        self_resamples: Whether the engine resamples audio internally. This is
            one of the traits (``role`` ``trait``) the spec declares in
            Capabilities rather than Properties -- alongside the per-mode
            ``diarization.always_on`` and the streaming trait flags; unlike
            those it is engine-global (a static property of the engine, not
            per-mode), so it lives at the top level alongside
            ``streaming_input`` / ``streaming_output``.

            It is **purely informational**: ``accepted_sample_rates`` remains
            authoritative for every resampling decision, so this
            flag has no decision power and does NOT change whether the standard
            resamples. It lets a client-side resampling engine (for example,
            faster-whisper, which declares ``accepted_sample_rates="any"``)
            advertise that incoming audio is downsampled inside the engine
            rather than by the standard. Absent ⇒ ``False`` (fail-closed).
    """

    batch: BatchCapabilities | None = None
    streaming: StreamingCapabilities | None = None
    streaming_input: TraitFlagCapability = Field(default_factory=TraitFlagCapability)
    streaming_output: TraitFlagCapability = Field(default_factory=TraitFlagCapability)
    self_resamples: TraitFlagCapability = Field(default_factory=TraitFlagCapability)

    @model_validator(mode="after")
    def _require_streaming_domain_for_streaming_flags(self) -> Capabilities:
        """Reject a supported streaming axis flag without a ``streaming`` domain.

        An omitted ``streaming`` domain means streaming is unsupported
        (fail-closed), so declaring ``streaming_input`` / ``streaming_output``
        as supported alongside it is self-contradictory: the flags advertise a
        transport axis for a mode the tree says does not exist (streaming input
        without a streaming-events domain is equally meaningless). Making the
        combination unrepresentable keeps the two signals from disagreeing.

        Returns:
            The validated capability tree.

        Raises:
            ValueError: If ``streaming_input`` or ``streaming_output`` is
                supported while ``streaming`` is ``None``.
        """
        if self.streaming is None:
            contradictory = [
                name
                for name, flag in (
                    ("streaming_input", self.streaming_input),
                    ("streaming_output", self.streaming_output),
                )
                if flag.is_supported
            ]
            if contradictory:
                raise ValueError(
                    f"{' and '.join(contradictory)} declared supported, but the "
                    "'streaming' capabilities domain is omitted -- an omitted domain "
                    "means streaming is unsupported (fail-closed), so the declaration "
                    "is self-contradictory. Declare streaming=StreamingCapabilities(...) "
                    "or drop the flag(s)."
                )
        return self

    def supports(self, dot_path: str) -> bool:
        """Return whether the capability at ``dot_path`` is supported.

        The standard support query. It does not say whether a request can
        ask for the feature; :meth:`can_request` answers that. Walks the tree
        segment by segment; any missing segment returns ``False``
        (fail-closed). Resolving a present mode-domain or container also
        returns ``True``.

        Args:
            dot_path: Dotted capability path without the ``capabilities.``
                prefix (for example, ``"batch.word_timestamps"``,
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

    def role_at(self, dot_path: str) -> CapabilityRole | None:
        """Return the role of the node at ``dot_path``, or ``None``.

        A capability node answers ``"requestable"`` or ``"trait"``, and a
        constraint submodel answers ``"constraints"``. ``None`` means the path
        states no role: a container or mode domain (it groups nodes), an
        absent path, a bare ``x_*`` grouping dict (it carries neither
        ``supported`` nor ``mode``), or a raw ``x_*`` node whose ``role`` is
        missing or is not exactly one of the three values. The role does not depend on
        support: an unsupported node still has one. To ask whether a request
        can use the feature, call :meth:`can_request`.

        Args:
            dot_path: Dotted capability path without the ``capabilities.``
                prefix (for example, ``"batch.diarization.always_on"``).

        Returns:
            The node's role, or ``None``.
        """
        return _derive_role(self._resolve(dot_path))

    def can_request(self, dot_path: str) -> bool:
        """Return whether a request can ask for the feature at ``dot_path``.

        ``True`` only when the node is supported and its role is
        ``requestable``. An unsupported requestable node, a supported trait
        node, a ``constraints`` submodel, a container, and an absent path all
        answer ``False`` (fail-closed, like :meth:`supports`). A standard feature is
        requested through its ``RuntimeParams`` field, and an ``x_*`` feature
        through the engine's ``provider_params``.

        Args:
            dot_path: Dotted capability path without the ``capabilities.``
                prefix (for example, ``"batch.diarization"``).

        Returns:
            ``True`` if a request can ask for the feature, otherwise ``False``.
        """
        node = self._resolve(dot_path)
        return _derive_supported(node) and _derive_role(node) == "requestable"

    def node_at(self, dot_path: str) -> _CapabilityNode | None:
        """Return the typed capability *node* at ``dot_path``, or ``None``.

        Unlike :meth:`supports` (which returns a bool), this returns the capability
        node object itself so callers can inspect its constraints / enums (for example,
        a ``WordTimestampsCapability`` to validate a requested granularity against
        :attr:`WordTimestampsCapability.granularities`). Returns ``None`` if the path
        is absent or does not resolve to a capability node.

        Args:
            dot_path: Dotted capability path without the ``capabilities.``
                prefix (for example, ``"batch.word_timestamps"``).

        Returns:
            The capability node, or ``None``.
        """
        node = self._resolve(dot_path)
        return node if isinstance(node, _CapabilityNode) else None

    def iter_supported_paths(self) -> Iterator[str]:
        """Yield the dot-path of every supported node in the tree.

        The paths are exactly the paths of :meth:`iter_queryable_paths` at
        which :meth:`supports` answers ``True``. Support is read at the node a
        path addresses, so an unsupported ancestor hides nothing: a supported
        ``x_*`` node on an unsupported feature, or inside a ``constraints``
        submodel, is yielded. A typed ``constraints`` submodel is never
        yielded, because it is not a capability, and neither is a raw dict
        that carries neither ``supported`` nor ``mode``. :meth:`covers`
        compares these paths to check that an effective tree only narrows a
        declared one.

        Yields:
            Dot-paths of supported capability nodes and present containers.
        """
        yield from _iter_paths(self, prefix="")

    def iter_queryable_paths(self) -> Iterator[str]:
        """Yield the dot-path of every NODE in the tree -- supported or not.

        The node set is fixed by the two-layer isomorphism: exactly the
        paths at which :meth:`to_protocol_dict` renders a JSON object and at
        which :meth:`supports` resolves a model/dict -- capability nodes,
        containers, constraint submodels, and ``x_*`` extension subtrees
        (typed or raw-dict; model extras pass the same ``x_*`` gate as every
        other traversal, so a non-extension unknown key is not a node).
        Scalar field values (a ``supported`` bool, a ``mode`` token, a
        ``granularities`` list) are field internals, not nodes: neither
        yielded nor descended. ``None`` children (an absent mode domain, a
        ``constraints=None``) are skipped.

        Unlike :meth:`iter_supported_paths` (the supported-only view behind
        ``effective ⊆ declared``), UNSUPPORTED nodes are yielded and
        descended, so a consumer can verify the fail-closed ``False`` answers
        too -- for example, an unsupported feature's ``constraints`` submodel MUST
        probe ``False``. The compliance suite sweeps this set to assert a
        hand-written ``supports()`` agrees with the tree on every node.

        Yields:
            Dot-paths of every capability node, container, submodel, and
            extension subtree in the tree.
        """
        yield from _iter_node_paths(self, prefix="")

    def covers(self, other: Capabilities) -> bool:
        """Return whether ``other`` is a valid narrowing of this tree.

        Enforces the normative ``effective ⊆ declared`` invariant:
        the effective set may only *close* declared capabilities, never widen
        them. This checks three things:

        * **Set containment** -- every path that :meth:`iter_supported_paths`
          yields for ``other`` is also supported here (no feature is enabled
          that this tree did not declare), a path under an unsupported node
          included.
        * **Same role** -- at every path both trees hold, :meth:`role_at`
          gives the same answer, whether or not the node is supported there.
          A path that ``other`` omits is a narrowing.
        * **Constraint narrowing** -- where both trees support a bounded or
          enum/mode node, ``other``'s limits MUST be no looser than this tree's
          (for example, a smaller-or-equal ``max``, a subset of ``granularities``, a
          ``mode`` that is the same or a reduction). A widening (declared
          ``max=2`` -> effective ``max=999``) is rejected.

        A restriction this tree states must be met by a well-formed value in
        ``other``, and a comparison that cannot be made counts as a widening.
        This tree states a restriction only with a string ``mode``, an
        ``int`` bound that is not a ``bool``, or a nonempty list of strings
        in ``granularities``. Where it states one, a node that ``other``
        still supports widens this tree when it drops the value or gives it
        in the wrong shape: a ``None`` or ``1`` mode, a ``6.5``, ``"6"``, or
        ``True`` bound, or a ``granularities`` that is not a list of
        strings. An empty, omitted, or ``None`` ``granularities`` means
        every granularity, in either tree: in ``other`` it widens a nonempty
        list in this tree, and in this tree it restricts nothing. A value of
        any other shape in this tree restricts nothing too. The constraint
        narrowing compares only these standard fields, on a standard node and
        an ``x_*`` node alike: ``mode``, ``granularities``, and the upper
        bounds ``max``, ``max_tokens``, ``max_terms``, ``max_chars_per_term``,
        ``max_words_per_term``, and ``max_speakers`` in ``constraints``. A
        vendor's other fields mean what the vendor says, and nothing compares
        them.

        Args:
            other: A (typically narrowed, effective) capability tree.

        Returns:
            ``True`` if ``other`` is a subset narrowing of this tree.
        """
        mine = set(self.iter_supported_paths())
        for path in other.iter_supported_paths():
            if path not in mine:
                return False
        # A node's role is data (design D3 in
        # docs/internal/feat_plan/capability-kinds.md), and it does not depend on
        # support, so the role of every node both trees hold must agree,
        # supported or not: a change in either direction changes what
        # role_at() and can_request() answer, and neither direction narrows.
        # A path the effective tree omits is a narrowing the loop above
        # already allowed, so only shared paths are compared.
        for path in other.iter_queryable_paths():
            effective_node = other._resolve(path)
            declared_node = self._resolve(path)
            if declared_node is None:
                continue
            if _derive_role(declared_node) != _derive_role(effective_node):
                return False
        # Where both support a node, the effective node must not be looser.
        for path in other.iter_supported_paths():
            declared_node = self._resolve(path)
            effective_node = other._resolve(path)
            if declared_node is None or effective_node is None:  # pragma: no cover
                # Defensive: set-containment above guarantees every `other` path
                # also resolves here, so neither side resolves to None in
                # practice. Kept as a guard against a future traversal change.
                continue
            if not _node_narrows(declared_node, effective_node):
                return False
        return True

    def to_protocol_dict(self) -> dict[str, Any]:
        """Return the capability JSON document: the tree as a JSON-ready dict.

        This is the form the protocol defines for clients in other languages,
        and what ``GET /v1/capabilities/{model}`` returns. Flag and bounded
        nodes carry ``supported`` as a real field, but enum/mode nodes derive it
        from ``mode`` (a Python property, absent from ``model_dump``). This method
        adds the derived boolean to every capability node and every present
        container below the root, so a client never has to special-case shapes
        or know the ``"none"``/``"unsupported"`` sentinels (enum/mode nodes'
        ``supported`` is server-injected). The root object itself carries no
        ``supported`` key (it is the container of all modes, not a capability),
        and neither does a ``constraints`` submodel, which is not a capability
        node; an absent mode domain serializes as ``null`` (fail-closed). A raw
        ``x_*`` dict gets ``supported`` only when it carries ``supported`` or
        ``mode``. Every typed capability node and constraint submodel also
        carries its ``role`` field. A container carries none, and a raw ``x_*``
        node carries one only when its declaration does.

        Returns:
            The capability JSON document: ``supported`` on every capability
            node and present container below the root, none on the root or on
            a ``constraints`` submodel, and ``null`` for an absent mode domain.
        """
        return cast("dict[str, Any]", _to_canonical(self, inject_supported=False))

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
        # An extra key on a typed node resolves only inside the ``x_*`` extension
        # namespace. A non-extension unknown segment (for example, a typo of
        # a real field) is fail-closed -- treated as absent -- so it never reads
        # as a supported capability. Keys *inside* a raw ``x_*`` subtree (the dict
        # branch below) are the vendor's own and are not filtered.
        if not _is_extension_key(part):
            return None
        extra: dict[str, Any] = node.model_extra or {}
        return extra.get(part)
    if isinstance(node, dict):
        return cast("dict[str, object]", node).get(part)
    return None


def _derive_supported(node: object) -> bool:
    """Derive the ``is_supported`` boolean for a resolved node.

    Args:
        node: A capability node, container, or raw dict/value.

    Returns:
        ``True`` if the node represents a supported capability or a present
        container.
    """
    if isinstance(node, _CapabilityNode):
        return node.is_supported
    if isinstance(node, _Container):
        # A present container (mode domain or grouping) counts as supported.
        return True
    if isinstance(node, BaseModel):
        # A non-capability BaseModel (a `constraints` submodel) is NOT a
        # capability node: `supports("<feature>.constraints")` must
        # be fail-CLOSED, never report the feature as supported via its limits.
        return False
    if isinstance(node, dict):
        mapping = cast("dict[str, object]", node)
        # An explicit `supported` is the authoritative flag for the flag and
        # bounded shapes and is read as a STRICT boolean: only a real ``True`` counts
        # as supported. A non-bool (for example, the STRING "false", truthy in Python, or
        # a number) is a malformed declaration and is fail-closed to ``False`` --
        # never silently promoted to supported. An explicit `supported` is also
        # checked BEFORE `mode`: a `mode` sub-key on the same node MUST NOT raise
        # an explicit ``supported: false`` back to true (fail-closed).
        if "supported" in mapping:
            return mapping["supported"] is True
        if "mode" in mapping:
            mode = mapping["mode"]
            # A `mode` value MUST be a string mode token. A non-string (bool,
            # number, None, and so on) is a malformed declaration and is fail-CLOSED to
            # ``False`` -- never silently promoted to supported --
            # mirroring the strict-boolean reading of `supported` above. Without
            # the ``isinstance`` guard, ``True``/``1`` would pass ``not in`` (a
            # frozenset of strings) and be wrongly reported as supported.
            return isinstance(mode, str) and mode not in _UNSUPPORTED_MODES
        # A raw dict with neither `supported` nor `mode` is NOT a
        # capability node -- it is non-capability data (a bare `constraints` dict
        # like {"max": 5}, a structural grouping, or a malformed vendor node that
        # forgot its flag). Probing it is fail-CLOSED (``False``), the same stance
        # taken for a typed `constraints` submodel above and the same predicate
        # `_to_canonical` uses (it injects `supported` only when `mode`/`supported`
        # is present). Previously this returned ``True`` ("present container
        # dict"), so `supports("batch.x_thing")` read ``True`` while
        # `to_protocol_dict()` emitted no `supported` key for the same node -- the
        # two layers disagreed (violating the two-layer isomorphism) and a
        # flag-less malformed x_* node read as supported. A vendor
        # that wants a queryable x_* node MUST mark it `supported`/`mode`
        # explicitly (x_* gating rules are the same as standard).
        return False
    return False


def _derive_role(node: object) -> CapabilityRole | None:
    """Derive the role of a resolved node, or ``None`` when it states none.

    Args:
        node: A capability node, constraint submodel, container, raw dict,
            or ``None``.

    Returns:
        The node's role, or ``None``. A typed node answers the value its
        class fixes.
        A raw dict answers its ``role`` key only when the dict is a
        capability node (it carries ``supported`` or ``mode``) and the key
        holds exactly one of the three values. A bare grouping dict and any
        other value answer ``None``, so a vendor node never reads as
        requestable by accident.
    """
    if isinstance(node, (_CapabilityNode, _ConstraintNode)):
        return node.role
    if isinstance(node, dict):
        mapping = cast("dict[str, object]", node)
        # Only a capability node states a role: a dict is one exactly when it carries
        # ``supported`` or ``mode``, the same predicate _derive_supported and
        # _to_canonical use. A bare grouping dict groups nodes and states
        # nothing, whatever keys it carries.
        if "supported" not in mapping and "mode" not in mapping:
            return None
        role = mapping.get("role")
        if role in ("requestable", "trait", "constraints"):
            return role
    return None


def _to_canonical(node: object, *, inject_supported: bool) -> Any:
    """Recursively convert a capability tree, or part of one, to its JSON-ready form.

    Mirrors ``model_dump(mode="json")`` but injects a derived ``supported``
    boolean at every capability node (``_CapabilityNode``) and present container
    (``_Container``). Constraint submodels are not capabilities and get no
    ``supported`` key. See :meth:`Capabilities.to_protocol_dict`.

    Args:
        node: A model, container, list, dict, or scalar to convert.
        inject_supported: Whether to add ``supported`` to this node if it is a
            capability node or container (``False`` only for the root).

    Returns:
        A JSON-serializable representation of ``node``.
    """
    if isinstance(node, BaseModel):
        out: dict[str, Any] = {}
        for name in type(node).model_fields:
            out[name] = _to_canonical(getattr(node, name), inject_supported=True)
        for key, value in (node.model_extra or {}).items():
            out[key] = _to_canonical(value, inject_supported=True)
        if inject_supported and isinstance(node, (_CapabilityNode, _Container)):
            out["supported"] = _derive_supported(node)
        return out
    if isinstance(node, list):
        return [_to_canonical(item, inject_supported=True) for item in cast("list[object]", node)]
    if isinstance(node, dict):
        mapping = cast("dict[str, object]", node)
        out_dict: dict[str, Any] = {
            key: _to_canonical(value, inject_supported=True) for key, value in mapping.items()
        }
        # A JSON-sourced x_* capability lands here as a raw dict (not a
        # typed _CapabilityNode). Inject the derived `supported` so cross-language
        # clients get the same uniform probe the typed path provides.
        # A dict is a capability node iff it carries `mode` or `supported`; a bare
        # `constraints` dict (for example, {"max": 5}) has neither and is left untouched.
        if inject_supported and ("mode" in mapping or "supported" in mapping):
            out_dict["supported"] = _derive_supported(mapping)
        return out_dict
    return node


def _iter_paths(node: object, prefix: str) -> Iterator[str]:
    """Recursively yield the supported dot-paths under ``node``.

    The walk visits every node whose path
    :meth:`Capabilities.iter_queryable_paths` yields, supported or not. It
    yields a path exactly when
    :func:`_derive_supported` answers ``True`` for the path's node, which is
    the answer :meth:`Capabilities.supports` gives for that path. Support is
    a fact about the addressed node, so an unsupported ancestor hides
    nothing.

    The standard children of an unsupported standard feature add no path.
    Its ``constraints`` submodel is never a capability
    (:func:`_derive_supported` answers ``False`` for it). A trait child such
    as ``always_on`` is not supported there either, because the parent's
    validator rejects that pair when the parent is built. An ``x_*`` node on
    the feature, or on its ``constraints`` submodel, is the vendor's own
    node. :meth:`Capabilities.supports` and
    :meth:`Capabilities.to_protocol_dict` read it as the vendor declared it,
    so the walk yields its path when it is supported.

    A bare grouping dict, with neither ``supported`` nor ``mode``, is not a
    capability, so its own path is not yielded, while a nested explicit
    capability (``{"supported": true}``) under it is. A bare ``constraints``
    dict such as ``{"max": 5}`` adds no path either: it is not a capability,
    and its values are not nodes.

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
        # Walk every node, supported or not. Support decides whether a path is
        # yielded, never whether its subtree is walked, so the set agrees with
        # supports() at every path. to_protocol_dict() renders the subtree of
        # an unsupported node too, so the walk and the capability JSON
        # document cover the same nodes.
        if isinstance(child, (BaseModel, dict)):
            yield from _iter_paths(cast("object", child), path)


def _iter_node_paths(node: object, prefix: str) -> Iterator[str]:
    """Recursively yield every node path under ``node``, supported or not.

    A node is any :class:`~pydantic.BaseModel` or dict child reachable through
    :func:`_children` (which applies the ``x_*`` gate to model extras) --
    the same object set :meth:`Capabilities.to_protocol_dict` renders
    as JSON objects. Scalars and lists are field internals, ``None`` children
    are absent domains; neither is a node.

    Args:
        node: A pydantic model or dict to walk.
        prefix: The accumulated dot-path prefix.

    Yields:
        Every node dot-path, in traversal order.
    """
    for name, child in _children(node):
        if not isinstance(child, (BaseModel, dict)):
            continue
        path = f"{prefix}.{name}" if prefix else name
        yield path
        yield from _iter_node_paths(cast("object", child), path)


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
        # Only ``x_*`` extension extras are queryable capabilities;
        # other unknown keys (forward-compat tolerated on parse, or a typo) MUST
        # NOT pollute the supported-path set used by the ``effective ⊆ declared``
        # comparison. Mirror the same gate as :func:`_get_child`.
        items.extend((key, value) for key, value in extra.items() if _is_extension_key(key))
        return items
    if isinstance(node, dict):
        return list(cast("dict[str, object]", node).items())
    return []


#: Constraint fields whose semantics are an *upper bound* (effective ≤ declared).
#: REGISTRY-DRIFT INVARIANT: this hand-maintained set MUST list every
#: upper-bound (``gt``-constrained ``int``) field across all ``*Constraints``
#: submodels. A new ``max_*`` field that is added but not registered here would
#: make :func:`_node_narrows` SILENTLY stop rejecting a widening of it -- a
#: fail-open the compliance suite would then miss (adding new constraint fields
#: is an expected evolution path). The guard
#: ``test_max_constraint_fields_registry_covers_every_bounded_field`` walks the
#: submodels and fails if any bounded field is unregistered (or stale).
_MAX_CONSTRAINT_FIELDS = frozenset(
    {"max", "max_tokens", "max_terms", "max_chars_per_term", "max_words_per_term", "max_speakers"}
)

#: enum/mode reductions: declared mode -> the set of modes that are no looser.
#: A mapping value is the set of effective modes accepted for that declared mode
#: (always includes the declared mode itself plus any strictly weaker mode).
#:
#: GLOBAL-TOKEN INVARIANT: reductions are keyed by the *bare* token,
#: not by node type, so this is sound ONLY while the standard enum/mode families
#: have **disjoint** token sets (they do today). A future enum node that reuses an
#: existing token (for example, ``none`` / ``final``) with a *different* strength order
#: would silently inherit the order below. A new standard enum node therefore MUST
#: NOT reuse an existing token with a different order; the guard
#: ``test_mode_reduction_tokens_are_globally_unique_per_enum_family`` enforces
#: token disjointness and that this map stays in lockstep with the declared enum
#: tokens. (``x_*`` experimental enum nodes that reuse a token are vendor risk and
#: are out of scope; an unknown token is fail-closed in :func:`_node_narrows`.)
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

    This is the per-node half of the ``effective ⊆ declared`` invariant. It
    compares the standard restrictions a node can state: its ``mode``, its
    ``granularities`` enumeration, and the upper bounds in its
    ``constraints``. A node that states none of them, such as a flag node,
    always passes, because set containment already covered it.
    :meth:`Capabilities.covers` calls this only at paths that both trees
    support.

    Args:
        declared: The declared node.
        effective: The effective node at the same path.

    Returns:
        ``True`` if ``effective`` does not widen ``declared``.
    """
    # One rule covers every restriction below. A restriction the declared
    # node states must be met by a well-formed effective value that narrows
    # it. A comparison that cannot be made, because the effective value is
    # missing, None, or not the shape the field has, proves no narrowing, so
    # it fails (fail-closed), as a changed mode with no known strength order
    # does. A restriction the declared node does not state, or states in a
    # shape this function does not read, restricts nothing, so any effective
    # value narrows it.

    # enum/mode nodes: the effective mode must be the declared mode or a
    # reduction of it. A token outside _MODE_REDUCTIONS, such as the token of
    # an x_* enum node or a token from a later version, has no known strength
    # order, so a change from or to such a token fails rather than passing as
    # a narrowing. A dropped mode fails too. The effective node is still
    # supported here, so a missing, None, or non-string mode proves no
    # narrowing.
    declared_mode = _read_attr(declared, "mode")
    effective_mode = _read_attr(effective, "mode")
    if isinstance(declared_mode, str):
        if not isinstance(effective_mode, str):
            return False
        if declared_mode != effective_mode:
            allowed = _MODE_REDUCTIONS.get(declared_mode)
            if allowed is None or effective_mode not in allowed:
                return False

    # granularities: the effective enumeration must be a subset of the
    # declared one. An empty, omitted, or None enumeration means "not
    # enumerated, so every granularity" on either side, by the
    # specification's rule on empty enumerations, which
    # granularity_offers_all also states. So a declared node that did not
    # enumerate is narrowed by any effective value, and an effective node
    # that stops enumerating widens a finite declared enumeration. A value
    # that is not a list of strings is not an enumeration. On the declared
    # side it restricts nothing, and on the effective side it proves no
    # narrowing.
    declared_grans = _enumeration(_read_attr(declared, "granularities"))
    if declared_grans:
        effective_grans = _enumeration(_read_attr(effective, "granularities"))
        if not effective_grans or not effective_grans <= declared_grans:
            return False

    # constraints: no upper bound may grow. A declared finite bound must be
    # met by an effective finite bound no larger than it. An effective bound
    # that is missing, None, a bool, a float, or a string states no readable
    # bound where the declared node had one, which is a widening. Dropping
    # the whole constraints submodel makes the same claim for every bound at
    # once. The only typed node whose `constraints` may be None is
    # CandidateLanguagesCapability, where None means no maximum, and a raw
    # x_* or dict node may omit the key. A declared constraints submodel
    # with no finite bound, such as an all-None PromptConstraints, restricts
    # nothing, so losing it widens nothing.
    declared_c = _read_attr(declared, "constraints")
    if declared_c is not None:
        effective_c = _read_attr(effective, "constraints")
        for field in _MAX_CONSTRAINT_FIELDS:
            d_val = _bound(_read_attr(declared_c, field))
            if d_val is None:
                continue
            e_val = _bound(_read_attr(effective_c, field)) if effective_c is not None else None
            if e_val is None or e_val > d_val:
                return False
    # A declared node without constraints states no bound, so the effective
    # node narrows it whether or not it has constraints.
    return True


def _enumeration(value: object) -> frozenset[str] | None:
    """Read an enumeration field's value as the set of tokens it offers.

    Args:
        value: The field's value on a node, ``None`` when the field is absent
            or holds ``None``.

    Returns:
        The tokens as a set. The set is empty when the node did not
        enumerate, which means every token: an empty list, an absent field,
        and ``None`` all read this way. ``None`` when the value is not a list
        of strings, because such a value is not an enumeration at all.
    """
    if value is None:
        return frozenset()
    if not isinstance(value, list):
        return None
    tokens = cast("list[object]", value)
    if not all(isinstance(token, str) for token in tokens):
        return None
    return frozenset(cast("list[str]", tokens))


def _bound(value: object) -> int | None:
    """Read a constraint field's value as a finite upper bound.

    Args:
        value: The field's value on a ``constraints`` submodel or dict,
            ``None`` when the field is absent or holds ``None``.

    Returns:
        The bound when the value is an ``int``. A ``bool`` does not count,
        although Python treats it as an ``int``. ``None`` for any other
        value, because an absent, ``None``, or malformed value states no
        bound.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _read_attr(node: object, name: str) -> object:
    """Read the protocol field ``name`` of a typed node or a dict, else ``None``.

    On a typed node, only a declared field counts. An extra key with that
    name is either an ``x_*`` node, which is never a field, or an unknown key
    that parsing tolerates (spec R2). Neither carries the field's meaning, so
    the narrowing comparison must not read it. On a raw dict, every key is
    the vendor's field.

    Args:
        node: A pydantic model, dict, or other value.
        name: The field or key name.

    Returns:
        The value, or ``None`` if the field is absent.
    """
    if isinstance(node, BaseModel):
        if name in type(node).model_fields:
            return getattr(node, name)
        return None
    if isinstance(node, dict):
        return cast("dict[str, object]", node).get(name)
    return None


__all__ = [
    "BatchCapabilities",
    "CandidateLanguagesCapability",
    "CandidateLanguagesConstraints",
    "CapabilityRole",
    "Capabilities",
    "DiarizationCapability",
    "DiarizationConstraints",
    "FinalityCapability",
    "TraitFlagCapability",
    "GuidanceCapabilities",
    "granularity_offers_all",
    "LanguageCapabilities",
    "ModeName",
    "PhraseHintsCapability",
    "PhraseHintsConstraints",
    "PromptCapability",
    "PromptConstraints",
    "ReconnectCapability",
    "LanguageOverrideCapability",
    "StreamTimestampsCapability",
    "StreamingCapabilities",
    "StreamingGuidanceCapabilities",
    "WordTimestampGranularityName",
    "WordTimestampsCapability",
]
