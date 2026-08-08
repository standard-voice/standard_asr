# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Init configuration models for ASR engines (spec, section "Init Config").

``BaseConfig`` is the discriminated base for an engine's install/deploy-time
configuration: a discriminator ``engine`` (the entrypoint-derived ``engine_id``)
plus "relevant-only" optional standard fields (provided via the applicability
mixins below) plus engine-declared fields.

Key normative behaviors implemented here:

* **Credential safety:** credential fields MUST use a masking carrier --
  ``SecretStr`` (or ``SecretBytes`` for byte credentials), exactly one per
  field -- and be marked secret; :meth:`BaseConfig.public_dump` returns a
  sanitized dump for ``/v1/models``, persistence, and telemetry. Plaintext is
  materialized only on demand via ``get_secret_value()``.
* **Env fallback:** ``STANDARD_ASR_<NORMENGINE>__<NORMFIELD>`` (double
  underscore boundary) with normalization and collision detection; composite
  fields are JSON-decoded; priority is explicit > env > error.
* **Applicability:** a standard field is applicable iff it appears in the
  model -- engines compose the mixins they need so auto-UI renders the right
  form without per-field hiding.
* **Lazy purity:** ``__init__`` capturing config MUST be pure (no FS,
  GPU, or network); materialization happens later under ``allow_downloads()``.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    ForwardRef,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    SecretBytes,
    SecretStr,
    SerializeAsAny,
    ValidationError,
    WrapSerializer,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic_core import PydanticCustomError, PydanticUndefined

from standard_asr.contract.exceptions import ConfigError, ConfigurationRequiredError
from standard_asr.runtime.redaction import config_error_from_validation

logger = logging.getLogger(__name__)

#: Mask emitted in place of any secret-marked field value by ``public_dump``.
SECRET_MASK = "**********"

#: Pydantic types that genuinely mask their value in ``repr``/``str``/dump.
_SECRET_TYPES: tuple[type[Any], ...] = (SecretStr, SecretBytes)

EngineNameT = TypeVar("EngineNameT", bound=str, covariant=True)
_ConfigT = TypeVar("_ConfigT", bound="BaseConfig[Any]")

#: Prefix for environment-variable credential fallback.
ENV_PREFIX = "STANDARD_ASR"


def secret_field(default: Any = None, *, description: str = "") -> Any:
    """Build a ``Field`` for a write-only credential rendered as a password.

    Use together with a ``SecretStr`` or ``SecretBytes`` annotation (exactly
    one carrier, optionally unioned with ``None``). The ``json_schema_extra``
    marks the field secret so auto-UI renders a password / write-only input and
    REST exposes it POST-only. :class:`BaseConfig` enforces the carrier
    annotation AND the default's shape at class-definition time, masks the
    value in :meth:`BaseConfig.public_dump`, and preserves the secret's exact
    contents (no whitespace stripping) so a paste error is never silently
    swallowed.

    Args:
        default: Field default. ``None`` for an optional credential (the
            annotation must union ``None``), ``...`` (Ellipsis) for a
            required one, or a carrier instance (e.g.
            ``SecretStr("preset")``) -- a plain string is rejected at class
            definition (defaults are not validated, so it would leak
            plaintext).
        description: Field description.

    Returns:
        A configured pydantic ``Field``.
    """
    return Field(
        default=default,
        description=description,
        json_schema_extra={"format": "password", "writeOnly": True, "secret": True},
    )


def _is_secret_marked(field: Any) -> bool:
    """Return whether a pydantic field is marked secret via ``json_schema_extra``.

    Args:
        field: A pydantic ``FieldInfo``.

    Returns:
        ``True`` if the field's ``json_schema_extra`` carries ``secret=True``.
    """
    extra = field.json_schema_extra
    return isinstance(extra, dict) and extra.get("secret") is True


def _secret_carrier(annotation: Any) -> type[SecretStr] | type[SecretBytes] | None:
    """Resolve the single masking carrier type of a secret annotation.

    The ONE authority on what a secret-marked field may be annotated as: the
    definition-time guard rejects any annotation this returns ``None`` for,
    and the whitespace-preserving pre-validator wraps raw strings into
    exactly the carrier this returns. The legal shapes are **exactly**
    ``SecretStr``, ``SecretBytes``, or either unioned with ``None`` --
    nothing else:

    * a **plaintext union member** (``SecretStr | int``, ``SecretStr | Path``)
      would let a constructed value bypass masking entirely: the field is
      advertised as a password in the schema while ``repr``/``model_dump``
      emit the plaintext value (the old ``any()``-over-members check accepted
      these);
    * **two carriers** (``SecretStr | SecretBytes``) make the pre-validator's
      raw-string wrap ambiguous (which type preserves the caller's intent?);
    * a **generic container** (``list[SecretStr]``) is half-protected: hidden
      by the UI markers while its items leak through the secret pipeline;
    * a **subclass** of a carrier is rejected by the exact-type rule: the
      pre-validator wraps into the carrier itself, which the subclass field
      would then reject at validation.

    Args:
        annotation: The field's resolved annotation.

    Returns:
        The carrier type (``SecretStr`` or ``SecretBytes``), or ``None`` when
        the annotation is not a legal scalar secret shape.
    """
    if annotation is SecretStr or annotation is SecretBytes:
        return annotation
    # Only union members are unwrapped (both typing.Union and the PEP 604
    # ``X | Y`` form). Any other parametrized origin is a generic container and
    # MUST NOT satisfy the requirement.
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(members) == 1 and (members[0] is SecretStr or members[0] is SecretBytes):
            return members[0]
    return None


def _annotation_serialization_gap(annotation: Any) -> str | None:
    """Describe an authored serialization hazard reachable in an annotation.

    The enumerated, accident-grade replacement for the deleted core-schema
    closure proof (see the trust model in AGENTS.md): it walks the DECLARED
    annotation -- unions, generic containers, ``Annotated`` metadata -- and
    names the shapes an honest author actually writes:

    * an **undeclared value shape** (``Any`` / ``object`` / bare ``dict`` /
      bare ``list``): pydantic serializes whatever object such a field holds
      at runtime by that object's OWN rules, the JSON Schema for it is empty
      (a settings UI cannot render it), and its env value has no defined
      reading. Declare a submodel or a typed container instead.
    * a **``SerializeAsAny`` marker**: the dump follows the RUNTIME object,
      so the declared type no longer bounds what ``public_dump`` emits.
    * a nested submodel carrying its own serialization hooks
      (``@computed_field`` / ``@model_serializer`` / ``@field_serializer`` /
      ``Annotated`` serializers): those run inside the parent's dump.

    Channels only a core-schema prover could see (a custom
    ``__get_pydantic_core_schema__`` installing a serializer) require an
    author actively smuggling code past the enumeration -- an adversary,
    out of scope by the trust model.

    Args:
        annotation: The field's resolved annotation.

    Returns:
        A description of the first hazard found, or ``None``.
    """
    if annotation is Any or annotation is object:
        return "an undeclared value shape (Any/object)"
    if annotation is dict or annotation is list:
        return "an undeclared value shape (a bare, unparametrized container)"
    # Gate on get_origin BEFORE any issubclass, exactly like
    # _nested_models_in_annotation: a parametrized generic (list[X], ...)
    # satisfies isinstance(_, type) on Python 3.10 (but not 3.11+), so an
    # isinstance-first branch hands list[X] to issubclass -- a TypeError
    # that fired at every config class definition on the 3.10 floor.
    if get_origin(annotation) is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return _nested_model_serialization_gap(annotation)
        # A non-model class or a non-class leaf (Literal value, TypeVar,
        # None) carries no hooks.
        return None
    for arg in get_args(annotation):
        if arg is SerializeAsAny or isinstance(arg, cast("type", SerializeAsAny)):
            return "a DUCK-TYPED serialization marker (SerializeAsAny)"
        gap = _annotation_serialization_gap(arg)
        if gap is not None:
            return gap
    return None


def _nested_model_serialization_gap(
    model: type[BaseModel], _seen: set[type[BaseModel]] | None = None
) -> str | None:
    """Describe a serialization hook on a nested submodel (or deeper), if any.

    A nested submodel's OWN hooks run inside the parent's dump (its computed
    field is emitted as a key of the nested object), so the config-class
    decorator scan alone is blind to them. Recurses through the submodel's
    fields; cycle-safe.

    Args:
        model: The nested model class to inspect.
        _seen: Visited classes (cycle guard), threaded through recursion.

    Returns:
        A description of the first hook found, or ``None``.
    """
    seen: set[type[BaseModel]] = _seen if _seen is not None else set()
    if model in seen:
        return None
    seen.add(model)
    decorators = model.__pydantic_decorators__
    for name in decorators.computed_fields:
        return f"nested submodel {model.__name__!r} declares a computed field {name!r}"
    for name in (*decorators.model_serializers, *decorators.field_serializers):
        return f"nested submodel {model.__name__!r} declares an author-defined serializer {name!r}"
    for name, field in model.model_fields.items():
        if any(isinstance(meta, (PlainSerializer, WrapSerializer)) for meta in field.metadata):
            return (
                f"nested submodel {model.__name__!r} field {name!r} carries an "
                f"author-defined serializer (PlainSerializer/WrapSerializer)"
            )
        for nested in _nested_models_in_annotation(field.annotation):
            gap = _nested_model_serialization_gap(nested, seen)
            if gap is not None:
                return gap
    return None


def _annotation_admits_none(annotation: Any) -> bool:
    """Return whether a (secret) annotation admits ``None`` as a value.

    Args:
        annotation: The field's resolved annotation.

    Returns:
        ``True`` if the annotation is a union containing ``None``.
    """
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return type(None) in get_args(annotation)
    return False


def _annotation_has_unresolved_ref(annotation: Any) -> bool:
    """Return whether a field annotation is or contains an unresolved forward ref.

    Pydantic leaves a reference to a model that is not yet defined at class-creation
    time as a bare :class:`typing.ForwardRef` (its ``get_args`` is empty), so
    :func:`_nested_models_in_annotation` cannot see the submodel behind it. The
    nested-secret guard must therefore fail **closed** on such an annotation rather
    than pass it unchecked: a credential buried in a define-after submodel would
    otherwise slip the scan. A resolved or imported submodel never survives as a
    ``ForwardRef``, so this never rejects a legitimately-resolvable annotation.

    Args:
        annotation: The field's annotation.

    Returns:
        ``True`` if the annotation is a ``ForwardRef`` or contains one in its
        arguments; ``False`` otherwise.
    """
    if isinstance(annotation, ForwardRef):
        return True
    if get_origin(annotation) is Literal:
        # Literal arguments are *values* (e.g. strings), not type annotations, so
        # they must not be recursed into (a Literal["x"] arg is not a ForwardRef).
        return False
    return any(_annotation_has_unresolved_ref(arg) for arg in get_args(annotation))


def _nested_models_in_annotation(annotation: Any) -> list[type[BaseModel]]:
    """Return every ``BaseModel`` subclass reachable in a field annotation.

    Walks unions (``X | Y``) and generic containers (``list[...]``,
    ``dict[str, ...]``, ``tuple[..., ...]``, ``Optional[...]`` etc.) so a
    submodel nested anywhere in the annotation is found, e.g. the ``Auth`` in
    ``Auth``, ``Auth | None``, ``list[Auth]``, or ``dict[str, Auth]``. Scalar
    secret types (``SecretStr`` / ``SecretBytes``) are intentionally **not**
    returned: they are the legitimate top-level secret carriers and the
    nested-secret guard must not flag them.

    Args:
        annotation: The field's resolved annotation.

    Returns:
        The list of ``BaseModel`` subclasses found (possibly empty; duplicates
        possible -- the caller deduplicates via a visited set).
    """
    # Gate on get_origin, NOT isinstance(annotation, type): a parametrized generic
    # (list[X], dict[str, X], ...) satisfies isinstance(_, type) on Python 3.10
    # (but not 3.11+), so an isinstance-first walk would treat list[Auth] as a leaf
    # and never reach the nested Auth -- silently disabling the secret guard on 3.10.
    # A real class has get_origin() is None; a generic/union has a non-None origin.
    if get_origin(annotation) is None:
        if isinstance(annotation, type):
            if issubclass(annotation, _SECRET_TYPES):
                return []
            if issubclass(annotation, BaseModel):
                return [annotation]
        return []
    found: list[type[BaseModel]] = []
    for arg in get_args(annotation):
        found.extend(_nested_models_in_annotation(arg))
    return found


def _annotation_contains_secret_carrier(annotation: Any) -> bool:
    """Return whether a secret carrier type is reachable in the annotation.

    Walks unions and generic containers exactly like
    :func:`_nested_models_in_annotation` and reports any ``SecretStr`` /
    ``SecretBytes`` (or subclass) found -- ``SecretStr``, ``SecretStr |
    None``, ``list[SecretStr]``, ``dict[str, SecretBytes]`` all count. The
    class hook uses this to require the secret MARKER wherever a carrier
    appears: an unmarked carrier field is half-protected -- pydantic masks
    its dumps, but it bypasses the whitespace-preserving validator (its
    raw string input is silently stripped, the exact credential rewrite
    the pipeline forbids), the password/write-only schema markers, and
    ``public_dump``'s defensive by-name mask.

    Args:
        annotation: The field's resolved annotation.

    Returns:
        ``True`` when a carrier type is reachable anywhere in it.
    """
    if get_origin(annotation) is None:
        return isinstance(annotation, type) and issubclass(annotation, _SECRET_TYPES)
    return any(_annotation_contains_secret_carrier(arg) for arg in get_args(annotation))


def _nested_credential_path(
    model: type[BaseModel], _visited: set[type[BaseModel]] | None = None
) -> tuple[str, str] | None:
    """Return the first credential-shaped field in ``model``, recursively.

    Searches ``model``'s own fields and every nested ``BaseModel`` reachable
    through their annotations, for BOTH credential signals: the secret
    MARKER (``secret_field``) and a bare CARRIER annotation
    (``SecretStr``/``SecretBytes`` without the marker). The secret pipeline
    (the carrier-enforcing class hook, the whitespace-preserving validator,
    and the masking dump) only operates on a :class:`BaseConfig`'s *own*
    scalar fields, so a nested marked secret is silently unprotected --
    its plaintext leaks through ``public_dump`` / ``repr`` /
    ``model_dump`` -- and a nested bare carrier is half-broken: its raw
    string input is silently whitespace-stripped, ``reveal_dump`` never
    unwraps it (the engine cannot reach the credential through the
    documented API), and a nested model's own serialization hooks could
    unwrap it into the "masked" dump. Detecting both lets the class hook
    reject the shape at definition time. A ``visited`` set guards against
    recursive/self-referential model graphs.

    Args:
        model: The model to search.
        _visited: Models already visited (cycle guard; internal).

    Returns:
        ``(dotted_path, description)`` for the first credential-shaped
        field found (e.g. ``("auth.token", "is marked secret
        (secret_field)")``), or ``None`` if the model graph carries none.
    """
    visited: set[type[BaseModel]] = _visited if _visited is not None else set()
    if model in visited:
        return None
    visited.add(model)
    for name, field in model.model_fields.items():
        if _is_secret_marked(field):
            return name, "is marked secret (secret_field)"
        if _annotation_contains_secret_carrier(field.annotation):
            return name, "is annotated with a secret carrier (SecretStr/SecretBytes)"
        for nested in _nested_models_in_annotation(field.annotation):
            sub = _nested_credential_path(nested, visited)
            if sub is not None:
                return f"{name}.{sub[0]}", sub[1]
    return None


#: Core-schema node kinds that ACCEPT KEYED INPUT and therefore have an
#: extra-keys policy: a pydantic model, a ``TypedDict``, and a pydantic
#: dataclass. All three express the policy the same way on the 2.5 floor and
#: on current pydantic -- ``config.extra_fields_behavior``, ABSENT for the
#: default -- and all three DEFAULT to silently dropping unknown keys.
_KEYED_INPUT_NODE_KINDS: frozenset[str] = frozenset({"model", "typed-dict", "dataclass"})


def _nested_input_surface_gap(cls: type[BaseModel]) -> str | None:
    """Find a nested input container whose extra-keys policy is not ``forbid``.

    Guard 0 closes the config's OWN input surface; this is the same rule
    applied at every depth, because the closure claim is about the whole
    accepted INPUT, not about the top-level mapping: a nested options model
    left on pydantic's default (``extra="ignore"``, and the same default for
    ``TypedDict`` and pydantic dataclasses) accepts a mistyped nested key
    and silently drops it -- the caller reads their setting as applied while
    the engine runs on the field's default, which is exactly the silent
    wrong result the top-level rule exists to prevent. The walk is over the
    CORE SCHEMA, not the annotations, for the reason every proof here is
    (annotations cannot see through containers, unions, or a custom
    ``__get_pydantic_core_schema__``), and it reads the one policy key both
    the 2.5 floor and current pydantic emit (``config.extra_fields_behavior``,
    with the node-level ``extra_behavior`` mirror as fallback). A container
    whose policy cannot be read as ``"forbid"`` is a gap -- including a
    future pydantic that stops emitting the key (fail-closed: the version
    matrix then fails loudly at class definition, never silently).

    ``serialization`` subtrees are skipped (they run on dump, not on input)
    and so is ``metadata`` (JSON-schema generation artifacts); neither
    accepts input. The root node -- the config class itself -- is skipped too: Guard
    0 owns it with its own message.

    Args:
        cls: The config class whose schema is walked.

    Returns:
        A description of the first open nested input container, or ``None``
        when every reachable one forbids undeclared keys.
    """
    try:
        stack: list[tuple[str, object]] = [(cls.__name__, cls.__pydantic_core_schema__)]
        seen: set[int] = set()
        while stack:
            where, current = stack.pop()
            if isinstance(current, dict):
                node = cast("dict[str, Any]", current)
                if id(node) in seen:
                    continue
                seen.add(id(node))
                kind: object = node.get("type")
                held: object = node.get("cls")
                if isinstance(kind, str) and kind in _KEYED_INPUT_NODE_KINDS and held is not cls:
                    name = (
                        held.__name__
                        if isinstance(held, type)
                        else f"the {kind} container at {where}"
                    )
                    node_config: object = node.get("config")
                    policy: object = (
                        cast("dict[str, Any]", node_config).get("extra_fields_behavior")
                        if isinstance(node_config, dict)
                        else None
                    ) or node.get("extra_behavior")
                    if policy != "forbid":
                        return (
                            f"{where} reaches nested input container {name} whose "
                            f"extra-keys policy is {policy!r}, not 'forbid'"
                        )
                for key, value in node.items():
                    # The shared non-input set, not an inline tuple: "default"
                    # matters here too (a dict default spelling {"type":
                    # "model"} read as an open nested input container).
                    if key in _NON_INPUT_SCHEMA_KEYS:
                        continue
                    if key == "fields" and isinstance(value, dict):
                        for field_name, field_node in cast("dict[str, Any]", value).items():
                            stack.append((f"{where}.{field_name}", field_node))
                        continue
                    stack.append((where, value))
            elif isinstance(current, (list, tuple)):
                stack.extend((where, item) for item in cast("Sequence[object]", current))
        return None
    except BaseException:  # noqa: BLE001 - unvettable means open
        return "the model's core schema could not be introspected"


#: Core-schema kinds that accept STRUCTURED input -- a mapping or a
#: sequence. A bare environment string can never be one of these, so a field
#: whose schema reaches one is decoded as JSON first. Listed by kind rather
#: than by annotation ORIGIN because the origin whitelist was demonstrably
#: partial: ``Mapping[str, int]``, ``Sequence[str]``, a ``TypedDict`` and a
#: dataclass all pass every class-definition guard (they are ordinary,
#: fully-declared config shapes) yet had no origin entry, so each was
#: unreachable through its own documented env convention. Widening the
#: whitelist would only postpone the next miss -- named tuples, type
#: aliases, a future pydantic kind -- which is the same fail-open pattern
#: the serialization proof already replaced with a sweep of the artifact
#: that actually runs.
_STRUCTURED_SCHEMA_KINDS: frozenset[str] = frozenset(
    {
        "list",
        "set",
        "frozenset",
        "tuple",
        "tuple-positional",
        "tuple-variable",
        "dict",
        "typed-dict",
        "model",
        "dataclass",
        "generator",
    }
)

#: Core-schema kinds a bare environment string CAN be validated as.
#: ``json`` (the ``Json[T]`` annotation) terminates the walk as RAW: its
#: contract is that the input IS the JSON document text — pydantic's own
#: ``json`` validator decodes it. Pre-decoding (the wrong verdict for
#: ``Json[list[str]]``'s inner ``list``) feeds pydantic the decoded VALUE,
#: which the ``json`` validator rejects, so the field passed the
#: constructor but could not be fed through its own documented env
#: convention. Do NOT descend into the inner document schema: it describes
#: the decoded result, not how the env string arrives. (``json-or-python``
#: is genuinely a wrapper — both its halves still describe Python-side
#: input — so it keeps descending.)
_SCALAR_SCHEMA_KINDS: frozenset[str] = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "decimal",
        "complex",
        "date",
        "datetime",
        "time",
        "timedelta",
        "url",
        "multi-host-url",
        "uuid",
        "enum",
        "literal",
        "json",
    }
)

#: Node keys whose subtree is not part of the field's INPUT shape.
#: ``default`` is DATA, not schema: a ``default`` wrapper node stores the
#: field's literal default value under this key, so a plain dict default
#: whose payload happens to spell ``{"type": "str"}`` (or ``"model"``) would
#: otherwise be misread as a schema node -- a spurious definition-time
#: rejection of a legal config.
_NON_INPUT_SCHEMA_KEYS: frozenset[str] = frozenset({"serialization", "metadata", "default"})


def _env_codec(field_schema: object) -> Literal["raw", "json", "ambiguous"]:
    """Classify how an env string reaches this field, from its core schema.

    An environment variable is always a bare string, so each field is either
    ``"raw"`` (the string IS the value: ``str``, ``int``, ``SecretStr``,
    ``Path``, an enum...) or ``"json"`` (the string is a JSON document to
    decode first: a list, a mapping, a submodel, a ``TypedDict``, a
    dataclass). A ``Json[T]`` field is RAW even though it names a structured
    shape: the annotation's own contract is that the input IS the JSON
    document text (pydantic's ``json`` validator decodes it), consistent
    with the explicit constructor, which takes the string and rejects the
    decoded value. The classification reads the **core schema** -- the artifact
    validation actually executes -- rather than the annotation, so a shape
    that has no annotation ORIGIN to match (``Mapping``/``Sequence``, a
    ``TypedDict``, a dataclass) is classified by what it validates.

    The walk descends through wrappers (``nullable``, ``default``, unions,
    ``json-or-python``, validator functions) and STOPS at the first
    structured kind: a ``dict[str, str | int]``'s member schemas describe
    what is inside the JSON document, not how the document arrives.

    The asymmetry between the two answers is deliberate. Guessing ``"raw"``
    for something structured fails LOUDLY at construction (pydantic rejects
    the string), while guessing ``"json"`` for something scalar silently
    REINTERPRETS it -- ``"123"`` becomes the integer 123, ``"null"`` becomes
    ``None`` -- which is the cardinal sin. So ``"json"`` is returned only on
    positive evidence, and anything unrecognized stays ``"raw"``.

    A field whose schema reaches BOTH (``str | list[str]``) is
    ``"ambiguous"``: no rule can tell whether ``"123"`` is that string or
    that JSON number, and the two paths would disagree with the explicit
    constructor, which always takes the string. The caller rejects it at
    class definition rather than picking one silently.

    Args:
        field_schema: The field's core-schema node.

    Returns:
        ``"raw"``, ``"json"``, or ``"ambiguous"``.
    """
    scalar = False
    structured = False
    stack: list[object] = [field_schema]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            node = cast("dict[str, Any]", current)
            if id(node) in seen:
                continue
            seen.add(id(node))
            kind = node.get("type")
            # pragma-no-branch: with 'default' payloads skipped, every dict
            # this walk visits is a pydantic-core schema node, and those
            # always carry a str "type". Kept as the walk's own guard (a
            # non-str would fall through to the descent loop, fail-closed
            # into "ambiguous", never crash).
            if isinstance(kind, str):  # pragma: no branch
                if kind in _STRUCTURED_SCHEMA_KINDS:
                    structured = True
                    # Do NOT descend: the members describe the document's
                    # CONTENTS, not how the document itself arrives.
                    continue
                if kind in _SCALAR_SCHEMA_KINDS:
                    scalar = True
                    continue
            for key, value in node.items():
                if key in _NON_INPUT_SCHEMA_KEYS:
                    continue
                stack.append(value)
        elif isinstance(current, (list, tuple)):
            stack.extend(cast("Sequence[object]", current))
    if scalar and structured:
        return "ambiguous"
    return "json" if structured else "raw"


def _env_codecs(cls: type[BaseModel]) -> dict[str, Literal["raw", "json"]]:
    """Classify every field of a config class, or raise on an ambiguous one.

    Computed once at class definition (see
    :meth:`BaseConfig.__pydantic_init_subclass__`) so the ambiguity is a
    definition-time error rather than a surprise that depends on which env
    var happens to be set.

    Args:
        cls: The config class.

    Returns:
        The per-field codec, keyed by field name.

    Raises:
        TypeError: If a field's schema reaches both a scalar and a
            structured shape, or if the schema cannot be introspected.
    """
    try:
        fields: dict[str, Any] = {}

        def collect(node: object) -> None:
            if isinstance(node, dict):
                mapping = cast("dict[str, Any]", node)
                if mapping.get("type") == "model-fields" and isinstance(
                    mapping.get("fields"), dict
                ):
                    for name, field_node in cast("dict[str, Any]", mapping["fields"]).items():
                        fields.setdefault(name, field_node)
                for key, value in mapping.items():
                    if key not in _NON_INPUT_SCHEMA_KEYS:
                        collect(value)
            elif isinstance(node, (list, tuple)):
                for item in cast("Sequence[object]", node):
                    collect(item)

        collect(cls.__pydantic_core_schema__)
    except BaseException as exc:  # noqa: BLE001 - unvettable means unusable
        raise TypeError(
            f"{cls.__name__}'s core schema could not be introspected, so the "
            f"env-variable codec for its fields cannot be established. Every "
            f"field must be classifiable as a bare string or a JSON document "
            f"for STANDARD_ASR_<ENGINE>__<FIELD> to be well-defined."
        ) from exc
    codecs: dict[str, Literal["raw", "json"]] = {}
    for name in cls.model_fields:
        codec = _env_codec(fields.get(name))
        if codec == "ambiguous":
            raise TypeError(
                f"{cls.__name__}.{name} accepts BOTH a scalar and a structured "
                f"shape (e.g. str | list[str]), so its "
                f"STANDARD_ASR_<ENGINE>__{name.upper()} value has no defined "
                f"reading. An env string is always a string, so no rule can "
                f"tell whether '123' is that string or that JSON number. "
                f"Decoding it would also disagree with the explicit "
                f"constructor, which always takes the string. Declare one "
                f"shape (a list field, or a scalar field), or model the "
                f"alternatives as a named submodel."
            )
        codecs[name] = codec
    return codecs


def _normalize_segment(value: str) -> str:
    """Normalize an env-var segment: uppercase, non-alphanumeric *runs* to one ``_``.

    A **run** of one-or-more non-alphanumerics collapses to a single ``_`` (not
    one ``_`` per char). This keeps a segment free of ``__``, so the ``__``
    double-underscore engine/field separator (:func:`env_var_name`) stays
    unambiguously parseable: a segment can never itself contain the separator
    sequence (e.g. ``"openai--api"`` -> ``"OPENAI_API"``, never ``"OPENAI__API"``).

    Args:
        value: Engine id or field name segment.

    Returns:
        The normalized uppercase segment.
    """
    return re.sub(r"[^A-Z0-9]+", "_", value.upper())


def env_var_name(engine_id: str, field_name: str) -> str:
    """Return the environment variable name for an engine config field.

    The engine and field segments are joined by a **double underscore**
    (``STANDARD_ASR_<ENGINE>__<FIELD>``) so the boundary between them is
    unambiguous for the realistic name space. With a single-underscore separator
    the engine/field split was not recoverable -- ``env_var_name("openai",
    "api_key")`` and ``env_var_name("openai-api", "key")`` both produced
    ``STANDARD_ASR_OPENAI_API_KEY``, so two different engines could silently read
    each other's credentials. Because :func:`_normalize_segment` collapses each
    non-alphanumeric *run* to a single ``_``, an interior single ``_`` (folded
    from ``-`` / ``.``) can never be mistaken for the ``__`` boundary. This
    relies on ``engine_id`` being entrypoint-derived and PEP 503-normalized (no
    leading/trailing separator) and ``field_name`` being a Python identifier:
    a pathological ``engine_id`` ending in a separator combined with a
    ``field_name`` starting with one is out of that space. Same-class collisions
    (two fields of one config normalizing alike) are still caught by
    :meth:`BaseConfig.env_overrides`.

    Args:
        engine_id: The engine identifier.
        field_name: The standard config field name.

    Returns:
        The fully qualified environment variable name.
    """
    return f"{ENV_PREFIX}_{_normalize_segment(engine_id)}__{_normalize_segment(field_name)}"


#: The serialization methods whose identity :class:`BaseConfig` OWNS. An
#: override of any of these -- direct or inherited from a mixin / intermediate
#: base -- is the same "customization rematerializes a sibling secret" act the
#: serializer-schema guards (5/5b/5c) reject, one dispatch entry over, and is
#: refused at class definition (:func:`_reject_security_method_override`).
#: ``public_dump`` / ``reveal_dump`` are owned by ``BaseConfig``; the pydantic
#: dump primitives and ``__iter__`` (which ``dict(model)`` walks) are owned by
#: ``BaseModel``. Keyed by name; the owning class is resolved at guard time.
_SECURITY_OWNED_METHODS: tuple[str, ...] = (
    "public_dump",
    "reveal_dump",
    "model_dump",
    "model_dump_json",
    "__iter__",
)


def _reject_security_method_override(cls: type) -> None:
    """Reject a subclass overriding a security-owned serialization method.

    :meth:`BaseConfig.public_dump`'s "safe for ``/v1/models``" contract rests
    on TWO proofs: the serializer SCHEMA is closed (guards 5/5b/5c reject
    ``@computed_field`` / ``@model_serializer`` / ``@field_serializer`` /
    ``Annotated`` serializers / ``SerializeAsAny``), AND the DISPATCH is fixed.
    An ordinary Python override of ``model_dump`` / ``model_dump_json`` (or
    ``__iter__``, which ``dict(model)`` walks) never enters the decorator
    registry or the core schema, so the schema proof cannot see it -- yet it
    runs INSIDE ``public_dump`` and can rematerialize a sibling secret under a
    key the by-name mask never touches (``dumped["authorization"] = "Bearer "
    + self.api_key.get_secret_value()``). This closes that entry point.

    Resolution is STATIC (:func:`inspect.getattr_static`, no descriptor
    dispatch) and walks the full MRO, so an override carried by a mixin or an
    intermediate base is caught exactly like a direct one. The runtime dumps
    additionally call the base implementations UNBOUND, so a bypass of this
    definition-time gate still cannot dispatch to a plugin override -- this is
    defense in depth.

    Args:
        cls: The concrete config subclass being defined.

    Raises:
        TypeError: If ``cls`` (or anything in its MRO below the owning base)
            overrides ``public_dump`` / ``reveal_dump`` / ``model_dump`` /
            ``model_dump_json`` / ``__iter__``.
    """
    owners = {
        "public_dump": BaseConfig,
        "reveal_dump": BaseConfig,
        "model_dump": BaseModel,
        "model_dump_json": BaseModel,
        "__iter__": BaseModel,
    }
    for name in _SECURITY_OWNED_METHODS:
        base_impl = inspect.getattr_static(owners[name], name)
        actual = inspect.getattr_static(cls, name, base_impl)
        if actual is not base_impl:
            raise TypeError(
                f"{cls.__name__} overrides the security-owned serialization method "
                f"{name!r}. Config serialization is a CLOSED surface: public_dump "
                f"(masked, for /v1/models/persistence/telemetry) and reveal_dump "
                f"(plaintext, in-process only) are the whole contract, and they rest "
                f"on model_dump / model_dump_json / __iter__ dispatching to pydantic's "
                f"own machinery over the closed schema. An override runs inside that "
                f"dump and can rematerialize a sibling secret under a key the by-name "
                f"mask never sees (the same reason @model_serializer / @computed_field "
                f"/ @field_serializer are rejected). Derive values in a plain @property "
                f"or in the engine, never in the config's serialization."
            )


class BaseConfig(BaseModel, Generic[EngineNameT]):
    """Base class for ASR engine init configuration models.

    Attributes:
        engine: Discriminator equal to the entrypoint-derived ``engine_id``.
        strict: Global policy for unsupported standard parameters. ``True``
            raises ``UnsupportedFeatureError``; ``False`` is best_effort
            (ignore + diagnostic).

    Raises:
        ValueError: If validation fails.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        # Accept fields by their attribute name as well as their alias:
        # env fallback keys by attribute name (e.g. ``api_key``), but a credential
        # may declare a provider-native alias (e.g. ElevenLabs ``xi-api-key``).
        # Without this, loading such a field from env trips ``extra="forbid"``.
        # ``populate_by_name`` (not the newer ``validate_by_name``) is used for
        # pydantic >= 2.5 compatibility (the lower-bounds CI lane).
        populate_by_name=True,
        # Engine configs commonly carry `model_*` fields (e.g. `model_path`).
        # Opt out of pydantic's `model_` protected namespace so subclasses do not
        # warn (the warning fires on older pydantic, e.g. the lower-bounds 2.5).
        protected_namespaces=(),
    )

    engine: EngineNameT = Field(
        ..., description="Engine discriminator (entrypoint-derived engine_id)."
    )
    strict: bool = Field(
        default=True,
        description="Unsupported-parameter policy: True=strict, False=best_effort.",
    )
    allow_private_urls: bool = Field(
        default=False,
        description=(
            "Opt-in to relax the SSRF policy so an AudioUrl may target a "
            "private/loopback/link-local address (HTTPS is still required). "
            "False by default; set True only for a trusted internal endpoint."
        ),
    )

    #: Base fields that MUST NOT be sourced from the environment. Env
    #: fallback covers **every other field** -- the standard config fields
    #: (credentials, endpoint routing, device, language, download root) AND any
    #: engine-declared field (e.g. ``beam_size``, ``model_path``), each gaining a
    #: ``STANDARD_ASR_<ENGINE>__<FIELD>`` entry: env coverage of the
    #: full config surface is intentional DX, not just the mixin fields. Excluded
    #: are only the three fields where an env override would be a silent
    #: security/correctness downgrade: the ``engine`` identity (entrypoint-
    #: derived, never user-set), and the ``strict`` / ``allow_private_urls``
    #: fail-loud safety defaults (the environment must not silently flip
    #: best_effort on or relax the SSRF guard with no diagnostic).
    _ENV_EXCLUDED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"engine", "strict", "allow_private_urls"}
    )

    #: How each field reads its ``STANDARD_ASR_<ENGINE>__<FIELD>`` value: as
    #: the bare string, or as a JSON document to decode first. Derived from
    #: the core schema at class definition (:func:`_env_codecs`), so an
    #: unreadable field is a loud definition-time error rather than a
    #: surprise that depends on which variable happens to be set.
    _ENV_CODECS: ClassVar[dict[str, Literal["raw", "json"]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Enforce definition-time invariants: secret annotations, flat aliases.

        Definition-time guards, so a credential leak or an unclassifiable
        construction failure can never reach runtime:

        0. The input surface stays CLOSED, at every depth. The config itself
           must keep ``extra="forbid"`` (``BaseConfig``'s default; ``allow``
           stores undeclared caller data and emits it from
           :meth:`public_dump`, ``ignore`` silently drops a mistyped key),
           and every nested input container its schema reaches -- a
           submodel, a ``TypedDict``, a pydantic dataclass -- must forbid
           undeclared keys too: pydantic's default for all three silently
           DROPS them, so a typo'd nested option reads as applied while the
           engine runs on the field's default.
        1. A field marked ``secret=True`` (via :func:`secret_field`) must
           resolve to exactly ONE masking carrier (:func:`_secret_carrier`):
           ``SecretStr`` or ``SecretBytes``, optionally unioned with ``None``
           and nothing else. A plain annotation (``str | None``) or a
           plaintext union member (``SecretStr | int``) would be hidden from
           REST/auto-UI while leaking plaintext in ``repr``/``str``/
           ``model_dump``/:meth:`public_dump`; two carriers make raw-string
           wrapping ambiguous; a container of secrets (e.g.
           ``list[SecretStr]``) is only half-protected.
        1b. The field's DEFAULT must uphold the same contract (pydantic does
           not validate defaults): required, ``None`` (only when the
           annotation admits it), or an instance of the carrier itself. A
           plain-string default would live on instances as raw ``str`` --
           plaintext ``repr``/``model_dump`` and a crash in
           ``model_dump_json``/:meth:`public_dump`. A ``default_factory`` is
           rejected outright (it runs at construction, unvetted by this
           guard).
        2. A secret marker on a field of a **nested submodel** (the standard
           encourages per-model-family submodels) is rejected outright. The secret
           pipeline -- the enforcement here, the whitespace-preserving validator,
           and ``public_dump``'s masking -- only operates on a ``BaseConfig``'s
           *own* scalar fields, so a secret nested one level down is silently
           unprotected and leaks plaintext through ``public_dump`` / ``repr`` /
           ``model_dump``. Credentials MUST therefore be modeled as top-level
           scalar ``SecretStr`` fields on the config, not buried in a submodel.
        3. A **non-string validation alias** -- ``AliasPath``, or an
           ``AliasChoices`` carrying one -- is rejected. The whole config
           surface is built on FLAT single-token field resolution: the env
           convention maps one ``STANDARD_ASR_<ENGINE>__<FIELD>`` variable to
           one field name (spec IC.4), and the absent-vs-invalid classifier
           behind :class:`~standard_asr.contract.exceptions.ConfigurationRequiredError`
           resolves each error ``loc`` back to a single string token. A path
           alias populates a field from NESTED input, which neither surface
           can express -- pre-guard it produced an environment-dependent
           compliance verdict (pure absence misclassified as a plugin
           defect). An ``AliasChoices`` of plain strings is fine: each choice
           resolves like a string alias.
        4. The flat input-key vocabulary is UNIQUE across fields: a field's
           alias (or ``AliasChoices`` choice) colliding with another field's
           name or alias would let one caller key silently populate two
           independent settings (``populate_by_name`` fills both) and makes
           loc-token resolution ambiguous. Every key has exactly one owning
           field.
        5. The config's serialization surface is CLOSED: author-defined
           serialization hooks -- ``@computed_field``, ``@model_serializer``,
           ``@field_serializer`` (declared here or inherited), and
           ``PlainSerializer``/``WrapSerializer`` metadata on a field --
           are rejected outright. They run INSIDE ``model_dump``, i.e.
           inside :meth:`public_dump`'s "masked" serialization, where they
           can rematerialize a sibling secret in plaintext (a computed
           ``authorization`` property, a ``field_serializer`` reading
           ``self.api_key``) under keys the by-name mask never touches --
           and they break the dump-is-the-declared-input-surface contract
           (G1.3/G3.1: auto-UI renders ``model_fields``; ``extra="forbid"``
           rejects a computed key on reload). Derived values belong on a
           plain ``@property`` or in the engine, never in the dump.
        6. A field whose annotation contains a secret CARRIER
           (``SecretStr``/``SecretBytes``, anywhere in it) must carry the
           secret MARKER: an unmarked carrier is half-protected -- pydantic
           masks its dumps, but the whitespace-preserving validator skips
           it (the raw string input is silently stripped: a padded
           credential is rewritten with no diagnostic), and the schema
           never renders it as a password/write-only input.
        8. The security-owned serialization methods keep their IDENTITY:
           ``public_dump`` / ``reveal_dump`` (owned here) and
           ``model_dump`` / ``model_dump_json`` / ``__iter__`` (owned by
           ``BaseModel``) MUST NOT be overridden -- directly or through a
           mixin / intermediate base. Guard 5 closes the serializer SCHEMA,
           but an ordinary Python override of these is the same
           customization one DISPATCH entry over: it never enters the
           decorator registry or the core schema, yet it runs inside
           ``public_dump`` and can rematerialize a sibling secret under a
           key the by-name mask never touches. Resolved statically over the
           MRO; the runtime dumps also call the base implementations unbound
           (defense in depth). See :func:`_reject_security_method_override`.

        Args:
            **kwargs: Forwarded subclass keyword arguments.

        Raises:
            TypeError: If a secret-marked field's annotation does not resolve
                to exactly one carrier (``SecretStr``/``SecretBytes``,
                optionally with ``None``), if its default violates the
                carrier contract (plain value, ``None`` on a non-optional
                annotation, or a ``default_factory``), if a nested submodel
                reachable from any field carries a secret-marked or
                carrier-annotated field, if a field's annotation contains an
                unresolved forward reference the nested-secret scan cannot
                vet, if a field declares a non-string validation alias
                (``AliasPath`` / ``AliasChoices`` with a non-string entry),
                if two fields claim the same flat input key, if the class
                declares or inherits an author serialization hook, if the
                class overrides a security-owned serialization method
                (``public_dump`` / ``reveal_dump`` / ``model_dump`` /
                ``model_dump_json`` / ``__iter__``), if a
                carrier-annotated field lacks the secret marker, if the
                class reopens its own input surface (``extra != "forbid"``),
                if a nested input container reachable from the schema
                does not forbid undeclared keys, or if a field's env value
                has no defined reading -- its schema reaches both a scalar
                and a structured shape, or the core schema cannot be
                introspected at all (Guard 7, ``_env_codecs``).
        """
        super().__pydantic_init_subclass__(**kwargs)
        # Guard 0: the input surface stays CLOSED. ``extra="forbid"`` is not
        # a stylistic default here -- the flat input-key vocabulary (guard 4),
        # the absent-vs-invalid classifier behind
        # ``ConfigurationRequiredError``, and the typo-names-the-key DX all
        # rest on every accepted key belonging to a declared field. Reopening
        # it breaks all three AND public_dump's mask, which can only mask
        # fields it knows: ``allow`` stores a misplaced credential and emits
        # it verbatim, while ``ignore`` silently swallows a mistyped
        # credential key and lets the engine run unauthenticated.
        extra_behavior = cls.model_config.get("extra")
        if extra_behavior != "forbid":
            raise TypeError(
                f"{cls.__name__} sets model_config extra={extra_behavior!r}; a config's "
                f"input surface must stay closed (extra='forbid', BaseConfig's default). "
                f"'allow' stores undeclared caller data on the instance and emits it from "
                f"public_dump, where the by-name secret mask cannot reach it; 'ignore' "
                f"silently drops a mistyped key (a typo'd credential then reads as absent). "
                f"Declare every accepted key as a field."
            )
        # Guard 5 (class level): the serialization surface is closed. The
        # decorator registry aggregates inherited hooks too, so a hook
        # smuggled in through a mixin base is caught the same way.
        decorators = cls.__pydantic_decorators__
        hooks = [
            *(f"@computed_field {name!r}" for name in decorators.computed_fields),
            *(f"@model_serializer {name!r}" for name in decorators.model_serializers),
            *(f"@field_serializer {name!r}" for name in decorators.field_serializers),
        ]
        if hooks:
            raise TypeError(
                f"{cls.__name__} declares (or inherits) author serialization "
                f"hooks: {', '.join(sorted(hooks))}. A config's serialization "
                f"surface is closed: these hooks run inside model_dump -- i.e. "
                f"inside public_dump's masked serialization for /v1/models, "
                f"persistence, and telemetry -- where they can rematerialize a "
                f"sibling secret in plaintext under keys the by-name mask never "
                f"touches, and their output diverges from the declared input "
                f"fields (extra='forbid' rejects such keys on reload; auto-UI "
                f"renders model_fields). Use a plain @property for in-process "
                f"convenience values, or derive the value in the engine."
            )
        for name, field in cls.model_fields.items():
            # Guard 5c (field level): Annotated serializer metadata is the
            # same open lane as the decorators -- it rewrites the dumped
            # value inside public_dump's "masked" serialization. (The schema
            # proof above already rejects it; this arm names the annotation.)
            if any(isinstance(meta, (PlainSerializer, WrapSerializer)) for meta in field.metadata):
                raise TypeError(
                    f"{cls.__name__}.{name} carries PlainSerializer/WrapSerializer "
                    f"annotation metadata. A config's serialization surface is "
                    f"closed (the dump IS the declared input surface, and for a "
                    f"secret field the serializer receives the unwrapped carrier); "
                    f"drop the serializer -- model_dump(mode='json') already "
                    f"renders every supported input type."
                )
            # Guard 5d (enumerated, accident-grade -- the trust-model
            # replacement for the deleted core-schema closure proof): the
            # DECLARED annotation must not carry an undeclared value shape,
            # a SerializeAsAny marker, or a nested submodel with its own
            # serialization hooks. See _annotation_serialization_gap.
            if any(
                meta is SerializeAsAny or isinstance(meta, cast("type", SerializeAsAny))
                for meta in field.metadata
            ):
                raise TypeError(
                    f"{cls.__name__}.{name} declares a DUCK-TYPED serialization "
                    f"marker (SerializeAsAny): the dump then follows the RUNTIME "
                    f"object, so the declared type no longer bounds what "
                    f"public_dump emits (a runtime subclass's computed field can "
                    f"rematerialize a credential under a key the by-name mask "
                    f"never looks at). Declare the exact model you mean to dump."
                )
            if field.exclude:
                raise TypeError(
                    f"{cls.__name__}.{name} is excluded from the dump "
                    f"(exclude=True). public_dump is documented for "
                    f"persistence: silently losing a declared input on dump "
                    f"means reloading yields a different config with no "
                    f"diagnostic. Drop exclude=True, or keep the value out of "
                    f"the config."
                )
            shape_gap = _annotation_serialization_gap(field.annotation)
            if shape_gap is not None:
                raise TypeError(
                    f"{cls.__name__}.{name} declares {shape_gap}. A config's dump "
                    f"is the declared input surface (public_dump for /v1/models, "
                    f"persistence, telemetry; the JSON Schema renders settings "
                    f"UIs; env values need a defined reading), so every field "
                    f"must be a fully declared shape with no author serialization "
                    f"hooks. Declare a typed submodel or container, and derive "
                    f"values in a @property or in the engine."
                )
            # Guard 6: a carrier anywhere in the annotation requires the
            # MARKER -- an unmarked SecretStr field skips the whitespace-
            # preserving validator, so its raw string input is silently
            # stripped (a padded credential rewritten with no diagnostic),
            # and the schema never renders it write-only.
            if not _is_secret_marked(field) and _annotation_contains_secret_carrier(
                field.annotation
            ):
                raise TypeError(
                    f"{cls.__name__}.{name} is annotated with a secret carrier "
                    f"(SecretStr/SecretBytes) but is not marked secret. Declare "
                    f"it with secret_field(...) so it joins the secret pipeline "
                    f"(exact-contents whitespace preservation, password/"
                    f"write-only schema, public_dump masking); an unmarked "
                    f"carrier silently strips credential whitespace and renders "
                    f"as an ordinary input in auto-UI."
                )
            if _is_secret_marked(field):
                carrier = _secret_carrier(field.annotation)
                if carrier is None:
                    raise TypeError(
                        f"{cls.__name__}.{name} is marked secret (secret_field) but its "
                        f"annotation {field.annotation!r} is not exactly SecretStr or "
                        f"SecretBytes (optionally unioned with None, and with nothing "
                        f"else). A plaintext union member (e.g. SecretStr | int) lets a "
                        f"constructed value bypass masking -- schema says password, "
                        f"repr/model_dump emit plaintext; two carriers "
                        f"(SecretStr | SecretBytes) make raw-string wrapping ambiguous; "
                        f"containers of secrets (e.g. list[SecretStr]) are not masked "
                        f"by the secret pipeline; model multiple credentials as "
                        f"separate scalar fields."
                    )
                # Guard 1b: the DEFAULT must uphold the carrier contract too.
                # pydantic does not validate defaults (validate_default is
                # off, deliberately -- it would silently coerce), so a plain
                # string default would live on instances as raw str: plaintext
                # in repr/model_dump, and a crash in model_dump_json /
                # public_dump when the secret serializer calls
                # get_secret_value() on it. Legal defaults: required
                # (secret_field(default=...)), None (when the annotation
                # admits it), or an instance of the carrier itself.
                if field.default_factory is not None:
                    raise TypeError(
                        f"{cls.__name__}.{name} is marked secret (secret_field) but "
                        f"declares a default_factory. Factories run at construction "
                        f"and are not vetted by this definition-time guard, so a "
                        f"factory returning plaintext would bypass masking. Use a "
                        f"{carrier.__name__} instance default, None, or make the "
                        f"field required (secret_field(default=...))."
                    )
                default = field.default
                if default is None:
                    if not _annotation_admits_none(field.annotation):
                        raise TypeError(
                            f"{cls.__name__}.{name} is marked secret (secret_field) "
                            f"with default None, but its annotation "
                            f"{field.annotation!r} does not admit None -- instances "
                            f"built without the field would silently violate the "
                            f"annotation (defaults are not validated). Union the "
                            f"annotation with None for an optional credential, or "
                            f"make the field required (secret_field(default=...))."
                        )
                elif default is not PydanticUndefined and type(default) is not carrier:
                    # The default's VALUE is never echoed here: it is the very
                    # plaintext credential this guard exists to protect, and
                    # definition-time TypeErrors land in CI logs.
                    raise TypeError(
                        f"{cls.__name__}.{name} is marked secret (secret_field) but "
                        f"its default is a {type(default).__name__}, not a "
                        f"{carrier.__name__} instance. Defaults are not validated, "
                        f"so a plaintext default leaks through repr/model_dump and "
                        f"crashes the secret serializer (model_dump_json/"
                        f"public_dump). Wrap it: "
                        f"secret_field(default={carrier.__name__}(...))."
                    )
            # Guard 2a: fail closed on an annotation pydantic could not resolve at
            # hook time (a define-after submodel left as a bare ForwardRef). Guard 2b
            # below scans only RESOLVED nested models, so an unresolved ref would
            # pass fail-open and a secret buried in that submodel would leak.
            if _annotation_has_unresolved_ref(field.annotation):
                raise TypeError(
                    f"{cls.__name__}.{name} has an unresolved forward-reference "
                    f"annotation {field.annotation!r}, so the nested-secret guard "
                    f"cannot verify no credential is buried in it. Define the submodel "
                    f"BEFORE this config (or import it) so it resolves, and keep "
                    f"credentials as top-level scalar SecretStr fields."
                )
            # Guard 2b: reject a credential buried in a nested submodel --
            # whether secret-MARKED or a bare CARRIER annotation. Neither is
            # threaded through this hook (nested models are not BaseConfig
            # subclasses) nor masked by public_dump's by-name pass; a marked
            # one leaks plaintext outright, and a bare carrier is
            # half-broken (whitespace silently stripped, unreachable by
            # reveal_dump, unwrappable by the submodel's own hooks).
            for nested in _nested_models_in_annotation(field.annotation):
                leak = _nested_credential_path(nested)
                if leak is not None:
                    leak_path, leak_reason = leak
                    raise TypeError(
                        f"{cls.__name__}.{name} reaches a nested submodel whose field "
                        f"{nested.__name__}.{leak_path} {leak_reason}. "
                        f"The secret pipeline (exact-contents whitespace "
                        f"preservation, public_dump masking, reveal_dump "
                        f"unwrapping) operates only on a BaseConfig's own scalar "
                        f"fields, so a credential nested in a submodel is "
                        f"unprotected or unreachable. Promote it to a top-level "
                        f"scalar SecretStr field (secret_field) on {cls.__name__}."
                    )
            # Guard 3: flat aliases only (see the docstring). Rejecting the
            # non-string forms HERE keeps the runtime classifier's
            # single-token loc resolution sound instead of guessing at error
            # time what a path alias's failure means.
            va = field.validation_alias
            if va is None or isinstance(va, str):
                continue
            if isinstance(va, AliasChoices):
                if all(isinstance(choice, str) for choice in va.choices):
                    continue
                raise TypeError(
                    f"{cls.__name__}.{name}: AliasChoices with a non-string entry "
                    "(e.g. AliasPath) is unsupported on BaseConfig -- the flat "
                    "STANDARD_ASR_<ENGINE>__<FIELD> env mapping (spec IC.4) and "
                    "the absent-vs-invalid config classifier resolve every "
                    "failure to a single string token, which a path choice "
                    "cannot provide. Use plain string aliases only."
                )
            raise TypeError(
                f"{cls.__name__}.{name}: validation_alias={type(va).__name__} is "
                "unsupported on BaseConfig -- nested/indexed population cannot "
                "map onto the flat STANDARD_ASR_<ENGINE>__<FIELD> env convention "
                "(spec IC.4) or the absent-vs-invalid config classifier. Declare "
                "the nested value as a submodel field (JSON env values are "
                "supported) or use a plain string alias."
            )
        # Guard 4: the flat input-key vocabulary is UNIQUE across fields. With
        # populate_by_name, pydantic happily fills TWO fields from one input
        # key when a field's alias collides with another field's name (or
        # alias) -- one caller key silently controlling two independent
        # settings. It also breaks _canonical_field_name's resolution (a
        # token that is one field's canonical name AND another's alias is
        # ambiguous, but the canonical short-circuit would pick one). Every
        # key therefore has exactly one owning field, enforced here so the
        # runtime never has to disambiguate.
        key_owner: dict[str, str] = {}
        for name, field in cls.model_fields.items():
            for key in cls._flat_input_keys(name, field):
                owner = key_owner.setdefault(key, name)
                if owner != name:
                    raise TypeError(
                        f"{cls.__name__}: input key {key!r} is claimed by both "
                        f"field {owner!r} and field {name!r} (as canonical name, "
                        f"alias, validation_alias, or an AliasChoices choice). "
                        f"One input key populating two fields lets a single "
                        f"caller value silently control two independent "
                        f"settings and makes loc-token resolution ambiguous; "
                        f"every key must belong to exactly one field."
                    )
        # NOTE (trust model, AGENTS.md): there is deliberately no TOTAL
        # schema proof behind public_dump anymore. Guards 5/5c above enumerate
        # the authored serializer shapes (@computed_field / @model_serializer /
        # @field_serializer / Annotated serializer metadata); the channels only
        # a core-schema prover could see (SerializeAsAny, a custom
        # __get_pydantic_core_schema__) require an author actively smuggling a
        # credential past the mask -- an adversary, out of scope by definition.
        # Guard 0, applied at DEPTH (after 5c, whose dump-channel message owns
        # the extra='allow' case): every nested input container -- a submodel,
        # a TypedDict, a pydantic dataclass -- must forbid undeclared keys.
        # pydantic's default for all three is to silently DROP them, so a
        # typo'd nested option reads as applied while the engine runs on the
        # field's default: the silent wrong result Guard 0 exists to prevent,
        # one level down.
        # Guard 7: every field's env value has a DEFINED reading. Classified
        # from the core schema (not the annotation's origin, which was
        # demonstrably partial), and an ambiguous field -- one accepting both
        # a scalar and a structured shape -- is refused here rather than
        # silently reinterpreted at whichever env var happens to be set.
        cls._ENV_CODECS = _env_codecs(cls)
        gap = _nested_input_surface_gap(cls)
        if gap is not None:
            raise TypeError(
                f"{cls.__name__}'s input surface is not closed: {gap}. "
                f"pydantic's default ('ignore') silently drops an undeclared "
                f"nested key -- a caller's typo'd option (or a misplaced "
                f"credential key) vanishes with no diagnostic and the engine "
                f"runs on the field's default. Close the container: "
                f"model_config = ConfigDict(extra='forbid') on a submodel, "
                f"__pydantic_config__ = ConfigDict(extra='forbid') on a "
                f"TypedDict, or dataclass(config=ConfigDict(extra='forbid'))."
            )
        # Guard 8: the SECURITY-OWNED serialization methods keep their identity.
        # Guards 5/5b/5c close the pydantic serializer SCHEMA (decorators,
        # Annotated serializers, SerializeAsAny, ...), but public_dump's "safe
        # for /v1/models" contract also rests on the DISPATCH: an ordinary
        # Python override of model_dump / model_dump_json (or __iter__, which
        # reveal_dump once walked) is the SAME "customization rematerializes a
        # sibling secret" act, one entry point over -- it never touches the
        # decorator registry or the core schema, so the closure proof cannot
        # see it. A plugin adding ``dumped["authorization"] = "Bearer " +
        # self.api_key.get_secret_value()`` in a model_dump override leaked the
        # plaintext through public_dump under a key the by-name mask never
        # looks at. Resolved statically (getattr_static: no descriptor
        # dispatch), so a mixin/intermediate base that carries the override is
        # caught the same as a direct one. The runtime dumps ALSO call the base
        # implementations unbound (belt and braces), so this is defense in
        # depth, not the sole gate.
        _reject_security_method_override(cls)

    @model_validator(mode="before")
    @classmethod
    def _preserve_secret_whitespace(cls, data: Any) -> Any:
        """Wrap raw secret strings before global whitespace stripping.

        ``str_strip_whitespace=True`` silently trims every plain ``str`` input,
        including a raw credential passed via ``from_env`` (which hands the
        constructor a plain ``str`` that pydantic strips *before* coercing it
        to the secret type). Trimming a credential can mask a paste error and
        produce a silently-wrong secret. Running before field validation, this
        wraps any raw ``str`` destined for a secret-marked field into the
        field's own carrier first (``SecretStr``, or ``SecretBytes`` via
        UTF-8 -- mirroring pydantic's lax ``str -> bytes`` coercion), so its
        contents bypass stripping; non-secret routing fields (``base_url``
        etc.) keep the convenience strip.

        The wrap only sees MAPPING input, so the input domain is closed to
        mappings here: ``model_validate(obj, from_attributes=True)`` would
        otherwise extract raw attribute strings BEHIND this validator's back
        and run them through ``str_strip_whitespace`` -- the exact silent
        credential rewrite this validator exists to prevent (a padded
        ``api_key`` attribute came out trimmed, with no diagnostic). No
        legitimate construction path loses anything: keyword construction,
        ``model_validate`` on any mapping (``MappingProxyType`` included),
        ``model_validate_json``, and env fallback all arrive as mappings, and
        pydantic short-circuits an already-validated instance before any
        validator runs (under ``revalidate_instances`` it re-presents the
        instance's ``__dict__`` -- a mapping whose secrets are already
        carrier-wrapped). Attribute extraction is simply not part of the
        config input contract.

        Args:
            data: The raw constructor input (a mapping when called positionally
                with keyword data, or whatever a caller handed to
                ``model_validate``).

        Returns:
            A shallow copy of the input mapping with raw secret strings wrapped
            (the caller's original mapping is never mutated).

        Raises:
            PydanticCustomError: If the input is not a mapping (surfaces to the
                caller inside the usual ``ValidationError``). The message is a
                fixed authored string -- it never echoes the input.
        """
        if not isinstance(data, Mapping):
            # Mapping, not dict: ``model_validate`` accepts any mapping
            # (``MappingProxyType``, immutable config views), and gating on
            # ``dict`` silently skipped the wrap for those -- the credential
            # then went through ``str_strip_whitespace`` and lost its exact
            # contents, the precise silent rewrite this validator forbids.
            raise PydanticCustomError(
                "standard_asr_config_mapping_required",
                "Configuration input must be a mapping of field names to "
                "values. Attribute extraction (from_attributes=True) is not "
                "part of the config input contract: it bypasses the secret "
                "whitespace-preservation wrap, so a padded credential would "
                "be silently trimmed. Pass a dict (or any Mapping) instead.",
            )
        # Operate on a shallow copy so a caller's input mapping is never
        # mutated (no spooky action at a distance): e.g.
        # ``Cloud.model_validate(d)`` must leave ``d['api_key']`` the plain
        # str the caller passed, not silently swap it for a SecretStr in
        # their mapping (a read-only Mapping also becomes writable this
        # way). A shallow copy is sufficient: we only rebind whole values
        # (str -> SecretStr), never mutate nested objects.
        mapping = dict(cast("Mapping[Any, Any]", data))
        for name, field in cls.model_fields.items():
            if not _is_secret_marked(field):
                continue
            carrier = _secret_carrier(field.annotation)
            if carrier is None:  # pragma: no cover - definition guard rejects it
                continue
            # Wrap EVERY input key this field can be populated from -- the
            # canonical name, a string alias, a string validation_alias, or
            # any AliasChoices choice (the definition-time guard admits only
            # all-string choices). Wrapping just one key was a real leak of
            # bytes-fidelity: with both an alias key and the canonical name
            # present, pydantic populates from the alias, so wrapping only
            # the canonical value left the winning raw string subject to
            # str_strip_whitespace's silent trim.
            #
            # The wrap targets the field's OWN carrier: wrapping a raw string
            # into SecretStr for a SecretBytes field made every env/alias
            # string construction fail (SecretBytes rejects SecretStr), while
            # plain pydantic would have coerced the bare str. UTF-8 mirrors
            # pydantic's own lax str->bytes coercion, with the whitespace
            # preserved exactly.
            for key in cls._flat_input_keys(name, field):
                if key not in mapping:
                    continue
                value = mapping[key]
                if isinstance(value, str):
                    mapping[key] = (
                        SecretStr(value)
                        if carrier is SecretStr
                        else SecretBytes(value.encode("utf-8"))
                    )
                elif isinstance(value, (bytes, bytearray)):
                    # Bytes-like credentials get the same exact-contents
                    # contract as strings: pydantic's lax bytes -> str
                    # coercion for a SecretStr field runs the DECODED text
                    # through str_strip_whitespace, silently trimming a
                    # padded credential exactly like the raw-str path this
                    # validator already guards.
                    if carrier is SecretBytes:
                        mapping[key] = SecretBytes(bytes(value))
                    else:
                        try:
                            decoded = bytes(value).decode("utf-8")
                        except UnicodeDecodeError:
                            # Not UTF-8: leave the raw value for pydantic's
                            # own coercion to reject loudly (never guess an
                            # encoding for a credential).
                            continue
                        mapping[key] = SecretStr(decoded)
        return mapping

    @classmethod
    def model_validate_json(
        cls: type[_ConfigT],
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: Any | None = None,
        **kwargs: Any,
    ) -> _ConfigT:
        """Validate a JSON document against this config (python-mode delegation).

        Overridden because pydantic's native JSON pipeline is incompatible
        with the whitespace-preserving secret wrap: the wrap replaces a raw
        secret string with its carrier INSTANCE (``SecretStr``), which
        JSON-mode field validation rejects outright (its secret schema
        accepts only the JSON string form) -- so ``model_validate_json``
        failed for EVERY config document carrying a secret value. Skipping
        the wrap in JSON mode is not an option either:
        ``str_strip_whitespace`` applies in JSON mode too, so the padded
        credential would be silently trimmed -- the exact rewrite the wrap
        exists to prevent. The one path that preserves both contracts is to
        parse the document and validate the resulting mapping in python
        mode, where the wrap is defined.

        Grammar parity holds because :func:`json.loads` and pydantic's
        parser accept the same token set for the config value space (both
        accept ``NaN``/``Infinity`` tokens; field validation then treats the
        resulting float identically on both routes -- pinned by test). A
        document :func:`json.loads` rejects is delegated to pydantic's own
        JSON parser so the canonical ``json_invalid`` error surfaces; that
        delegation is REQUIRED to raise -- if the parsers ever diverged and
        pydantic accepted such a document, its JSON field pipeline would
        run without the wrap (the silent trim again), so a delegation that
        returns is refused outright (fail closed) rather than handed back.

        Args:
            json_data: The JSON document.
            strict: Field strictness, applied by the python-mode validation
                of the parsed document.
            context: Validation context, forwarded unchanged.
            **kwargs: Any further ``model_validate`` keyword arguments a
                newer pydantic accepts (e.g. ``by_alias``), forwarded
                unchanged.

        Returns:
            The validated config.

        Raises:
            ValidationError: If the document is malformed JSON (pydantic's
                canonical ``json_invalid`` error), is not a JSON object
                (the config mapping-required error), or fails field
                validation.
            ConfigError: If pydantic's parser accepted a document
                :func:`json.loads` rejected (a parser divergence this
                override refuses to validate around).
        """
        try:
            parsed: Any = json.loads(json_data)
        except ValueError:
            # Malformed JSON: pydantic's own parser produces the canonical
            # json_invalid ValidationError (parsing fails before any field
            # validation, so the wrap-incompatible JSON field pipeline is
            # never reached for the raising path this branch requires).
            super().model_validate_json(json_data, strict=strict, context=context, **kwargs)
            # Reaching here needs a document pydantic's parser accepts and
            # json.loads rejects -- no such document is known (kept as
            # fail-closed containment rather than reached in tests).
            raise ConfigError(  # pragma: no cover
                "Internal parser divergence: pydantic's JSON parser accepted a "
                "configuration document the standard layer's parser rejected. "
                "Refusing to return a result validated without the secret "
                "whitespace-preservation wrap."
            )
        return cls.model_validate(parsed, strict=strict, context=context, **kwargs)

    @classmethod
    def model_validate_strings(
        cls: type[_ConfigT],
        obj: Any,
        *,
        strict: bool | None = None,
        context: Any | None = None,
        **kwargs: Any,
    ) -> _ConfigT:
        """Validate string-valued data against this config (env-grammar delegation).

        Overridden for the same reason as :meth:`model_validate_json`:
        pydantic's native strings pipeline is incompatible with the
        whitespace-preserving secret wrap -- the wrap replaces a raw secret
        string with its carrier INSTANCE (``SecretStr``), which strings-mode
        field validation rejects (its secret schema accepts only the string
        form) -- so ``model_validate_strings`` failed for EVERY config
        supplying a secret value, blaming the caller's own credential field
        for a wrong type it never passed. Skipping the wrap is not an option
        either: ``str_strip_whitespace`` applies in strings mode too, so the
        padded credential would be silently trimmed.

        The string grammar is the config surface's OWN, shared with
        :meth:`from_env` / :meth:`env_overrides` rather than pydantic's
        strings mode: a scalar field takes the raw string (python-mode lax
        coercion handles ``"4"`` / ``"true"``), a STRUCTURED field (list /
        mapping / submodel -- the fields ``_ENV_CODECS`` marks ``"json"``)
        takes a JSON document, and on a JSON error the raw string is kept so
        construction still fails loudly. One grammar for every string-valued
        source (env vars, CLI ``key=value`` pairs, query params), not two
        subtly different ones.

        Args:
            obj: The string-valued mapping; keys may use any of a field's
                flat input keys (canonical name, string alias/
                ``validation_alias``, an ``AliasChoices`` choice).
            strict: Field strictness, applied by the python-mode validation
                of the decoded mapping. ``strict=True`` therefore rejects
                the string spelling of a non-string scalar (``"4"`` for an
                ``int`` field) -- leave it unset for string-valued sources,
                exactly as ``from_env`` does.
            context: Validation context, forwarded unchanged.
            **kwargs: Any further ``model_validate`` keyword arguments a
                newer pydantic accepts, forwarded unchanged.

        Returns:
            The validated config.

        Raises:
            ValidationError: If ``obj`` is not a mapping (the config
                mapping-required error) or the decoded mapping fails field
                validation.
        """
        if not isinstance(obj, Mapping):
            # Not a mapping: python-mode validation surfaces the canonical
            # mapping-required error (the wrap's own input-domain guard).
            return cls.model_validate(obj, strict=strict, context=context, **kwargs)
        key_to_field = {
            key: name
            for name, field in cls.model_fields.items()
            for key in cls._flat_input_keys(name, field)
        }
        decoded = dict(cast("Mapping[Any, Any]", obj))
        for key, value in decoded.items():
            field_name = key_to_field.get(key)
            if field_name is None or not isinstance(value, str):
                # An unknown key fails loudly downstream (extra="forbid");
                # a non-string value is outside the strings contract and is
                # judged as-is by field validation, never reinterpreted.
                continue
            if cls._ENV_CODECS.get(field_name, "raw") == "json":
                try:
                    decoded[key] = json.loads(value)
                except json.JSONDecodeError:
                    # Keep the raw string: construction fails loudly with
                    # the field named (never silently drop).
                    continue
        return cls.model_validate(decoded, strict=strict, context=context, **kwargs)

    def public_dump(self) -> dict[str, Any]:
        """Return a serialization with secrets masked (the default path).

        This is the **masked** half of the secret-serialization contract and is
        the serialization to use for ``/v1/models``, persistence, and telemetry.
        ``SecretStr`` / ``SecretBytes`` fields are rendered as
        :data:`SECRET_MASK` (never plaintext). As a defensive measure, **any**
        secret-marked field is masked by name, so even a value that
        (hypothetically) slipped through as plaintext is never emitted. The
        default pydantic serializers (``model_dump`` / ``model_dump_json``)
        likewise mask the secret carriers; use :meth:`reveal_dump` only when
        plaintext is genuinely required in-process.

        Trusting ``model_dump`` here rests on the definition-time guards
        that reject the authored serializer shapes -- the three serializer
        decorators, ``Annotated`` serializer metadata, nested credential
        carriers, fields excluded from the dump (Guards 5/5c/6) -- so the
        dump contains the declared input fields, serialized by pydantic's
        own machinery. Per the trust model (AGENTS.md) this is an
        enumeration, not a proof: the channels only a core-schema prover
        could see (``SerializeAsAny``, a custom
        ``__get_pydantic_core_schema__``) require an author actively
        smuggling a credential past the mask -- an adversary, out of scope.

        **The guards bound what the schema installs in the dump, not the
        CONTENTS of the values author code constructed.** Config code that
        copies a secret out of its carrier into non-secret state -- an
        after-validator writing ``self.api_key.get_secret_value()`` into a
        declared ``str`` field, or onto display state a trusted serializer
        renders by builtin dispatch (a validator-returned ``Path`` SUBCLASS
        whose ``__str__`` embeds the credential; pydantic's own audited
        ``ser_path`` calls ``str()`` on the runtime value) -- emits that
        plaintext here, and no serialization mechanism can prevent it: both
        shapes are the same author act, and the leaking value passes every
        type or dispatch audit (a plain ``str`` whose content nothing can
        classify as secret). The value-level envelope is and remains the
        carrier contract (IC.3): credentials live in secret-marked
        ``SecretStr``/``SecretBytes`` fields, are masked here by name, and
        MUST NOT be copied out of the carrier into any other field or
        object state. The boundary is pinned executable in
        ``test_secret_extraction_is_the_closure_boundary``; moving it means
        updating this contract, the spec, and that test together.

        Returns:
            A JSON-safe dict with credentials masked.
        """
        # Unbound BaseModel.model_dump, not self.model_dump: the call still
        # uses THIS instance's serializer / core schema, but never dispatches
        # to a subclass's Python model_dump override -- which could add a
        # rematerialized secret under a key the by-name mask never checks.
        # Guard 8 already refuses such an override at definition; this is the
        # matching runtime half (defense in depth).
        dumped = BaseModel.model_dump(self, mode="json")
        for name, field in type(self).model_fields.items():
            if _is_secret_marked(field) and dumped.get(name) is not None:
                dumped[name] = SECRET_MASK
        return dumped

    def reveal_dump(self) -> dict[str, Any]:
        """Return a serialization with secrets materialized as plaintext.

        This is the **reveal** half of the secret-serialization contract: the
        explicit, named counterpart to :meth:`public_dump`. Use it **only** for
        in-process calls into an engine SDK that needs the raw credential (e.g.
        an ``Authorization`` header). The result contains plaintext secrets and
        MUST NEVER be logged, persisted, sent to ``/v1/models``, or emitted as
        telemetry -- those paths use :meth:`public_dump`.

        ``SecretStr`` / ``SecretBytes`` fields are unwrapped via
        ``get_secret_value()``; all other fields keep their Python values (no
        JSON coercion), so a credential is returned as the engine SDK expects it.

        Returns:
            A dict with secret-marked fields materialized to plaintext.
        """
        # Read the DECLARED fields straight from the instance state, not
        # dict(self): the latter walks __iter__, which a subclass could
        # override to inject an extra key (Guard 8 refuses that override, and
        # this is the matching runtime half). model_fields is the closed
        # declared surface; extra="forbid" keeps __dict__ to exactly it.
        state = self.__dict__
        revealed: dict[str, Any] = {}
        for name in type(self).model_fields:
            value: Any = state.get(name)
            if isinstance(value, _SECRET_TYPES):
                revealed[name] = value.get_secret_value()
            else:
                revealed[name] = value
        return revealed

    @classmethod
    def from_env(
        cls: type[_ConfigT],
        engine_id: str,
        *,
        environ: Mapping[str, str] | None = None,
        **explicit: Any,
    ) -> _ConfigT:
        """Construct a config, filling unset fields from the environment.

        Applies the normative priority **explicit > env > (required-missing
        error)**: each field that is not supplied in ``explicit`` under **any
        of its flat input keys** (canonical name, string alias/
        ``validation_alias``, or an ``AliasChoices`` choice -- the
        :meth:`_flat_input_keys` vocabulary) is filled from its
        ``STANDARD_ASR_<NORMENGINE>__<NORMFIELD>`` environment variable
        (collision detected), and the merged mapping is then passed to the
        constructor. Alias-awareness is what makes "explicit wins" true for
        aliased fields: a caller passing ``apiKey=...`` suppresses the
        ``api_key`` env fallback instead of colliding with it under
        ``extra="forbid"``. Because construction does the field coercion,
        ``SecretStr`` credentials are wrapped (and so masked in
        ``repr``/``str``/``public_dump``) instead of being handed around as raw
        plaintext -- avoiding the leak footgun of passing a plaintext
        ``{field: secret}`` dict through application code.

        Note on explicit ``None``: "absent" means the key is **not present** in
        ``explicit``, not "present with value ``None``". A key passed explicitly
        as ``None`` is a value and wins over env (priority is "explicit wins",
        not "explicit-non-None wins"). A wrapper that forwards optional kwargs
        with ``None`` defaults therefore disables the env fallback for those
        fields; drop ``None`` keys before calling (``{k: v for k, v in kwargs
        if v is not None}``) if env fallback should apply.

        The ``engine`` discriminator is never read from the environment; it is
        the entrypoint-derived identity and defaults on each engine's subclass.
        The ``strict`` safety policy is likewise excluded so the environment can
        never silently downgrade fail-loud to best_effort (see
        :attr:`_ENV_EXCLUDED_FIELDS`).

        Args:
            engine_id: The engine identifier used to build env var names.
            environ: Environment mapping (defaults to ``os.environ``).
            **explicit: Explicitly supplied field values (highest priority).

        Returns:
            A validated config instance.

        Raises:
            ConfigError: If two field names collide on the same env var, or if
                construction fails (an invalid value, or a required field missing
                from both ``explicit`` and the environment) -- wrapped from
                pydantic's ``ValidationError`` with the offending input scrubbed.
                ``ConfigError`` is a ``ValueError`` subclass, so existing
                ``except ValueError`` handlers keep working.
        """
        # "explicit wins" must be ALIAS-AWARE: env_overrides keys by canonical
        # field name, but the caller may legally supply the same field under
        # any of its flat input keys (alias / validation_alias / AliasChoices
        # choice). A blind dict merge kept BOTH keys, and under
        # extra="forbid" pydantic populated from the alias and rejected the
        # canonical env key as extra -- a loud failure where the contract
        # says the explicit value wins. The env fallback for a field is
        # therefore dropped whenever the caller supplied that field under
        # ANY of its keys; a caller passing two keys for one field is still
        # loudly rejected by pydantic (that is a caller mistake, not merge
        # policy).
        merged: dict[str, Any] = {
            name: value
            for name, value in cls.env_overrides(engine_id, environ=environ).items()
            if not any(
                key in explicit for key in cls._flat_input_keys(name, cls.model_fields[name])
            )
        }
        merged.update(explicit)  # explicit wins over env.
        try:
            return cls(**merged)
        except ValidationError as exc:
            # Surface construction failures as the standard layer's ConfigError
            # (catchable as ConfigError, not a raw pydantic ValidationError), with
            # the echoed input scrubbed so a mis-placed secret never leaks (EC-1).
            error = config_error_from_validation(
                exc, prefix=f"Invalid configuration for engine {engine_id!r}"
            )
            if cls._failure_is_absent_env_config(exc, merged):
                # EVERY failure is an absent, ENVIRONMENT-FILLABLE required
                # field: nothing was supplied and nothing was invalid -- a
                # fact about the environment (e.g. a credential not set on
                # this machine), not about any value or declaration. Raise
                # the narrow subtype so consumers (the compliance suite's
                # skip) can distinguish it from a real configuration defect.
                raise ConfigurationRequiredError(str(error), details=error.details) from exc
            raise error from exc

    @classmethod
    def _failure_is_absent_env_config(cls, exc: ValidationError, merged: Mapping[str, Any]) -> bool:
        """Return whether a construction failure means "required config absent".

        The :class:`~standard_asr.contract.exceptions.ConfigurationRequiredError`
        classification (which the compliance suite SKIPS rather than fails)
        must hold only when setting environment variables could actually fix
        the failure. A bare ``all(type == "missing")`` test over-matched:

        * ``engine`` (and the other :attr:`_ENV_EXCLUDED_FIELDS`) can NEVER be
          environment-supplied -- a subclass that forgot to pin its ``engine``
          discriminator default is a DECLARATION bug, and classifying it as
          "credential absent" made compliance skip a broken plugin.
        * A nested ``missing`` (``loc`` deeper than one level) means an outer
          value WAS supplied but is incomplete -- a defect in the supplied
          value, not absence.
        * A field reported missing despite appearing in the merged input (an
          alias/population mismatch) is likewise a defect, not absence.

        Alias resolution: an ALIASED field (a provider-native wire name such
        as ``Field(alias="xi-api-key")``, populated by attribute name via
        ``populate_by_name``) reports its ALIAS in ``loc`` -- pydantic's
        errors are alias-keyed. Rejecting that token as unknown re-created
        the env-dependent verdict this classifier exists to kill: a fully
        supported credentialed config failed compliance on a clean CI and
        passed with the env var set. Each top-level token is therefore
        resolved back to its canonical own-field name (field name, string
        ``alias``, or string ``validation_alias``; an ambiguous token stays
        fail-closed).

        Args:
            exc: The construction-time validation error.
            merged: The merged explicit + environment inputs handed to the
                constructor.

        Returns:
            ``True`` when EVERY error is a top-level ``missing`` on an
            environment-fillable, genuinely-unsupplied own field.
        """
        for entry in exc.errors():
            loc = entry.get("loc", ())
            if entry.get("type") != "missing" or len(loc) != 1:
                return False
            # A non-string loc entry (a bare index) can never name an own
            # field; str() folds it into the unknown-name rejection.
            name = cls._canonical_field_name(str(loc[0]))
            if name is None or name in cls._ENV_EXCLUDED_FIELDS:
                return False
            # "Supplied" means supplied under ANY of the field's flat input
            # keys (same vocabulary as the resolver above): a value present
            # under an AliasChoices key with the error keyed by another
            # choice is still a supplied-but-invalid defect, never absence.
            if any(key in merged for key in cls._flat_input_keys(name, cls.model_fields[name])):
                return False
        return True

    @classmethod
    def _flat_input_keys(cls, name: str, field: FieldInfo) -> tuple[str, ...]:
        """Return every flat input key a field can be VALIDATED from.

        THE single alias vocabulary, shared by the three surfaces that must
        agree on it or drift into silent bugs: the secret
        whitespace-preserving pre-validator (which key(s) to wrap), the
        absence classifier's supplied-input check, and
        :meth:`_canonical_field_name`'s loc-token resolution. All three ask
        the same question -- "can this key populate this field?" -- so the
        vocabulary must model pydantic's VALIDATION direction exactly:

        * the canonical field name (``populate_by_name`` is set on
          :class:`BaseConfig` and is load-bearing: the env fallback keys by
          field name);
        * ``validation_alias`` when set -- a plain string, or the string
          choices of an ``AliasChoices`` (the definition-time guard admits
          nothing else);
        * ``alias`` ONLY when no ``validation_alias`` overrides it. Pydantic
          treats ``alias`` as both directions until a ``validation_alias``
          is given, at which point ``alias`` is serialization-only and
          rejected on input (verified: with ``alias="a",
          validation_alias="b"``, ``{"a": ...}`` yields ``extra_forbidden``
          on ``a`` AND ``missing`` on ``b``). Including it made all three
          surfaces believe a key could populate a field that pydantic will
          never accept from it -- and made the definition-time collision
          guard reject a legal declaration whose serialization alias merely
          spelled another field's name.

        ``AliasPath`` never reaches here -- it is rejected at class
        definition.

        Args:
            name: The canonical field name.
            field: The field's ``FieldInfo``.

        Returns:
            The de-duplicated keys, canonical name first.
        """
        keys: list[str] = [name]
        # Every supported pydantic MIRRORS a bare ``alias`` into
        # ``validation_alias``, so the latter alone is the complete
        # validation vocabulary (verified on the 2.5 floor and current). The
        # fallback keeps that from being an unstated assumption: if a future
        # pydantic stops mirroring, an aliased field must not silently lose
        # its input key -- the secret pre-validator would skip it and
        # ``str_strip_whitespace`` would rewrite a credential.
        va = field.validation_alias if field.validation_alias is not None else field.alias
        if isinstance(va, str):
            keys.append(va)
        elif isinstance(va, AliasChoices):
            # Definition-time guard 3 admits only all-string choices.
            keys.extend(choice for choice in va.choices if isinstance(choice, str))
        seen: set[str] = set()
        out: list[str] = []
        for key in keys:
            if key not in seen:
                seen.add(key)
                out.append(key)
        return tuple(out)

    @classmethod
    def _canonical_field_name(cls, token: str) -> str | None:
        """Resolve a validation-error ``loc`` token to an own-field name.

        Consults :meth:`_flat_input_keys` -- the same alias vocabulary the
        secret pre-validator and the absence classifier use, so the three
        surfaces can never drift on what counts as "this field's key".

        Args:
            token: The top-level ``loc`` entry from a pydantic error --
                either a field name or (for aliased fields) the alias string
                pydantic keys its errors by.

        Returns:
            The canonical field name, or ``None`` when the token names no own
            field or matches more than one (ambiguity stays fail-closed: the
            caller then classifies the failure as a defect, never absence).
        """
        if token in cls.model_fields:
            return token
        matches = [
            field_name
            for field_name, field in cls.model_fields.items()
            if token in cls._flat_input_keys(field_name, field)
        ]
        return matches[0] if len(matches) == 1 else None

    @classmethod
    def env_overrides(
        cls, engine_id: str, *, environ: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        """Collect config overrides from environment variables.

        Only fields absent from explicit config should be filled from these;
        the caller applies priority (explicit > env). Collisions (two fields
        normalizing to the same env var) are rejected.

        Security note: the returned dict holds **raw plaintext** values,
        including any credential fields, because ``SecretStr`` wrapping happens
        only at construction. Prefer :meth:`from_env`, which merges and
        constructs in one step (so secrets are wrapped/masked); treat the dict
        returned here as sensitive and never log it.

        Args:
            engine_id: The engine identifier used to build env var names.
            environ: Environment mapping (defaults to ``os.environ``).

        Returns:
            A dict of ``{field_name: value}`` discovered in the environment.

        Raises:
            ConfigError: If two field names collide on the same env var.
        """
        env = os.environ if environ is None else environ
        seen: dict[str, str] = {}
        overrides: dict[str, Any] = {}
        for field_name in cls.model_fields:
            if field_name in cls._ENV_EXCLUDED_FIELDS:
                continue
            var = env_var_name(engine_id, field_name)
            if var in seen:
                raise ConfigError(
                    f"Env var collision: fields {seen[var]!r} and {field_name!r} "
                    f"both normalize to {var!r}."
                )
            seen[var] = field_name
            if var not in env:
                continue
            raw = env[var]
            if cls._ENV_CODECS.get(field_name, "raw") == "json":
                # A structured field (list, mapping, submodel, TypedDict,
                # dataclass): a bare env string never coerces into one, so
                # parse JSON first. On a JSON error keep the raw string so
                # construction still fails loudly (never silently drop).
                try:
                    overrides[field_name] = json.loads(raw)
                except json.JSONDecodeError:
                    overrides[field_name] = raw
            else:
                overrides[field_name] = raw
        return overrides


class DeviceConfigMixin(BaseModel):
    """Applicability mixin: compute-device selection.

    Attributes:
        device: Compute device (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    device: str | None = Field(default=None, description="Compute device.")


class LanguageConfigMixin(BaseModel):
    """Applicability mixin: default language selection.

    Attributes:
        default_language: Default language (BCP-47 or ``"auto"``). Required when
            the engine exposes a language axis.
        default_candidate_languages: Default candidate languages.
    """

    default_language: str | None = Field(
        default=None, description="Default language (BCP-47 or 'auto')."
    )
    default_candidate_languages: list[str] | None = Field(
        default=None, description="Default candidate languages."
    )


class DownloadConfigMixin(BaseModel):
    """Applicability mixin: model download / cache location.

    Attributes:
        download_root: Root directory for model artifacts. Priority: explicit >
            ``STANDARD_ASR_MODEL_DIR`` > library default > ``~/.cache``.
    """

    download_root: Path | None = Field(
        default=None, description="Root directory for downloaded model artifacts."
    )


class CredentialsConfigMixin(BaseModel):
    """Applicability mixin: cloud credentials and endpoint routing.

    Credentials (``api_key``) are secret; endpoint routing fields
    (``base_url`` / ``region`` / ``org_id``) are not secret and may be logged.

    Attributes:
        api_key: Secret API key / token.
        base_url: Non-secret API base URL.
        region: Non-secret service region.
        org_id: Non-secret organization id.
    """

    api_key: SecretStr | None = secret_field(description="Secret API key / token.")
    base_url: str | None = Field(default=None, description="API base URL.")
    region: str | None = Field(default=None, description="Service region.")
    org_id: str | None = Field(default=None, description="Organization id.")


__all__ = [
    "ENV_PREFIX",
    "SECRET_MASK",
    "BaseConfig",
    "CredentialsConfigMixin",
    "DeviceConfigMixin",
    "DownloadConfigMixin",
    "LanguageConfigMixin",
    "env_var_name",
    "secret_field",
]
