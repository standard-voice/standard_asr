# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Init configuration models for ASR engines (spec, section "Init Config").

``BaseConfig`` is the discriminated base for an engine's install/deploy-time
configuration: a discriminator ``engine`` (the entrypoint-derived ``engine_id``)
plus "relevant-only" optional standard fields (provided via the applicability
mixins below) plus engine-declared fields.

Key normative behaviours implemented here:

* **Credential safety (IC.3):** credential fields MUST use ``SecretStr`` and be
  marked secret; :meth:`BaseConfig.public_dump` returns a sanitized dump for
  ``/v1/models``, persistence, and telemetry. Plaintext is materialized only
  on demand via ``SecretStr.get_secret_value()``.
* **Env fallback (IC.4):** ``STANDARD_ASR_<NORMENGINE>__<NORMFIELD>`` (double
  underscore boundary) with normalization and collision detection; composite
  fields are JSON-decoded; priority is explicit > env > error.
* **Applicability (IC.5):** a standard field is applicable iff it appears in the
  model -- engines compose the mixins they need so auto-UI renders the right
  form without per-field hiding.
* **Lazy purity (IC.9):** ``__init__`` capturing config MUST be pure (no FS,
  GPU, or network); materialization happens later under ``allow_downloads()``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import types
from collections.abc import Mapping
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
    BaseModel,
    ConfigDict,
    Field,
    SecretBytes,
    SecretStr,
    ValidationError,
    model_validator,
)

from .error_redaction import config_error_from_validation
from .exceptions import ConfigError

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

    Use together with a ``SecretStr`` annotation. The ``json_schema_extra``
    marks the field secret so auto-UI renders a password / write-only input and
    REST exposes it POST-only. :class:`BaseConfig` enforces the ``SecretStr``
    annotation at class-definition time, masks the value in
    :meth:`BaseConfig.public_dump`, and preserves the secret's exact contents
    (no whitespace stripping) so a paste error is never silently swallowed.

    Args:
        default: Field default (use ``None`` for optional credentials).
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


def _annotation_is_secret_type(annotation: Any) -> bool:
    """Return whether an annotation is a *scalar* masking secret type.

    Accepts ``SecretStr`` / ``SecretBytes`` directly or as a member of an
    ``Optional`` / ``Union`` (e.g. ``SecretStr | None``). A generic container
    parametrized by a secret type (``list[SecretStr]``, ``dict[str, SecretStr]``,
    ``tuple[SecretStr, ...]``) is deliberately **not** accepted: the
    whitespace-preserving wrapper and the masking dump paths only handle scalar
    secrets, so a container field would be half-protected (hidden by the UI
    markers while its items leak through the secret pipeline).

    Args:
        annotation: The field's resolved annotation.

    Returns:
        ``True`` if the annotation is (or unions in) a scalar secret type.
    """
    if isinstance(annotation, type) and issubclass(annotation, _SECRET_TYPES):
        return True
    # Only union members are unwrapped (both typing.Union and the PEP 604
    # ``X | Y`` form). Any other parametrized origin is a generic container and
    # MUST NOT satisfy the requirement.
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return any(_annotation_is_secret_type(arg) for arg in get_args(annotation))
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


def _secret_marked_field_path(
    model: type[BaseModel], _visited: set[type[BaseModel]] | None = None
) -> str | None:
    """Return the dotted path to the first secret-marked field in ``model``, recursively.

    Searches ``model``'s own fields and every nested ``BaseModel`` reachable
    through their annotations. The IC.3 secret pipeline (the SecretStr-enforcing
    class hook, the whitespace-preserving validator, and the masking dump) only
    operates on a :class:`BaseConfig`'s *own* scalar fields, so a secret marker
    on a nested submodel field is silently unprotected -- its plaintext leaks
    through ``public_dump`` / ``repr`` / ``model_dump``. Detecting it lets the
    class hook reject the shape at definition time. A ``visited`` set guards
    against recursive/self-referential model graphs.

    Args:
        model: The model to search.
        _visited: Models already visited (cycle guard; internal).

    Returns:
        The dotted path (e.g. ``"auth.token"``) to the first secret-marked
        field found, or ``None`` if the model graph carries none.
    """
    visited: set[type[BaseModel]] = _visited if _visited is not None else set()
    if model in visited:
        return None
    visited.add(model)
    for name, field in model.model_fields.items():
        if _is_secret_marked(field):
            return name
        for nested in _nested_models_in_annotation(field.annotation):
            sub = _secret_marked_field_path(nested, visited)
            if sub is not None:
                return f"{name}.{sub}"
    return None


#: Container origins whose env value is parsed as JSON (a bare env string can
#: never coerce into one of these). ``BaseModel`` subclasses are handled
#: separately (they are types, not parametrized origins).
_JSON_CONTAINER_ORIGINS: tuple[Any, ...] = (list, dict, set, tuple, frozenset)


def _annotation_needs_json_env(annotation: Any) -> bool:
    """Return whether an env value for this annotation must be JSON-decoded first.

    A standard field may be a composite type -- ``LanguageConfigMixin``'s
    ``default_candidate_languages: list[str] | None`` is a spec-named Init Config
    field (spec 3.1) -- but an environment variable is always a bare string, and
    pydantic will not coerce ``"en,ja"`` (or even ``'["en","ja"]'``) into a
    ``list[str]``. Without this, a composite standard field is unreachable
    through its own env convention (``list_type`` ``ValidationError``). For such
    fields the env value is parsed as JSON first (the pydantic-settings
    precedent), falling back to the raw string on a JSON error so a malformed
    value still fails loudly at construction. Scalars (``str``, ``int``, ``bool``,
    ``SecretStr``, ``Path``, ...) are returned ``False`` so they keep passing
    through verbatim -- including credentials, whose exact bytes must not be
    reinterpreted.

    Args:
        annotation: The field's resolved annotation.

    Returns:
        ``True`` if the env value should be JSON-decoded before construction.
    """
    # Match container/union origins FIRST, before any isinstance check. A parametrized
    # generic (list[str], dict[...]) is an instance of `type` on Python 3.10 (but not
    # 3.11+); an isinstance-first check would misclassify list[str] as a scalar leaf
    # there and skip JSON-decoding it, leaving a composite standard field unreachable
    # via env on 3.10. Resolving the origin first sidesteps that version difference.
    origin = get_origin(annotation)
    if origin in _JSON_CONTAINER_ORIGINS:
        return True
    # Unwrap Optional/Union: composite iff any non-None member is composite.
    if origin is Union or origin is types.UnionType:
        return any(
            _annotation_needs_json_env(arg) for arg in get_args(annotation) if arg is not type(None)
        )
    # No container/union origin: composite only if it is a bare submodel class. A
    # generic was already handled above, so isinstance here is never fooled by list[...].
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


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
    :meth:`BaseConfig.env_overrides` (spec IC.4).

    Args:
        engine_id: The engine identifier.
        field_name: The standard config field name.

    Returns:
        The fully qualified environment variable name.
    """
    return f"{ENV_PREFIX}_{_normalize_segment(engine_id)}__{_normalize_segment(field_name)}"


class BaseConfig(BaseModel, Generic[EngineNameT]):
    """Base class for ASR engine init configuration models.

    Args:
        engine: Discriminator equal to the entrypoint-derived ``engine_id``.
        strict: Global policy for unsupported standard parameters. ``True``
            raises ``UnsupportedFeatureError``; ``False`` is best_effort
            (ignore + diagnostic).

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        # Accept fields by their attribute name as well as their alias (IC.4):
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
            "Opt-in to relax the R5 SSRF policy so an AudioUrl may target a "
            "private/loopback/link-local address (HTTPS is still required). "
            "False by default; set True only for a trusted internal endpoint."
        ),
    )

    #: Base fields that MUST NOT be sourced from the environment (IC.4). Env
    #: fallback covers **every other field** -- the standard config fields
    #: (credentials, endpoint routing, device, language, download root) AND any
    #: engine-declared field (e.g. ``beam_size``, ``model_path``), each gaining a
    #: ``STANDARD_ASR_<ENGINE>__<FIELD>`` entry (spec IC.4): env coverage of the
    #: full config surface is intentional DX, not just the mixin fields. Excluded
    #: are only the three fields where an env override would be a silent
    #: security/correctness downgrade: the ``engine`` identity (entrypoint-
    #: derived, never user-set), and the ``strict`` / ``allow_private_urls``
    #: fail-loud safety defaults (the environment must not silently flip
    #: best_effort on or relax the SSRF guard with no diagnostic).
    _ENV_EXCLUDED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"engine", "strict", "allow_private_urls"}
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Enforce that secret-marked fields use a masking secret annotation (IC.3).

        Two definition-time guards, so a credential leak can never reach runtime:

        1. A field marked ``secret=True`` (via :func:`secret_field`) but annotated
           with a plain type (e.g. ``str | None``) would be hidden from
           REST/auto-UI while leaking plaintext in ``repr``/``str``/
           ``model_dump``/:meth:`public_dump`. A container of secrets (e.g.
           ``list[SecretStr]``) is equally rejected: the masking/whitespace-
           preserving paths handle only scalar secrets, so it would be
           half-protected.
        2. A secret marker on a field of a **nested submodel** (IC.8 encourages
           per-model-family submodels) is rejected outright. The IC.3 secret
           pipeline -- the enforcement here, the whitespace-preserving validator,
           and ``public_dump``'s masking -- only operates on a ``BaseConfig``'s
           *own* scalar fields, so a secret nested one level down is silently
           unprotected and leaks plaintext through ``public_dump`` / ``repr`` /
           ``model_dump``. Credentials MUST therefore be modeled as top-level
           scalar ``SecretStr`` fields on the config, not buried in a submodel.

        Args:
            **kwargs: Forwarded subclass keyword arguments.

        Raises:
            TypeError: If a secret-marked field is not annotated as a scalar
                ``SecretStr`` or ``SecretBytes`` (optionally unioned with
                ``None``), or if a nested submodel reachable from any field
                carries a secret-marked field, or if a field's annotation contains
                an unresolved forward reference the nested-secret scan cannot vet.
        """
        super().__pydantic_init_subclass__(**kwargs)
        for name, field in cls.model_fields.items():
            if _is_secret_marked(field) and not _annotation_is_secret_type(field.annotation):
                raise TypeError(
                    f"{cls.__name__}.{name} is marked secret (secret_field) but its "
                    f"annotation {field.annotation!r} is not a scalar "
                    f"SecretStr/SecretBytes (optionally unioned with None). Containers "
                    f"of secrets (e.g. list[SecretStr]) are not masked by the secret "
                    f"pipeline; model multiple credentials as separate scalar fields "
                    f"(spec IC.3)."
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
                    f"credentials as top-level scalar SecretStr fields (spec IC.3)."
                )
            # Guard 2b: reject a secret marker buried in a nested submodel. It is
            # not threaded through this hook (nested models are not BaseConfig
            # subclasses) nor masked by public_dump (which walks only top-level
            # fields), so it would leak silently.
            for nested in _nested_models_in_annotation(field.annotation):
                leak_path = _secret_marked_field_path(nested)
                if leak_path is not None:
                    raise TypeError(
                        f"{cls.__name__}.{name} reaches a nested submodel whose field "
                        f"{nested.__name__}.{leak_path} is marked secret (secret_field). "
                        f"The IC.3 secret pipeline masks only a BaseConfig's own scalar "
                        f"fields, so a secret nested in a submodel leaks plaintext through "
                        f"public_dump / repr / model_dump. Promote the credential to a "
                        f"top-level scalar SecretStr field on {cls.__name__} (spec IC.3)."
                    )

    @model_validator(mode="before")
    @classmethod
    def _preserve_secret_whitespace(cls, data: Any) -> Any:
        """Wrap raw secret strings before global whitespace stripping.

        ``str_strip_whitespace=True`` silently trims every plain ``str`` input,
        including a raw credential passed via ``from_env`` (which hands the
        constructor a plain ``str`` that pydantic strips *before* coercing it to
        ``SecretStr``). Trimming a credential can mask a paste error and produce a
        silently-wrong secret. Running before field validation, this wraps any
        raw ``str`` destined for a secret-marked field into ``SecretStr`` first,
        so its contents bypass stripping; non-secret routing fields (``base_url``
        etc.) keep the convenience strip.

        Args:
            data: The raw constructor input (a mapping when called positionally
                with keyword data; passed through unchanged otherwise).

        Returns:
            A shallow copy of the input mapping with raw secret strings wrapped
            (the caller's original mapping is never mutated), or the input
            unchanged when it is not a mapping.
        """
        if not isinstance(data, dict):
            return data
        # Operate on a shallow copy so a caller's input dict is never mutated
        # (no spooky action at a distance): e.g. ``Cloud.model_validate(d)`` must
        # leave ``d['api_key']`` the plain str the caller passed, not silently
        # swap it for a SecretStr in their mapping. A shallow copy is sufficient:
        # we only rebind whole values (str -> SecretStr), never mutate nested
        # objects.
        mapping = dict(cast("dict[Any, Any]", data))
        for name, field in cls.model_fields.items():
            if not _is_secret_marked(field):
                continue
            key = name if name in mapping else field.alias
            if key is None or key not in mapping:
                continue
            value = mapping[key]
            if isinstance(value, str):
                mapping[key] = SecretStr(value)
        return mapping

    def public_dump(self) -> dict[str, Any]:
        """Return a serialization with secrets masked (IC.3, the default path).

        This is the **masked** half of the secret-serialization contract and is
        the serialization to use for ``/v1/models``, persistence, and telemetry.
        ``SecretStr`` fields are rendered as :data:`SECRET_MASK` (never
        plaintext). As a defensive measure, **any** secret-marked field is masked
        by name, so even a value that (hypothetically) slipped through as
        plaintext is never emitted. The default pydantic serializers
        (``model_dump`` / ``model_dump_json``) likewise mask ``SecretStr``; use
        :meth:`reveal_dump` only when plaintext is genuinely required in-process.

        Returns:
            A JSON-safe dict with credentials masked.
        """
        dumped = self.model_dump(mode="json")
        for name, field in type(self).model_fields.items():
            if _is_secret_marked(field) and dumped.get(name) is not None:
                dumped[name] = SECRET_MASK
        return dumped

    def reveal_dump(self) -> dict[str, Any]:
        """Return a serialization with secrets materialized as plaintext (IC.3).

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
        revealed = dict(self)
        for name, value in revealed.items():
            if isinstance(value, _SECRET_TYPES):
                revealed[name] = value.get_secret_value()
        return revealed

    @classmethod
    def from_env(
        cls: type[_ConfigT],
        engine_id: str,
        *,
        environ: Mapping[str, str] | None = None,
        **explicit: Any,
    ) -> _ConfigT:
        """Construct a config, filling unset fields from the environment (IC.4).

        Applies the normative priority **explicit > env > (required-missing
        error)**: each field whose name is **not a key** in ``explicit`` is
        filled from its ``STANDARD_ASR_<NORMENGINE>__<NORMFIELD>`` environment
        variable (collision detected), and the merged mapping is then passed to
        the constructor. Because construction does the field coercion,
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
        if v is not None}``) if env fallback should apply (spec IC.4).

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
        merged: dict[str, Any] = dict(cls.env_overrides(engine_id, environ=environ))
        merged.update(explicit)  # explicit wins over env.
        try:
            return cls(**merged)
        except ValidationError as exc:
            # Surface construction failures as the standard layer's ConfigError
            # (catchable as ConfigError, not a raw pydantic ValidationError), with
            # the echoed input scrubbed so a mis-placed secret never leaks (EC-1).
            raise config_error_from_validation(
                exc, prefix=f"Invalid configuration for engine {engine_id!r}"
            ) from exc

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
        for field_name, field in cls.model_fields.items():
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
            if _annotation_needs_json_env(field.annotation):
                # Composite field (e.g. list[str]): a bare env string never
                # coerces, so parse JSON first. On a JSON error keep the raw
                # string so construction still fails loudly (never silently drop).
                try:
                    overrides[field_name] = json.loads(raw)
                except json.JSONDecodeError:
                    overrides[field_name] = raw
            else:
                overrides[field_name] = raw
        return overrides


class DeviceConfigMixin(BaseModel):
    """Applicability mixin: compute-device selection.

    Args:
        device: Compute device (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    device: str | None = Field(default=None, description="Compute device.")


class LanguageConfigMixin(BaseModel):
    """Applicability mixin: default language selection.

    Args:
        default_language: Default language (BCP-47 or ``"auto"``). Required when
            the engine exposes a language axis (spec IC.6 / LANG R1).
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

    Args:
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

    Args:
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
