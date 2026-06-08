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
* **Env fallback (IC.4):** ``STANDARD_ASR_<NORMENGINE>_<NORMFIELD>`` with
  normalization and collision detection; priority is explicit > env > error.
* **Applicability (IC.5):** a standard field is applicable iff it appears in the
  model -- engines compose the mixins they need so auto-UI renders the right
  form without per-field hiding.
* **Lazy purity (IC.9):** ``__init__`` capturing config MUST be pure (no FS,
  GPU, or network); materialization happens later under ``allow_downloads()``.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretBytes,
    SecretStr,
    model_validator,
)

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
    """Return whether an annotation resolves to a masking secret type.

    Accepts ``SecretStr`` / ``SecretBytes`` directly or as a member of an
    ``Optional`` / ``Union`` (e.g. ``SecretStr | None``).

    Args:
        annotation: The field's resolved annotation.

    Returns:
        ``True`` if the annotation is (or unions in) a masking secret type.
    """
    if isinstance(annotation, type) and issubclass(annotation, _SECRET_TYPES):
        return True
    # Unwrap Optional/Union: any masking member satisfies the requirement.
    args = getattr(annotation, "__args__", None)
    if args:
        return any(_annotation_is_secret_type(arg) for arg in args)
    return False


def _normalize_segment(value: str) -> str:
    """Normalize an env-var segment: uppercase, non-alphanumerics to ``_``.

    Args:
        value: Engine id or field name segment.

    Returns:
        The normalized uppercase segment.
    """
    return re.sub(r"[^A-Z0-9]", "_", value.upper())


def env_var_name(engine_id: str, field_name: str) -> str:
    """Return the environment variable name for an engine config field.

    Args:
        engine_id: The engine identifier.
        field_name: The standard config field name.

    Returns:
        The fully qualified environment variable name.
    """
    return f"{ENV_PREFIX}_{_normalize_segment(engine_id)}_{_normalize_segment(field_name)}"


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

    #: Base fields that MUST NOT be sourced from the environment (IC.4). Env
    #: fallback covers standard *config* fields (credentials, endpoint routing,
    #: device, language, download root) only -- never the ``engine`` identity
    #: (entrypoint-derived) nor the ``strict`` safety policy, which would let the
    #: environment silently downgrade fail-loud to best_effort with no diagnostic.
    _ENV_EXCLUDED_FIELDS: ClassVar[frozenset[str]] = frozenset({"engine", "strict"})

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Enforce that secret-marked fields use a masking secret annotation (IC.3).

        A field marked ``secret=True`` (via :func:`secret_field`) but annotated
        with a plain type (e.g. ``str | None``) would be hidden from REST/auto-UI
        while leaking plaintext in ``repr``/``str``/``model_dump``/
        :meth:`public_dump`. Fail loud at class-definition time so the leak can
        never reach runtime.

        Args:
            **kwargs: Forwarded subclass keyword arguments.

        Raises:
            TypeError: If a secret-marked field is not annotated ``SecretStr`` or
                ``SecretBytes`` (optionally unioned with ``None``).
        """
        super().__pydantic_init_subclass__(**kwargs)
        for name, field in cls.model_fields.items():
            if _is_secret_marked(field) and not _annotation_is_secret_type(field.annotation):
                raise TypeError(
                    f"{cls.__name__}.{name} is marked secret (secret_field) but its "
                    f"annotation {field.annotation!r} is not SecretStr/SecretBytes. "
                    f"Secret-marked fields MUST use SecretStr to avoid plaintext leaks "
                    f"(spec IC.3)."
                )

    @model_validator(mode="before")
    @classmethod
    def _preserve_secret_whitespace(cls, data: Any) -> Any:
        """Wrap raw secret strings before global whitespace stripping (X-EL-5).

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
            The (possibly mutated) input mapping.
        """
        if not isinstance(data, dict):
            return data
        mapping = cast("dict[Any, Any]", data)
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
        environ: dict[str, str] | None = None,
        **explicit: Any,
    ) -> _ConfigT:
        """Construct a config, filling unset fields from the environment (IC.4).

        Applies the normative priority **explicit > env > (required-missing
        error)**: each standard field absent from ``explicit`` is filled from its
        ``STANDARD_ASR_<NORMENGINE>_<NORMFIELD>`` environment variable (collision
        detected), and the merged mapping is then passed to the constructor.
        Because construction does the field coercion, ``SecretStr`` credentials
        are wrapped (and so masked in ``repr``/``str``/``public_dump``) instead
        of being handed around as raw plaintext -- avoiding the leak footgun of
        passing a plaintext ``{field: secret}`` dict through application code.

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
            ConfigError: If two field names collide on the same env var.
            ValueError: If construction fails (e.g. a required field is missing
                from both ``explicit`` and the environment).
        """
        merged: dict[str, Any] = dict(cls.env_overrides(engine_id, environ=environ))
        merged.update(explicit)  # explicit wins over env.
        return cls(**merged)

    @classmethod
    def env_overrides(
        cls, engine_id: str, *, environ: dict[str, str] | None = None
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
            if var in env:
                overrides[field_name] = env[var]
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
