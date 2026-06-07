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
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .exceptions import ConfigError

logger = logging.getLogger(__name__)

EngineNameT = TypeVar("EngineNameT", bound=str, covariant=True)
_ConfigT = TypeVar("_ConfigT", bound="BaseConfig[Any]")

#: Prefix for environment-variable credential fallback.
ENV_PREFIX = "STANDARD_ASR"


def secret_field(default: Any = None, *, description: str = "") -> Any:
    """Build a ``Field`` for a write-only credential rendered as a password.

    Use together with a ``SecretStr`` annotation. The ``json_schema_extra``
    marks the field secret so auto-UI renders a password / write-only input and
    REST exposes it POST-only.

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
        validate_assignment=True,
    )

    engine: EngineNameT = Field(
        ..., description="Engine discriminator (entrypoint-derived engine_id)."
    )
    strict: bool = Field(
        default=True,
        description="Unsupported-parameter policy: True=strict, False=best_effort.",
    )

    def public_dump(self) -> dict[str, Any]:
        """Return a serialization with secrets masked.

        Suitable for ``/v1/models``, persistence, and telemetry. ``SecretStr``
        fields are rendered as ``"**********"`` (never plaintext).

        Returns:
            A JSON-safe dict with credentials masked.
        """
        return self.model_dump(mode="json")

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
            if field_name == "engine":
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
    "BaseConfig",
    "CredentialsConfigMixin",
    "DeviceConfigMixin",
    "DownloadConfigMixin",
    "LanguageConfigMixin",
    "env_var_name",
    "secret_field",
]
