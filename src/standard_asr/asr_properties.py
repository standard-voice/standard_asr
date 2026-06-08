# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Static engine properties (identity and I/O boundaries).

``Properties`` carries an engine's *static identity* -- values that do not
change with feature flags or runtime mode: its id, the audio shapes it accepts,
its sample-rate boundaries, and the language axis it exposes. Behavioural
support lives in :mod:`standard_asr.capabilities`, not here (spec, section
"Capabilities", rule R7: limits that only make sense when a feature is supported
belong on that feature's capability node).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .audio_input import InputKind
from .discovery import validate_engine_id, validate_model_name
from .exceptions import EntrypointValidationError
from .language import AUTO, is_valid_bcp47, normalize_bcp47


class BaseProperties(BaseModel):
    """Base class for ASR engine static properties.

    Args:
        engine_id: Engine identifier (PEP 503 normalized).
        model_name: Model preset name within the engine.
        protocol_version: Standard ASR protocol version supported by the engine.
        accepted_input: Audio shapes the engine accepts (MUST be non-empty).
        native_sample_rate: The model's native sample rate in Hz.
        accepted_sample_rates: Sample rates the engine accepts, or ``"any"``.
        required_input_sample_rate: Sample rate the wire protocol hard-requires
            (e.g. 24000 for OpenAI Realtime), if any.
        max_file_size: Maximum file/payload size in bytes, if any.
        max_audio_duration: Maximum audio duration in seconds, if any.
        wire_encodings: Wire encodings supported for streaming, if any.
        selectable_languages: Languages the application may explicitly select
            (BCP-47 tags plus optional ``"auto"``). Empty means the engine has
            no language axis.
        detectable_languages: Languages detectable in ``auto`` mode; required
            when ``"auto"`` is selectable.
        description: Optional human-readable description.
        extra: Engine-specific metadata.

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        # `model_name` is a deliberate, central field of this protocol. Opt out of
        # pydantic's `model_` protected namespace so it does not warn (the warning
        # fires on older pydantic, e.g. the lower-bounds lane's 2.5).
        protected_namespaces=(),
    )

    engine_id: str = Field(..., description="Engine identifier (PEP 503 normalized).")
    model_name: str = Field(..., description="Model preset name within the engine.")
    protocol_version: str = Field(
        ..., description="Standard ASR protocol version supported by the engine."
    )

    # Audio I/O boundaries (spec AI section 3.2).
    accepted_input: set[InputKind] = Field(..., description="Audio shapes the engine accepts.")
    native_sample_rate: int = Field(..., gt=0, description="The model's native sample rate in Hz.")
    accepted_sample_rates: list[int] | Literal["any"] = Field(
        ..., description="Accepted input sample rates in Hz, or 'any'."
    )
    required_input_sample_rate: int | None = Field(
        default=None,
        gt=0,
        description="Sample rate the wire protocol hard-requires, if any.",
    )
    max_file_size: int | None = Field(
        default=None, gt=0, description="Maximum file/payload size in bytes, if any."
    )
    max_audio_duration: float | None = Field(
        default=None, gt=0, description="Maximum audio duration in seconds, if any."
    )
    wire_encodings: list[str] | None = Field(
        default=None, description="Wire encodings supported for streaming, if any."
    )

    # Language identity (spec LANG section 3).
    selectable_languages: list[str] = Field(
        default_factory=list,
        description="Languages the application may select (BCP-47 plus 'auto').",
    )
    detectable_languages: list[str] = Field(
        default_factory=list,
        description="Languages detectable in 'auto' mode.",
    )

    description: str | None = Field(
        default=None, description="Optional human-readable description."
    )
    extra: dict[str, Any] = Field(default_factory=dict, description="Engine-specific metadata.")

    @field_validator("accepted_input")
    @classmethod
    def _validate_accepted_input(cls, value: set[InputKind]) -> set[InputKind]:
        """Ensure the engine declares at least one accepted input shape.

        Args:
            value: The accepted input kinds.

        Returns:
            The validated set.

        Raises:
            ValueError: If empty.
        """
        if not value:
            raise ValueError("accepted_input must not be empty.")
        return value

    @field_validator("accepted_sample_rates")
    @classmethod
    def _validate_sample_rates(
        cls, value: list[int] | Literal["any"]
    ) -> list[int] | Literal["any"]:
        """Validate the accepted sample-rate list.

        Args:
            value: A list of positive rates, or ``"any"``.

        Returns:
            The validated value.

        Raises:
            ValueError: If the list is empty or holds non-positive entries.
        """
        if not isinstance(value, list):
            # Defensive: pydantic's ``list[int] | Literal["any"]`` coercion has
            # already rejected any non-"any" string before this after-validator
            # runs, so the inner guard is unreachable via normal validation.
            if value != "any":  # pragma: no cover - unreachable post-coercion
                raise ValueError("accepted_sample_rates string value must be 'any'.")
            return value
        if not value:
            raise ValueError("accepted_sample_rates must not be empty (or use 'any').")
        if any(rate <= 0 for rate in value):
            raise ValueError("accepted_sample_rates entries must be positive.")
        return value

    @field_validator("wire_encodings")
    @classmethod
    def _validate_wire_encodings(cls, value: list[str] | None) -> list[str] | None:
        """Validate and normalize the streaming wire-encoding allowlist.

        ``wire_encodings`` is the fail-closed gate that
        :meth:`~standard_asr.asr_interface.EngineBase.ensure_stream_format_supported`
        checks at session establishment. ``None`` means "unconstrained"; a
        concrete list MUST therefore be usable:

        - An empty ``[]`` (distinct from ``None``) would reject every encoding,
          silently bricking streaming -- rejected here.
        - Entries are stripped and lowercased so a declared ``"PCM_S16LE"`` and a
          requested ``"pcm_s16le"`` do not falsely mismatch (encodings are
          case-insensitive identifiers).
        - Blank and duplicate entries are rejected as declaration errors.

        Args:
            value: The declared wire encodings, or ``None``.

        Returns:
            The normalized list, or ``None`` when unconstrained.

        Raises:
            ValueError: If the list is empty, or holds a blank or duplicate
                entry.
        """
        if value is None:
            return None
        if not value:
            raise ValueError(
                "wire_encodings must not be an empty list (use None for "
                "'unconstrained'; an empty list rejects every streaming encoding)."
            )
        normalized: list[str] = []
        for encoding in value:
            cleaned = encoding.strip().lower()
            if not cleaned:
                raise ValueError("wire_encodings entries must not be blank.")
            if cleaned in normalized:
                raise ValueError(f"wire_encodings has a duplicate entry: {cleaned!r}.")
            normalized.append(cleaned)
        return normalized

    @field_validator("selectable_languages")
    @classmethod
    def _validate_selectable(cls, value: list[str]) -> list[str]:
        """Validate and normalize selectable languages (BCP-47 plus 'auto').

        Args:
            value: Selectable language tags.

        Returns:
            Normalized list preserving the reserved ``auto`` token.

        Raises:
            ValueError: If any tag is invalid.
        """
        normalized: list[str] = []
        for tag in value:
            if tag == AUTO:
                normalized.append(AUTO)
                continue
            if not is_valid_bcp47(tag):
                raise ValueError(f"Invalid BCP 47 language tag: {tag!r}")
            normalized.append(normalize_bcp47(tag))
        return normalized

    @field_validator("detectable_languages")
    @classmethod
    def _validate_detectable(cls, value: list[str]) -> list[str]:
        """Validate and normalize detectable languages (BCP-47 only).

        Args:
            value: Detectable language tags.

        Returns:
            Normalized list of tags.

        Raises:
            ValueError: If any tag is invalid or is the reserved ``auto`` token.
        """
        normalized: list[str] = []
        for tag in value:
            if tag == AUTO:
                raise ValueError("detectable_languages must not contain 'auto'.")
            if not is_valid_bcp47(tag):
                raise ValueError(f"Invalid BCP 47 language tag: {tag!r}")
            normalized.append(normalize_bcp47(tag))
        return normalized

    @model_validator(mode="after")
    def _validate_auto_requires_detectable(self) -> BaseProperties:
        """Require ``detectable_languages`` when ``auto`` is selectable.

        Returns:
            The validated model.

        Raises:
            ValueError: If ``auto`` is selectable but no detectable languages
                are declared.
        """
        if AUTO in self.selectable_languages and not self.detectable_languages:
            raise ValueError("detectable_languages is required when 'auto' is selectable.")
        return self

    @model_validator(mode="after")
    def _validate_required_rate_reachable(self) -> BaseProperties:
        """Require ``required_input_sample_rate`` to be an accepted rate (R7).

        The standard resamples wire/array input to ``required_input_sample_rate``
        when set (spec example E: ``required=24000``, ``accepted=[24000]``). If a
        concrete ``accepted_sample_rates`` list does not contain the required
        rate, the engine's own contract is contradictory (the resample target is
        unreachable), so this fails loudly at declaration time rather than at a
        session establishment far from the bug.

        Returns:
            The validated model.

        Raises:
            ValueError: If a finite ``required_input_sample_rate`` is not in a
                concrete ``accepted_sample_rates`` list.
        """
        req = self.required_input_sample_rate
        rates = self.accepted_sample_rates
        if req is not None and isinstance(rates, list) and req not in rates:
            raise ValueError(
                f"required_input_sample_rate={req} must be in accepted_sample_rates "
                f"{rates} (the standard resamples to the required rate; spec R7)."
            )
        return self

    @field_validator("engine_id")
    @classmethod
    def _validate_engine_id_field(cls, value: str) -> str:
        """Validate the engine identifier.

        Args:
            value: Engine identifier string.

        Returns:
            Validated engine identifier.

        Raises:
            ValueError: If the engine identifier is invalid.
        """
        try:
            validate_engine_id(value)
        except EntrypointValidationError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @field_validator("model_name")
    @classmethod
    def _validate_model_name_field(cls, value: str) -> str:
        """Validate the model name.

        Args:
            value: Model name string (may be empty for defaults).

        Returns:
            Validated model name.

        Raises:
            ValueError: If the model name is invalid.
        """
        try:
            validate_model_name(value)
        except EntrypointValidationError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @property
    def has_language_axis(self) -> bool:
        """Whether the engine exposes a language axis.

        Returns:
            ``True`` if any selectable language is declared.
        """
        return bool(self.selectable_languages)

    @property
    def supports_auto(self) -> bool:
        """Whether automatic language detection is selectable.

        Returns:
            ``True`` if ``"auto"`` is in ``selectable_languages``.
        """
        return AUTO in self.selectable_languages

    @property
    def self_describes_sample_rate(self) -> bool:
        """Whether the engine accepts any sample rate.

        Returns:
            ``True`` if ``accepted_sample_rates`` is ``"any"``.
        """
        return self.accepted_sample_rates == "any"

    @property
    def model_id(self) -> str:
        """Return the fully qualified model identifier (engine/model).

        Returns:
            Model identifier in ``engine/model`` format.
        """
        return f"{self.engine_id}/{self.model_name}"


__all__ = ["BaseProperties"]
