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

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .audio_input import InputKind
from .discovery import validate_engine_id, validate_model_name
from .exceptions import EntrypointValidationError
from .language import AUTO, is_valid_bcp47, normalize_bcp47


class SampleRateRange(BaseModel):
    """A continuous inclusive range of accepted input sample rates.

    The third ``accepted_sample_rates`` variant (besides an explicit ``list[int]``
    and ``"any"``), for engines whose I/O boundary is a *range* rather than a
    discrete set -- e.g. AWS Transcribe accepts any rate in ``[8000, 48000]``
    (research 4). Without it such an engine must either enumerate a few points
    (forcing the standard to needlessly resample an in-range rate it would accept,
    losing quality) or declare ``"any"`` (over-declaring, so an out-of-range rate
    is passed through and fails vendor-side instead of being negotiated). Both
    betray the standard's "negotiate before the call" promise. On the
    wire it serializes as ``{"min": 8000, "max": 48000}``.

    Args:
        min: Lowest accepted rate in Hz (inclusive, ``> 0``).
        max: Highest accepted rate in Hz (inclusive, ``>= min``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    min: int = Field(..., gt=0, description="Lowest accepted sample rate in Hz (inclusive).")
    max: int = Field(..., gt=0, description="Highest accepted sample rate in Hz (inclusive).")

    @model_validator(mode="after")
    def _validate_order(self) -> SampleRateRange:
        """Require ``min <= max``.

        Returns:
            The validated range.

        Raises:
            ValueError: If ``min`` exceeds ``max``.
        """
        if self.min > self.max:
            raise ValueError(f"SampleRateRange min ({self.min}) must be <= max ({self.max}).")
        return self

    def contains(self, rate: int) -> bool:
        """Return whether ``rate`` falls within the inclusive range.

        Args:
            rate: A sample rate in Hz.

        Returns:
            ``True`` if ``min <= rate <= max``.
        """
        return self.min <= rate <= self.max


#: The declared ``accepted_sample_rates`` value: a discrete list, a continuous
#: range, or ``"any"`` (every rate). Use :func:`sample_rate_accepted` and
#: :func:`nearest_accepted_sample_rate` to query/target it uniformly so a new
#: variant can never be silently mishandled by a stray ``isinstance(..., list)``.
AcceptedSampleRates = Union[list[int], SampleRateRange, Literal["any"]]


def sample_rate_accepted(accepted: AcceptedSampleRates, rate: int) -> bool:
    """Return whether ``rate`` is accepted by an ``accepted_sample_rates`` value.

    The single membership predicate for all three variants, so every
    call site (R7 reachability checks, batch resampling decision, streaming
    session validation) agrees on what "accepted" means.

    Args:
        accepted: The declared accepted sample rates.
        rate: A candidate sample rate in Hz.

    Returns:
        ``True`` if ``accepted`` admits ``rate``: always for ``"any"``,
        ``rate in accepted`` for a list, ``min <= rate <= max`` for a range.
    """
    if isinstance(accepted, SampleRateRange):
        return accepted.contains(rate)
    if isinstance(accepted, list):
        return rate in accepted
    return True  # "any"


def nearest_accepted_sample_rate(accepted: AcceptedSampleRates, source: int) -> int:
    """Return the accepted rate closest to ``source`` (resample target choice).

    Only meaningful when ``source`` is NOT already accepted (the caller checks
    that first). For a list, the nearest member preferring not to upsample (R7's
    anti-upsampling spirit). For a range, ``source`` clamped into ``[min, max]``
    -- the closest reachable in-range rate, which never upsamples when ``source``
    is above the range. ``"any"`` accepts everything, so it can never reach here
    with an unaccepted rate.

    Args:
        accepted: The declared accepted sample rates (a list or range).
        source: The input waveform's current sample rate in Hz.

    Returns:
        An accepted target sample rate in Hz.

    Raises:
        ValueError: If ``accepted`` is ``"any"`` (no finite target to choose;
            an ``"any"`` engine accepts ``source`` directly and never resamples).
    """
    if isinstance(accepted, SampleRateRange):
        # Clamp into the range: the nearest reachable in-range point.
        return min(accepted.max, max(accepted.min, source))
    if isinstance(accepted, list):
        # Nearest member, preferring a non-upsampling rate on ties (R7).
        return min(accepted, key=lambda rate: (abs(rate - source), rate > source))
    raise ValueError("nearest_accepted_sample_rate is undefined for 'any'.")


def _normalize_language_list(value: list[str], *, field: str, allow_auto: bool) -> list[str]:
    """Validate, canonicalize, and de-duplicate a declared language list.

    Shared by the ``selectable_languages`` and ``detectable_languages``
    validators. BCP-47 matching is case-insensitive, so the reserved ``auto``
    token MUST be recognised *after* normalization -- otherwise an upper-case
    ``"AUTO"`` would pass the literal pre-normalization guard, be canonicalized to
    ``"auto"`` by :func:`normalize_bcp47` (``"AUTO"`` validates as a 4-letter
    primary subtag), and land in the list despite the reserved-token ban
    (spec §LANG term ``auto``: it MUST NOT appear in a candidate list,
    and ``detectable_languages`` is the candidate-validation source). For the same
    reason duplicates are detected on the *canonical* form so case-only variants
    (``"en-US"`` / ``"EN-US"``) are caught; a duplicate is a declaration error and
    is rejected (mirroring ``wire_encodings``), naming the colliding original
    spellings so the author can find the copy-paste mistake. Order is preserved.

    Args:
        value: The raw declared tags.
        field: The field name, for error messages.
        allow_auto: Whether the canonical ``auto`` token is permitted (``True``
            for ``selectable_languages``, ``False`` for ``detectable_languages``).

    Returns:
        The canonical tags, order preserved, with ``auto`` kept verbatim when
        allowed.

    Raises:
        ValueError: If a tag is not valid BCP-47, if ``auto`` appears where it is
            not allowed, or if two tags collapse to the same canonical form.
    """
    normalized: list[str] = []
    # Map each canonical tag to the original spelling that first produced it, so a
    # collision message can show both conflicting inputs.
    seen: dict[str, str] = {}
    for tag in value:
        # Normalize FIRST, then test against the reserved token: a case variant of
        # "auto" must be recognised as the reserved token, not a BCP-47 tag.
        canonical = AUTO if tag == AUTO else (normalize_bcp47(tag) if is_valid_bcp47(tag) else None)
        if canonical is None:
            raise ValueError(f"Invalid BCP 47 language tag: {tag!r}")
        if canonical == AUTO and not allow_auto:
            raise ValueError(f"{field} must not contain 'auto'.")
        if canonical in seen:
            raise ValueError(
                f"{field} has a duplicate language: {tag!r} and {seen[canonical]!r} both "
                f"normalize to {canonical!r}."
            )
        seen[canonical] = tag
        normalized.append(canonical)
    return normalized


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
        description: Optional human-readable, display-only description. MUST NOT
            carry machine-readable negotiation/gating data -- that belongs in
            Capabilities (the free-form ``extra`` metadata pocket was
            removed; it duplicated the blanket metadata §C deliberately dropped).

    Returns:
        None.

    Raises:
        ValueError: If validation fails.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        # Plugins declare properties as a subclass with class-level defaults (the
        # documented pattern in adapting_engine.md).
        # pydantic skips field validators on defaults unless this is set, which
        # would let every declaration-time check below be bypassed on that most
        # common path -- illegal engine_id, negative sample rates, or
        # un-normalized wire_encodings would flow through silently.
        validate_default=True,
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
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = Field(
        ...,
        description="Accepted input sample rates in Hz: a discrete list, a "
        "{min,max} range, or 'any'.",
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
        default=None,
        description=(
            "Optional human-readable, display-only description. MUST NOT carry "
            "machine-readable negotiation/gating data (use Capabilities)."
        ),
    )

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
        cls, value: list[int] | SampleRateRange | Literal["any"]
    ) -> list[int] | SampleRateRange | Literal["any"]:
        """Validate the accepted sample-rate declaration.

        Args:
            value: A list of positive rates, a :class:`SampleRateRange`, or
                ``"any"``.

        Returns:
            The validated value.

        Raises:
            ValueError: If a list is empty, holds non-positive entries, or holds
                a duplicate rate (a declaration error). A ``SampleRateRange`` and
                ``"any"`` self-validate / need no list checks.
        """
        if isinstance(value, SampleRateRange):
            # A range self-validates (min>0, max>0, min<=max) in its own model.
            return value
        if not isinstance(value, list):
            # Defensive: pydantic's union coercion has already rejected any
            # non-"any" string before this after-validator runs, so the inner
            # guard is unreachable via normal validation.
            if value != "any":  # pragma: no cover - unreachable post-coercion
                raise ValueError("accepted_sample_rates string value must be 'any'.")
            return value
        if not value:
            raise ValueError("accepted_sample_rates must not be empty (or use 'any').")
        if any(rate <= 0 for rate in value):
            raise ValueError("accepted_sample_rates entries must be positive.")
        # A repeated rate is a declaration error (a copy-paste slip), rejected
        # like the duplicate guards on wire_encodings and the language lists
        # -- same declaration-hygiene contract across the module.
        if len(set(value)) != len(value):
            raise ValueError(f"accepted_sample_rates has duplicate entries: {value}.")
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
            Normalized list preserving the reserved ``auto`` token, with no
            (post-normalization) duplicates.

        Raises:
            ValueError: If any tag is invalid, or two entries collapse to the
                same tag after normalization (a declaration error).
        """
        return _normalize_language_list(value, field="selectable_languages", allow_auto=True)

    @field_validator("detectable_languages")
    @classmethod
    def _validate_detectable(cls, value: list[str]) -> list[str]:
        """Validate and normalize detectable languages (BCP-47 only).

        Args:
            value: Detectable language tags.

        Returns:
            Normalized list of tags with no (post-normalization) duplicates.

        Raises:
            ValueError: If any tag is invalid, is the reserved ``auto`` token, or
                two entries collapse to the same tag after normalization.
        """
        return _normalize_language_list(value, field="detectable_languages", allow_auto=False)

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
        concrete ``accepted_sample_rates`` (a list OR a range) does not admit the
        required rate, the engine's own contract is contradictory (the resample
        target is unreachable), so this fails loudly at declaration time rather
        than at a session establishment far from the bug. ``"any"`` admits every
        rate and so skips the check.

        Returns:
            The validated model.

        Raises:
            ValueError: If a finite ``required_input_sample_rate`` is not admitted
                by a concrete ``accepted_sample_rates`` (list or range).
        """
        req = self.required_input_sample_rate
        rates = self.accepted_sample_rates
        if req is not None and rates != "any" and not sample_rate_accepted(rates, req):
            raise ValueError(
                f"required_input_sample_rate={req} must be accepted by accepted_sample_rates "
                f"{rates!r} (the standard resamples to the required rate; spec R7)."
            )
        return self

    @model_validator(mode="after")
    def _validate_native_rate_reachable(self) -> BaseProperties:
        """Require ``native_sample_rate`` to be an accepted rate (R7).

        An engine whose ``native_sample_rate`` is not admitted by a concrete
        ``accepted_sample_rates`` (list or range) is self-contradictory: an 8 kHz
        telephony model that declares ``native_sample_rate=8000`` but excludes
        8000 from ``accepted_sample_rates`` would have its own native input
        silently upsampled (e.g. to 16 kHz), degrading quality. Per spec R7 an
        8 kHz model is a distinct native model, not a low-rate variant. Mirror the
        ``required_input_sample_rate`` reachability invariant and fail loudly at
        declaration time. ``"any"`` accepts every rate and so skips the check.

        Returns:
            The validated model.

        Raises:
            ValueError: If ``native_sample_rate`` is not admitted by a concrete
                ``accepted_sample_rates`` (list or range).
        """
        native = self.native_sample_rate
        rates = self.accepted_sample_rates
        if rates != "any" and not sample_rate_accepted(rates, native):
            raise ValueError(
                f"native_sample_rate={native} must be accepted by accepted_sample_rates "
                f"{rates!r} (otherwise the engine's own native-rate input would be "
                "silently resampled; an 8 kHz telephony model is a distinct native "
                "model, not a low-rate variant; spec R7)."
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
    def accepts_any_sample_rate(self) -> bool:
        """Whether the engine accepts any input sample rate.

        Renamed from ``self_describes_sample_rate``: that name
        clashed with the spec §AI 3.1 sense of "self-describing sample rate" --
        a property of the *audio carrier* (a file/stream header that states its
        own rate), an unrelated concept. This predicate is purely about the
        engine's declared I/O boundary: ``accepted_sample_rates == "any"``.

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


__all__ = ["BaseProperties", "SampleRateRange"]
