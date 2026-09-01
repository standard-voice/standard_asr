# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Artifact status, acquisition, action, and progress data models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated, Final

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    UrlConstraints,
    field_validator,
    model_validator,
)

from standard_asr.contract.capabilities import ModeName
from standard_asr.contract.params import RuntimeParams
from standard_asr.contract.results import Diagnostic

_CODE_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
_ARTIFACT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"

ArtifactState = Annotated[str, StringConstraints(pattern=_CODE_PATTERN)]
"""Open vocabulary for one logical artifact requirement's state."""

ArtifactReadiness = Annotated[str, StringConstraints(pattern=_CODE_PATTERN)]
"""Open vocabulary for aggregate inference readiness."""

ArtifactActionKind = Annotated[str, StringConstraints(pattern=_CODE_PATTERN)]
"""Open vocabulary for an external action required from an operator."""

ArtifactAcquisitionBlocker = Annotated[str, StringConstraints(pattern=_CODE_PATTERN)]
"""Open vocabulary for a reason that prevents explicit acquisition."""

ArtifactProgressPhase = Annotated[str, StringConstraints(pattern=_CODE_PATTERN)]
"""Open vocabulary for an artifact operation's current phase."""

ArtifactProgressUnit = Annotated[str, StringConstraints(pattern=_CODE_PATTERN)]
"""Open vocabulary for artifact progress-count units."""

HttpsUrl = Annotated[
    AnyUrl,
    UrlConstraints(allowed_schemes=["https"], host_required=True),
]
"""An HTTPS display URL that Standard ASR never dereferences."""

_ArtifactId = Annotated[str, StringConstraints(pattern=_ARTIFACT_ID_PATTERN)]


ARTIFACT_READY: Final = "ready"
ARTIFACT_MISSING: Final = "missing"
ARTIFACT_INCOMPLETE: Final = "incomplete"
ARTIFACT_CORRUPT: Final = "corrupt"
ARTIFACT_UNKNOWN: Final = "unknown"

ARTIFACTS_READY: Final = "ready"
ARTIFACTS_UNAVAILABLE: Final = "unavailable"
ARTIFACTS_UNKNOWN: Final = "unknown"
ARTIFACTS_NOT_APPLICABLE: Final = "not_applicable"

ARTIFACT_ACTION_ACCEPT_TERMS: Final = "accept_terms"
ARTIFACT_ACTION_AUTHENTICATE: Final = "authenticate"
ARTIFACT_ACTION_REQUEST_ACCESS: Final = "request_access"
ARTIFACT_ACTION_PROVIDE_ARTIFACTS: Final = "provide_artifacts"
ARTIFACT_ACTION_INSTALL_EXTERNAL: Final = "install_external"
ARTIFACT_ACTION_OTHER: Final = "other"

ARTIFACT_BLOCKER_DOWNLOADS_DISABLED: Final = "downloads_disabled"
ARTIFACT_BLOCKER_ACTION_REQUIRED: Final = "action_required"
ARTIFACT_BLOCKER_UNSUPPORTED: Final = "unsupported"

ARTIFACT_PROGRESS_RESOLVING: Final = "resolving"
ARTIFACT_PROGRESS_TRANSFERRING: Final = "transferring"
ARTIFACT_PROGRESS_EXTRACTING: Final = "extracting"
ARTIFACT_PROGRESS_CONVERTING: Final = "converting"
ARTIFACT_PROGRESS_VERIFYING: Final = "verifying"
ARTIFACT_PROGRESS_FINALIZING: Final = "finalizing"
ARTIFACT_PROGRESS_UNIT_BYTES: Final = "bytes"
ARTIFACT_PROGRESS_UNIT_FILES: Final = "files"

_KNOWN_READINESS_VALUES = frozenset(
    {ARTIFACTS_READY, ARTIFACTS_UNAVAILABLE, ARTIFACTS_UNKNOWN, ARTIFACTS_NOT_APPLICABLE}
)
_KNOWN_UNAVAILABLE_STATES = frozenset({ARTIFACT_MISSING, ARTIFACT_INCOMPLETE, ARTIFACT_CORRUPT})
_KNOWN_BLOCKERS = frozenset(
    {
        ARTIFACT_BLOCKER_ACTION_REQUIRED,
        ARTIFACT_BLOCKER_DOWNLOADS_DISABLED,
        ARTIFACT_BLOCKER_UNSUPPORTED,
    }
)


class ArtifactContext(BaseModel):
    """Request information available before audio reaches an engine.

    Attributes:
        mode: Requested inference mode, or ``None`` for engine resolution.
        params: Per-request parameters that can affect artifact requirements.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: ModeName | None = None
    params: RuntimeParams = Field(default_factory=RuntimeParams)


class ArtifactAction(BaseModel):
    """Describe an external action needed before artifact acquisition.

    Attributes:
        kind: Machine-readable action classification.
        message: Human-readable instructions for the operator.
        url: Optional HTTPS page that explains or completes the action.

    Raises:
        ValueError: If ``url`` embeds user information.
    """

    # ``revalidate_instances`` because pydantic's default accepts an existing
    # instance of this class verbatim when it is nested in another model: only
    # MODEL-level validators rerun, so a field-level constraint (the ``kind``
    # token shape, the HTTPS ``url``) never gets a second look. An engine reaches
    # that state without any exotic API -- ``model_copy(update=...)`` skips
    # validation by design, and adjusting one field of a template action is the
    # obvious thing to write. Re-validating here keeps "nested in a report" and
    # "constructed directly" the same contract.
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    kind: ArtifactActionKind
    message: str
    url: HttpsUrl | None = None

    @field_validator("url")
    @classmethod
    def _url_has_no_user_information(cls, value: AnyUrl | None) -> AnyUrl | None:
        """Reject credentials embedded in an action URL.

        Args:
            value: Validated HTTPS URL, or ``None``.

        Returns:
            The validated URL unchanged.

        Raises:
            ValueError: If the URL contains a username or password.
        """
        if value is not None and (value.username is not None or value.password is not None):
            raise ValueError("Artifact action URLs must not contain user information.")
        return value


class ArtifactRequirement(BaseModel):
    """Report one logical artifact requirement for a resolved context.

    Attributes:
        artifact_id: Engine-scoped identifier used to correlate reports and
            progress events.
        label: Human-readable display label.
        state: Current requirement state.
        required_for_inference: Whether the selected execution path needs the
            requirement.
        can_acquire_now: Whether explicit acquisition can run immediately.
        may_acquire_during_inference: Whether normal inference can acquire the
            requirement.
        source_is_mutable: Whether refresh can re-resolve a mutable source.
        acquisition_blocker: Reason explicit acquisition cannot run.
        required_actions: External actions known to block acquisition.
        location: Absolute logical file or directory root, if known.
        size_bytes: Present logical size, if known.
        expected_size_bytes: Complete logical size, if known.
        artifact_version: Opaque revision or version, if known.

    Raises:
        ValueError: If acquisition fields contradict the requirement state, a
            size is negative, the location is relative, or the identifier is
            malformed.
    """

    # ``revalidate_instances`` for the reason given on :class:`ArtifactAction`:
    # without it, only the model-level validator below reruns when a requirement
    # is nested in a report, so a negative size, a malformed ``artifact_id``, or
    # a RELATIVE ``location`` -- which the protocol states MUST be absolute --
    # reaches the report and its JSON projection unchallenged. The template maps
    # the resulting ValidationError to EngineContractError, which is the fault
    # ownership AR.2 assigns to a hook that returns an invalid requirement.
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    artifact_id: _ArtifactId
    label: str
    state: ArtifactState
    required_for_inference: bool
    can_acquire_now: bool
    may_acquire_during_inference: bool
    source_is_mutable: bool
    acquisition_blocker: ArtifactAcquisitionBlocker | None = None
    required_actions: tuple[ArtifactAction, ...] = ()
    location: Path | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    expected_size_bytes: int | None = Field(default=None, ge=0)
    artifact_version: str | None = None

    @field_validator("location")
    @classmethod
    def _location_is_absolute(cls, value: Path | None) -> Path | None:
        """Require an absolute location when one is reported.

        Args:
            value: Logical artifact root, or ``None``.

        Returns:
            The validated path unchanged.

        Raises:
            ValueError: If the path is relative.
        """
        if value is not None and not value.is_absolute():
            raise ValueError("Artifact locations must be absolute paths.")
        return value

    @model_validator(mode="after")
    def _validate_acquisition_state(self) -> ArtifactRequirement:
        """Reject contradictory state, blocker, and action combinations.

        Returns:
            The validated requirement.

        Raises:
            ValueError: If the requirement carries an invalid combination.
        """
        if self.state == ARTIFACTS_NOT_APPLICABLE:
            raise ValueError(
                "ArtifactRequirement.state cannot be 'not_applicable'; use report applicability."
            )

        if self.state == ARTIFACT_READY:
            if self.can_acquire_now or self.acquisition_blocker is not None:
                raise ValueError(
                    "A ready artifact cannot be acquired now and cannot carry a blocker."
                )
            if self.required_actions:
                raise ValueError("A ready artifact cannot carry required actions.")
            return self

        if self.can_acquire_now != (self.acquisition_blocker is None):
            raise ValueError(
                "For a non-ready artifact, can_acquire_now must be true exactly "
                "when acquisition_blocker is absent."
            )
        if self.can_acquire_now and self.required_actions:
            raise ValueError("An artifact that can be acquired now cannot carry required actions.")
        if self.acquisition_blocker in _KNOWN_BLOCKERS and bool(self.required_actions) != (
            self.acquisition_blocker == ARTIFACT_BLOCKER_ACTION_REQUIRED
        ):
            raise ValueError(
                "Required actions must be present exactly when acquisition_blocker "
                "is the standard 'action_required' blocker."
            )
        return self


def _derive_readiness(
    applicable: bool, requirements: tuple[ArtifactRequirement, ...]
) -> ArtifactReadiness:
    """Derive aggregate readiness from required artifact states.

    Args:
        applicable: Whether artifact management applies to the resolved context.
        requirements: The logical artifact requirements resolved for the context.

    Returns:
        A canonical readiness token.
    """
    if not applicable:
        return ARTIFACTS_NOT_APPLICABLE
    required = tuple(item for item in requirements if item.required_for_inference)
    if all(item.state == ARTIFACT_READY for item in required):
        return ARTIFACTS_READY
    if any(item.state in _KNOWN_UNAVAILABLE_STATES for item in required):
        return ARTIFACTS_UNAVAILABLE
    return ARTIFACTS_UNKNOWN


class ArtifactReport(BaseModel):
    """Report artifact requirements and aggregate inference readiness.

    Attributes:
        mode: Concrete inference mode represented by the report.
        applicable: Whether artifact management applies to the resolved context.
        requirements: Every logical requirement the resolved context depends on.
        diagnostics: Non-fatal decisions made while producing the report.
        readiness: Stored aggregate inference-readiness verdict.

    Raises:
        ValueError: If applicability, identifiers, or a recognized readiness
            value contradicts the requirements.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: ModeName
    applicable: bool
    requirements: tuple[ArtifactRequirement, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    readiness: ArtifactReadiness

    @model_validator(mode="after")
    def _readiness_matches_requirements(self) -> ArtifactReport:
        """Check applicability, unique identifiers, and known readiness values.

        Returns:
            The validated report.

        Raises:
            ValueError: If the report is internally inconsistent.
        """
        if not self.applicable and self.requirements:
            raise ValueError("A non-applicable artifact report cannot carry requirements.")
        artifact_ids = [requirement.artifact_id for requirement in self.requirements]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("Artifact identifiers must be unique within one report.")

        expected = _derive_readiness(self.applicable, self.requirements)
        if self.readiness in _KNOWN_READINESS_VALUES and self.readiness != expected:
            raise ValueError(
                f"Artifact report readiness {self.readiness!r} does not match "
                f"the derived readiness {expected!r}."
            )
        return self

    @classmethod
    def from_requirements(
        cls,
        *,
        mode: ModeName,
        applicable: bool,
        requirements: Iterable[ArtifactRequirement] = (),
        diagnostics: Iterable[Diagnostic] = (),
    ) -> ArtifactReport:
        """Build a report with canonical aggregate readiness.

        Args:
            mode: Concrete inference mode represented by the report.
            applicable: Whether artifact management applies to the context.
            requirements: Logical artifact requirements.
            diagnostics: Non-fatal status diagnostics.

        Returns:
            A validated report with derived readiness.
        """
        requirement_tuple = tuple(requirements)
        return cls(
            mode=mode,
            applicable=applicable,
            requirements=requirement_tuple,
            diagnostics=tuple(diagnostics),
            readiness=_derive_readiness(applicable, requirement_tuple),
        )


class ArtifactProgress(BaseModel):
    """Describe one ordered artifact-acquisition progress update.

    Attributes:
        phase: Current operation phase.
        artifact_id: Related logical requirement, if the update is specific.
        completed_units: Completed byte or file count, if known.
        total_units: Total byte or file count, if known.
        unit: Unit for the supplied counts.

    Raises:
        ValueError: If counts and their unit disagree or completed units exceed
            total units.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: ArtifactProgressPhase
    artifact_id: _ArtifactId | None = None
    completed_units: int | None = Field(default=None, ge=0)
    total_units: int | None = Field(default=None, ge=0)
    unit: ArtifactProgressUnit | None = None

    @model_validator(mode="after")
    def _validate_counts(self) -> ArtifactProgress:
        """Require coherent progress counts and units.

        Returns:
            The validated progress update.

        Raises:
            ValueError: If counts and their unit disagree or completed units
                exceed total units.
        """
        has_count = self.completed_units is not None or self.total_units is not None
        if has_count != (self.unit is not None):
            raise ValueError("Progress unit must be present exactly when a count is present.")
        if (
            self.completed_units is not None
            and self.total_units is not None
            and self.completed_units > self.total_units
        ):
            raise ValueError("completed_units cannot exceed total_units.")
        return self


ArtifactProgressCallback = Callable[[ArtifactProgress], None]
"""Callback that observes validated artifact progress updates."""


__all__ = [
    "ARTIFACTS_NOT_APPLICABLE",
    "ARTIFACTS_READY",
    "ARTIFACTS_UNAVAILABLE",
    "ARTIFACTS_UNKNOWN",
    "ARTIFACT_ACTION_ACCEPT_TERMS",
    "ARTIFACT_ACTION_AUTHENTICATE",
    "ARTIFACT_ACTION_INSTALL_EXTERNAL",
    "ARTIFACT_ACTION_OTHER",
    "ARTIFACT_ACTION_PROVIDE_ARTIFACTS",
    "ARTIFACT_ACTION_REQUEST_ACCESS",
    "ARTIFACT_BLOCKER_ACTION_REQUIRED",
    "ARTIFACT_BLOCKER_DOWNLOADS_DISABLED",
    "ARTIFACT_BLOCKER_UNSUPPORTED",
    "ARTIFACT_CORRUPT",
    "ARTIFACT_INCOMPLETE",
    "ARTIFACT_MISSING",
    "ARTIFACT_PROGRESS_CONVERTING",
    "ARTIFACT_PROGRESS_EXTRACTING",
    "ARTIFACT_PROGRESS_FINALIZING",
    "ARTIFACT_PROGRESS_RESOLVING",
    "ARTIFACT_PROGRESS_TRANSFERRING",
    "ARTIFACT_PROGRESS_UNIT_BYTES",
    "ARTIFACT_PROGRESS_UNIT_FILES",
    "ARTIFACT_PROGRESS_VERIFYING",
    "ARTIFACT_READY",
    "ARTIFACT_UNKNOWN",
    "ArtifactAcquisitionBlocker",
    "ArtifactAction",
    "ArtifactActionKind",
    "ArtifactContext",
    "ArtifactProgress",
    "ArtifactProgressCallback",
    "ArtifactProgressPhase",
    "ArtifactProgressUnit",
    "ArtifactReadiness",
    "ArtifactReport",
    "ArtifactRequirement",
    "ArtifactState",
    "HttpsUrl",
]
