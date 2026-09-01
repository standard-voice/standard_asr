# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model-artifact lifecycle contract models and errors."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from standard_asr.contract.artifacts import (
    ARTIFACT_ACTION_ACCEPT_TERMS,
    ARTIFACT_BLOCKER_ACTION_REQUIRED,
    ARTIFACT_BLOCKER_DOWNLOADS_DISABLED,
    ARTIFACT_BLOCKER_UNSUPPORTED,
    ARTIFACT_CORRUPT,
    ARTIFACT_INCOMPLETE,
    ARTIFACT_MISSING,
    ARTIFACT_PROGRESS_EXTRACTING,
    ARTIFACT_PROGRESS_UNIT_BYTES,
    ARTIFACT_READY,
    ARTIFACT_UNKNOWN,
    ARTIFACTS_NOT_APPLICABLE,
    ARTIFACTS_READY,
    ARTIFACTS_UNAVAILABLE,
    ARTIFACTS_UNKNOWN,
    ArtifactAction,
    ArtifactContext,
    ArtifactProgress,
    ArtifactReport,
    ArtifactRequirement,
)
from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactProgressCallbackError,
    ArtifactStatusError,
    ArtifactUnavailableError,
    StandardASRError,
    StructuredError,
)
from standard_asr.contract.params import RuntimeParams
from standard_asr.contract.results import Diagnostic


def _requirement(
    state: str,
    *,
    artifact_id: str = "asr_weights",
    required: bool = True,
    can_acquire_now: bool | None = None,
    blocker: str | None = None,
    actions: tuple[ArtifactAction, ...] = (),
) -> ArtifactRequirement:
    if state == ARTIFACT_READY:
        can_acquire_now = False if can_acquire_now is None else can_acquire_now
    elif can_acquire_now is None:
        can_acquire_now = blocker is None
    return ArtifactRequirement(
        artifact_id=artifact_id,
        label="ASR weights",
        state=state,
        required_for_inference=required,
        can_acquire_now=can_acquire_now,
        may_acquire_during_inference=False,
        source_is_mutable=False,
        acquisition_blocker=blocker,
        required_actions=actions,
    )


def test_artifact_context_defaults_are_closed_and_independent() -> None:
    first = ArtifactContext()
    second = ArtifactContext()
    assert first.mode is None
    assert first.params == RuntimeParams()
    assert first.params is not second.params
    with pytest.raises(ValidationError):
        ArtifactContext.model_validate({"extra_field": True})


@pytest.mark.parametrize("token", ["future_state", "x_vendor_special"])
def test_open_state_vocabulary_preserves_future_tokens(token: str) -> None:
    requirement = _requirement(token, blocker=ARTIFACT_BLOCKER_UNSUPPORTED)
    assert requirement.state == token
    assert requirement.model_dump()["state"] == token


@pytest.mark.parametrize("token", ["", "Upper", "has-hyphen", "1starts_with_digit"])
def test_open_state_vocabulary_rejects_malformed_tokens(token: str) -> None:
    with pytest.raises(ValidationError):
        _requirement(token, blocker=ARTIFACT_BLOCKER_UNSUPPORTED)


def test_artifact_action_accepts_https_without_credentials() -> None:
    action = ArtifactAction.model_validate(
        {
            "kind": ARTIFACT_ACTION_ACCEPT_TERMS,
            "message": "Accept the model terms, and retry.",
            "url": "https://example.com/models/terms",
        }
    )
    assert action.url is not None
    assert action.url.scheme == "https"


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/terms",
        "https://user@example.com/terms",
        "https://user:secret@example.com/terms",
    ],
)
def test_artifact_action_rejects_unsafe_display_url(url: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactAction.model_validate(
            {"kind": ARTIFACT_ACTION_ACCEPT_TERMS, "message": "Act.", "url": url}
        )


def test_ready_requirement_has_no_acquisition_work() -> None:
    requirement = _requirement(ARTIFACT_READY)
    assert requirement.can_acquire_now is False
    assert requirement.acquisition_blocker is None
    assert requirement.required_actions == ()


@pytest.mark.parametrize(
    "updates",
    [
        {"can_acquire_now": True},
        {"blocker": ARTIFACT_BLOCKER_UNSUPPORTED},
        {"actions": (ArtifactAction(kind=ARTIFACT_ACTION_ACCEPT_TERMS, message="Accept terms."),)},
    ],
)
def test_ready_requirement_rejects_acquisition_fields(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _requirement(ARTIFACT_READY, **updates)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("can_acquire_now", "blocker"),
    [
        (True, ARTIFACT_BLOCKER_UNSUPPORTED),
        (False, None),
    ],
)
def test_nonready_requirement_requires_exact_blocker_equivalence(
    can_acquire_now: bool, blocker: str | None
) -> None:
    with pytest.raises(ValidationError, match="exactly"):
        _requirement(
            ARTIFACT_MISSING,
            can_acquire_now=can_acquire_now,
            blocker=blocker,
        )


def test_action_required_blocker_and_actions_are_bidirectional() -> None:
    action = ArtifactAction(kind=ARTIFACT_ACTION_ACCEPT_TERMS, message="Accept terms.")
    blocked = _requirement(
        ARTIFACT_MISSING,
        can_acquire_now=False,
        blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
        actions=(action,),
    )
    assert blocked.required_actions == (action,)

    with pytest.raises(ValidationError, match="exactly"):
        _requirement(
            ARTIFACT_MISSING,
            can_acquire_now=False,
            blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
        )
    with pytest.raises(ValidationError, match="exactly"):
        _requirement(
            ARTIFACT_MISSING,
            can_acquire_now=False,
            blocker=ARTIFACT_BLOCKER_UNSUPPORTED,
            actions=(action,),
        )
    with pytest.raises(ValidationError, match="cannot carry required actions"):
        _requirement(
            ARTIFACT_MISSING,
            can_acquire_now=True,
            blocker=None,
            actions=(action,),
        )


def test_future_blocker_preserves_optional_future_actions() -> None:
    action = ArtifactAction(kind="future_action", message="Complete the future action.")
    requirement = _requirement(
        ARTIFACT_MISSING,
        can_acquire_now=False,
        blocker="future_blocker",
        actions=(action,),
    )
    assert requirement.acquisition_blocker == "future_blocker"
    assert requirement.required_actions == (action,)


def test_requirement_rejects_aggregate_only_state() -> None:
    with pytest.raises(ValidationError, match="cannot be 'not_applicable'"):
        _requirement(ARTIFACTS_NOT_APPLICABLE, blocker=ARTIFACT_BLOCKER_UNSUPPORTED)


@pytest.mark.parametrize("artifact_id", ["", "/tmp/model", "has space", "../weights"])
def test_requirement_rejects_non_identifier_artifact_id(artifact_id: str) -> None:
    with pytest.raises(ValidationError):
        _requirement(
            ARTIFACT_MISSING,
            artifact_id=artifact_id,
            blocker=ARTIFACT_BLOCKER_UNSUPPORTED,
        )


def test_requirement_validates_location_and_sizes(tmp_path: Path) -> None:
    data = _requirement(
        ARTIFACT_MISSING,
        blocker=ARTIFACT_BLOCKER_DOWNLOADS_DISABLED,
        can_acquire_now=False,
    ).model_dump()
    data.update(
        location=tmp_path / "weights",
        size_bytes=0,
        expected_size_bytes=1024,
    )
    requirement = ArtifactRequirement(**data)
    assert requirement.location == tmp_path / "weights"

    data["location"] = Path("relative/weights")
    with pytest.raises(ValidationError, match="absolute"):
        ArtifactRequirement(**data)

    data["location"] = None
    data["size_bytes"] = -1
    with pytest.raises(ValidationError):
        ArtifactRequirement(**data)


def test_report_derives_not_applicable_and_ready_shapes() -> None:
    not_applicable = ArtifactReport.from_requirements(mode="batch", applicable=False)
    assert not_applicable.readiness == ARTIFACTS_NOT_APPLICABLE

    empty_applicable = ArtifactReport.from_requirements(mode="streaming", applicable=True)
    assert empty_applicable.readiness == ARTIFACTS_READY

    optional_missing = ArtifactReport.from_requirements(
        mode="batch",
        applicable=True,
        requirements=[
            _requirement(
                ARTIFACT_MISSING,
                required=False,
                blocker=ARTIFACT_BLOCKER_UNSUPPORTED,
            )
        ],
    )
    assert optional_missing.readiness == ARTIFACTS_READY


def test_report_derives_ready_for_a_required_requirement_that_is_ready() -> None:
    # The most ordinary production case, and the one the other ready assertions
    # cannot reach: both of them leave the REQUIRED sequence empty (no
    # requirements at all, and one optional requirement), so
    # ``all(... for item in required)`` returns True vacuously and its body
    # never runs. Only a required, ready requirement proves the ready verdict
    # comes from inspecting a requirement rather than from an empty sequence.
    report = ArtifactReport.from_requirements(
        mode="batch",
        applicable=True,
        requirements=[_requirement(ARTIFACT_READY)],
    )
    assert report.readiness == ARTIFACTS_READY
    assert report.requirements[0].required_for_inference is True


@pytest.mark.parametrize(
    "state",
    [ARTIFACT_MISSING, ARTIFACT_INCOMPLETE, ARTIFACT_CORRUPT],
)
def test_report_derives_unavailable_for_known_required_failure(state: str) -> None:
    report = ArtifactReport.from_requirements(
        mode="batch",
        applicable=True,
        requirements=[_requirement(state, blocker=ARTIFACT_BLOCKER_UNSUPPORTED)],
    )
    assert report.readiness == ARTIFACTS_UNAVAILABLE


@pytest.mark.parametrize("state", [ARTIFACT_UNKNOWN, "future_state"])
def test_report_derives_unknown_conservatively(state: str) -> None:
    report = ArtifactReport.from_requirements(
        mode="batch",
        applicable=True,
        requirements=[_requirement(state, blocker=ARTIFACT_BLOCKER_UNSUPPORTED)],
    )
    assert report.readiness == ARTIFACTS_UNKNOWN


def test_report_known_unavailable_state_precedes_future_state() -> None:
    report = ArtifactReport.from_requirements(
        mode="batch",
        applicable=True,
        requirements=(
            item
            for item in [
                _requirement(ARTIFACT_MISSING, blocker=ARTIFACT_BLOCKER_UNSUPPORTED),
                _requirement(
                    "future_state",
                    artifact_id="aligner",
                    blocker=ARTIFACT_BLOCKER_UNSUPPORTED,
                ),
            ]
        ),
        diagnostics=[Diagnostic(code="artifact_inspected", message="Inspected artifacts.")],
    )
    assert report.readiness == ARTIFACTS_UNAVAILABLE
    assert report.diagnostics[0].code == "artifact_inspected"


def test_report_rejects_known_mismatch_but_preserves_future_readiness() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        ArtifactReport(
            mode="batch",
            applicable=True,
            readiness=ARTIFACTS_UNAVAILABLE,
        )

    future = ArtifactReport(
        mode="batch",
        applicable=True,
        readiness="future_readiness",
    )
    assert future.readiness == "future_readiness"
    assert future.model_dump()["readiness"] == "future_readiness"


def test_report_rejects_requirements_when_not_applicable_and_duplicate_ids() -> None:
    ready = _requirement(ARTIFACT_READY)
    with pytest.raises(ValidationError, match="non-applicable"):
        ArtifactReport(
            mode="batch",
            applicable=False,
            requirements=(ready,),
            readiness=ARTIFACTS_NOT_APPLICABLE,
        )
    with pytest.raises(ValidationError, match="unique"):
        ArtifactReport.from_requirements(
            mode="batch",
            applicable=True,
            requirements=(ready, ready),
        )


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"size_bytes": -1}, "greater than or equal to 0"),
        ({"expected_size_bytes": -5}, "greater than or equal to 0"),
        ({"artifact_id": "Bad Id!"}, "artifact_id"),
        ({"location": Path("relative/dir")}, "absolute"),
    ],
)
def test_report_revalidates_nested_requirements_that_skipped_validation(
    update: dict[str, object], match: str
) -> None:
    # ``model_copy(update=...)`` does not validate -- adjusting one field of a
    # template requirement is the obvious thing for an engine to write, and it
    # is enough to produce a value the constructor would have rejected. Nesting
    # such an instance in a report reruns only MODEL-level validators by
    # default, so each of these violates a FIELD constraint and would otherwise
    # reach the report and its JSON projection unchallenged. A relative
    # ``location`` is the sharpest of them: the protocol states it MUST be
    # absolute, so the leak crosses the wire contract.
    forged = _requirement(ARTIFACT_READY).model_copy(update=update)
    with pytest.raises(ValidationError, match=match):
        ArtifactReport.from_requirements(mode="batch", applicable=True, requirements=(forged,))


def test_requirement_revalidates_a_nested_action_that_skipped_validation() -> None:
    # The guard has to sit on each leaf an engine authors: pydantic accepts a
    # nested instance of the exact class verbatim at every level, so without it
    # a forged action rides into the requirement that carries it.
    action = ArtifactAction(kind=ARTIFACT_ACTION_ACCEPT_TERMS, message="Accept the license.")
    forged = action.model_copy(update={"kind": "NOT A TOKEN"})
    with pytest.raises(ValidationError, match="kind"):
        _requirement(
            ARTIFACT_MISSING,
            can_acquire_now=False,
            blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
            actions=(forged,),
        )


def test_report_revalidates_actions_nested_two_levels_down() -> None:
    # Both hops skipped validation: the action was forged with model_copy, then
    # swapped into an already-built requirement with model_copy again. The
    # report must still refuse it, which only holds if the requirement
    # re-validates AND its actions re-validate in turn.
    action = ArtifactAction(kind=ARTIFACT_ACTION_ACCEPT_TERMS, message="Accept the license.")
    forged_action = action.model_copy(update={"kind": "NOT A TOKEN"})
    requirement = _requirement(
        ARTIFACT_MISSING,
        can_acquire_now=False,
        blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
        actions=(action,),
    )
    forged_requirement = requirement.model_copy(update={"required_actions": (forged_action,)})
    with pytest.raises(ValidationError, match="kind"):
        ArtifactReport.from_requirements(
            mode="batch", applicable=True, requirements=(forged_requirement,)
        )


def test_report_round_trips_paths_actions_and_open_tokens(tmp_path: Path) -> None:
    action = ArtifactAction.model_validate(
        {
            "kind": "x_vendor_license",
            "message": "Complete the external action.",
            "url": "https://example.com/action",
        }
    )
    requirement = ArtifactRequirement(
        artifact_id="weights:v1",
        label="Weights",
        state="future_state",
        required_for_inference=True,
        can_acquire_now=False,
        may_acquire_during_inference=True,
        source_is_mutable=True,
        acquisition_blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
        required_actions=(action,),
        location=tmp_path,
        size_bytes=5,
        expected_size_bytes=10,
        artifact_version="main",
    )
    report = ArtifactReport.from_requirements(
        mode="batch", applicable=True, requirements=(requirement,)
    )
    dumped = report.model_dump(mode="json")
    assert ArtifactReport.model_validate(dumped) == report


def test_progress_accepts_open_phases_and_coherent_counts() -> None:
    progress = ArtifactProgress(
        phase=ARTIFACT_PROGRESS_EXTRACTING,
        artifact_id="weights",
        completed_units=4,
        total_units=10,
        unit=ARTIFACT_PROGRESS_UNIT_BYTES,
    )
    assert progress.completed_units == 4
    assert ArtifactProgress(phase="x_vendor_indexing").phase == "x_vendor_indexing"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"completed_units": 1},
        {"total_units": 1},
        {"unit": ARTIFACT_PROGRESS_UNIT_BYTES},
        {
            "completed_units": 2,
            "total_units": 1,
            "unit": ARTIFACT_PROGRESS_UNIT_BYTES,
        },
        {"completed_units": -1, "unit": ARTIFACT_PROGRESS_UNIT_BYTES},
    ],
)
def test_progress_rejects_incoherent_counts(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ArtifactProgress.model_validate({"phase": ARTIFACT_PROGRESS_EXTRACTING, **kwargs})


def test_artifact_errors_carry_structured_context() -> None:
    report = ArtifactReport.from_requirements(mode="batch", applicable=False)
    action = ArtifactAction(kind=ARTIFACT_ACTION_ACCEPT_TERMS, message="Accept terms.")

    status = ArtifactStatusError("Status failed.", hint="Retry the status query.")
    assert isinstance(status, StructuredError)
    assert status.hint == "Retry the status query."

    unavailable = ArtifactUnavailableError(
        "Artifacts are unavailable.",
        reason="missing",
        report=report,
        hint="Acquire the artifacts.",
    )
    assert unavailable.reason == "missing"
    assert unavailable.report == report

    acquisition = ArtifactAcquisitionError(
        "Acquisition failed.",
        reason="action_required",
        report=report,
        required_actions=(action,),
        retriable_after=2.5,
        hint="Complete the action, and retry.",
    )
    assert acquisition.required_actions == (action,)
    assert acquisition.retriable_after == 2.5
    assert acquisition.report == report

    callback = ArtifactProgressCallbackError("Callback failed.", report=report)
    assert callback.report is report
    assert isinstance(callback, StandardASRError)


@pytest.mark.parametrize("delay", [-1.0, math.inf, math.nan])
def test_acquisition_error_rejects_invalid_retry_delay(delay: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ArtifactAcquisitionError(
            "Acquisition failed.",
            reason="failed",
            retriable_after=delay,
        )


@pytest.mark.parametrize(
    "reason",
    ["missing", "incomplete", "corrupt", "unknown", "action_required", "downloads_disabled"],
)
def test_unavailable_error_accepts_every_closed_reason(reason: str) -> None:
    report = ArtifactReport.from_requirements(mode="batch", applicable=False)
    error = ArtifactUnavailableError(
        "Artifacts are unavailable.",
        reason=cast("Any", reason),
        report=report,
    )
    assert error.reason == reason


@pytest.mark.parametrize(
    "reason", ["downloads_disabled", "action_required", "unsupported", "busy", "failed"]
)
def test_acquisition_error_accepts_every_closed_reason(reason: str) -> None:
    error = ArtifactAcquisitionError("Acquisition failed.", reason=cast("Any", reason))
    assert error.reason == reason


def test_artifact_errors_reject_a_reason_outside_the_closed_vocabulary() -> None:
    # The Literal annotation does not run, and consumers branch on the token
    # (the CLI's exit-code split); a typo'd reason must fail at the raise
    # site that authored it, not misclassify the failure downstream.
    report = ArtifactReport.from_requirements(mode="batch", applicable=False)
    with pytest.raises(ValueError, match="'action_requred' is not one of"):
        ArtifactUnavailableError(
            "Artifacts are unavailable.",
            reason=cast("Any", "action_requred"),
            report=report,
        )
    with pytest.raises(ValueError, match="'action_requred' is not one of"):
        ArtifactAcquisitionError("Acquisition failed.", reason=cast("Any", "action_requred"))


def test_artifact_errors_reject_a_wrong_typed_reason() -> None:
    # A reason that cannot be hashed previously crashed the CLI's
    # set-membership exit split (`exc.reason in {...}` raises TypeError for a
    # list) inside its exception handler; the constructor now owns that
    # failure.
    report = ArtifactReport.from_requirements(mode="batch", applicable=False)
    with pytest.raises(TypeError, match="reason must be one of"):
        ArtifactUnavailableError(
            "Artifacts are unavailable.", reason=cast("Any", []), report=report
        )
    with pytest.raises(TypeError, match="reason must be one of"):
        ArtifactAcquisitionError("Acquisition failed.", reason=cast("Any", []))


def test_artifact_errors_require_a_typed_report() -> None:
    with pytest.raises(TypeError, match="report must be an ArtifactReport, not None"):
        ArtifactUnavailableError(
            "Artifacts are unavailable.", reason="missing", report=cast("Any", None)
        )
    with pytest.raises(TypeError, match="report must be an ArtifactReport, not dict"):
        ArtifactUnavailableError(
            "Artifacts are unavailable.", reason="missing", report=cast("Any", {})
        )
    with pytest.raises(TypeError, match="report must be an ArtifactReport, not dict"):
        ArtifactAcquisitionError("Acquisition failed.", reason="failed", report=cast("Any", {}))
    assert ArtifactAcquisitionError("Acquisition failed.", reason="failed").report is None


def test_artifact_errors_revalidate_the_attached_report() -> None:
    # The CLI's exception handler walks report.requirements and each
    # requirement's actions. An isinstance check proves the class, not the
    # contents: model_copy(update=...) skips validation by design, so a report
    # whose requirements never met a validator would crash the reporting
    # boundary instead of the raise site that attached it.
    report = ArtifactReport.from_requirements(
        mode="batch", applicable=True, requirements=(_requirement(ARTIFACT_MISSING),)
    )
    forged = report.model_copy(update={"requirements": ({"artifact_id": "asr_weights"},)})
    with pytest.raises(ValidationError):
        ArtifactUnavailableError("Artifacts are unavailable.", reason="missing", report=forged)
    with pytest.raises(ValidationError):
        ArtifactAcquisitionError("Acquisition failed.", reason="failed", report=forged)
    # The model-level validator is skipped too: a stored readiness that
    # contradicts the requirements must fail the same way.
    contradictory = report.model_copy(update={"readiness": ARTIFACTS_READY})
    with pytest.raises(ValidationError, match="does not match the derived readiness"):
        ArtifactUnavailableError(
            "Artifacts are unavailable.", reason="missing", report=contradictory
        )


def test_acquisition_error_revalidates_directly_attached_actions() -> None:
    # A directly attached action crosses no model boundary, so without
    # constructor re-validation a model_copy(update=...) value that skipped
    # the HTTPS/user-information validators would ride the error into the
    # CLI's action rendering (AR.8 forbids credentials there).
    action = ArtifactAction(kind=ARTIFACT_ACTION_ACCEPT_TERMS, message="Accept terms.")
    forged = action.model_copy(update={"url": "https://user:secret@example.com/terms"})
    with pytest.raises(ValidationError):
        ArtifactAcquisitionError(
            "Acquisition failed.",
            reason="action_required",
            required_actions=(forged,),
        )


def test_acquisition_error_normalizes_and_rejects_action_payloads() -> None:
    action = ArtifactAction(kind=ARTIFACT_ACTION_ACCEPT_TERMS, message="Accept terms.")
    error = ArtifactAcquisitionError(
        "Acquisition failed.",
        reason="action_required",
        required_actions=cast("Any", [action]),
    )
    assert error.required_actions == (action,)
    assert type(error.required_actions) is tuple
    with pytest.raises(TypeError, match="items must be ArtifactAction, not str"):
        ArtifactAcquisitionError(
            "Acquisition failed.",
            reason="action_required",
            required_actions=cast("Any", ("accept terms",)),
        )
