# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the EngineBase inference-artifact lifecycle template."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal, cast

import pytest

from standard_asr.audio.input import InputKind
from standard_asr.contract.artifacts import (
    ARTIFACT_ACTION_PROVIDE_ARTIFACTS,
    ARTIFACT_BLOCKER_ACTION_REQUIRED,
    ARTIFACT_BLOCKER_DOWNLOADS_DISABLED,
    ARTIFACT_BLOCKER_UNSUPPORTED,
    ARTIFACT_MISSING,
    ARTIFACT_PROGRESS_FINALIZING,
    ARTIFACT_PROGRESS_RESOLVING,
    ARTIFACT_PROGRESS_TRANSFERRING,
    ARTIFACT_READY,
    ARTIFACT_UNKNOWN,
    ARTIFACTS_NOT_APPLICABLE,
    ArtifactAction,
    ArtifactContext,
    ArtifactProgress,
    ArtifactProgressCallback,
    ArtifactReport,
    ArtifactRequirement,
)
from standard_asr.contract.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    StreamingCapabilities,
)
from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactProgressCallbackError,
    ArtifactStatusError,
    ConfigError,
    EngineContractError,
    InvalidProviderParamError,
    ProtocolCompatibilityError,
)
from standard_asr.contract.metadata import (
    NO_ARTIFACT_LIFECYCLE,
    ArtifactDeclaration,
    DeclaredEngineMetadata,
)
from standard_asr.contract.params import (
    ProviderParams,
    RuntimeParams,
    WordTimestampGranularity,
)
from standard_asr.contract.properties import BaseProperties, SampleRateRange
from standard_asr.contract.results import Diagnostic, TranscriptionResult
from standard_asr.runtime.config import BaseConfig, LanguageConfigMixin
from standard_asr.runtime.interface import EngineBase, require_artifact_protocol


class _ArtifactConfig(LanguageConfigMixin, BaseConfig[Literal["artifact-test"]]):
    engine: Literal["artifact-test"] = "artifact-test"
    default_language: str | None = "en"


class _ArtifactProperties(BaseProperties):
    engine_id: str = "artifact-test"
    model_name: str = "fixture"
    protocol_version: str = "0.2.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    selectable_languages: list[str] = ["en", "fr", "auto"]
    detectable_languages: list[str] = ["en", "fr"]


class _OutsideLineArtifactProperties(_ArtifactProperties):
    protocol_version: str = "0.1.0"


class _OwnProviderParams(ProviderParams):
    beam_size: int = 1


class _ForeignProviderParams(ProviderParams):
    temperature: float = 0.0


_ARTIFACT_DECLARATION = ArtifactDeclaration(
    applicable=True,
    supports_explicit_acquisition=True,
    may_acquire_during_inference=True,
)
_ARTIFACT_METADATA = DeclaredEngineMetadata(artifacts=_ARTIFACT_DECLARATION)
_NO_EXPLICIT_METADATA = DeclaredEngineMetadata(
    artifacts=ArtifactDeclaration(
        applicable=True,
        supports_explicit_acquisition=False,
        may_acquire_during_inference=True,
    )
)
_NO_ARTIFACT_METADATA = DeclaredEngineMetadata(artifacts=NO_ARTIFACT_LIFECYCLE)
_BATCH_CAPABILITIES = DeclaredCapabilities(batch=BatchCapabilities())
_STREAMING_CAPABILITIES = DeclaredCapabilities(streaming=StreamingCapabilities())


def _requirement(
    state: str = ARTIFACT_MISSING,
    *,
    artifact_id: str = "weights",
    required: bool = True,
    can_acquire_now: bool | None = None,
    inference_acquisition: bool = False,
    mutable: bool = False,
    blocker: str | None = None,
    actions: tuple[ArtifactAction, ...] = (),
) -> ArtifactRequirement:
    """Build a coherent requirement for interface tests."""
    if state == ARTIFACT_READY:
        can_acquire_now = False
        blocker = None
        actions = ()
    elif can_acquire_now is None:
        can_acquire_now = blocker is None
    return ArtifactRequirement(
        artifact_id=artifact_id,
        label=f"Artifact {artifact_id}",
        state=state,
        required_for_inference=required,
        can_acquire_now=can_acquire_now,
        may_acquire_during_inference=inference_acquisition,
        source_is_mutable=mutable,
        acquisition_blocker=blocker,
        required_actions=actions,
    )


def _ready(requirement: ArtifactRequirement) -> ArtifactRequirement:
    """Return the ready form of a requirement."""
    return ArtifactRequirement.model_validate(
        {
            **requirement.model_dump(),
            "state": ARTIFACT_READY,
            "can_acquire_now": False,
            "acquisition_blocker": None,
            "required_actions": (),
        }
    )


class _ArtifactEngine(EngineBase):
    properties: ClassVar[BaseProperties] = _ArtifactProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _BATCH_CAPABILITIES
    declared_metadata: ClassVar[DeclaredEngineMetadata] = _ARTIFACT_METADATA
    provider_params_type: ClassVar[type[ProviderParams] | None] = _OwnProviderParams

    config: BaseConfig[str]
    applicable: bool
    requirements: tuple[ArtifactRequirement, ...]
    engine_diagnostics: tuple[Diagnostic, ...]
    captured_contexts: list[ArtifactContext]
    acquired: list[tuple[ArtifactRequirement, ...]]
    refresh_values: list[bool]
    status_calls: int
    status_errors: dict[int, Exception]
    native_error: Exception | None
    emit_progress: bool
    drop_attempted: bool
    leave_attempted_nonready: bool

    def __init__(
        self,
        requirements: tuple[ArtifactRequirement, ...] = (),
        *,
        applicable: bool = True,
        status_errors: dict[int, Exception] | None = None,
        native_error: Exception | None = None,
        emit_progress: bool = False,
        drop_attempted: bool = False,
        leave_attempted_nonready: bool = False,
    ) -> None:
        self.config = _ArtifactConfig(strict=True)
        self.applicable = applicable
        self.requirements = requirements
        self.engine_diagnostics = ()
        self.captured_contexts = []
        self.acquired = []
        self.refresh_values = []
        self.status_calls = 0
        self.status_errors = status_errors or {}
        self.native_error = native_error
        self.emit_progress = emit_progress
        self.drop_attempted = drop_attempted
        self.leave_attempted_nonready = leave_attempted_nonready

    def _transcribe(self, prepared: object, params: RuntimeParams) -> TranscriptionResult:
        return TranscriptionResult(text="unused")

    def _artifact_requirements(
        self,
        context: ArtifactContext,
    ) -> tuple[bool, tuple[ArtifactRequirement, ...], tuple[Diagnostic, ...]]:
        self.status_calls += 1
        self.captured_contexts.append(context)
        error = self.status_errors.get(self.status_calls)
        if error is not None:
            raise error
        return self.applicable, self.requirements, self.engine_diagnostics

    def _acquire_artifacts(
        self,
        context: ArtifactContext,
        requirements: tuple[ArtifactRequirement, ...],
        refresh: bool,
        progress: Any,
    ) -> None:
        self.captured_contexts.append(context)
        self.acquired.append(requirements)
        self.refresh_values.append(refresh)
        if self.emit_progress and progress is not None:
            progress(
                ArtifactProgress(
                    phase=ARTIFACT_PROGRESS_TRANSFERRING,
                    artifact_id=requirements[0].artifact_id,
                )
            )
        if self.native_error is not None:
            raise self.native_error
        target_ids = {requirement.artifact_id for requirement in requirements}
        if self.drop_attempted:
            self.requirements = tuple(
                requirement
                for requirement in self.requirements
                if requirement.artifact_id not in target_ids
            )
        elif not self.leave_attempted_nonready:
            self.requirements = tuple(
                _ready(requirement)
                if requirement.artifact_id in target_ids and requirement.state != ARTIFACT_READY
                else requirement
                for requirement in self.requirements
            )


class _StreamingArtifactEngine(_ArtifactEngine):
    declared_capabilities: ClassVar[DeclaredCapabilities] = _STREAMING_CAPABILITIES


class _NoModesArtifactEngine(_ArtifactEngine):
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities()


class _NoArtifactEngine(_ArtifactEngine):
    declared_capabilities: ClassVar[DeclaredCapabilities] = DeclaredCapabilities()
    declared_metadata = _NO_ARTIFACT_METADATA


class _NoExplicitArtifactEngine(_ArtifactEngine):
    declared_metadata = _NO_EXPLICIT_METADATA


class _OutsideLineEngine(_ArtifactEngine):
    properties: ClassVar[BaseProperties] = _OutsideLineArtifactProperties()
    declared_metadata = cast("DeclaredEngineMetadata", None)


class _InvalidStatusHookEngine(_ArtifactEngine):
    raw_output: object

    def __init__(self, raw_output: object) -> None:
        super().__init__()
        self.raw_output = raw_output

    def _artifact_requirements(
        self,
        context: ArtifactContext,
    ) -> tuple[bool, tuple[ArtifactRequirement, ...], tuple[Diagnostic, ...]]:
        del context
        return cast(Any, self.raw_output)


class _InvalidProgressEngine(_ArtifactEngine):
    def _acquire_artifacts(
        self,
        context: ArtifactContext,
        requirements: tuple[ArtifactRequirement, ...],
        refresh: bool,
        progress: Any,
    ) -> None:
        if progress is not None:
            progress("not-progress")
        super()._acquire_artifacts(context, requirements, refresh, progress=None)


class _ConstructedInvalidProgressEngine(_ArtifactEngine):
    def _acquire_artifacts(
        self,
        context: ArtifactContext,
        requirements: tuple[ArtifactRequirement, ...],
        refresh: bool,
        progress: Any,
    ) -> None:
        if progress is not None:
            progress(
                ArtifactProgress.model_construct(
                    phase="INVALID PHASE",
                    completed_units=2,
                    total_units=1,
                    unit=None,
                )
            )
        super()._acquire_artifacts(context, requirements, refresh, progress=None)


class _WrongAcquireReturnEngine(_ArtifactEngine):
    def _acquire_artifacts(
        self,
        context: ArtifactContext,
        requirements: tuple[ArtifactRequirement, ...],
        refresh: bool,
        progress: Any,
    ) -> None:
        super()._acquire_artifacts(context, requirements, refresh, progress)
        return cast(Any, "wrong")


class _AddsRequiredRequirementEngine(_ArtifactEngine):
    def _acquire_artifacts(
        self,
        context: ArtifactContext,
        requirements: tuple[ArtifactRequirement, ...],
        refresh: bool,
        progress: Any,
    ) -> None:
        super()._acquire_artifacts(context, requirements, refresh, progress)
        self.requirements = (
            *self.requirements,
            _requirement(artifact_id="new_required"),
        )


def test_status_uses_best_effort_gating_even_when_engine_is_strict() -> None:
    engine = _ArtifactEngine()
    engine.engine_diagnostics = (
        Diagnostic(level="info", code="engine_status", message="Native status inspected."),
    )
    report = engine.artifact_status(
        ArtifactContext(
            params=RuntimeParams(
                language="fr",
                word_timestamps=WordTimestampGranularity.WORD,
            )
        )
    )

    captured = engine.captured_contexts[-1]
    assert captured.params.language == "en"
    assert captured.params.word_timestamps is None
    assert {diagnostic.param for diagnostic in report.diagnostics} >= {
        "language",
        "word_timestamps",
    }
    assert report.diagnostics[-1].code == "engine_status"


def test_status_rejects_wrong_engine_provider_params_before_hook() -> None:
    engine = _ArtifactEngine()
    context = ArtifactContext(params=RuntimeParams(provider_params=_ForeignProviderParams()))

    with pytest.raises(InvalidProviderParamError):
        engine.artifact_status(context)
    assert engine.status_calls == 0


def test_acquire_rejects_wrong_engine_provider_params_before_hook() -> None:
    engine = _ArtifactEngine((_requirement(),))
    context = ArtifactContext(params=RuntimeParams(provider_params=_ForeignProviderParams()))

    with pytest.raises(InvalidProviderParamError):
        engine.acquire_artifacts(context)
    assert engine.status_calls == 0


def test_status_prefers_batch_as_omitted_mode() -> None:
    engine = _ArtifactEngine()
    report = engine.artifact_status()
    assert report.mode == "batch"
    assert engine.captured_contexts[-1].mode == "batch"


def test_status_falls_back_to_only_streaming_mode() -> None:
    engine = _StreamingArtifactEngine()
    report = engine.artifact_status()
    assert report.mode == "streaming"
    assert engine.captured_contexts[-1].mode == "streaming"


def test_status_accepts_an_explicit_supported_mode() -> None:
    engine = _ArtifactEngine()
    report = engine.artifact_status(ArtifactContext(mode="batch"))
    assert report.mode == "batch"


@pytest.mark.parametrize(
    ("engine", "mode"),
    [
        (_ArtifactEngine(), "streaming"),
        (_StreamingArtifactEngine(), "batch"),
    ],
)
def test_status_rejects_an_explicit_unsupported_mode(
    engine: _ArtifactEngine,
    mode: Literal["batch", "streaming"],
) -> None:
    with pytest.raises(ValueError, match="is not supported"):
        engine.artifact_status(ArtifactContext(mode=mode))
    assert engine.status_calls == 0


def test_nonapplicable_engine_without_modes_uses_batch_report_fallback() -> None:
    engine = _NoArtifactEngine()
    report = engine.artifact_status()
    assert report.mode == "batch"
    assert report.applicable is False
    assert report.readiness == ARTIFACTS_NOT_APPLICABLE
    assert engine.status_calls == 0


def test_applicable_engine_without_modes_requires_explicit_mode() -> None:
    engine = _NoModesArtifactEngine()
    with pytest.raises(ConfigError, match="mode is required"):
        engine.artifact_status()
    assert engine.status_calls == 0


@pytest.mark.parametrize(
    "raw_output",
    [
        None,
        (True, ()),
        (1, (), ()),
        (True, [], ()),
        (True, (object(),), ()),
        (True, (), []),
        (True, (), (object(),)),
    ],
)
def test_status_rejects_invalid_hook_return_shape(raw_output: object) -> None:
    engine = _InvalidStatusHookEngine(raw_output)
    with pytest.raises(EngineContractError, match="_artifact_requirements"):
        engine.artifact_status()


def test_status_rejects_awaitable_hook_result() -> None:
    async def result() -> tuple[bool, tuple[ArtifactRequirement, ...], tuple[Diagnostic, ...]]:
        return True, (), ()

    engine = _InvalidStatusHookEngine(result())
    with pytest.raises(EngineContractError, match="awaitable"):
        engine.artifact_status()


def test_status_requires_hook_when_artifacts_are_declared_applicable() -> None:
    class _MissingStatusHookEngine(_ArtifactEngine):
        _artifact_requirements = EngineBase._artifact_requirements

    with pytest.raises(EngineContractError, match="must override"):
        _MissingStatusHookEngine().artifact_status()


def test_protocol_1_1_status_requires_authored_metadata() -> None:
    class _MissingMetadataEngine(_ArtifactEngine):
        declared_metadata = cast("DeclaredEngineMetadata", None)

    engine = _MissingMetadataEngine()
    with pytest.raises(EngineContractError, match="DeclaredEngineMetadata"):
        engine.artifact_status()
    assert engine.status_calls == 0


def test_protocol_1_1_status_requires_a_typed_artifacts_section() -> None:
    # The outer isinstance proves the class, not the section's type. A metadata
    # object whose artifacts section is a raw mapping -- what
    # model_copy(update=...) stores when handed one -- passes that check, and
    # every caller then reads .applicable off a dict. That
    # escaped as a bare AttributeError from a public lifecycle method; an
    # invalid declaration is an engine fault (AR.2), so it must be typed.
    class _UntypedSectionEngine(_ArtifactEngine):
        declared_metadata: ClassVar[DeclaredEngineMetadata] = _ARTIFACT_METADATA.model_copy(
            update={
                "artifacts": {
                    "applicable": False,
                    "supports_explicit_acquisition": False,
                    "may_acquire_during_inference": False,
                }
            }
        )

    engine = _UntypedSectionEngine()
    with pytest.raises(EngineContractError, match="must be an ArtifactDeclaration"):
        engine.artifact_status()
    assert engine.status_calls == 0


def test_status_enforces_declaration_values_that_skipped_validation() -> None:
    # The isinstance proves the section's class, not its values: model_copy
    # stores the string "false", which is TRUTHY, so the AR.2 narrowing guard
    # accepted can_acquire_now=True while canonical_json() publishes false for
    # the same metadata -- two verdicts for one declaration. Re-validation
    # applies construction semantics ("false" coerces to False), and the
    # guard then rejects the report as widening the static bound.
    class _CoercibleDeclarationEngine(_ArtifactEngine):
        declared_metadata: ClassVar[DeclaredEngineMetadata] = _ARTIFACT_METADATA.model_copy(
            update={
                "artifacts": _ARTIFACT_DECLARATION.model_copy(
                    update={"supports_explicit_acquisition": "false"}
                )
            }
        )

    assert _CoercibleDeclarationEngine.declared_metadata.canonical_json() == {
        "artifacts": {
            "applicable": True,
            "supports_explicit_acquisition": False,
            "may_acquire_during_inference": True,
        }
    }
    engine = _CoercibleDeclarationEngine((_requirement(),))
    with pytest.raises(EngineContractError, match="explicit acquisition that declared metadata"):
        engine.artifact_status()


def test_status_rejects_declaration_values_the_model_would_reject() -> None:
    class _GarbageDeclarationEngine(_ArtifactEngine):
        declared_metadata: ClassVar[DeclaredEngineMetadata] = _ARTIFACT_METADATA.model_copy(
            update={"artifacts": _ARTIFACT_DECLARATION.model_copy(update={"applicable": "maybe"})}
        )

    engine = _GarbageDeclarationEngine()
    with pytest.raises(EngineContractError, match="fails re-validation"):
        engine.artifact_status()
    assert engine.status_calls == 0


def test_artifact_protocol_guard_requires_typed_properties() -> None:
    with pytest.raises(EngineContractError, match="requires engine properties"):
        require_artifact_protocol(object())


def test_base_status_hook_returns_nonapplicable_empty_shape() -> None:
    engine = _NoArtifactEngine()
    assert EngineBase._artifact_requirements(  # pyright: ignore[reportPrivateUsage]
        engine, ArtifactContext(mode="batch")
    ) == (
        False,
        (),
        (),
    )


def test_status_normalizes_cross_field_report_validation_as_contract_error() -> None:
    engine = _InvalidStatusHookEngine((False, (_requirement(),), ()))
    with pytest.raises(EngineContractError, match="invalid artifact report"):
        engine.artifact_status()


def test_status_normalizes_field_level_requirement_validation_as_contract_error() -> None:
    # The field-level twin of the case above, end to end through the template.
    # A hook returning a requirement whose FIELD constraint was skipped -- here
    # a relative location, which the protocol says MUST be absolute -- is an
    # engine fault, so it must arrive as EngineContractError (AR.2) and not as
    # the raw ValidationError, and it must not reach the caller as a report.
    forged = _requirement(ARTIFACT_READY).model_copy(update={"location": Path("relative/dir")})
    engine = _InvalidStatusHookEngine((True, (forged,), ()))
    with pytest.raises(EngineContractError, match="invalid artifact report"):
        engine.artifact_status()


def test_status_rejects_hook_diagnostics_that_skipped_validation() -> None:
    # The requirements re-validate when the report nests them, but plain
    # Diagnostic keeps the framework-wide 1.0 config, so a model_copy(...)
    # diagnostic crossed the report boundary verbatim: a level token outside
    # the closed vocabulary published through every projection. AR.2 makes
    # invalid hook output an EngineContractError at this seam.
    engine = _ArtifactEngine((_requirement(ARTIFACT_READY),))
    engine.engine_diagnostics = (
        Diagnostic(code="artifact_note", message="Note.").model_copy(update={"level": "fatal"}),
    )
    with pytest.raises(EngineContractError, match="invalid Diagnostic"):
        engine.artifact_status()


@pytest.mark.parametrize("provided", [float("nan"), object()], ids=["nan", "object"])
def test_status_rejects_hook_diagnostic_values_with_no_json_form(provided: object) -> None:
    # A NaN ``provided`` silently became JSON null, and a non-JSON object
    # deferred the crash to the wire projection, long after the hook boundary
    # that owns the fault.
    engine = _ArtifactEngine((_requirement(ARTIFACT_READY),))
    engine.engine_diagnostics = (
        Diagnostic(code="artifact_note", message="Note.").model_copy(update={"provided": provided}),
    )
    with pytest.raises(EngineContractError, match="invalid Diagnostic"):
        engine.artifact_status()


def test_status_wraps_unexpected_native_failure() -> None:
    native = RuntimeError("native status failed")
    engine = _ArtifactEngine(status_errors={1: native})
    with pytest.raises(ArtifactStatusError) as caught:
        engine.artifact_status()
    assert caught.value.__cause__ is native


@pytest.mark.parametrize(
    "error",
    [
        ArtifactStatusError("typed status"),
        ConfigError("typed config"),
        EngineContractError("typed contract"),
    ],
)
def test_status_preserves_typed_hook_failure(error: Exception) -> None:
    engine = _ArtifactEngine(status_errors={1: error})
    with pytest.raises(type(error)) as caught:
        engine.artifact_status()
    assert caught.value is error


def test_status_rejects_dynamic_explicit_acquisition_widening() -> None:
    engine = _NoExplicitArtifactEngine((_requirement(),))
    with pytest.raises(EngineContractError, match="explicit acquisition"):
        engine.artifact_status()


def test_status_rejects_dynamic_inference_acquisition_widening() -> None:
    class _NoImplicitArtifactEngine(_ArtifactEngine):
        declared_metadata = DeclaredEngineMetadata(
            artifacts=ArtifactDeclaration(
                applicable=True,
                supports_explicit_acquisition=True,
                may_acquire_during_inference=False,
            )
        )

    requirement = _requirement(inference_acquisition=True)
    with pytest.raises(EngineContractError, match="inference acquisition"):
        _NoImplicitArtifactEngine((requirement,)).artifact_status()


def test_status_rejects_dynamic_applicability_widening() -> None:
    report = ArtifactReport.from_requirements(mode="batch", applicable=True)
    with pytest.raises(EngineContractError, match="reports applicability"):
        EngineBase._artifact_report_matches_declaration(  # pyright: ignore[reportPrivateUsage]
            report, NO_ARTIFACT_LIFECYCLE
        )


def test_status_hook_requires_an_exact_tuple() -> None:
    class _TupleSubclass(tuple[Any, ...]):
        pass

    engine = _InvalidStatusHookEngine(_TupleSubclass((True, (), ())))
    with pytest.raises(EngineContractError, match="three-item tuple"):
        engine.artifact_status()


def test_acquire_noops_for_nonapplicable_report() -> None:
    engine = _NoArtifactEngine()
    report = engine.acquire_artifacts()
    assert report.readiness == ARTIFACTS_NOT_APPLICABLE
    assert engine.acquired == []


def test_acquire_materializes_runnable_requirement_and_rechecks_status() -> None:
    engine = _ArtifactEngine((_requirement(),))
    report = engine.acquire_artifacts()
    assert [item.artifact_id for item in engine.acquired[0]] == ["weights"]
    assert report.requirements[0].state == ARTIFACT_READY
    assert engine.status_calls == 2


def test_acquire_final_report_preserves_original_context_diagnostics() -> None:
    engine = _ArtifactEngine((_requirement(),))
    context = ArtifactContext(params=RuntimeParams(word_timestamps=WordTimestampGranularity.WORD))

    report = engine.acquire_artifacts(context)

    assert any(diagnostic.param == "word_timestamps" for diagnostic in report.diagnostics)
    assert engine.captured_contexts[1].params.word_timestamps is None


def test_acquire_allows_local_work_when_downloads_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    engine = _ArtifactEngine((_requirement(mutable=False),))
    report = engine.acquire_artifacts()
    assert report.requirements[0].state == ARTIFACT_READY
    assert len(engine.acquired) == 1


def test_refresh_targets_only_unblocked_mutable_requirements() -> None:
    action = ArtifactAction(
        kind=ARTIFACT_ACTION_PROVIDE_ARTIFACTS,
        message="Complete the external prerequisite.",
    )
    mutable = _requirement(ARTIFACT_READY, artifact_id="mutable", mutable=True)
    immutable = _requirement(ARTIFACT_READY, artifact_id="immutable", mutable=False)
    action_blocked = _requirement(
        artifact_id="gated",
        required=False,
        mutable=True,
        can_acquire_now=False,
        blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
        actions=(action,),
    )
    engine = _ArtifactEngine((mutable, immutable, action_blocked))

    report = engine.acquire_artifacts(refresh=True)

    assert [[item.artifact_id for item in call] for call in engine.acquired] == [["mutable"]]
    assert engine.refresh_values == [True]
    assert report.requirements[-1] == action_blocked


def test_refresh_with_only_immutable_ready_requirements_is_noop() -> None:
    immutable = _requirement(ARTIFACT_READY, mutable=False)
    engine = _ArtifactEngine((immutable,))
    report = engine.acquire_artifacts(refresh=True)
    assert report.requirements == (immutable,)
    assert engine.acquired == []


def test_refresh_rejects_any_mutable_source_when_downloads_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    action = ArtifactAction(
        kind=ARTIFACT_ACTION_PROVIDE_ARTIFACTS,
        message="Complete the external prerequisite.",
    )
    blocked_optional = _requirement(
        required=False,
        mutable=True,
        can_acquire_now=False,
        blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
        actions=(action,),
    )
    engine = _ArtifactEngine((blocked_optional,))

    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts(refresh=True)
    assert caught.value.reason == "downloads_disabled"
    assert engine.acquired == []


def test_mutable_refresh_requires_effective_explicit_acquisition() -> None:
    ready_mutable = _requirement(ARTIFACT_READY, mutable=True)
    engine = _NoExplicitArtifactEngine((ready_mutable,))
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts(refresh=True)
    assert caught.value.reason == "unsupported"
    assert engine.acquired == []


def test_required_blocker_precedence_and_action_projection() -> None:
    action = ArtifactAction(
        kind=ARTIFACT_ACTION_PROVIDE_ARTIFACTS,
        message="Provide the required files.",
    )
    requirements = (
        _requirement(
            artifact_id="unsupported",
            can_acquire_now=False,
            blocker=ARTIFACT_BLOCKER_UNSUPPORTED,
        ),
        _requirement(
            artifact_id="disabled",
            can_acquire_now=False,
            blocker=ARTIFACT_BLOCKER_DOWNLOADS_DISABLED,
        ),
        _requirement(
            artifact_id="action",
            can_acquire_now=False,
            blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
            actions=(action,),
        ),
    )
    engine = _ArtifactEngine(requirements)
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value.reason == "action_required"
    assert caught.value.required_actions == (action,)


def test_unknown_blocker_projects_to_unsupported_reason() -> None:
    requirement = _requirement(
        can_acquire_now=False,
        blocker="x_vendor_policy",
    )
    engine = _ArtifactEngine((requirement,))
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value.reason == "unsupported"
    assert caught.value.report is not None
    assert caught.value.report.requirements[0].acquisition_blocker == "x_vendor_policy"


def test_downloads_disabled_blocker_projects_to_download_reason() -> None:
    requirement = _requirement(
        can_acquire_now=False,
        blocker=ARTIFACT_BLOCKER_DOWNLOADS_DISABLED,
    )
    engine = _ArtifactEngine((requirement,))
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value.reason == "downloads_disabled"


def test_optional_blocker_returns_report_without_error() -> None:
    optional = _requirement(
        required=False,
        can_acquire_now=False,
        blocker=ARTIFACT_BLOCKER_UNSUPPORTED,
    )
    engine = _ArtifactEngine((optional,))
    report = engine.acquire_artifacts()
    assert report.requirements == (optional,)
    assert engine.acquired == []


def test_mixed_acquisition_excludes_blocked_target_then_raises() -> None:
    action = ArtifactAction(
        kind=ARTIFACT_ACTION_PROVIDE_ARTIFACTS,
        message="Provide the aligner.",
    )
    runnable = _requirement(artifact_id="base")
    blocked = _requirement(
        artifact_id="aligner",
        can_acquire_now=False,
        blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
        actions=(action,),
    )
    engine = _ArtifactEngine((runnable, blocked))
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert [[item.artifact_id for item in call] for call in engine.acquired] == [["base"]]
    assert caught.value.reason == "action_required"
    # AR.3 makes the partial path re-query status and raise against the FRESH
    # report, not the preflight it started from. Without these two assertions an
    # implementation that skipped the re-query, or attached the stale preflight,
    # passed: the reason and the target exclusion above are identical either
    # way. The retained report is what an application renders, so a stale one
    # would show `base` still non-ready after it was just acquired.
    assert engine.status_calls == 2
    assert caught.value.report is not None
    states = {item.artifact_id: item.state for item in caught.value.report.requirements}
    assert states["base"] == ARTIFACT_READY
    assert states["aligner"] != ARTIFACT_READY


def test_progress_delivers_template_and_engine_events_in_order() -> None:
    engine = _ArtifactEngine((_requirement(),), emit_progress=True)
    phases: list[str] = []
    report = engine.acquire_artifacts(progress=lambda event: phases.append(event.phase))
    assert phases == [
        ARTIFACT_PROGRESS_RESOLVING,
        ARTIFACT_PROGRESS_TRANSFERRING,
        ARTIFACT_PROGRESS_FINALIZING,
    ]
    assert report.requirements[0].state == ARTIFACT_READY


def test_callback_failure_is_reported_after_successful_acquisition() -> None:
    engine = _ArtifactEngine((_requirement(),), emit_progress=True)
    seen: list[str] = []

    def fail(event: ArtifactProgress) -> None:
        seen.append(event.phase)
        raise RuntimeError("observer failed")

    with pytest.raises(ArtifactProgressCallbackError) as caught:
        engine.acquire_artifacts(progress=fail)
    assert seen == [ARTIFACT_PROGRESS_RESOLVING]
    assert caught.value.report.requirements[0].state == ARTIFACT_READY
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert engine.status_calls == 2


def test_native_failure_takes_precedence_over_callback_failure() -> None:
    native = RuntimeError("native acquisition failed")
    engine = _ArtifactEngine((_requirement(),), native_error=native)

    def fail(event: ArtifactProgress) -> None:
        del event
        raise ValueError("observer failed")

    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts(progress=fail)
    assert caught.value.reason == "failed"
    assert caught.value.__cause__ is native


def test_final_status_failure_takes_precedence_over_callback_failure() -> None:
    native_status = RuntimeError("final status failed")
    engine = _ArtifactEngine(
        (_requirement(),),
        status_errors={2: native_status},
    )

    def fail(event: ArtifactProgress) -> None:
        del event
        raise ValueError("observer failed")

    with pytest.raises(ArtifactStatusError) as caught:
        engine.acquire_artifacts(progress=fail)
    assert isinstance(caught.value.__cause__, ArtifactStatusError)
    assert caught.value.__cause__.__cause__ is native_status


def test_invalid_progress_value_becomes_contract_error_after_success() -> None:
    engine = _InvalidProgressEngine((_requirement(),))
    with pytest.raises(EngineContractError, match="not ArtifactProgress"):
        engine.acquire_artifacts(progress=lambda event: None)
    assert engine.requirements[0].state == ARTIFACT_READY


def test_model_construct_cannot_bypass_progress_validation() -> None:
    engine = _ConstructedInvalidProgressEngine((_requirement(),))
    delivered: list[ArtifactProgress] = []

    with pytest.raises(EngineContractError, match="invalid ArtifactProgress"):
        engine.acquire_artifacts(progress=delivered.append)

    assert delivered == [ArtifactProgress(phase=ARTIFACT_PROGRESS_RESOLVING)]
    assert engine.requirements[0].state == ARTIFACT_READY


def test_async_progress_callback_fails_after_success_without_leaking_coroutine() -> None:
    engine = _ArtifactEngine((_requirement(),))

    async def callback(event: ArtifactProgress) -> None:
        del event

    with pytest.raises(ArtifactProgressCallbackError) as caught:
        engine.acquire_artifacts(progress=cast("ArtifactProgressCallback", callback))

    assert isinstance(caught.value.__cause__, EngineContractError)
    assert caught.value.report.requirements[0].state == ARTIFACT_READY


def test_cancelled_progress_callback_does_not_cancel_acquisition() -> None:
    engine = _ArtifactEngine((_requirement(),))

    def callback(event: ArtifactProgress) -> None:
        del event
        raise asyncio.CancelledError

    with pytest.raises(ArtifactProgressCallbackError) as caught:
        engine.acquire_artifacts(progress=callback)

    assert isinstance(caught.value.__cause__, asyncio.CancelledError)
    assert caught.value.report.requirements[0].state == ARTIFACT_READY


def test_unexpected_native_acquisition_failure_is_normalized() -> None:
    native = RuntimeError("native failed")
    engine = _ArtifactEngine((_requirement(),), native_error=native)
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value.reason == "failed"
    assert caught.value.__cause__ is native
    assert caught.value.report is not None
    assert caught.value.report.requirements[0].state == ARTIFACT_MISSING


def test_typed_native_acquisition_failure_is_preserved() -> None:
    native = ArtifactAcquisitionError("typed failure", reason="busy")
    engine = _ArtifactEngine((_requirement(),), native_error=native)
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value is native


def test_typed_hook_error_without_a_report_gains_the_preflight() -> None:
    # AR.5: the error preserves the full report. A hook-level helper may
    # raise without one (no preflight in its scope); the template has the
    # preflight and must backfill it rather than surface a structured
    # error stripped of its status context (round-16 review).
    native = ArtifactAcquisitionError("gated companion", reason="action_required")
    engine = _ArtifactEngine((_requirement(),), native_error=native)
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value is native
    assert caught.value.report is not None
    assert caught.value.report.requirements[0].state == ARTIFACT_MISSING
    assert caught.value.reason == "action_required"


def test_typed_hook_error_keeps_its_own_report() -> None:
    # A hook that DID attach a report keeps it: the template never
    # overwrites the more specific context with the preflight.
    own_report = ArtifactReport.from_requirements(
        mode="batch", applicable=True, requirements=(_ready(_requirement()),)
    )
    native = ArtifactAcquisitionError("with report", reason="busy", report=own_report)
    engine = _ArtifactEngine((_requirement(),), native_error=native)
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    # Equality, not identity: the constructor re-validates the attached
    # report and stores the rebuilt copy. The preflight report would differ
    # (its requirement is missing, not ready).
    assert caught.value.report == own_report


def test_engine_contract_failure_from_acquisition_hook_is_preserved() -> None:
    native = EngineContractError("hook contract")
    engine = _ArtifactEngine((_requirement(),), native_error=native)
    with pytest.raises(EngineContractError) as caught:
        engine.acquire_artifacts()
    assert caught.value is native


def test_acquisition_hook_must_return_none() -> None:
    engine = _WrongAcquireReturnEngine((_requirement(),))
    with pytest.raises(EngineContractError, match="_acquire_artifacts"):
        engine.acquire_artifacts()


def test_explicit_declaration_requires_acquisition_hook_override() -> None:
    class _MissingAcquireHookEngine(_ArtifactEngine):
        _acquire_artifacts = EngineBase._acquire_artifacts

    engines = (
        _MissingAcquireHookEngine((_requirement(),)),
        _MissingAcquireHookEngine((_requirement(ARTIFACT_READY),)),
        _MissingAcquireHookEngine(applicable=False),
    )
    for engine in engines:
        with pytest.raises(EngineContractError, match="must override"):
            engine.acquire_artifacts()


def test_attempted_nonready_artifact_is_a_failed_acquisition() -> None:
    engine = _ArtifactEngine(
        (_requirement(),),
        leave_attempted_nonready=True,
    )
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value.reason == "failed"


def test_attempted_optional_unknown_artifact_is_not_reported_as_success() -> None:
    engine = _ArtifactEngine(
        (_requirement(ARTIFACT_UNKNOWN, required=False),),
        leave_attempted_nonready=True,
    )
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value.reason == "failed"


def test_attempted_artifact_missing_from_final_report_is_a_contract_error() -> None:
    engine = _ArtifactEngine(
        (_requirement(),),
        drop_attempted=True,
    )
    with pytest.raises(EngineContractError, match="omitted a target"):
        engine.acquire_artifacts()


def test_refresh_target_missing_from_final_report_is_a_contract_error() -> None:
    engine = _ArtifactEngine(
        (_requirement(ARTIFACT_READY, mutable=True),),
        drop_attempted=True,
    )
    with pytest.raises(EngineContractError, match="omitted a target"):
        engine.acquire_artifacts(refresh=True)


def test_new_required_nonready_requirement_in_final_report_is_failure() -> None:
    engine = _AddsRequiredRequirementEngine((_requirement(),))
    with pytest.raises(ArtifactAcquisitionError) as caught:
        engine.acquire_artifacts()
    assert caught.value.reason == "failed"
    assert caught.value.report is not None
    assert [item.artifact_id for item in caught.value.report.requirements] == [
        "weights",
        "new_required",
    ]


def test_artifact_guard_survives_the_feature_table_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Freeze-day counterexample (review round 23): the guard must not depend
    # on the feature table, whose artifact entry the first stable release
    # removes. Under frozen constants -- and an EMPTY table -- the guard
    # admits the stable line and its additive minors, and rejects the
    # pre-stable line.
    import standard_asr.contract.protocol_version as protocol_version_module

    monkeypatch.setattr(protocol_version_module, "SUPPORTED_PROTOCOL_MAJOR", 1)
    monkeypatch.setattr(protocol_version_module, "CURRENT_PROTOCOL_VERSION", "1.0.0")
    monkeypatch.setattr(
        protocol_version_module,
        "PROTOCOL_FEATURE_MINIMUMS",
        MappingProxyType(dict[str, str]()),
    )

    class _Holder:
        properties: BaseProperties

    for accepted in ("1.0.0", "1.1.0"):
        stub = _Holder()
        stub.properties = _ArtifactProperties(protocol_version=accepted)
        assert require_artifact_protocol(stub) is None
    stub = _Holder()
    stub.properties = _ArtifactProperties(protocol_version="0.2.0")
    with pytest.raises(ProtocolCompatibilityError):
        require_artifact_protocol(stub)


@pytest.mark.parametrize("operation", ["status", "acquire"])
def test_outside_line_guard_precedes_metadata_and_hooks(operation: str) -> None:
    engine = _OutsideLineEngine((_requirement(),))
    with pytest.raises(ProtocolCompatibilityError) as caught:
        if operation == "status":
            engine.artifact_status()
        else:
            engine.acquire_artifacts()
    assert caught.value.protocol_version == "0.1.0"
    assert engine.status_calls == 0
