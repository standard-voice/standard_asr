# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typed engine-metadata declaration surface."""

from __future__ import annotations

import math
import warnings

import pytest
from pydantic import ValidationError

from standard_asr.contract.metadata import (
    NO_ARTIFACT_LIFECYCLE,
    ArtifactDeclaration,
    DeclaredEngineMetadata,
)


def test_no_artifact_constant_is_an_explicit_frozen_declaration() -> None:
    assert NO_ARTIFACT_LIFECYCLE == ArtifactDeclaration(
        applicable=False,
        supports_explicit_acquisition=False,
        may_acquire_during_inference=False,
    )
    with pytest.raises(ValidationError, match="frozen"):
        NO_ARTIFACT_LIFECYCLE.applicable = True


@pytest.mark.parametrize(
    ("explicit", "implicit"),
    [(True, False), (False, True), (True, True)],
)
def test_artifact_paths_require_applicable_lifecycle(explicit: bool, implicit: bool) -> None:
    with pytest.raises(ValidationError, match="must be false"):
        ArtifactDeclaration(
            applicable=False,
            supports_explicit_acquisition=explicit,
            may_acquire_during_inference=implicit,
        )


def test_applicable_external_artifact_can_have_no_acquisition_path() -> None:
    declaration = ArtifactDeclaration(
        applicable=True,
        supports_explicit_acquisition=False,
        may_acquire_during_inference=False,
    )
    assert declaration.applicable is True


def test_metadata_requires_authored_artifact_section() -> None:
    with pytest.raises(ValidationError, match="artifacts"):
        DeclaredEngineMetadata.model_validate({})


def test_declaration_revalidates_an_instance_that_skipped_validation() -> None:
    # ``model_copy(update=...)`` skips every validator by design, and pydantic
    # accepts an existing instance verbatim at validation boundaries unless
    # the model opts into ``revalidate_instances`` -- so a declaration that
    # violates its own invariant would cross ``model_validate`` and nested
    # metadata construction unchallenged.
    forged = NO_ARTIFACT_LIFECYCLE.model_copy(update={"supports_explicit_acquisition": True})
    with pytest.raises(ValidationError, match="must be false"):
        ArtifactDeclaration.model_validate(forged)
    with pytest.raises(ValidationError, match="must be false"):
        DeclaredEngineMetadata(artifacts=forged)


def test_declaration_revalidation_applies_construction_semantics() -> None:
    # A coercible stored value ("false") behaves exactly as if it had been
    # authored at the constructor: re-validation coerces it to False rather
    # than reading its truthiness.
    forged = NO_ARTIFACT_LIFECYCLE.model_copy(update={"applicable": "false"})
    assert ArtifactDeclaration.model_validate(forged).applicable is False


def test_metadata_preserves_unknown_json_sections_in_canonical_projection() -> None:
    metadata = DeclaredEngineMetadata.model_validate(
        {
            "artifacts": NO_ARTIFACT_LIFECYCLE.model_dump(),
            "future_section": {"nested": [1, "two", None]},
        }
    )
    projected = metadata.canonical_json()
    assert projected["future_section"] == {"nested": [1, "two", None]}
    assert DeclaredEngineMetadata.model_validate(projected).canonical_json() == projected


def test_canonical_json_rejects_a_declaration_that_skipped_validation() -> None:
    # An isinstance check at the call sites proves the class, not the contents.
    # ``model_copy(update=...)`` and ``model_construct`` both leave a section
    # that never met a validator, and pydantic then SERIALIZES it with a
    # warning rather than refusing -- so the malformed declaration reached
    # ``show`` and a 200 from GET /v1/metadata/..., where the server contract
    # promises a scrubbed 500 for an invalid declaration.
    metadata = DeclaredEngineMetadata.model_validate(
        {"artifacts": NO_ARTIFACT_LIFECYCLE.model_dump()}
    )
    forged = metadata.model_copy(update={"artifacts": {"applicable": "not-a-bool"}})
    assert isinstance(forged, DeclaredEngineMetadata)
    with pytest.raises(ValidationError):
        forged.canonical_json()


def test_canonical_json_reports_one_error_type_under_warnings_as_errors() -> None:
    # The intermediate dump of a mistyped section emits a pydantic serializer
    # warning. Callers running with warnings-as-errors must still get the
    # documented ValidationError, not a UserWarning that depends on their
    # warning filter, so the dump suppresses it and the revalidation decides.
    metadata = DeclaredEngineMetadata.model_validate(
        {"artifacts": NO_ARTIFACT_LIFECYCLE.model_dump()}
    )
    forged = metadata.model_copy(update={"artifacts": {"applicable": "not-a-bool"}})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValidationError):
            forged.canonical_json()


def test_metadata_rejects_python_only_extra_value() -> None:
    with pytest.raises(ValidationError):
        DeclaredEngineMetadata.model_validate(
            {
                "artifacts": NO_ARTIFACT_LIFECYCLE,
                "future_section": object(),
            }
        )


def test_metadata_rejects_nonfinite_extra_value() -> None:
    with pytest.raises(ValidationError):
        DeclaredEngineMetadata.model_validate(
            {
                "artifacts": NO_ARTIFACT_LIFECYCLE,
                "future_section": {"ratio": math.inf},
            }
        )


def test_metadata_rejects_non_string_extra_key_at_any_depth() -> None:
    with pytest.raises(ValidationError, match="JSON object keys must be strings"):
        DeclaredEngineMetadata.model_validate(
            {
                "artifacts": NO_ARTIFACT_LIFECYCLE.model_dump(),
                "future_section": {1: "not-json"},
            }
        )


def test_metadata_rejects_non_mapping_input() -> None:
    with pytest.raises(ValidationError):
        DeclaredEngineMetadata.model_validate("not-a-mapping")
