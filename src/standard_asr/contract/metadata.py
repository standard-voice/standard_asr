# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Static, typed metadata declared by an engine class."""

from __future__ import annotations

from typing import Final, cast

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

from standard_asr.contract._json_extra import JsonExtraModel


class ArtifactDeclaration(BaseModel):
    """Declare the artifact-lifecycle behavior available across a model preset.

    Each field is an upper bound over every supported configuration and request
    context. The configured engine reports narrower effective behavior through
    its artifact status report.

    Attributes:
        applicable: Whether any supported context uses separately supplied
            persistent artifacts inside the engine lifecycle. It does NOT
            promise an engine-owned acquisition path: an engine whose artifacts
            an operator or the operating system supplies declares it true with
            both acquisition fields false.
        supports_explicit_acquisition: Whether any supported context permits an
            explicit artifact acquisition before inference.
        may_acquire_during_inference: Whether normal inference can acquire
            artifacts in any supported context.

    Raises:
        ValueError: If the lifecycle is not applicable but an acquisition
            path is declared.
    """

    # ``revalidate_instances`` for the reason given on the artifact models
    # (``ArtifactAction``): pydantic accepts an existing instance verbatim at
    # every validation boundary, so a ``model_copy(update=...)`` declaration
    # whose fields never met a validator -- the obvious way to derive a
    # variant of a template declaration -- would cross ``model_validate`` and
    # nested construction unchallenged. Re-validating applies the model's own
    # construction semantics to the stored values wherever the declaration is
    # validated again (the runtime template does so before gating on it).
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    applicable: bool
    supports_explicit_acquisition: bool
    may_acquire_during_inference: bool

    @model_validator(mode="after")
    def _acquisition_paths_require_applicability(self) -> ArtifactDeclaration:
        """Reject acquisition behavior without an applicable lifecycle.

        Returns:
            The validated declaration.

        Raises:
            ValueError: If either acquisition path is declared while the
                artifact lifecycle is not applicable.
        """
        if not self.applicable and (
            self.supports_explicit_acquisition or self.may_acquire_during_inference
        ):
            raise ValueError(
                "supports_explicit_acquisition and may_acquire_during_inference "
                "must be false when applicable is false."
            )
        return self


NO_ARTIFACT_LIFECYCLE: Final = ArtifactDeclaration(
    applicable=False,
    supports_explicit_acquisition=False,
    may_acquire_during_inference=False,
)
"""Authored declaration for a model with no inference-artifact lifecycle."""


class DeclaredEngineMetadata(JsonExtraModel):
    """Structured static metadata declared by an engine class.

    Known sections are typed and required according to their protocol version.
    Unknown string keys with JSON values survive parsing and serialization so a
    later protocol minor can add a sibling section without breaking an older
    reader.

    Attributes:
        artifacts: Static artifact-lifecycle declaration for the model preset.
    """

    artifacts: ArtifactDeclaration

    def canonical_json(self) -> dict[str, JsonValue]:
        """Return the canonical JSON projection of the metadata.

        The value is re-validated first. An ``isinstance`` check at the call
        sites proves the class, not the contents: ``model_copy(update=...)``
        and ``model_construct`` both build an instance whose ``artifacts``
        section never met a validator, and pydantic then serializes it with a
        warning rather than an error. Projecting that unchallenged makes
        ``show`` print a malformed declaration and ``GET /v1/metadata/...``
        answer 200, where the server contract promises a scrubbed 500 for an
        invalid declaration. Re-validating makes that promise the code path.

        Returns:
            A JSON-ready mapping that preserves unknown metadata sections.

        Raises:
            ValidationError: If the declaration does not satisfy its own
                contract. Callers own the projection: the metadata endpoint's
                fault boundary safe-logs it and answers a scrubbed 500, and the
                CLI reports it as an engine fault.
        """
        # ``warnings=False`` on the intermediate dump only: serializing a
        # section that is not its declared type emits a pydantic serializer
        # warning, and a caller running with warnings-as-errors would get that
        # warning instead of the ValidationError this method promises. The
        # warning carries no information the re-validation below does not, and
        # suppressing it per call keeps the failure one deterministic type
        # without touching global warning state from a threaded server.
        validated = type(self).model_validate(self.model_dump(mode="python", warnings=False))
        return cast("dict[str, JsonValue]", validated.model_dump(mode="json"))


__all__ = [
    "ArtifactDeclaration",
    "DeclaredEngineMetadata",
    "NO_ARTIFACT_LIFECYCLE",
]
