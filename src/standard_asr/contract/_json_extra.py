# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Internal base for forward-compatible JSON declaration models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, model_validator

from standard_asr.contract.results import require_json_string_keys

#: The JSON value space for unknown fields on typed declaration models.
#:
#: A declaration is visible through both Python and JSON. Validate unknown
#: values when the declaration is constructed so an unencodable value cannot
#: survive until a metadata endpoint tries to serialize it. The adapter owns
#: ``allow_inf_nan=False`` because model config does not propagate into a
#: standalone ``TypeAdapter``.
#:
#: A typed ``__pydantic_extra__`` annotation is not used here. Pydantic 2.5
#: builds typed extras only from an eagerly evaluated annotation, while this
#: project uses deferred annotations. The explicit adapter keeps the same
#: behavior on the supported pydantic floor and on later releases.
_EXTRA_VALUE_ADAPTER = TypeAdapter(dict[str, JsonValue], config=ConfigDict(allow_inf_nan=False))


class JsonExtraModel(BaseModel):
    """Accept unknown string keys while restricting their values to JSON.

    The base supports additive declaration fields without accepting Python-only
    values. Unknown keys survive parsing and serialization. Each key must be an
    exact ``str`` at every nesting level, and each value must fit the JSON value
    space. Subclasses can add subject-specific key checks by overriding
    :meth:`_validate_extra_keys`.
    """

    model_config = ConfigDict(frozen=True, extra="allow", allow_inf_nan=False)

    @classmethod
    def _validate_extra_keys(cls, extras: Mapping[str, object]) -> None:
        """Validate subject-specific constraints on unknown keys.

        Args:
            extras: Unknown fields whose keys are exact strings and whose nested
                key domains have already been checked.

        Returns:
            None.
        """

    @model_validator(mode="before")
    @classmethod
    def _extras_are_json_values(cls, data: Any) -> Any:
        """Canonicalize every unknown field into the JSON value space.

        Args:
            data: The raw constructor input.

        Returns:
            The input with unknown values replaced by their validated JSON
            representations.
        """
        if not isinstance(data, Mapping):
            return data
        mapping = cast("Mapping[Any, Any]", data)
        declared = cls.model_fields
        extras: dict[Any, Any] = {
            key: value for key, value in mapping.items() if key not in declared
        }
        if not extras:
            return cast("Any", data)

        # Check the key domain before pydantic can coerce a bytes key to its
        # string spelling and collide with a declared or extension field.
        require_json_string_keys(extras)
        exact_string_extras = cast("dict[str, object]", extras)
        cls._validate_extra_keys(exact_string_extras)
        validated = _EXTRA_VALUE_ADAPTER.validate_python(exact_string_extras)
        merged = dict(mapping)
        merged.update(validated)
        return merged


__all__: list[str] = []
