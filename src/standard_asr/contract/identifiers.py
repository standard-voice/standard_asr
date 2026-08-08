# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Syntactic validation for engine and model identifiers.

An entry-point name has the shape ``<engine_id>/<model_name>``. These two
segments are protocol-level identity strings: the properties model declares
them as fields, and plugin discovery parses them out of entry-point names.
Both need the same surface-syntax check, so the validators live here in the
contract layer -- the lowest layer -- and both consumers depend downward on it.

The checks are purely syntactic. Canonicalization of an engine id to its
routing identity (PEP 503 normalization) is a separate, discovery-layer step.
"""

from __future__ import annotations

import logging
import re

from standard_asr.contract.exceptions import EntrypointValidationError

logger = logging.getLogger(__name__)

_ENGINE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*\Z")
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+%:-]*\Z")


def validate_engine_id(engine_id: str) -> None:
    """Validate the *declared* form of an engine identifier.

    This checks the surface syntax only. Canonicalization to the PEP 503
    routing identity is performed by :func:`parse_entrypoint_name`; a
    non-canonical-but-valid id such as ``my_engine`` passes here and is folded
    to its canonical ``my-engine`` form downstream.

    Args:
        engine_id: Engine identifier string.

    Returns:
        None.

    Raises:
        EntrypointValidationError: If the engine identifier is invalid.
    """
    if "/" in engine_id:
        raise EntrypointValidationError(f"engine_id must not contain '/' (got {engine_id!r})")
    if not _ENGINE_ID_RE.match(engine_id):
        raise EntrypointValidationError(
            "engine_id contains unsupported characters. Allowed: lowercase ASCII "
            "letters, digits, '.', '_' and '-'."
        )


def validate_model_name(model_name: str) -> None:
    """Validate and log guidance for a model name.

    Args:
        model_name: Model name string (may be empty for defaults).

    Returns:
        None.

    Raises:
        EntrypointValidationError: If the model name is invalid.
    """
    if model_name == "":
        logger.warning(
            "model_name is empty for a standard_asr.models entry point. "
            "Empty names are allowed but discouraged; document the default clearly."
        )
        return
    if "/" in model_name:
        raise EntrypointValidationError(f"model_name must not contain '/' (got {model_name!r})")
    if not _MODEL_NAME_RE.match(model_name):
        raise EntrypointValidationError(
            "model_name contains unsupported characters. Allowed characters: "
            "letters, digits, '.', '_', '+', '%', ':', '-'."
        )
