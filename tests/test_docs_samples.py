# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Executable checks for the code samples in the engine-author guide.

The guide's "Minimal batch engine" block is the first code an engine author
copies. A sample that does not run is the most expensive documentation defect
we can ship, and prose review cannot catch it -- only execution can. These
tests extract the sample from the Markdown and run it, so the guide cannot
drift away from the library without a red test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_GUIDE = Path(__file__).resolve().parent.parent / "docs" / "for_asr_dev" / "adapting_engine.md"


def _first_python_block(markdown: str) -> str:
    """Return the first fenced ``python`` block of a Markdown document.

    Args:
        markdown: The full text of the Markdown file.

    Returns:
        The source inside the first ```python fence.

    Raises:
        AssertionError: If the document contains no ```python fence.
    """
    match = re.search(r"```python\n(.*?)```", markdown, re.DOTALL)
    assert match is not None, "the guide must contain a python sample"
    return match.group(1)


@pytest.fixture
def minimal_engine_namespace() -> dict[str, Any]:
    """Execute the guide's minimal-engine sample and return its namespace.

    Returns:
        The globals the sample defined, including ``MyEngine`` and ``MyConfig``.
    """
    source = _first_python_block(_GUIDE.read_text(encoding="utf-8"))

    # The sample calls a placeholder for the author's own inference code.
    # __name__ lets pydantic resolve the sample's annotations, and
    # dont_inherit keeps this module's `from __future__ import annotations`
    # out of the sample -- an author who copies the block does not get it.
    def my_model_infer(audio: object) -> str:
        return "hello"

    namespace: dict[str, Any] = {
        "__name__": "guide_sample",
        "my_model_infer": my_model_infer,
    }
    exec(  # noqa: S102
        compile(source, str(_GUIDE), "exec", dont_inherit=True),
        namespace,
    )
    return namespace


def test_guide_minimal_engine_transcribes(minimal_engine_namespace: dict[str, Any]) -> None:
    # The sample must survive a real transcribe call. It previously did not:
    # it declared selectable_languages without a default_language, so IC.6
    # made every transcribe raise EngineContractError on the first line.
    engine = minimal_engine_namespace["MyEngine"]()
    result = engine.transcribe((np.zeros(16000, dtype=np.float32), 16000))

    assert result.text == "hello"


def test_guide_minimal_engine_exposes_config_schema(
    minimal_engine_namespace: dict[str, Any],
) -> None:
    # config_type is read off the CLASS so a settings UI can render the schema
    # without constructing the engine. Without it the schema surface goes dark
    # and compliance warns (missing_config_type).
    engine_class = minimal_engine_namespace["MyEngine"]

    assert engine_class.config_type is minimal_engine_namespace["MyConfig"]
    assert "properties" in engine_class.config_type.model_json_schema()


def test_guide_minimal_engine_declares_a_usable_default_language(
    minimal_engine_namespace: dict[str, Any],
) -> None:
    # IC.6: an engine that declares a language axis must supply a default that
    # is actually selectable, otherwise the axis names languages nobody can get.
    config = minimal_engine_namespace["MyConfig"]()
    properties = minimal_engine_namespace["MyProps"]()

    assert config.default_language in properties.selectable_languages
