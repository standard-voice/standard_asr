# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Executable checks for the code samples in the engine-author guide.

The guide's "Minimal batch engine" block is the first code an engine author
copies. A sample that does not run is the most expensive documentation defect
this repo can ship, and prose review cannot catch it -- only execution can. These
tests extract the sample from the Markdown and run it, so the guide cannot
drift away from the library without a red test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_GUIDE = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "content"
    / "engine-authors"
    / "adapt-an-asr-system.md"
)


def _python_block_under_heading(markdown: str, heading: str) -> str:
    """Return the first fenced ``python`` block after a Markdown heading.

    Anchoring on the heading instead of document position keeps the tests
    aimed at the intended sample even when an unrelated fence is added
    earlier in the document.

    Args:
        markdown: The full text of the Markdown file.
        heading: The exact heading line, for example ``"## Minimal batch engine"``.

    Returns:
        The source inside the first ```python fence after ``heading``.

    Raises:
        AssertionError: If the heading or its ```python fence is missing.
    """
    start = markdown.find(heading)
    assert start != -1, f"the guide must contain the heading {heading!r}"
    match = re.search(r"```python\n(.*?)```", markdown[start:], re.DOTALL)
    assert match is not None, f"no python sample under {heading!r}"
    return match.group(1)


@pytest.fixture
def minimal_engine_namespace() -> dict[str, Any]:
    """Execute the guide's minimal-engine sample and return its namespace.

    Returns:
        The globals the sample defined, including ``MyEngine`` and ``MyConfig``.
    """
    source = _python_block_under_heading(
        _GUIDE.read_text(encoding="utf-8"), "## Minimal batch engine"
    )

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


def test_guide_minimal_engine_survives_auto_language(
    minimal_engine_namespace: dict[str, Any],
) -> None:
    # The sample declares "auto" selectable, so language="auto" is a legal
    # call. The sample previously echoed params.language into
    # detected_language, and the reserved "auto" is rejected there -- every
    # auto-detect request crashed as a misreported engine fault.
    from standard_asr import RuntimeParams

    engine = minimal_engine_namespace["MyEngine"]()
    result = engine.transcribe(
        (np.zeros(16000, dtype=np.float32), 16000), RuntimeParams(language="auto")
    )

    assert result.text == "hello"
    assert result.detected_language != "auto"


def test_guide_minimal_engine_rejects_unknown_init_option(
    minimal_engine_namespace: dict[str, Any],
) -> None:
    # The sample previously accepted **kw and discarded it, so a mistyped
    # option ran the engine on defaults -- a silent wrong result. The config
    # is extra="forbid", so forwarding the kwargs makes the typo fail loudly.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        minimal_engine_namespace["MyEngine"](bogus_option=1)


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


def test_guide_minimal_engine_pins_its_protocol_version() -> None:
    # AR.1: a plugin declares the protocol version it implements, and bumps it
    # only after it fully implements the newer contract. The sample previously
    # assigned CURRENT_PROTOCOL_VERSION -- the version of the INSTALLED core --
    # so a plugin copied from it would silently re-declare whatever protocol a
    # later core implements, a claim consumers trust before member lookup.
    source = _python_block_under_heading(
        _GUIDE.read_text(encoding="utf-8"), "## Minimal batch engine"
    )

    assert "CURRENT_PROTOCOL_VERSION" not in source
    assert 'protocol_version: str = "1.1.0"' in source
