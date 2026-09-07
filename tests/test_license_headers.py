# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Every Python file carries the SPDX license header."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
#: The directories that hold this project's own Python files, the working
#: notes under ``docs/internal`` included. Anything else under the checkout,
#: such as a virtual environment or ``node_modules``, belongs to a tool, not
#: to the project.
_SOURCE_ROOTS = ("src", "tests", "docs/site/scripts", "scripts", "docs/internal")
_HEADER = (
    "# SPDX-FileCopyrightText: 2026 Standard Voice Contributors\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)


def test_every_python_file_has_the_spdx_header() -> None:
    missing = sorted(
        str(path.relative_to(_ROOT))
        for root in _SOURCE_ROOTS
        for path in (_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
        and not path.read_text(encoding="utf-8").startswith(_HEADER)
    )
    assert missing == [], f"Python files without the SPDX header: {missing}"
