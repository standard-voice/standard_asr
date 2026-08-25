# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Counterexample tests for the API-reference generator (docs/site/scripts).

Each test pins a defect class the docs site shipped at least once: lost
constructor signatures, invalid variadic defaults, dropped positional-only
markers, hidden dunder protocol methods, quoted forward references,
cross-references that silently fail to resolve, declared exports that
silently vanish from the reference, and schema fields hidden for lack of
prose or shown without their defaults.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

griffe = pytest.importorskip("griffe")

_SCRIPT = Path(__file__).resolve().parent.parent / "docs" / "site" / "scripts" / "dump_api.py"
_spec = importlib.util.spec_from_file_location("dump_api", _SCRIPT)
assert _spec is not None and _spec.loader is not None
dump_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dump_api)

_FIXTURE = '''
"""Fixture module; see :class:`Widget`."""

__all__ = ["Carrier", "Gadget", "Model", "Plain", "Widget", "clamp", "combine"]


class Widget:
    """A widget.

    Attributes:
        size: The size.
    """

    size: int
    flags: int

    def __init__(self, size: int = 1) -> None:
        self.size = size

    def resize(self, size: int) -> None:
        """Resize; see :meth:`resize` and :class:`Plain`."""

    def __enter__(self) -> "Widget":
        """Enter the widget context."""
        return self

    def _hidden(self) -> None:
        """Private helper."""

    @classmethod
    def create(cls, size: int) -> "Widget":
        """Build a widget."""
        raise NotImplementedError

    @staticmethod
    def unit() -> int:
        """Return the unit size."""
        raise NotImplementedError

    @property
    def area(self) -> int:
        """The widget's area."""
        return self.size * self.size


class Plain:
    """Documented in the class docstring only.

    Args:
        name: The name.
    """

    def __init__(self, name: str) -> None:
        self.name = name


class Gadget:
    """A gadget."""

    def __init__(self, level: int) -> None:
        """Build a gadget.

        Args:
            level: The level.
        """
        self.level = level


class Carrier:
    """A record documented through its constructor.

    Args:
        payload: The payload.
    """

    payload: str

    def __init__(self, payload: str) -> None:
        self.payload = payload


class Model:
    """A schema model.

    Attributes:
        code: The code.
    """

    code: str = Field(..., description="Machine code.")
    timeout: float | None = Field(default=LIMIT, gt=0.0)
    items: list[str] = Field(default_factory=list)
    mode: str = "fast"
    plain: int


def combine(first: int, /, *args: int, **kwargs: int) -> "Widget":
    """Combine; see :attr:`Widget.size` and :func:`missing_thing`."""
    raise NotImplementedError


def clamp(value: int, *, limit: int = 10) -> int:
    """Clamp a value."""
    raise NotImplementedError
'''


@pytest.fixture(scope="module")
def record(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Serialize the fixture module once for the whole test module."""
    root = tmp_path_factory.mktemp("fixture")
    (root / "fixmod.py").write_text(_FIXTURE, encoding="utf-8")
    pkg = griffe.load(
        "fixmod",
        search_paths=[root],
        docstring_parser=griffe.Parser.google,
    )
    assert isinstance(pkg, griffe.Module)
    module = dump_api._serialize_module(pkg)
    symbols = dump_api._build_symbol_index([module], pkg)
    unresolved = dump_api._link_xrefs([module], symbols)
    return {"module": module, "symbols": symbols, "unresolved": unresolved}


def _member(record: dict[str, Any], name: str) -> dict[str, Any]:
    members = {member["name"]: member for member in record["module"]["members"]}
    return members[name]


def test_init_without_docstring_keeps_constructor_signature(record: dict[str, Any]) -> None:
    """A class documenting its params in the class docstring keeps its signature."""
    plain = _member(record, "Plain")
    assert [param["name"] for param in plain["parameters"]] == ["name"]


def test_documented_init_sections_reach_the_class_record(record: dict[str, Any]) -> None:
    """Args documented on __init__ surface in the class docstring sections."""
    gadget = _member(record, "Gadget")
    parameter_items = [
        item
        for section in gadget["docstring"]
        if section["kind"] == "parameters"
        for item in section["items"]
    ]
    assert [item["name"] for item in parameter_items] == ["level"]


def test_variadic_parameters_carry_no_synthetic_default(record: dict[str, Any]) -> None:
    """The synthetic () / {} defaults on *args / **kwargs are not published."""
    combine = _member(record, "combine")
    by_name = {param["name"]: param for param in combine["parameters"]}
    assert "default" not in by_name["args"]
    assert "default" not in by_name["kwargs"]


def test_positional_only_kind_is_published(record: dict[str, Any]) -> None:
    """The renderer needs the kind to emit the / marker."""
    combine = _member(record, "combine")
    by_name = {param["name"]: param for param in combine["parameters"]}
    assert by_name["first"]["kind"] == "positional-only"


def test_quoted_forward_reference_loses_its_quotes(record: dict[str, Any]) -> None:
    """A quoted return annotation publishes the name, not the literal."""
    combine = _member(record, "combine")
    text = "".join(token["text"] for token in combine["returns"])
    assert text == "Widget"


def test_documented_dunder_is_published_and_private_is_not(record: dict[str, Any]) -> None:
    """__enter__ with a docstring is public protocol surface; _hidden is not."""
    widget = _member(record, "Widget")
    names = [method["name"] for method in widget["methods"]]
    assert "__enter__" in names
    assert "_hidden" not in names
    assert "__init__" not in names


def test_relative_xrefs_resolve_against_their_scope(record: dict[str, Any]) -> None:
    """Bare, class-relative, and attribute targets resolve; unknowns stay put."""
    import json

    text = json.dumps(record["module"])
    assert "(xref:fixmod.Widget)" in text  # module docstring, page scope
    assert "(xref:fixmod.Widget.resize)" in text  # self-reference, class scope
    assert "(xref:fixmod.Plain)" in text  # sibling class from method scope
    assert "(xref:fixmod.Widget.size)" in text  # attribute target
    assert "(xref:missing_thing)" in text  # unknown: left as written
    assert record["unresolved"] == {"missing_thing": 1}


def test_annotated_field_publishes_without_prose(record: dict[str, Any]) -> None:
    """A public annotated field is contract even without a description.

    ``StreamDeadlines`` shipped with no ``Attributes:`` section, and the
    prose-gated policy rendered it as a model with no fields at all; the
    field now publishes (and anchors) regardless, so missing prose shows
    as an empty description instead of a missing field.
    """
    widget = _member(record, "Widget")
    by_name = {attr["name"]: attr for attr in widget["attributes"]}
    assert by_name["flags"]["published"] is True
    assert "fixmod.Widget.size" in record["symbols"]
    assert "fixmod.Widget.flags" in record["symbols"]


def test_constructor_documented_field_renders_there_instead(record: dict[str, Any]) -> None:
    """An annotated field the merged constructor table already documents
    (dataclass-style classes) is not republished as an attribute row."""
    carrier = _member(record, "Carrier")
    by_name = {attr["name"]: attr for attr in carrier["attributes"]}
    assert by_name["payload"]["published"] is False
    assert "fixmod.Carrier.payload" not in record["symbols"]


def test_pydantic_field_calls_are_unpacked(record: dict[str, Any]) -> None:
    """A ``Field(...)`` default publishes its facts, not the raw call.

    Required stays default-less (pydantic's own semantics), factories show
    as calls, constraints survive verbatim, and ``description=`` becomes
    the fallback prose.
    """
    model = _member(record, "Model")
    by_name = {attr["name"]: attr for attr in model["attributes"]}
    assert by_name["code"]["default"] is None
    assert by_name["code"]["field_description"] == "Machine code."
    assert by_name["timeout"]["default"] == "LIMIT"
    assert by_name["timeout"]["constraints"] == ["gt=0.0"]
    assert by_name["items"]["default"] == "list()"
    assert by_name["mode"]["default"] == "'fast'"
    assert by_name["plain"]["default"] is None
    assert all(by_name[name]["published"] for name in by_name)


def test_binding_labels_are_published(record: dict[str, Any]) -> None:
    """The classmethod / staticmethod binding survives into the record."""
    widget = _member(record, "Widget")
    labels = {method["name"]: method["labels"] for method in widget["methods"]}
    assert "classmethod" in labels["create"]
    assert "staticmethod" in labels["unit"]


def test_property_serializes_as_an_attribute(record: dict[str, Any]) -> None:
    """griffe models @property as an attribute; a callable projection of it
    (``Segment.timestamp_status()``) would publish wrong usage, so this pins
    the group it lands in."""
    widget = _member(record, "Widget")
    assert "area" not in [method["name"] for method in widget["methods"]]
    by_name = {attr["name"]: attr for attr in widget["attributes"]}
    assert "property" in by_name["area"]["labels"]
    assert by_name["area"]["published"] is True


def test_phantom_all_name_fails_the_build(tmp_path: Path) -> None:
    """A name declared in ``__all__`` with no member behind it must not be
    dropped from the reference under a green build."""
    (tmp_path / "phantommod.py").write_text(
        '"""Module."""\n\n__all__ = ["ghost"]\n',
        encoding="utf-8",
    )
    pkg = griffe.load("phantommod", search_paths=[tmp_path], docstring_parser=griffe.Parser.google)
    assert isinstance(pkg, griffe.Module)
    with pytest.raises(RuntimeError, match="declares ghost"):
        dump_api._serialize_module(pkg)


def test_unresolvable_export_fails_the_build(tmp_path: Path) -> None:
    """A re-export whose target griffe cannot resolve must not be dropped
    from the reference under a green build."""
    (tmp_path / "brokenmod.py").write_text(
        '"""Module."""\n\nfrom missing_dependency import thing\n\n__all__ = ["thing"]\n',
        encoding="utf-8",
    )
    pkg = griffe.load("brokenmod", search_paths=[tmp_path], docstring_parser=griffe.Parser.google)
    assert isinstance(pkg, griffe.Module)
    with pytest.raises(RuntimeError, match="brokenmod.thing does not resolve"):
        dump_api._serialize_module(pkg)


_FACADE_PKG = {
    "__init__.py": '''
"""Facade page; see :meth:`fixpkg.impl.Widget.resize` and :class:`Thing`."""

from fixpkg.impl import Widget

__all__ = ["Widget"]
''',
    "impl.py": '''
"""Concrete module without a documentation page."""


class Widget:
    """A widget."""

    def resize(self, size: int) -> None:
        """Resize the widget."""
''',
    "alt.py": '''
"""Second facade page."""

from fixpkg.impl import Widget

__all__ = ["Thing", "Widget"]


class Thing:
    """One of two distinct classes named Thing."""
''',
    "tools.py": '''
"""Tools page; see :class:`Widget`."""

__all__ = ["Thing"]


class Thing:
    """The other class named Thing."""
''',
}


@pytest.fixture(scope="module")
def facade(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Serialize a package whose class is documented through re-exports."""
    root = tmp_path_factory.mktemp("facade")
    pkg_dir = root / "fixpkg"
    pkg_dir.mkdir()
    for filename, content in _FACADE_PKG.items():
        (pkg_dir / filename).write_text(content, encoding="utf-8")
    pkg = griffe.load(
        "fixpkg",
        search_paths=[root],
        docstring_parser=griffe.Parser.google,
    )
    assert isinstance(pkg, griffe.Module)
    modules = [
        dump_api._serialize_module(dump_api._module_page(pkg, dotted))
        for dotted in ("fixpkg", "fixpkg.alt", "fixpkg.tools")
    ]
    symbols = dump_api._build_symbol_index(modules, pkg)
    unresolved = dump_api._link_xrefs(modules, symbols)
    return {"modules": modules, "symbols": symbols, "unresolved": unresolved}


def test_canonical_child_paths_resolve(facade: dict[str, Any]) -> None:
    """A method cited by its defining-module path links to the facade page.

    ``standard_asr.runtime.config.BaseConfig.from_env`` in a docstring must
    reach ``BaseConfig.from_env`` on the page that documents the re-export.
    """
    import json

    entry = facade["symbols"]["fixpkg.impl.Widget.resize"]
    assert entry["page"] == "fixpkg"
    assert entry["anchor"] == "Widget.resize"
    text = json.dumps(facade["modules"][0]["docstring"])
    assert "(xref:fixpkg.impl.Widget.resize)" in text
    assert "fixpkg.impl.Widget.resize" not in facade["unresolved"]


def test_bare_name_resolves_when_every_candidate_is_one_symbol(facade: dict[str, Any]) -> None:
    """``Widget`` is documented on two pages, but both are the same class,
    so a bare reference links to the highest-priority page instead of
    failing as ambiguous. Two distinct classes named ``Thing`` stay
    ambiguous and unresolved."""
    import json

    tools_text = json.dumps(facade["modules"][2]["docstring"])
    assert "(xref:fixpkg.Widget)" in tools_text
    facade_text = json.dumps(facade["modules"][0]["docstring"])
    assert "(xref:Thing)" in facade_text
    assert facade["unresolved"] == {"Thing": 1}
