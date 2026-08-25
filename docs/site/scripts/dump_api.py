# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Dump the public API surface of ``standard_asr`` to neutral JSON.

The docs site renders its API reference from this data. Griffe reads the
source tree; this script selects the public surface (the modules below and
their exported members) and writes structured, renderer-neutral data:
signatures as token lists, docstring sections as parsed Google-style
blocks, and source locations for view-source links. It never renders
HTML or Markdown; presentation belongs to the site renderer.

Run from ``docs/site``::

    pnpm generate:api
"""

from __future__ import annotations

import ast
import json
import re
import sys
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import Any, cast

import griffe

# The documented module pages, in cross-reference priority order: the
# first page that documents a symbol wins the target, so concrete modules
# come before the facades and a reference resolves to the symbol's home.
# (The site renders pages by path; this order never reaches navigation.)
MODULE_PAGES = [
    "standard_asr.contract.results",
    "standard_asr.contract.capabilities",
    "standard_asr.contract.exceptions",
    "standard_asr.runtime.streaming",
    "standard_asr.audio.wire",
    "standard_asr.compliance",
    "standard_asr.engine",
    "standard_asr",
]

# Sphinx-style roles that appear in docstrings. Each becomes a neutral
# ``xref:`` Markdown link the renderer resolves against the symbol index.
_ROLE_RE = re.compile(r":(?:class|func|meth|mod|data|attr|exc):`(~?)([A-Za-z0-9_.]+)`")

# An reStructuredText literal block: a line ending in ``::``, a blank line,
# then an indented run. Markdown renders the indented run as code already;
# this converts the marker and the block into an explicit fence.
_LITERAL_BLOCK_RE = re.compile(r"::\n\n((?:(?:    .*)?\n)+(?:    .*)?)", re.MULTILINE)


def _roles_to_xrefs(text: str) -> str:
    """Rewrite Sphinx markup in ``text`` into neutral Markdown.

    Roles become ``xref:`` links the renderer resolves against the symbol
    index; ``::`` literal blocks become fenced code blocks.

    Args:
        text: Docstring Markdown that may contain ``:class:`` -style roles.

    Returns:
        The rewritten text.
    """

    def repl(match: re.Match[str]) -> str:
        shorten, path = match.groups()
        label = path.rsplit(".", 1)[-1] if shorten else path
        return f"[`{label}`](xref:{path})"

    def fence(match: re.Match[str]) -> str:
        body = match.group(1)
        lines = [line[4:] if line.startswith("    ") else line for line in body.split("\n")]
        code = "\n".join(lines).strip("\n")
        return f":\n\n```python\n{code}\n```\n"

    return _LITERAL_BLOCK_RE.sub(fence, _ROLE_RE.sub(repl, text))


def _annotation_tokens(
    annotation: str | griffe.Expr | None,
) -> list[dict[str, str]] | None:
    """Flatten a griffe annotation expression into linkable tokens.

    Args:
        annotation: A griffe expression, a plain string, or ``None``.

    Returns:
        Token dicts with ``text`` and, for resolvable names, ``target``
        (the canonical dotted path); ``None`` when there is no annotation.
    """
    if annotation is None:
        return None
    if isinstance(annotation, str):
        # A stringized forward reference keeps its quotes in the source;
        # the published annotation is the name, not the string literal.
        stripped = annotation.strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "'\"":
            stripped = stripped[1:-1]
        return [{"text": stripped}]
    tokens: list[dict[str, str]] = []
    for part in annotation.iterate(flat=True):
        if isinstance(part, griffe.ExprName):
            tokens.append({"text": part.name, "target": part.canonical_path})
        else:
            text = str(part)
            if tokens and "target" not in tokens[-1]:
                tokens[-1]["text"] += text
            else:
                tokens.append({"text": text})
    return tokens


def _docstring_sections(obj: griffe.Object) -> list[dict[str, Any]]:
    """Serialize an object's parsed docstring into neutral sections.

    Args:
        obj: Any griffe object whose docstring was parsed as Google style.

    Returns:
        A list of section dicts; empty when the object has no docstring.
    """
    if obj.docstring is None:
        return []
    sections: list[dict[str, Any]] = []
    for section in obj.docstring.parsed:
        kind = section.kind.value
        if kind == "text":
            sections.append({"kind": "text", "value": _roles_to_xrefs(section.value)})
        elif kind in {"parameters", "other parameters", "attributes"}:
            sections.append(
                {
                    "kind": "attributes" if kind == "attributes" else "parameters",
                    "items": [
                        {
                            "name": item.name,
                            "annotation": _annotation_tokens(item.annotation),
                            "description": _roles_to_xrefs(item.description),
                            **(
                                {"default": str(item.value)}
                                if getattr(item, "value", None) is not None
                                else {}
                            ),
                        }
                        for item in section.value
                    ],
                }
            )
        elif kind in {"returns", "yields", "receives"}:
            sections.append(
                {
                    "kind": kind,
                    "items": [
                        {
                            "name": item.name or None,
                            "annotation": _annotation_tokens(item.annotation),
                            "description": _roles_to_xrefs(item.description),
                        }
                        for item in section.value
                    ],
                }
            )
        elif kind in {"raises", "warns"}:
            sections.append(
                {
                    "kind": kind,
                    "items": [
                        {
                            "annotation": _annotation_tokens(item.annotation),
                            "description": _roles_to_xrefs(item.description),
                        }
                        for item in section.value
                    ],
                }
            )
        elif kind == "examples":
            parts = [
                text if part_kind.value == "text" else f"```python\n{text}\n```"
                for part_kind, text in section.value
            ]
            sections.append({"kind": "examples", "value": "\n\n".join(parts)})
        elif kind == "admonition":
            description = section.value.description
            # A doctest body (Google's singular "Example:") is code, not
            # prose: fence it, or the renderer parses ">>>" as a blockquote.
            if description.lstrip().startswith(">>>"):
                value = f"```python\n{description}\n```"
            else:
                value = _roles_to_xrefs(description)
            sections.append(
                {
                    "kind": "admonition",
                    "type": section.title or "note",
                    "value": value,
                }
            )
        elif kind == "deprecated":
            sections.append({"kind": "deprecated", "value": _roles_to_xrefs(section.value)})
        else:
            sections.append({"kind": "text", "value": _roles_to_xrefs(str(section.value))})
    return sections


def _source_location(obj: griffe.Object) -> dict[str, Any] | None:
    """Return the source location for a view-source link, when known.

    Args:
        obj: A resolved (non-alias) griffe object.

    Returns:
        A dict with ``file`` (repo-relative path) and ``line``, or ``None``.
    """
    try:
        filepath = obj.relative_package_filepath
    except ValueError:
        return None
    if obj.lineno is None:
        return None
    return {"file": f"src/{filepath}", "line": obj.lineno}


def _parameters(func: griffe.Function) -> list[dict[str, Any]]:
    """Serialize a function's parameters.

    Args:
        func: The griffe function.

    Returns:
        Parameter dicts in declaration order, ``self`` and ``cls`` excluded.
    """
    params: list[dict[str, Any]] = []
    for param in func.parameters:
        if param.name in {"self", "cls"}:
            continue
        kind = param.kind.value if param.kind else "positional or keyword"
        # griffe synthesizes "()" / "{}" defaults for *args / **kwargs;
        # rendering them would publish invalid Python.
        has_default = param.default is not None and not kind.startswith("variadic")
        params.append(
            {
                "name": param.name,
                "kind": kind,
                "annotation": _annotation_tokens(param.annotation),
                **({"default": str(param.default)} if has_default else {}),
            }
        )
    return params


def _serialize_function(func: griffe.Function, name: str) -> dict[str, Any]:
    """Serialize a function or method.

    Args:
        func: The resolved griffe function.
        name: The name to publish (the alias name on a facade page).

    Returns:
        The neutral function record.
    """
    return {
        "kind": "function",
        "name": name,
        "async": "async" in func.labels,
        "labels": sorted(func.labels),
        "parameters": _parameters(func),
        "returns": _annotation_tokens(func.returns),
        "docstring": _docstring_sections(func),
        "source": _source_location(func),
    }


def _attribute_value_facts(value: str | griffe.Expr | None) -> dict[str, Any]:
    """Unpack an attribute's assigned value into display facts.

    A top-level pydantic ``Field(...)`` call is opened up: ``default`` or
    ``default_factory`` becomes the shown default (absent for a required
    field, matching pydantic's own semantics), ``description`` is kept as
    fallback prose, and every other keyword (``gt``, ``ge``, ...) is
    preserved verbatim as a constraint. Any other value passes through
    unchanged as the default, so no assignment is ever hidden.

    Args:
        value: The attribute's griffe value expression, if any.

    Returns:
        A dict with ``default`` (``str | None``), ``constraints``
        (``list[str]``), and ``field_description`` (``str | None``).
    """
    facts: dict[str, Any] = {"default": None, "constraints": [], "field_description": None}
    if value is None:
        return facts
    if not (
        isinstance(value, griffe.ExprCall)
        and str(value.function).rsplit(".", maxsplit=1)[-1] == "Field"
    ):
        facts["default"] = str(value)
        return facts
    for argument in value.arguments:
        if isinstance(argument, griffe.ExprKeyword):
            name, text = argument.name, str(argument.value)
        else:
            # Field's only positional parameter is the default.
            name, text = "default", str(argument)
        if name == "default":
            # A literal ``...`` default is pydantic's required marker.
            facts["default"] = None if text == "..." else text
        elif name == "default_factory":
            facts["default"] = f"{text}()"
        elif name == "description":
            try:
                facts["field_description"] = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                # Not a literal (a name or a call): keep the source text.
                facts["field_description"] = text
        else:
            facts["constraints"].append(f"{name}={text}")
    return facts


def _serialize_attribute(attr: griffe.Attribute, name: str) -> dict[str, Any]:
    """Serialize a module attribute, class field, or enum member.

    Args:
        attr: The resolved griffe attribute.
        name: The name to publish.

    Returns:
        The neutral attribute record.
    """
    return {
        "kind": "attribute",
        "name": name,
        "annotation": _annotation_tokens(attr.annotation),
        **_attribute_value_facts(attr.value),
        "labels": sorted(attr.labels),
        "docstring": _docstring_sections(attr),
        "source": _source_location(attr),
    }


def _class_members(cls: griffe.Class) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect the public attributes and methods of a class.

    Args:
        cls: The resolved griffe class.

    Returns:
        A pair ``(attributes, methods)``. Attributes keep declaration
        order and include undocumented annotated fields; whether one
        publishes is decided in :func:`_serialize_class`, which sees the
        class docstring and the constructor signature. Methods keep
        declaration order and skip undocumented ones.
    """
    attributes: list[dict[str, Any]] = []
    methods: list[dict[str, Any]] = []
    for name, member in cls.members.items():
        if member.is_alias:
            continue
        # Private members stay private, but a documented dunder is public
        # protocol surface (__aenter__, __iter__, ...): publish it.
        if name.startswith("_") and not (
            name.startswith("__")
            and name.endswith("__")
            and name != "__init__"
            and member.is_function
            and member.docstring is not None
        ):
            continue
        # griffe represents a @property as an Attribute carrying the
        # "property" label, so it lands in the attributes group -- the
        # right one: its rendering is attribute access, not a call.
        if member.is_attribute:
            attributes.append(_serialize_attribute(member, name))  # type: ignore[arg-type]
        elif member.is_function:
            if member.docstring is None:
                continue
            methods.append(_serialize_function(member, name))  # type: ignore[arg-type]
    return attributes, methods


def _serialize_class(cls: griffe.Class, name: str) -> dict[str, Any]:
    """Serialize a class, merging ``__init__`` into the class signature.

    Args:
        cls: The resolved griffe class.
        name: The name to publish.

    Returns:
        The neutral class record.
    """
    attributes, methods = _class_members(cls)
    init = cls.members.get("__init__")
    parameters: list[dict[str, Any]] = []
    init_sections: list[dict[str, Any]] = []
    # Merge __init__ into the class record the way mkdocstrings'
    # merge_init_into_class did: the constructor signature always, and its
    # docstring sections (the D107-sanctioned home for parameter docs is
    # either place) after the class's own.
    if isinstance(init, griffe.Function):
        parameters = _parameters(init)
        init_sections = _docstring_sections(init)
    sections = _docstring_sections(cls) + init_sections
    # A public annotated field always publishes: its name, type, and
    # default are contract regardless of prose, and silently dropping one
    # from the reference is exactly the drift this generator exists to
    # prevent. Two carve-outs keep the pages clean: a field the merged
    # constructor table already documents (dataclass-style classes)
    # renders there instead, and an unannotated attribute still needs a
    # description -- from the class docstring's ``Attributes:`` section
    # (an enum member's home) or its own docstring -- because griffe also
    # reports plumbing like ``model_config`` as attributes.
    described = {
        item["name"]
        for section in sections
        if section["kind"] == "attributes"
        for item in section["items"]
    }
    parameter_names = {parameter["name"] for parameter in parameters}
    for attr in attributes:
        own_text = any(section["kind"] == "text" for section in attr["docstring"])
        attr["published"] = (
            attr["name"] in described
            or own_text
            or (attr["annotation"] is not None and attr["name"] not in parameter_names)
        )
    return {
        "kind": "class",
        "name": name,
        "bases": [_annotation_tokens(base) for base in cls.bases],
        "labels": sorted(cls.labels),
        "parameters": parameters,
        "docstring": sections,
        "attributes": attributes,
        "methods": methods,
        "source": _source_location(cls),
    }


def _module_member_names(module: griffe.Module) -> list[str]:
    """Return the publishable member names of a module, in order.

    Args:
        module: The resolved griffe module.

    Returns:
        ``__all__`` order when the module declares it; otherwise the
        public, locally defined members in source order.

    Raises:
        RuntimeError: If ``__all__`` declares a name griffe cannot see.
            A declared export must publish or fail the build; it never
            silently vanishes from the reference.
    """
    if module.exports is not None:
        exported = [name if isinstance(name, str) else name.name for name in module.exports]
        missing = [name for name in exported if name not in module.members]
        if missing:
            raise RuntimeError(
                f"{module.path}.__all__ declares {', '.join(sorted(missing))}, "
                "which griffe does not see as a member"
            )
        return exported
    names = [
        name
        for name, member in module.members.items()
        if not name.startswith("_") and not member.is_alias and not member.is_module
    ]
    names.sort(key=lambda name: module.members[name].lineno or 0)
    return names


def _module_page(pkg: griffe.Module, dotted: str) -> griffe.Module:
    """Resolve a dotted page path to its griffe module.

    Args:
        pkg: The loaded root package.
        dotted: The page's dotted module path.

    Returns:
        The named module.

    Raises:
        TypeError: If the path names something other than a module.
    """
    module = pkg if dotted == pkg.name else pkg[dotted.removeprefix(f"{pkg.name}.")]
    if not isinstance(module, griffe.Module):
        raise TypeError(f"{dotted} is not a module")
    return module


def _serialize_module(module: griffe.Module) -> dict[str, Any]:
    """Serialize one documented module page.

    Args:
        module: The resolved griffe module.

    Returns:
        The neutral module record with resolved members.

    Raises:
        RuntimeError: If a declared export cannot publish: an alias that
            does not resolve, a target outside the package, or a kind
            this generator does not render. A dropped export would be
            silent doc drift, so the build fails instead.
    """
    members: list[dict[str, Any]] = []
    for name in _module_member_names(module):
        member = module.members[name]
        try:
            target = member.final_target if isinstance(member, griffe.Alias) else member
        except (griffe.AliasResolutionError, KeyError) as error:
            raise RuntimeError(
                f"public export {module.path}.{name} does not resolve: {error}"
            ) from error
        if target.package is not module.package:
            raise RuntimeError(
                f"public export {module.path}.{name} resolves outside the package "
                f"({target.path}); render it explicitly or stop exporting it"
            )
        if isinstance(target, griffe.Class):
            members.append(_serialize_class(target, name))
        elif isinstance(target, griffe.Function):
            members.append(_serialize_function(target, name))
        elif isinstance(target, griffe.Attribute):
            members.append(_serialize_attribute(target, name))
        else:
            raise RuntimeError(
                f"public export {module.path}.{name} is a {target.kind.value}, "
                "which this generator does not render"
            )
    return {
        "path": module.path,
        "docstring": _docstring_sections(module),
        "members": members,
    }


def _build_symbol_index(modules: list[dict[str, Any]], pkg: griffe.Module) -> dict[str, Any]:
    """Map canonical and facade paths to their documentation anchors.

    Args:
        modules: The serialized module records, in ``MODULE_PAGES`` order.
        pkg: The loaded package, for canonical-path lookups.

    Returns:
        ``{dotted_path: {"page": module_path, "anchor": member_name,
        "canonical": canonical_path}}``. The ``canonical`` field lets the
        xref resolver recognize two keys as the same symbol; ``main``
        strips it before writing.
    """
    index: dict[str, Any] = {}
    for record in modules:
        page = record["path"]
        # The page itself is a target too (:mod: references).
        index.setdefault(page, {"page": page, "anchor": None, "canonical": page})
        module = _module_page(pkg, page)
        for member in record["members"]:
            name = member["name"]
            # A member reached through a re-export keeps its canonical
            # (defining-module) path as an index key too, so a docstring
            # can cite either path. Its children follow below.
            # _serialize_module resolved this alias or raised, so
            # final_target cannot fail here.
            canonical = f"{page}.{name}"
            raw = module.members.get(name)
            if isinstance(raw, griffe.Alias):
                canonical = raw.final_target.path
            entry = {"page": page, "anchor": name, "canonical": canonical}
            index.setdefault(f"{page}.{name}", entry)
            index.setdefault(canonical, entry)
            for group in ("methods", "attributes"):
                for sub in member.get(group, []):
                    if group == "attributes" and not sub["published"]:
                        continue
                    sub_entry = {
                        "page": page,
                        "anchor": f"{name}.{sub['name']}",
                        "canonical": f"{canonical}.{sub['name']}",
                    }
                    index.setdefault(f"{page}.{name}.{sub['name']}", sub_entry)
                    index.setdefault(f"{canonical}.{sub['name']}", sub_entry)
    return index


_XREF_RE = re.compile(r"\(xref:([A-Za-z0-9_.]+)\)")


def _link_xrefs(modules: list[dict[str, Any]], symbols: dict[str, Any]) -> dict[str, int]:
    """Resolve relative xref targets against the symbol index, in place.

    Docstrings reference symbols the way Python code does -- bare or
    class-relative names -- while the index keys dotted paths. Each target
    is tried against its enclosing scopes (class, then page), then as an
    exact index key, then as an anchor across all pages. An anchor match
    counts only when every candidate is the same symbol (one canonical
    path); the winner is the earliest key, so the ``MODULE_PAGES`` priority
    order decides which page a multiply-documented symbol links to. An
    ambiguous or unknown target is left as written; the renderer keeps the
    code-span label and drops the link.

    Args:
        modules: The serialized module records, mutated in place.
        symbols: The symbol index from ``_build_symbol_index``.

    Returns:
        The unresolved targets, mapped to their occurrence counts.
    """
    by_anchor: dict[str, list[str]] = {}
    for key, entry in symbols.items():
        by_anchor.setdefault(entry["anchor"], []).append(key)
    unresolved: dict[str, int] = {}

    def resolve(target: str, scopes: list[str]) -> str | None:
        for scope in scopes:
            key = f"{scope}.{target}"
            if key in symbols:
                return key
        if target in symbols:
            return target
        candidates = by_anchor.get(target, [])
        canonicals = {symbols[key]["canonical"] for key in candidates}
        if len(canonicals) == 1:
            return candidates[0]
        return None

    def rewrite(value: str, scopes: list[str]) -> str:
        def repl(match: re.Match[str]) -> str:
            resolved = resolve(match.group(1), scopes)
            if resolved is None:
                target = match.group(1)
                unresolved[target] = unresolved.get(target, 0) + 1
                return match.group(0)
            return f"(xref:{resolved})"

        return _XREF_RE.sub(repl, value)

    def walk(node: object, scopes: list[str]) -> None:
        if isinstance(node, dict):
            record = cast(dict[str, Any], node)
            for key, value in record.items():
                if key in {"value", "description"} and isinstance(value, str):
                    record[key] = rewrite(value, scopes)
                else:
                    walk(value, scopes)
        elif isinstance(node, list):
            for item in cast(list[Any], node):
                walk(item, scopes)

    for record in modules:
        page = record["path"]
        walk(record["docstring"], [page])
        for member in record["members"]:
            walk(member, [f"{page}.{member['name']}", page])
    return unresolved


def main() -> None:
    """Load the package, serialize the documented pages, write the JSON.

    Raises:
        SystemExit: If the unresolved cross-reference set drifts from
            ``xref_baseline.txt``. The JSON is written first, so the
            artifact is available while debugging the drift.
    """
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("api.json")
    pkg = griffe.load(
        "standard_asr",
        docstring_parser=griffe.Parser.google,
        resolve_aliases=True,
    )
    if not isinstance(pkg, griffe.Module):
        raise TypeError("griffe did not load standard_asr as a module")
    modules: list[dict[str, Any]] = []
    for dotted in MODULE_PAGES:
        modules.append(_serialize_module(_module_page(pkg, dotted)))
    # Cross-reference resolution prefers the concrete module page; the list
    # above is ordered so the first writer wins.
    symbols = _build_symbol_index(modules, pkg)
    unresolved = _link_xrefs(modules, symbols)
    payload = {
        "package": "standard-asr",
        "version": dist_version("standard-asr"),
        "modules": modules,
        "symbols": {
            key: {"page": entry["page"], "anchor": entry["anchor"]}
            for key, entry in symbols.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    member_count = sum(len(record["members"]) for record in modules)
    print(
        f"wrote {out_path} ({member_count} members, {len(symbols)} symbols, "
        f"{sum(unresolved.values())} unresolved xrefs)"
    )
    # The unresolved set is pinned: the baseline lists the external,
    # private, and undocumented targets that stay unlinked on purpose, so
    # a typo or a rename fails the build instead of degrading a link.
    baseline_path = Path(__file__).with_name("xref_baseline.txt")
    baseline = {
        line
        for line in baseline_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    new = sorted(set(unresolved) - baseline)
    stale = sorted(baseline - set(unresolved))
    for target in new:
        print(f"unresolved xref not in the baseline: {target} (x{unresolved[target]})")
    for target in stale:
        print(f"baseline entry no longer unresolved: {target}")
    if new or stale:
        raise SystemExit(
            f"cross-reference drift against {baseline_path.name}: fix the "
            "reference, or edit the baseline when the change is intentional"
        )


if __name__ == "__main__":
    main()
