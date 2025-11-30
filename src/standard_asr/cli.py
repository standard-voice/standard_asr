"""Command line entry point for Standard ASR developer utilities."""

from __future__ import annotations

import argparse
from typing import Callable, Iterable

from .compliance import check_entrypoints
from .discovery import discover_models


def _add_models_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register ``models`` subcommands."""

    models_parser = subparsers.add_parser(
        "models", help="Inspect discovered Standard ASR models."
    )
    models_sub = models_parser.add_subparsers(dest="models_command", required=True)

    list_parser = models_sub.add_parser("list", help="List available models.")
    list_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on invalid entry points during discovery.",
    )
    list_parser.add_argument(
        "--on-conflict",
        choices=["warn_keep_first", "replace"],
        default="warn_keep_first",
        help="Strategy for duplicate model keys.",
    )
    list_parser.set_defaults(func=_cmd_models_list)

    show_parser = models_sub.add_parser(
        "show", help="Display details about a single model."
    )
    show_parser.add_argument("name", help="Model key in '<engine>/<model>' format.")
    show_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on invalid entry points during discovery.",
    )
    show_parser.set_defaults(func=_cmd_models_show)


def _add_compliance_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register ``compliance`` subcommands."""

    compliance_parser = subparsers.add_parser(
        "compliance",
        help="Run compliance helpers to validate plugin behaviour.",
    )
    compliance_sub = compliance_parser.add_subparsers(
        dest="compliance_command", required=True
    )

    ep_parser = compliance_sub.add_parser(
        "entrypoints",
        help="Verify entry point visibility and basic factory behaviour.",
    )
    ep_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on invalid entry points at discovery time.",
    )
    ep_parser.add_argument(
        "--no-instantiate",
        dest="instantiate",
        action="store_false",
        help="Skip instantiation attempts during compliance checks.",
    )
    ep_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress warnings in output.",
    )
    ep_parser.set_defaults(func=_cmd_compliance_entrypoints, instantiate=True)


def build_parser() -> argparse.ArgumentParser:
    """Construct the root argument parser for the CLI."""

    parser = argparse.ArgumentParser(description="Standard ASR plugin utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_models_subcommands(subparsers)
    _add_compliance_subcommands(subparsers)
    return parser


def _cmd_models_list(args: argparse.Namespace) -> int:
    registry = discover_models(strict=args.strict, on_conflict=args.on_conflict)
    names = registry.names()
    if not names:
        print("No Standard ASR models were discovered.")
        return 0

    width = max(len(name) for name in names)
    print("Discovered models:")
    for name in names:
        spec = registry.spec(name)
        model_label = spec.model_name or "<default>"
        print(f" - {name.ljust(width)}  engine={spec.engine_id}  model={model_label}")
    return 0


def _cmd_models_show(args: argparse.Namespace) -> int:
    registry = discover_models(strict=args.strict)
    spec = registry.spec(args.name)
    model_label = spec.model_name or "<default>"
    print(f"Model: {spec.key}")
    print(f"  Engine ID   : {spec.engine_id}")
    print(f"  Model name  : {model_label}")
    print(f"  Module      : {spec.entry_point.module}")
    print(f"  Attribute   : {spec.entry_point.attr}")
    print(f"  Value       : {spec.entry_point.value}")
    return 0


def _cmd_compliance_entrypoints(args: argparse.Namespace) -> int:
    report = check_entrypoints(
        strict_discovery=args.strict, instantiate=args.instantiate
    )

    if report.passed:
        print("✅ Entry point compliance checks passed.")
    else:
        print("❌ Entry point compliance checks failed.")

    def _emit(issues: Iterable, prefix: str) -> None:
        for issue in issues:
            location = issue.model or "<registry>"
            print(f"{prefix} {location}: {issue.message}")

    if not args.quiet:
        _emit(report.iter_level("warning"), "⚠️ Warning")
    _emit(report.iter_level("error"), "❌ Error")

    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m standard_asr.cli`` or console script."""

    parser = build_parser()
    args = parser.parse_args(argv)
    command: Callable[[argparse.Namespace], int] = args.func
    return command(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
