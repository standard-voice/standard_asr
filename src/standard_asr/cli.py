# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Command line entry point for Standard ASR utilities."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any, Callable, Iterable, cast

from .compliance import ComplianceIssue, check_entrypoints
from .discovery import discover_models
from .exceptions import (
    AudioProcessingError,
    ConfigError,
    DiscoveryError,
    EntrypointValidationError,
    FactoryLoadError,
    TranscriptionError,
)
from .runtime import ensure_cache_dir, resolve_cache_dir
from .runtime_params import RuntimeParams


def _add_models_subcommands(subparsers: Any) -> None:
    """Register ``models`` subcommands.

    Args:
        subparsers: Subparser collection for the root CLI.

    Returns:
        None.

    Raises:
        None.
    """
    models_parser = subparsers.add_parser("models", help="Inspect discovered Standard ASR models.")
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

    show_parser = models_sub.add_parser("show", help="Display details about a single model.")
    show_parser.add_argument("name", help="Model key in '<engine>/<model>' format.")
    show_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on invalid entry points during discovery.",
    )
    show_parser.set_defaults(func=_cmd_models_show)

    cache_parser = models_sub.add_parser(
        "cache", help="Show or create the Standard ASR cache directory."
    )
    cache_parser.add_argument(
        "--ensure",
        action="store_true",
        help="Create the cache directory if it does not exist.",
    )
    cache_parser.set_defaults(func=_cmd_models_cache)

    prepare_parser = models_sub.add_parser(
        "prepare", help="Warm up a model (download/load weights if required)."
    )
    prepare_parser.add_argument("name", help="Model key in '<engine>/<model>' format.")
    prepare_parser.add_argument(
        "--options",
        help="JSON string of transcription options passed to the engine.",
    )
    prepare_parser.set_defaults(func=_cmd_models_prepare)


def _add_compliance_subcommands(subparsers: Any) -> None:
    """Register ``compliance`` subcommands.

    Args:
        subparsers: Subparser collection for the root CLI.

    Returns:
        None.

    Raises:
        None.
    """
    compliance_parser = subparsers.add_parser(
        "compliance",
        help="Run compliance helpers to validate plugin behaviour.",
    )
    compliance_sub = compliance_parser.add_subparsers(dest="compliance_command", required=True)

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


def _add_transcribe_subcommand(subparsers: Any) -> None:
    """Register the ``transcribe`` subcommand.

    Args:
        subparsers: Subparser collection for the root CLI.

    Returns:
        None.

    Raises:
        None.
    """
    parser = subparsers.add_parser("transcribe", help="Transcribe an audio file.")
    parser.add_argument("name", help="Model key in '<engine>/<model>' format.")
    parser.add_argument("audio", help="Path to audio file to transcribe.")
    parser.add_argument(
        "--options",
        help="JSON string of transcription options passed to the engine.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON transcription result.",
    )
    parser.set_defaults(func=_cmd_transcribe)


def _add_serve_subcommand(subparsers: Any) -> None:
    """Register the ``serve`` subcommand.

    Args:
        subparsers: Subparser collection for the root CLI.

    Returns:
        None.

    Raises:
        None.
    """
    parser = subparsers.add_parser("serve", help="Start the FastAPI server for Standard ASR.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level.")
    parser.set_defaults(func=_cmd_serve)


def build_parser() -> argparse.ArgumentParser:
    """Construct the root argument parser for the CLI.

    Args:
        None.

    Returns:
        Configured argument parser.

    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description="Standard ASR utilities")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show stack traces for unexpected errors.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_models_subcommands(subparsers)
    _add_compliance_subcommands(subparsers)
    _add_transcribe_subcommand(subparsers)
    _add_serve_subcommand(subparsers)
    _add_doctor_subcommand(subparsers)
    return parser


def _add_doctor_subcommand(subparsers: Any) -> None:
    """Register the ``doctor`` subcommand.

    Args:
        subparsers: Subparser collection for the root CLI.

    Returns:
        None.
    """
    parser = subparsers.add_parser("doctor", help="Diagnose plugin dependency (numpy) conflicts.")
    parser.set_defaults(func=_cmd_doctor)


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Handle the ``doctor`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (``1`` if a conflict was detected, else ``0``).
    """
    from .doctor import diagnose, format_report

    report = diagnose()
    print(format_report(report))
    return 1 if report.has_conflict else 0


def _cmd_models_list(args: argparse.Namespace) -> int:
    """Handle ``models list`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.

    Raises:
        None.
    """
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
    """Handle ``models show`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.

    Raises:
        None.
    """
    registry = discover_models(strict=args.strict)
    spec = registry.spec(args.name)
    model_label = spec.model_name or "<default>"
    print(f"Model: {spec.key}")
    print(f"  Engine ID   : {spec.engine_id}")
    print(f"  Model name  : {model_label}")
    print(f"  Module      : {spec.entry_point.module}")
    print(f"  Attribute   : {spec.entry_point.attr}")
    print(f"  Value       : {spec.entry_point.value}")
    _print_declared_capabilities(spec)
    return 0


def _print_declared_capabilities(spec: Any) -> None:
    """Print an engine's DeclaredCapabilities without instantiating it.

    Spec §264 lists ``standard-asr models show`` as a consumer of
    DeclaredCapabilities. The capabilities are read from the engine *class*
    (ClassVar), so no engine is constructed and no credentials are resolved.

    Args:
        spec: The model :class:`~standard_asr.discovery.ModelSpec`.
    """
    try:
        engine_class = spec.engine_class()
    except FactoryLoadError as exc:
        print(f"  Capabilities: <unavailable: {exc}>")
        return
    caps = getattr(engine_class, "declared_capabilities", None)
    if caps is None:
        print("  Capabilities: <none declared>")
        return
    print("  Capabilities:")
    rendered = json.dumps(caps.model_dump(mode="json"), indent=2, sort_keys=True)
    for line in rendered.splitlines():
        print(f"    {line}")


def _cmd_models_cache(args: argparse.Namespace) -> int:
    """Handle ``models cache`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.

    Raises:
        None.
    """
    path = ensure_cache_dir() if args.ensure else resolve_cache_dir()
    print(str(path))
    return 0


def _cmd_models_prepare(args: argparse.Namespace) -> int:
    """Handle ``models prepare`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.

    Raises:
        None.
    """
    registry = discover_models()
    asr = registry.create(args.name)

    prepare = getattr(asr, "prepare", None)
    if callable(prepare):
        prepare()
        print("✅ Model prepare complete.")
    else:
        # No prepare() hook: there is nothing to warm up or download. Never fire
        # a real transcribe as a stand-in -- for cloud/commercial engines that
        # would be a billable request with side effects (lazy / no-surprise).
        print("ℹ️  Engine declares no prepare() step; nothing to warm up.")
    return 0


def _cmd_compliance_entrypoints(args: argparse.Namespace) -> int:
    """Handle ``compliance entrypoints`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.

    Raises:
        None.
    """
    report = check_entrypoints(strict_discovery=args.strict, instantiate=args.instantiate)

    if report.passed:
        print("✅ Entry point compliance checks passed.")
    else:
        print("❌ Entry point compliance checks failed.")

    def _emit(issues: Iterable[ComplianceIssue], prefix: str) -> None:
        for issue in issues:
            location = issue.model or "<registry>"
            print(f"{prefix} {location}: {issue.message}")

    if not args.quiet:
        _emit(report.iter_level("warning"), "⚠️ Warning")
    _emit(report.iter_level("error"), "❌ Error")

    return 0 if report.passed else 1


def _cmd_transcribe(args: argparse.Namespace) -> int:
    """Handle ``transcribe`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.

    Raises:
        None.
    """
    registry = discover_models()
    asr = registry.create(args.name)

    params = _parse_options(args.options)
    result = asr.transcribe(args.audio, params)

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(result.text)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Handle ``serve`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.

    Raises:
        None.
    """
    try:
        from .server import run
    except ImportError:
        print(
            "FastAPI server dependencies are missing. Install with: "
            "pip install 'standard-asr[server]'."
        )
        return 1

    try:
        run(host=args.host, port=args.port, reload=args.reload, log_level=args.log_level)
    except ImportError as exc:
        print(str(exc))
        return 1
    return 0


def _parse_options(raw: str | None) -> RuntimeParams | None:
    """Parse a JSON options string into :class:`RuntimeParams`.

    Only the portable standard-set fields are supported on the CLI; engine
    ``provider_params`` are not constructible without the engine type.

    Args:
        raw: Raw JSON string.

    Returns:
        Parsed runtime parameters, or ``None``.

    Raises:
        ValueError: If JSON does not decode to an object.
    """
    if raw is None:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Options JSON must decode to an object.")
    return RuntimeParams.model_validate(cast(dict[str, Any], payload))


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m standard_asr.cli`` or console script.

    Args:
        argv: Optional list of CLI arguments.

    Returns:
        Exit code.

    Raises:
        None.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    command: Callable[[argparse.Namespace], int] = args.func
    try:
        return command(args)
    except EntrypointValidationError as exc:
        _print_error(str(exc))
        return 2
    except AudioProcessingError as exc:
        _print_error(str(exc))
        return 2
    except (ConfigError, DiscoveryError, ValueError) as exc:
        _print_error(str(exc))
        return 2
    except TranscriptionError as exc:
        _print_error(str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001
        _print_error(str(exc))
        if args.debug:
            traceback.print_exc()
        return 1


def _print_error(message: str) -> None:
    """Print a CLI error message to stderr.

    Args:
        message: Error message to emit.

    Returns:
        None.

    Raises:
        None.
    """
    print(message, file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
