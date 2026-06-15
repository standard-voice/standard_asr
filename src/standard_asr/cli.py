# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Command line entry point for Standard ASR utilities."""

from __future__ import annotations

import argparse
import inspect
import io
import json
import sys
import traceback
from typing import IO, Any, Callable, Iterable, cast

from pydantic import ValidationError

from .asr_interface import EngineBase
from .audio_format import AudioFormat
from .compliance import (
    ComplianceIssue,
    ComplianceReport,
    check_entrypoints,
    check_provider_params_swap_safety,
    check_recommended_wire_format,
    check_streaming_param_gating,
    check_sync_bridge,
    prepare_requires_arguments,
)
from .discovery import ModelRegistry, ModelSpec, discover_models
from .error_redaction import sanitized_validation_message
from .exceptions import (
    AudioProcessingError,
    ConfigError,
    DiscoveryError,
    EntrypointValidationError,
    FactoryLoadError,
    TranscriptionError,
)
from .results import Diagnostic
from .runtime import ensure_cache_dir, resolve_cache_dir
from .runtime_params import RuntimeParams, WireRuntimeParams

#: ASCII status markers. The CLI prints transcripts and decorative status lines
#: to stdout/stderr; on Windows a redirected stream defaults to the ANSI code
#: page with ``errors="strict"`` (PEP 686's UTF-8 default only lands in Python
#: 3.15, but the project supports 3.10+), where emoji raise ``UnicodeEncodeError``
#: and crash the CLI. Decorative markers therefore stay ASCII; the
#: transcript text itself is made loss-lessly printable by forcing UTF-8 on the
#: output streams (see :func:`_ensure_utf8_stream`) rather than dropping
#: characters, since a corrupted transcript is the cardinal silent-wrong-result
#: sin.
_OK = "[OK]"
_FAIL = "[FAIL]"
_WARN = "[WARN]"
_INFO = "[INFO]"

#: Copy-pasteable examples shown at the bottom of ``standard-asr --help`` so the
#: top-level help alone is enough to get started (no drilling into per-command
#: ``--help`` to learn the common invocations).
_EPILOG = """\
Examples:
  standard-asr list                                   # what engines/models are installed
  standard-asr show faster-whisper/large-v3           # properties, capabilities, config schema
  standard-asr transcribe faster-whisper/tiny a.wav   # transcribe an audio file
  standard-asr prepare faster-whisper/tiny            # pre-download / warm up weights
  standard-asr serve --port 8000                      # expose every engine over HTTP + WebSocket
  standard-asr doctor                                 # diagnose environment / dependency issues
"""


def _add_inspection_subcommands(subparsers: Any) -> None:
    """Register the model-inspection verbs as flat top-level commands.

    ``list`` / ``show`` / ``cache`` / ``prepare`` are registered directly on the
    root parser (not nested under a ``models`` group), so the common commands are
    visible in ``standard-asr --help`` without a second-level menu.

    Args:
        subparsers: Subparser collection for the root CLI.

    Returns:
        None.

    Raises:
        None.
    """
    list_parser = subparsers.add_parser("list", help="List discovered models.")
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
    list_parser.set_defaults(func=_cmd_list)

    show_parser = subparsers.add_parser(
        "show", help="Show a model's properties, capabilities, and config schema."
    )
    show_parser.add_argument("name", help="Model key in '<engine>/<model>' format.")
    show_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on invalid entry points during discovery.",
    )
    show_parser.set_defaults(func=_cmd_show)

    cache_parser = subparsers.add_parser(
        "cache", help="Show (or create) the Standard ASR cache directory."
    )
    cache_parser.add_argument(
        "--ensure",
        action="store_true",
        help="Create the cache directory if it does not exist.",
    )
    cache_parser.set_defaults(func=_cmd_cache)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Warm up a model (download/load weights if required)."
    )
    prepare_parser.add_argument("name", help="Model key in '<engine>/<model>' format.")
    _add_init_config_args(prepare_parser)
    prepare_parser.set_defaults(func=_cmd_prepare)


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

    run_parser = compliance_sub.add_parser(
        "run",
        help="Run the full compliance suite (entry points + streaming gating).",
    )
    run_parser.add_argument(
        "names",
        nargs="*",
        help="Model keys to check (default: every discovered model).",
    )
    run_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on invalid entry points at discovery time.",
    )
    run_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress warnings in output.",
    )
    run_parser.add_argument(
        "--include-bridge",
        action="store_true",
        help=(
            "Also run the sync-bridge check, which opens a streaming session "
            "(off by default: it may bill / connect for cloud engines)."
        ),
    )
    run_parser.set_defaults(func=_cmd_compliance_run)


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
    _add_init_config_args(parser)
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
    # No --reload: uvicorn's auto-reload requires an import-string app, but
    # serve() passes a configured FastAPI instance (so byte caps are honored),
    # which uvicorn rejects under reload by exiting. A flag that can only fail is
    # worse than none; for dev-reload, run uvicorn directly.
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
    parser = argparse.ArgumentParser(
        prog="standard-asr",
        description="Standard ASR -- a universal interface for speech-to-text engines.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show stack traces for unexpected errors.",
    )
    # required=False so a bare `standard-asr` prints help instead of an argparse
    # "arguments are required" error; main() routes the no-command case to
    # parser.print_help().
    subparsers = parser.add_subparsers(dest="command")
    _add_inspection_subcommands(subparsers)
    _add_transcribe_subcommand(subparsers)
    _add_serve_subcommand(subparsers)
    _add_doctor_subcommand(subparsers)
    _add_compliance_subcommands(subparsers)
    return parser


def _add_doctor_subcommand(subparsers: Any) -> None:
    """Register the ``doctor`` subcommand.

    Args:
        subparsers: Subparser collection for the root CLI.

    Returns:
        None.

    Raises:
        None.
    """
    parser = subparsers.add_parser("doctor", help="Diagnose plugin dependency (numpy) conflicts.")
    parser.set_defaults(func=_cmd_doctor)


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Handle the ``doctor`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: ``0`` iff the report verdict is clean (no conflict detected
        and conflict analysis ran -- ``DoctorReport.is_clean``), else ``1``.

    Raises:
        None.
    """
    from .doctor import diagnose, format_report

    report = diagnose()
    print(format_report(report))
    return 0 if report.is_clean else 1


def _cmd_list(args: argparse.Namespace) -> int:
    """Handle the ``list`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.

    Raises:
        EntrypointValidationError: In ``--strict`` mode, when discovery finds an
            invalid entry point or an IC.2 engine-identity collision.
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


def _cmd_show(args: argparse.Namespace) -> int:
    """Handle the ``show`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.

    Raises:
        EntrypointValidationError: In ``--strict`` mode when discovery finds an
            invalid entry point, or when the requested model name is unknown or
            malformed.
    """
    registry = discover_models(strict=args.strict)
    spec = registry.spec(args.name)
    model_label = spec.model_name or "<default>"
    print(f"Model: {spec.model_id}")
    print(f"  Engine ID   : {spec.engine_id}")
    print(f"  Model name  : {model_label}")
    print(f"  Module      : {spec.entry_point.module}")
    print(f"  Attribute   : {spec.entry_point.attr}")
    print(f"  Value       : {spec.entry_point.value}")
    _print_declared_capabilities(spec)
    return 0


def _print_declared_capabilities(spec: Any) -> None:
    """Print an engine's DeclaredCapabilities without instantiating it.

    Spec §264 lists ``standard-asr show`` as a consumer of
    DeclaredCapabilities. The capabilities are read from the engine *class*
    (ClassVar), so no engine is constructed and no credentials are resolved.

    Args:
        spec: The model :class:`~standard_asr.discovery.ModelSpec`.

    Returns:
        None.

    Raises:
        None.
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
    # Render the *canonical* JSON (the single capability serialization shared with
    # `GET /v1/.../capabilities`), so a `show` output can be compared
    # field-for-field with the wire view: every node carries the derived
    # `supported` boolean and the reader never has to know the "none"/"unsupported"
    # sentinels (spec §C R6; G.5.2 two-layer isomorphism).
    canonical_json = getattr(caps, "canonical_json", None)
    if not callable(canonical_json):
        # `declared_capabilities` is not a DeclaredCapabilities model (e.g. an
        # engine mis-declared it as a dict). discovery.py consumes metadata
        # defensively via getattr; mirror that here so the rest of `show`
        # (Engine ID, Module, ...) still renders and the author is pointed at the
        # precise diagnostic instead of an opaque AttributeError.
        type_name = type(caps).__name__
        print(
            f"  Capabilities: <invalid: declared_capabilities is not a "
            f"DeclaredCapabilities model (got {type_name}); run "
            f"'standard-asr compliance entrypoints' for diagnostics>"
        )
        return
    print("  Capabilities:")
    rendered = json.dumps(canonical_json(), indent=2, sort_keys=True)
    for line in rendered.splitlines():
        print(f"    {line}")


def _cmd_cache(args: argparse.Namespace) -> int:
    """Handle the ``cache`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.

    Raises:
        OSError: With ``--ensure``, when the cache directory cannot be created.
        RuntimeError: When the home directory cannot be resolved while computing
            the default cache location.
    """
    path = ensure_cache_dir() if args.ensure else resolve_cache_dir()
    print(str(path))
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    """Handle the ``prepare`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.

    Raises:
        EntrypointValidationError: When the named model is not installed.
        FactoryLoadError: When the engine's entry point cannot be loaded.
        ConfigError: When the engine factory rejects its configuration (a missing
            or invalid field, wrapped from pydantic and secret-scrubbed), when a
            ``--config`` / ``--set`` value is malformed, or when the engine
            declares an invalid prepare() hook (a coroutine, a non-callable
            attribute, or one that requires arguments).
    """
    registry = discover_models()
    asr = registry.create(args.name, **_parse_init_config(args))

    prepare = getattr(asr, "prepare", None)
    if prepare is None or _is_base_prepare(asr):
        # No warm-up hook: either a structural engine that declares none, or an
        # EngineBase subclass that inherited the base no-op (spec IC.11). There is
        # nothing to warm up or download. Never fire a real transcribe as a
        # stand-in -- for cloud/commercial engines that would be a billable
        # request with side effects (lazy / no-surprise).
        print(f"{_INFO} Engine declares no prepare() step; nothing to warm up.")
        return 0
    if inspect.iscoroutinefunction(prepare):
        # The spec defines prepare() as a *synchronous* zero-argument hook (spec
        # §IC.11). An `async def prepare` returns an un-awaited coroutine that
        # `callable()` would accept and silently report as "complete" without ever
        # warming up -- a silent false success. Fail loudly instead.
        raise ConfigError(
            f"engine {args.name!r} declares prepare() as a coroutine function; "
            "the prepare() warm-up hook MUST be a synchronous zero-argument "
            "method (spec Init Config IC.11)."
        )
    if not callable(prepare):
        # A non-callable `prepare` attribute is a declaration bug: it can never be
        # a warm-up hook. Reject it rather than treat it as "no hook" so the
        # author sees the mistake.
        raise ConfigError(
            f"engine {args.name!r} exposes a non-callable 'prepare' attribute; "
            "the prepare() warm-up hook MUST be a synchronous zero-argument "
            "method (spec Init Config IC.11)."
        )
    if prepare_requires_arguments(prepare):
        # The hook is sync and callable, but the IC.11 contract also requires it
        # to be invocable with no arguments. A prepare() that demands parameters
        # can never be driven by the toolchain; reject it with the same structured
        # error its coroutine/non-callable siblings raise rather than letting the
        # call below fail with a bare TypeError. (The compliance suite records the
        # same defect as 'prepare_hook_requires_args' via the shared predicate.)
        raise ConfigError(
            f"engine {args.name!r} declares prepare() with required parameters; "
            "the prepare() warm-up hook MUST be a synchronous zero-argument "
            "method (spec Init Config IC.11)."
        )
    prepare()
    print(f"{_OK} Model prepare complete.")
    return 0


def _is_base_prepare(asr: Any) -> bool:
    """Return whether ``asr.prepare`` is the inherited EngineBase no-op.

    :class:`~standard_asr.asr_interface.EngineBase` provides a default no-op
    :meth:`~standard_asr.asr_interface.EngineBase.prepare` (spec IC.11), so every
    EngineBase engine has a callable ``prepare``. An engine that did not override
    it has nothing to warm up; distinguishing the inherited no-op from a real
    override lets the CLI report "nothing to warm up" instead of a misleading
    "prepare complete". A structural (non-EngineBase) engine returns
    ``False`` here and is handled by the ``prepare is None`` branch when it
    declares no hook.

    Args:
        asr: The constructed engine instance.

    Returns:
        ``True`` when ``asr`` is an EngineBase whose ``prepare`` is not overridden.

    Raises:
        None.
    """
    if not isinstance(asr, EngineBase):
        return False
    prepare = inspect.getattr_static(asr, "prepare", None)
    return prepare is EngineBase.__dict__["prepare"]


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
        print(f"{_OK} Entry point compliance checks passed.")
    else:
        print(f"{_FAIL} Entry point compliance checks failed.")

    if not args.quiet:
        _emit_issues(report.iter_level("warning"), f"{_WARN} Warning")
    _emit_issues(report.iter_level("error"), f"{_FAIL} Error")

    return 0 if report.passed else 1


def _emit_issues(issues: Iterable[ComplianceIssue], prefix: str) -> None:
    """Print compliance issues with a status prefix, one per line.

    Args:
        issues: The issues to print.
        prefix: Leading status label (already an ASCII marker).

    Returns:
        None.

    Raises:
        None.
    """
    for issue in issues:
        location = issue.model or "<registry>"
        # Include the machine-readable code so CI can grep a stable
        # identifier rather than the rewordable human message.
        print(f"{prefix} {location} [{issue.code}]: {issue.message}")


def _scope_entrypoints_report(
    report: ComplianceReport, registry: ModelRegistry, named: set[str]
) -> ComplianceReport:
    """Scope an entry-point report's per-engine issues to a named model subset.

    ``check_entrypoints`` loops the whole discovered registry (its IC.2 /
    RuntimeParams-closedness invariants are registry-global by design), so when the
    user asks for a named subset an unrelated co-installed plugin's per-engine
    failure would otherwise count toward -- and fail -- the named run. Keep an issue
    iff it is a registry-global invariant (``model is None`` -- e.g. RuntimeParams
    closedness, no-entry-points), names a requested model, or reports an IC.2
    engine_id collision (``shadowed_engine_ids``; §IC.2 mandates these surface on
    every run, keyed by a bare engine_id). Registry-global invariants are never
    dropped.

    Args:
        report: The full-registry entry-point report.
        registry: The discovered registry (for ``shadowed_engine_ids``).
        named: The requested model keys.

    Returns:
        A new :class:`ComplianceReport` carrying only the in-scope issues.

    Raises:
        None.
    """
    kept = [
        issue
        for issue in report.issues
        if issue.model is None
        or issue.model in named
        or issue.model in registry.shadowed_engine_ids
    ]
    return ComplianceReport(registry=report.registry, issues=kept)


def _cmd_compliance_run(args: argparse.Namespace) -> int:
    """Handle ``compliance run`` command (the full one-command suite).

    Delivers G.2.1's "one command validates compliance" promise beyond the entry
    point checks: it runs ``check_entrypoints`` and then, for every selected
    model that constructs without arguments, the provider_params swap-safety
    check (Runtime R3 / §5.4) and -- for an engine that declares a streaming axis
    -- the streaming parameter-gating check. The sync-bridge check
    opens a real streaming session, so it is **opt-in** (``--include-bridge``) --
    for a cloud engine that is a billable connection. The event-sequence check
    needs a recorded event stream the CLI cannot synthesize, so it stays a library
    API (``standard_asr.compliance.check_event_sequence``) and the output points
    the author at it rather than silently omitting that dimension.

    Engines that require constructor arguments (e.g. credentials) are reported as
    skipped with the reason, not failed: their entry point metadata was already
    validated, and the standard layer cannot supply real credentials.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: ``0`` when every executed check passed, else ``1``.

    Raises:
        EntrypointValidationError: In ``--strict`` mode, when discovery finds an
            invalid entry point or an IC.2 engine-identity collision.
    """
    registry = discover_models(strict=args.strict)

    entrypoints = check_entrypoints(registry=registry, strict_discovery=args.strict)
    names = args.names or registry.names()
    if args.names:
        # Scope the entry-point report's per-engine issues to the named subset so an
        # unrelated co-installed plugin's failure does not fail your named run; the
        # registry-global invariants (IC.2 collisions, RuntimeParams closedness) stay.
        entrypoints = _scope_entrypoints_report(entrypoints, registry, set(args.names))
    reports: list[ComplianceReport] = [entrypoints]
    if entrypoints.passed:
        print(f"{_OK} Entry point compliance checks passed.")
    else:
        print(f"{_FAIL} Entry point compliance checks failed.")

    for name in names:
        reports.extend(_run_instance_checks(registry, name, include_bridge=args.include_bridge))

    if not args.quiet:
        for report in reports:
            _emit_issues(report.iter_level("warning"), f"{_WARN} Warning")
    for report in reports:
        _emit_issues(report.iter_level("error"), f"{_FAIL} Error")

    # The streaming event-sequence dimension cannot be exercised from the CLI (it
    # needs an author-recorded event stream); name it explicitly so the author
    # does not read a green run as "all five dimensions covered".
    print(
        f"{_INFO} Streaming event-sequence is not run here; cover it with "
        "standard_asr.compliance.check_event_sequence in your tests "
        "(see docs/for_asr_dev/plugin_entrypoints.md)."
    )

    passed = all(report.passed for report in reports)
    print(f"{_OK} Compliance run passed." if passed else f"{_FAIL} Compliance run failed.")
    return 0 if passed else 1


def _run_instance_checks(
    registry: ModelRegistry, name: str, *, include_bridge: bool
) -> list[ComplianceReport]:
    """Run the instantiation-level compliance checks for one model.

    Constructs the engine without arguments; an engine that needs configuration
    is skipped (reported, not failed). For every constructed engine it runs
    ``check_provider_params_swap_safety`` (spec Runtime R3 / §5.4 -- an
    unconditional MUST for any engine, streaming or not). For a streaming engine
    it additionally runs ``check_streaming_param_gating`` and, when
    ``include_bridge`` is set, ``check_sync_bridge``.

    Args:
        registry: The discovered model registry.
        name: The model key to check.
        include_bridge: Whether to also run the session-opening sync-bridge check.

    Returns:
        The reports produced for this model (possibly empty).

    Raises:
        None.
    """
    try:
        spec = registry.spec(name)
    except DiscoveryError as exc:
        return [_single_error_report(name, "unknown_model", f"unknown model: {exc}")]

    if not _spec_is_zero_arg(spec):
        print(
            f"{_INFO} {name}: skipped streaming checks "
            "(engine requires constructor arguments, e.g. credentials)."
        )
        return []

    try:
        engine = registry.create(name)
    except (ConfigError, DiscoveryError, FactoryLoadError, ValueError) as exc:
        # Construction surfaced a client-side configuration problem; report it
        # as a single error for this model rather than aborting the whole run.
        return [
            _single_error_report(
                name, "engine_construction_failed", f"could not construct engine: {exc}"
            )
        ]

    # provider_params swap safety (Runtime R3 / spec §5.4) is an unconditional
    # MUST for any engine that exposes a provider_params_type, streaming or not,
    # so it runs for every constructed engine. The check itself is a no-op (an
    # immediate pass) for an engine that declares no provider_params_type. Like
    # the streaming check it is typed against EngineBase; a discovered engine that
    # passed check_entrypoints is one in practice, and the check guards every
    # surface access so a structural non-EngineBase is contained, not crashed.
    reports: list[ComplianceReport] = [check_provider_params_swap_safety(cast(EngineBase, engine))]

    if _engine_supports(engine, "streaming_input") or _engine_supports(engine, "streaming_output"):
        reports.append(check_streaming_param_gating(cast(EngineBase, engine)))
        # Self-consistency of the recommended wire format (AW-2): cheap, and passes
        # trivially for an output-only engine, so it runs for any streaming engine.
        reports.append(check_recommended_wire_format(cast(EngineBase, engine)))
        if include_bridge:
            reports.append(_run_sync_bridge(engine, name))
    return reports


def _run_sync_bridge(engine: Any, name: str) -> ComplianceReport:
    """Run the sync-bridge check against a streaming engine.

    Builds a session factory from the engine's first declared wire encoding and
    its native sample rate. A streaming engine that declares no usable
    ``wire_encodings`` cannot be bridged from the CLI; that is reported as an
    error (it is also flagged by ``check_entrypoints``).

    Args:
        engine: The constructed engine instance.
        name: The model key (for messages).

    Returns:
        The sync-bridge :class:`ComplianceReport`.

    Raises:
        ValidationError: When the engine's first declared wire encoding is blank,
            so a valid :class:`AudioFormat` cannot be built for the session.
    """
    audio_format = _streaming_audio_format(cast(EngineBase, engine))
    if audio_format is None:
        return _single_error_report(
            name,
            "sync_bridge_no_wire_format",
            "cannot run sync-bridge: engine declares streaming but no usable wire "
            "format (no declared sample rate to open a bare-frame session with).",
        )

    def _factory() -> Any:
        return engine.start_transcription(audio_format=audio_format)

    return check_sync_bridge(_factory)


def _streaming_audio_format(engine: EngineBase) -> AudioFormat | None:
    """Return the engine's recommended minimal streaming wire :class:`AudioFormat`.

    Thin CLI-side delegate to
    :meth:`~standard_asr.asr_interface.EngineBase.recommended_wire_format`, the
    single source of truth (AW-2), so the sync-bridge runner cannot drift from the
    compliance gating probe on the format a streaming engine is opened with.

    Args:
        engine: The constructed engine instance.

    Returns:
        A valid :class:`AudioFormat`, or ``None`` when the engine declares no
        usable sample rate to open a bare-frame session with.

    Raises:
        None.
    """
    return engine.recommended_wire_format()


def _spec_is_zero_arg(spec: ModelSpec) -> bool:
    """Return whether a model's factory constructs without arguments.

    Args:
        spec: The model spec.

    Returns:
        ``True`` when the factory has no required parameters, mirroring the
        compliance suite's own zero-arg test.

    Raises:
        None.
    """
    try:
        factory = spec.load_factory()
    except FactoryLoadError:
        return False
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    return not any(
        parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        for parameter in signature.parameters.values()
    )


def _engine_supports(engine: Any, dot_path: str) -> bool:
    """Query an engine's capability support defensively.

    Args:
        engine: The constructed engine instance.
        dot_path: A capability dot-path.

    Returns:
        ``True`` when supported; ``False`` if the engine lacks a ``supports``
        method or it raises (fail-closed, mirroring the runtime's gating).

    Raises:
        None.
    """
    supports = getattr(engine, "supports", None)
    if not callable(supports):
        return False
    try:
        return bool(supports(dot_path))
    except Exception:  # noqa: BLE001 - a broken declaration is treated as unsupported
        return False


def _single_error_report(name: str, code: str, message: str) -> ComplianceReport:
    """Build a one-error :class:`ComplianceReport` for a single model.

    Args:
        name: The model key the error pertains to.
        code: The machine-readable, stable issue code.
        message: The error message.

    Returns:
        A failing report carrying exactly one error issue.

    Raises:
        None.
    """
    return ComplianceReport(
        registry=ModelRegistry({}),
        issues=[ComplianceIssue(level="error", code=code, message=message, model=name)],
    )


def _add_init_config_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--config`` / ``--set`` engine init-config flags.

    These let the CLI supply an engine's *init* configuration (e.g. ``device``,
    ``compute_type``) -- previously reachable only through
    ``STANDARD_ASR_<ENGINE>__<FIELD>`` env vars, which were undiscoverable from
    ``--help``. ``--options`` is separate: it carries *runtime* params, not init
    config.

    Args:
        parser: The subcommand parser to extend.

    Returns:
        None.

    Raises:
        None.
    """
    parser.add_argument(
        "--config",
        metavar="JSON",
        help=(
            "Engine init-config as a JSON object, e.g. "
            '--config \'{"device": "cpu"}\'. Merged under --set. Run '
            "'standard-asr show <model>' to see the config schema."
        ),
    )
    parser.add_argument(
        "--set",
        dest="set_",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "Set one init-config field (repeatable), e.g. "
            "--set device=cpu --set compute_type=int8. Overrides --config. For "
            "secrets (api_key, tokens) prefer the STANDARD_ASR_<ENGINE>__<FIELD> "
            "env vars -- command-line values are visible in shell history."
        ),
    )


def _parse_init_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build an engine init-config mapping from ``--config`` and ``--set``.

    ``--config`` supplies a base JSON object; each ``--set KEY=VALUE`` then
    overrides or adds a field (``--set`` wins). ``--set`` values are strings --
    the engine's pydantic config coerces them (``"5"`` -> ``5``), exactly like the
    env-var path. A construction failure is surfaced by ``registry.create`` as a
    scrubbed :class:`ConfigError` (the value is never echoed).

    Args:
        args: Parsed CLI arguments (reads ``--config`` and ``--set``).

    Returns:
        The init-config mapping to splat into ``registry.create``.

    Raises:
        ConfigError: If ``--config`` is not a JSON object, or a ``--set`` item is
            not ``KEY=VALUE`` with a non-empty key.
    """
    config: dict[str, Any] = {}
    raw_config: str | None = getattr(args, "config", None)
    if raw_config:
        try:
            parsed: object = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"--config must be a JSON object: {exc}.") from exc
        if not isinstance(parsed, dict):
            raise ConfigError(
                '--config must be a JSON object, e.g. --config \'{"device": "cpu"}\'.'
            )
        config.update(cast("dict[str, Any]", parsed))
    for item in getattr(args, "set_", None) or ():
        field, sep, value = item.partition("=")
        field = field.strip()
        if not sep or not field:
            raise ConfigError(
                "Each --set must be KEY=VALUE with a non-empty key, e.g. --set device=cpu."
            )
        config[field] = value
    return config


def _cmd_transcribe(args: argparse.Namespace) -> int:
    """Handle ``transcribe`` command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.

    Raises:
        EntrypointValidationError: When the named model is not installed.
        FactoryLoadError: When the engine's entry point cannot be loaded.
        ValueError: When ``--options`` is not a valid portable params object, or
            from the engine on an invalid candidate-language list.
        ConfigError: On an invalid language configuration, a malformed
            ``--config`` / ``--set``, or when the engine factory rejects its
            configuration (wrapped from pydantic and secret-scrubbed).
        AudioProcessingError: On a decode, size, missing-sample-rate, or
            incompatible-input failure in the conversion pipeline (includes its
            ``IncompatibleAudioInputError`` / ``UnsafeAudioUrlError`` subclasses).
        UnsupportedFeatureError: In strict mode, on an unsupported parameter or a
            non-selectable language.
        InvalidProviderParamError: On wrong ``provider_params`` (swap-safety).
        TranscriptionError: On an engine-execution failure during transcription.
    """
    registry = discover_models()
    asr = registry.create(args.name, **_parse_init_config(args))

    params = _parse_options(args.options)
    result = asr.transcribe(args.audio, params)

    if args.json:
        # The JSON view already carries `diagnostics`; the text view renders them
        # to stderr below so neither path drops them.
        print(result.model_dump_json(indent=2))
    else:
        print(result.text)
        _render_diagnostics(result.diagnostics)
    return 0


def _render_diagnostics(diagnostics: Iterable[Diagnostic]) -> None:
    """Render transcription diagnostics to stderr (text mode).

    The runtime attaches a structured :class:`~standard_asr.results.Diagnostic`
    for every lossy step (an ad-hoc resample, a bare-array sample-rate
    assumption, a guidance degrade, ...). The default text output prints only the
    transcript, so without this the provenance warnings vanish on the surface
    end users reach most -- a silent degrade, which the project forbids.
    They go to **stderr** so stdout stays a clean, pipeable
    transcript, mirroring the "errors to stderr" convention (cli.md §2). The
    ``--json`` view already carries them on the result.

    Args:
        diagnostics: The diagnostics attached to the result.

    Returns:
        None.

    Raises:
        None.
    """
    for diag in diagnostics:
        _print_error(f"{_WARN} diagnostic [{diag.code}]: {diag.message}")


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
        _print_error(
            "FastAPI server dependencies are missing. Install with: "
            "pip install 'standard-asr[server]'."
        )
        return 1

    try:
        run(host=args.host, port=args.port, log_level=args.log_level)
    except ImportError as exc:
        _print_error(str(exc))
        return 1
    return 0


def _parse_options(raw: str | None) -> RuntimeParams | None:
    """Parse a JSON options string into :class:`RuntimeParams`.

    Mirrors the server's untyped-wire rule (D5): validation goes through
    :class:`WireRuntimeParams`, the portable-only wire view, so an options
    object that includes the engine-specific ``provider_params`` escape hatch
    is rejected with a clear validation error -- it is not constructible from
    untyped JSON and must never reach the engine unvalidated. The validated
    portable params are then promoted to the internal :class:`RuntimeParams`.

    The pydantic ``ValidationError`` raised by an invalid options object is
    **not** surfaced verbatim: ``str(ValidationError)`` echoes the offending
    ``input`` value, so a secret mis-pasted into ``--options`` (e.g.
    ``{"api_key": "sk-..."}``, rejected by ``extra="forbid"``) would otherwise be
    reflected to stderr and bleed into CI logs / bug reports. It is re-raised as
    a ``ValueError`` carrying the shared sanitized message (the same scrub the
    server applies to its 422 body, spec server.md §1) so the field name and
    validator message are kept but the value is dropped.

    Args:
        raw: Raw JSON string.

    Returns:
        Parsed runtime parameters, or ``None``.

    Raises:
        ValueError: If JSON does not decode to an object, or the object is not
            a valid portable params object (including when it carries a
            ``provider_params`` key). The message never echoes the submitted
            value.
    """
    if raw is None:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Options JSON must decode to an object.")
    try:
        validated = WireRuntimeParams.model_validate(cast(dict[str, Any], payload))
    except ValidationError as exc:
        raise ValueError(sanitized_validation_message(exc)) from exc
    return validated.to_runtime_params()


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m standard_asr.cli`` or console script.

    Args:
        argv: Optional list of CLI arguments.

    Returns:
        Exit code.

    Raises:
        None.
    """
    # Force UTF-8 on the output streams before anything is printed so a
    # transcript (or any non-ASCII text) survives a redirected/piped stream on
    # Windows, where the default is the ANSI code page with errors="strict".
    # Decorative status markers are ASCII regardless.
    _ensure_utf8_stream(sys.stdout)
    _ensure_utf8_stream(sys.stderr)

    parser = build_parser()
    args = parser.parse_args(argv)
    command: Callable[[argparse.Namespace], int] | None = getattr(args, "func", None)
    if command is None:
        # Bare `standard-asr` with no subcommand: print help and exit cleanly
        # rather than an argparse "arguments are required" error (friendlier
        # first-run UX; the subparsers are registered with required=False).
        parser.print_help()
        return 0
    try:
        return command(args)
    except EntrypointValidationError as exc:
        _print_error(str(exc))
        _debug_traceback(args)
        return 2
    except AudioProcessingError as exc:
        _print_error(str(exc))
        _debug_traceback(args)
        return 2
    except ValidationError as exc:
        # pydantic's str(ValidationError) echoes the offending input_value verbatim,
        # leaking a SecretStr-bound credential in plaintext. ValidationError IS a
        # ValueError, so without this branch it falls into the generic one below and
        # prints the secret; catch it first and emit the shared sanitized message (the
        # same scrub _parse_options applies), matching the server's construction-path
        # redaction (RR-014).
        _print_error(sanitized_validation_message(exc, prefix="Invalid configuration"))
        _debug_traceback(args)
        return 2
    except (ConfigError, DiscoveryError, ValueError) as exc:
        _print_error(str(exc))
        _debug_traceback(args)
        return 2
    except TranscriptionError as exc:
        _print_error(str(exc))
        _debug_traceback(args)
        return 1
    except Exception as exc:  # noqa: BLE001
        _print_error(str(exc))
        _debug_traceback(args)
        return 1


def _debug_traceback(args: argparse.Namespace) -> None:
    """Emit a stack trace to stderr when ``--debug`` is set.

    cli.md describes ``--debug`` as emitting "stack traces for unexpected
    errors", but the trace was previously printed only in the final
    ``except Exception`` branch, so an error caught by a named branch (e.g. an
    engine-internal failure surfacing as a ``ValueError`` from ``_transcribe``)
    had no trace even with ``--debug``. Routing every branch through
    this helper makes the flag uniform. ``getattr`` keeps it safe for the
    argparse error paths where ``--debug`` may be absent from the namespace.

    Args:
        args: Parsed CLI arguments.

    Returns:
        None.

    Raises:
        None.
    """
    if getattr(args, "debug", False):
        traceback.print_exc()


def _ensure_utf8_stream(stream: IO[str]) -> None:
    """Reconfigure a text stream to UTF-8 when it is not already UTF-8.

    On Windows a stdout/stderr redirected to a file or pipe defaults to the
    process ANSI code page (e.g. cp1252) with ``errors="strict"``; printing a
    non-ASCII transcript then raises ``UnicodeEncodeError`` and crashes the CLI
    (PEP 686's UTF-8 default only lands in Python 3.15, but this project targets
    3.10+). Forcing UTF-8 -- never ``errors="replace"`` -- keeps the transcript
    loss-less; replacing characters would silently corrupt the result, the
    cardinal sin. A no-op on streams already UTF-8 (the common POSIX case) or
    that do not support ``reconfigure`` (already-wrapped/replaced test streams).

    Args:
        stream: The text stream to reconfigure (``sys.stdout`` / ``sys.stderr``).

    Returns:
        None.

    Raises:
        None.
    """
    encoding = getattr(stream, "encoding", None)
    if isinstance(encoding, str) and encoding.lower().replace("-", "") in {"utf8"}:
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, io.UnsupportedOperation, OSError):
            # The stream cannot be reconfigured (already detached, or a custom
            # buffer). Leave it as-is rather than crash: callers that print only
            # ASCII status markers are unaffected, and a genuinely un-encodable
            # transcript still fails loudly rather than silently corrupting.
            return


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
