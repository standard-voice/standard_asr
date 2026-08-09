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

from standard_asr.audio.format import AudioFormat
from standard_asr.compliance import (
    DEFAULT_SYNC_BRIDGE_TIMEOUT,
    ComplianceIssue,
    ComplianceReport,
    check_entrypoints,
    check_provider_params_swap_safety,
    check_streaming_param_gating,
    check_sync_bridge,
    prepare_requires_arguments,
    validate_bridge_timeout,
)
from standard_asr.contract.exceptions import (
    AudioProcessingError,
    ConfigError,
    ConfigurationRequiredError,
    DiscoveryError,
    EngineContractError,
    EntrypointValidationError,
    FactoryLoadError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from standard_asr.contract.params import RuntimeParams, WireRuntimeParams
from standard_asr.contract.results import Diagnostic, TranscriptionResult
from standard_asr.plugins.discovery import ModelRegistry, ModelSpec, discover_models
from standard_asr.runtime.downloads import ensure_cache_dir, resolve_cache_dir
from standard_asr.runtime.interface import EngineBase, StandardASR
from standard_asr.runtime.protocol_boundary import (
    require_sync_result,
    safe_type_name,
    sync_result_defect,
)
from standard_asr.runtime.redaction import (
    chain_has_validation_error,
    safe_exception_summary,
    safe_str,
    sanitized_validation_message,
)


class _EngineFault(Exception):
    """CLI-internal envelope: an ENGINE-side fault caught at the execution seam.

    ``main``'s top-level arms classify by exception class, but the
    ``ValueError`` family is class-ambiguous: a marked usage ``ValueError``
    from ``_parse_options`` and an engine SDK's internal ``ValueError`` from
    inside ``transcribe()`` share a type while owning opposite faults. The
    seam is what knows: :func:`_run_engine_call` wraps the engine invocation
    and envelopes what escapes it, so ``main`` maps this envelope -- not a
    class guess -- to the engine-fault exit (1), mirroring the server's
    scrubbed 500 for the same seam. Never part of the public API.
    """

    def __init__(self, original: BaseException) -> None:
        self.original = original
        super().__init__("engine execution fault")


def _run_engine_call(invoke: Callable[[], Any]) -> Any:
    """Run one engine invocation, classifying its escapes by the seam.

    Everything the ``ValueError`` family can mean at THIS seam is
    engine-side: ``--options`` content was already validated by
    ``_parse_options`` (its usage errors are raised before the engine runs),
    audio input problems have their own type (``AudioProcessingError``),
    strict-mode rejections theirs (``UnsupportedFeatureError``), and the CLI
    user cannot supply ``provider_params`` at all (the wire view rejects
    them), so an ``InvalidProviderParamError`` here is engine misbehavior
    too. A raw ``ValidationError`` is the structural-engine
    invalid-internal-model case the server maps to a scrubbed 500. The one
    exception is :class:`ConfigError` (with its
    :class:`ConfigurationRequiredError` subtype): configuration is
    invoker-owned at the CLI regardless of WHEN the engine checks it (a
    deferred credential check surfaces at first transcribe), so it
    propagates to ``main``'s caller-actionable arm (exit 2). Engine faults
    that already carry their own non-``ValueError`` types
    (``TranscriptionError``, ``EngineContractError``) propagate untouched to
    their exit-1 arms.

    Args:
        invoke: A zero-argument closure performing the engine call.

    Returns:
        Whatever ``invoke`` returns.

    Raises:
        _EngineFault: Wrapping any ``ValueError``-family escape that is not
            a ``ConfigError``.
        BaseException: Everything else, unchanged.
    """
    try:
        return invoke()
    except ConfigError:
        raise
    except ValueError as exc:
        raise _EngineFault(exc) from exc


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
  standard-asr show faster-whisper/large-v3           # identity, capabilities, config schema
  standard-asr transcribe faster-whisper/tiny a.wav   # transcribe an audio file
  standard-asr prepare faster-whisper/tiny            # pre-download / warm up weights
  standard-asr serve --port 8000                      # expose every engine over HTTP + WebSocket
  standard-asr doctor                                 # diagnose environment / dependency issues
"""


def _add_strict_discovery_flag(parser: Any) -> None:
    """Register the shared ``--strict-discovery`` flag on a subparser.

    One registration site for the six discovery-driven commands so the flag
    name, default, and the collision-explaining help text can never drift
    between them (the same single-owner pattern as ``_add_init_config_args``).

    Args:
        parser: The subcommand parser to extend.

    Returns:
        None.

    Raises:
        None.
    """
    parser.add_argument(
        "--strict-discovery",
        action="store_true",
        help=(
            "Fail on invalid plugin entry points during discovery. (Named "
            "--strict-discovery, not --strict. 'strict' alone is the engine's "
            "strict/best_effort parameter-gating policy -- an init-config field "
            "set via --set strict=..., a different setting.)"
        ),
    )


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
    list_parser = subparsers.add_parser("list", help="List discovered models.", allow_abbrev=False)
    _add_strict_discovery_flag(list_parser)
    list_parser.add_argument(
        "--on-conflict",
        choices=["warn_keep_first", "replace"],
        default="warn_keep_first",
        help="Strategy for duplicate model keys.",
    )
    list_parser.set_defaults(func=_cmd_list)

    show_parser = subparsers.add_parser(
        "show",
        help="Show a model's identity, capabilities, and config schema.",
        allow_abbrev=False,
    )
    show_parser.add_argument("name", help="Model key in '<engine>/<model>' format.")
    _add_strict_discovery_flag(show_parser)
    show_parser.set_defaults(func=_cmd_show)

    cache_parser = subparsers.add_parser(
        "cache",
        help="Show (or create) the Standard ASR cache directory.",
        allow_abbrev=False,
    )
    cache_parser.add_argument(
        "--ensure",
        action="store_true",
        help="Create the cache directory if it does not exist.",
    )
    cache_parser.set_defaults(func=_cmd_cache)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Warm up a model (download/load weights if required).",
        allow_abbrev=False,
    )
    prepare_parser.add_argument("name", help="Model key in '<engine>/<model>' format.")
    _add_strict_discovery_flag(prepare_parser)
    _add_init_config_args(prepare_parser)
    prepare_parser.set_defaults(func=_cmd_prepare)


def _positive_finite_seconds(value: str) -> float:
    """Parse an argparse seconds value that MUST be finite and strictly positive.

    A bare ``type=float`` accepts ``0``, negatives, ``inf`` and ``nan``; fed
    into ``check_sync_bridge`` those turn into a ``Thread.join`` timeout that
    returns immediately (``<= 0``) or never (``inf``) -- a false
    "did not terminate" verdict or a hang, blamed on the engine instead of the
    flag. Reject them as usage errors at parse time.

    Args:
        value: The raw command-line token.

    Returns:
        The parsed seconds as a ``float``.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a number, not finite,
            or not strictly positive.
    """
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from exc
    try:
        return validate_bridge_timeout(seconds)
    except ValueError as exc:
        # One rule, one owner (compliance.validate_bridge_timeout); the CLI
        # only converts the library's ValueError into an argparse usage error.
        raise argparse.ArgumentTypeError(str(exc)) from exc


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
        help="Run compliance helpers to validate plugin behavior.",
        allow_abbrev=False,
    )
    compliance_sub = compliance_parser.add_subparsers(dest="compliance_command", required=True)

    ep_parser = compliance_sub.add_parser(
        "entrypoints",
        help="Verify entry point visibility and basic factory behavior.",
        allow_abbrev=False,
    )
    _add_strict_discovery_flag(ep_parser)
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
        allow_abbrev=False,
    )
    run_parser.add_argument(
        "names",
        nargs="*",
        help="Model keys to check (default: every discovered model).",
    )
    _add_strict_discovery_flag(run_parser)
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
    run_parser.add_argument(
        "--bridge-timeout",
        type=_positive_finite_seconds,
        # None sentinel (not 5.0): the effective default is applied in
        # _cmd_compliance_run so an EXPLICIT --bridge-timeout without
        # --include-bridge can be rejected loudly -- with a literal default the
        # two are indistinguishable and the flag was silently inert.
        default=None,
        metavar="SECONDS",
        help=(
            "Timeout in seconds granted to each phase of the sync-bridge check "
            "-- session establishment, then the bridged open/end-of-audio/"
            f"drain/close (default: {DEFAULT_SYNC_BRIDGE_TIMEOUT}). Only "
            "meaningful together with --include-bridge (passing it alone is a "
            "usage error, not a silent no-op). Raise it for engines with slow "
            "session setup or teardown. This flag is how you act on the check's "
            "'re-run with a larger timeout' advice."
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
    parser = subparsers.add_parser(
        "transcribe", help="Transcribe an audio file.", allow_abbrev=False
    )
    parser.add_argument("name", help="Model key in '<engine>/<model>' format.")
    parser.add_argument("audio", help="Path to audio file to transcribe.")
    parser.add_argument(
        "--options",
        help="JSON string of transcription options passed to the engine.",
    )
    _add_strict_discovery_flag(parser)
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
    parser = subparsers.add_parser(
        "serve", help="Start the FastAPI server for Standard ASR.", allow_abbrev=False
    )
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
        # No prefix abbreviation, here and on every subparser: argparse's
        # default abbreviation is implicit magic ("--strict" would silently
        # parse as --strict-discovery, resurrecting the very strict-vs-
        # strict-discovery confusion the flag rename exists to prevent), and
        # any abbreviation a user scripts today can turn ambiguous -- a
        # behavior change -- the day a new flag lands.
        allow_abbrev=False,
        description="Standard ASR -- a universal interface for speech-to-text engines.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Show a stack trace on any error path (a validation error prints a "
            "scrubbed summary instead)."
        ),
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
    parser = subparsers.add_parser(
        "doctor", help="Diagnose plugin dependency (numpy) conflicts.", allow_abbrev=False
    )
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
    from standard_asr.toolchain.doctor import diagnose, format_report

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
        EntrypointValidationError: In ``--strict-discovery`` mode, when discovery finds an
            invalid entry point or an engine-identity collision.
    """
    registry = discover_models(strict=args.strict_discovery, on_conflict=args.on_conflict)
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
        EntrypointValidationError: In ``--strict-discovery`` mode when discovery finds an
            invalid entry point, or when the requested model name is unknown or
            malformed (a caller-fixable key: exit 2).
        FactoryLoadError: When the model IS registered but its plugin cannot
            be imported/resolved. The metadata gathered so far and a
            sanitized capabilities line are printed FIRST -- the operator
            still gets everything ``show`` could learn -- and the error then
            propagates so ``main`` reports the installation fault
            (exit 1). Reporting 0 here would tell a script the model is
            usable when nothing about the engine could be read.
        _EngineFault: When reading the engine class's
            ``declared_capabilities`` runs a plugin descriptor that raises
            the ``ValueError`` family (exit 1).
    """
    registry = discover_models(strict=args.strict_discovery)
    spec = registry.spec(args.name)
    model_label = spec.model_name or "<default>"
    print(f"Model: {spec.model_id}")
    print(f"  Engine ID   : {spec.engine_id}")
    print(f"  Model name  : {model_label}")
    print(f"  Module      : {spec.entry_point.module}")
    print(f"  Attribute   : {spec.entry_point.attr}")
    print(f"  Value       : {spec.entry_point.value}")
    _print_declared_capabilities(spec)
    _print_config_schema(registry, args.name)
    return 0


def _print_declared_capabilities(spec: Any) -> None:
    """Print an engine's DeclaredCapabilities without instantiating it.

    ``standard-asr show`` is a consumer of DeclaredCapabilities. The
    capabilities are read from the engine *class* (ClassVar), so no engine is
    constructed and no credentials are resolved.

    Args:
        spec: The model :class:`~standard_asr.plugins.discovery.ModelSpec`.

    Returns:
        None.

    Raises:
        FactoryLoadError: Re-raised after rendering the sanitized
            unavailable line, so ``show`` reports the installation fault
            through ``main``'s exit-1 arm instead of a silent success.
        _EngineFault: When the class-level ``declared_capabilities`` lookup
            executes a plugin descriptor that raises the ``ValueError``
            family -- the same engine-fault seam as an instance call.
    """
    try:
        engine_class = spec.engine_class()
    except FactoryLoadError as exc:
        # The FactoryLoadError chain can hold a plugin's import-time
        # ValidationError; render through the total safe boundary.
        # Then RE-RAISE: printing the fault while returning success made
        # `show` the one consumer that swallowed a broken installation --
        # the caller's key was fine, so it is not exit 2 either.
        print(f"  Capabilities: <unavailable: {safe_exception_summary(exc)}>")
        raise
    # A class-level attribute read is still plugin code: `declared_capabilities`
    # can be a descriptor/property on the engine class (or its metaclass), so
    # the lookup goes through the engine-fault seam like every other engine call.
    caps = _run_engine_call(lambda: getattr(engine_class, "declared_capabilities", None))
    if caps is None:
        print("  Capabilities: <none declared>")
        return
    # Render the *canonical* JSON (the single capability serialization shared with
    # `GET /v1/.../capabilities`), so a `show` output can be compared
    # field-for-field with the wire view: every node carries the derived
    # `supported` boolean and the reader never has to know the "none"/"unsupported"
    # sentinels.
    canonical_json = _run_engine_call(lambda: getattr(caps, "canonical_json", None))
    if not callable(canonical_json):
        # `declared_capabilities` is not a DeclaredCapabilities model (for example, an
        # engine mis-declared it as a dict). discovery.py consumes metadata
        # defensively via getattr; mirror that here so the rest of `show`
        # (Engine ID, Module, and so on) still renders and the author is pointed at the
        # precise diagnostic instead of an opaque AttributeError.
        type_name = safe_type_name(caps)
        print(
            f"  Capabilities: <invalid: declared_capabilities is not a "
            f"DeclaredCapabilities model (got {type_name}); run "
            f"'standard-asr compliance entrypoints' for diagnostics>"
        )
        return
    print("  Capabilities:")
    rendered = json.dumps(_run_engine_call(canonical_json), indent=2, sort_keys=True)
    for line in rendered.splitlines():
        print(f"    {line}")


def _print_config_schema(registry: ModelRegistry, name: str) -> None:
    """Print an engine's init-config JSON Schema without instantiating it.

    ``standard-asr show`` is a consumer of the same class-level ``config_type``
    schema the server exposes at ``GET /v1/config-schema/...``: it lets an author
    (or a settings UI) see an engine's init fields -- ``device``,
    ``compute_type``, credentials -- **before** constructing it, since
    construction may need the very values the schema describes. The schema is read
    from the engine class, so no engine is built and no credentials are resolved.
    An engine that declares no ``config_type`` gets an explicit "no init config"
    line rather than silence, so a reader never mistakes an omission for an empty
    schema.

    Args:
        registry: The discovered model registry.
        name: Model key in ``<engine>/<model>`` format.
    """
    try:
        schema = registry.config_schema(name)
    except FactoryLoadError as exc:
        print(f"  Config schema: <unavailable: {exc}>")
        return
    if schema is None:
        print("  Config schema: <none: engine declares no init config (config_type)>")
        return
    print("  Config schema:")
    rendered = json.dumps(schema, indent=2, sort_keys=True)
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
            or invalid field, wrapped from pydantic and secret-scrubbed) or a
            ``--config`` / ``--set`` value is malformed -- configuration the
            invoking user owns, so a usage surface (exit 2 via ``main``'s
            ``ConfigError`` arm).
        EngineContractError: When the engine declares an invalid prepare()
            hook (a coroutine function, a non-callable attribute, or one that
            requires arguments) or breaks the call's runtime boundary (a
            sync ``prepare()`` returning an awaitable -- a wrapper delegating
            to an ``async def``, reporting success without ever warming up --
            or any non-``None`` value). Declaration and behavior are the same
            fault owner: the engine author (exit 1) -- no flag or env var the
            CLI user controls can fix either.
        _EngineFault: When the warm-up call itself escapes with a bare
            ``ValueError`` or raw ``ValidationError`` (the execution seam;
            exit 1 via ``main``'s envelope arm).
    """
    registry = discover_models(strict=args.strict_discovery)
    asr = registry.create(args.name, **_parse_init_config(args))

    # The BINDING is already engine code: `prepare` may be a property or any
    # descriptor, so the lookup itself runs the plugin and belongs inside the
    # engine-fault seam -- outside it, a descriptor raising ValueError was
    # reported as the invoker's usage error (exit 2).
    prepare = _run_engine_call(lambda: getattr(asr, "prepare", None))
    if prepare is None or _is_base_prepare(asr):
        # No warm-up hook: either a structural engine that declares none, or an
        # EngineBase subclass that inherited the base no-op. There is
        # nothing to warm up or download. Never fire a real transcribe as a
        # stand-in -- for cloud/commercial engines that would be a billable
        # request with side effects (lazy / no-surprise).
        print(f"{_INFO} Engine declares no prepare() step; nothing to warm up.")
        return 0
    if inspect.iscoroutinefunction(prepare):
        # The spec defines prepare() as a *synchronous* zero-argument hook. An
        # `async def prepare` returns an un-awaited coroutine that
        # `callable()` would accept and silently report as "complete" without ever
        # warming up -- a silent false success. Fail loudly instead.
        # EngineContractError, not ConfigError: a declaration-shape defect is
        # the engine author's contract violation (exit 1) -- no flag or env
        # var the CLI user owns can fix it, and the compliance suite already
        # classifies the same shapes as engine defects.
        raise EngineContractError(
            f"engine {args.name!r} declares prepare() as a coroutine function; "
            "the prepare() warm-up hook MUST be a synchronous zero-argument "
            "method."
        )
    if not callable(prepare):
        # A non-callable `prepare` attribute is a declaration bug: it can never be
        # a warm-up hook. Reject it rather than treat it as "no hook" so the
        # author sees the mistake.
        raise EngineContractError(
            f"engine {args.name!r} exposes a non-callable 'prepare' attribute; "
            "the prepare() warm-up hook MUST be a synchronous zero-argument "
            "method."
        )
    if prepare_requires_arguments(prepare):
        # The hook is sync and callable, but the contract also requires it
        # to be invocable with no arguments. A prepare() that demands parameters
        # can never be driven by the toolchain; reject it with the same structured
        # error its coroutine/non-callable siblings raise rather than letting the
        # call below fail with a bare TypeError. (The compliance suite records the
        # same defect as 'prepare_hook_requires_args' via the shared predicate.)
        raise EngineContractError(
            f"engine {args.name!r} declares prepare() with required parameters; "
            "the prepare() warm-up hook MUST be a synchronous zero-argument "
            "method."
        )
    # The declaration checks above cannot see a sync wrapper that DELEGATES to
    # an async def (iscoroutinefunction is False); only the returned value
    # betrays it. Requiring a strict None return kills that silent false
    # success -- and any other returned value -- at the runtime boundary.
    # The call itself runs behind the execution seam: a bare ValueError or
    # raw ValidationError escaping a warm-up is an engine fault (exit 1).
    require_sync_result(_run_engine_call(prepare), "prepare()", expected_type=type(None))
    print(f"{_OK} Model prepare complete.")
    return 0


def _is_base_prepare(asr: Any) -> bool:
    """Return whether ``asr.prepare`` is the inherited EngineBase no-op.

    :class:`~standard_asr.runtime.interface.EngineBase` provides a default no-op
    :meth:`~standard_asr.runtime.interface.EngineBase.prepare`, so every
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
    report = check_entrypoints(strict_discovery=args.strict_discovery, instantiate=args.instantiate)

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


def _cmd_compliance_run(args: argparse.Namespace) -> int:
    """Handle ``compliance run`` command (the full one-command suite).

    Delivers the "one command validates compliance" promise beyond the entry
    point checks: it runs ``check_entrypoints`` and then, for every selected
    model that constructs without arguments, the provider_params swap-safety
    check and -- for an engine that declares a streaming axis
    -- the streaming parameter-gating check. The sync-bridge check
    opens a real streaming session, so it is **opt-in** (``--include-bridge``) --
    for a cloud engine that is a billable connection. Two checks stay library-only
    because each needs recorded data the CLI cannot synthesize -- the
    event-sequence check (``check_event_sequence``, a streaming event stream) and
    the transcription-result check (``check_transcription_result``, a batch
    result); the output names both rather than silently omitting those dimensions.

    Engines that require constructor arguments (for example, credentials) are reported as
    skipped with the reason, not failed: their entry point metadata was already
    validated, and the standard layer cannot supply real credentials.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: ``0`` when every executed check passed; ``1`` when any
        executed check failed; ``2`` on a usage error (``--bridge-timeout``
        without ``--include-bridge``).

    Raises:
        EntrypointValidationError: In ``--strict-discovery`` mode, when
            discovery finds an invalid entry point or an engine-identity
            collision.
    """
    # An explicit --bridge-timeout without --include-bridge is a usage error,
    # not a silent no-op: the author would read "Compliance run passed" as
    # "the bridge ran within my budget" when the bridge never executed.
    if args.bridge_timeout is not None and not args.include_bridge:
        _print_error(
            "--bridge-timeout has no effect without --include-bridge; add "
            "--include-bridge to run the sync-bridge check (or drop "
            "--bridge-timeout)."
        )
        return 2
    bridge_timeout = (
        args.bridge_timeout if args.bridge_timeout is not None else DEFAULT_SYNC_BRIDGE_TIMEOUT
    )

    registry = discover_models(strict=args.strict_discovery)

    # Scope the per-engine checks -- including the instance BEHAVIORAL probes
    # (construction, the supports() sweep, the start_transcription() refusal
    # probe: a model load; for a cloud engine, potentially a billable call) --
    # to the named subset AT THE SOURCE. Filtering the report afterwards
    # discarded only the verdicts while the user still paid the probes' side
    # effects on co-installed plugins they never named. Registry-global
    # invariants (engine-identity collisions, RuntimeParams closedness) still
    # evaluate the whole environment inside check_entrypoints.
    entrypoints = check_entrypoints(
        registry=registry, strict_discovery=args.strict_discovery, names=args.names or None
    )
    names = args.names or registry.names()
    reports: list[ComplianceReport] = [entrypoints]
    if entrypoints.passed:
        print(f"{_OK} Entry point compliance checks passed.")
    else:
        print(f"{_FAIL} Entry point compliance checks failed.")

    for name in names:
        try:
            reports.extend(
                _run_instance_checks(
                    registry,
                    name,
                    include_bridge=args.include_bridge,
                    bridge_timeout=bridge_timeout,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one model's fault is one model's verdict
            # Defense in depth behind _run_instance_checks's own per-fault
            # arms: this command's contract is a verdict for EVERY model in
            # one run (G2.1), so nothing a single model does -- including a
            # check implementation crashing on a shape nobody anticipated --
            # may deny the others theirs. The two failures stay
            # distinguishable: engine_construction_failed is the plugin's
            # construction, compliance_check_crashed is the suite itself
            # falling over. Both fail the run. BaseException is not caught:
            # KeyboardInterrupt/SystemExit are the operator's control flow.
            reports.append(
                _single_error_report(
                    name,
                    "compliance_check_crashed",
                    f"compliance checks crashed for this model: {safe_exception_summary(exc)}",
                )
            )

    if not args.quiet:
        for report in reports:
            _emit_issues(report.iter_level("warning"), f"{_WARN} Warning")
    for report in reports:
        _emit_issues(report.iter_level("error"), f"{_FAIL} Error")

    # Two dimensions cannot be exercised from the CLI, because each needs
    # author-recorded data the standard layer cannot synthesize: a streaming
    # engine's event stream (check_event_sequence) and a batch result
    # (check_transcription_result). Name both -- whatever the engine's shape -- so
    # a green run is never read as "every dimension covered"; the second is the one
    # a batch-only engine would otherwise never hear about.
    print(
        f"{_INFO} Two checks are not run here (each needs recorded data the CLI "
        "cannot synthesize): check_event_sequence for a streaming engine's event "
        "stream, and check_transcription_result for a batch result. Cover them with "
        "standard_asr.compliance in your tests "
        "(see docs/for_asr_dev/plugin_entrypoints.md)."
    )

    passed = all(report.passed for report in reports)
    print(f"{_OK} Compliance run passed." if passed else f"{_FAIL} Compliance run failed.")
    return 0 if passed else 1


def _run_instance_checks(
    registry: ModelRegistry,
    name: str,
    *,
    include_bridge: bool,
    bridge_timeout: float = DEFAULT_SYNC_BRIDGE_TIMEOUT,
) -> list[ComplianceReport]:
    """Run the instantiation-level compliance checks for one model.

    Constructs the engine without arguments; an engine that needs configuration
    is skipped (reported, not failed) -- BOTH shapes: a factory whose signature
    requires arguments, and a zero-arg factory that raises
    ``ConfigurationRequiredError`` because the required credential/config is
    absent from the environment (the same classification ``check_entrypoints``
    applies, so one ``compliance run`` never issues two contradictory verdicts
    for one engine). Any OTHER ``ConfigError`` is a defect and fails. For every
    constructed engine it runs
    ``check_provider_params_swap_safety`` (an unconditional MUST for any engine,
    streaming or not). The ``recommended_wire_format()`` self-consistency
    round-trip is NOT run here: it is unconditional for every engine (spec
    §3.1) and lives in the entrypoint-layer instance checks
    (``check_entrypoints``), which the same ``compliance run`` already
    executes -- running it here too would double-report every defect. For a
    streaming engine this function additionally runs
    ``check_streaming_param_gating`` and, when
    ``include_bridge`` is set, ``check_sync_bridge`` with ``bridge_timeout``
    as its per-phase timeout (the CLI's ``--bridge-timeout``). The bridge runs
    for every streaming-axis engine; classification lives in the library:
    ``check_sync_bridge(engine=...)`` reports an output-only engine (which
    cannot accept the bridge's bare frames) as a passing, structured
    ``sync_bridge_not_applicable`` warning in the report set -- never a
    spurious failure, and never an ad-hoc CLI print a machine consumer would
    miss.

    Args:
        registry: The discovered model registry.
        name: The model key to check.
        include_bridge: Whether to also run the session-opening sync-bridge check.
        bridge_timeout: Per-phase timeout in seconds (``float``) handed to
            ``check_sync_bridge`` -- session establishment, then the bridged
            open/end-of-audio/drain/close, each granted this budget.

    Returns:
        The reports produced for this model (possibly empty).

    Raises:
        Exception: Only from a check implementation itself falling over --
            every plugin-owned fault becomes a report. The caller wraps
            this in its own per-model envelope so even that cannot deny
            the other models their verdicts.
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
    except ConfigurationRequiredError as exc:
        # Required configuration (typically a credential) is ABSENT from this
        # environment -- the SAME state check_entrypoints classifies as a
        # factory_requires_config WARNING skip in the same `compliance run`
        # invocation (from_env raises the narrow subtype automatically when
        # construction fails solely on missing required fields). Failing it
        # here while the entrypoint layer waives it gave one engine two
        # contradictory verdicts in one command. The skip is deliberately
        # scoped to ABSENCE: any other ConfigError (invalid supplied value,
        # inconsistent declaration, a wrapped construction ValidationError)
        # is a defect and falls to the error branch below -- waiving those
        # would let a broken plugin read as green-with-warning.
        # Typically from_env's sanitized ConfigurationRequiredError, but an
        # engine building config another way raises its own -- one total
        # boundary for every embedded exception text.
        print(
            f"{_INFO} {name}: skipped instance checks (factory requires "
            f"configuration not present in this environment: "
            f"{safe_exception_summary(exc)})."
        )
        return []
    except Exception as exc:  # noqa: BLE001 - a per-model verdict, never an abort
        # Construction failed for a non-absence reason. The catch is BROAD by
        # design: `registry.create` wraps only a construction-time
        # ValidationError (into ConfigError), so a factory's own
        # RuntimeError/TypeError/OSError -- an SDK failing to initialize, a
        # missing native library, a permission error on a model directory --
        # propagates verbatim. Naming a few types let exactly those escape
        # `for name in names` and abort the WHOLE multi-model run: every
        # LATER model lost its verdict, which is precisely the one-command
        # compliance guarantee (G2.1) this command exists to provide, and
        # check_entrypoints's earlier isolated call cannot compensate (this
        # is a second, independent construction). BaseException is
        # deliberately NOT caught: KeyboardInterrupt and SystemExit are the
        # operator's own control flow, not a plugin verdict.
        # safe_exception_summary, not str(exc): an engine-authored wrapper
        # (for example, `raise ConfigError(f"bad: {ve}") from ve`) may embed a chained
        # ValidationError's input echo, and this message lands in the report.
        return [
            _single_error_report(
                name,
                "engine_construction_failed",
                f"could not construct engine: {safe_exception_summary(exc)}",
            )
        ]

    # provider_params swap safety is an unconditional
    # MUST for any engine that exposes a provider_params_type, streaming or not,
    # so it runs for every constructed engine. The check itself is a no-op (an
    # immediate pass) for an engine that declares no provider_params_type.
    # No casts anywhere below: every check is typed against the StandardASR
    # protocol (or a narrower one) and reads non-protocol conveniences like
    # effective_capabilities defensively -- a structural engine gets the same
    # verdicts as an EngineBase one, never an AttributeError dressed up as a
    # compliance failure.
    reports: list[ComplianceReport] = [check_provider_params_swap_safety(engine)]

    if _engine_supports(engine, "streaming_input") or _engine_supports(engine, "streaming_output"):
        reports.append(check_streaming_param_gating(engine))
        if include_bridge:
            # No CLI-side capability pre-gate: check_sync_bridge itself
            # classifies an establishment refusal against the engine's declared
            # streaming_input (engine= below), so an output-only engine yields
            # a STRUCTURED, --quiet-respecting sync_bridge_not_applicable
            # warning inside the report set -- one layer, one message, visible
            # to machine consumers -- instead of an ad-hoc print here that a
            # script could never see.
            reports.append(_run_sync_bridge(engine, name, timeout=bridge_timeout))
    return reports


def _run_sync_bridge(
    engine: Any, name: str, *, timeout: float = DEFAULT_SYNC_BRIDGE_TIMEOUT
) -> ComplianceReport:
    """Run the sync-bridge check against a streaming engine.

    Builds a session factory from the engine's first declared wire encoding and
    its native sample rate. An engine that recommends no usable wire format is
    reported as ``sync_bridge_not_applicable`` (passing) when it verifiably
    does not declare ``streaming_input``, and as a hard
    ``sync_bridge_no_wire_format`` error otherwise -- with a message naming
    the actual fault: a ``streaming_input`` engine without a reachable
    bare-frame format is a real declaration problem, while an engine whose
    ``supports()`` raised is unverifiable (the message says so rather than
    asserting a declaration that was never observed).

    Args:
        engine: The constructed engine instance.
        name: The model key (for messages).
        timeout: Per-phase timeout in seconds handed to
            :func:`~standard_asr.compliance.check_sync_bridge` (the CLI's
            ``--bridge-timeout``).

    Returns:
        The sync-bridge :class:`ComplianceReport`.
    """
    try:
        audio_format = _streaming_audio_format(engine)
    except Exception as exc:  # noqa: BLE001 - a per-model report, never an abort
        # The entrypoint-layer instance checks report the same fault as
        # recommended_wire_format_raised; guarding here keeps a broken
        # implementation from escaping _run_instance_checks and aborting the
        # WHOLE multi-model run -- every other model's reports must survive.
        return _single_error_report(
            name,
            "sync_bridge_setup_failed",
            f"cannot run sync-bridge: recommended_wire_format() raised: "
            f"{safe_exception_summary(exc)} "
            "(also reported by the entrypoint-layer wire-format check).",
        )
    format_defect = sync_result_defect(audio_format, expected_type=(AudioFormat, type(None)))
    if format_defect is not None:
        # This is a REAL consumer: the value flows into
        # ``start_transcription(audio_format=...)`` below, so EVERY defect shape
        # must stop here, not just the awaitable. An `async def`
        # recommendation hands back an awaitable (the shared boundary has
        # already CLOSED the stray coroutine); a wrong-typed value (a str, a
        # dict, a duck object) would otherwise be handed to the engine and
        # could trigger model loads, connections, or a secondary crash blamed
        # on the bridge. Report with the SAME codes the compliance layer uses
        # for these defect classes -- and let the VERDICT pick the arm: the
        # value's metadata already trained the boundary's containment once,
        # so re-inspecting it here could raise out of the error path.
        if format_defect.kind == "awaitable":
            return _single_error_report(
                name,
                "protocol_member_not_synchronous",
                "cannot run sync-bridge: recommended_wire_format() returned an "
                "awaitable; the StandardASR protocol pins it as a SYNCHRONOUS "
                "member (also reported by the compliance checks).",
            )
        if format_defect.kind == "unclassifiable":
            return _single_error_report(
                name,
                "protocol_member_unclassifiable_result",
                f"cannot run sync-bridge: recommended_wire_format() {format_defect.clause}; "
                "a result nobody can safely classify violates the protocol "
                "outright (also reported by the compliance checks).",
            )
        return _single_error_report(
            name,
            "protocol_member_wrong_return_type",
            f"cannot run sync-bridge: recommended_wire_format() {format_defect.clause}; "
            "opening a session with it would misblame the bridge for a "
            "wrong-return-type defect (also reported by the compliance checks).",
        )
    if audio_format is None:
        try:
            raw_declared = engine.supports("streaming_input")
        except Exception:  # noqa: BLE001 - a broken supports() cannot earn a pass
            raw_declared = None
        streaming_input_declared: bool | None
        if sync_result_defect(raw_declared, expected_type=bool) is not None:
            # An awaitable (closed by the shared boundary -- bool() would
            # fabricate a "declares streaming_input" verdict from a modality
            # defect), None (the raise arm above), or any non-bool: no
            # verifiable declaration -- same fail-closed arm as a raise.
            streaming_input_declared = None
        else:
            streaming_input_declared = raw_declared
        if streaming_input_declared is False:
            # An output-only engine with no recommendable wire format: the
            # bridge (which feeds bare frames) has nothing to test -- the same
            # not-applicable verdict check_sync_bridge itself reaches for an
            # output-only engine that CAN construct a format. A hard error
            # here failed compliant output-only engines on a property of the
            # check's shape.
            return ComplianceReport(
                registry=None,
                issues=[
                    ComplianceIssue(
                        level="warning",
                        code="sync_bridge_not_applicable",
                        message=(
                            "Sync-bridge check not applicable: the engine does "
                            "not declare streaming_input and recommends no "
                            "bare-frame wire format. The bridge feeds bare PCM "
                            "frames; this is a property of the check, not an "
                            "engine failure."
                        ),
                        model=name,
                    )
                ],
            )
        if streaming_input_declared is None:
            # Fail-closed like check_sync_bridge, and HONESTLY: supports()
            # raised, or answered with the wrong shape (an awaitable / a
            # non-bool), so whether the engine declares streaming_input is
            # unverifiable -- the message must not assert a declaration that
            # was never observed.
            return _single_error_report(
                name,
                "sync_bridge_no_wire_format",
                "cannot run sync-bridge: the engine recommends no usable wire "
                "format, and its own supports() raised or returned a "
                "non-boolean/awaitable while verifying streaming_input -- a "
                "broken capability surface cannot earn a not-applicable pass; "
                "fix supports() first (the entry-point checks flag it too).",
            )
        return _single_error_report(
            name,
            "sync_bridge_no_wire_format",
            "cannot run sync-bridge: engine declares streaming_input but no usable "
            "wire format (no declared sample rate to open a bare-frame session "
            "with).",
        )

    def _factory() -> Any:
        return engine.start_transcription(audio_format=audio_format)

    return check_sync_bridge(_factory, timeout=timeout, model=name, engine=engine)


def _streaming_audio_format(engine: StandardASR) -> AudioFormat | None:
    """Return the engine's recommended minimal streaming wire :class:`AudioFormat`.

    Thin CLI-side delegate to the protocol's
    :meth:`~standard_asr.runtime.interface.StandardASR.recommended_wire_format`, the
    single source of truth, so the sync-bridge runner cannot drift from the
    compliance gating probe on the format a streaming engine is opened with.
    Typed against the protocol (not ``EngineBase``): a structural engine
    implements the member itself, and this helper must not require the base
    class.

    Args:
        engine: The constructed engine instance.

    Returns:
        A valid :class:`AudioFormat`, or ``None`` when the engine declares no
        usable sample rate to open a bare-frame session with.

    Raises:
        Exception: Whatever the engine's ``recommended_wire_format()`` raises
            (a structural engine's implementation is arbitrary code); the sole
            caller guards the call and converts a raise into a per-model
            ``sync_bridge_setup_failed`` report.
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

    Fail-closed on every malformed answer, mirroring the runtime's gating:
    only a literal ``True`` counts as supported. A truthy non-bool
    (``"false"``, an object) is a wrong-return-type defect that ``bool()``
    coercion would silently promote to a capability verdict, and an awaitable
    (an ``async def`` supports) is both truthy AND a coroutine that would
    leak never-awaited -- the shared sync-call boundary CLOSES it. This helper
    has no issue channel; the compliance checks report the underlying defect
    (``protocol_member_not_synchronous`` / ``protocol_member_wrong_return_type``)
    -- the CLI pre-gate's only job is to not act on a lie and not leak.

    Args:
        engine: The constructed engine instance.
        dot_path: A capability dot-path.

    Returns:
        ``True`` when supported; ``False`` if the engine lacks a ``supports``
        method, it raises, or it returns anything but a real ``bool``.

    Raises:
        None.
    """
    supports = getattr(engine, "supports", None)
    if not callable(supports):
        return False
    try:
        value = supports(dot_path)
    except Exception:  # noqa: BLE001 - a broken declaration is treated as unsupported
        return False
    return sync_result_defect(value, expected_type=bool) is None and value is True


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

    These let the CLI supply an engine's *init* configuration (for example, ``device``,
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
        FactoryLoadError: When the engine's entry point cannot be loaded (an
            installation fault: exit 1 via ``main``'s arm).
        ValueError: When ``--options`` is not a valid portable params object
            (raised by ``_parse_options`` BEFORE the engine runs -- the only
            usage-owned ``ValueError`` on this path; exit 2).
        _EngineFault: When the engine call itself escapes with the
            ``ValueError`` family -- a bare SDK ``ValueError``, a raw
            ``ValidationError`` from an internal model, an
            ``InvalidProviderParamError`` the CLI user cannot have caused
            (the wire view rejects ``provider_params``). The execution seam
            classifies these as engine faults (exit 1).
        ConfigError: On an invalid language configuration value, a malformed
            ``--config`` / ``--set``, or when the engine factory rejects its
            configuration (wrapped from pydantic and secret-scrubbed) --
            invoker-owned config, exit 2 even when the engine defers the
            check to first transcribe.
        AudioProcessingError: On a decode, size, missing-sample-rate, or
            incompatible-input failure in the conversion pipeline (includes its
            ``IncompatibleAudioInputError`` / ``UnsafeAudioUrlError`` subclasses).
        UnsupportedFeatureError: In strict mode, on an unsupported parameter, a
            non-selectable language, or a valid-but-unreachable candidate list
            (non-detectable / over-``max``).
        TranscriptionError: On an engine-execution failure during transcription.
        EngineContractError: When ``transcribe()`` broke the sync-call
            boundary (returned an awaitable or a non-``TranscriptionResult``),
            or the engine's language declaration is malformed (a bad declared
            tag, a missing IC.6 ``default_language``) -- engine faults
            surfaced loudly at the boundary (exit 1) instead of a secondary
            ``AttributeError`` below.
    """
    registry = discover_models(strict=args.strict_discovery)
    asr = registry.create(args.name, **_parse_init_config(args))

    params = _parse_options(args.options)
    # The execution seam: what escapes the engine call here as a bare
    # ValueError / raw ValidationError is an engine fault (exit 1 through
    # _EngineFault), never a usage error -- every caller-fixable input was
    # already validated above (see _run_engine_call).
    result = _run_engine_call(lambda: asr.transcribe(args.audio, params))
    require_sync_result(result, "transcribe()", expected_type=TranscriptionResult)

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

    The runtime attaches a structured :class:`~standard_asr.contract.results.Diagnostic`
    for every lossy step (an ad-hoc resample, a bare-array sample-rate
    assumption, a guidance degrade, and so on). The default text output prints only the
    transcript, so without this the provenance warnings vanish on the surface
    end users reach most -- a silent degrade, which the project forbids.
    They go to **stderr** so stdout stays a clean, pipeable
    transcript, mirroring the "errors to stderr" convention. The
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
        from standard_asr.toolchain.server import run
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

    Mirrors the server's untyped-wire rule: validation goes through
    :class:`WireRuntimeParams`, the portable-only wire view, so an options
    object that includes the engine-specific ``provider_params`` escape hatch
    is rejected with a clear validation error -- it is not constructible from
    untyped JSON and must never reach the engine unvalidated. The validated
    portable params are then promoted to the internal :class:`RuntimeParams`.

    The pydantic ``ValidationError`` raised by an invalid options object is
    **not** surfaced verbatim: ``str(ValidationError)`` echoes the offending
    ``input`` value, so a secret mis-pasted into ``--options`` (for example,
    ``{"api_key": "sk-..."}``, rejected by ``extra="forbid"``) would otherwise be
    reflected to stderr and bleed into CI logs / bug reports. It is re-raised as
    a ``ValueError`` carrying the shared sanitized message (the same scrub the
    server applies to its 422 body) so the field name and
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
        # Marked input-echo-free: the message is the sanitized loc/msg summary,
        # so the safe error boundary keeps it instead of withholding it.
        raise ValueError(sanitized_validation_message(exc)) from exc
    return validated.to_runtime_params()


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m standard_asr.toolchain.cli`` or console script.

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
    # Every arm reports through _print_exception -- the ONE safe boundary --
    # never bare str(exc): the normal (no --debug) error line prints
    # unconditionally, and a wrapper that copied a chained ValidationError's
    # truncated input echo into its own message would otherwise leak it to
    # stderr before any traceback logic could scrub anything.
    try:
        return command(args)
    except EntrypointValidationError as exc:
        _print_exception(exc)
        _debug_traceback(args)
        return 2
    except AudioProcessingError as exc:
        _print_exception(exc)
        _debug_traceback(args)
        return 2
    except _EngineFault as exc:
        # The execution seam already classified this (see _run_engine_call):
        # an engine-side fault the CLI user did not cause. Raw
        # ValidationError gets the operator-audience scrub (this line lands
        # in CI logs and bug reports); anything else renders its own message
        # through the safe boundary. Exit 1, the server's scrubbed-500 twin.
        original = exc.original
        if isinstance(original, ValidationError):
            _print_error(
                sanitized_validation_message(
                    original,
                    prefix="Engine produced an invalid model (engine fault)",
                )
            )
        else:
            _print_exception(original)
        _debug_traceback(args)
        return 1
    except FactoryLoadError as exc:
        # A REGISTERED model whose plugin fails to import/resolve: the
        # installation is broken, not the invocation -- the same state the
        # server maps to a scrubbed 500 (never 404). Must precede the
        # DiscoveryError arm below (FactoryLoadError subclasses it); only
        # unknown/malformed model keys (EntrypointValidationError) are the
        # caller's exit 2.
        _print_exception(exc)
        _debug_traceback(args)
        return 1
    except ValidationError as exc:
        # BACKSTOP for a raw ValidationError from any seam the _EngineFault
        # envelope does not wrap (the execution seam is enveloped above):
        # still an ENGINE fault, never usage -- every caller-originating
        # pydantic failure is classified upstream (--options by
        # _parse_options, init config by registry.create's ConfigError wrap,
        # flags by argparse) -- the CLI twin of the server's scrubbed 500.
        # Hence exit 1 (engine/runtime failure), not 2 (usage), and the
        # OPERATOR audience: this line reports a fault the CLI user did not
        # cause and lands in CI logs and bug reports.
        #
        # str(exc) is never printed: it echoes the offending input_value
        # verbatim, leaking a SecretStr-bound credential in plaintext, and
        # ValidationError IS a ValueError, so without this arm it would fall
        # into the ValueError family below and print the secret.
        _print_error(
            sanitized_validation_message(
                exc,
                prefix="Engine produced an invalid model (engine fault)",
            )
        )
        _debug_traceback(args)
        return 1
    except (ConfigError, DiscoveryError, UnsupportedFeatureError, ValueError) as exc:
        # UnsupportedFeatureError is named explicitly: it is a StructuredError,
        # NOT a ValueError, yet a strict-mode rejection of an unsupported /
        # unreachable request parameter is a usage error (exit 2) exactly like
        # the ValueError family -- without this it fell into the generic
        # runtime-failure branch below and scripts read a caller mistake as an
        # internal failure (exit 1).
        _print_exception(exc)
        _debug_traceback(args)
        return 2
    except TranscriptionError as exc:
        _print_exception(exc)
        _debug_traceback(args)
        return 1
    except Exception as exc:  # noqa: BLE001
        _print_exception(exc)
        _debug_traceback(args)
        return 1


def _debug_traceback(args: argparse.Namespace) -> None:
    """Emit a stack trace to stderr when ``--debug`` is set.

    ``--debug`` promises a stack trace on any error path, but the trace was
    previously printed only in the final
    ``except Exception`` branch, so an error caught by a named branch (for example, an
    engine-internal failure surfacing as a ``ValueError`` from ``_transcribe``)
    had no trace even with ``--debug``. Routing every branch through
    this helper makes the flag uniform. ``getattr`` keeps it safe for the
    argparse error paths where ``--debug`` may be absent from the namespace.
    When the active chain carries a pydantic ``ValidationError`` the trace is
    the scrubbed one-line chain summary, never ``traceback.print_exc()`` --
    the native formatter re-renders every link's raw message, ``input_value``
    echo included, and a diagnostics flag must not double as a
    credential-leak opt-in.

    Args:
        args: Parsed CLI arguments.

    Returns:
        None.

    Raises:
        None.
    """
    if not getattr(args, "debug", False):
        return
    exc = sys.exc_info()[1]
    if exc is not None and chain_has_validation_error(exc):
        # traceback.print_exc() would re-render the raw chained
        # ValidationError -- input_value echo included -- undoing the scrub
        # every normal CLI path applies. Debug mode is a diagnostics opt-in,
        # not a credential-leak opt-in: print the scrubbed chain summary.
        _print_error(safe_exception_summary(exc))
        return
    traceback.print_exc()


def _ensure_utf8_stream(stream: IO[str]) -> None:
    """Reconfigure a text stream to UTF-8 when it is not already UTF-8.

    On Windows a stdout/stderr redirected to a file or pipe defaults to the
    process ANSI code page (for example, cp1252) with ``errors="strict"``; printing a
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


def _print_exception(exc: BaseException) -> None:
    """Print a caught exception to stderr through the ONE safe boundary.

    Every ``except`` arm in :func:`main` reports through this helper -- never
    bare ``str(exc)``. The distinction ``--debug`` cannot make: debug output
    is opt-in, but the NORMAL error line prints unconditionally, so a wrapper
    that copied a pydantic ``ValidationError``'s (possibly truncated) input
    echo into its own message (``raise RuntimeError(f"...: {exc}") from
    exc``) would leak a mis-placed credential to stderr before any traceback
    logic runs. Semantics:

    * chain WITHOUT a ``ValidationError``: the exception's own authored
      message (via :func:`~standard_asr.runtime.redaction.safe_str`, so a
      raising ``__str__`` degrades to a placeholder instead of crashing
      error reporting);
    * chain WITH one: the input-echo-free one-line summary
      (:func:`~standard_asr.runtime.redaction.safe_exception_summary`) --
      sanitized loc/msg for the ``ValidationError`` link, every other
      link's own message kept.

    Args:
        exc: The caught exception to report.

    Returns:
        None.

    Raises:
        None.
    """
    if chain_has_validation_error(exc):
        _print_error(safe_exception_summary(exc))
        return
    message = safe_str(exc)
    if message is None:
        message = f"{safe_type_name(exc)}: <exception str() failed>"
    _print_error(message)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
