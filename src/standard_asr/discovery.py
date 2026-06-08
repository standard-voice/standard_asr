# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Plugin discovery system for Standard ASR models in the current Python environment.

**Quick Start:**

    >>> from standard_asr import discover_models
    >>> registry = discover_models()
    >>> asr = registry.create("faster-whisper/large-v3")

**Key Concepts:**

- **Entry Point Group:** ``standard_asr.models``
- **Entry Point Name:** ``<engine_id>/<model_name>`` (e.g., ``faster-whisper/large-v3``)
- **ModelRegistry:** Container of all discovered ASR engine factories.
- **ModelSpec:** Metadata for a single entry point.

**For Plugin Authors:** See ``docs/for_asr_dev/plugin_entrypoints.md``.
"""

from __future__ import annotations

import inspect
import logging
import re
import typing
from dataclasses import dataclass
from importlib.metadata import EntryPoint, EntryPoints, entry_points
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Protocol,
    final,
)

from .exceptions import EntrypointValidationError, FactoryLoadError

if TYPE_CHECKING:  # pragma: no cover
    from .asr_interface import StandardASR


logger = logging.getLogger(__name__)

ENTRYPOINT_GROUP: str = "standard_asr.models"


class ASRFactory(Protocol):
    """Callable that creates a ``StandardASR`` instance.

    Plugin entry points must resolve to a callable matching this protocol.
    Typically a function or class constructor that accepts optional configuration.

    Example:
        >>> def create_asr(**kwargs) -> StandardASR:
        ...     return MyASREngine(**kwargs)
    """

    def __call__(self, *args: Any, **kwargs: Any) -> "StandardASR": ...


_ENGINE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*\Z")
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+%:-]*\Z")


def pep503_normalize(name: str) -> str:
    """Normalize *name* according to PEP 503 rules.

    Args:
        name: Original distribution name.

    Returns:
        Normalized name in lowercase with runs of ``[-_.]`` replaced by ``-``.
    """

    return re.sub(r"[-_.]+", "-", name.lower())


def _validate_engine_id(engine_id: str) -> None:
    """Validate and log guidance for an engine identifier.

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
    canonical = pep503_normalize(engine_id)
    if canonical != engine_id:
        logger.info(
            "engine_id %r is not PEP 503 normalized. Recommended form: %r.",
            engine_id,
            canonical,
        )


def _validate_model_name(model_name: str) -> None:
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


def validate_engine_id(engine_id: str) -> None:
    """Validate an engine identifier.

    Args:
        engine_id: Engine identifier string.

    Returns:
        None.

    Raises:
        EntrypointValidationError: If the engine identifier is invalid.
    """
    _validate_engine_id(engine_id)


def validate_model_name(model_name: str) -> None:
    """Validate a model name.

    Args:
        model_name: Model name string (may be empty for defaults).

    Returns:
        None.

    Raises:
        EntrypointValidationError: If the model name is invalid.
    """
    _validate_model_name(model_name)


def parse_entrypoint_name(name: str) -> tuple[str, str]:
    """Parse an entry point name into ``(engine_id, model_name)``.

    Args:
        name: Entry point name declared in pyproject.toml.

    Returns:
        Tuple containing engine identifier and model name (possibly empty).

    Raises:
        EntrypointValidationError: If the name does not meet formatting rules.
    """

    if "/" not in name:
        engine_id, model_name = name, ""
    else:
        parts = name.split("/")
        if len(parts) != 2:
            raise EntrypointValidationError(
                f"Invalid entry point name {name!r}. Use '<engine_id>/<model_name>'."
            )
        engine_id, model_name = parts[0], parts[1]
    _validate_engine_id(engine_id)
    _validate_model_name(model_name)
    return engine_id, model_name


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Metadata for a discovered ASR model entry point.

    Attributes:
        key: Full entry point name (``engine_id/model_name``).
        engine_id: Engine identifier (e.g., ``faster-whisper``).
        model_name: Model preset name (e.g., ``large-v3``), or empty for default.
        entry_point: The underlying ``importlib.metadata.EntryPoint`` object.

    Note:
        Instances are created by ``discover_models()``. Use ``load_factory()``
        to get the callable that constructs the ASR engine.
    """

    key: str
    engine_id: str
    model_name: str
    entry_point: EntryPoint

    def load_factory(self) -> ASRFactory:
        """Load the factory callable for this entry point.

        Returns:
            Callable that creates a ``StandardASR`` instance when invoked.

        Raises:
            FactoryLoadError: Entry point failed to load or is not callable.
        """
        try:
            target = self.entry_point.load()
        except Exception as exc:  # noqa: BLE001
            message = f"Failed to load entry point target for {self.key!r}: {exc!r}"
            raise FactoryLoadError(message) from exc
        if not callable(target):
            raise FactoryLoadError(
                f"Entry point target for {self.key!r} is not callable "
                f"(got {type(target).__name__})."
            )
        return target  # type: ignore[return-value]

    def engine_class(self) -> type["StandardASR"]:
        """Resolve the engine **class** without instantiating it.

        This enables reading class-level ``ClassVar`` metadata
        (``declared_capabilities``, ``properties``, ``provider_params_type``)
        without calling the factory -- which spec §3.1 / §C requires to be
        possible "without instantiation or authentication". Instantiating a
        cloud engine would force credential resolution and a heavy ``__init__``,
        turning an unauthenticated metadata read into a denial-of-service vector.

        The entry-point target is *loaded* (its module is imported) but never
        *called*. Resolution rules:

        - If the target is itself a class, it is returned directly.
        - If the target is a function (the common factory pattern), its return
          type annotation is resolved (via :func:`typing.get_type_hints`) and,
          if it names a concrete class, that class is returned.

        Returns:
            The engine class declaring the static metadata.

        Raises:
            FactoryLoadError: The target failed to load, or the class cannot be
                determined without calling the factory (e.g. a factory with no
                concrete return annotation). Callers SHOULD fall back to
                instantiation only when they explicitly accept that cost.
        """
        target = self.load_factory()
        if inspect.isclass(target):
            return self._ensure_engine_class(target)

        try:
            hints = typing.get_type_hints(target)
        except Exception as exc:  # noqa: BLE001
            raise FactoryLoadError(
                f"Cannot resolve the engine class for {self.key!r} without "
                f"instantiation: failed to read the factory's type hints ({exc!r}). "
                "Annotate the factory with a concrete engine return type."
            ) from exc

        returned = hints.get("return")
        if inspect.isclass(returned):
            return self._ensure_engine_class(returned)

        raise FactoryLoadError(
            f"Cannot resolve the engine class for {self.key!r} without "
            "instantiation. The entry point is a factory whose return annotation "
            f"is not a concrete class (got {returned!r}). Either expose the engine "
            "class directly as the entry point, or annotate the factory with the "
            "concrete engine return type so its static metadata is readable "
            "without calling it."
        )

    def _ensure_engine_class(self, cls: type) -> type["StandardASR"]:
        """Validate ``cls`` is recognisably an engine class, then cast.

        ``StandardASR`` is a ``runtime_checkable`` :class:`typing.Protocol` with
        non-method (``ClassVar``) members, so ``issubclass`` against it raises
        ``TypeError``; engines are also structural and need not subclass
        :class:`~standard_asr.asr_interface.EngineBase`. We therefore duck-type a
        **minimal** signal: the class must expose at least one member of the
        engine surface. This converts a misconfigured entry point that resolves
        to a wholly unrelated class into a clear
        :class:`~standard_asr.exceptions.FactoryLoadError` instead of a later
        ``AttributeError``.

        The check is intentionally permissive: per-attribute validation (e.g. a
        class that has ``transcribe`` but is missing ``declared_capabilities`` /
        ``properties``) is the job of the compliance suite, which emits precise
        diagnostics; and metadata readers consume these attributes defensively
        via ``getattr``, so a degenerate-but-intentional engine is still
        tolerated. We only reject classes that look nothing like an engine.

        Args:
            cls: The resolved candidate engine class.

        Returns:
            ``cls``, typed as a Standard ASR engine class.

        Raises:
            FactoryLoadError: If ``cls`` exposes none of the engine surface.
        """
        markers = ("transcribe", "declared_capabilities", "properties", "supports")
        if not any(hasattr(cls, marker) for marker in markers):
            raise FactoryLoadError(
                f"Entry point {self.key!r} resolves to {cls.__name__!r}, which does not "
                "expose any Standard ASR engine surface (none of "
                f"{', '.join(markers)}). Check the entry-point target."
            )
        return typing.cast("type[StandardASR]", cls)


@final
class ModelRegistry:
    """Container for discovered ASR engine factories.

    ModelRegistry holds the results of plugin discovery and provides methods to
    list, query, and instantiate ASR engines. It does **not** perform discovery
    itself—use ``discover_models()`` to create a populated registry.

    **Typical Usage:**

        >>> from standard_asr import discover_models
        >>> registry = discover_models()
        >>>
        >>> # List all available models
        >>> registry.names()  # ['faster-whisper/large-v3', 'whisper/base', ...]
        >>>
        >>> # Create an ASR instance
        >>> asr = registry.create("faster-whisper/large-v3", device="cuda")
        >>> result = asr.transcribe(audio)

    **Key Methods:**

    - ``names()``: List all discovered model keys.
    - ``by_engine(engine_id)``: List models for a specific engine.
    - ``create(name, **kwargs)``: Instantiate an ASR engine.
    - ``spec(name)``: Get metadata for a model.

    Note:
        Use ``discover_models()`` to create a ModelRegistry. Do not instantiate directly
        unless you're providing custom entry points for testing.
    """

    def __init__(
        self,
        specs: Mapping[str, ModelSpec],
        *,
        shadowed_engine_ids: set[str] | None = None,
    ) -> None:
        """Initialize with a mapping of model specs (internal use).

        Args:
            specs: Mapping of ``engine_id/model_name`` keys to specs.
            shadowed_engine_ids: Engine ids contributed by more than one
                distribution (IC.2 identity collision). Routing on these is
                ambiguous; consumers may surface or reject them.
        """
        self._specs: dict[str, ModelSpec] = dict(specs)
        self._shadowed_engine_ids: set[str] = set(shadowed_engine_ids or set())

    @property
    def shadowed_engine_ids(self) -> set[str]:
        """Engine ids provided by more than one distribution (IC.2).

        Returns:
            A copy of the set of ambiguous engine ids. Empty when discovery
            found no engine-identity collisions.
        """
        return set(self._shadowed_engine_ids)

    def names(self) -> list[str]:
        """List all discovered model keys, sorted alphabetically.

        Returns:
            List of model keys (e.g., ``['faster-whisper/large-v3', 'whisper/base']``).
        """
        return sorted(self._specs.keys())

    def by_engine(self, engine_id: str) -> list[str]:
        """List all model keys for a specific engine.

        Args:
            engine_id: Engine identifier (e.g., ``faster-whisper``).

        Returns:
            List of matching model keys, sorted alphabetically.

        Example:
            >>> registry.by_engine("faster-whisper")
            ['faster-whisper/', 'faster-whisper/large-v3', 'faster-whisper/small']
        """
        return sorted(key for key, spec in self._specs.items() if spec.engine_id == engine_id)

    def spec(self, name: str) -> ModelSpec:
        """Get metadata for a model.

        Args:
            name: Model key in ``engine_id/model_name`` format.

        Returns:
            ``ModelSpec`` containing entry point metadata.

        Raises:
            EntrypointValidationError: Model not found or invalid name format.
        """
        engine_id, model_name = parse_entrypoint_name(name)
        key = f"{engine_id}/{model_name}"
        try:
            return self._specs[key]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise EntrypointValidationError(
                f"Model {key!r} not found. Available models: {available}"
            ) from exc

    def get_factory(self, name: str) -> ASRFactory:
        """Get the factory callable for a model (without instantiating).

        Args:
            name: Model key in ``engine_id/model_name`` format.

        Returns:
            Callable that creates a ``StandardASR`` instance.

        Raises:
            EntrypointValidationError: Model not found.
            FactoryLoadError: Entry point failed to load.
        """
        return self.spec(name).load_factory()

    def engine_class(self, name: str) -> type["StandardASR"]:
        """Resolve a model's engine class without instantiating it.

        Use this to read class-level metadata (``declared_capabilities``,
        ``properties``, ``provider_params_type``) for discovery, UI generation,
        and REST endpoints without paying the cost (or auth requirements) of
        constructing the engine. See :meth:`ModelSpec.engine_class`.

        Args:
            name: Model key in ``engine_id/model_name`` format.

        Returns:
            The engine class.

        Raises:
            EntrypointValidationError: Model not found.
            FactoryLoadError: Entry point failed to load, or the class cannot be
                determined without calling the factory.
        """
        return self.spec(name).engine_class()

    def create(self, name: str, /, *args: Any, **kwargs: Any) -> "StandardASR":
        """Create an ASR engine instance.

        This is the **primary method** for instantiating ASR engines. It loads the
        factory and invokes it with the provided arguments.

        Args:
            name: Model key (e.g., ``"faster-whisper/large-v3"``).
            *args: Positional arguments passed to the factory.
            **kwargs: Keyword arguments passed to the factory (e.g., ``device="cuda"``).

        Returns:
            ``StandardASR`` instance ready for transcription.

        Raises:
            EntrypointValidationError: Model not found.
            FactoryLoadError: Factory failed to load or execute.

        Example:
            >>> asr = registry.create("faster-whisper/large-v3", device="cuda")
            >>> result = asr.transcribe(audio)
        """
        factory = self.get_factory(name)
        engine_id = self.spec(name).engine_id
        if engine_id in self._shadowed_engine_ids:
            # IC.2: surface the ambiguity again at the point of use, not only at
            # discovery -- routing to a shadowed engine_id is never silent.
            logger.warning(
                "Creating model %r whose engine_id %r is provided by more than one "
                "distribution; config.engine routing is ambiguous. Install only one "
                "provider for this engine_id.",
                name,
                engine_id,
            )
        return factory(*args, **kwargs)

    def __len__(self) -> int:  # pragma: no cover
        return len(self._specs)

    def __iter__(self) -> Iterator[str]:  # pragma: no cover
        return iter(self.names())

    def __repr__(self) -> str:  # pragma: no cover
        return f"ModelRegistry({self.names()!r})"


def _gather_entry_points(eps: Iterable[EntryPoint] | None = None) -> EntryPoints:
    """Gather entry points from the standard_asr.models group (internal)."""
    if eps is not None:
        return EntryPoints(list(eps))
    return entry_points(group=ENTRYPOINT_GROUP)


def discover_models(
    eps: Iterable[EntryPoint] | None = None,
    *,
    strict: bool = False,
    on_conflict: str = "warn_keep_first",
) -> ModelRegistry:
    """Discover all installed ASR plugins and return a registry.

    This is the **main entry point** for the plugin discovery system. It scans
    the ``standard_asr.models`` entry point group and returns a ``ModelRegistry``
    containing all discovered ASR engine factories.

    Args:
        eps: Custom entry points for testing. Leave ``None`` for normal discovery.
        strict: If ``True``, raise on invalid entry points. Default: ``False`` (warn only).
        on_conflict: How to handle duplicate model keys:

            - ``"warn_keep_first"``: Keep first, warn about duplicates (default).
            - ``"replace"``: Use latest, warn about replacement.

    Returns:
        ``ModelRegistry`` containing all discovered models.

    Raises:
        EntrypointValidationError: (strict mode) Invalid entry points detected.
        ValueError: Unknown ``on_conflict`` value.

    Example:
        >>> from standard_asr import discover_models
        >>> registry = discover_models()
        >>> print(registry.names())
        ['faster-whisper/large-v3', 'whisper/base', ...]
        >>> asr = registry.create("faster-whisper/large-v3")
    """

    if on_conflict not in {"warn_keep_first", "replace"}:
        raise ValueError("on_conflict must be 'warn_keep_first' or 'replace'.")

    found = _gather_entry_points(eps)
    logger.debug("Discovering Standard ASR models: %d entry points located.", len(found))

    specs: MutableMapping[str, ModelSpec] = {}
    errors: list[str] = []
    # engine_id -> set of distribution names that contribute it (IC.2).
    engine_dists: dict[str, set[str]] = {}

    for ep in found:
        if ep.group != ENTRYPOINT_GROUP:
            logger.debug("Skipping entry point with unexpected group: %r", ep)
            continue
        try:
            engine_id, model_name = parse_entrypoint_name(ep.name)
            key = f"{engine_id}/{model_name}"
            spec = ModelSpec(key=key, engine_id=engine_id, model_name=model_name, entry_point=ep)
        except EntrypointValidationError as exc:
            dist = getattr(ep, "dist", None)
            dist_label = f" (dist={dist})" if dist is not None else ""
            message = f"Invalid entry point name {ep.name!r}{dist_label}: {exc}"
            if strict:
                errors.append(message)
            else:
                logger.warning(message)
            continue

        engine_dists.setdefault(engine_id, set()).add(_dist_name(ep))

        if key in specs and on_conflict == "warn_keep_first":
            logger.warning(
                "Duplicate model key %r detected. Keeping %r; ignoring %r.",
                key,
                specs[key].entry_point,
                ep,
            )
            continue
        if key in specs and on_conflict == "replace":
            logger.warning(
                "Duplicate model key %r detected. Replacing %r with %r.",
                key,
                specs[key].entry_point,
                ep,
            )

        specs[key] = spec

    # IC.2 engine-identity collision: the same engine_id MUST come from a single
    # distribution. Two different distributions both claiming engine_id="whisper"
    # (even with distinct model names) make ``config.engine`` routing ambiguous.
    shadowed: dict[str, set[str]] = {
        engine_id: dists for engine_id, dists in engine_dists.items() if len(dists) > 1
    }
    for engine_id, dists in shadowed.items():
        message = (
            f"Engine-identity collision: engine_id {engine_id!r} is provided by "
            f"multiple distributions ({', '.join(sorted(dists))}). "
            "config.engine routing is ambiguous; install only one provider for "
            "this engine_id, or have authors choose distinct engine_ids."
        )
        if strict:
            errors.append(message)
        else:
            logger.warning(message)

    if strict and errors:
        joined = "\n".join(f"- {message}" for message in errors)
        raise EntrypointValidationError("Invalid entry points detected:\n" + joined)

    registry = ModelRegistry(specs, shadowed_engine_ids=set(shadowed))
    logger.info("Discovered %d Standard ASR model(s).", len(registry))
    return registry


def _dist_name(ep: EntryPoint) -> str:
    """Return the PEP 503-normalized distribution name for *ep*.

    Args:
        ep: The entry point to inspect.

    Returns:
        The normalized distribution name, or ``"<unknown>"`` when the entry
        point has no associated distribution (e.g. test-injected entry points).
    """
    dist = getattr(ep, "dist", None)
    name = getattr(dist, "name", None) if dist is not None else None
    return pep503_normalize(name) if name else "<unknown>"


__all__ = [
    "ASRFactory",
    "ENTRYPOINT_GROUP",
    "ModelRegistry",
    "ModelSpec",
    "discover_models",
    "parse_entrypoint_name",
    "pep503_normalize",
]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    registry = discover_models()
    print("Discovered models:")
    for name in registry.names():
        print(f" - {name}")
