"""Plugin discovery utilities for Standard ASR models.

This module implements discovery for entry points defined under the
``standard_asr.models`` group. Entry point names must follow the format
``<engine_id>/<model_name>`` where ``model_name`` may be empty to denote the
engine's default model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from importlib.metadata import EntryPoint, EntryPoints, entry_points
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Protocol,
    TYPE_CHECKING,
    final,
)

from .exceptions import EntrypointValidationError, FactoryLoadError

if TYPE_CHECKING:  # pragma: no cover
    from .asr_interface import StandardASR


logger = logging.getLogger(__name__)

ENTRYPOINT_GROUP: str = "standard_asr.models"


class ASRFactory(Protocol):
    """Callable protocol describing Standard ASR model factories."""

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
        raise EntrypointValidationError(
            f"engine_id must not contain '/' (got {engine_id!r})"
        )
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
        raise EntrypointValidationError(
            f"model_name must not contain '/' (got {model_name!r})"
        )
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
    """Immutable specification describing a discovered ASR model entry point."""

    key: str
    engine_id: str
    model_name: str
    entry_point: EntryPoint

    def load_factory(self) -> ASRFactory:
        """Load and validate the target callable for this entry point.

        Returns:
            A callable that constructs a ``StandardASR`` implementation.

        Raises:
            FactoryLoadError: If loading fails or the target is not callable.
        """

        try:
            target = self.entry_point.load()
        except Exception as exc:  # noqa: BLE001
            message = f"Failed to load entry point target for {self.key!r}: {exc!r}"
            raise FactoryLoadError(message) from exc
        if not callable(target):
            raise FactoryLoadError(
                f"Entry point target for {self.key!r} is not callable (got {type(target).__name__})."
            )
        return target  # type: ignore[return-value]


@final
class ModelRegistry:
    """Immutable registry of all discovered Standard ASR model factories."""

    def __init__(self, specs: Mapping[str, ModelSpec]) -> None:
        self._specs: dict[str, ModelSpec] = dict(specs)

    def names(self) -> list[str]:
        """Return discovered model keys sorted lexicographically."""

        return sorted(self._specs.keys())

    def by_engine(self, engine_id: str) -> list[str]:
        """Return model keys associated with *engine_id*."""

        return sorted(
            key for key, spec in self._specs.items() if spec.engine_id == engine_id
        )

    def spec(self, name: str) -> ModelSpec:
        """Return the :class:`ModelSpec` for *name* (``engine/model``)."""

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
        """Retrieve the callable factory for *name*."""

        return self.spec(name).load_factory()

    def create(self, name: str, /, *args: Any, **kwargs: Any) -> "StandardASR":
        """Instantiate a Standard ASR implementation for *name*.

        Args:
            name: Entry point key in ``engine/model`` format.
            *args: Positional arguments forwarded to the factory.
            **kwargs: Keyword arguments forwarded to the factory.

        Returns:
            Instance created by the target callable.
        """

        factory = self.get_factory(name)
        return factory(*args, **kwargs)

    def __len__(self) -> int:  # pragma: no cover
        return len(self._specs)

    def __iter__(self) -> Iterator[str]:  # pragma: no cover
        return iter(self.names())

    def __repr__(self) -> str:  # pragma: no cover
        return f"ModelRegistry({self.names()!r})"


def _gather_entry_points(eps: Iterable[EntryPoint] | None = None) -> EntryPoints:
    """Return entry points for the Standard ASR models group."""

    if eps is not None:
        return EntryPoints(list(eps))
    return entry_points(group=ENTRYPOINT_GROUP)


def discover_models(
    eps: Iterable[EntryPoint] | None = None,
    *,
    strict: bool = False,
    on_conflict: str = "warn_keep_first",
) -> ModelRegistry:
    """Discover installed Standard ASR model entry points.

    Args:
        eps: Optional iterable of entry points to inspect (primarily for testing).
        strict: When ``True`` invalid entry points cause an immediate error.
        on_conflict: Conflict strategy - ``warn_keep_first`` (default) keeps the first
            discovered entry; ``replace`` swaps to the latest entry while warning.

    Returns:
        Populated :class:`ModelRegistry` instance.

    Raises:
        EntrypointValidationError: When ``strict`` is true and invalid entries exist.
        ValueError: If ``on_conflict`` is not a supported value.
    """

    if on_conflict not in {"warn_keep_first", "replace"}:
        raise ValueError("on_conflict must be 'warn_keep_first' or 'replace'.")

    found = _gather_entry_points(eps)
    logger.debug(
        "Discovering Standard ASR models: %d entry points located.", len(found)
    )

    specs: MutableMapping[str, ModelSpec] = {}
    errors: list[str] = []

    for ep in found:
        if ep.group != ENTRYPOINT_GROUP:
            logger.debug("Skipping entry point with unexpected group: %r", ep)
            continue
        try:
            engine_id, model_name = parse_entrypoint_name(ep.name)
            key = f"{engine_id}/{model_name}"
            spec = ModelSpec(
                key=key, engine_id=engine_id, model_name=model_name, entry_point=ep
            )
        except EntrypointValidationError as exc:
            dist = getattr(ep, "dist", None)
            dist_label = f" (dist={dist})" if dist is not None else ""
            message = f"Invalid entry point name {ep.name!r}{dist_label}: {exc}"
            if strict:
                errors.append(message)
            else:
                logger.warning(message)
            continue

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

    if strict and errors:
        joined = "\n".join(f"- {message}" for message in errors)
        raise EntrypointValidationError("Invalid entry points detected:\n" + joined)

    registry = ModelRegistry(specs)
    logger.info("Discovered %d Standard ASR model(s).", len(registry))
    return registry


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
