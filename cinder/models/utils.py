"""Build model components from lightweight specs."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

from torch import nn

__all__ = ["ModuleSpec", "build_module", "check_sizes"]


# A module, registry key, factory, or (key_or_factory, keyword_arguments) pair.
ModuleSpec: TypeAlias = nn.Module | str | Callable[..., nn.Module] | Sequence[Any]


def build_module(
    spec: ModuleSpec,
    registry: Mapping[str, Callable[..., nn.Module]] | None = None,
    *args: Any,
    **kwargs: Any,
) -> nn.Module:
    """Build a module from an instance, registry key, factory, or spec pair.

    Instances are returned unchanged. A ``(key_or_factory, params)`` pair adds
    keyword arguments, for example ``("cosine", {"num_components": 64})``. The
    caller's ``args`` and ``kwargs`` are passed alongside ``params``, and
    duplicate keywords raise a ``TypeError``. Factories include module classes,
    ``functools.partial`` objects and Hydra ``_partial_`` nodes; the sequence
    and mapping checks also accept OmegaConf containers.
    """
    if isinstance(spec, nn.Module):
        return spec

    target, params = spec, {}
    # Strings are sequences too, but they are registry keys.
    if not isinstance(spec, str) and isinstance(spec, Sequence):
        if len(spec) != 2 or not isinstance(spec[1], Mapping):
            raise ValueError(
                f"a sequence spec must be a (key/factory, params) pair, got {spec!r}"
            )
        target, params = spec

    if isinstance(target, str):
        if registry is None:
            raise ValueError(f"no registry given to resolve the key {target!r}")
        if target not in registry:
            raise KeyError(
                f"unknown key {target!r}; expected one of {sorted(registry)}"
            )
        target = registry[target]

    if not callable(target):
        raise TypeError(
            "spec must be an nn.Module, a registry key, a factory, or a "
            f"(key/factory, params) pair, got {spec!r}"
        )
    module = target(*args, **params, **kwargs)
    if not isinstance(module, nn.Module):
        raise TypeError(
            f"module factory must return an nn.Module, got {type(module).__name__}"
        )
    return module


def check_sizes(module: nn.Module, name: str, **sizes: Any) -> None:
    """Check the size attributes a module exposes, skipping absent ones.

    Ready-made modules and custom factories may ignore the sizes they are given;
    a check at construction fails more clearly than a later forward pass.
    """
    for attr, expected in sizes.items():
        actual = getattr(module, attr, expected)
        if actual != expected:
            raise ValueError(f"{name}.{attr} must be {expected}, got {actual}")
