"""Shared helpers for assembling model components from lightweight specs."""

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from torch import nn

__all__ = ["ModuleSpec", "broadcast", "build_module"]


# A module spec: a ready-made module, a registry key, a factory, or a
# ``(key_or_factory, params)`` pair carrying keyword arguments.
ModuleSpec = nn.Module | str | Callable[..., nn.Module] | Sequence[Any]


def broadcast(value: int | Sequence[int], length: int, name: str) -> list[int]:
    """Broadcast an int to a list of ``length``, or validate a given sequence."""
    values = [value] * length if isinstance(value, int) else list(value)
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    return values


def build_module(
    spec: ModuleSpec,
    registry: Mapping[str, Callable[..., nn.Module]] | None = None,
    *args: Any,
    **kwargs: Any,
) -> nn.Module:
    """Resolve a module spec into an instantiated :class:`~torch.nn.Module`.

    A spec is one of:

    - an :class:`~torch.nn.Module` instance -- returned unchanged;
    - a registry key such as ``"mlp"`` (needs ``registry``);
    - a factory: an ``nn.Module`` subclass, or any callable returning one --
      e.g. ``functools.partial(TimmEncoder, model_name=...)`` or a Hydra
      ``_partial_: true`` node;
    - a ``(key_or_factory, params)`` pair whose ``params`` mapping supplies
      keyword arguments, e.g. ``("cos", {"num_components": 64})``.

    ``*args`` / ``**kwargs`` are injected by the caller (e.g. ``in_features``)
    and passed alongside the spec's ``params``, so a key given in both is an
    error rather than a silent override.

    Sequences and mappings are matched structurally, so OmegaConf containers
    coming straight from a config work without conversion.
    """
    if isinstance(spec, nn.Module):
        return spec

    target: Any = spec
    params: Mapping[str, Any] = {}
    # A (key_or_factory, params) pair. `str` is itself a Sequence, so it is
    # excluded explicitly.
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
            raise KeyError(f"unknown key {target!r}; expected one of {sorted(registry)}")
        target = registry[target]

    if not callable(target):
        raise ValueError(
            "spec must be an nn.Module, a registry key, a factory, or a "
            f"(key/factory, params) pair, got {spec!r}"
        )
    return target(*args, **params, **kwargs)
