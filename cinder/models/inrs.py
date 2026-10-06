"""Implicit neural representations: coordinates ``(*, C)`` to values ``(*, D)``.

Ported from ``neurofield.models`` (``mlp``, ``siren``, ``finer``, ``gauss``,
``wire``, ``rff``, ``pemlp``, ``instant_ngp`` and ``futon``) with unchanged
implementations, except that :class:`WIRE` is neurofield's real-valued
``RealWIRE``, ``pemlp`` and ``instant_ngp`` contribute only their encodings,
:class:`PositionalEncoding` and :class:`HashEncoding`, the polynomial bases are
omitted, and :class:`FUTONEncoding` factors ``combiner(basis(x))`` out of
:class:`FUTON` so that the modulators can reuse it.
:data:`INRS` lists the networks CINDER builds as
``inr(in_features=..., out_features=...)``.

FUTON evaluates ``decoder(combiner(basis(x)))`` in three stages:

1. A **basis** maps coordinates ``(N, C)`` in ``[-1, 1]`` to one feature tensor
   ``(N, K_c)`` per axis, where ``K_c = num_components[c]``.
2. A **combiner** fuses the per-axis features into one ``(N, out_features)``
   tensor.
3. A **decoder**, ``nn.Linear`` or :class:`MLP`, maps the fused features to the
   output.

All bases share three options:

- ``num_components``: an int shared by all axes, or one count per axis.
- ``normalize``: L2-normalize each feature vector (not each basis function).
- ``grid_size``: tabulate each axis on ``linspace(-1, 1, size)`` at
  construction. Coordinates on the grid read the table and others are evaluated
  directly. Table lookups carry no coordinate gradients, so leave this ``None``
  when gradients with respect to ``x`` are needed.

The local bases :class:`TriangleBasis` and :class:`LanczosBasis` default to
sparse mode: each axis returns an ``(N, K_c)`` :class:`RCSMatrix` that stores
only the few nonzero taps per point, and the combiners contract it directly.
"""

import math
from collections.abc import Callable, Sequence
from functools import reduce
from itertools import product
from operator import mul
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rcs_matrix import RCSMatrix
from .utils import ModuleSpec, build_module

__all__ = [
    "MLP",
    "ReLULayer",
    "SineLayer",
    "SIREN",
    "FinerLayer",
    "FINER",
    "GaussLayer",
    "Gauss",
    "GaborLayer",
    "WIRE",
    "RFFEncoding",
    "RFF",
    "PositionalEncoding",
    "HashEncoding",
    "CosineBasis",
    "SincBasis",
    "TriangleBasis",
    "LanczosBasis",
    "HadamardCombiner",
    "CPCombiner",
    "TRCombiner",
    "FUTONEncoding",
    "FUTON",
    "BASES",
    "COMBINERS",
    "DECODERS",
    "INRS",
]


# MLP ----------------------------------------------------------------------------------


class ReLULayer(nn.Module):
    """Linear projection followed by ReLU: ``(..., in_features) -> (..., out_features)``."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.linear(x))


class MLP(nn.Module):
    """MLP with standard or custom hidden layers and a linear output layer.

    Maps ``(..., in_features)`` to ``(..., out_features)``.
    By default, hidden layers are linear projections followed by ReLU.
    Subclasses apply custom initialization after ``super().__init__``.

    Args:
        in_features: Input width.
        out_features: Output width.
        hidden_features: Hidden width; ``None`` defaults to ``in_features``.
        hidden_layers: Number of hidden layers; zero gives a single linear layer
            when ``layer_class`` is omitted. Custom layers require at least one.
        activation: Callable applied after each standard hidden linear layer;
            ``None`` selects ReLU. Module instances are shared across hidden
            layers. Cannot be combined with ``layer_class``.
        layer_class: Complete hidden-layer class receiving input/output widths
            and ``**layer_kwargs``. No additional activation is applied.
        output_activation: Callable applied after the linear output layer;
            ``None`` leaves the output unchanged.
        **layer_kwargs: Constructor arguments for the hidden layers only.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int | None = None,
        hidden_layers: int = 2,
        activation: Callable[[Tensor], Tensor] | None = None,
        *,
        layer_class: type[nn.Module] | None = None,
        output_activation: Callable[[Tensor], Tensor] | None = None,
        **layer_kwargs: Any,
    ) -> None:
        super().__init__()

        min_hidden_layers = 0 if layer_class is None else 1
        if hidden_layers < min_hidden_layers:
            raise ValueError(
                f"hidden_layers must be >= {min_hidden_layers}, got {hidden_layers}"
            )
        if layer_class is None:
            layer_class = nn.Linear
            activation = torch.relu if activation is None else activation
        elif activation is not None:
            raise ValueError("Custom layer_class supplies its own activation.")

        if hidden_features is None:
            hidden_features = in_features

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.activation = activation if activation is not None else nn.Identity()

        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            layers.append(layer_class(in_features, hidden_features, **layer_kwargs))
            in_features = hidden_features
        layers.append(nn.Linear(in_features, out_features))
        self.layers = nn.ModuleList(layers)

        self.output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.output_activation(self.layers[-1](x))


# SIREN --------------------------------------------------------------------------------


class SineLayer(nn.Module):
    """Sine layer ``sin(omega * linear(x))`` used by :class:`SIREN`.

    Maps ``(..., in_features)`` to ``(..., out_features)``; ``omega`` controls
    frequency.
    """

    def __init__(
        self, in_features: int, out_features: int, omega: float = 30.0
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.omega = omega

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega * self.linear(x))


class SIREN(MLP):
    """Sinusoidal representation network (Sitzmann et al., NeurIPS 2020).

    An :class:`MLP` with sine hidden layers, a linear output layer,
    and SIREN weight initialization. Biases keep PyTorch's defaults.

    Args:
        in_features: Number of input coordinates.
        out_features: Number of output channels.
        hidden_features: Hidden width.
        hidden_layers: Number of sine layers; must be at least one.
        omega: Frequency multiplier in every sine layer.
        output_activation: Optional callable applied to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        omega: float = 30.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=SineLayer,
            omega=omega,
            output_activation=output_activation,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with SIREN bounds, leaving biases unchanged."""
        _siren_init_(self.layers, self.layers[0].omega)


@torch.no_grad()
def _siren_init_(layers: nn.ModuleList, omega: float) -> None:
    """Initialize SIREN-style weights in place (Sitzmann et al., Sec. 3.2).

    The first layer uses ``U(-1/n, 1/n)`` and later layers, including the
    linear output, ``U(-sqrt(6/n)/omega, sqrt(6/n)/omega)``, where ``n`` is
    the layer's input width. Biases are left unchanged.
    """
    for i, layer in enumerate(layers):
        if i == 0:
            bound = 1 / layer.in_features
        else:
            bound = math.sqrt(6 / layer.in_features) / omega
        getattr(layer, "linear", layer).weight.uniform_(-bound, bound)


# FINER --------------------------------------------------------------------------------


class FinerLayer(nn.Module):
    """Variable-periodic sine layer ``sin(omega * (abs(z) + 1) * z)``.

    Here ``z = linear(x)``. The input-dependent scale ``abs(z) + 1`` is
    detached to keep its derivative out of the gradient. Maps
    ``(..., in_features)`` to ``(..., out_features)``.
    """

    def __init__(
        self, in_features: int, out_features: int, omega: float = 30.0
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.omega = omega

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear(x)
        scale = z.detach().abs() + 1
        return torch.sin(self.omega * scale * z)


class FINER(MLP):
    """Variable-periodic coordinate MLP (Liu et al., CVPR 2024).

    Uses :class:`FinerLayer` hidden layers and SIREN weight initialization.
    Biases keep PyTorch's defaults; the paper's enlarged first-layer bias
    range is not exposed here.

    Args:
        in_features: Number of input coordinates.
        out_features: Number of output channels.
        hidden_features: Hidden width.
        hidden_layers: Number of variable-periodic layers; must be at least one.
        omega: Base frequency multiplier in every hidden layer.
        output_activation: Optional callable applied to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        omega: float = 30.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=FinerLayer,
            omega=omega,
            output_activation=output_activation,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with SIREN bounds, leaving biases unchanged."""
        _siren_init_(self.layers, self.layers[0].omega)


# Gauss --------------------------------------------------------------------------------


class GaussLayer(nn.Module):
    """Gaussian layer ``exp(-(scale * linear(x))**2)``.

    Maps ``(..., in_features)`` to ``(..., out_features)``. Larger ``scale``
    produces narrower Gaussians.
    """

    def __init__(
        self, in_features: int, out_features: int, scale: float = 10.0
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.scale = scale

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-((self.scale * self.linear(x)) ** 2))


class Gauss(MLP):
    """Gaussian coordinate MLP (Ramasinghe and Lucey, ECCV 2022).

    Uses :class:`GaussLayer` hidden layers and PyTorch's default initialization.

    Args:
        in_features: Number of input coordinates.
        out_features: Number of output channels.
        hidden_features: Hidden width.
        hidden_layers: Number of Gaussian layers; must be at least one.
        scale: Inverse Gaussian width in every hidden layer.
        output_activation: Optional callable applied to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        scale: float = 10.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=GaussLayer,
            scale=scale,
            output_activation=output_activation,
        )


# WIRE ---------------------------------------------------------------------------------


class GaborLayer(nn.Module):
    """Real Gabor layer ``cos(omega * z1) * exp(-(scale * z2)**2)``.

    A linear projection produces ``z1`` and ``z2`` with ``out_features``
    channels each. ``omega`` controls frequency and ``scale`` controls the
    inverse Gaussian width. Leading input dimensions are preserved.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 20.0,
        scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.omega = omega
        self.scale = scale

        self.linear = nn.Linear(in_features, 2 * out_features)

    def forward(self, x: Tensor) -> Tensor:
        projection = self.linear(x)
        frequency_projection, scale_projection = projection.chunk(2, dim=-1)
        return torch.cos(self.omega * frequency_projection) * torch.exp(
            -((self.scale * scale_projection) ** 2)
        )


class WIRE(MLP):
    """Real-valued WIRE (Saragadam et al., CVPR 2023) with :class:`GaborLayer`.

    The paper's WIRE uses complex Gabor wavelets; here each wavelet is formed
    from two real projections. Uses PyTorch's default initialization and a
    linear output layer. The hidden width is reduced by ``sqrt(2)`` as in the
    complex WIRE; each Gabor layer projects to twice that width to form its
    two wavelet terms.

    Args:
        in_features: Number of input coordinates.
        out_features: Number of output channels.
        hidden_features: Nominal width, divided by ``sqrt(2)`` and rounded down.
        hidden_layers: Number of Gabor layers; must be at least one.
        omega: Frequency multiplier.
        scale: Inverse Gaussian width.
        output_activation: Optional callable applied to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        omega: float = 20.0,
        scale: float = 10.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        hidden_features = int(hidden_features / math.sqrt(2))

        super().__init__(
            in_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=GaborLayer,
            omega=omega,
            scale=scale,
            output_activation=output_activation,
        )


# RFF ----------------------------------------------------------------------------------


class RFFEncoding(nn.Module):
    """Random Fourier encoding ``[sin(2πBx), cos(2πBx)]``.

    The fixed frequency buffer ``B`` has shape ``(num_frequencies,
    in_features)`` and entries sampled from ``Normal(0, sigma**2)`` at
    construction. All sines precede all cosines; the output width is
    ``2 * num_frequencies`` and leading input dimensions are preserved.

    See Tancik et al., "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains", NeurIPS 2020.
    """

    def __init__(
        self, in_features: int, num_frequencies: int = 256, sigma: float = 10.0
    ) -> None:
        super().__init__()

        frequencies = torch.randn(num_frequencies, in_features) * sigma
        self.register_buffer("B", frequencies)

        self.out_features = 2 * num_frequencies

    def forward(self, x: Tensor) -> Tensor:
        projection = 2 * math.pi * x @ self.B.T
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class RFF(nn.Module):
    """ReLU MLP with random Fourier features (Tancik et al., NeurIPS 2020).

    Maps ``(..., in_features)`` to ``(..., out_features)``.

    Args:
        in_features: Number of input coordinates.
        out_features: Number of output channels.
        hidden_features: Hidden width.
        hidden_layers: Number of ReLU layers; must be at least one.
        num_frequencies: Number of random frequency vectors in :class:`RFFEncoding`.
        sigma: Standard deviation of the frequency vectors.
        output_activation: Optional callable applied to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        num_frequencies: int = 256,
        sigma: float = 10.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()

        self.encoding = RFFEncoding(in_features, num_frequencies, sigma)
        self.mlp = MLP(
            self.encoding.out_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=ReLULayer,
            output_activation=output_activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.encoding(x))


# Positional encoding ------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Positional encoding ``[x, sin(2**k * x), cos(2**k * x)]``.

    Uses ``num_frequencies`` bands, with ``k`` starting at zero, without the
    π factor in NeRF Eq. (4). The input is followed by all sines, then all
    cosines; within each block, bands vary faster than coordinates.
    The output width is ``in_features * (2 * num_frequencies + 1)`` and
    leading input dimensions are preserved. Bands are stored in ``freqs``.

    See Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance
    Fields for View Synthesis", ECCV 2020.
    """

    def __init__(self, in_features: int, num_frequencies: int = 10) -> None:
        super().__init__()

        freqs = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freqs", freqs)

        self.out_features = in_features * (2 * num_frequencies + 1)

    def forward(self, x: Tensor) -> Tensor:
        projection = x.unsqueeze(-1) * self.freqs
        return torch.cat(
            [x, projection.sin().flatten(-2), projection.cos().flatten(-2)], dim=-1
        )


# Instant-NGP --------------------------------------------------------------------------

# tiny-cuda-nn's coherent prime hash factors. The first coordinate is left
# unscaled (factor 1) for cache coherence.
_PRIMES = (1, 2654435761, 805459861)


def _level_scales(
    num_levels: int, base_resolution: int, max_resolution: int
) -> list[float]:
    """Per-level grid scales, computed in float32 like instant-ngp/tiny-cuda-nn.

    The ceiling of a scale sets the level's resolution, so matching the
    reference's float32 rounding keeps table sizes identical.
    """
    f32 = np.float32
    ratio = f32(max_resolution) / f32(base_resolution)
    growth = np.exp(np.log(ratio) / f32(max(num_levels - 1, 1)))
    log2_growth = np.log2(growth)
    return [
        float(f32(2) ** (f32(level) * log2_growth) * f32(base_resolution) - f32(1))
        for level in range(num_levels)
    ]


class HashEncoding(nn.Module):
    """Multiresolution hash encoding (Müller et al., SIGGRAPH 2022).

    Level ``l`` scales unit-cube positions by ``N_min * b**l - 1``, with growth
    ``b = (N_max / N_min) ** (1 / (L - 1))``, and adds a half-cell offset so
    levels are staggered (paper, Appendix A). Its grid has
    ``ceil(scale) + 1`` vertices per axis. Grids whose vertices fit in the
    table are indexed densely; finer grids are hashed. Features of the
    ``2**C`` surrounding vertices are interpolated multilinearly and
    concatenated across levels.

    Args:
        in_features: Number of input coordinates ``C`` (1, 2, or 3).
        num_levels: Number of resolution levels ``L``.
        features_per_level: Feature channels ``F`` per table entry.
        log2_hashmap_size: Base-2 logarithm of the maximum entries ``T`` per level.
        base_resolution: Coarsest resolution ``N_min``.
        max_resolution: Finest resolution ``N_max`` over the ``[-1, 1]`` domain;
            the reference uses 2048 for NeRF and SDFs, ``max(H, W) / 2`` for
            images, and the voxel resolution for volumes.

    Shape:
        - Input: :math:`(*, C)` in ``[-1, 1]``.
        - Output: :math:`(*, L F)`.

    The learnable ``embeddings`` concatenate all levels and start uniformly
    in ``[-1e-4, 1e-4]``.

    The forward pass handles every level at once, as one batched tensor
    operation per step rather than a loop over levels: the arithmetic is the
    same, so the features are identical, but a pass costs a few dozen kernel
    launches instead of a few hundred, which is what sets the encoding's time
    on the small batches a ray marcher or a training step feeds it.
    """

    def __init__(
        self,
        in_features: int,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
    ) -> None:
        super().__init__()
        if not 1 <= in_features <= len(_PRIMES):
            raise ValueError(f"in_features must be in [1, {len(_PRIMES)}]")
        self.in_features = in_features
        self.num_levels = num_levels
        self.features_per_level = features_per_level

        self.scales = _level_scales(num_levels, base_resolution, max_resolution)
        self.resolutions = [math.ceil(scale) + 1 for scale in self.scales]
        self.table_sizes = [
            -(-min(resolution**in_features, 2**log2_hashmap_size) // 8) * 8
            for resolution in self.resolutions
        ]
        self.offsets = [0]
        for size in self.table_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + size)
        # The per-level constants as tensors, for the batched forward pass. A
        # level whose vertices outnumber its table is hashed, the others are
        # indexed densely.
        levels = {
            "level_scales": torch.tensor(self.scales, dtype=torch.float32),
            "level_resolutions": torch.tensor(self.resolutions, dtype=torch.long),
            "level_table_sizes": torch.tensor(self.table_sizes, dtype=torch.long),
            "level_offsets": torch.tensor(self.offsets, dtype=torch.long),
            "level_hashed": torch.tensor(
                [
                    size < resolution**in_features
                    for resolution, size in zip(self.resolutions, self.table_sizes)
                ]
            ),
        }
        for name, value in levels.items():
            self.register_buffer(name, value, persistent=False)

        self.register_buffer(
            "corners",
            torch.tensor(list(product((0, 1), repeat=in_features)), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "primes",
            torch.tensor(_PRIMES[:in_features], dtype=torch.long),
            persistent=False,
        )
        self.embeddings = nn.Parameter(
            torch.empty(sum(self.table_sizes), features_per_level).uniform_(-1e-4, 1e-4)
        )

    @property
    def out_features(self) -> int:
        """Number of output features, ``num_levels * features_per_level``."""
        return self.num_levels * self.features_per_level

    def forward(self, x: Tensor) -> Tensor:
        batch_shape = x.shape[:-1]
        unit_x = (x.reshape(-1, self.in_features) + 1) / 2

        # Every level at once: (N, L, C) positions, (N, L, 2^C, C) vertices.
        position = unit_x.unsqueeze(1) * self.level_scales.view(1, -1, 1) + 0.5
        grid = position.floor()
        fraction = (position - grid).unsqueeze(2)
        vertices = grid.long().unsqueeze(2) + self.corners

        # Spatial hash; the reference's uint32 wraparound only affects bits
        # above log2(table_size), so int64 arithmetic gives the same index.
        hashed = vertices * self.primes
        hash_index = hashed[..., 0]
        for dim in range(1, self.in_features):
            hash_index = hash_index ^ hashed[..., dim]
        # Dense stride index with the first coordinate varying fastest.
        resolution = self.level_resolutions.view(1, -1, 1)
        dense_index = vertices[..., -1]
        for dim in range(self.in_features - 2, -1, -1):
            dense_index = dense_index * resolution + vertices[..., dim]
        index = torch.where(self.level_hashed.view(1, -1, 1), hash_index, dense_index)
        index = index % self.level_table_sizes.view(1, -1, 1)

        corner_features = self.embeddings[self.level_offsets.view(1, -1, 1) + index]
        weights = torch.where(self.corners.bool(), fraction, 1 - fraction).prod(-1)
        out = (weights.unsqueeze(-1) * corner_features).sum(2)  # (N, L, F)
        return out.reshape(*batch_shape, self.out_features)


def _broadcast(value: int | Sequence[int], length: int, name: str) -> list[int]:
    """Expand a shared int to ``length`` values, or check a per-axis sequence."""
    values = [value] * length if isinstance(value, int) else list(value)
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    return values


def _grid_position(x: Tensor, size: int) -> Tensor:
    """Map coordinates in ``[-1, 1]`` to continuous indices in ``[0, size - 1]``."""
    return (x + 1) / 2 * (size - 1)


# Bases --------------------------------------------------------------------------------


class _Basis(nn.Module):
    """Base class for coordinate bases.

    Subclasses implement :meth:`_evaluate` for one axis and call
    :meth:`_build_cache` once the buffers it needs are registered.
    """

    # Distance, in grid steps, within which a coordinate reads the cached table.
    _grid_atol = 1e-4

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        min_components: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_components = _broadcast(num_components, in_features, "num_components")
        if min(self.num_components) < min_components:
            raise ValueError(
                f"num_components must be >= {min_components}, "
                f"got {self.num_components}"
            )
        self.normalize = normalize
        self.grid_size: list[int] | None = None

    def _evaluate(self, x: Tensor, axis: int) -> Tensor:
        """Evaluate one axis at coordinates ``(*,)``, returning ``(*, K_axis)``."""
        raise NotImplementedError

    def _normalize(self, features: Tensor) -> Tensor:
        return F.normalize(features, dim=-1) if self.normalize else features

    def _build_cache(self, grid_size: int | Sequence[int] | None) -> None:
        """Tabulate every axis on a regular grid; ``None`` disables the cache."""
        if grid_size is None:
            return
        self.grid_size = _broadcast(grid_size, self.in_features, "grid_size")
        for axis, size in enumerate(self.grid_size):
            table = self._evaluate(torch.linspace(-1.0, 1.0, size), axis)
            self.register_buffer(f"_cache_{axis}", table, persistent=False)

    def _lookup(self, x: Tensor, axis: int, size: int) -> Tensor:
        """Read on-grid coordinates from the cache and evaluate the rest."""
        if size == 1:
            return self._evaluate(x, axis)

        position = _grid_position(x, size)
        index = position.round()
        on_grid = (
            ((position - index).abs() < self._grid_atol)
            & (position >= -self._grid_atol)
            & (position <= size - 1 + self._grid_atol)
        )
        features = self.get_buffer(f"_cache_{axis}")[index.clamp(0, size - 1).long()]
        if not bool(on_grid.all()):
            off_grid = ~on_grid
            features[off_grid] = self._evaluate(x[off_grid], axis)
        return features

    def _axis_features(self, x: Tensor, axis: int) -> Tensor:
        if self.grid_size is None:
            return self._evaluate(x, axis)
        return self._lookup(x, axis, self.grid_size[axis])

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        return tuple(
            self._axis_features(x[..., axis], axis) for axis in range(self.in_features)
        )


class CosineBasis(_Basis):
    r"""Cosine basis :math:`\cos(k \pi u)` with :math:`u = (x + 1) / 2`.

    Axis ``c`` uses frequencies :math:`k = 0, \dots, K_c - 1`. This is the
    FUTON paper's basis without its :math:`\sqrt{2}` factor; see
    :class:`FUTON`.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of frequencies ``K_c``, shared or per axis.
        normalize: L2-normalize each feature vector.
        grid_size: Per-axis cache size; see the module notes.

    Shape:
        - Input: :math:`(*, C)`.
        - Output: ``C`` tensors of shape :math:`(*, K_c)`.
    """

    frequencies: Tensor

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(in_features, num_components, normalize)
        frequencies = math.pi * torch.arange(max(self.num_components))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self._build_cache(grid_size)

    def _evaluate(self, x: Tensor, axis: int) -> Tensor:
        u = ((x + 1) / 2).unsqueeze(-1)
        frequencies = self.frequencies[: self.num_components[axis]]
        return self._normalize(torch.cos(frequencies * u))


class SincBasis(_Basis):
    r"""Cardinal sine basis centered on a uniform grid over ``[-1, 1]``.

    Component ``k`` is :math:`\operatorname{sinc}((x - \mu_k) / w)`, with
    centers :math:`\mu_k = -1 + k w` and spacing :math:`w = 2 / (K_c - 1)`.
    Each component equals one at its own center and zero at all other
    centers, so ``normalize=False`` gives cardinal (Whittaker) interpolation.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of centers ``K_c >= 2``, shared or per axis.
        normalize: L2-normalize each feature vector.
        grid_size: Per-axis cache size; see the module notes.

    Shape:
        - Input: :math:`(*, C)`.
        - Output: ``C`` tensors of shape :math:`(*, K_c)`.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(in_features, num_components, normalize, min_components=2)
        self.width = [2.0 / (count - 1) for count in self.num_components]
        for axis, count in enumerate(self.num_components):
            centers = torch.linspace(-1.0, 1.0, count)
            self.register_buffer(f"centers_{axis}", centers, persistent=False)
        self._build_cache(grid_size)

    def _evaluate(self, x: Tensor, axis: int) -> Tensor:
        centers = self.get_buffer(f"centers_{axis}")
        return self._normalize(
            torch.sinc((x.unsqueeze(-1) - centers) / self.width[axis])
        )


class _LocalBasis(_Basis):
    """Compact kernels centered on ``linspace(-1, 1, K_c)``.

    Subclasses implement :meth:`_kernel` on offsets ``t`` measured in grid
    steps. The kernel must vanish for ``|t| >= radius``, so each coordinate
    touches at most ``2 * radius`` consecutive components; sparse mode
    evaluates only those taps.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        radius: int,
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
        sparse: bool = True,
    ) -> None:
        if radius < 1:
            raise ValueError(f"radius must be >= 1, got {radius}")
        super().__init__(
            in_features, num_components, normalize, min_components=2 * radius
        )
        self.radius = int(radius)
        self.sparse = bool(sparse)
        if not self.sparse:
            self._build_cache(grid_size)

    def _kernel(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def _evaluate(self, x: Tensor, axis: int) -> Tensor:
        size = self.num_components[axis]
        centers = torch.arange(size, device=x.device, dtype=x.dtype)
        t = _grid_position(x, size).unsqueeze(-1) - centers
        return self._normalize(self._kernel(t))

    def _evaluate_sparse(self, x: Tensor, axis: int) -> RCSMatrix:
        """Evaluate the ``2 * radius`` taps around each point as an RCS matrix."""
        size, num_taps = self.num_components[axis], 2 * self.radius
        position = _grid_position(x.reshape(-1), size)
        # Clamping keeps each segment inside the basis; taps shifted by the
        # clamp lie outside the kernel support and evaluate to zero.
        start = (position.floor().long() - (self.radius - 1)).clamp_(0, size - num_taps)
        offsets = torch.arange(num_taps, device=x.device)
        t = position.unsqueeze(-1) - (start.unsqueeze(-1) + offsets).to(position.dtype)
        return RCSMatrix(self._normalize(self._kernel(t)), start, size)

    def _axis_features(self, x: Tensor, axis: int) -> Tensor:
        if self.sparse:
            return self._evaluate_sparse(x, axis)
        return super()._axis_features(x, axis)


class TriangleBasis(_LocalBasis):
    r"""Triangle (hat) basis for piecewise-linear interpolation on ``[-1, 1]``.

    Component ``k`` is :math:`\max(0, 1 - |x - \mu_k| / w)` with centers spaced
    by :math:`w = 2 / (K_c - 1)`, so at most two components are nonzero. With
    ``normalize=False``, a linear map of these features equals sampling a 1D
    feature grid with ``align_corners=True``; a CP combiner and MLP decoder
    then give the TensoRF-CP parametrization.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of centers ``K_c >= 2``, shared or per axis.
        normalize: L2-normalize each feature vector.
        grid_size: Per-axis cache size in dense mode; ignored when sparse.
        sparse: Return :class:`RCSMatrix` features with two taps per point.

    Shape:
        - Input: :math:`(*, C)` in dense mode, :math:`(N, C)` in sparse mode.
        - Output: ``C`` features of shape :math:`(*, K_c)`, or
          :math:`(N, K_c)` RCS matrices in sparse mode.

    References:
        Chen et al., "TensoRF: Tensorial Radiance Fields", ECCV 2022.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
        sparse: bool = True,
    ) -> None:
        super().__init__(in_features, num_components, 1, normalize, grid_size, sparse)

    def _kernel(self, t: Tensor) -> Tensor:
        return (1.0 - t.abs()).clamp(min=0.0)


class LanczosBasis(_LocalBasis):
    r"""Windowed-sinc (Lanczos) basis centered on a uniform grid over ``[-1, 1]``.

    With grid offset :math:`t = (x - \mu_k) / w` and radius :math:`a`, the
    kernel is :math:`\operatorname{sinc}(t) \operatorname{sinc}(t / a)` for
    :math:`|t| < a` and zero elsewhere. Each component equals one at its own
    center and zero at all other centers; at most ``2 * radius`` are nonzero.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of centers ``K_c >= 2 * radius``, shared or per
            axis.
        radius: Kernel radius in grid steps: ``2`` gives Lanczos-2 (four taps),
            ``3`` gives Lanczos-3 (six taps).
        normalize: L2-normalize each feature vector.
        grid_size: Per-axis cache size in dense mode; ignored when sparse.
        sparse: Return :class:`RCSMatrix` features with ``2 * radius`` taps per
            point.

    Shape:
        - Input: :math:`(*, C)` in dense mode, :math:`(N, C)` in sparse mode.
        - Output: ``C`` features of shape :math:`(*, K_c)`, or
          :math:`(N, K_c)` RCS matrices in sparse mode.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        radius: int = 2,
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
        sparse: bool = True,
    ) -> None:
        super().__init__(
            in_features, num_components, radius, normalize, grid_size, sparse
        )

    def _kernel(self, t: Tensor) -> Tensor:
        return torch.where(
            t.abs() < self.radius,
            torch.sinc(t) * torch.sinc(t / self.radius),
            t.new_zeros(()),
        )


# Combiners ----------------------------------------------------------------------------


def _project(features: Tensor, linear: nn.Linear) -> Tensor:
    """Apply ``linear`` to dense or RCS features without densifying the latter."""
    if isinstance(features, RCSMatrix):
        out = features @ linear.weight.T
        return out if linear.bias is None else out + linear.bias
    return linear(features)


class _Combiner(nn.Module):
    """Base class for combiners; ``in_features[c]`` is the width ``K_c`` of axis ``c``."""

    out_features: int

    def __init__(self, in_features: Sequence[int]) -> None:
        super().__init__()
        self.in_features = list(in_features)


class HadamardCombiner(_Combiner):
    """Elementwise product of per-axis features that share one width ``K``.

    RCS features are densified before multiplication.

    Args:
        in_features: Feature width of each axis; all entries must be equal.

    Shape:
        - Input: ``C`` tensors of shape :math:`(*, K)`.
        - Output: :math:`(*, K)`.
    """

    def __init__(self, in_features: Sequence[int]) -> None:
        super().__init__(in_features)
        if len(set(self.in_features)) != 1:
            raise ValueError(
                f"Expected all in_features entries to be equal, got {self.in_features}"
            )
        self.out_features = self.in_features[0]

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        return reduce(
            mul,
            (f.to_dense() if isinstance(f, RCSMatrix) else f for f in features),
        )


class CPCombiner(_Combiner):
    r"""Canonical polyadic (CP) combiner.

    Projects each axis to ``rank`` channels and multiplies the projections
    elementwise. ``linears[c].weight`` stores the transposed CP factor
    :math:`\mathbf{U}^{(c)\top}` of shape ``(rank, K_c)`` (FUTON paper,
    Eqs. 10–11). RCS features are projected without densifying.

    Args:
        in_features: Feature width ``K_c`` of each axis.
        rank: CP rank, which is also the output width.
        bias: Add a bias to each projection; the paper's model has none.

    Shape:
        - Input: ``C`` tensors of shape :math:`(N, K_c)`.
        - Output: :math:`(N, \text{rank})`.

    References:
        Kolda and Bader, "Tensor Decompositions and Applications", SIAM Review 2009.
    """

    def __init__(
        self, in_features: Sequence[int], rank: int, bias: bool = False
    ) -> None:
        super().__init__(in_features)
        self.rank = rank
        self.out_features = rank
        self.linears = nn.ModuleList(
            nn.Linear(width, rank, bias=bias) for width in self.in_features
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize factors with Kaiming-uniform weights and zero biases."""
        for linear in self.linears:
            nn.init.kaiming_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        projections = (
            _project(feature, linear) for feature, linear in zip(features, self.linears)
        )
        return reduce(mul, projections)


class TRCombiner(_Combiner):
    r"""Tensor-ring (TR) combiner.

    Axis ``c`` contracts its features with a core of shape
    ``(rank, K_c, rank)`` into a ``rank x rank`` matrix. The ordered product of
    these matrices is flattened to ``rank**2`` features, leaving the ring
    closure (e.g. a trace) to the decoder. This extends the paper's CP model.

    Args:
        in_features: Feature width ``K_c`` of each axis.
        rank: Tensor-ring rank; the output width is ``rank**2``.

    Shape:
        - Input: ``C`` tensors of shape :math:`(N, K_c)`.
        - Output: :math:`(N, \text{rank}^2)`.

    References:
        Zhao et al., "Tensor Ring Decomposition", arXiv 2016.
    """

    def __init__(self, in_features: Sequence[int], rank: int) -> None:
        super().__init__(in_features)
        self.rank = rank
        self.out_features = rank**2
        self.cores = nn.ParameterList(
            nn.Parameter(torch.empty(rank, width, rank)) for width in self.in_features
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the cores with Kaiming-uniform weights."""
        for core in self.cores:
            nn.init.kaiming_uniform_(core)

    @staticmethod
    def _transfer(features: Tensor, core: Tensor) -> Tensor:
        """Contract ``(N, K)`` features with an ``(R, K, R)`` core into ``(N, R, R)``."""
        if isinstance(features, RCSMatrix):
            # Fold the rank axes so the contraction is one sparse matmul.
            rank, width, _ = core.shape
            flat = features @ core.permute(1, 0, 2).reshape(width, rank * rank)
            return flat.unflatten(-1, (rank, rank))
        return torch.einsum("...k,rks->...rs", features, core)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        matrices = map(self._transfer, features, self.cores)
        return reduce(torch.matmul, matrices).flatten(start_dim=-2)


# FUTON --------------------------------------------------------------------------------


class FUTONEncoding(nn.Module):
    r"""FUTON coordinate encoding ``combiner(basis(x))``, without a decoder.

    Args:
        in_features: Number of coordinate axes ``C``.
        basis: Basis spec, built as ``basis(in_features, **kwargs)``. Custom
            bases must expose ``num_components``, one count per axis.
        combiner: Combiner spec, built as
            ``combiner(basis.num_components, **kwargs)``. Custom combiners must
            expose ``out_features``.

    Shape:
        - Input: :math:`(*, C)` coordinates in ``[-1, 1]``.
        - Output: :math:`(*, \text{out\_features})`, the combiner's width.
    """

    def __init__(
        self, in_features: int, basis: ModuleSpec, combiner: ModuleSpec
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.basis = build_module(basis, BASES, in_features)
        self.combiner = build_module(combiner, COMBINERS, self.basis.num_components)
        self.out_features = self.combiner.out_features

    def forward(self, x: Tensor) -> Tensor:
        # Sparse bases expect a flat batch of points.
        features = self.combiner(self.basis(x.reshape(-1, self.in_features)))
        return features.reshape(*x.shape[:-1], self.out_features)


class FUTON(nn.Module):
    r"""Fourier Tensor Network for implicit neural representations.

    Computes ``output_activation(decoder(combiner(basis(x))))``. Each stage is
    a module instance, a registry key, a module class, or a
    ``(key_or_class, kwargs)`` pair. Registry keys:

    - Bases: ``"cosine"``, ``"sinc"``, ``"triangle"``, ``"lanczos"``.
    - Combiners: ``"hadamard"``, ``"cp"``, ``"tr"``.
    - Decoders: ``"linear"``, ``"mlp"``.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of output channels.
        basis: Basis spec; see :class:`FUTONEncoding`.
        combiner: Combiner spec; see :class:`FUTONEncoding`.
        decoder: Decoder spec, built as
            ``decoder(combiner.out_features, out_features, **kwargs)``.
        output_activation: Callable applied to the output; ``None`` uses the
            identity.

    Shape:
        - Input: :math:`(*, C)` coordinates in ``[-1, 1]``.
        - Output: :math:`(*, \text{out\_features})`.

    Relation to the paper:
        The cosine basis, CP combiner, and linear decoder implement Eq. (9):
        ``encoding.combiner.linears[c].weight`` stores
        :math:`\mathbf{U}^{(c)\top}` and ``decoder.weight`` stores
        :math:`\mathbf{V}`; the decoder bias is an addition. Coordinates are
        mapped internally from ``[-1, 1]`` to ``[0, 1]``. The basis omits the
        paper's :math:`\sqrt{2}` factor and L2-normalizes features by default;
        ``normalize=False`` recovers Eq. (1) up to per-frequency constants
        absorbed by the factors. Experiments use ``torch.tanh`` at the output.
        The other bases, the Hadamard and tensor-ring combiners, and the MLP
        decoder are extensions.

    Example::

        model = FUTON(
            in_features=2,
            out_features=3,
            basis=("cosine", {"num_components": 256}),
            combiner=("cp", {"rank": 256}),
            decoder="linear",
            output_activation=torch.tanh,
        )
        rgb = model(torch.rand(64, 64, 2) * 2 - 1)  # (64, 64, 3)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: ModuleSpec,
        combiner: ModuleSpec,
        decoder: ModuleSpec,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.encoding = FUTONEncoding(in_features, basis, combiner)
        self.decoder = build_module(
            decoder, DECODERS, self.encoding.out_features, out_features
        )
        self.output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.output_activation(self.decoder(self.encoding(x)))


# Registries ---------------------------------------------------------------------------

BASES: dict[str, type[nn.Module]] = {
    "cosine": CosineBasis,
    "sinc": SincBasis,
    "triangle": TriangleBasis,
    "lanczos": LanczosBasis,
}

COMBINERS: dict[str, type[nn.Module]] = {
    "hadamard": HadamardCombiner,
    "cp": CPCombiner,
    "tr": TRCombiner,
}

DECODERS: dict[str, type[nn.Module]] = {
    "linear": nn.Linear,
    "mlp": MLP,
}


# INRs, built with ``in_features`` and ``out_features``.
INRS: dict[str, type[nn.Module]] = {
    "linear": nn.Linear,
    "mlp": MLP,
    "siren": SIREN,
    "finer": FINER,
    "gauss": Gauss,
    "wire": WIRE,
    "rff": RFF,
    "futon": FUTON,
}
