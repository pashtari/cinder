"""Modulators: condition an INR on spatial encoder maps.

A modulator wraps a model and predicts ``y = INR_theta(z)(phi(x, z))`` from query
coordinates ``x`` and encoder maps ``z``. Every modulator follows one protocol,
:class:`BaseModulator`: ``forward(coords, conds) -> (B, *, out_features)``, where
coordinates with a batch axis of 1 are shared by the batch.

- **Input modulators** change what the INR reads. :class:`FUTONGate` gates a
  coordinate encoding with the maps sampled at each query, so the condition varies
  within an image.
- **Weight modulators** change the INR's parameters. :class:`WeightDisplacement`
  displaces each matrix per image, so the condition selects a function.

The wrapped model can itself be a modulator, which is how :class:`ListModulators`
composes several. Modulators are chosen by spec from :data:`MODULATORS`, and their
parts from :data:`FUSIONS` and :data:`CONDITIONERS`.

Shape symbols: ``B`` batch, ``*`` query shape, ``D`` coordinate dimension, ``E``
map channels, ``F`` gate features, ``S`` maps.
"""

from collections.abc import Callable, Mapping, Sequence
from functools import partial, reduce
from math import prod
from operator import add, attrgetter, mul

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call, vmap

from .inrs import INRS, FUTONEncoding
from .utils import ModuleSpec, build_module, check_sizes

__all__ = [
    "BaseModulator",
    "GridSampler",
    "Camera",
    "ProductFusion",
    "ResidualFusion",
    "SumFusion",
    "ConvexFusion",
    "AttentionFusion",
    "FUTONGate",
    "WeightConditioner",
    "LinearWeightConditioner",
    "MLPWeightConditioner",
    "WeightDisplacement",
    "ListModulators",
    "FUSIONS",
    "CONDITIONERS",
    "MODULATORS",
]


def _as_shapes(
    cond_shape: Sequence[int] | Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Return one validated shape per map; a lone ``(E, *grid)`` is one map."""
    shapes = tuple(cond_shape)
    if not any(isinstance(shape, Sequence) for shape in shapes):
        shapes = (shapes,)
    for shape in shapes:
        if (
            not isinstance(shape, Sequence)
            or len(shape) == 0
            or any(not isinstance(size, int) or size < 1 for size in shape)
        ):
            raise ValueError(
                "cond_shape must be a shape (E, *grid) of positive integers, "
                f"or a sequence of such shapes, got {cond_shape}"
            )
    return tuple(tuple(shape) for shape in shapes)


class BaseModulator(nn.Module):
    """Base class for modulators, which condition an INR on encoder maps.

    The wrapped model, ``inr``, is a plain INR from :data:`~cinder.models.inrs.INRS`
    or another modulator, which then reads ``(coords, conds)`` too. Subclasses
    build it with :meth:`build_inr` once they know the width it reads, and
    implement :meth:`forward`.

    Args:
        inr: Spec of the wrapped model.
        in_features: Width of the coordinates the modulator reads.
        out_features: Prediction width.
        cond_shape: Shape ``(E, *grid)`` of a conditioning map without the batch
            axis, or one such shape per map. Stored as a tuple of shapes.
    """

    def __init__(
        self,
        inr: ModuleSpec,
        in_features: int,
        out_features: int,
        cond_shape: Sequence[int] | Sequence[Sequence[int]],
    ) -> None:
        super().__init__()
        for name, size in (
            ("in_features", in_features),
            ("out_features", out_features),
        ):
            if not isinstance(size, int) or size < 1:
                raise ValueError(f"{name} must be a positive integer, got {size}")
        self.in_features = in_features
        self.out_features = out_features
        self.cond_shape = _as_shapes(cond_shape)
        self._inr_spec = inr

    def forward(self, coords: Tensor, conds: Tensor | Sequence[Tensor]) -> Tensor:
        """Predict at the coordinates, image by image.

        Args:
            coords: Query coordinates ``(B, *, in_features)``, or
                ``(1, *, in_features)`` to share them across the batch.
            conds: Conditioning maps, one ``(B, E, *grid)`` tensor per shape.

        Returns:
            Predictions ``(B, *, out_features)``.
        """
        raise NotImplementedError

    def build_inr(self, in_features: int) -> nn.Module:
        """Build the wrapped model from the ``inr`` spec, reading ``in_features``."""
        spec = self._inr_spec
        # Drop the spec, or a ready-made module would be registered twice.
        del self._inr_spec
        sizes = dict(in_features=in_features, out_features=self.out_features)
        module = build_module(spec, INRS, **sizes)
        check_sizes(module, "inr", **sizes)
        return module

    def check_inputs(
        self, coords: Tensor, conds: Tensor | Sequence[Tensor]
    ) -> tuple[Tensor, ...]:
        """Check the forward arguments and return the maps as a tuple."""
        conds = (conds,) if isinstance(conds, Tensor) else tuple(conds)
        if len(conds) != len(self.cond_shape):
            raise ValueError(
                f"expected {len(self.cond_shape)} conditioning map(s), got {len(conds)}"
            )
        for i, (cond, shape) in enumerate(zip(conds, self.cond_shape)):
            if not isinstance(cond, Tensor):
                raise TypeError(f"conditioning map {i} must be a Tensor")
            if tuple(cond.shape[1:]) != shape:
                raise ValueError(
                    f"conditioning map {i} must have shape {shape} after the batch "
                    f"axis, got {tuple(cond.shape)}"
                )
            if cond.shape[0] != conds[0].shape[0]:
                raise ValueError("conditioning maps must have the same batch size")
            if cond.device != conds[0].device:
                raise ValueError("conditioning maps must be on the same device")

        batch_size = conds[0].shape[0]
        if (
            coords.ndim < 2
            or coords.shape[-1] != self.in_features
            or coords.shape[0] not in (1, batch_size)
        ):
            raise ValueError(
                f"coords must have shape ({batch_size}, *, {self.in_features}) or "
                f"(1, *, {self.in_features}), got {tuple(coords.shape)}"
            )
        return conds


# Sampling -----------------------------------------------------------------------------


class GridSampler(nn.Module):
    """Read a feature map at continuous coordinates.

    Each map is interpolated where it lies instead of being resampled onto a
    common grid.

    Args:
        mode: Interpolation, as in :func:`~torch.nn.functional.grid_sample`.
        padding_mode: How coordinates outside ``[-1, 1]`` are resolved.
        align_corners: Whether ``-1`` and ``1`` are the corner pixels' centers.
            ``True`` matches CINDER's coordinates, so a query on a map's own
            grid node reads that pixel exactly.

    Shape:
        - coords: ``(*, D)`` in ``[-1, 1]``; values: ``(B, F, *grid)`` with ``D``
          grid axes (``grid_sample`` supports 2 or 3).
        - Output: ``(B, *, F)``.
    """

    def __init__(
        self,
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, coords: Tensor, values: Tensor) -> Tensor:
        if coords.ndim < 1 or coords.shape[-1] not in (2, 3):
            raise ValueError("coords must have 2 or 3 spatial dimensions")
        if values.ndim != coords.shape[-1] + 2:
            raise ValueError(
                f"`values` must have {coords.shape[-1]} spatial dimensions to "
                f"match the coordinates, got shape {tuple(values.shape)}"
            )
        batch, num_features = values.shape[:2]
        *query_shape, dim = coords.shape

        # Sample half-precision maps in float32: subpixel coordinates would
        # round, and CPU grid_sample has no half or bfloat16 kernel.
        out_dtype = values.dtype
        if values.dtype in (torch.float16, torch.bfloat16):
            values = values.float()

        # grid_sample reads the last axis as (x, y[, z]), the reverse of "ij"
        # order, and wants a grid shaped like its output: the M queries run
        # along the first sampling axis and the other axes stay singleton.
        grid = coords.flip(-1).reshape(-1, *[1] * (dim - 1), dim)  # (M, 1[, 1], D)
        grid = grid.to(values.dtype).expand(batch, *grid.shape)
        with torch.autocast(device_type=values.device.type, enabled=False):
            sampled = F.grid_sample(
                values,
                grid,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )  # (B, F, M, 1[, 1])

        # `.mT` moves channels last as a strided view, avoiding a copy.
        queried = sampled.flatten(2).mT  # (B, M, F)
        return queried.reshape(batch, *query_shape, num_features).to(out_dtype)


class Camera(nn.Module):
    """Project ``(*, 3)`` coordinates to a view's ``(*, 2)`` grid coordinates.

    A pinhole camera with a ``3 x 4`` matrix ``P`` maps ``x`` to
    ``P[:2] x~ / P[2] x~``, where ``x~ = (x, 1)``, both in CINDER's coordinates,
    ``[-1, 1]`` per axis. As ``sample_at`` of :class:`FUTONGate`, it reads a
    view's maps where each point projects. The projection runs in float32 even
    under autocast, since a half-precision division would move the samples by a
    fraction of a pixel.

    Args:
        matrix: The ``(3, 4)`` camera matrix.
    """

    def __init__(self, matrix: Tensor) -> None:
        super().__init__()
        self.register_buffer("matrix", torch.as_tensor(matrix, dtype=torch.float32))

    @classmethod
    def fit(cls, points: Tensor, projections: Tensor) -> "Camera":
        """Fit the camera to ``(N, 3)`` points and their ``(N, 2)`` projections.

        The direct linear transform finds the matrix, up to scale, as the null
        vector of the projection constraints, in float64.
        """
        points, projections = points.double(), projections.double()
        homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
        zeros = torch.zeros_like(homogeneous)
        constraints = torch.cat(
            [
                torch.cat([homogeneous, zeros, -projections[:, :1] * homogeneous], 1),
                torch.cat([zeros, homogeneous, -projections[:, 1:] * homogeneous], 1),
            ]
        )
        matrix = torch.linalg.svd(constraints, full_matrices=False).Vh[-1].view(3, 4)
        return cls(matrix / matrix[2, :3].norm())

    def forward(self, coords: Tensor) -> Tensor:
        with torch.autocast(device_type=coords.device.type, enabled=False):
            projected = coords.float() @ self.matrix[:, :3].T + self.matrix[:, 3]
            return projected[..., :2] / projected[..., 2:]


def _coordinate_maps(
    sample_at: Sequence[ModuleSpec | None] | None,
    cond_shape: tuple[tuple[int, ...], ...],
    in_features: int,
) -> nn.ModuleList:
    """Build the module that maps the coordinates to each map's grid.

    ``None``, for every map or one, samples at the coordinates themselves, so that
    map needs ``in_features`` grid axes.
    """
    if sample_at is None:
        sample_at = [None] * len(cond_shape)
    if len(sample_at) != len(cond_shape):
        raise ValueError(
            f"sample_at needs one entry per map, got {len(sample_at)} "
            f"for {len(cond_shape)} maps"
        )
    if any(
        spec is None and len(shape) != in_features + 1
        for shape, spec in zip(cond_shape, sample_at)
    ):
        raise ValueError(
            f"maps are sampled at {in_features}D coordinates, so "
            f"cond_shape grids need {in_features} axes, got {cond_shape}"
        )
    return nn.ModuleList(
        nn.Identity() if spec is None else build_module(spec) for spec in sample_at
    )


# Fusions ------------------------------------------------------------------------------


class _Fusion(nn.Module):
    """Base class for the rules that combine per-map factors into one gate.

    Each map, sampled at the queries, gives one factor. With a single map every
    rule except :class:`ResidualFusion` returns that factor.

    Args:
        num_maps: Number of maps ``S``.
        out_features: Width ``F`` of each factor.

    Shape:
        - Input: ``S`` tensors of shape ``(B, *, F)``.
        - Output: ``(B, *, F)``.
    """

    def __init__(self, num_maps: int, out_features: int) -> None:
        super().__init__()
        self.num_maps = num_maps
        self.out_features = out_features

    def forward(self, factors: Sequence[Tensor]) -> Tensor:
        raise NotImplementedError


class ProductFusion(_Fusion):
    """Multiply the factors, so the gate opens only where every map agrees."""

    def forward(self, factors: Sequence[Tensor]) -> Tensor:
        # `reduce` rather than `math.prod`/`sum`, which start from a scalar and
        # so add a pass over the (B, M, F) gate and copy a lone factor.
        return reduce(mul, factors)


class ResidualFusion(_Fusion):
    """Multiply ``1 + factor``, so extra maps cannot drive the gate to zero."""

    def forward(self, factors: Sequence[Tensor]) -> Tensor:
        return reduce(mul, (1 + f for f in factors))


class SumFusion(_Fusion):
    """Add the factors: one linear projection of all maps' channels at the query."""

    def forward(self, factors: Sequence[Tensor]) -> Tensor:
        return reduce(add, factors)


class ConvexFusion(_Fusion):
    """Add the factors with learned weights that sum to one."""

    def __init__(self, num_maps: int, out_features: int) -> None:
        super().__init__(num_maps, out_features)
        self.mix_logits = nn.Parameter(torch.zeros(num_maps))  # uniform at init

    def forward(self, factors: Sequence[Tensor]) -> Tensor:
        # (B, *, F, S) @ (S,) -> (B, *, F)
        return torch.stack(factors, dim=-1) @ self.mix_logits.softmax(0)


class AttentionFusion(_Fusion):
    """Add the factors with weights scored at each query.

    Unlike :class:`ConvexFusion`, the mixture follows the coordinate instead of
    being fixed for the whole image.
    """

    def __init__(self, num_maps: int, out_features: int) -> None:
        super().__init__(num_maps, out_features)
        self.scores = nn.Parameter(torch.zeros(num_maps, out_features))  # uniform

    def forward(self, factors: Sequence[Tensor]) -> Tensor:
        scored = [factor @ score for factor, score in zip(factors, self.scores)]
        weights = torch.stack(scored, dim=-1).softmax(-1)  # (B, *, S)
        stacked = torch.stack(factors, dim=-1)  # (B, *, F, S)
        return (stacked * weights.unsqueeze(-2)).sum(-1)


# Fusions, built with ``num_maps`` and ``out_features``.
FUSIONS: dict[str, type[nn.Module]] = {
    "product": ProductFusion,
    "residual": ResidualFusion,
    "sum": SumFusion,
    "convex": ConvexFusion,
    "attention": AttentionFusion,
}


# Input modulation ---------------------------------------------------------------------


class FUTONGate(BaseModulator):
    """Gate a FUTON encoding of the coordinates with encoder maps sampled there.

    For query coordinates ``x`` and maps ``z_s``::

        y = inr(LayerNorm(encoding(x) * fusion([sample(proj_s(z_s), x) for s])))

    Each map is projected to the encoding width ``F`` and sampled on its own
    grid, at the query coordinates or where ``sample_at`` maps them, and the
    coordinates must be shared by the batch. The wrapped model reads the gated
    features of each image as its coordinates.

    Args:
        inr: Spec of the wrapped model, built with ``in_features=F``.
        in_features: Coordinate dimension ``D``.
        out_features: Prediction width.
        cond_shape: See :class:`BaseModulator`.
        basis: Basis spec from :data:`~cinder.models.inrs.BASES`.
        combiner: Combiner spec from :data:`~cinder.models.inrs.COMBINERS`.
        bias: Add a bias to each map projection.
        fusion: Spec from :data:`FUSIONS`.
        sample_at: One spec per map of a module that maps the ``(*, D)``
            coordinates to the ``(*, d)`` grid coordinates where the map is
            sampled, for maps on another grid than the queries, such as the
            camera of an X-ray view of a volume. ``None``, for every map or one,
            samples at the coordinates themselves, so that map needs ``D`` grid
            axes.
    """

    def __init__(
        self,
        inr: ModuleSpec,
        in_features: int,
        out_features: int,
        cond_shape: Sequence[int] | Sequence[Sequence[int]],
        *,
        basis: ModuleSpec = ("cosine", {"num_components": 256}),
        combiner: ModuleSpec = ("cp", {"rank": 256}),
        bias: bool = False,
        fusion: ModuleSpec = "sum",
        sample_at: Sequence[ModuleSpec | None] | None = None,
    ) -> None:
        super().__init__(inr, in_features, out_features, cond_shape)
        self.encoding = FUTONEncoding(in_features, basis, combiner)
        num_features = self.encoding.out_features
        self.sampler = GridSampler()
        self.sample_at = _coordinate_maps(sample_at, self.cond_shape, in_features)
        sizes = dict(num_maps=len(self.cond_shape), out_features=num_features)
        self.fusion = build_module(fusion, FUSIONS, **sizes)
        check_sizes(self.fusion, "fusion", **sizes)
        self.projections = nn.ModuleList(
            nn.Linear(shape[0], num_features, bias=bias) for shape in self.cond_shape
        )
        self.norm = nn.LayerNorm(num_features)
        # Built last, so seeded initialization draws the gate's parameters first.
        self.inr = self.build_inr(num_features)

    def forward(self, coords: Tensor, conds: Tensor | Sequence[Tensor]) -> Tensor:
        conds = self.check_inputs(coords, conds)
        if coords.shape[0] != 1:
            raise ValueError(
                "the gate needs coordinates shared by the batch, (1, *, D)"
            )
        # One expression, so the (1, *, F) encoding and the (B, *, F) gate are freed
        # before the wrapped model reads the gated features as its coordinates.
        gated = self.norm(self.encoding(coords) * self._fuse_maps(coords[0], conds))
        if isinstance(self.inr, BaseModulator):
            return self.inr(gated, conds)
        return self.inr(gated)

    def _fuse_maps(self, coords: Tensor, conds: Sequence[Tensor]) -> Tensor:
        """Fuse the maps sampled at ``coords`` into the ``(B, *, F)`` gate."""
        factors = []
        for cond, projection, sample_at in zip(conds, self.projections, self.sample_at):
            # Project before sampling: interpolation is linear, and a map has far
            # fewer positions than the query grid.
            projected = projection(cond.movedim(1, -1)).movedim(-1, 1)  # (B, F, *grid)
            factors.append(self.sampler(sample_at(coords), projected))  # (B, *, F)
        return self.fusion(factors)


# Weight modulation --------------------------------------------------------------------


class WeightConditioner(nn.Module):
    """Predict a weight displacement from a ``(positions, channels)`` condition.

    Two paths read the condition's axes. The direct path maps positions to
    weight rows and channels to columns; the crossed path swaps those roles. A
    learned mixture and strength combine them::

        displacement = strength * ((1 - mix) * direct + mix * crossed)

    Args:
        in_shape: Condition shape ``(positions, channels)`` without the batch.
        out_shape: Weight shape ``(rows, columns)``.
        projection: Factory ``(in_features, out_features) -> nn.Module`` for each
            of the four axis projections.

    Shape:
        - condition: ``(B, positions, channels)``.
        - Output: additive displacement ``(B, rows, columns)``.
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_shape: tuple[int, int],
        projection: Callable[[int, int], nn.Module],
    ) -> None:
        super().__init__()
        self.in_shape = tuple(in_shape)
        self.out_shape = tuple(out_shape)
        positions, channels = self.in_shape
        rows, cols = self.out_shape

        self.positions_to_rows = projection(positions, rows)
        self.channels_to_cols = projection(channels, cols)
        self.channels_to_rows = projection(channels, rows)
        self.positions_to_cols = projection(positions, cols)
        # An equal blend and a small update at init, so both paths get gradients.
        self.mix_logit = nn.Parameter(torch.zeros(()))
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, condition: Tensor) -> Tensor:
        if condition.ndim != 3 or tuple(condition.shape[1:]) != self.in_shape:
            positions, channels = self.in_shape
            raise ValueError(
                f"condition must have shape (B, {positions}, {channels}), "
                f"got {tuple(condition.shape)}"
            )

        # Each projection acts on the last axis; transposes reach the other one.
        direct = self.channels_to_cols(condition)
        direct = self.positions_to_rows(direct.mT).mT

        crossed = self.positions_to_cols(condition.mT)
        crossed = self.channels_to_rows(crossed.mT).mT

        mix = torch.sigmoid(self.mix_logit)
        return self.strength * ((1 - mix) * direct + mix * crossed)


class LinearWeightConditioner(WeightConditioner):
    """Weight conditioner with linear axis projections."""

    def __init__(self, in_shape: tuple[int, int], out_shape: tuple[int, int]) -> None:
        super().__init__(in_shape, out_shape, projection=nn.Linear)


class MLPWeightConditioner(WeightConditioner):
    """Weight conditioner with two-layer MLP axis projections.

    ``expansion`` sets each hidden width relative to the projection's input.
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_shape: tuple[int, int],
        expansion: float = 2.0,
    ) -> None:
        def projection(in_features: int, out_features: int) -> nn.Module:
            hidden_features = int(in_features * expansion)
            return nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.GELU(),
                nn.Linear(hidden_features, out_features),
            )

        super().__init__(in_shape, out_shape, projection=projection)


# Weight conditioners, built with ``in_shape`` and ``out_shape``.
CONDITIONERS: dict[str, type[nn.Module]] = {
    "linear": LinearWeightConditioner,
    "mlp": MLPWeightConditioner,
}


def _select_weight_names(
    inr: nn.Module, names: Sequence[str] | None
) -> tuple[str, ...]:
    """Return the matrix parameters to displace, in the order given."""
    parameters = dict(inr.named_parameters())
    if names is None:
        names = [name for name, parameter in parameters.items() if parameter.ndim == 2]
    names = tuple(names)
    if not names:
        raise ValueError(f"{type(inr).__name__} has no matrix parameters to modulate")
    if len(set(names)) != len(names):
        raise ValueError("weight_names must not contain duplicates")
    for name in names:
        if name not in parameters:
            raise KeyError(
                f"{type(inr).__name__} has no parameter {name!r}; "
                f"expected one of {sorted(parameters)}"
            )
        if parameters[name].ndim != 2:
            raise ValueError(f"only matrix parameters can be modulated, got {name!r}")
    return names


class WeightDisplacement(BaseModulator):
    """Displace the wrapped model's weight matrices per image.

    The final map ``(B, E, *grid)`` is read as ``(B, positions, E)`` without
    pooling, whatever its grid rank; an ``(E,)`` map is one position. Each
    selected matrix gets its own conditioner, which predicts an additive
    displacement from it. The model then runs once per image under
    :func:`torch.func.vmap`, so every query in an image uses the same weights.

    Args:
        inr: Spec of the wrapped model, built with ``in_features``.
        in_features: Coordinate width, passed on unchanged.
        out_features: Prediction width.
        cond_shape: See :class:`BaseModulator`.
        conditioner: Spec from :data:`CONDITIONERS`, built for each selected
            matrix with ``in_shape=(positions, E)`` and ``out_shape=weight.shape``.
        weight_names: Matrices of the wrapped model to displace. Defaults to
            every matrix; biases and other vectors stay shared across images.
        init_scale: Scale of the selected matrices at initialization. With ``0``,
            each image's matrices start from its displacement alone.

    Note:
        Under ``vmap``, a FUTON basis in the wrapped model needs
        ``sparse=False``.
    """

    def __init__(
        self,
        inr: ModuleSpec,
        in_features: int,
        out_features: int,
        cond_shape: Sequence[int] | Sequence[Sequence[int]],
        *,
        conditioner: ModuleSpec = "mlp",
        weight_names: Sequence[str] | None = None,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__(inr, in_features, out_features, cond_shape)
        self.inr = self.build_inr(in_features)
        self.weight_names = _select_weight_names(self.inr, weight_names)
        if init_scale != 1.0:
            with torch.no_grad():
                for name in self.weight_names:
                    self.inr.get_parameter(name).mul_(init_scale)
        # A ModuleList in weight_names order: ModuleDict keys cannot contain dots.
        self.conditioners = nn.ModuleList()
        for name in self.weight_names:
            sizes = dict(
                in_shape=self.in_shape,
                out_shape=tuple(self.inr.get_parameter(name).shape),
            )
            module = build_module(conditioner, CONDITIONERS, **sizes)
            check_sizes(module, "conditioner", **sizes)
            self.conditioners.append(module)

    @property
    def in_shape(self) -> tuple[int, int]:
        """Shape ``(positions, E)`` of the flattened final map."""
        channels, *grid = self.cond_shape[-1]
        return (prod(grid), channels)

    def flatten_final_map(self, conds: Sequence[Tensor]) -> Tensor:
        """Return the final map as a ``(B, positions, E)`` view."""
        final = conds[-1]
        positions, channels = self.in_shape
        # reshape, not flatten(2): a channel-only (B, E) map is one position.
        return final.reshape(final.shape[0], channels, positions).mT

    def forward(self, coords: Tensor, conds: Tensor | Sequence[Tensor]) -> Tensor:
        conds = self.check_inputs(coords, conds)
        condition = self.flatten_final_map(conds)
        # attrgetter, not get_parameter: an outer weight modulator's
        # functional_call swaps the parameters for plain tensors.
        params = {
            name: attrgetter(name)(self.inr) + conditioner(condition)
            for name, conditioner in zip(self.weight_names, self.conditioners)
        }
        wraps_modulator = isinstance(self.inr, BaseModulator)

        def decode(
            params: dict[str, Tensor], coords: Tensor, conds: Sequence[Tensor]
        ) -> Tensor:
            if not wraps_modulator:
                return functional_call(self.inr, params, (coords,))
            # vmap strips the batch axis, so the wrapped modulator reads one image.
            args = (coords.unsqueeze(0), [cond.unsqueeze(0) for cond in conds])
            return functional_call(self.inr, params, args).squeeze(0)

        # Shared coordinates stay unbatched rather than being copied per image.
        if coords.shape[0] == 1:
            return vmap(decode, in_dims=(0, None, 0))(params, coords[0], conds)
        return vmap(decode)(params, coords, conds)


# Modulators, built with ``inr``, ``in_features``, ``out_features`` and ``cond_shape``.
MODULATORS: dict[str, type[nn.Module]] = {
    "futon": FUTONGate,
    "displacement": WeightDisplacement,
}


# ListModulators -----------------------------------------------------------------------


def _as_list(specs: ModuleSpec | Sequence[ModuleSpec]) -> list[ModuleSpec]:
    """Wrap a single spec, including a ``(key, params)`` pair, in a list."""
    is_pair = (
        isinstance(specs, Sequence)
        and len(specs) == 2
        and isinstance(specs[1], Mapping)
    )
    if isinstance(specs, str) or is_pair or not isinstance(specs, Sequence):
        return [specs]
    return list(specs)


class ListModulators(BaseModulator):
    """Compose modulators, innermost first.

    The first modulator wraps the INR and each later one wraps the model built so
    far, so ``[displacement, gate]`` is ``gate(displacement(inr))``: per-image
    weights reading a gated coordinate encoding. A weight modulator displaces
    every matrix of the model it wraps, including earlier modulators'.

    Args:
        inr: INR spec, built by the first modulator.
        in_features: Coordinate dimension ``D``.
        out_features: Prediction width.
        cond_shape: See :class:`BaseModulator`.
        modulators: A spec from :data:`MODULATORS`, or a sequence of them,
            innermost first. Pass specs rather than built modules, so that each
            can wrap the one before.
    """

    def __init__(
        self,
        inr: ModuleSpec,
        in_features: int,
        out_features: int,
        cond_shape: Sequence[int] | Sequence[Sequence[int]],
        modulators: ModuleSpec | Sequence[ModuleSpec],
    ) -> None:
        specs = _as_list(modulators)
        if not specs:
            raise ValueError(
                "give at least one modulator; without one the conditions are ignored"
            )
        if any(isinstance(spec, nn.Module) for spec in specs):
            raise TypeError("modulators must be specs, not built modules")
        for spec in specs:
            inr = partial(
                build_module, spec, MODULATORS, inr=inr, cond_shape=cond_shape
            )
        super().__init__(inr, in_features, out_features, cond_shape)
        self.inr = self.build_inr(in_features)

    @property
    def modulators(self) -> tuple[BaseModulator, ...]:
        """The modulators, innermost first; the first one's ``inr`` is the INR."""
        modulators, module = [], self.inr
        while isinstance(module, BaseModulator):
            modulators.append(module)
            module = module.inr
        return tuple(reversed(modulators))

    def forward(self, coords: Tensor, conds: Tensor | Sequence[Tensor]) -> Tensor:
        return self.inr(coords, conds)
