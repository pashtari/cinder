"""
Conditional FUTON: Fourier Tensor Network for *conditional* implicit neural
representations.

This is a self-contained port of the modular FUTON design, specialized for
conditional decoding. A model is the composition of four stages:

1. **Basis** -- maps coordinates in ``[-1, 1]`` of shape ``(*, C)`` to a tuple
   of ``C`` per-coordinate feature tensors ``(*, K_c)``.
2. **Combiner** -- fuses the per-coordinate features into a single coordinate
   feature tensor ``(*, F)`` (Hadamard product or a tensor-network contraction).
3. **Conditioner** -- folds a conditioning map ``z`` of shape
   ``(B, cond_dim, *grid)`` into the coordinate features, lifting ``(*, F)`` to
   ``(B, *, F)`` by interpolating ``z`` at the coordinates and gating with it.
4. **Decoder** -- maps the conditioned features to the output ``(B, *, out_features)``.

Compared to the unconditional FUTON, only the conditioner is new: it gates *any*
combiner's output with a learned projection of ``z``, read off the image at the
queried coordinate. With the CP combiner this is exactly the canonical-polyadic
decomposition over coordinates *and* condition.
"""

from typing import cast
import math
from collections.abc import Sequence
from functools import reduce
from operator import mul


import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .rcs_matrix import RCSMatrix
from .utils import ModuleSpec, broadcast, build_module

__all__ = [
    "CosineBasis",
    "TriangleBasis",
    "SincBasis",
    "LanczosBasis",
    "Hadamard",
    "CanonicalPolyadicCombiner",
    "CPCombiner",
    "MLP",
    "BASES",
    "COMBINERS",
    "DECODERS",
    "ConditionalFUTON",
]


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------


class _Basis(nn.Module):
    """Base class for feature bases.

    A basis maps coordinates of shape ``(*, C)`` to a tuple of ``C``
    per-coordinate feature tensors. Subclasses implement :meth:`_eval_axis`
    (evaluate the basis for one coordinate axis) and expose :attr:`out_size`,
    the number of features produced per coordinate; the shared :meth:`forward`
    and the lookup-table cache are provided here.

    On-grid caching:
        Coordinates produced on a regular grid (``linspace(*domain, size)`` per
        axis) take only ``size`` distinct values, so the basis can be evaluated
        once per grid point and stored in a lookup table. When ``grid_size`` is
        passed to a subclass, :meth:`forward` gathers the table for coordinates
        that lie on the grid and computes the rest on the fly (see
        :meth:`_eval_axis_cached`). The output is exact either way. Gradients
        w.r.t. on-grid coordinates are zero; the bases have no learnable
        parameters, so this is harmless.
    """

    # Coordinate domain the basis is defined on; subclasses may override.
    domain: tuple[float, float] = (-1.0, 1.0)
    # Tolerance (in grid-index units) for treating a coordinate as on-grid.
    _grid_atol: float = 1e-4

    def __init__(self, in_features: int, normalize: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.normalize = normalize
        self.grid_size: list[int] | None = None

    def _normalize(self, features: Tensor) -> Tensor:
        """L2-normalize along the feature axis, or pass through if disabled."""
        return F.normalize(features, dim=-1) if self.normalize else features

    @property
    def out_size(self) -> list[int]:
        raise NotImplementedError("Subclasses must define the out_size property.")

    def _eval_axis(self, coord: Tensor, axis: int) -> Tensor:
        """Evaluate the basis for one coordinate axis: ``(*,) -> (*, out_size[axis])``."""
        raise NotImplementedError

    def _init_cache(self, grid_size: int | Sequence[int] | None) -> None:
        """Precompute per-axis lookup tables for on-grid queries (no-op if ``None``).

        Subclasses call this at the end of ``__init__`` (after their buffers are
        registered, since the tables are built by evaluating the basis).
        """
        if grid_size is None:
            return
        self.grid_size = broadcast(grid_size, self.in_features, "grid_size")
        lo, hi = self.domain
        for axis, size in enumerate(self.grid_size):
            table = self._eval_axis(torch.linspace(lo, hi, size), axis)
            self.register_buffer(f"_cache_{axis}", table, persistent=False)

    def _eval_axis_cached(self, coord: Tensor, axis: int, size: int) -> Tensor:
        """Look up on-grid coordinates in the cache; compute the rest on the fly.

        A coordinate is "on-grid" when it maps to an integer ``linspace`` index
        (within :attr:`_grid_atol`); those are gathered from the precomputed
        table, while off-grid coordinates are evaluated exactly.
        """
        if size == 1:  # a one-point cache buys nothing; just compute.
            return self._eval_axis(coord, axis)

        lo, hi = self.domain
        pos = (coord - lo) / (hi - lo) * (size - 1)  # continuous grid index
        idx = pos.round()
        on_grid = (
            ((pos - idx).abs() < self._grid_atol)
            & (pos >= -self._grid_atol)
            & (pos <= size - 1 + self._grid_atol)
        )

        table = cast(Tensor, getattr(self, f"_cache_{axis}"))
        out = table[idx.clamp(0, size - 1).long()]  # cheap gather everywhere
        if not bool(on_grid.all()):  # evaluate the off-grid points exactly
            off = ~on_grid
            out[off] = self._eval_axis(coord[off], axis)
        return out

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        if self.grid_size is not None:
            return tuple(
                self._eval_axis_cached(x[..., axis], axis, size)
                for axis, size in enumerate(self.grid_size)
            )
        return tuple(
            self._eval_axis(x[..., axis], axis) for axis in range(self.in_features)
        )


class CosineBasis(_Basis):
    r"""Cosine feature mapping.

    Maps each coordinate to :math:`[\cos(0), \cos(\pi u), \dots, \cos((K-1)\pi u)]`
    with :math:`u = (x + 1)/2 \in [0, 1]`, then L2-normalizes.

    Args:
        in_features: Number of input coordinates (C).
        num_components: Frequency count per coordinate. An int applies the
            same K to all coordinates; a sequence sets per-coordinate counts.
        normalize: L2-normalize the per-coordinate features (default ``True``).
        grid_size: Optional on-grid lookup-table acceleration; see
            :class:`_Basis`. An int applies the same grid size to all
            coordinates; a sequence sets per-coordinate sizes.

    Shape:
        - Input: :math:`(*, C)` with values in [-1, 1].
        - Output: Tuple of C tensors; the c-th tensor has shape :math:`(*, K_c)`.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(in_features, normalize)
        self.num_components = broadcast(num_components, in_features, "num_components")

        omega = math.pi * torch.arange(max(self.num_components))
        self.register_buffer("omega", omega, persistent=False)
        self._init_cache(grid_size)

    @property
    def out_size(self) -> list[int]:
        return self.num_components

    def _eval_axis(self, coord: Tensor, axis: int) -> Tensor:
        coord = ((coord + 1) / 2).unsqueeze(-1)  # map [-1, 1] -> [0, 1]
        omega = cast(Tensor, self.omega)[: self.num_components[axis]]
        return self._normalize(torch.cos(omega * coord))


class SincBasis(_Basis):
    r"""Cardinal-sine (Whittaker-Shannon) basis.

    Each coordinate is expanded into ``K`` sinc functions whose centers are
    uniformly spaced over :math:`[-1, 1]`. The c-th feature is

    .. math:: S_c(x) = \operatorname{sinc}\!\left((x - \mu_c) / w\right),

    with :math:`\operatorname{sinc}(t) = \sin(\pi t) / (\pi t)`,
    :math:`\mu_c = -1 + 2c/(K-1)` and the center spacing :math:`w = 2/(K-1)`.
    Each sinc equals 1 at its own center and 0 at every other center (its zero
    crossings land on the grid), giving the ideal band-limited interpolation
    basis.

    Unlike :class:`TriangleBasis`, sinc has global (slowly decaying) support, so
    the dense output is not sparse.

    Args:
        in_features: Number of input coordinates (C).
        num_components: Number of sinc functions per coordinate (each K >= 2).
            An int applies the same K to all coordinates; a sequence sets
            per-coordinate counts.
        normalize: L2-normalize the per-coordinate features (default ``True``).
            Pass ``False`` to preserve the cardinal interpolation property.
        grid_size: Optional on-grid lookup-table acceleration; see :class:`_Basis`.

    Shape:
        - Input: :math:`(*, C)` with values in [-1, 1].
        - Output: Tuple of C tensors; the c-th tensor has shape :math:`(*, K_c)`.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(in_features, normalize)
        self.num_components = broadcast(num_components, in_features, "num_components")
        if any(k < 2 for k in self.num_components):
            raise ValueError(
                f"Expected every num_components >= 2, got {self.num_components}"
            )
        self.width = [2.0 / (k - 1) for k in self.num_components]

        # Per-axis centers, as in TriangleBasis: linspace(-1, 1, K_c) per coordinate.
        for axis, k in enumerate(self.num_components):
            centers = torch.linspace(-1.0, 1.0, k)
            self.register_buffer(f"centers_{axis}", centers, persistent=False)
        self._init_cache(grid_size)

    @property
    def out_size(self) -> list[int]:
        return self.num_components

    def _eval_axis(self, coord: Tensor, axis: int) -> Tensor:
        centers = cast(Tensor, getattr(self, f"centers_{axis}"))
        sinc = torch.sinc((coord.unsqueeze(-1) - centers) / self.width[axis])
        return self._normalize(sinc)


class _LocalBasis(_Basis):
    r"""Base class for compactly supported (row-sparse) interpolation bases.

    Subclasses place ``K`` copies of a kernel on ``linspace(-1, 1, K)`` and
    evaluate it at the grid-relative offset :math:`t = (x - \mu_c)/w`, where
    :math:`w = 2/(K-1)` is the center spacing -- so ``t`` is just the distance
    measured in grid-index units. The kernel vanishes outside :math:`|t| < a`,
    where :attr:`bandwidth` ``= a`` is its radius in grid steps, hence at most
    ``2a`` *consecutive* centers are nonzero: exactly the row-contiguous
    sparse pattern of :class:`RCSMatrix`. Subclasses
    implement :meth:`_kernel`.

    Sparse mode:
        :meth:`forward` returns :class:`RCSMatrix` features of shape
        ``(M, K)`` holding only the ``(M, 2a)`` taps, so the dense block is
        never built and the combiner's factor-matrix products contract
        directly on the compact form. Evaluation is ``O(M a)`` rather than
        ``O(M K)``, and the output matches ``sparse=False`` to floating-point
        precision.

    Note:
        Sparse mode flattens the leading dimensions, so features are
        ``(M, K)`` with ``M = prod(batch_shape)``; :class:`FUTON` flattens its
        input and restores the shape on the output, so this is invisible in
        normal use. On-grid caching (see :class:`_Basis`) applies to dense
        mode only, so ``grid_size`` is ignored when ``sparse``.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        bandwidth: int,
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
        sparse: bool = True,
    ) -> None:
        super().__init__(in_features, normalize)
        self.num_components = broadcast(num_components, in_features, "num_components")
        if bandwidth < 1:
            raise ValueError(f"Expected bandwidth >= 1, got {bandwidth}")
        # Validate against the argument, not an attribute, so the checks stay
        # correct however the assignments below are reordered.
        num_taps = 2 * bandwidth
        if any(k < num_taps for k in self.num_components):
            raise ValueError(
                f"Expected every num_components >= 2 * bandwidth = {num_taps}, "
                f"got {self.num_components}"
            )
        self.bandwidth = int(bandwidth)
        self.sparse = bool(sparse)

        if not self.sparse:
            self._init_cache(grid_size)

    @property
    def out_size(self) -> list[int]:
        return self.num_components

    def _kernel(self, t: Tensor) -> Tensor:
        """Evaluate the kernel at grid-relative offsets ``t``.

        Must return zero wherever ``|t| >= bandwidth``.
        """
        raise NotImplementedError

    def _grid_pos(self, coord: Tensor, axis: int) -> Tensor:
        """Continuous grid index of ``coord``, i.e. ``(x - mu_0) / w`` in [0, K-1]."""
        lo, hi = self.domain
        return (coord - lo) / (hi - lo) * (self.num_components[axis] - 1)

    def _eval_axis(self, coord: Tensor, axis: int) -> Tensor:
        # Offsets to every center, in grid-index units.
        centers = torch.arange(
            self.num_components[axis], device=coord.device, dtype=coord.dtype
        )
        t = self._grid_pos(coord, axis).unsqueeze(-1) - centers
        return self._normalize(self._kernel(t))

    def _eval_axis_sparse(self, coord: Tensor, axis: int) -> RCSMatrix:
        """Evaluate only the ``2a`` in-support taps, as an ``(M, K)`` RCSMatrix."""
        K, a = self.num_components[axis], self.bandwidth
        pos = self._grid_pos(coord.reshape(-1), axis)
        # The in-support centers are floor(pos) - a + 1 ... floor(pos) + a.
        # Clamping keeps the segment inside [0, K); taps the clamp pushes out
        # of the support evaluate to zero, so the result stays exact.
        start = (pos.floor().long() - (a - 1)).clamp_(0, K - 2 * a)
        offsets = torch.arange(2 * a, device=coord.device)
        t = pos.unsqueeze(-1) - (start.unsqueeze(-1) + offsets).to(pos.dtype)
        return RCSMatrix(self._normalize(self._kernel(t)), start, K)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        if not self.sparse:
            return super().forward(x)
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        return tuple(
            self._eval_axis_sparse(x[..., axis], axis) for axis in range(self.in_features)
        )


class TriangleBasis(_LocalBasis):
    r"""Triangle-pulse (tent / hat) basis.

    Each coordinate is expanded into ``K`` triangular pulses whose centers are
    uniformly spaced over :math:`[-1, 1]` and whose half-width equals the center
    spacing :math:`2 / (K - 1)`. The c-th feature is

    .. math:: T_c(x) = \max\!\left(0,\ 1 - |x - \mu_c| / w\right),

    with :math:`\mu_c = -1 + 2c/(K-1)` and :math:`w = 2/(K-1)`; each pulse
    reaches zero at its neighbors' centers, so the (un-normalized) pulses form a
    partition of unity (piecewise-linear interpolation).

    Equivalence to feature grids:
        With ``normalize=False`` the un-normalized tent expansion followed by a
        per-axis linear map (e.g. :class:`CanonicalPolyadicCombiner`) is exactly
        linear interpolation (``align_corners=True``) of a learnable 1D feature
        grid of resolution ``K``. A FUTON with this basis, a CP combiner, and an
        MLP decoder is therefore a reparametrization of TensoRF-CP (Chen et al.,
        ECCV 2022). L2-normalization breaks the identity (a normalized tent is
        no longer a partition of unity).

    Row-contiguous sparsity:
        Only the two pulses bracketing a coordinate are nonzero, and they are
        adjacent: this is the ``bandwidth = 1`` case of :class:`_LocalBasis`
        (two taps), including its sparse mode.

    Args:
        in_features: Number of input coordinates (C).
        num_components: Number of triangle pulses per coordinate (each K >= 2).
            An int applies the same K to all coordinates; a sequence sets
            per-coordinate counts.
        normalize: L2-normalize the per-coordinate features (default ``True``).
            Pass ``False`` for the partition-of-unity form (see above).
        grid_size: Optional on-grid lookup-table acceleration for dense mode;
            see :class:`_Basis`.
        sparse: Return compact :class:`RCSMatrix` features (default ``True``).

    Shape:
        - Input: :math:`(*, C)` with values in [-1, 1].
        - Output: Tuple of C tensors; the c-th has shape :math:`(*, K_c)`
          (dense) or is an :math:`(M, K_c)` ``RCSMatrix`` (sparse).
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
        sparse: bool = True,
    ) -> None:
        # bandwidth=1: the two pulses bracketing the coordinate.
        super().__init__(in_features, num_components, 1, normalize, grid_size, sparse)

    def _kernel(self, t: Tensor) -> Tensor:
        return (1.0 - t.abs()).clamp(min=0.0)


class LanczosBasis(_LocalBasis):
    r"""Lanczos (windowed-sinc) basis -- a compactly supported :class:`SincBasis`.

    Each coordinate is expanded into ``K`` Lanczos kernels centered on
    ``linspace(-1, 1, K)``. Writing :math:`t = (x - \mu_c) / w` for the
    grid-relative offset (center spacing :math:`w = 2/(K-1)`) and :math:`a`
    for the :attr:`bandwidth`, the c-th feature is

    .. math::
        \Lambda_c(x) = \begin{cases}
            \operatorname{sinc}(t)\,\operatorname{sinc}(t/a) & |t| < a, \\
            0 & \text{otherwise,}
        \end{cases}

    a sinc windowed by the central lobe of a second, :math:`a` times wider
    sinc. :class:`SincBasis` is the :math:`a \to \infty` limit. Like sinc, the
    kernel is cardinal -- it is 1 at its own center and 0 at every other
    center -- so with ``normalize=False`` the un-normalized expansion is
    exactly Lanczos resampling, the standard high-quality alternative to the
    linear interpolation of :class:`TriangleBasis` (``bandwidth = 1``).

    Row-contiguous sparsity:
        Only the :math:`2a` centers nearest to ``x`` are in support, and they
        are *consecutive*, so sparse mode applies; see :class:`_LocalBasis`.

    Args:
        in_features: Number of input coordinates (C).
        num_components: Number of Lanczos kernels per coordinate (each
            ``K >= 2 * bandwidth``). An int applies the same K to all
            coordinates; a sequence sets per-coordinate counts.
        bandwidth: Kernel radius :math:`a >= 1`, giving ``2a`` taps of support
            (``bandwidth=2`` is the classic Lanczos-2 with 4 taps,
            ``bandwidth=3`` Lanczos-3 with 6). Larger values approach
            :class:`SincBasis`.
        normalize: L2-normalize the per-coordinate features (default ``True``).
            Pass ``False`` to preserve the cardinal interpolation property.
        grid_size: Optional on-grid lookup-table acceleration for dense mode;
            see :class:`_Basis`.
        sparse: Return compact :class:`RCSMatrix` features (default ``True``).

    Shape:
        - Input: :math:`(*, C)` with values in [-1, 1].
        - Output: Tuple of C tensors; the c-th has shape :math:`(*, K_c)`
          (dense) or is an :math:`(M, K_c)` ``RCSMatrix`` (sparse).
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        bandwidth: int = 2,
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
        sparse: bool = True,
    ) -> None:
        super().__init__(
            in_features, num_components, bandwidth, normalize, grid_size, sparse
        )

    def _kernel(self, t: Tensor) -> Tensor:
        a = self.bandwidth
        return torch.where(
            t.abs() < a, torch.sinc(t) * torch.sinc(t / a), t.new_zeros(())
        )


# ---------------------------------------------------------------------------
# Combiners (mostly based on tensor networks)
# ---------------------------------------------------------------------------


def _to_dense(features: Tensor) -> Tensor:
    """Materialize compact (RCS) features; pass dense features through."""
    return features.to_dense() if isinstance(features, RCSMatrix) else features


def _project(features: Tensor, linear: nn.Linear) -> Tensor:
    """Apply ``linear`` to dense or compact (RCS) features.

    Compact features are contracted on their row-sparse form, so the dense
    ``(M, K)`` block is never materialized.
    """
    if isinstance(features, RCSMatrix):
        out = features @ linear.weight.transpose(0, 1)
        return out if linear.bias is None else out + linear.bias
    return linear(features)


class _Combiner(nn.Module):
    """Base class for feature combiners.

    A combiner fuses the per-coordinate features produced by a basis (a
    sequence of ``C`` tensors with feature sizes :attr:`in_size`) into a single
    feature tensor. Subclasses implement :meth:`forward` and expose
    :attr:`out_features`.
    """

    def __init__(self, in_size: Sequence[int]) -> None:
        super().__init__()
        self.in_size = list(in_size)

    @property
    def out_features(self) -> int:
        raise NotImplementedError("Subclasses must define the out_features property.")

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        raise NotImplementedError


class HadamardCombiner(_Combiner):
    """Element-wise product of features, without learnable parameters.

    Args:
        in_size: Per-axis feature sizes, which must all be equal.

    Raises:
        ValueError: If the feature sizes differ.
    """

    def __init__(self, in_size: Sequence[int]) -> None:
        super().__init__(in_size)
        if not all(k == self.in_size[0] for k in self.in_size):
            raise ValueError(f"all elements of in_size must be equal, got {self.in_size}")

    @property
    def out_features(self) -> int:
        return self.in_size[0]

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        # RCSMatrix implements only matmul (any other op raises), so compact
        # features are densified before the element-wise product.
        return reduce(mul, (_to_dense(phi) for phi in features))


class CanonicalPolyadicCombiner(_Combiner):
    """Canonical polyadic (CP / CANDECOMP-PARAFAC) feature combiner.

    Projects each per-coordinate feature into a shared rank-``R`` space with a
    learnable linear map, then takes the Hadamard product across coordinates.

    Args:
        in_size: Per-axis feature sizes ``K_c`` produced by the basis.
        rank: Shared CP rank ``R``.
        bias: Whether each per-axis linear map has a bias.
    """

    def __init__(self, in_size: Sequence[int], rank: int, bias: bool = False) -> None:
        super().__init__(in_size)
        self.rank = rank
        self.linears = nn.ModuleList(
            [nn.Linear(k, rank, bias=bias) for k in self.in_size]
        )
        self.hadamard = HadamardCombiner([rank] * len(self.in_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the factor matrices with Kaiming-uniform sampling."""
        for linear in self.linears:
            nn.init.kaiming_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    @property
    def out_features(self) -> int:
        return self.rank

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        projected = (_project(phi, proj) for phi, proj in zip(features, self.linears))
        return self.hadamard(projected)


class TensorRingCombiner(_Combiner):
    """Tensor-ring feature combiner.

    Contracts each per-coordinate feature with a learnable core tensor to form
    an ``R x R`` transfer matrix, chain-multiplies the matrices, and flattens
    the result to a length-``R^2`` vector.

    Args:
        in_size: Per-axis feature sizes ``K_c`` produced by the basis.
        rank: Tensor-ring rank ``R``; the combiner outputs ``R ** 2`` features.
    """

    def __init__(self, in_size: Sequence[int], rank: int) -> None:
        super().__init__(in_size)
        self.rank = rank
        self.cores = nn.ParameterList(
            [nn.Parameter(torch.empty(rank, k, rank)) for k in self.in_size]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the core tensors with Kaiming-uniform sampling."""
        for core in self.cores:
            nn.init.kaiming_uniform_(core)

    @property
    def out_features(self) -> int:
        return self.rank**2

    @staticmethod
    def _transfer(phi: Tensor, core: Tensor) -> Tensor:
        """Contract one feature with its core: ``(..., K) x (R, K, R) -> (..., R, R)``."""
        if isinstance(phi, RCSMatrix):
            # Fold the core's two rank axes into one so the contraction is a
            # single (M, K) @ (K, R^2) product on the compact form.
            rank, k, _ = core.shape
            flat = phi @ core.permute(1, 0, 2).reshape(k, rank * rank)
            return flat.unflatten(-1, (rank, rank))
        return torch.einsum("...k,rks->...rs", phi, core)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        matrices = (self._transfer(phi, core) for phi, core in zip(features, self.cores))
        # Chain-multiply the transfer matrices, then flatten: (..., R, R) -> (..., R^2)
        state = reduce(torch.matmul, matrices)
        return state.flatten(start_dim=-2)


# ---------------------------------------------------------------------------
# Conditioner
# ---------------------------------------------------------------------------


class Conditioner(nn.Module):
    r"""Fold a spatial conditioning map into the combined coordinate features.

    The condition is an image-shaped embedding ``z`` of shape
    ``(B, cond_dim, *grid)`` -- e.g. an encoder's feature map. It is projected
    into the combiner's feature space, interpolated at the query coordinates,
    and multiplied with the coordinate features: a spatially varying,
    FiLM-style gate that lifts ``(*, F)`` to ``(B, *, F)``,

    .. math:: y_{b,\dots,f} = \phi_{\dots,f}\;\bigl(W z_b\bigr)(x_{\dots})_f .

    With a CP combiner this is simply one more canonical-polyadic factor, now
    read off the image *at* the coordinate rather than shared across it.

    Note:
        Interpolation is linear, so projecting the grid before sampling gives
        exactly the same result as sampling and then projecting, but runs the
        ``cond_dim -> out_features`` map over ``prod(grid)`` positions instead
        of over every queried coordinate -- and only the projected features
        are gathered. On a 12x12 grid feeding 384x384 queries that measured
        ~5x faster and ~2.7x lighter. The orders stop being equivalent if the
        projection is made non-linear, which would require sampling first.

    Args:
        cond_dim: Channel count ``E`` of the conditioning map.
        out_features: Combiner feature dimension ``F`` to modulate.
        bias: Whether the projection learns an additive bias.

    Shape:
        - x: :math:`(*, C)` in ``[-1, 1]``, features: :math:`(*, F)`,
          z: :math:`(B, E, *grid)` with ``len(grid) == C``.
        - Output: :math:`(B, *, F)`.
    """

    def __init__(self, cond_dim: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.out_features = out_features
        self.proj = nn.Linear(cond_dim, out_features, bias=bias)

    @staticmethod
    def _sample(x: Tensor, z: Tensor) -> Tensor:
        """Interpolate ``z`` at ``x``: ``(*, C) x (B, F, *grid) -> (B, *, F)``."""
        if z.ndim != x.shape[-1] + 2:
            raise ValueError(
                f"`z` must have {x.shape[-1]} spatial dimensions to match the "
                f"coordinates, got shape {tuple(z.shape)}"
            )
        batch, features = z.shape[:2]
        # grid_sample has no bfloat16 CUDA kernel (as of torch 2.1), so under
        # autocast the interpolation is done in fp32 and cast back. It is a
        # gather with bilinear weights, not a matmul, so nothing is gained by
        # running it in low precision anyway.
        out_dtype = z.dtype
        if torch.is_autocast_enabled() and z.is_cuda:
            z = z.float()
        # grid_sample reads the last axis as (x, y[, z]) -- the reverse of the
        # "ij" coordinate order -- and wants a grid shaped like its output, so
        # the queries are flattened onto a single sampling axis.
        grid = x.flip(-1).reshape(-1, *(1,) * (x.shape[-1] - 1), x.shape[-1])
        grid = grid.to(z.dtype).expand(batch, *grid.shape)
        with torch.autocast(device_type="cuda", enabled=False):
            sampled = F.grid_sample(
                z, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
        return (
            sampled.reshape(batch, features, -1)
            .mT.reshape(batch, *x.shape[:-1], features)
            .to(out_dtype)
        )

    def forward(self, x: Tensor, features: Tensor, z: Tensor) -> Tensor:
        z = self.proj(z.movedim(1, -1)).movedim(-1, 1)  # (B, E, *grid) -> (B, F, *grid)
        return features * self._sample(x, z)  # (*, F) * (B, *, F) -> (B, *, F)


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Multi-layer perceptron with a configurable activation.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        hidden_layers: Number of hidden Linear+activation blocks (default 2),
            so the module holds ``hidden_layers + 1`` linear layers.
        hidden_features: Hidden width (defaults to ``in_features``).
        activation: Activation between layers (defaults to :class:`~torch.nn.ReLU`).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int = 2,
        hidden_features: int | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features or in_features
        self.activation = activation if activation is not None else nn.ReLU()

        layers: list[nn.Module] = [
            nn.Linear(in_features, self.hidden_features),
            self.activation,
        ]
        for _ in range(hidden_layers - 1):
            layers += [
                nn.Linear(self.hidden_features, self.hidden_features),
                self.activation,
            ]
        layers.append(nn.Linear(self.hidden_features, out_features))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# Aliases
Hadamard = HadamardCombiner
CPCombiner = CanonicalPolyadic = CanonicalPolyadicCombiner
TRCombiner = TensorRing = TensorRingCombiner

BASES: dict[str, type[nn.Module]] = {
    "cosine": CosineBasis,
    "cos": CosineBasis,
    "triangle": TriangleBasis,
    "tent": TriangleBasis,
    "hat": TriangleBasis,
    "sinc": SincBasis,
    "lanczos": LanczosBasis,
}

COMBINERS: dict[str, type[nn.Module]] = {
    "cp": CPCombiner,
    "tr": TRCombiner,
    "ring": TRCombiner,
}

DECODERS: dict[str, type[nn.Module]] = {
    "linear": nn.Linear,
    "mlp": MLP,
}

# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ConditionalFUTON(nn.Module):
    r"""Conditional Fourier Tensor Network (FUTON).

    Decodes a per-coordinate output conditioned on an external vector ``z``:
    ``basis(x) -> combiner -> conditioner(z) -> decoder``. See the module
    docstring for the stage-by-stage description.

    Args:
        in_features: Dimensionality of the input coordinates (C).
        out_features: Dimensionality of the output (D).
        cond_dim: Channel count (E) of the conditioning map.
        basis: Spec for the feature basis (see :data:`BASES`).
        combiner: Spec for the feature combiner (see :data:`COMBINERS`).
        decoder: Spec for the decoder (see :data:`DECODERS`).
        output_activation: Applied to the decoder output. Defaults to identity
            (the conditional decoder typically produces logits).

    Each spec is an :class:`nn.Module` instance, a registry key / nn.Module
    subclass used with default params (e.g. ``"linear"``), or a
    ``(name_or_type, params)`` pair, e.g. ``("cos", {"num_components": 64})``.

    Shape:
        - x: :math:`(*, C)` coordinates in ``[-1, 1]``; z: :math:`(B, E, *grid)`
          with ``len(grid) == C``.
        - Output: :math:`(B, *, D)`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cond_dim: int,
        basis: ModuleSpec = ("cosine", {"num_components": 256}),
        combiner: ModuleSpec = ("cp", {"rank": 256}),
        decoder: ModuleSpec = ("mlp", {"hidden_layers": 1}),
        output_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cond_dim = cond_dim

        self.basis = build_module(basis, BASES, in_features)
        self.combiner = build_module(combiner, COMBINERS, self.basis.out_size)
        self.norm = nn.LayerNorm(self.combiner.out_features)
        self.conditioner = Conditioner(cond_dim, self.combiner.out_features)
        self.decoder = build_module(
            decoder, DECODERS, self.combiner.out_features, out_features
        )
        self.output_activation = output_activation or nn.Identity()

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        # x: (N, 2) or (H, W, 2) coordinates in [-1, 1]
        # z: (B, E, H', W'), a batch of conditioning 2D image embeddings
        # Flatten the query axes into one: the row-sparse bases collapse them
        # anyway, so this puts every basis on the same (M, C) contract. The
        # original coordinate shape is restored on the output.
        coord_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])  # (M, C)

        pos_feats = self.basis(x)  # [(M, K_1), ..., (M, K_C)]
        combined = self.combiner(pos_feats)  # (M, R), R = combiner.out_features

        conditioned = self.conditioner(x, combined, z)  # (B, M, R)

        conditioned = self.norm(conditioned)
        out = self.decoder(conditioned)  # (B, M, D)
        out = self.output_activation(out)
        return out.reshape(out.shape[0], *coord_shape, out.shape[-1])  # (B, *, D)
