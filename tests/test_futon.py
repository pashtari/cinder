"""Unit tests for the conditional FUTON building blocks.

Focused coverage for the two pieces added/ported in the rewrite: the
:class:`Conditioner` (a spatially varying gate that interpolates a conditioning
feature map at the query coordinates) and the on-grid lookup-table cache shared
by all bases.
"""

import pytest
import torch

from cinder.models.futon import (
    CanonicalPolyadicCombiner,
    Conditioner,
    ConditionalFUTON,
    CosineBasis,
)


# ---------------------------------------------------------------------------
# Conditioner
# ---------------------------------------------------------------------------


def test_conditioner_shape_and_gate():
    """Lifts (*, F) + (B, E, H', W') -> (B, *, F), gating by the sampled condition."""
    torch.manual_seed(0)
    B, H, W, F, E = 3, 4, 5, 7, 6
    cond = Conditioner(cond_dim=E, out_features=F)

    features = torch.randn(H, W, F)  # combined coordinate features
    z = torch.randn(B, E, 3, 8)  # conditioning feature map
    coords = _grid(H, W)

    out = cond(coords, features, z)
    assert out.shape == (B, H, W, F)

    # Reference: sample the map at the coordinates, project, then gate.
    sampled = Conditioner._sample(coords, z)  # (B, H, W, E)
    expected = features * cond.proj(sampled)
    assert torch.allclose(out, expected, atol=1e-5)

    # Flat coordinates keep the same contract: (N, 2) -> (B, N, F).
    flat = coords.reshape(-1, 2)[:9]
    assert cond(flat, torch.randn(9, F), z).shape == (B, 9, F)


def test_conditioner_samples_at_the_right_coordinates():
    """coords[..., 0] indexes rows and coords[..., 1] columns, exactly on-grid."""
    # Channel 0 is a pure row ramp, channel 1 a pure column ramp, so a correctly
    # sampled point reproduces its own (row, col) coordinate.
    z = torch.zeros(1, 2, 3, 5)
    z[0, 0] = torch.linspace(-1, 1, 3)[:, None]
    z[0, 1] = torch.linspace(-1, 1, 5)[None, :]
    probe = torch.tensor([[-1.0, 1.0], [1.0, -1.0], [0.0, 0.0], [0.5, -0.25]])

    sampled = Conditioner._sample(probe, z)[0]  # (4, 2)
    assert torch.allclose(sampled, probe, atol=1e-6)

    # At the map's own grid nodes the interpolation is exact (align_corners).
    nodes = _grid(3, 5)
    assert torch.allclose(Conditioner._sample(nodes, z), z.movedim(1, -1), atol=1e-6)


def test_conditioner_has_no_bias_by_default():
    cond = Conditioner(cond_dim=8, out_features=4)
    assert cond.proj.bias is None
    # z = 0 -> projection 0 -> output 0 (pure multiplicative gate).
    out = cond(_grid(3, 4), torch.randn(3, 4, 4), torch.zeros(5, 8, 2, 2))
    assert torch.count_nonzero(out) == 0


def test_conditioner_matches_cp_factor_semantics():
    """With a CP combiner, conditioning is one more (spatial) CP factor."""
    torch.manual_seed(0)
    B, R, E = 2, 5, 8
    in_size = [4, 6]
    combiner = CanonicalPolyadicCombiner(in_size, rank=R, bias=False)
    cond = Conditioner(cond_dim=E, out_features=R)

    coords = _grid(3, 3)
    feats = [torch.randn(3, 3, k) for k in in_size]
    z = torch.randn(B, E, 4, 4)

    out = cond(coords, combiner(feats), z)  # (B, 3, 3, R)

    # Manual CP: Hadamard of the per-axis rank-R projections and the condition
    # factor, the latter read off the map at each coordinate.
    proj_coords = [lin(f) for f, lin in zip(feats, combiner.linears)]
    proj_cond = cond.proj(Conditioner._sample(coords, z))  # (B, 3, 3, R)
    expected = proj_coords[0] * proj_coords[1] * proj_cond
    assert torch.allclose(out, expected, atol=1e-5)


def test_conditioner_projection_commutes_with_sampling():
    """Projecting the grid then sampling equals sampling then projecting.

    Interpolation is linear, so the two orders agree; the conditioner projects
    first (cheaper -- see its docstring), and this pins the equivalence.
    """
    torch.manual_seed(0)
    cond = Conditioner(cond_dim=6, out_features=4)
    coords, z = _grid(7, 5), torch.randn(2, 6, 3, 4)

    project_first = cond(coords, torch.ones(7, 5, 4), z)  # gate by ones -> condition
    sample_first = cond.proj(Conditioner._sample(coords, z))
    assert torch.allclose(project_first, sample_first, atol=1e-5)


# ---------------------------------------------------------------------------
# On-grid caching
# ---------------------------------------------------------------------------


def _grid(h, w):
    ys = torch.linspace(-1, 1, h)
    xs = torch.linspace(-1, 1, w)
    return torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)  # (h, w, 2)


def test_cache_exact_on_full_grid():
    """Cached basis output is bit-for-bit equal to the uncached output on-grid."""
    H, W = 12, 9
    plain = CosineBasis(2, num_components=16)
    cached = CosineBasis(2, num_components=16, grid_size=[H, W])
    assert cached.grid_size == [H, W]

    coords = _grid(H, W)
    for p, c in zip(plain(coords), cached(coords)):
        assert torch.equal(p, c)


def test_cache_recomputes_off_grid_points_exactly():
    """A mix of on-grid and off-grid coords still yields the exact basis."""
    H = W = 8
    plain = CosineBasis(2, num_components=10)
    cached = CosineBasis(2, num_components=10, grid_size=[H, W])

    coords = _grid(H, W).reshape(-1, 2).clone()  # (N, 2), all on-grid
    # Knock a few points off the grid by a non-integer index offset.
    coords[::7] += 0.013
    coords = coords.clamp(-1, 1)

    for p, c in zip(plain(coords), cached(coords)):
        assert torch.allclose(p, c, atol=1e-6)


def test_cache_anisotropic_grid_and_components():
    """Per-axis grid sizes and per-axis component counts stay consistent."""
    plain = CosineBasis(2, num_components=[6, 9])
    cached = CosineBasis(2, num_components=[6, 9], grid_size=[10, 14])

    coords = _grid(10, 14)
    out_plain = plain(coords)
    out_cached = cached(coords)
    assert [t.shape[-1] for t in out_cached] == [6, 9]
    for p, c in zip(out_plain, out_cached):
        assert torch.equal(p, c)


@pytest.mark.parametrize("basis", ["cosine", "sinc", "tent", "triangle", "hat", "lanczos"])
def test_grid_and_flat_coordinates_agree(basis):
    """Grid-shaped coords give the same result as the equivalent flat list.

    Regression test: the row-sparse local bases (tent/lanczos, ``sparse=True``
    by default) collapse the leading coordinate axes, so ``ConditionalFUTON``
    must flatten its queries and restore the shape. Without that, grid-shaped
    coordinates raised a broadcast error against the conditioner's
    ``(B, H, W, F)`` output -- or, when a spatial axis was 1, silently produced
    a wrong shape. CINDER only passes flat coordinates while subsampling during
    training, so this surfaced at the first validation pass.
    """
    torch.manual_seed(0)
    B, E, H, W = 3, 8, 6, 5
    model = ConditionalFUTON(
        in_features=2,
        out_features=4,
        cond_dim=E,
        basis=(basis, {"num_components": 8}),
        combiner=("cp", {"rank": 9}),
    ).eval()

    z = torch.randn(B, E, 4, 7)
    coords = _grid(H, W)
    with torch.no_grad():
        on_grid = model(coords, z)
        flat = model(coords.reshape(-1, 2), z)

    assert on_grid.shape == (B, H, W, 4)
    assert flat.shape == (B, H * W, 4)
    assert torch.allclose(on_grid.reshape(B, H * W, 4), flat, atol=1e-6)

    # A degenerate single-column grid must keep its shape (it used to broadcast).
    with torch.no_grad():
        assert model(_grid(H, 1), z).shape == (B, H, 1, 4)


def test_full_model_caching_is_exact():
    """End-to-end: enabling the cache does not change ConditionalFUTON output."""
    torch.manual_seed(0)
    H = W = 16
    kw = dict(in_features=2, out_features=3, cond_dim=8)
    plain = ConditionalFUTON(basis=("cosine", {"num_components": 24}), **kw)
    cached = ConditionalFUTON(
        basis=("cosine", {"num_components": 24, "grid_size": [H, W]}), **kw
    )
    cached.load_state_dict(plain.state_dict())  # share weights

    coords = _grid(H, W)
    z = torch.randn(4, 8, 5, 6)  # conditioning feature map
    with torch.no_grad():
        assert torch.allclose(plain(coords, z), cached(coords, z), atol=1e-6)
