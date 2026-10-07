"""Unit tests for the modulators.

Focused coverage for the modulator protocol -- every modulator is built from
``(inr, in_features, out_features, cond_shape)``, reads ``(coords, conds)`` and
predicts ``(B, *, out_features)`` -- and for its implementations: the
:class:`FUTONGate`, which samples the conditioning maps at the query coordinates
so that it varies in space; the :class:`WeightDisplacement` and the per-image INR
it yields; and :class:`ListModulators`, which nests them. Also covers the sampler,
fusion rules and conditioners they are assembled from.
"""

import weakref
from functools import partial
from math import prod

import pytest
import torch
from torch import nn
from torch.func import functional_call

from cinder.models.inrs import (
    MLP,
    SIREN,
    CosineBasis,
    CPCombiner,
    FUTONEncoding,
    PositionalEncoding,
)
from cinder.models.modulators import (
    FUSIONS,
    MODULATORS,
    AttentionFusion,
    BaseModulator,
    Camera,
    FeatureConcat,
    FUTONGate,
    GridSampler,
    LinearWeightConditioner,
    ListModulators,
    MLPWeightConditioner,
    SumFusion,
    WeightConditioner,
    WeightDisplacement,
)

# A small encoding, so the modulator tests stay light. Rank 9 is the width the
# gate works in and the width the INR is built to read.
STAGES = {"basis": ("cosine", {"num_components": 8}), "combiner": ("cp", {"rank": 9})}
SMALL_INR = ("mlp", {"hidden_layers": 1})
SMALL_SIREN = ("siren", {"hidden_features": 16, "hidden_layers": 1})
COND_SHAPE = ((5, 3, 3), (7, 6, 5))


def _grid(h, w):
    ys = torch.linspace(-1, 1, h)
    xs = torch.linspace(-1, 1, w)
    return torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)  # (h, w, 2)


def _run(model, coords, conds):
    """Predict at ``(*, D)`` coordinates shared by the batch, as CINDER does."""
    return model(coords.unsqueeze(0), conds)


def _inr(model):
    """The INR at the bottom of a list of modulators."""
    return model.modulators[0].inr


def _input_modulator(inr=SMALL_INR, cond_shape=COND_SHAPE, **gate):
    """Input-only, with the small encoding; ``gate`` overrides the gate's knobs."""
    return ListModulators(inr, 2, 3, cond_shape, ("futon", {**STAGES, **gate}))


def _weight_modulator(inr=SMALL_SIREN, cond_shape=COND_SHAPE, **displacement):
    """Weight-only; ``displacement`` are the modulator's knobs."""
    return ListModulators(inr, 2, 3, cond_shape, ("displacement", displacement))


def _hybrid(inr=SMALL_SIREN, cond_shape=COND_SHAPE, **displacement):
    """A displacement of one INR, wrapped by a gate."""
    modulators = [("displacement", displacement), ("futon", STAGES)]
    return ListModulators(inr, 2, 3, cond_shape, modulators)


def _bare_gate(cond_shape=COND_SHAPE, **gate):
    """A gate around the identity, so its predictions are the gated features."""
    return FUTONGate(nn.Identity(), 2, 9, cond_shape, **{**STAGES, **gate})


def _gate_features(gate, coords, conds):
    """norm(encode(coords) * sum of the sampled factors), rebuilt from the parts."""
    factors = [
        gate.sampler(coords, projection(z.movedim(1, -1)).movedim(-1, 1))
        for z, projection in zip(conds, gate.projections)
    ]
    return gate.norm(gate.encoding(coords) * sum(factors))


class Affine(BaseModulator):
    """A map-independent input modulator: an affine warp of the coordinates."""

    def __init__(self, inr, in_features, out_features, cond_shape, width=2):
        super().__init__(inr, in_features, out_features, cond_shape)
        self.linear = nn.Linear(in_features, width)
        self.inr = self.build_inr(width)

    def forward(self, coords, conds):
        conds = self.check_inputs(coords, conds)
        return self.inr(self.linear(coords), conds)


# ---------------------------------------------------------------------------
# GridSampler
# ---------------------------------------------------------------------------


def test_grid_sampler_reads_at_the_right_coordinates():
    """coords[..., 0] indexes rows and coords[..., 1] columns, exactly on-grid."""
    # Channel 0 is a pure row ramp, channel 1 a pure column ramp, so a correctly
    # sampled point reproduces its own (row, col) coordinate.
    z = torch.zeros(1, 2, 3, 5)
    z[0, 0] = torch.linspace(-1, 1, 3)[:, None]
    z[0, 1] = torch.linspace(-1, 1, 5)[None, :]
    probe = torch.tensor([[-1.0, 1.0], [1.0, -1.0], [0.0, 0.0], [0.5, -0.25]])

    sampled = GridSampler()(probe, z)[0]  # (4, 2)
    assert torch.allclose(sampled, probe, atol=1e-6)

    # At the map's own grid nodes the interpolation is exact (align_corners).
    nodes = _grid(3, 5)
    assert torch.allclose(GridSampler()(nodes, z), z.movedim(1, -1), atol=1e-6)


def test_grid_sampler_honours_its_interpolation_knobs():
    """`mode` and `padding_mode` reach grid_sample rather than being fixed."""
    peak = torch.zeros(1, 1, 3, 3)
    peak[0, 0, 1, 1] = 1.0
    between = torch.tensor([[0.25, 0.0]])  # a quarter step past the centre row

    assert torch.allclose(GridSampler()(between, peak), torch.full((1, 1, 1), 0.75))
    nearest = GridSampler(mode="nearest")(between, peak)
    assert torch.allclose(nearest, torch.ones(1, 1, 1))  # snaps to the peak

    # Outside the domain `border` clamps to the edge, `zeros` falls away.
    far, ones = torch.tensor([[3.0, 3.0]]), torch.ones(1, 1, 2, 2)
    assert torch.allclose(GridSampler()(far, ones), torch.ones(1, 1, 1))
    zeros = GridSampler(padding_mode="zeros")(far, ones)
    assert torch.allclose(zeros, torch.zeros(1, 1, 1))


def test_grid_sampler_rejects_a_map_of_the_wrong_rank():
    with pytest.raises(ValueError, match="spatial dimensions"):
        GridSampler()(_grid(3, 4), torch.zeros(1, 2, 5))  # 1 spatial axis, needs 2


# ---------------------------------------------------------------------------
# FUTONGate: the gate
# ---------------------------------------------------------------------------


def _gating(cond_shape, width, **kwargs):
    """A modulated INR whose gate is ``width`` wide, for exercising the gate."""
    gate = {
        "basis": ("cosine", {"num_components": 4}),
        "combiner": ("cp", {"rank": width}),
        **kwargs,
    }
    return ListModulators("mlp", 2, 1, cond_shape, ("futon", gate))


def _factors(model, coords, maps):
    """The per-map factors the gate builds, rebuilt from the model's own parts."""
    gate = model.modulators[0]
    maps = [maps] if isinstance(maps, torch.Tensor) else list(maps)
    return [
        gate.sampler(coords, projection(z.movedim(1, -1)).movedim(-1, 1))
        for z, projection in zip(maps, gate.projections)
    ]


def _decode(model, features, coord_shape):
    """What `forward` does after the gate: norm, INR, restore the query shape."""
    out = _inr(model)(model.modulators[0].norm(features))
    return out.reshape(out.shape[0], *coord_shape, model.out_features)


def test_gate_multiplies_the_encoding_by_the_sampled_condition():
    """forward == decode(encode(coords) * gate), and the shape contract holds."""
    torch.manual_seed(0)
    B, H, W, F, E = 3, 4, 5, 7, 6
    model = _gating(((E, 3, 8),), F).eval()
    coords, flat = _grid(H, W), _grid(H, W).reshape(-1, 2)
    z = torch.randn(B, E, 3, 8)  # conditioning feature map

    with torch.no_grad():
        out = _run(model, coords, z)
        gated = model.modulators[0].encoding(flat) * _factors(model, flat, z)[0]
        expected = _decode(model, gated, (H, W))
    assert out.shape == (B, H, W, 1)
    assert torch.allclose(out, expected, atol=1e-5)

    # Flat coordinates keep the same contract: (N, 2) -> (B, N, D).
    assert _run(model, flat, z).shape == (B, H * W, 1)


def test_gate_projects_over_channels():
    """A spatially constant map gates by that vector through the projection.

    Computed from the projection's weight rather than by re-running the
    model's own projection step, so this pins the channel-wise contract
    independently of how `forward` moves its axes.
    """
    torch.manual_seed(0)
    E, F, H, W = 4, 3, 2, 5
    model = _gating(((E, 3, 6),), F).eval()
    gate = model.modulators[0]
    channels = torch.randn(E)
    z = channels.reshape(1, E, 1, 1).expand(1, E, 3, 6)  # same vector everywhere

    with torch.no_grad():
        factor = gate.projections[0](channels)  # (F,), the factor at every query
        gated = (gate.encoding(_grid(H, W).reshape(-1, 2)) * factor).unsqueeze(0)
        expected = _decode(model, gated, (H, W))
        assert torch.allclose(_run(model, _grid(H, W), z), expected, atol=1e-5)


def test_gate_has_no_bias_by_default():
    """z = 0 -> projection 0 -> the gate closes, the same way everywhere."""
    torch.manual_seed(0)
    model = _gating(((8, 2, 2),), 4).eval()
    assert all(p.bias is None for p in model.modulators[0].projections)

    with torch.no_grad():
        out = _run(model, _grid(3, 4), torch.zeros(5, 8, 2, 2))
    # The gated features are zero at every coordinate, so the INR sees the same
    # input everywhere and every query gets the same answer.
    assert torch.allclose(out, out[:, :1, :1].expand_as(out), atol=1e-6)


def test_gate_matches_cp_factor_semantics():
    """With a CP combiner, conditioning is one more (spatial) CP factor."""
    torch.manual_seed(0)
    B, R, E, H, W = 2, 5, 8, 3, 3
    model = _gating(((E, 4, 4),), R).eval()
    coords, flat = _grid(H, W), _grid(H, W).reshape(-1, 2)
    z = torch.randn(B, E, 4, 4)

    with torch.no_grad():
        # Manual CP: Hadamard of the per-axis rank-R projections and the
        # condition factor, the latter read off the map at each coordinate.
        encoding = model.modulators[0].encoding
        axes = [
            lin(f) for f, lin in zip(encoding.basis(flat), encoding.combiner.linears)
        ]
        gated = axes[0] * axes[1] * _factors(model, flat, z)[0]
        expected = _decode(model, gated, (H, W))
        assert torch.allclose(_run(model, coords, z), expected, atol=1e-5)


def test_gate_projection_commutes_with_sampling():
    """Projecting the grid then sampling equals sampling then projecting.

    Interpolation is linear, so the two orders agree; `forward` projects first
    (cheaper -- see `FUTONGate._fuse_maps`), and this pins the equivalence.
    """
    torch.manual_seed(0)
    model = _gating(((6, 3, 4),), 4)
    gate = model.modulators[0]
    coords, z = _grid(7, 5), torch.randn(2, 6, 3, 4)

    project_first = _factors(model, coords, z)[0]
    sample_first = gate.projections[0](gate.sampler(coords, z))
    assert torch.allclose(project_first, sample_first, atol=1e-5)


def test_gate_multiplies_the_per_map_factors():
    """Every map contributes one factor, read on its own grid.

    Under ``product`` the maps meet multiplicatively -- each is one more CP
    factor over the same coordinates -- and none of them is resampled to a
    common resolution first.
    """
    torch.manual_seed(0)
    B, F, E, H, W = 2, 5, (4, 6), 6, 7
    model = _gating(((4, 3, 4), (6, 5, 5)), F, fusion="product").eval()
    assert model.cond_shape == ((4, 3, 4), (6, 5, 5))
    assert [p.in_features for p in model.modulators[0].projections] == list(E)

    coords, flat = _grid(H, W), _grid(H, W).reshape(-1, 2)
    zs = [torch.randn(B, 4, 3, 4), torch.randn(B, 6, 5, 5)]  # different grids

    with torch.no_grad():
        out = _run(model, coords, zs)
        features = model.modulators[0].encoding(flat)
        a, b = _factors(model, flat, zs)
        assert torch.allclose(out, _decode(model, features * a * b, (H, W)), atol=1e-5)
        # A product, not a sum: the two differ, so this pins which one is built.
        summed = _decode(model, features * (a + b), (H, W))
    assert not torch.allclose(out, summed, atol=1e-4)

    # Equal full map shapes let us swap values and pin the projection pairing.
    same = _gating(((4, 3, 4), (4, 3, 4)), F, fusion="product").eval()
    a, b = torch.randn(B, 4, 3, 4), torch.randn(B, 4, 3, 4)
    with torch.no_grad():
        assert not torch.allclose(
            _run(same, coords, [a, b]), _run(same, coords, [b, a]), atol=1e-4
        )


def test_gate_single_map_is_unchanged_by_the_sequence_path():
    """A bare tensor and a one-element sequence keep the same single-map behaviour."""
    torch.manual_seed(0)
    model = _gating(((6, 3, 4),), 4).eval()
    coords, z = _grid(7, 5), torch.randn(2, 6, 3, 4)

    with torch.no_grad():
        assert torch.equal(_run(model, coords, z), _run(model, coords, [z]))


@pytest.mark.parametrize("fusion", FUSIONS)
def test_gate_fusion_rules(fusion):
    """Each rule combines the per-map factors exactly as it advertises."""
    torch.manual_seed(0)
    B, F, H, W = 2, 5, 6, 7
    model = _gating(((4, 3, 4), (6, 5, 5)), F, fusion=fusion).eval()
    rule = model.modulators[0].fusion
    with torch.no_grad():  # both mixers start uniform; give them something to say
        for name in ("mix_logits", "scores"):
            if hasattr(rule, name):
                getattr(rule, name).normal_()
    coords, flat = _grid(H, W), _grid(H, W).reshape(-1, 2)
    zs = [torch.randn(B, 4, 3, 4), torch.randn(B, 6, 5, 5)]  # different grids

    with torch.no_grad():
        out = _run(model, coords, zs)
        features = model.modulators[0].encoding(flat)
        a, b = _factors(model, flat, zs)

        if fusion == "product":
            gate = a * b
        elif fusion == "residual":
            gate = (1 + a) * (1 + b)
        elif fusion == "sum":
            gate = a + b
        elif fusion == "convex":
            w = rule.mix_logits.softmax(0)
            assert torch.allclose(w.sum(), torch.ones(())) and (w > 0).all()
            gate = w[0] * a + w[1] * b
        else:  # attention: weights vary with the coordinate
            scores = rule.scores
            w = torch.stack([a @ scores[0], b @ scores[1]], dim=-1).softmax(-1)
            assert torch.allclose(w.sum(-1), torch.ones(1)) and w.shape == (B, H * W, 2)
            assert w[..., 0].std() > 1e-3  # genuinely varying, not a uniform mix
            gate = w[..., :1] * a + w[..., 1:] * b

        assert torch.allclose(out, _decode(model, features * gate, (H, W)), atol=1e-5)

        # Whatever the rule, one map is passed through untouched by it -- except
        # `residual`, whose whole point is the +1.
        single = _gating(((4, 3, 4),), F, fusion=fusion).eval()
        factor = _factors(single, flat, zs[0])[0]
        one = (1 + factor) if fusion == "residual" else factor
        plain = single.modulators[0].encoding(flat) * one
        assert torch.allclose(
            _run(single, coords, zs[0]), _decode(single, plain, (H, W)), atol=1e-5
        )


def test_gate_rejects_an_unknown_fusion():
    with pytest.raises(KeyError, match="unknown key"):
        _gating(((3, 2, 2), (4, 3, 3)), 2, fusion="mean")

    # A rule is a spec like any other: a key, a pair, or a ready-made module.
    ready = _gating(((4, 3, 4),), 2, fusion=SumFusion(1, 2)).modulators[0].fusion
    assert isinstance(ready, SumFusion)
    with pytest.raises(ValueError, match="fusion.num_maps must be 2"):
        _gating(((3, 2, 2), (4, 3, 3)), 2, fusion=SumFusion(1, 2))


def test_gate_bias_is_per_projection():
    """Each map's projection carries its own bias; the gate sees their product."""
    torch.manual_seed(0)
    H, W = 2, 3
    model = _gating(((3, 2, 2), (4, 3, 3)), 2, bias=True, fusion="product").eval()
    gate = model.modulators[0]
    with torch.no_grad():
        for projection, value in zip(gate.projections, (1.5, 2.5)):
            projection.weight.zero_()
            projection.bias.fill_(value)

    zs = [torch.zeros(1, 3, 2, 2), torch.zeros(1, 4, 3, 3)]
    with torch.no_grad():  # zero weights -> the biases alone gate, everywhere
        gated = (gate.encoding(_grid(H, W).reshape(-1, 2)) * (1.5 * 2.5)).unsqueeze(0)
        expected = _decode(model, gated, (H, W))
        assert torch.allclose(_run(model, _grid(H, W), zs), expected, atol=1e-5)


def test_gate_rejects_mismatched_maps():
    model = _gating(((3, 2, 2), (4, 3, 3)), 2)
    coords = _grid(2, 3)

    with pytest.raises(ValueError, match="conditioning map"):  # too few maps
        _run(model, coords, [torch.zeros(1, 3, 2, 2)])
    with pytest.raises(ValueError, match="shape"):  # right count, wrong channels
        _run(model, coords, [torch.zeros(1, 3, 2, 2), torch.zeros(1, 5, 3, 3)])


def test_gate_stands_alone():
    """Around the identity, the gate predicts norm(encode(x) * (a + b))."""
    torch.manual_seed(0)
    gate = _bare_gate()
    flat = _grid(6, 5).reshape(-1, 2)
    zs = [torch.randn(4, 5, 3, 3), torch.randn(4, 7, 6, 5)]

    out = _run(gate, flat, zs)
    assert out.shape == (4, 30, 9) and gate.encoding.out_features == 9
    assert torch.allclose(out, _gate_features(gate, flat, zs), atol=1e-6)


@pytest.mark.parametrize("basis", ["cosine", "lanczos"])
def test_standalone_gate_preserves_query_shape_and_accepts_one_map(basis):
    gate = _bare_gate(((5, 4, 5),), basis=(basis, {"num_components": 8}))
    coords, feature_map = _grid(3, 4), torch.randn(2, 5, 4, 5)
    expected = _run(gate, coords.reshape(-1, 2), [feature_map])
    actual = _run(gate, coords, feature_map)
    torch.testing.assert_close(actual, expected.reshape(2, 3, 4, 9))


def test_sum_gate_releases_sampled_factors_before_normalization():
    """Summed factors must not occupy memory during the larger LayerNorm step."""
    gate = _bare_gate(((3, 4, 4), (4, 3, 3), (5, 2, 2), (6, 1, 1)), fusion="sum")
    maps = [
        torch.randn(2, channels, size, size, requires_grad=True)
        for channels, size in ((3, 4), (4, 3), (5, 2), (6, 1))
    ]
    references = []
    live_at_normalization = []

    def observe_sample(module, inputs, output):
        references.append(weakref.ref(output))

    def observe_normalization(module, inputs):
        live_at_normalization.append(sum(ref() is not None for ref in references))

    gate.sampler.register_forward_hook(observe_sample)
    gate.norm.register_forward_pre_hook(observe_normalization)
    coords = _grid(5, 6).reshape(-1, 2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = _run(gate, coords, maps)
    assert out.requires_grad
    assert len(references) == len(maps)
    assert live_at_normalization == [0]
    out.float().square().mean().backward()
    assert all(m.grad is not None and torch.isfinite(m.grad).all() for m in maps)


def test_sum_gate_matches_direct_composition_outputs_and_gradients():
    """Releasing intermediates preserves the gate and its training gradients."""
    torch.manual_seed(0)
    gate = _bare_gate(
        ((3, 4, 4), (4, 3, 3), (5, 2, 2)),
        basis=("lanczos", {"num_components": 8}),
        fusion="sum",
    )
    coords = _grid(5, 6).reshape(-1, 2)
    maps = [
        torch.randn(2, channels, size, size, requires_grad=True)
        for channels, size in ((3, 4), (4, 3), (5, 2))
    ]
    actual = _run(gate, coords, maps)
    expected = _gate_features(gate, coords, maps)
    torch.testing.assert_close(actual, expected)

    probe = torch.randn_like(actual)
    inputs = (*maps, *gate.parameters())
    actual_gradients = torch.autograd.grad((actual * probe).sum(), inputs)
    expected_gradients = torch.autograd.grad((expected * probe).sum(), inputs)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("cond_shape", [(5, 6), (5, 2, 3, 4), ((5, 3, 3), (7, 4))])
@pytest.mark.parametrize(
    "build", [_bare_gate, _input_modulator, _hybrid], ids=["gate", "input", "hybrid"]
)
def test_gate_needs_one_grid_axis_per_coordinate(build, cond_shape):
    """The gate samples maps at the coordinates; the weight path alone does not."""
    with pytest.raises(ValueError, match="cond_shape grids need 2 axes"):
        build(cond_shape=cond_shape)

    model = _weight_modulator(cond_shape=cond_shape)
    conds = [torch.randn(2, *shape) for shape in model.cond_shape]
    assert _run(model, _grid(2, 3), conds).shape == (2, 2, 3, 3)


# ---------------------------------------------------------------------------
# ListModulators: input modulation
# ---------------------------------------------------------------------------


def test_input_modulation_forwards_the_fusion_knob():
    model = _input_modulator(fusion="attention")
    assert isinstance(model.modulators[0].fusion, AttentionFusion)
    zs = [torch.randn(4, 5, 3, 3), torch.randn(4, 7, 6, 5)]
    assert _run(model, _grid(6, 5), zs).shape == (4, 6, 5, 3)


def test_input_modulation_accepts_several_maps():
    """End-to-end through the modulators: condition shapes and maps as sequences."""
    torch.manual_seed(0)
    model = _input_modulator().eval()
    assert model.cond_shape == COND_SHAPE

    zs = [torch.randn(4, 5, 3, 3), torch.randn(4, 7, 6, 5)]
    with torch.no_grad():
        assert _run(model, _grid(6, 5), zs).shape == (4, 6, 5, 3)
        assert _run(model, _grid(6, 5).reshape(-1, 2), zs).shape == (4, 30, 3)


@pytest.mark.parametrize(
    "inr", ["mlp", "siren", "finer", "gauss", "wire", "rff", "futon"]
)
def test_input_modulation_wraps_any_inr(inr):
    """The INR is called directly, so every one of them can be modulated."""
    torch.manual_seed(0)
    params = (
        {**STAGES, "decoder": "linear"} if inr == "futon" else {"hidden_features": 16}
    )
    model = _input_modulator(inr=(inr, params), cond_shape=((5, 3, 3),))

    out = _run(model, _grid(4, 5), torch.randn(2, 5, 3, 3))
    assert out.shape == (2, 4, 5, 3) and torch.isfinite(out).all()

    out.sum().backward()  # gradients reach the encoding, the gate and the INR
    gate = model.modulators[0]
    for part in (gate.encoding, gate.projections, _inr(model)):
        grads = [p.grad for p in part.parameters() if p.requires_grad]
        assert grads and all(g is not None for g in grads)


def test_input_modulation_builds_its_encoding_and_sizes_the_rest_to_it():
    """basis -> combiner set the width; the gate, the norm and the INR follow."""
    kw = dict(in_features=2, out_features=3, cond_shape=((5, 3, 3),))
    slot = dict(modulators=("futon", STAGES))

    model = ListModulators(inr=SMALL_INR, **kw, **slot)
    gate = model.modulators[0]
    assert isinstance(gate, FUTONGate)
    assert isinstance(gate.encoding, FUTONEncoding)
    assert isinstance(gate.encoding.basis, CosineBasis)
    assert gate.encoding.basis.num_components == [8, 8]
    assert isinstance(gate.encoding.combiner, CPCombiner)
    assert gate.encoding.out_features == 9
    assert isinstance(gate.norm, nn.LayerNorm)
    assert tuple(gate.norm.normalized_shape) == (9,)
    assert isinstance(gate.inr, MLP)
    assert (gate.inr.in_features, gate.inr.out_features) == (9, 3)

    # A bare registry key, a factory, and a ready-made module all resolve.
    assert isinstance(_inr(ListModulators(inr="mlp", **kw, **slot)), MLP)

    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return MLP(**kwargs)

    ListModulators(inr=factory, **kw, **slot)
    assert seen == {"in_features": 9, "out_features": 3}

    ready = MLP(9, 3)
    assert _inr(ListModulators(inr=ready, **kw, **slot)) is ready

    # The injected sizes win: a partial's own value is silently replaced, and
    # a (key, params) pair carrying one is a duplicate keyword.
    fixed = ListModulators(inr=partial(MLP, in_features=4), **kw, **slot)
    assert _inr(fixed).in_features == 9
    with pytest.raises(TypeError):
        ListModulators(inr=("mlp", {"in_features": 4}), **kw, **slot)

    # A ready-made INR must already read the width the encoding produces.
    with pytest.raises(ValueError, match="inr.in_features must be 9"):
        ListModulators(inr=MLP(4, 3), **kw, **slot)
    with pytest.raises(ValueError, match="inr.out_features must be 3"):
        ListModulators(inr=MLP(9, 7), **kw, **slot)


def test_input_modulation_composes_the_stages_in_order():
    """out = inr(norm(encode(x) * gate))."""
    torch.manual_seed(0)
    model = _input_modulator().eval()
    gate = model.modulators[0]
    coords = _grid(6, 5)
    zs = [torch.randn(4, 5, 3, 3), torch.randn(4, 7, 6, 5)]

    with torch.no_grad():
        out = _run(model, coords, zs)
        flat = coords.reshape(-1, 2)
        a, b = _factors(model, flat, zs)
        conditioned = gate.encoding(flat) * (a + b)  # the shipped rule is `sum`
        expected = gate.inr(gate.norm(conditioned))
        unnormed = gate.inr(conditioned)
    assert out.shape == (4, 6, 5, 3)
    assert torch.allclose(out, expected.reshape(4, 6, 5, 3), atol=1e-6)
    # The norm is not skipped.
    assert not torch.allclose(out, unnormed.reshape(4, 6, 5, 3), atol=1e-4)


def test_input_modulation_keeps_the_inr_output_activation():
    """The INR is called as it is, so its output activation still applies."""
    torch.manual_seed(0)
    plain = _input_modulator(cond_shape=((5, 3, 3),)).eval()
    squashed_inr = ("mlp", {**SMALL_INR[1], "output_activation": nn.Sigmoid()})
    squashed = _input_modulator(inr=squashed_inr, cond_shape=((5, 3, 3),)).eval()
    squashed.load_state_dict(plain.state_dict())  # share weights

    coords, z = _grid(4, 3), torch.randn(2, 5, 3, 3)
    with torch.no_grad():
        expected = _run(plain, coords, z).sigmoid()
        assert torch.allclose(_run(squashed, coords, z), expected, atol=1e-6)


@pytest.mark.parametrize("basis", ["cosine", "sinc", "lanczos"])
def test_input_modulation_grid_and_flat_coordinates_agree(basis):
    """Grid-shaped coords give the same result as the equivalent flat list.

    Regression test: the row-sparse local basis (lanczos, ``sparse=True`` by
    default) collapses the leading coordinate axes, which once broke grid-shaped
    coordinates against the gate's ``(B, H, W, F)`` output -- or, when a spatial
    axis was 1, silently produced a wrong shape.
    """
    torch.manual_seed(0)
    B, E, H, W = 3, 8, 6, 5
    stages = {
        "basis": (basis, {"num_components": 8}),
        "combiner": ("cp", {"rank": 9}),
        "decoder": ("mlp", {"hidden_layers": 1}),
    }
    model = ListModulators(("futon", stages), 2, 4, ((E, 4, 7),), "futon").eval()

    z = torch.randn(B, E, 4, 7)
    coords = _grid(H, W)
    with torch.no_grad():
        on_grid = _run(model, coords, z)
        flat = _run(model, coords.reshape(-1, 2), z)

    assert on_grid.shape == (B, H, W, 4)
    assert flat.shape == (B, H * W, 4)
    assert torch.allclose(on_grid.reshape(B, H * W, 4), flat, atol=1e-6)

    # A degenerate single-column grid must keep its shape (it used to broadcast).
    with torch.no_grad():
        assert _run(model, _grid(H, 1), z).shape == (B, H, 1, 4)


def test_input_modulation_caching_is_exact():
    """Enabling the basis cache does not change the modulated output."""
    torch.manual_seed(0)
    H = W = 16
    combiner = ("cp", {"rank": 9})
    plain_basis = ("cosine", {"num_components": 24})
    cached_basis = ("cosine", {"num_components": 24, "grid_size": [H, W]})
    plain = _input_modulator(
        cond_shape=((8, 5, 6),), basis=plain_basis, combiner=combiner
    )
    cached = _input_modulator(
        cond_shape=((8, 5, 6),), basis=cached_basis, combiner=combiner
    )
    cached.load_state_dict(plain.state_dict())  # share weights

    coords = _grid(H, W)
    z = torch.randn(4, 8, 5, 6)  # conditioning feature map
    with torch.no_grad():
        assert torch.allclose(
            _run(plain, coords, z), _run(cached, coords, z), atol=1e-6
        )


# ---------------------------------------------------------------------------
# WeightConditioner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "conditioner_cls", [LinearWeightConditioner, MLPWeightConditioner]
)
def test_weight_conditioner_reads_positions_then_channels(conditioner_cls):
    conditioner = conditioner_cls(in_shape=(16, 6), out_shape=(8, 9))
    condition = torch.randn(3, 16, 6)
    assert conditioner(condition).shape == (3, 8, 9)

    # Matrix orientation is explicit; raw spatial maps belong to WeightDisplacement.
    for invalid in (condition.mT, torch.randn(3, 6, 4, 4)):
        with pytest.raises(ValueError, match="condition"):
            conditioner(invalid)


@pytest.mark.parametrize(
    "conditioner_cls", [LinearWeightConditioner, MLPWeightConditioner]
)
def test_weight_conditioner_returns_a_scaled_mixture_of_two_readings(conditioner_cls):
    """The conditioner returns strength * ((1 - mix) * direct + mix * crossed)."""
    torch.manual_seed(0)
    conditioner = conditioner_cls(in_shape=(6, 4), out_shape=(8, 9))
    with torch.no_grad():  # move off the initialization, which is symmetric
        for p in conditioner.parameters():
            p.normal_()
    condition = torch.randn(3, 6, 4)

    out = conditioner(condition)
    direct = conditioner.positions_to_rows(
        conditioner.channels_to_cols(condition).mT
    ).mT
    crossed = conditioner.channels_to_rows(
        conditioner.positions_to_cols(condition.mT).mT
    ).mT
    mix = torch.sigmoid(conditioner.mix_logit)
    expected = conditioner.strength * ((1 - mix) * direct + mix * crossed)

    assert out.shape == (3, 8, 9)
    assert torch.allclose(out, expected, atol=1e-6)
    # The two branches read the condition's axes the opposite way round, so
    # neither is the other's copy.
    assert not torch.allclose(direct, crossed, atol=1e-4)


@pytest.mark.parametrize("positions", [1, 4])
def test_weight_conditioner_displacement_rank_follows_the_positions(positions):
    """Each branch contracts one axis, so the grid is what the rank is spent on."""
    torch.manual_seed(0)
    rows, cols = 32, 40
    conditioner = LinearWeightConditioner(
        in_shape=(positions, 64), out_shape=(rows, cols)
    )
    with torch.no_grad():
        for p in conditioner.parameters():
            p.normal_(0, 0.3)
        delta = conditioner(torch.randn(3, positions, 64))

    ranks = [int(torch.linalg.matrix_rank(d)) for d in delta]
    assert all(r < min(rows, cols) for r in ranks)  # low-rank, not full
    # Two separable branches, plus their biases.
    assert all(r <= 2 * positions + 3 for r in ranks)


# ---------------------------------------------------------------------------
# WeightDisplacement
# ---------------------------------------------------------------------------


def _displaced(displacement, conds, image):
    """The wrapped model's parameters for one image, displaced by hand."""
    condition = displacement.flatten_maps(conds)
    return {
        name: displacement.inr.get_parameter(name) + conditioner(condition)[image]
        for name, conditioner in zip(
            displacement.weight_names, displacement.conditioners
        )
    }


def test_weight_displacement_gives_each_image_its_own_inr():
    """The INR runs once per image, with that image's displaced weights."""
    torch.manual_seed(0)
    model = _weight_modulator().eval()
    (displacement,) = model.modulators
    coords = _grid(6, 5)
    flat = coords.reshape(-1, 2)
    zs = [torch.randn(4, 5, 3, 3), torch.randn(4, 7, 6, 5)]

    with torch.no_grad():
        out = _run(model, coords, zs)
        flat_out = _run(model, flat, zs)
    assert out.shape == (4, 6, 5, 3)
    assert flat_out.shape == (4, 30, 3)
    assert torch.allclose(out.reshape(4, 30, 3), flat_out, atol=1e-6)
    # Different images, different functions.
    assert not torch.allclose(out[0], out[1], atol=1e-4)

    # And each is exactly the INR run with that image's displaced weights.
    assert displacement.weight_names == ("layers.0.linear.weight", "layers.1.weight")
    with torch.no_grad():
        one = functional_call(displacement.inr, _displaced(displacement, zs, 2), coords)
    assert torch.allclose(out[2], one, atol=1e-6)


def test_weight_displacement_stands_alone():
    """It wraps a ready-made INR, and every query of an image shares its weights."""
    torch.manual_seed(0)
    inr = SIREN(2, 3, hidden_features=16, hidden_layers=1)
    displacement = WeightDisplacement(inr, 2, 3, COND_SHAPE)
    assert displacement.inr is inr and displacement.in_shape == (30, 7)
    assert [c.out_shape for c in displacement.conditioners] == [(16, 2), (3, 16)]

    zs = [torch.randn(4, 5, 3, 3), torch.randn(4, 7, 6, 5)]
    flat = _grid(6, 5).reshape(-1, 2)
    out = _run(displacement, flat, zs)
    assert out.shape == (4, 30, 3)
    # A subset of the queries gets the same predictions: the weights depend on
    # the maps only.
    subset = torch.tensor([0, 7, 29])
    torch.testing.assert_close(_run(displacement, flat[subset], zs), out[:, subset])

    # Coordinates may also differ per image, (B, *, D).
    per_image = torch.rand(4, 5, 2) * 2 - 1
    out = displacement(per_image, zs)
    for b in range(4):
        one = displacement(per_image[b : b + 1], [z[b : b + 1] for z in zs])
        torch.testing.assert_close(out[b : b + 1], one)


def test_weight_displacement_leaves_the_inr_alone_at_zero_strength():
    """With the displacement switched off, it is the unmodulated INR."""
    torch.manual_seed(0)
    model = _weight_modulator().eval()
    inr = _inr(model)
    with torch.no_grad():
        for conditioner in model.modulators[-1].conditioners:
            conditioner.strength.zero_()

    coords = _grid(4, 5)
    zs = [torch.randn(3, 5, 3, 3), torch.randn(3, 7, 6, 5)]
    with torch.no_grad():
        out, base = _run(model, coords, zs), inr(coords)
    assert torch.allclose(out, base.expand_as(out), atol=1e-6)  # same for every image

    # A changed base matrix must take effect without rebuilding the displacement.
    with torch.no_grad():
        inr.layers[0].linear.weight.mul_(0.5)
        changed, base = _run(model, coords, zs), inr(coords)
    assert not torch.allclose(changed, out)
    torch.testing.assert_close(changed, base.expand_as(changed))


@pytest.mark.parametrize("cond_size", [(2, 3), (2, 3, 4)])
def test_weight_displacement_preserves_every_final_map_position(cond_size):
    spatial_dims = len(cond_size)
    cond_shape = ((5, *([3] * spatial_dims)), (7, *cond_size))
    displacement = WeightDisplacement(
        nn.Linear(spatial_dims, 3), spatial_dims, 3, cond_shape
    )
    positions = prod(cond_size)
    assert displacement.cond_shape == cond_shape
    assert displacement.in_shape == (positions, 7)
    assert all(t.in_shape == displacement.in_shape for t in displacement.conditioners)

    first = torch.full((2, 5, *([3] * spatial_dims)), -100.0)
    last = torch.arange(2 * 7 * positions, dtype=torch.float32).reshape(
        2, 7, *cond_size
    )
    condition = displacement.flatten_maps([first, last])
    expected = torch.arange(2 * 7, dtype=last.dtype).reshape(
        2, 1, 7
    ) * positions + torch.arange(positions, dtype=last.dtype).reshape(1, positions, 1)
    torch.testing.assert_close(condition, expected)
    assert condition.untyped_storage().data_ptr() == last.untyped_storage().data_ptr()

    # Equal position counts cannot hide a changed spatial arrangement.
    coords = torch.zeros(1, 4, spatial_dims)
    wrong_grid = last.reshape(2, 7, *reversed(cond_size))
    with pytest.raises(ValueError, match="shape"):
        displacement(coords, [first, wrong_grid])


@pytest.mark.parametrize("hybrid", [False, True], ids=["weight", "hybrid"])
def test_weight_displacement_uses_final_map_and_hybrid_keeps_multiscale_input(hybrid):
    torch.manual_seed(0)
    build = _hybrid if hybrid else _weight_modulator
    model = build(cond_shape=((5, 3, 3), (7, 4, 4))).eval()
    coords = _grid(4, 5)
    maps = [
        torch.randn(2, 5, 3, 3, requires_grad=True),
        torch.randn(2, 7, 4, 4, requires_grad=True),
    ]
    changed_early = [torch.randn_like(maps[0]), maps[1]]
    out = _run(model, coords, maps)
    changed = _run(model, coords, changed_early)
    if hybrid:
        assert not torch.allclose(out, changed)
    else:
        torch.testing.assert_close(out, changed, rtol=0, atol=0)
    changed_final = [maps[0], torch.randn_like(maps[1])]
    assert not torch.allclose(out, _run(model, coords, changed_final))

    out.square().mean().backward()
    assert maps[1].grad is not None and torch.isfinite(maps[1].grad).all()
    assert torch.count_nonzero(maps[1].grad) > 0
    if hybrid:
        assert maps[0].grad is not None and torch.isfinite(maps[0].grad).all()
        assert torch.count_nonzero(maps[0].grad) > 0
    else:
        assert maps[0].grad is None


def test_weight_displacement_selects_the_matrix_parameters():
    """Weights are modulated; biases and other vectors stay shared."""
    displacement = _weight_modulator().modulators[-1]
    assert displacement.weight_names == ("layers.0.linear.weight", "layers.1.weight")
    assert len(displacement.conditioners) == 2
    # The default conditioner is the MLP one.
    assert all(isinstance(t, WeightConditioner) for t in displacement.conditioners)
    assert all(isinstance(t, MLPWeightConditioner) for t in displacement.conditioners)
    linear = _weight_modulator(conditioner="linear").modulators[-1].conditioners[0]
    assert isinstance(linear, WeightConditioner)
    assert isinstance(linear, LinearWeightConditioner)

    # An explicit selection is honoured, and checked.
    one = _weight_modulator(weight_names=["layers.1.weight"]).modulators[-1]
    assert one.weight_names == ("layers.1.weight",)
    assert one.conditioners[0].out_shape == (3, 16)

    with pytest.raises(KeyError, match="no parameter"):
        _weight_modulator(weight_names=["layers.9.weight"])
    with pytest.raises(ValueError, match="only matrix parameters"):
        _weight_modulator(weight_names=["layers.1.bias"])
    with pytest.raises(ValueError, match="duplicates"):
        _weight_modulator(weight_names=["layers.1.weight", "layers.1.weight"])


@pytest.mark.parametrize("inr", ["mlp", "siren", "finer", "gauss", "wire", "rff"])
def test_weight_displacement_wraps_any_inr(inr):
    """Any INR can be modulated: it is called through `functional_call`."""
    torch.manual_seed(0)
    model = _weight_modulator(
        inr=(inr, {"hidden_features": 16, "hidden_layers": 1}),
        cond_shape=((5, 3, 3),),
    )
    # The INR reads the coordinates themselves, unlike behind a gate.
    assert model.modulators[-1].in_features == 2

    out = _run(model, _grid(4, 5), torch.randn(2, 5, 3, 3))
    assert out.shape == (2, 4, 5, 3) and torch.isfinite(out).all()

    out.sum().backward()  # gradients reach the conditioners and the INR alike
    for part in (model.modulators[-1].conditioners, _inr(model)):
        grads = [p.grad for p in part.parameters() if p.requires_grad]
        assert grads and all(g is not None for g in grads)


@pytest.mark.parametrize("init_scale", [0.0, 0.5])
def test_displacement_scales_the_selected_matrices_at_init(init_scale):
    """Only the displaced matrices are rescaled, and nothing else is drawn."""
    torch.manual_seed(0)
    reference = WeightDisplacement(SMALL_SIREN, 2, 3, COND_SHAPE)
    torch.manual_seed(0)
    scaled = WeightDisplacement(SMALL_SIREN, 2, 3, COND_SHAPE, init_scale=init_scale)
    for name, parameter in reference.state_dict().items():
        expected = parameter
        if name.removeprefix("inr.") in scaled.weight_names:
            expected = init_scale * parameter
        torch.testing.assert_close(scaled.state_dict()[name], expected, rtol=0, atol=0)

    coords = _grid(3, 4)
    zs = [torch.randn(2, 5, 3, 3), torch.randn(2, 7, 6, 5)]
    assert torch.isfinite(_run(scaled, coords, zs)).all()


def test_displacement_checks_ready_made_conditioner_shapes():
    # A (1, 2) update would broadcast over this (3, 2) weight, concealing an
    # incorrectly wired ready-made conditioner.
    conditioner = LinearWeightConditioner((4, 5), (1, 2))
    with pytest.raises(ValueError, match="conditioner.out_shape"):
        WeightDisplacement(nn.Linear(2, 3), 2, 3, ((5, 2, 2),), conditioner=conditioner)


@pytest.mark.parametrize(
    "cond_shape",
    [
        ((5, 2, 0),),
        ((5, 2, 1.5),),
        ((),),
        (5, 2, 0),
        (5, 2, 1.5),
    ],
)
def test_displacement_requires_positive_integer_condition_shapes(cond_shape):
    with pytest.raises(ValueError, match="cond_shape"):
        WeightDisplacement(nn.Linear(2, 3), 2, 3, cond_shape)


@pytest.mark.parametrize("grid", [(), (6,), (2, 3), (1, 2, 3)])
def test_displacement_reads_positions_whatever_the_grid_rank(grid):
    """Only the number of positions matters, not how the grid lays them out."""
    modules = []
    for shape in ((5, *grid), (5, prod(grid))):
        torch.manual_seed(0)
        modules.append(
            WeightDisplacement(nn.Linear(2, 3), 2, 3, shape, conditioner="linear")
        )
    displacement, flat = modules
    assert displacement.in_shape == (prod(grid), 5)

    coords = _grid(2, 3).reshape(-1, 2)
    feature_map = torch.randn(2, 5, *grid)
    torch.testing.assert_close(
        _run(displacement, coords, feature_map),
        _run(flat, coords, feature_map.reshape(2, 5, prod(grid))),
        rtol=0,
        atol=0,
    )


# ---------------------------------------------------------------------------
# ListModulators: composing modulators
# ---------------------------------------------------------------------------


def test_hybrid_nests_the_gate_the_displacement_and_the_inr():
    """The INR reads the gate's width; the displacement is shaped to the INR."""
    model = _hybrid()
    displacement, gate = model.modulators
    assert model.inr is gate and gate.inr is displacement
    assert isinstance(gate, FUTONGate) and isinstance(displacement, WeightDisplacement)
    assert _inr(model).in_features == gate.encoding.out_features == 9
    assert displacement.in_features == 9  # the displacement follows the gate
    assert displacement.in_shape == (30, 7)
    assert displacement.weight_names == ("layers.0.linear.weight", "layers.1.weight")
    assert displacement.conditioners[0].out_shape == (16, 9)  # (hidden, F)


def test_hybrid_gates_the_input_and_displaces_the_weights():
    """out[b] = INR_theta(z)[b](phi(x, z)[b]), image by image."""
    torch.manual_seed(0)
    model = _hybrid().eval()
    displacement, gate = model.modulators
    coords = _grid(6, 5)
    flat = coords.reshape(-1, 2)
    zs = [torch.randn(4, 5, 3, 3), torch.randn(4, 7, 6, 5)]

    with torch.no_grad():
        out = _run(model, coords, zs)
        phi = _gate_features(gate, flat, zs)  # (4, 30, 9)
        for b in range(4):
            weights = _displaced(displacement, zs, b)
            one = functional_call(displacement.inr, weights, (phi[b],))  # (30, 3)
            assert torch.allclose(out[b].reshape(-1, 3), one, atol=1e-6)
    assert out.shape == (4, 6, 5, 3)
    assert not torch.allclose(out[0], out[1], atol=1e-4)


def test_hybrid_reduces_to_input_only_at_zero_strength():
    """Switch the displacement off and the hybrid is the gated INR."""
    torch.manual_seed(0)
    hybrid = _hybrid().eval()
    displacement, gate = hybrid.modulators
    with torch.no_grad():
        for conditioner in displacement.conditioners:
            conditioner.strength.zero_()
    plain = _input_modulator(inr=SMALL_SIREN).eval()
    for part in ("encoding", "projections", "norm"):
        getattr(plain.modulators[0], part).load_state_dict(
            getattr(gate, part).state_dict()
        )
    _inr(plain).load_state_dict(_inr(hybrid).state_dict())

    coords = _grid(4, 5)
    zs = [torch.randn(3, 5, 3, 3), torch.randn(3, 7, 6, 5)]
    with torch.no_grad():
        assert torch.allclose(
            _run(hybrid, coords, zs), _run(plain, coords, zs), atol=1e-6
        )


def test_modulators_chain():
    """Each modulator reads what the one wrapping it passes on as coordinates."""
    torch.manual_seed(0)
    model = ListModulators(
        SMALL_INR, 2, 3, COND_SHAPE, ["displacement", ("futon", STAGES), Affine]
    ).eval()
    displacement, gate, affine = model.modulators
    assert [m.in_features for m in model.modulators] == [9, 2, 2]
    assert _inr(model).in_features == 9

    coords, flat = _grid(4, 5), _grid(4, 5).reshape(-1, 2)
    zs = [torch.randn(2, 5, 3, 3), torch.randn(2, 7, 6, 5)]
    with torch.no_grad():
        out = _run(model, coords, zs)
        # The gate encodes the warped coordinates and samples the maps there.
        phi = _gate_features(gate, affine.linear(flat), zs)  # (2, 20, 9)
        for b in range(2):
            weights = _displaced(displacement, zs, b)
            one = functional_call(displacement.inr, weights, (phi[b],))
            torch.testing.assert_close(out[b].reshape(-1, 3), one)


def test_a_gate_cannot_follow_a_gate():
    """A gate samples the maps at what it reads, which must be coordinates."""
    with pytest.raises(ValueError, match="cond_shape grids need 9 axes"):
        ListModulators(
            SMALL_INR, 2, 3, COND_SHAPE, [("futon", STAGES), ("futon", STAGES)]
        )


def test_weight_modulators_displace_everything_they_wrap():
    """An outer displacement also displaces the inner modulator's matrices.

    Restricted to the INR's matrices, the two displacements add up.
    """
    torch.manual_seed(0)
    model = ListModulators(
        SMALL_INR,
        2,
        3,
        COND_SHAPE,
        [
            ("displacement", {"conditioner": "linear"}),
            ("displacement", {"conditioner": "linear"}),
        ],
    )
    inner, outer = model.modulators
    assert outer.weight_names == (
        "inr.layers.0.weight",
        "inr.layers.1.weight",
        *(
            f"conditioners.{i}.{name}.weight"
            for i in range(2)
            for name in (
                "positions_to_rows",
                "channels_to_cols",
                "channels_to_rows",
                "positions_to_cols",
            )
        ),
    )

    inr_only = ["inr.layers.0.weight", "inr.layers.1.weight"]
    model = ListModulators(
        SMALL_INR,
        2,
        3,
        COND_SHAPE,
        ["displacement", ("displacement", {"weight_names": inr_only})],
    ).eval()
    inner, outer = model.modulators
    coords = _grid(4, 5).reshape(-1, 2)
    zs = [torch.randn(2, 5, 3, 3), torch.randn(2, 7, 6, 5)]
    with torch.no_grad():
        out = _run(model, coords, zs)
        outer_deltas = [c(outer.flatten_maps(zs)) for c in outer.conditioners]
        inner_deltas = [c(inner.flatten_maps(zs)) for c in inner.conditioners]
        for b in range(2):
            weights = {
                name: inner.inr.get_parameter(name) + d_outer[b] + d_inner[b]
                for name, d_outer, d_inner in zip(
                    inner.weight_names, outer_deltas, inner_deltas
                )
            }
            one = functional_call(inner.inr, weights, (coords,))
            torch.testing.assert_close(out[b], one)


def test_a_displacement_of_the_inr_commutes_with_a_gate():
    """Displacing only the INR, its place around the gate does not matter."""
    torch.manual_seed(0)
    gate_outside = _hybrid().eval()
    displacement_outside = ListModulators(
        SMALL_SIREN,
        2,
        3,
        COND_SHAPE,
        [
            ("futon", STAGES),
            (
                "displacement",
                {"weight_names": ["inr.layers.0.linear.weight", "inr.layers.1.weight"]},
            ),
        ],
    ).eval()
    displacement_a, gate_a = gate_outside.modulators
    gate_b, displacement_b = displacement_outside.modulators
    for part in ("encoding", "projections", "norm"):
        getattr(gate_b, part).load_state_dict(getattr(gate_a, part).state_dict())
    displacement_b.conditioners.load_state_dict(
        displacement_a.conditioners.state_dict()
    )
    _inr(displacement_outside).load_state_dict(_inr(gate_outside).state_dict())

    coords = _grid(4, 5)
    zs = [torch.randn(2, 5, 3, 3), torch.randn(2, 7, 6, 5)]
    with torch.no_grad():
        torch.testing.assert_close(
            _run(displacement_outside, coords, zs), _run(gate_outside, coords, zs)
        )


def test_list_modulators_accepts_one_spec_but_no_module():
    """A key or a (key, params) pair is a list of one modulator."""
    torch.manual_seed(0)
    for spec in ("displacement", ("futon", STAGES), [("futon", STAGES)]):
        model = ListModulators(SMALL_INR, 2, 3, COND_SHAPE, spec)
        assert len(model.modulators) == 1
    # A built modulator already wraps its INR, so it cannot wrap the next one.
    with pytest.raises(TypeError, match="not built modules"):
        ListModulators(SMALL_INR, 2, 3, COND_SHAPE, _bare_gate())


@pytest.mark.parametrize("basis", ["cosine", "lanczos"])
def test_hybrid_grid_and_flat_coordinates_agree(basis):
    """The gate runs outside the displacement's vmap, so a sparse basis works."""
    torch.manual_seed(0)
    B, E, H, W = 3, 8, 6, 5
    gate = {"basis": (basis, {"num_components": 8}), "combiner": ("cp", {"rank": 9})}
    model = ListModulators(
        SMALL_SIREN, 2, 4, ((E, 4, 7),), ["displacement", ("futon", gate)]
    ).eval()

    z = torch.randn(B, E, 4, 7)
    coords = _grid(H, W)
    with torch.no_grad():
        on_grid = _run(model, coords, z)
        flat = _run(model, coords.reshape(-1, 2), z)
        assert on_grid.shape == (B, H, W, 4)
        assert torch.allclose(on_grid.reshape(B, H * W, 4), flat, atol=1e-6)
        assert _run(model, _grid(H, 1), z).shape == (B, H, 1, 4)


def test_hybrid_gradients_reach_every_part():
    torch.manual_seed(0)
    model = _hybrid()
    zs = [torch.randn(2, 5, 3, 3), torch.randn(2, 7, 6, 5)]
    _run(model, _grid(4, 5), zs).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_list_modulators_registers_the_inr_once():
    """Each part belongs to exactly one modulator; the INR to the innermost."""
    model = _hybrid()
    assert sum(isinstance(module, SIREN) for module in model.modules()) == 1
    names = [name for name, _ in model.named_parameters()]
    assert len(names) == len(set(names)) == len(list(model.parameters()))
    assert {name.split(".")[0] for name in names} == {"inr"}


def test_list_modulators_needs_a_modulator():
    with pytest.raises(ValueError, match="at least one"):
        ListModulators(SMALL_INR, 2, 3, ((5, 3, 3),), [])


@pytest.mark.parametrize("build", [_input_modulator, _weight_modulator, _hybrid])
def test_list_modulators_checks_maps_for_every_mode(build):
    """The same guard whatever the modulators; a bare tensor is one map."""
    torch.manual_seed(0)
    model = build(cond_shape=((5, 2, 2),)).eval()
    z = torch.randn(1, 5, 2, 2)
    with torch.no_grad():
        assert torch.equal(_run(model, _grid(2, 3), z), _run(model, _grid(2, 3), [z]))

    model = build()
    with pytest.raises(ValueError, match="conditioning map"):  # too few maps
        _run(model, _grid(2, 3), [torch.zeros(1, 5, 3, 3)])
    with pytest.raises(ValueError, match="shape"):  # right count, wrong channels
        _run(model, _grid(2, 3), [torch.zeros(1, 5, 3, 3), torch.zeros(1, 9, 6, 5)])


@pytest.mark.parametrize("build", [_input_modulator, _weight_modulator, _hybrid])
def test_list_modulators_reject_broadcast_batches_and_wrong_coordinate_width(build):
    model = build()
    maps = [torch.randn(2, 5, 3, 3), torch.randn(1, 7, 6, 5)]
    with pytest.raises(ValueError, match="same batch size"):
        _run(model, _grid(2, 3), maps)
    with pytest.raises(ValueError, match="coords must have shape"):
        _run(model, torch.zeros(2, 4), [maps[0], maps[1].expand(2, -1, -1, -1)])


@pytest.mark.parametrize("kind", ["gate", "displacement"])
def test_modulators_check_their_maps_and_inputs(kind):
    shapes = ((5, 3, 3), (7, 4, 4))
    if kind == "gate":
        module = _bare_gate(shapes)
    else:
        module = WeightDisplacement(nn.Linear(2, 3), 2, 3, shapes, conditioner="linear")

    coords = _grid(2, 3).reshape(-1, 2)
    a, b = torch.randn(2, 5, 3, 3), torch.randn(2, 7, 4, 4)
    _run(module, coords, [a, b])
    for maps in ([a], [a, b, b]):
        with pytest.raises(ValueError, match="conditioning map"):
            _run(module, coords, maps)
    with pytest.raises(ValueError, match="same batch size"):
        _run(module, coords, [a, b[:1]])
    with pytest.raises(ValueError, match="shape"):
        _run(module, coords, [a, b.unsqueeze(-1)])

    # Coordinates are shared, (1, *, D), or per image, (B, *, D).
    for bad in (coords, coords.expand(3, -1, -1), torch.zeros(1, 6, 3)):
        with pytest.raises(ValueError, match="coords must have shape"):
            module(bad, [a, b])
    per_image = coords.expand(2, -1, -1)
    if kind == "gate":
        with pytest.raises(ValueError, match="shared by the batch"):
            module(per_image, [a, b])
    else:
        assert module(per_image, [a, b]).shape == (2, 6, 3)


@pytest.mark.parametrize(
    "modulators", ["displacement", ["displacement", ("futon", STAGES)]]
)
def test_list_modulators_require_condition_shapes(modulators):
    with pytest.raises(TypeError, match="cond_shape"):
        ListModulators(
            inr=SMALL_INR, in_features=2, out_features=3, modulators=modulators
        )


@pytest.mark.parametrize("cond_shape", [(), ((0, 3, 3),), ((3, 3, 3), (0, 2, 2))])
def test_modulators_require_nonempty_positive_condition_shapes(cond_shape):
    with pytest.raises(ValueError, match="cond_shape"):
        _input_modulator(cond_shape=cond_shape)


@pytest.mark.parametrize("kind", ["gate", "displacement", "list"])
@pytest.mark.parametrize("shape", [(5, 3, 3), [5, 3, 3], torch.Size([5, 3, 3])])
def test_a_lone_condition_shape_is_one_map(kind, shape):
    """``(C, H, W)`` builds the same module as ``((C, H, W),)``."""

    def build(cond_shape):
        torch.manual_seed(0)
        if kind == "gate":
            return _bare_gate(cond_shape)
        if kind == "displacement":
            return WeightDisplacement(
                nn.Linear(2, 3), 2, 3, cond_shape, conditioner="linear"
            )
        return ListModulators(
            SMALL_INR, 2, 3, cond_shape, ["displacement", ("futon", STAGES)]
        )

    lone, listed = build(shape), build(((5, 3, 3),))
    assert lone.cond_shape == listed.cond_shape == ((5, 3, 3),)
    if kind != "displacement":  # the displacement wraps one shared nn.Linear
        torch.testing.assert_close(
            lone.state_dict(), listed.state_dict(), rtol=0, atol=0
        )

    coords, feature_map = _grid(2, 3).reshape(-1, 2), torch.randn(2, 5, 3, 3)
    torch.testing.assert_close(
        _run(lone, coords, feature_map), _run(listed, coords, [feature_map])
    )


@pytest.mark.parametrize("cond_shape", [((5, 3, 3), 7), (5, (3, 3), 3)])
def test_condition_shapes_do_not_mix_sizes_and_shapes(cond_shape):
    with pytest.raises(ValueError, match="cond_shape"):
        _input_modulator(cond_shape=cond_shape)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grid_sampler_low_precision_preserves_subpixel_queries(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # Close queries near the border round to the same bfloat16 coordinate if
    # the grid is cast to the feature-map dtype before interpolation.
    values = torch.arange(256, device=device, dtype=dtype).reshape(1, 1, 1, 256)
    values.requires_grad_()
    coords = torch.tensor([[0.0, 0.990], [0.0, 0.994]], device=device)
    sampler = GridSampler()
    actual = sampler(coords, values)
    expected = sampler(coords, values.float()).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


@pytest.mark.parametrize("mode", ["input", "weight", "hybrid"])
def test_modulators_support_volume_conditions(mode):
    modulators = {
        "input": [("futon", STAGES)],
        "weight": ["displacement"],
        "hybrid": ["displacement", ("futon", STAGES)],
    }[mode]
    model = ListModulators(
        ("mlp", {"hidden_features": 8, "hidden_layers": 1}),
        3,
        2,
        ((4, 2, 3, 4), (5, 3, 2, 2)),
        modulators,
    )
    coords = torch.rand(2, 3, 4, 3) * 2 - 1
    maps = [torch.randn(2, 4, 2, 3, 4), torch.randn(2, 5, 3, 2, 2)]
    out = _run(model, coords, maps)
    assert out.shape == (2, 2, 3, 4, 2)
    out.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hybrid_sparse_gate_cuda_autocast_gradients(dtype):
    gate = ("futon", {**STAGES, "basis": ("lanczos", {"num_components": 8})})
    model = ListModulators(SMALL_INR, 2, 3, ((5, 3, 4),), ["displacement", gate]).cuda()
    maps = torch.randn(2, 5, 3, 4, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=dtype):
        out = _run(model, _grid(4, 5).cuda(), maps)
        loss = out.float().square().mean()
    loss.backward()
    assert out.shape == (2, 4, 5, 3) and torch.isfinite(out).all()
    assert maps.grad is not None and torch.isfinite(maps.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


@pytest.mark.parametrize(
    "modulator",
    [
        partial(FUTONGate, **STAGES),
        partial(WeightDisplacement, conditioner="linear"),
        partial(ListModulators, modulators=["displacement", ("futon", STAGES)]),
    ],
    ids=["gate", "displacement", "list"],
)
def test_modulators_share_one_constructor_and_forward(modulator):
    """Every modulator is built from (inr, sizes) and predicts from (coords, conds)."""
    module = modulator(
        inr=SMALL_INR, in_features=2, out_features=3, cond_shape=[[5, 3, 3], [7, 6, 5]]
    )
    assert isinstance(module, BaseModulator)
    assert module.cond_shape == COND_SHAPE
    assert (module.in_features, module.out_features) == (2, 3)

    coords = _grid(2, 3)
    conds = [torch.randn(2, *shape) for shape in COND_SHAPE]
    out = module(coords=coords.unsqueeze(0), conds=conds)
    assert out.shape == (2, 2, 3, 3)


@pytest.mark.parametrize("build", [_input_modulator, _weight_modulator, _hybrid])
@pytest.mark.parametrize("changed_map", [0, 1])
def test_full_condition_shapes_reject_wrong_spatial_arrangements(build, changed_map):
    cond_shape = ((5, 2, 3), (7, 3, 4))
    model = build(cond_shape=cond_shape)
    conds = [torch.randn(2, *shape) for shape in cond_shape]
    channels, height, width = cond_shape[changed_map]
    conds[changed_map] = conds[changed_map].reshape(2, channels, width, height)

    # Matching channel counts and pixel counts are insufficient, even for the
    # earlier map which weight-only modulation does not read.
    with pytest.raises(ValueError, match="shape"):
        _run(model, _grid(2, 3), conds)


@pytest.mark.parametrize("kind", ["gate", "displacement", "list"])
def test_condition_guard_rejects_mixed_devices_without_using_a_gpu(kind):
    conds = [torch.empty(2, 5, 3, 3), torch.empty(2, 7, 6, 5, device="meta")]
    if kind == "gate":
        module = _bare_gate()
    elif kind == "displacement":
        module = WeightDisplacement(nn.Linear(2, 3), 2, 3, COND_SHAPE)
    else:
        module = _weight_modulator()
    with pytest.raises(ValueError, match="same device"):
        _run(module, _grid(2, 3), conds)


class _SelectAxes(nn.Module):
    """Parallel projection onto the given coordinate axes."""

    def __init__(self, axes):
        super().__init__()
        self.axes = list(axes)

    def forward(self, coords):
        return coords[..., self.axes]


def test_gate_samples_each_map_where_sample_at_maps_the_coordinates():
    """3D queries, two 2D maps: each view is sampled at its own projection."""
    torch.manual_seed(0)
    cond_shape = ((5, 6, 7), (5, 6, 7))
    views = [_SelectAxes((0, 2)), partial(_SelectAxes, (1, 2))]  # module or factory
    gate = FUTONGate(nn.Identity(), 3, 9, cond_shape, **STAGES, sample_at=views)
    coords = torch.rand(1, 11, 3) * 2 - 1
    conds = [torch.randn(2, 5, 6, 7), torch.randn(2, 5, 6, 7)]
    out = gate(coords, conds)
    assert out.shape == (2, 11, 9)

    # Sampling the first view at (x, z) and the second at (y, z) by hand.
    factors = [
        gate.sampler(coords[0][:, axes], projection(z.movedim(1, -1)).movedim(-1, 1))
        for z, projection, axes in zip(conds, gate.projections, ([0, 2], [1, 2]))
    ]
    expected = gate.norm(gate.encoding(coords) * sum(factors))
    torch.testing.assert_close(out, expected)

    with pytest.raises(ValueError, match="one entry per map"):
        FUTONGate(nn.Identity(), 3, 9, cond_shape, **STAGES, sample_at=views[:1])
    # A map sampled at the coordinates themselves needs one grid axis per axis.
    with pytest.raises(ValueError, match="grids need 3 axes"):
        FUTONGate(nn.Identity(), 3, 9, cond_shape, **STAGES, sample_at=[None, views[1]])


def test_displacement_can_read_several_maps():
    """cond_maps concatenates the chosen maps along positions."""
    cond_shape = ((5, 3, 3), (7, 6, 5), (5, 2, 2))
    both = WeightDisplacement(
        nn.Linear(2, 3), 2, 3, cond_shape, conditioner="linear", cond_maps=(0, 2)
    )
    assert both.in_shape == (9 + 4, 5)
    conds = [torch.randn(2, *shape) for shape in cond_shape]
    condition = both.flatten_maps(conds)
    assert condition.shape == (2, 13, 5)
    torch.testing.assert_close(condition[:, :9], conds[0].flatten(2).mT)
    torch.testing.assert_close(condition[:, 9:], conds[2].flatten(2).mT)
    assert both(torch.zeros(1, 4, 2), conds).shape == (2, 4, 3)

    last = WeightDisplacement(nn.Linear(2, 3), 2, 3, cond_shape, conditioner="linear")
    assert last.cond_maps == (2,) and last.in_shape == (4, 5)
    for bad in ((0, 1), (0, 0)):
        with pytest.raises(ValueError, match="cond_maps"):
            WeightDisplacement(nn.Linear(2, 3), 2, 3, cond_shape, cond_maps=bad)


def test_concat_feeds_the_sampled_maps_and_the_code_to_the_inr():
    torch.manual_seed(0)
    cond_shape = ((5, 6, 7), (3, 4, 4))
    code = partial(PositionalEncoding, num_frequencies=2)
    concat = FeatureConcat(nn.Identity(), 2, 9, cond_shape, encoding=code)
    coords = torch.rand(1, 11, 2) * 2 - 1
    conds = [torch.randn(2, *shape) for shape in cond_shape]
    out = concat(coords, conds)

    sampled = [concat.sampler(coords[0], z) for z in conds]
    expected = torch.cat([*sampled, concat.encoding(coords).expand(2, -1, -1)], -1)
    torch.testing.assert_close(out, expected)
    assert out.shape == (2, 11, 5 + 3 + concat.encoding.out_features)
    assert MODULATORS["concat"] is FeatureConcat


def test_concat_without_a_code_reads_the_coordinates():
    inr = ("mlp", {"hidden_features": 8, "hidden_layers": 1})
    concat = FeatureConcat(inr, 2, 3, (5, 6, 7))
    assert concat.inr.in_features == 5 + 2
    out = concat(torch.rand(1, 4, 2) * 2 - 1, torch.randn(2, 5, 6, 7))
    assert out.shape == (2, 4, 3)


def test_concat_samples_each_map_where_sample_at_maps_the_coordinates():
    cond_shape = ((5, 6, 7), (5, 6, 7))
    views = [_SelectAxes((0, 2)), partial(_SelectAxes, (1, 2))]
    concat = FeatureConcat(nn.Identity(), 3, 9, cond_shape, sample_at=views)
    coords = torch.rand(1, 11, 3) * 2 - 1
    conds = [torch.randn(2, 5, 6, 7), torch.randn(2, 5, 6, 7)]
    expected = torch.cat(
        [
            concat.sampler(coords[0][:, [0, 2]], conds[0]),
            concat.sampler(coords[0][:, [1, 2]], conds[1]),
            coords.expand(2, -1, -1),
        ],
        dim=-1,
    )
    torch.testing.assert_close(concat(coords, conds), expected)

    with pytest.raises(ValueError, match="one entry per map"):
        FeatureConcat(nn.Identity(), 3, 9, cond_shape, sample_at=views[:1])
    with pytest.raises(ValueError, match="grids need 3 axes"):
        FeatureConcat(nn.Identity(), 3, 9, cond_shape, sample_at=[None, views[1]])


def test_concat_needs_coordinates_shared_by_the_batch():
    concat = FeatureConcat(nn.Identity(), 2, 9, (5, 6, 7))
    with pytest.raises(ValueError, match="shared by the batch"):
        concat(torch.rand(2, 4, 2), torch.randn(2, 5, 6, 7))


def test_concat_composes_with_weight_displacement():
    torch.manual_seed(0)
    model = ListModulators(
        ("mlp", {"hidden_features": 8, "hidden_layers": 1}),
        2,
        3,
        (5, 6, 7),
        modulators=[("displacement", {"conditioner": "linear"}), "concat"],
    )
    displacement, concat = model.modulators
    assert isinstance(concat, FeatureConcat)
    assert displacement.inr.in_features == 5 + 2
    out = model(torch.rand(1, 4, 2) * 2 - 1, torch.randn(2, 5, 6, 7))
    assert out.shape == (2, 4, 3)


def test_camera_divides_by_the_projective_coordinate_in_float32():
    matrix = torch.tensor(
        [[2.0, 0.0, 0.0, 0.1], [0.0, 3.0, 0.0, -0.2], [0.0, 0.0, 0.5, 4.0]]
    )
    coords = torch.rand(6, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        projected = Camera(matrix)(coords)
    homogeneous = coords @ matrix[:, :3].T + matrix[:, 3]
    assert projected.dtype == torch.float32
    torch.testing.assert_close(projected, homogeneous[:, :2] / homogeneous[:, 2:])


def test_camera_fit_recovers_the_projection_of_points():
    truth = Camera(
        torch.tensor(
            [[1.2, 0.1, 0.0, 0.05], [0.0, 0.9, 0.2, -0.1], [0.05, 0.0, 0.1, 3.0]]
        )
    )
    points = torch.rand(200, 3, generator=torch.Generator().manual_seed(0)) * 2 - 1
    fitted = Camera.fit(points, truth(points))
    torch.testing.assert_close(fitted(points), truth(points), atol=1e-5, rtol=0)
