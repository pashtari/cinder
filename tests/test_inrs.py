"""Unit tests for the INRs.

The MLP and FUTON sections are neurofield's ``tests/test_mlp.py`` and
``tests/test_futon_bases.py``, with only the imports changed.
"""

import math
from functools import partial

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cinder.models import inrs
from cinder.models.inrs import (
    FUTON,
    INRS,
    CosineBasis,
    CPCombiner,
    HadamardCombiner,
    HashEncoding,
    LanczosBasis,
    PositionalEncoding,
    SincBasis,
    TRCombiner,
    TriangleBasis,
)
from cinder.models.rcs_matrix import RCSMatrix
from cinder.models.utils import build_module

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]


def _grid(h, w):
    ys = torch.linspace(-1, 1, h)
    xs = torch.linspace(-1, 1, w)
    return torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)  # (h, w, 2)


# Registry -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "params"),
    [
        ("linear", {}),
        ("mlp", {"hidden_layers": 1}),
        ("siren", {"hidden_features": 16, "hidden_layers": 1}),
        ("finer", {"hidden_features": 16, "hidden_layers": 1}),
        ("gauss", {"hidden_features": 16, "hidden_layers": 1}),
        ("wire", {"hidden_features": 16, "hidden_layers": 1}),
        ("rff", {"hidden_features": 16, "hidden_layers": 1, "num_frequencies": 8}),
        (
            "futon",
            {
                "basis": ("cosine", {"num_components": 8}),
                "combiner": ("cp", {"rank": 9}),
                "decoder": "linear",
            },
        ),
    ],
)
def test_registered_inrs_build_from_input_and_output_widths(key, params):
    """CINDER builds every INR as ``inr(in_features=..., out_features=...)``."""
    assert set(INRS) == {
        "linear",
        "mlp",
        "siren",
        "finer",
        "gauss",
        "wire",
        "rff",
        "futon",
    }
    torch.manual_seed(0)
    model = build_module((key, params), INRS, in_features=2, out_features=3)
    assert isinstance(model, INRS[key])
    out = model(_grid(4, 5))
    assert out.shape == (4, 5, 3) and not out.is_complex()
    assert torch.isfinite(out).all()


# On-grid caching ------------------------------------------------------------------------


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
    coords[::7] += 0.013  # knock a few points off the grid
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


def test_futon_caching_is_exact():
    """Enabling the basis cache does not change FUTON's output."""
    torch.manual_seed(0)
    H = W = 16
    kw = dict(
        in_features=2, out_features=3, combiner=("cp", {"rank": 9}), decoder="linear"
    )
    plain = FUTON(basis=("cosine", {"num_components": 24}), **kw)
    cached = FUTON(basis=("cosine", {"num_components": 24, "grid_size": [H, W]}), **kw)
    cached.load_state_dict(plain.state_dict())

    with torch.no_grad():
        assert torch.allclose(plain(_grid(H, W)), cached(_grid(H, W)), atol=1e-6)


# MLP -----------------------------------------------------------------------------------


@pytest.mark.parametrize("activation", [None, nn.PReLU(init=0.25)])
def test_hidden_activation_is_shared(activation):
    model = inrs.MLP(1, 1, 1, 2, activation)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.fill_(1)
                layer.bias.zero_()

    output = model(torch.tensor([[-2.0], [2.0]]))
    expected = [[0.0], [2.0]] if activation is None else [[-0.125], [2.0]]
    torch.testing.assert_close(output, torch.tensor(expected))
    if activation is not None:
        assert model.get_submodule("activation") is activation
        assert [
            name
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter is activation.weight
        ] == ["activation.weight"]
        output.sum().backward()
        torch.testing.assert_close(activation.weight.grad, torch.tensor([-1.0]))


@pytest.mark.parametrize("activation", [None, torch.tanh])
def test_omitted_hidden_width_defaults_to_input_width(activation):
    model = inrs.MLP(2, 3, activation=activation)
    reference = inrs.MLP(2, 3, hidden_features=2, activation=activation)
    reference.load_state_dict(model.state_dict())
    x = torch.tensor([[-2.0, 1.0], [0.5, -1.0]])

    assert model.hidden_features == 2
    torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)


def test_hidden_activation_leaves_linear_output_signed():
    model = inrs.MLP(1, 1, 1, 1, activation=torch.relu)
    model.load_state_dict(
        {
            "layers.0.weight": torch.ones(1, 1),
            "layers.0.bias": torch.zeros(1),
            "layers.1.weight": -torch.ones(1, 1),
            "layers.1.bias": torch.zeros(1),
        }
    )

    torch.testing.assert_close(model(torch.tensor([[2.0]])), torch.tensor([[-2.0]]))


def test_hidden_layer_kwargs_do_not_affect_output_layer():
    model = inrs.MLP(2, 1, 3, 2, bias=False)

    assert all(layer.bias is None for layer in model.layers[:-1])
    assert model.layers[-1].bias is not None
    assert model(torch.zeros(4, 2)).shape == (4, 1)


@pytest.mark.parametrize(
    ("activation", "module_activation"),
    [
        (torch.tanh, nn.Tanh()),
        (F.relu, nn.ReLU()),
        (partial(F.leaky_relu, negative_slope=0.2), nn.LeakyReLU(0.2)),
        (lambda x: torch.sigmoid(x), nn.Sigmoid()),
    ],
    ids=["torch", "functional", "partial", "lambda"],
)
def test_callable_hidden_activations_match_modules(activation, module_activation):
    model = inrs.MLP(2, 3, 4, 2, activation=activation).double()
    reference = inrs.MLP(2, 3, 4, 2, activation=module_activation).double()
    assert model.state_dict().keys() == reference.state_dict().keys()
    reference.load_state_dict(model.state_dict())
    x = torch.tensor(
        [[-2.0, 1.0], [0.5, -1.0]], dtype=torch.float64, requires_grad=True
    )
    reference_x = x.detach().clone().requires_grad_()

    output = model(x)
    reference_output = reference(reference_x)
    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
    output.square().sum().backward()
    reference_output.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
    for parameter, reference_parameter in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(
            parameter.grad, reference_parameter.grad, rtol=0, atol=0
        )


@pytest.mark.parametrize("output_activation", [torch.sigmoid, nn.PReLU(init=0.25)])
def test_output_activation_accepts_callable_or_module(output_activation):
    model = inrs.MLP(1, 1, hidden_layers=0, output_activation=output_activation)
    with torch.no_grad():
        model.layers[0].weight.fill_(1)
        model.layers[0].bias.zero_()
    x = torch.tensor([[[-2.0], [2.0]]])

    output = model(x)
    torch.testing.assert_close(output, output_activation(x))
    if isinstance(output_activation, nn.Module):
        assert model.get_submodule("output_activation") is output_activation
        assert "output_activation.weight" in model.state_dict()
        output.sum().backward()
        torch.testing.assert_close(output_activation.weight.grad, torch.tensor([-2.0]))


def test_custom_layers_own_their_activation_and_receive_kwargs():
    model = inrs.MLP(1, 1, 1, 2, layer_class=inrs.SineLayer, omega=2.0)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.fill_(1)
                layer.bias.zero_()
    x = torch.tensor([[[-0.5], [0.5]]])

    torch.testing.assert_close(model(x), torch.sin(2 * torch.sin(2 * x)))


@pytest.mark.parametrize("hidden_features", [None, 4])
def test_zero_hidden_layers_skip_activation_and_preserve_signed_output(hidden_features):
    def unused_activation(x):
        raise AssertionError("A model without hidden layers must not call activation.")

    model = inrs.MLP(
        2, 1, hidden_features, hidden_layers=0, activation=unused_activation
    )
    model.load_state_dict(
        {"layers.0.weight": torch.tensor([[1.0, 2.0]]), "layers.0.bias": torch.zeros(1)}
    )

    torch.testing.assert_close(
        model(torch.tensor([[-1.0, -2.0]])), torch.tensor([[-5.0]])
    )


@pytest.mark.parametrize("activation", [nn.ReLU(), torch.relu])
def test_custom_layers_reject_an_additional_hidden_activation(activation):
    with pytest.raises(ValueError, match="activation"):
        inrs.MLP(2, 1, activation=activation, layer_class=inrs.SineLayer)


@pytest.mark.parametrize(
    ("hidden_layers", "layer_class", "message"),
    [(-1, None, ">= 0"), (-1, inrs.SineLayer, ">= 1"), (0, inrs.SineLayer, ">= 1")],
)
def test_invalid_hidden_layer_counts(hidden_layers, layer_class, message):
    with pytest.raises(ValueError, match=message):
        inrs.MLP(2, 1, hidden_layers=hidden_layers, layer_class=layer_class)


@pytest.mark.parametrize(
    ("layer_class", "hidden_prefix", "output_prefix"),
    [(None, "layers.0", "layers.1"), (inrs.ReLULayer, "layers.0.linear", "layers.1")],
)
def test_plain_and_custom_checkpoint_layouts(layer_class, hidden_prefix, output_prefix):
    checkpoint = {
        f"{hidden_prefix}.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        f"{hidden_prefix}.bias": torch.zeros(3),
        f"{output_prefix}.weight": torch.tensor([[1.0, 2.0, 3.0]]),
        f"{output_prefix}.bias": torch.tensor([0.5]),
    }
    model = inrs.MLP(2, 1, 3, 1, layer_class=layer_class)
    model.load_state_dict(checkpoint)

    x = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
    torch.testing.assert_close(model(x), torch.tensor([[14.5], [2.5]]))


@pytest.mark.parametrize("model_class", [inrs.SIREN, inrs.FINER])
@pytest.mark.parametrize("reset", [False, True])
def test_sine_models_initialize_and_reset_weights_without_changing_biases(
    model_class, reset
):
    model = model_class(2, 1, 8, hidden_layers=2, omega=30.0)
    linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    biases = [layer.bias.detach().clone() for layer in linear_layers]
    if reset:
        with torch.no_grad():
            for layer in linear_layers:
                layer.weight.fill_(99)
        model.reset_parameters()

    for index, (layer, bias) in enumerate(zip(linear_layers, biases)):
        bound = (
            1 / layer.in_features
            if index == 0
            else math.sqrt(6 / layer.in_features) / 30
        )
        assert layer.weight.abs().max().item() <= bound
        torch.testing.assert_close(layer.bias, bias, rtol=0, atol=0)


# FUTON --------------------------------------------------------------------------------


def reference_lanczos(x, num_components, radius):
    """Direct evaluation from the definition, over every center."""
    K, a = num_components, radius
    centers = torch.linspace(-1.0, 1.0, K, dtype=x.dtype, device=x.device)
    w = 2.0 / (K - 1)
    t = (x.unsqueeze(-1) - centers) / w
    kernel = torch.sinc(t) * torch.sinc(t / a)
    return torch.where(t.abs() < a, kernel, torch.zeros_like(kernel))


class TestKernel:
    @pytest.mark.parametrize("radius", [1, 2, 3, 4])
    @pytest.mark.parametrize("K", [16, 33])
    def test_matches_definition(self, radius, K):
        basis = LanczosBasis(1, K, radius=radius, normalize=False, sparse=False)
        x = torch.rand(500) * 2 - 1
        got = basis(x.unsqueeze(-1))[0]
        assert torch.allclose(got, reference_lanczos(x, K, radius), atol=1e-6)

    @pytest.mark.parametrize(
        "basis, taps",
        [
            (TriangleBasis(1, 32), 2),
            (LanczosBasis(1, 32, radius=1), 2),
            (LanczosBasis(1, 32), 4),  # default radius=2 -> Lanczos-2
            (LanczosBasis(1, 32, radius=3), 6),
        ],
    )
    def test_stores_two_taps_per_radius(self, basis, taps):
        """The RCS segment length is 2 * radius."""
        assert basis.radius * 2 == taps
        feats = basis((torch.rand(16) * 2 - 1).unsqueeze(-1))[0]
        assert feats.values.shape[1] == taps

    def test_support_is_l_contiguous(self):
        K, radius = 32, 2
        basis = LanczosBasis(1, K, radius=radius, normalize=False, sparse=False)
        dense = basis((torch.rand(200) * 2 - 1).unsqueeze(-1))[0]
        nz = dense != 0
        assert int(nz.sum(-1).max()) <= 2 * radius
        # every nonzero run is contiguous
        first = torch.argmax(nz.int(), dim=-1)
        last = nz.shape[-1] - 1 - torch.argmax(nz.int().flip(-1), dim=-1)
        assert bool((last - first < 2 * radius).all())

    def test_cardinal_at_grid_points(self):
        """Un-normalized kernels are 1 at their own center, 0 at the others."""
        K = 16
        basis = LanczosBasis(1, K, radius=2, normalize=False, sparse=False)
        dense = basis(torch.linspace(-1, 1, K).unsqueeze(-1))[0]
        assert torch.allclose(dense, torch.eye(K), atol=1e-6)

    def test_approaches_sinc_for_large_radius(self):
        """SincBasis is the radius -> infinity limit."""
        K = 64
        x = (torch.rand(100) * 2 - 1).unsqueeze(-1)
        sinc = SincBasis(1, K, normalize=False)(x)[0]
        errors = [
            (LanczosBasis(1, K, radius=a, normalize=False, sparse=False)(x)[0] - sinc)
            .abs()
            .max()
            .item()
            for a in (2, 8, 24)
        ]
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 0.05

    def test_partition_of_unity_of_interpolant(self):
        """Lanczos reconstruction of a constant signal is near-constant."""
        basis = LanczosBasis(1, 64, radius=3, normalize=False, sparse=False)
        x = (torch.rand(500) * 1.8 - 0.9).unsqueeze(-1)
        assert (basis(x)[0].sum(-1) - 1.0).abs().max() < 0.02


def reference_tent(x, num_components):
    """The tent formula as originally written, in coordinate units."""
    K = num_components
    centers = torch.linspace(-1.0, 1.0, K, dtype=x.dtype, device=x.device)
    width = 2.0 / (K - 1)
    return (1.0 - (x.unsqueeze(-1) - centers).abs() / width).clamp(min=0.0)


class TestTriangle:
    @pytest.mark.parametrize("K", [2, 8, 33, 64, 512])
    def test_matches_original_formula(self, K):
        """Refactoring to grid-index units must not change the values.

        Checked in float64: in float32 both this form and the original sit
        ~1e-5 from the exact value once K is large (cancellation in the
        ``x - mu_c`` subtraction), so a float32 comparison of the two would
        only be measuring rounding noise. See
        :meth:`test_float32_accuracy_matches_original`.
        """
        basis = TriangleBasis(1, K, normalize=False, sparse=False)
        x = torch.rand(500, dtype=torch.float64) * 2 - 1
        got = basis(x.unsqueeze(-1))[0]
        assert torch.allclose(got, reference_tent(x, K), atol=1e-12)

    @pytest.mark.parametrize("K", [64, 512])
    def test_float32_accuracy_matches_original(self, K):
        """In float32 the new form is no less accurate than the original."""
        x = torch.rand(2000) * 2 - 1
        truth = reference_tent(x.double(), K)
        new = TriangleBasis(1, K, normalize=False, sparse=False)(x.unsqueeze(-1))[0]
        old = reference_tent(x, K)
        err_new = (new.double() - truth).abs().max()
        err_old = (old.double() - truth).abs().max()
        assert err_new < 1e-4
        assert err_new < 4 * err_old

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("normalize", [True, False])
    def test_sparse_equals_dense(self, device, normalize):
        K = 64
        x = torch.rand(1000, 2, device=device) * 2 - 1
        dense = TriangleBasis(2, K, normalize=normalize, sparse=False).to(device)(x)
        sparse = TriangleBasis(2, K, normalize=normalize, sparse=True).to(device)(x)
        for d, s in zip(dense, sparse):
            assert isinstance(s, RCSMatrix)
            assert torch.allclose(s.to_dense(), d, atol=1e-6)

    def test_two_taps_only(self):
        feats = TriangleBasis(1, 32, normalize=False)(
            (torch.rand(500) * 2 - 1).unsqueeze(-1)
        )[0]
        assert feats.values.shape[1] == 2
        assert int((feats.to_dense() != 0).sum(-1).max()) <= 2

    def test_partition_of_unity(self):
        basis = TriangleBasis(1, 32, normalize=False, sparse=False)
        x = (torch.rand(500) * 2 - 1).unsqueeze(-1)
        assert torch.allclose(basis(x)[0].sum(-1), torch.ones(500), atol=1e-5)

    def test_cardinal_at_grid_points(self):
        K = 16
        basis = TriangleBasis(1, K, normalize=False, sparse=False)
        got = basis(torch.linspace(-1, 1, K).unsqueeze(-1))[0]
        assert torch.allclose(got, torch.eye(K), atol=1e-6)

    def test_grid_sample_equivalence(self):
        """Un-normalized tents + a linear map == linear interpolation on a grid."""
        K, R = 16, 3
        grid = torch.randn(K, R)
        x = torch.rand(200) * 2 - 1
        feats = TriangleBasis(1, K, normalize=False, sparse=True)(x.unsqueeze(-1))[0]
        pos = (x + 1) / 2 * (K - 1)
        i = pos.floor().clamp(0, K - 2).long()
        frac = (pos - i).unsqueeze(-1)
        expected = grid[i] * (1 - frac) + grid[i + 1] * frac
        assert torch.allclose(feats @ grid, expected, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_futon_equivalence(self, device):
        torch.manual_seed(0)
        x = torch.rand(4, 9, 2, device=device) * 2 - 1
        outs = []
        for sparse in (False, True):
            torch.manual_seed(1)
            model = FUTON(
                2,
                3,
                ("triangle", {"num_components": 32, "sparse": sparse}),
                ("cp", {"rank": 8}),
                "linear",
            ).to(device)
            outs.append(model(x))
        assert outs[0].shape == (4, 9, 3)
        assert torch.allclose(outs[1], outs[0], atol=1e-5)


class TestSparseEqualsDense:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("radius", [1, 2, 4])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_features_match(self, device, radius, normalize):
        K = 48
        x = torch.rand(1000, 2, device=device) * 2 - 1
        common = dict(radius=radius, normalize=normalize)
        dense = LanczosBasis(2, K, sparse=False, **common).to(device)(x)
        sparse = LanczosBasis(2, K, sparse=True, **common).to(device)(x)
        for d, s in zip(dense, sparse):
            assert isinstance(s, RCSMatrix)
            assert s.shape == d.shape
            assert torch.allclose(s.to_dense(), d, atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_boundaries_and_grid_points(self, device):
        """Clamped windows at the domain edges stay exact."""
        K = 32
        x = torch.cat(
            [
                torch.linspace(-1, 1, K, device=device),  # exactly on grid
                torch.tensor([-1.0, 1.0, -0.999, 0.999], device=device),
                torch.rand(200, device=device) * 2 - 1,
            ]
        ).unsqueeze(-1)
        common = dict(radius=3, normalize=False)
        dense = LanczosBasis(1, K, sparse=False, **common).to(device)(x)[0]
        sparse = LanczosBasis(1, K, sparse=True, **common).to(device)(x)[0]
        assert torch.allclose(sparse.to_dense(), dense, atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_per_axis_num_components(self, device):
        x = torch.rand(64, 3, device=device) * 2 - 1
        common = dict(num_components=[16, 24, 32], radius=2)
        dense = LanczosBasis(3, sparse=False, **common).to(device)(x)
        sparse = LanczosBasis(3, sparse=True, **common).to(device)(x)
        for d, s in zip(dense, sparse):
            assert torch.allclose(s.to_dense(), d, atol=1e-6)

    def test_grid_cache_matches(self):
        """Dense mode's on-grid lookup table stays exact."""
        K = 32
        basis = LanczosBasis(1, K, radius=2, sparse=False, grid_size=K)
        plain = LanczosBasis(1, K, radius=2, sparse=False)
        x = torch.cat([torch.linspace(-1, 1, K), torch.rand(50) * 2 - 1]).unsqueeze(-1)
        assert torch.allclose(basis(x)[0], plain(x)[0], atol=1e-6)


class TestCombiners:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("combiner_cls", [CPCombiner, TRCombiner])
    def test_matches_dense_path(self, device, combiner_cls):
        torch.manual_seed(0)
        K, C = 32, 2
        x = torch.rand(128, C, device=device) * 2 - 1
        combiner = combiner_cls([K] * C, rank=8).to(device)
        dense = LanczosBasis(C, K, radius=2, sparse=False).to(device)(x)
        sparse = LanczosBasis(C, K, radius=2, sparse=True).to(device)(x)
        assert torch.allclose(combiner(sparse), combiner(dense), atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_hadamard_densifies(self, device):
        K, C = 32, 2
        x = torch.rand(64, C, device=device) * 2 - 1
        combiner = HadamardCombiner([K] * C)
        dense = LanczosBasis(C, K, radius=2, sparse=False).to(device)(x)
        sparse = LanczosBasis(C, K, radius=2, sparse=True).to(device)(x)
        assert torch.allclose(combiner(sparse), combiner(dense), atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_cp_with_bias(self, device):
        torch.manual_seed(0)
        K, C = 24, 2
        x = torch.rand(64, C, device=device) * 2 - 1
        combiner = CPCombiner([K] * C, rank=6, bias=True).to(device)
        for linear in combiner.linears:
            nn.init.normal_(linear.bias)
        dense = LanczosBasis(C, K, radius=2, sparse=False).to(device)(x)
        sparse = LanczosBasis(C, K, radius=2, sparse=True).to(device)(x)
        assert torch.allclose(combiner(sparse), combiner(dense), atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradients_match(self, device):
        torch.manual_seed(0)
        K, C = 32, 2
        x = torch.rand(128, C, device=device) * 2 - 1
        grads = {}
        for sparse in (False, True):
            torch.manual_seed(1)
            combiner = CPCombiner([K] * C, rank=8).to(device)
            feats = LanczosBasis(C, K, radius=2, sparse=sparse).to(device)(x)
            combiner(feats).square().sum().backward()
            grads[sparse] = [p.grad.clone() for p in combiner.parameters()]
        for g_dense, g_sparse in zip(grads[False], grads[True]):
            assert torch.allclose(g_sparse, g_dense, atol=1e-4, rtol=1e-4)


class TestFuton:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("batch_shape", [(), (7,), (4, 5), (2, 8, 8)])
    def test_shapes_and_equivalence(self, device, batch_shape):
        torch.manual_seed(0)
        K, C = 32, 2
        x = torch.rand(*batch_shape, C, device=device) * 2 - 1
        outs = []
        for sparse in (False, True):
            torch.manual_seed(1)
            model = FUTON(
                C,
                3,
                ("lanczos", {"num_components": K, "radius": 3, "sparse": sparse}),
                ("cp", {"rank": 8}),
                "linear",
            ).to(device)
            outs.append(model(x))
        assert outs[0].shape == (*batch_shape, 3)
        assert torch.allclose(outs[1], outs[0], atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_other_bases_unaffected(self, device):
        """Flattening in FUTON.forward must not change existing bases."""
        torch.manual_seed(0)
        x = torch.rand(3, 6, 2, device=device) * 2 - 1
        for name in ("cosine", "sinc", "triangle"):
            torch.manual_seed(1)
            model = FUTON(
                2, 3, (name, {"num_components": 16}), ("cp", {"rank": 8}), "linear"
            ).to(device)
            out = model(x)
            flat = model(x.reshape(-1, 2))
            assert out.shape == (3, 6, 3)
            assert torch.allclose(out.reshape(-1, 3), flat, atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_training_step(self, device):
        torch.manual_seed(0)
        model = FUTON(
            2,
            1,
            ("lanczos", {"num_components": 64, "radius": 3}),
            ("cp", {"rank": 16}),
            "linear",
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = torch.rand(512, 2, device=device) * 2 - 1
        y = torch.sin(3 * x[:, :1]) * torch.cos(3 * x[:, 1:])
        losses = []
        for _ in range(60):
            opt.zero_grad()
            loss = (model(x) - y).square().mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.6 * losses[0]


@pytest.mark.parametrize("name", ["cosine", "sinc", "triangle", "lanczos"])
class TestNumComponents:
    def test_int_and_per_axis(self, name):
        from cinder.models.inrs import BASES

        def dense(feature):
            return feature.to_dense() if isinstance(feature, RCSMatrix) else feature

        x = torch.rand(20, 3) * 2 - 1
        shared = BASES[name](3, 8, normalize=False)
        per_axis = BASES[name](3, [4, 8, 6], normalize=False)
        assert shared.num_components == [8, 8, 8]
        assert per_axis.num_components == [4, 8, 6]
        features = [dense(f) for f in per_axis(x)]
        assert [f.shape for f in features] == [(20, 4), (20, 8), (20, 6)]
        # Axis 1 has the same count in both bases, so its features must match.
        assert torch.allclose(dense(shared(x)[1]), features[1])

    def test_wrong_length(self, name):
        from cinder.models.inrs import BASES

        with pytest.raises(ValueError):
            BASES[name](3, [8, 8])


class TestValidation:
    @pytest.mark.parametrize("radius", [0, -2])
    def test_radius_must_be_positive(self, radius):
        with pytest.raises(ValueError):
            LanczosBasis(1, 32, radius=radius)

    def test_num_components_at_least_twice_radius(self):
        with pytest.raises(ValueError):
            LanczosBasis(1, 4, radius=3)  # needs K >= 6
        with pytest.raises(ValueError):
            TriangleBasis(1, 1)  # needs K >= 2

    def test_wrong_in_features(self):
        basis = LanczosBasis(3, 16, radius=2)
        with pytest.raises(ValueError):
            basis(torch.rand(8, 2))

    def test_registry(self):
        from cinder.models.inrs import BASES

        assert BASES["lanczos"] is LanczosBasis
        assert BASES["triangle"] is TriangleBasis


@CUDA
class TestSparseIsFaster:
    def test_forward_beats_dense(self):
        """The compact path should win where K >> L (its whole purpose)."""
        torch.manual_seed(0)
        K, C, M = 512, 2, 200_000
        x = torch.rand(M, C, device="cuda") * 2 - 1
        combiner = CPCombiner([K] * C, rank=32).cuda()

        def run(sparse):
            basis = LanczosBasis(C, K, radius=4, sparse=sparse).cuda()
            return lambda: combiner(basis(x))

        import time

        times = {}
        for sparse in (False, True):
            fn = run(sparse)
            with torch.no_grad():
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    fn()
                torch.cuda.synchronize()
                times[sparse] = time.perf_counter() - t0
        assert times[True] < times[False], times


# Encodings ----------------------------------------------------------------------------


def test_hash_encoding_indexes_coarse_levels_densely_and_hashes_fine_ones():
    encoding = HashEncoding(
        3,
        num_levels=4,
        features_per_level=2,
        log2_hashmap_size=12,
        base_resolution=4,
        max_resolution=32,
    )
    for resolution, size in zip(encoding.resolutions, encoding.table_sizes):
        assert size == -(-min(resolution**3, 2**12) // 8) * 8  # rounded up to 8
    assert encoding.embeddings.shape == (sum(encoding.table_sizes), 2)
    out = encoding(torch.rand(5, 7, 3) * 2 - 1)
    assert out.shape == (5, 7, encoding.out_features) == (5, 7, 8)


def test_hash_encoding_reads_a_vertex_of_a_dense_level_exactly():
    encoding = HashEncoding(
        2,
        num_levels=1,
        features_per_level=1,
        log2_hashmap_size=10,
        base_resolution=4,
        max_resolution=4,
    )
    rows = torch.arange(encoding.embeddings.numel(), dtype=torch.float32)
    with torch.no_grad():
        encoding.embeddings.copy_(rows.view_as(encoding.embeddings))
    # The level reads unit positions at u * scale + 0.5, so vertex k sits at
    # u = (k - 0.5) / scale; dense rows run with the first coordinate fastest.
    vertex = torch.tensor([[1.0, 2.0]])
    x = (vertex - 0.5) / encoding.scales[0] * 2 - 1
    expected = 1 + encoding.resolutions[0] * 2
    assert encoding(x).item() == pytest.approx(expected, abs=1e-4)


def test_hash_encoding_trains_its_tables():
    encoding = HashEncoding(
        3, num_levels=2, log2_hashmap_size=8, base_resolution=2, max_resolution=8
    )
    encoding(torch.rand(10, 3) * 2 - 1).sum().backward()
    assert encoding.embeddings.grad.abs().sum() > 0


def test_hash_encoding_takes_one_to_three_axes():
    with pytest.raises(ValueError, match="in_features"):
        HashEncoding(4)


def test_positional_encoding_lists_the_input_then_the_sines_and_cosines():
    encoding = PositionalEncoding(2, num_frequencies=3)
    x = torch.tensor([[0.5, -0.25]])
    angles = (x.unsqueeze(-1) * torch.tensor([1.0, 2.0, 4.0])).flatten(-2)
    expected = torch.cat([x, angles.sin(), angles.cos()], dim=-1)
    assert encoding.out_features == 14
    torch.testing.assert_close(encoding(x), expected)
