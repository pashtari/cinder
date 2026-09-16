"""Smoke tests: the core model must import, instantiate, and run a forward pass.

These deliberately avoid downloading pretrained weights so they can run offline
and fast. Run with ``pytest tests/`` from the project root.
"""

import os
from functools import partial

import pytest
import torch

import cinder as cd
from cinder.models.inrs import SincBasis
from cinder.models.modulators import SumFusion

# A small gate, so the smoke tests stay light: the INR reads 16 features.
SMALL_GATE = {
    "basis": ("cosine", {"num_components": 16}),
    "combiner": ("cp", {"rank": 16}),
}


def _build_model(in_size=(64, 64), sampling_ratio=1.0):
    # encoder / inr / modulator are passed as factories; CINDER wires
    # in_channels / in_features / out_features / cond_shape.
    encoder = partial(
        cd.TimmEncoder,
        model_name="resnet18",
        pretrained=False,
        normalize=True,
        out_indices=[2],  # an intermediate CNN stage -> ((B, C, H', W'),)
    )
    # in/out features come from the modulator; hidden_features is read back below
    inr = partial(cd.MLP, hidden_layers=1, hidden_features=24)
    modulator = partial(  # cond_shape is inferred from the encoder by CINDER
        cd.ListModulators, modulators=("futon", {**SMALL_GATE})
    )
    return cd.CINDER(
        encoder=encoder,
        inr=inr,
        modulator=modulator,
        in_channels=3,
        out_channels=1,
        in_size=in_size,
        sampling_ratio=sampling_ratio,
    )


def _build_multiscale_model(in_size=(64, 64), out_indices=(-3, -2, -1)):
    """Same model, conditioned on several encoder stages at once."""
    encoder = partial(
        cd.TimmEncoder,
        model_name="resnet18",
        pretrained=False,
        normalize=True,
        out_indices=out_indices,
    )
    inr = ("mlp", {"hidden_layers": 1, "hidden_features": 24})  # a (key, params) spec
    modulator = partial(
        cd.ListModulators, modulators=("futon", {**SMALL_GATE, "fusion": "sum"})
    )
    return cd.CINDER(
        encoder=encoder,
        inr=inr,
        modulator=modulator,
        in_channels=3,
        out_channels=1,
        in_size=in_size,
    )


def test_modulator_forward():
    inr = partial(
        cd.FUTON,
        basis=("cosine", {"num_components": 16}),
        combiner=("cp", {"rank": 16}),
        decoder=("mlp", {"hidden_layers": 1}),
    )
    modulator = cd.ListModulators(
        inr=inr,
        in_features=2,
        out_features=3,
        cond_shape=((32, 5, 6),),
        modulators="futon",
    )
    coords = torch.rand(1, 8, 8, 2) * 2 - 1  # (1, H, W, 2) in [-1, 1], shared
    cond = torch.randn(4, 32, 5, 6)  # (B, C, H', W') feature map
    out = modulator(coords, cond)
    assert out.shape == (4, 8, 8, 3)


def test_cinder_full_forward():
    model = _build_model(in_size=(64, 64))
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    # (B, C_out, H, W)
    assert y.shape == (2, 1, 64, 64)


def test_cinder_coordinate_subsampling():
    model = _build_model(in_size=(64, 64), sampling_ratio=0.25)
    model.train()
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    n_total = 64 * 64
    n_expected = max(1, int(0.25 * n_total))
    # During training with sampling, output is (B, C_out, N_sampled)
    assert y.shape == (2, 1, n_expected)
    assert model.sample_indices is not None
    assert model.sample_indices.shape == (n_expected,)


def test_encoder_returns_one_map_per_stage():
    """`out_indices` selects the stages; each comes back on its own grid."""
    encoder = cd.TimmEncoder(
        model_name="resnet18", pretrained=False, out_indices=(-3, -2, -1)
    )
    assert encoder.out_indices == (2, 3, 4)  # resolved against resnet18's 5 stages
    assert encoder.embed_dims == (128, 256, 512)

    features = encoder(torch.randn(2, 3, 64, 64))
    assert isinstance(features, tuple) and len(features) == 3
    assert [tuple(f.shape) for f in features] == [
        (2, 128, 8, 8),
        (2, 256, 4, 4),
        (2, 512, 2, 2),
    ]
    # Each map gets its own LayerNorm, sized to that stage's channel count, so
    # a deep stage's scale cannot swamp a shallow one's through a shared one.
    assert len(encoder.layer_norms) == 3
    assert len({id(norm) for norm in encoder.layer_norms}) == 3
    for norm, dim, f in zip(encoder.layer_norms, encoder.embed_dims, features):
        assert tuple(norm.normalized_shape) == (dim,) == (f.shape[1],)
        assert torch.allclose(f.movedim(1, -1).mean(-1), torch.zeros(1), atol=1e-5)

    # A single stage, or the default (deepest stage), is still a 1-tuple.
    assert (
        len(
            cd.TimmEncoder("resnet18", pretrained=False, out_indices=[2])(
                torch.randn(1, 3, 64, 64)
            )
        )
        == 1
    )
    assert (
        len(cd.TimmEncoder("resnet18", pretrained=False)(torch.randn(1, 3, 64, 64)))
        == 1
    )


def test_encoder_resolves_out_indices_to_the_order_maps_arrive_in():
    """timm emits stages ascending however they were asked for; norms must follow.

    Regression: pairing `feature_info.channels()` (request order) with the maps
    (ascending) mismatched every non-ascending selection -- a LayerNorm size
    error on backbones whose stages differ in width, and a silent stage swap on
    those whose do not (any ViT).
    """
    descending = cd.TimmEncoder("resnet18", pretrained=False, out_indices=[-1, -3])
    assert descending.out_indices == (2, 4)
    assert descending.embed_dims == (128, 512)
    features = descending(torch.randn(1, 3, 64, 64))
    assert tuple(f.shape[1] for f in features) == descending.embed_dims

    # Same stage twice is one stage: no phantom map, no dead LayerNorm.
    duplicated = cd.TimmEncoder("resnet18", pretrained=False, out_indices=(-1, -1))
    assert duplicated.out_indices == (4,)
    assert duplicated.embed_dims == (512,)
    assert (
        len(duplicated.layer_norms) == len(duplicated(torch.randn(1, 3, 64, 64))) == 1
    )

    # Uniform-width backbone: the mismatch would be silent, so pin the order by
    # the resolved indices themselves.
    vit = cd.TimmEncoder(
        "vit_tiny_patch16_224", pretrained=False, out_indices=[-1, -3], img_size=64
    )
    assert vit.out_indices == (9, 11)


def test_encoder_canonicalizes_channels_last_backbones():
    """Swin & co. emit ``(B, H, W, C)``; the maps still come back channels-first."""
    encoder = cd.TimmEncoder(
        "swin_tiny_patch4_window7_224", pretrained=False, out_indices=(-2, -1)
    )
    assert encoder.channels_last  # the case the canonicalization is there for
    assert encoder.embed_dims == (384, 768)

    features = encoder(torch.randn(1, 3, 224, 224))
    assert [tuple(f.shape) for f in features] == [(1, 384, 14, 14), (1, 768, 7, 7)]


def test_cinder_multiscale_forward():
    """CINDER wires every stage's full shape into the modulator and decodes."""
    model = _build_multiscale_model(in_size=(64, 64))
    assert model.modulator.cond_shape == ((128, 8, 8), (256, 4, 4), (512, 2, 2))
    assert model.modulator.modulators[0].inr.hidden_features == 24  # the pair's params

    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, 64, 64))
    assert y.shape == (2, 1, 64, 64)

    # Gradients reach every stage's projection, not just one.
    model.train()
    model(torch.randn(2, 3, 64, 64)).sum().backward()
    for projection in model.modulator.modulators[0].projections:
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad) > 0


def test_cinder_hands_the_inr_to_the_modulator():
    """CINDER builds the modulator, which builds the INR to the sizes wired in."""
    model = _build_model(in_size=(64, 64))
    modulator = model.modulator
    assert isinstance(modulator, cd.ListModulators)
    (gate,) = modulator.modulators
    assert isinstance(gate, cd.FUTONGate)
    assert (modulator.in_features, modulator.out_features) == (2, 1)
    assert modulator.cond_shape == ((128, 8, 8),)

    assert modulator.inr is gate
    inr = gate.inr
    assert isinstance(inr, cd.MLP)
    # The INR reads the gate's encoding, so its input is that width.
    assert inr.in_features == gate.encoding.out_features == 16
    assert inr.out_features == 1
    assert inr.hidden_features == 24  # the partial's own params arrived


def test_cinder_injects_sizes_by_keyword_and_hands_the_inr_spec_through():
    """Exactly the injected keywords reach the modulator, with the INR spec as is."""
    encoder = partial(
        cd.TimmEncoder, model_name="resnet18", pretrained=False, out_indices=(-2, -1)
    )
    inr = partial(cd.MLP, hidden_layers=1)
    seen = {}

    def modulator(*args, **kwargs):
        assert not args, "sizes must be injected by keyword"
        seen.update(kwargs)
        gate = {
            "basis": ("cosine", {"num_components": 8}),
            "combiner": ("cp", {"rank": 8}),
        }
        return cd.ListModulators(modulators=("futon", gate), **kwargs)

    model = cd.CINDER(
        encoder=encoder,
        inr=inr,
        modulator=modulator,
        in_channels=3,
        out_channels=2,
        in_size=(32, 32),
    )
    assert seen == {
        "inr": inr,
        "in_features": 2,
        "out_features": 2,
        "cond_shape": ((256, 2, 2), (512, 1, 1)),
    }
    assert seen["inr"] is inr  # handed through untouched, not rebuilt or wrapped
    assert model.modulator.modulators[0].inr.out_features == 2


def test_hydra_model_config_builds_the_model():
    """The shipped model config wires `inr` + `modulator` through Hydra."""
    hydra = pytest.importorskip("hydra")
    pytest.importorskip("hydra_plugins.hydra_colorlog")  # configs/hydra/default.yaml
    from hydra import compose, initialize_config_dir

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(root, "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[  # keep it small and offline
                "dataset.crop_size=[32,32]",
                "model.encoder.model_name=resnet18",
                "model.encoder.pretrained=false",
                "model.encoder.out_indices=[-2,-1]",
            ],
        )
    model = hydra.utils.instantiate(cfg.model)

    assert isinstance(model, cd.CINDER)
    # A single modulator is configured as the modulator itself, with no list.
    gate = modulator = model.modulator
    assert isinstance(gate, cd.FUTONGate)
    assert isinstance(gate.inr, cd.MLP)
    assert gate.inr.activation is torch.relu  # the MLP's default
    assert modulator.cond_shape == ((256, 2, 2), (512, 1, 1))
    assert (modulator.in_features, modulator.out_features) == (2, 1)
    assert modulator.out_features == cfg.dataset.num_classes
    assert gate.encoding.out_features == gate.inr.in_features == 256
    assert modulator.out_features == 1
    assert isinstance(gate.fusion, SumFusion)
    assert gate.encoding.basis.num_components == [32, 32]  # ${dataset.crop_size}

    with torch.no_grad():
        assert model.eval()(torch.zeros(1, 3, 32, 32)).shape == (1, 1, 32, 32)


def test_cinder_drives_a_weight_modulated_inr():
    """The other modulator: the encoder displaces the INR's weights per image."""
    torch.manual_seed(0)
    encoder = partial(
        cd.TimmEncoder, model_name="resnet18", pretrained=False, out_indices=(-2, -1)
    )
    model = cd.CINDER(
        encoder=encoder,
        inr=partial(cd.SIREN, hidden_features=32, hidden_layers=2),
        modulator=partial(
            cd.ListModulators, modulators=("displacement", {"conditioner": "linear"})
        ),
        in_channels=3,
        out_channels=1,
        in_size=(64, 64),
    )
    modulator = model.modulator
    assert isinstance(modulator, cd.ListModulators)
    (displacement,) = modulator.modulators
    assert isinstance(displacement, cd.WeightDisplacement)
    assert modulator.cond_shape == ((256, 4, 4), (512, 2, 2))
    assert displacement.in_shape == (4, 512)  # all 2x2 positions, final-stage channels
    # The INR reads the coordinates themselves here, not an encoding of them.
    assert (displacement.inr.in_features, displacement.inr.out_features) == (2, 1)
    assert displacement.weight_names == tuple(
        name for name, p in displacement.inr.named_parameters() if p.ndim == 2
    )

    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, 64, 64))
    assert y.shape == (2, 1, 64, 64)

    # Gradients reach the conditioners that produce the weights.
    model.train()
    model(torch.randn(2, 3, 64, 64)).sum().backward()
    for conditioner in displacement.conditioners:
        assert torch.count_nonzero(conditioner.strength.grad) > 0


@pytest.mark.parametrize("model_name", ["weight_futon", "weight_siren", "weight_relu"])
def test_hydra_weight_model_config_builds_the_model(model_name):
    """Every weight config builds and backpropagates through its chosen INR."""
    hydra = pytest.importorskip("hydra")
    pytest.importorskip("hydra_plugins.hydra_colorlog")
    from hydra import compose, initialize_config_dir

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(root, "configs")
    width_override = (
        "model.inr.combiner.1.rank=16"
        if model_name == "weight_futon"
        else "model.inr.hidden_features=32"
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                f"model={model_name}",
                "dataset.crop_size=[32,48]",
                "model.encoder.model_name=resnet18",
                "model.encoder.pretrained=false",
                width_override,
            ],
        )
    model = hydra.utils.instantiate(cfg.model).eval()

    displacement = modulator = model.modulator
    assert isinstance(displacement, cd.WeightDisplacement)
    assert modulator.cond_shape == ((512, 1, 2),)
    assert displacement.in_shape == (2, 512)
    inr = displacement.inr
    if model_name == "weight_futon":
        assert isinstance(inr, cd.FUTON)
        assert inr.encoding.basis.num_components == [256, 256]
        assert isinstance(inr.encoding.basis, SincBasis)
        assert inr.encoding.out_features == 16
        assert isinstance(inr.decoder, cd.MLP)  # with the MLP's default ReLU
    elif model_name == "weight_siren":
        assert isinstance(inr, cd.SIREN)
        assert inr.hidden_features == 32
    else:
        assert isinstance(inr, cd.MLP) and not isinstance(inr, cd.SIREN)
        assert inr.activation is torch.relu  # the MLP's default
        assert inr.hidden_features == 32

    output = model(torch.randn(2, 3, 32, 48))
    assert output.shape == (2, 1, 32, 48)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    for conditioner in displacement.conditioners:
        assert conditioner.strength.grad is not None
        assert torch.isfinite(conditioner.strength.grad).all()
        assert torch.count_nonzero(conditioner.strength.grad) > 0


def test_cinder_drives_a_hybrid_modulated_inr():
    """Both mechanisms: the gate feeds the INR, whose matrices are displaced."""
    torch.manual_seed(0)
    encoder = partial(
        cd.TimmEncoder, model_name="resnet18", pretrained=False, out_indices=(-2, -1)
    )
    model = cd.CINDER(
        encoder=encoder,
        inr=partial(cd.MLP, hidden_layers=1, hidden_features=24),
        modulator=partial(
            cd.ListModulators,
            modulators=[
                ("displacement", {"conditioner": "linear"}),
                ("futon", SMALL_GATE),
            ],
        ),
        in_channels=3,
        out_channels=1,
        in_size=(64, 64),
    )
    modulator = model.modulator
    displacement, gate = modulator.modulators
    assert isinstance(gate, cd.FUTONGate)
    assert isinstance(displacement, cd.WeightDisplacement)
    # The INR reads the gate's 16 features, and its first matrix is (24, 16).
    assert displacement.inr.in_features == gate.encoding.out_features == 16
    assert displacement.in_features == 16  # the displacement follows the gate
    assert displacement.in_shape == (4, 512)  # the maps' rank (2), not the INR's width
    assert [c.out_shape for c in displacement.conditioners] == [(24, 16), (1, 24)]

    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, 64, 64))
    assert y.shape == (2, 1, 64, 64)

    # Gradients reach the gate's projections and the displacement's strengths.
    model.train()
    model(torch.randn(2, 3, 64, 64)).sum().backward()
    for projection in gate.projections:
        assert torch.count_nonzero(projection.weight.grad) > 0
    for conditioner in displacement.conditioners:
        assert torch.count_nonzero(conditioner.strength.grad) > 0


def test_hydra_hybrid_model_config_builds_the_model():
    """`futongate_weight_relu` inherits the gate and adds weight modulation."""
    hydra = pytest.importorskip("hydra")
    pytest.importorskip("hydra_plugins.hydra_colorlog")  # configs/hydra/default.yaml
    from hydra import compose, initialize_config_dir

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(root, "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[  # keep it small and offline
                "model=futongate_weight_relu",
                "dataset.crop_size=[32,32]",
                "model.encoder.model_name=resnet18",
                "model.encoder.pretrained=false",
                "model.encoder.out_indices=[-2,-1]",
            ],
        )
    model = hydra.utils.instantiate(cfg.model)

    modulator = model.modulator
    displacement, gate = modulator.modulators
    assert isinstance(gate, cd.FUTONGate)
    assert isinstance(displacement, cd.WeightDisplacement)
    assert displacement.inr.activation is torch.relu  # the MLP's default
    assert gate.encoding.basis.num_components == [32, 32]  # ${dataset.crop_size}
    assert displacement.inr.in_features == gate.encoding.out_features == 256
    assert displacement.in_shape == (1, 512)  # the encoder produces a 1x1 final map
    assert [c.out_shape for c in displacement.conditioners] == [(256, 256), (1, 256)]

    with torch.no_grad():
        assert model.eval()(torch.zeros(1, 3, 32, 32)).shape == (1, 1, 32, 32)


def test_cinder_rejects_ready_made_modules():
    """Instances bypass the size wiring, so every spec must be a factory."""
    encoder = partial(
        cd.TimmEncoder, model_name="resnet18", pretrained=False, out_indices=[2]
    )
    inr = partial(cd.MLP, hidden_layers=1)
    modulator = partial(cd.ListModulators, modulators=("futon", SMALL_GATE))
    kw = dict(in_channels=3, out_channels=1, in_size=(64, 64))

    with pytest.raises(TypeError, match="`inr` must be a factory"):
        cd.CINDER(encoder=encoder, inr=inr(16, 1), modulator=modulator, **kw)

    ready = modulator(inr=inr, in_features=2, out_features=1, cond_shape=((128, 8, 8),))
    with pytest.raises(TypeError, match="`modulator` must be a factory"):
        cd.CINDER(encoder=encoder, inr=inr, modulator=ready, **kw)

    with pytest.raises(TypeError, match="`encoder` must be a factory"):
        cd.CINDER(encoder=encoder(), inr=inr, modulator=modulator, **kw)


if __name__ == "__main__":
    test_modulator_forward()
    test_cinder_full_forward()
    test_cinder_coordinate_subsampling()
    test_encoder_returns_one_map_per_stage()
    test_encoder_resolves_out_indices_to_the_order_maps_arrive_in()
    test_encoder_canonicalizes_channels_last_backbones()
    test_cinder_multiscale_forward()
    test_cinder_hands_the_inr_to_the_modulator()
    test_cinder_injects_sizes_by_keyword_and_hands_the_inr_spec_through()
    test_hydra_model_config_builds_the_model()
    test_cinder_drives_a_weight_modulated_inr()
    for model_name in ("weight_futon", "weight_siren", "weight_relu"):
        test_hydra_weight_model_config_builds_the_model(model_name)
    test_cinder_drives_a_hybrid_modulated_inr()
    test_hydra_hybrid_model_config_builds_the_model()
    test_cinder_rejects_ready_made_modules()
    print("All smoke tests passed.")
