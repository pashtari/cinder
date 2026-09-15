"""CINDER composition and training contracts, without pretrained backbones."""

from functools import partial

import pytest
import torch
from torch import nn

from cinder.models.cinder import CINDER
from cinder.models.encoders import BaseEncoder
from cinder.models.modulators import FUTONGate, ListModulators, WeightDisplacement


class TinyEncoder(BaseEncoder):
    """A fixed output grid makes input-size validation independent of the encoder."""

    def __init__(self, in_channels):
        super().__init__(in_channels)
        self.embed_dims = (4,)
        self.projection = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.norm = nn.BatchNorm2d(4)
        self.pool = nn.AdaptiveAvgPool2d((3, 2))

    def forward(self, x):
        return (self.pool(self.norm(self.projection(x))),)


INPUT = (
    "futon",
    {
        "basis": ("cosine", {"num_components": 4}),
        "combiner": ("cp", {"rank": 8}),
    },
)
WEIGHT = ("displacement", {"conditioner": "linear"})
MODES = {
    "input": {"modulator": INPUT},
    "weight": {"modulator": WEIGHT},
    "hybrid": {"modulator": partial(ListModulators, modulators=[WEIGHT, INPUT])},
}
FACTORIES = {  # the same modulators, as factories instead of registry keys
    "input": partial(FUTONGate, **INPUT[1]),
    "weight": partial(WeightDisplacement, **WEIGHT[1]),
}
FACTORIES["hybrid"] = partial(
    ListModulators, modulators=[FACTORIES["weight"], FACTORIES["input"]]
)


def build_model(**kwargs):
    return CINDER(
        in_channels=3,
        out_channels=2,
        encoder=TinyEncoder,
        inr=("mlp", {"hidden_features": 8, "hidden_layers": 1}),
        in_size=(6, 8),
        **kwargs,
    )


@pytest.mark.parametrize("mode", MODES)
def test_registry_keys_and_factories_build_the_same_model(mode):
    torch.manual_seed(0)
    direct = build_model(**MODES[mode]).eval()
    torch.manual_seed(0)
    factory = build_model(modulator=FACTORIES[mode]).eval()
    torch.testing.assert_close(
        factory.state_dict(), direct.state_dict(), rtol=0, atol=0
    )
    x = torch.randn(2, 3, 6, 8)

    actual = direct(x)
    assert actual.shape == (2, 2, 6, 8)
    torch.testing.assert_close(actual, factory(x))

    actual.square().mean().backward()
    gradient = direct.encoder.projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_custom_modulator_factories_receive_their_sizes():
    seen = {}

    def gate(inr, in_features, out_features, cond_shape):
        seen["gate"] = (in_features, out_features, cond_shape)
        return FUTONGate(inr, in_features, out_features, cond_shape, **INPUT[1])

    def displacement(inr, in_features, out_features, cond_shape):
        seen["weight"] = (in_features, out_features, cond_shape)
        return WeightDisplacement(
            inr, in_features, out_features, cond_shape, **WEIGHT[1]
        )

    modulator = partial(ListModulators, modulators=[displacement, gate])
    model = build_model(modulator=modulator).eval()
    # The gate reads the coordinates; the displacement follows the gate's 8 features.
    assert seen == {"gate": (2, 2, ((4, 3, 2),)), "weight": (8, 2, ((4, 3, 2),))}
    assert model.modulator.modulators[0].inr.in_features == 8
    with torch.no_grad():
        assert model(torch.randn(2, 3, 6, 8)).shape == (2, 2, 6, 8)


def test_sampled_predictions_align_with_flat_targets_and_reset_in_eval():
    torch.manual_seed(0)
    model = build_model(**MODES["hybrid"], sampling_ratio=0.25, freeze_encoder=True)
    x = torch.randn(2, 3, 6, 8)
    with torch.no_grad():
        full = model.eval()(x)
        sampled = model.train()(x)
        indices = model.sample_indices
        assert indices is not None and indices.unique().numel() == 12
        assert sampled.shape == (2, 2, 12)
        torch.testing.assert_close(sampled, full.flatten(2)[..., indices])
        torch.testing.assert_close(model.eval()(x), full)
    assert model.sample_indices is None


def test_frozen_encoder_keeps_statistics_before_and_after_explicit_train():
    model = build_model(**MODES["input"], freeze_encoder=True)
    assert model.training and not model.encoder.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    statistics = {name: value.clone() for name, value in model.encoder.named_buffers()}
    x = torch.randn(2, 3, 6, 8) + 3

    model(x).square().mean().backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.modulator.parameters())
    model.train()
    assert model.training and not model.encoder.training
    model(x)
    for name, value in model.encoder.named_buffers():
        torch.testing.assert_close(value, statistics[name], rtol=0, atol=0)


def test_input_size_is_checked_when_encoder_feature_shapes_would_match():
    model = build_model(**MODES["input"]).eval()
    wrong_size = torch.randn(2, 3, 7, 8)
    feature_shapes = tuple(
        tuple(feature.shape[1:]) for feature in model.encoder(wrong_size)
    )
    assert feature_shapes == model.modulator.cond_shape
    with pytest.raises(ValueError, match="spatial size"):
        model(wrong_size)


@pytest.mark.parametrize("sampling_ratio", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_sampling_ratio_raises_value_error(sampling_ratio):
    with pytest.raises(ValueError, match="sampling_ratio"):
        build_model(**MODES["input"], sampling_ratio=sampling_ratio)


def test_modulator_is_required():
    with pytest.raises(TypeError, match="modulator"):
        build_model()
    with pytest.raises(ValueError, match="at least one"):
        build_model(modulator=partial(ListModulators, modulators=[]))
