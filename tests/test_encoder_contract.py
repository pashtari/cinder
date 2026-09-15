"""Backbone-independent contracts for spatial timm features."""

from collections import OrderedDict

import pytest
import torch
from timm.layers import Format
from timm.models._features import FeatureInfo
from torch import nn

from cinder.models.encoders import ChannelNorm, TimmEncoder


class FakeFeatures(nn.Module):
    """Expose timm's metadata while controlling feature values and geometry."""

    def __init__(
        self,
        outputs=None,
        channels=(3,),
        out_indices=(-1,),
        output_fmt=Format.NCHW,
        pretrained_cfg=None,
    ):
        super().__init__()
        self.outputs = outputs
        self.output_fmt = output_fmt
        self.pretrained_cfg = pretrained_cfg or {}
        self.feature_info = FeatureInfo(
            [
                {"num_chs": width, "reduction": 2**i, "module": f"stage{i}"}
                for i, width in enumerate(channels)
            ],
            out_indices,
        )

    def forward(self, x):
        return [x] if self.outputs is None else self.outputs


@pytest.fixture
def wrap(monkeypatch):
    def create(backbone, **kwargs):
        monkeypatch.setattr(
            "cinder.models.encoders.timm.create_model", lambda *a, **k: backbone
        )
        return TimmEncoder("fake", pretrained=False, **kwargs)

    return create


@pytest.mark.parametrize("output_fmt", [Format.NCHW, Format.NHWC, "NHWC", None])
def test_feature_layout_uses_metadata_even_when_axes_have_equal_lengths(
    wrap, output_fmt
):
    feature = torch.arange(54, dtype=torch.float32).reshape(2, 3, 3, 3)
    encoder = wrap(FakeFeatures([feature], output_fmt=output_fmt), normalize=False)

    (actual,) = encoder(torch.zeros(2, 3, 7, 11))

    expected = (
        feature.movedim(-1, 1) if output_fmt in (Format.NHWC, "NHWC") else feature
    )
    torch.testing.assert_close(actual, expected)
    assert encoder.output_fmt == Format.NCHW


def test_rectangular_stage_maps_keep_their_own_grids_and_dictionary_order(wrap):
    maps = OrderedDict(
        [
            ("shallow", torch.randn(2, 3, 5, 7)),
            ("deep", torch.randn(2, 8, 2, 3)),
        ]
    )
    encoder = wrap(
        FakeFeatures(maps, channels=(3, 8), out_indices=(0, 1)), normalize=False
    )

    actual = encoder(torch.zeros(2, 3, 20, 28))

    assert isinstance(actual, tuple)
    assert encoder.embed_dims == (3, 8)
    assert encoder.feature_reductions == (1, 2)
    for returned, expected in zip(actual, maps.values()):
        torch.testing.assert_close(returned, expected)


@pytest.mark.parametrize("output_fmt", ["NLC", "NCL", "CHWN"])
def test_token_or_unknown_layout_is_rejected_without_guessing_a_grid(wrap, output_fmt):
    with pytest.raises(ValueError, match="feature format"):
        wrap(FakeFeatures(output_fmt=output_fmt))


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "outputs, message",
    [
        ([], "feature maps"),
        ([torch.zeros(2, 3, 5, 7)] * 2, "feature maps"),
        ([torch.zeros(2, 35, 3)], "4D"),
        ([torch.zeros(2, 4, 5, 7)], "channels"),
        ([torch.zeros(1, 3, 5, 7)], "batch size"),
        (["not a tensor"], "Tensor|tensor|feature"),
        (torch.zeros(2, 3, 5, 7), "feature maps|list|tuple|sequence"),
    ],
    ids=[
        "missing-map",
        "extra-map",
        "tokens",
        "channels",
        "batch",
        "non-tensor",
        "bare-tensor",
    ],
)
def test_malformed_features_fail_before_normalization(
    wrap, normalize, outputs, message
):
    encoder = wrap(FakeFeatures(outputs), normalize=normalize)
    with pytest.raises((TypeError, ValueError, RuntimeError), match=message):
        encoder(torch.zeros(2, 3, 20, 28))


@pytest.mark.parametrize("index", [-4, 3])
def test_out_of_range_stage_indices_are_not_wrapped(wrap, index):
    with pytest.raises((ValueError, IndexError), match="out_indices"):
        wrap(FakeFeatures(channels=(3, 8, 16), out_indices=(index,)))


@pytest.mark.parametrize("normalize", [False, True])
def test_patch_dropout_is_rejected_for_dense_features(wrap, normalize):
    with pytest.raises(ValueError, match="patch_drop"):
        wrap(FakeFeatures(), normalize=normalize, patch_drop_rate=0.5)


@pytest.mark.parametrize("in_channels", [1, 3, 5])
@pytest.mark.parametrize(
    "pretrained_cfg, mean, std",
    [
        ({}, ChannelNorm.MEAN, ChannelNorm.STD),
        ({"mean": (0.5,) * 3, "std": (0.5,) * 3}, (0.5,) * 3, (0.5,) * 3),
        (
            {"mean": (0.1, 0.2, 0.3), "std": (0.2, 0.4, 0.5)},
            (0.1, 0.2, 0.3),
            (0.2, 0.4, 0.5),
        ),
    ],
    ids=["fallback", "symmetric", "custom"],
)
def test_input_normalization_follows_backbone_stats_and_cycles_channels(
    wrap, in_channels, pretrained_cfg, mean, std
):
    encoder = wrap(
        FakeFeatures(channels=(in_channels,), pretrained_cfg=pretrained_cfg),
        in_channels=in_channels,
        normalize=False,
    )
    x = torch.linspace(0, 1, 2 * in_channels * 5 * 7).reshape(2, in_channels, 5, 7)
    mean = x.new_tensor([mean[i % len(mean)] for i in range(in_channels)]).view(
        1, -1, 1, 1
    )
    std = x.new_tensor([std[i % len(std)] for i in range(in_channels)]).view(
        1, -1, 1, 1
    )

    (actual,) = encoder(x)

    torch.testing.assert_close(actual, (x - mean) / std)


def test_saved_learned_input_normalization_overrides_new_initialization(wrap):
    encoder = wrap(
        FakeFeatures(pretrained_cfg={"mean": (0.5,) * 3, "std": (0.5,) * 3}),
        normalize=False,
    )
    state = {
        "alpha": torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1),
        "beta": torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1, 1),
    }
    encoder.channel_norm.load_state_dict(state, strict=True)

    assert set(encoder.channel_norm.state_dict()) == {"alpha", "beta"}
    x = torch.randn(2, 3, 5, 7)
    (actual,) = encoder(x)
    torch.testing.assert_close(actual, state["alpha"] * x + state["beta"])
