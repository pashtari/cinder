"""Offline backbone checks for CINDER's spatial feature-map contract."""

from functools import partial

import pytest
import timm
import torch
from timm.layers import Format

import cinder as cd

# Use each family's native feature extractor with small inputs and shallow blocks.
# Reversed and repeated selections exercise the order that timm actually emits.
BACKBONES = [
    pytest.param("resnet18", {}, (64, 96), ((32, 48), (2, 3)), id="resnet"),
    pytest.param(
        "efficientnet_b0", {}, (64, 96), ((32, 48), (2, 3)), id="efficientnet"
    ),
    pytest.param(
        "convnext_tiny",
        {"depths": (1, 1, 1, 1)},
        (64, 96),
        ((16, 24), (2, 3)),
        id="convnext",
    ),
    pytest.param(
        "swin_tiny_patch4_window7_224",
        {"img_size": (64, 96), "depths": (1, 1, 1, 1), "embed_dim": 24},
        (64, 96),
        ((16, 24), (2, 3)),
        id="swin",
    ),
    pytest.param(
        "vit_tiny_patch16_224",
        {"img_size": (64, 96), "depth": 2},
        (64, 96),
        ((4, 6), (4, 6)),
        id="vit",
    ),
    pytest.param(
        "vit_small_patch14_reg4_dinov2",
        {"img_size": (56, 84), "depth": 2, "embed_dim": 96, "num_heads": 3},
        (56, 84),
        ((4, 6), (4, 6)),
        id="dinov2-registers",
    ),
    pytest.param(
        "vit_small_patch16_dinov3",
        {"img_size": (64, 96), "depth": 2, "embed_dim": 96, "num_heads": 3},
        (64, 96),
        ((4, 6), (4, 6)),
        id="dinov3-registers",
        marks=pytest.mark.skipif(
            not timm.is_model("vit_small_patch16_dinov3"),
            reason="DINOv3 is unavailable in this timm version",
        ),
    ),
    pytest.param(
        "deit_tiny_distilled_patch16_224",
        {"img_size": (64, 96), "depth": 2},
        (64, 96),
        ((4, 6), (4, 6)),
        id="deit-distilled",
    ),
    pytest.param(
        "pvt_v2_b0",
        {"depths": (1, 1, 1, 1)},
        (64, 96),
        ((16, 24), (2, 3)),
        id="pvt",
    ),
]


@pytest.mark.parametrize("model_name, kwargs, size, grids", BACKBONES)
@torch.no_grad()
def test_backbone_preserves_spatial_features(model_name, kwargs, size, grids):
    encoder = cd.TimmEncoder(
        model_name,
        pretrained=False,
        normalize=False,
        out_indices=(-1, 0, 0),
        **kwargs,
    ).eval()
    images = torch.rand(1, 3, *size)
    features = encoder(images)
    native = encoder.backbone(encoder.channel_norm(images))
    native_format = getattr(encoder.backbone, "output_fmt", Format.NCHW)

    assert isinstance(features, tuple) and len(features) == 2
    assert tuple(feature.shape[1] for feature in features) == encoder.embed_dims
    assert tuple(feature.shape[2:] for feature in features) == grids
    for feature, reference in zip(features, native):
        if native_format == Format.NHWC:
            reference = reference.movedim(-1, 1)
        torch.testing.assert_close(feature, reference, rtol=0, atol=0)

    # Check prefix removal and row-major patch placement against the token stream,
    # independently of timm's intermediate-feature reshaping.
    backbone = getattr(encoder.backbone, "model", None)
    if backbone is not None and hasattr(backbone, "num_prefix_tokens"):
        tokens = backbone.forward_features(encoder.channel_norm(images))
        prefix_count = backbone.num_prefix_tokens
        assert prefix_count == (
            5 if "dino" in model_name else 2 if "distilled" in model_name else 1
        )
        patches = tokens[:, prefix_count:]
        assert patches.shape[1] == grids[-1][0] * grids[-1][1]
        torch.testing.assert_close(features[-1].flatten(2).transpose(1, 2), patches)


@torch.no_grad()
def test_vit_preserves_rectangular_grid_with_dynamic_padding():
    encoder = cd.TimmEncoder(
        "vit_tiny_patch16_224",
        pretrained=False,
        normalize=False,
        depth=2,
        dynamic_img_size=True,
        dynamic_img_pad=True,
    ).eval()
    images = torch.rand(1, 3, 65, 97)
    (feature,) = encoder(images)
    assert feature.shape == (1, 192, 5, 7)
    tokens = encoder.backbone.model.forward_features(encoder.channel_norm(images))
    torch.testing.assert_close(feature.flatten(2).transpose(1, 2), tokens[:, 1:])


@pytest.mark.parametrize("backbone", ["vit", "swin"])
@pytest.mark.parametrize("mechanism", ["input", "weight", "hybrid"])
def test_transformer_features_drive_cinder_gradients(backbone, mechanism):
    torch.manual_seed(0)
    size = (32, 48)
    if backbone == "vit":
        encoder_kwargs = {
            "model_name": "vit_tiny_patch16_224",
            "depth": 2,
            "img_size": size,
        }
    else:
        encoder_kwargs = {
            "model_name": "swin_tiny_patch4_window7_224",
            "img_size": size,
            "depths": (1, 1, 1, 1),
            "embed_dim": 24,
        }
    modulators = []  # innermost first
    if mechanism in ("weight", "hybrid"):
        modulators.append(("displacement", {"conditioner": "linear"}))
    if mechanism in ("input", "hybrid"):
        modulators.append(
            (
                "futon",
                {
                    "basis": ("cosine", {"num_components": 8}),
                    "combiner": ("cp", {"rank": 8}),
                },
            )
        )
    model = cd.CINDER(
        in_channels=3,
        out_channels=2,
        in_size=size,
        encoder=partial(
            cd.TimmEncoder,
            pretrained=False,
            out_indices=(0, -1),
            **encoder_kwargs,
        ),
        inr=("mlp", {"hidden_layers": 1, "hidden_features": 16}),
        modulator=partial(cd.ListModulators, modulators=modulators),
    )
    images = torch.rand(1, 3, *size, requires_grad=True)
    prediction = model(images)
    assert prediction.shape == (1, 2, *size)
    prediction.square().mean().backward()

    for gradient in (images.grad, model.encoder.channel_norm.alpha.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    backbone_gradients = [
        parameter.grad
        for parameter in model.encoder.backbone.parameters()
        if parameter.grad is not None
    ]
    assert backbone_gradients
    assert all(torch.isfinite(gradient).all() for gradient in backbone_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in backbone_gradients)
    for modulator in model.modulator.modulators:
        if isinstance(modulator, cd.FUTONGate):
            for projection in modulator.projections:
                assert torch.count_nonzero(projection.weight.grad) > 0
        else:
            for conditioner in modulator.conditioners:
                assert torch.count_nonzero(conditioner.strength.grad) > 0
