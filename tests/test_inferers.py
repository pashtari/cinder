"""Sliding-window blending must cover every image pixel without losing precision."""

import pytest
import torch

from cinder.engine.inferers import MultiScaleFlipInferer, SlidingWindowInferer


@pytest.mark.parametrize("shape", [(2, 3, 5, 7), (2, 3, 25, 31)])
@pytest.mark.parametrize("mode", ["constant", "gaussian"])
def test_sliding_window_reconstructs_the_image(shape, mode):
    inputs = torch.rand(shape)
    inferer = SlidingWindowInferer((12, 16), sw_batch_size=3, mode=mode)

    result = inferer(inputs, lambda tile: tile + 2)

    torch.testing.assert_close(result, inputs + 2)


@pytest.mark.parametrize("sigma_scale", [0.125, 0.1, 0.001])
def test_half_precision_gaussian_blending_is_finite(sigma_scale):
    inputs = torch.ones(2, 1, 32, 32, dtype=torch.float16)
    inferer = SlidingWindowInferer((16, 16), sigma_scale=sigma_scale, sw_batch_size=3)

    result = inferer(inputs, lambda tile: tile)

    assert result.dtype == torch.float32
    torch.testing.assert_close(result, inputs.float())


def test_half_precision_overlapping_logits_do_not_overflow():
    inputs = torch.full((1, 1, 32, 32), 50000, dtype=torch.float16)
    inferer = SlidingWindowInferer((16, 16), mode="constant")

    result = inferer(inputs, lambda tile: tile)

    torch.testing.assert_close(result, inputs.float())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"roi_size": (0, 8)},
        {"sw_batch_size": 0},
        {"overlap": -0.1},
        {"overlap": 1},
        {"mode": "unknown"},
        {"sigma_scale": 0},
    ],
)
def test_sliding_window_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SlidingWindowInferer(**({"roi_size": (8, 8)} | kwargs))


@pytest.mark.parametrize("scales", [(), (0,), (-1,), (float("nan"),)])
def test_multiscale_rejects_invalid_scales(scales):
    with pytest.raises(ValueError, match="scales"):
        MultiScaleFlipInferer(SlidingWindowInferer((8, 8)), scales=scales)


@pytest.mark.parametrize("flip", [False, True])
def test_multiscale_flip_averages_views_back_at_input_size(flip):
    calls = []

    def inferer(inputs, predictor):
        calls.append(tuple(inputs.shape[-2:]))
        return predictor(inputs)

    image = torch.rand(1, 2, 8, 12)
    tta = MultiScaleFlipInferer(inferer, scales=(0.5, 1.0), flip=flip)

    # A constant image is unchanged by resizing and flipping, so the average is too.
    constant = torch.full_like(image, 3.0)
    torch.testing.assert_close(tta(constant, lambda x: x), constant)
    assert calls == [(4, 6), (4, 6), (8, 12), (8, 12)] if flip else [(4, 6), (8, 12)]

    # At the original scale, flipping back undoes the flip exactly.
    single = MultiScaleFlipInferer(inferer, scales=(1.0,), flip=flip)
    torch.testing.assert_close(single(image, lambda x: 2 * x), 2 * image)
