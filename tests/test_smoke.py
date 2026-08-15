"""Smoke tests: the core model must import, instantiate, and run a forward pass.

These deliberately avoid downloading pretrained weights so they can run offline
and fast. Run with ``pytest tests/`` from the project root.
"""

from functools import partial

import torch
import torch.nn as nn

import cinder as cd


def _build_model(in_size=(64, 64), sampling_ratio=1.0):
    # encoder / cinr are passed as factories; CINDER wires in_channels /
    # in_features / out_features (à la INRKMeans).
    encoder = partial(
        cd.TimmEncoder,
        model_name="resnet18",
        pretrained=False,
        normalize=True,
        out_index=2,  # an intermediate CNN stage -> (B, C, H', W')
    )
    cinr = partial(
        cd.ConditionalFUTON,  # cond_dim is inferred from the encoder by CINDER
        basis=("cosine", {"num_components": 16}),
        combiner=("cp", {"rank": 16}),
    )
    return cd.CINDER(
        encoder=encoder,
        cinr=cinr,
        in_channels=3,
        out_channels=1,
        in_size=in_size,
        sampling_ratio=sampling_ratio,
    )


def test_conditional_futon_forward():
    cinr = cd.ConditionalFUTON(
        in_features=2,
        out_features=3,
        cond_dim=32,
        basis=("cosine", {"num_components": 16}),
        combiner=("cp", {"rank": 16}),
    )
    coords = torch.rand(8, 8, 2) * 2 - 1  # (H, W, 2) in [-1, 1]
    cond = torch.randn(4, 32, 5, 6)  # (B, cond_dim, H', W') feature map
    out = cinr(coords, cond)
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
    assert model._sample_indices is not None
    assert model._sample_indices.shape == (n_expected,)


if __name__ == "__main__":
    test_conditional_futon_forward()
    test_cinder_full_forward()
    test_cinder_coordinate_subsampling()
    print("All smoke tests passed.")
