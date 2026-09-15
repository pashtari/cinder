"""Encoder shape probes preserve the model's numerical and training state."""

import pytest
import torch
from torch import nn

from cinder.models.encoders import BaseEncoder


class ProbeEncoder(BaseEncoder):
    def __init__(self):
        super().__init__(3)
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8))

    def forward(self, x):
        return (self.features(x),)


def test_output_shapes_uses_model_dtype_and_preserves_mixed_training_modes():
    encoder = ProbeEncoder().double().train()
    encoder.features[1].eval()
    modes = [module.training for module in encoder.modules()]
    running_mean = encoder.features[1].running_mean.clone()
    assert encoder.get_output_shapes((9, 11)) == ((8, 7, 9),)
    assert [module.training for module in encoder.modules()] == modes
    torch.testing.assert_close(encoder.features[1].running_mean, running_mean)
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_output_shapes_restores_modes_after_failed_probe():
    encoder = ProbeEncoder().train()
    encoder.features[1].eval()
    modes = [module.training for module in encoder.modules()]
    with pytest.raises(RuntimeError):
        encoder.get_output_shapes((1, 1))
    assert [module.training for module in encoder.modules()] == modes


def test_output_shapes_uses_buffers_for_parameterless_encoders():
    class BufferEncoder(BaseEncoder):
        def __init__(self):
            super().__init__(3)
            self.register_buffer("projection", torch.eye(3, dtype=torch.float64))

        def forward(self, x):
            return ((x.movedim(1, -1) @ self.projection).movedim(-1, 1),)

    assert BufferEncoder().get_output_shapes((4, 5)) == ((3, 4, 5),)
