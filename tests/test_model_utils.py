"""Contracts for assembling model components."""

import pytest
import torch

from cinder.models.utils import build_module


@pytest.mark.parametrize("result", [None, torch.zeros(1)])
def test_module_factory_must_return_a_module(result):
    with pytest.raises(TypeError, match="must return an nn.Module"):
        build_module(lambda: result)
