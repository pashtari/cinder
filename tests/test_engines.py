"""The engine factories train, evaluate and hand outputs to metrics and handlers."""

import pytest
import torch
from ignite.engine import Events
from ignite.metrics import Average
from torch import nn

from cinder.engine.engines import create_evaluator, create_trainer


def _batches(num_batches=3):
    generator = torch.Generator().manual_seed(0)
    return [
        (torch.randn(2, 3, generator=generator), torch.randn(2, 1, generator=generator))
        for _ in range(num_batches)
    ]


@pytest.mark.parametrize("amp", [False, True])
def test_trainer_steps_the_optimizer_and_outputs_a_float_loss(amp):
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    losses = []

    # AMP is a CUDA feature, so a CPU run ignores it and trains in float32.
    trainer = create_trainer(
        model, optimizer, nn.MSELoss(), torch.device("cpu"), amp=amp
    )
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda e: losses.append(e.state.output)
    )
    trainer.run(_batches(), max_epochs=1)

    assert len(losses) == 3 and all(isinstance(loss, float) for loss in losses)
    assert not torch.equal(model.weight, before)
    assert model.weight.dtype == torch.float32


def test_evaluator_uses_the_inferer_and_feeds_metrics():
    model = nn.Linear(3, 1)
    calls = []

    def inferer(inputs, predictor):
        calls.append(inputs.shape)
        return predictor(inputs) * 0  # zero predictions make the metric exact

    average = Average(output_transform=lambda output: output[0].mean())
    evaluator = create_evaluator(
        model, {"mean_prediction": average}, torch.device("cpu"), inferer=inferer
    )
    state = evaluator.run(_batches(2))

    assert calls == [(2, 3), (2, 3)]
    assert state.metrics == {"mean_prediction": 0.0}
    predictions, targets = state.output
    assert predictions.shape == targets.shape == (2, 1)
    assert not predictions.requires_grad and not model.training
