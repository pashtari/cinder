"""The engine factories train, evaluate and hand outputs to metrics and handlers."""

import pytest
import torch
from ignite.engine import Events
from ignite.metrics import Average
from torch import nn

from cinder.engine.engines import create_evaluator, create_trainer, fit


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


def test_evaluator_ignores_amp_on_the_cpu():
    model = nn.Linear(3, 1)
    device = torch.device("cpu")
    plain = create_evaluator(model, {}, device).run(_batches(1)).output
    autocast = create_evaluator(model, {}, device, amp=True).run(_batches(1)).output
    torch.testing.assert_close(plain, autocast)


def _fit(**kwargs):
    torch.manual_seed(0)
    error = Average(
        output_transform=lambda output: (output[0] - output[1]).abs().mean()
    )
    options = dict(
        max_iters=10,
        lr=0.1,
        val_loader=_batches(2),
        metrics={"mae": error},
        device=torch.device("cpu"),
        progress=False,
    )
    # The four batches cycle for as many iterations as needed.
    return fit(nn.Linear(3, 1), nn.MSELoss(), _batches(4), **(options | kwargs))


def test_fit_records_the_loss_and_the_validation_metrics():
    history = _fit(eval_interval=4, log_interval=2)
    losses = [record for record in history if "loss" in record]
    evaluations = [record for record in history if "mae" in record]
    assert [record["iteration"] for record in losses] == [2, 4, 6, 8, 10]
    assert [record["iteration"] for record in evaluations] == [4, 8, 10]
    times = [record["time"] for record in history]
    assert times == sorted(times)


def test_fit_is_reproducible_and_evaluates_at_the_end_by_default():
    def scores(history):  # the records without their timings
        return [{k: v for k, v in record.items() if k != "time"} for record in history]

    history = _fit(log_interval=5)
    assert scores(history) == scores(_fit(log_interval=5))
    assert [record["iteration"] for record in history if "mae" in record] == [10]


def test_fit_reports_its_progress(capsys):
    history = _fit(max_iters=4, val_loader=_batches(1), eval_interval=2, progress=True)
    output = capsys.readouterr()
    assert "Training" in output.err  # the bar
    assert "Iteration 2: mae" in output.out and "Iteration 4: mae" in output.out
    assert [record["iteration"] for record in history if "mae" in record] == [2, 4]
