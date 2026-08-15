"""Ignite engine factories for CINDER training and evaluation."""

import torch
from ignite.engine import Engine


def create_trainer(model, optimizer, loss_fn, device):
    """Create an Ignite training engine.

    Args:
        model: The model to train.
        optimizer: The optimizer.
        loss_fn: Loss function (e.g., ``cinder.SampledLoss``-wrapped).
        device: Target device.

    Returns:
        An ``ignite.engine.Engine`` that yields ``loss.item()`` per step.
    """

    def train_step(engine, batch):
        model.train()
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(inputs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        return loss.item()

    return Engine(train_step)


def create_evaluator(model, metrics, device, inferer=None):
    """Create an Ignite evaluation engine.

    When *inferer* is given, it is used for inference (e.g. sliding-window on
    large images); otherwise uses a direct forward pass.

    Args:
        model: The model to evaluate.
        metrics: Dict of ``{name: ignite.metrics.Metric}``.
        device: Target device.
        inferer: Optional callable ``(inputs, model) -> preds``.

    Returns:
        An ``ignite.engine.Engine`` that yields ``(preds, targets)`` per step.
    """

    @torch.no_grad()
    def eval_step(engine, batch):
        model.eval()
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if inferer is not None:
            preds = inferer(inputs, model)
        else:
            preds = model(inputs)

        return preds, targets

    evaluator = Engine(eval_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
