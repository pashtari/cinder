"""Ignite engines that train and evaluate CINDER."""

from collections.abc import Callable, Mapping

import torch
from ignite.engine import Engine
from ignite.metrics import Metric
from torch import Tensor, nn
from torch.optim import Optimizer

__all__ = ["create_trainer", "create_evaluator"]


def create_trainer(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    non_blocking: bool = True,
    amp: bool = False,
) -> Engine:
    """Create an engine that takes one optimizer step per batch.

    Args:
        model: Model to train.
        optimizer: Optimizer over the model's parameters.
        loss_fn: ``(predictions, targets) -> loss``.
        device: Device that each batch is moved to.
        non_blocking: Passed to ``Tensor.to`` when moving a batch.
        amp: Use CUDA autocast, with bfloat16 where supported and otherwise
            float16 with gradient scaling. Ignored on CPU.

    Returns:
        An engine whose output is the batch loss as a Python float, which the
        progress and TensorBoard handlers log.
    """
    use_amp = bool(amp) and device.type == "cuda"
    dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    )
    # Scaling keeps float16 gradients from underflowing; bfloat16 has no such issue.
    needs_scaler = use_amp and dtype is torch.float16
    # torch.amp.GradScaler supersedes torch.cuda.amp.GradScaler from PyTorch 2.3.
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=needs_scaler)

    def train_step(_: Engine, batch: tuple[Tensor, Tensor]) -> float:
        model.train()
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.item()

    return Engine(train_step)


def create_evaluator(
    model: nn.Module,
    metrics: Mapping[str, Metric],
    device: torch.device,
    non_blocking: bool = True,
    inferer: Callable[[Tensor, nn.Module], Tensor] | None = None,
) -> Engine:
    """Create an engine that predicts each batch and updates ``metrics``.

    Args:
        model: Model to evaluate.
        metrics: Ignite metrics to attach, keyed by the name they are logged as.
        device: Device that each batch is moved to.
        non_blocking: Passed to ``Tensor.to`` when moving a batch.
        inferer: ``(inputs, model) -> predictions``, such as sliding-window
            inference; ``None`` calls the model directly.

    Returns:
        An engine whose output is ``(predictions, targets)``.
    """

    @torch.no_grad()
    def eval_step(_: Engine, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        model.eval()
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        predictions = model(inputs) if inferer is None else inferer(inputs, model)
        return predictions, targets

    evaluator = Engine(eval_step)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    return evaluator
