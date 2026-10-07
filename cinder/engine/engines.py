"""Ignite engines that train and evaluate CINDER, and a loop that runs them."""

import time
from collections.abc import Callable, Iterable, Mapping

import torch
from ignite.engine import Engine, Events
from ignite.handlers import LRScheduler, ProgressBar, create_lr_scheduler_with_warmup
from ignite.metrics import Metric
from torch import Tensor, nn
from torch.optim import Optimizer

__all__ = ["create_trainer", "create_evaluator", "fit"]


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
    amp: bool = False,
) -> Engine:
    """Create an engine that predicts each batch and updates ``metrics``.

    Args:
        model: Model to evaluate.
        metrics: Ignite metrics to attach, keyed by the name they are logged as.
        device: Device that each batch is moved to.
        non_blocking: Passed to ``Tensor.to`` when moving a batch.
        inferer: ``(inputs, model) -> predictions``, such as sliding-window
            inference; ``None`` calls the model directly.
        amp: Predict under CUDA autocast, with bfloat16 where supported and
            otherwise float16, and return the predictions in float32. Ignored on
            CPU.

    Returns:
        An engine whose output is ``(predictions, targets)``.
    """
    use_amp = bool(amp) and device.type == "cuda"
    dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    )

    @torch.no_grad()
    def eval_step(_: Engine, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        model.eval()
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        if not use_amp:
            predictions = model(inputs) if inferer is None else inferer(inputs, model)
            return predictions, targets
        with torch.autocast("cuda", dtype=dtype):
            predictions = model(inputs) if inferer is None else inferer(inputs, model)
        return predictions.float(), targets

    evaluator = Engine(eval_step)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    return evaluator


def fit(
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    train_loader: Iterable,
    max_iters: int,
    lr: float,
    *,
    weight_decay: float = 0.01,
    warmup_iters: int | None = None,
    val_loader: Iterable | None = None,
    metrics: Mapping[str, Metric] | None = None,
    eval_interval: int | None = None,
    log_interval: int = 50,
    amp: bool = True,
    device: torch.device | None = None,
    seed: int = 0,
    progress: bool = True,
) -> list[dict[str, float]]:
    """Train a model with the recipe of the configs and record its convergence.

    AdamW warms up linearly from ``1e-6`` over ``warmup_iters`` steps, 1% of
    ``max_iters`` by default, under a cosine schedule that decays to zero at
    ``max_iters``, as ``configs/trainer/default.yaml`` does. The loader cycles
    as needed, and the model is evaluated on ``val_loader`` every
    ``eval_interval`` iterations and at the end.

    Args:
        model: Model to train in place.
        loss_fn: ``(predictions, targets) -> loss``, such as a
            :class:`~cinder.engine.losses.SampledLoss`.
        train_loader: Training batches ``(inputs, targets)``.
        max_iters: Number of optimizer steps.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_iters: Length of the linear warmup; ``None`` uses 1%.
        val_loader: Validation batches; ``None`` skips the evaluations.
        metrics: Ignite metrics computed on ``val_loader``.
        eval_interval: Iterations between evaluations; ``None`` evaluates only at
            the end.
        log_interval: Iterations between records of the training loss.
        amp: Train and evaluate under CUDA autocast; see :func:`create_trainer`.
        device: Training device; ``None`` selects CUDA when available.
        seed: Seed of the global PyTorch generator, which orders the batches.
        progress: Show a progress bar with the training loss, and print the
            metrics of every evaluation.

    Returns:
        The history: every ``log_interval`` iterations a record of the
        ``iteration``, the training ``time`` in seconds so far, which leaves the
        evaluations out, and the mean training ``loss`` since the previous
        record, and at every evaluation a record of the ``iteration``, ``time``
        and the metrics.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = create_trainer(model, optimizer, loss_fn, device, amp=amp)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
    warmup_iters = round(0.01 * max_iters) if warmup_iters is None else warmup_iters
    # Ignite's warmup needs at least two steps; shorter warmups are skipped.
    if warmup_iters > 1:
        scheduler = create_lr_scheduler_with_warmup(
            cosine, warmup_start_value=1e-6, warmup_duration=warmup_iters
        )
    else:
        scheduler = LRScheduler(cosine)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    if progress:
        bar = ProgressBar(persist=True, desc="Training")
        bar.attach(trainer, output_transform=lambda loss: {"loss": loss})

    history: list[dict[str, float]] = []
    losses: list[float] = []
    start, paused = time.perf_counter(), 0.0

    def elapsed() -> float:
        return time.perf_counter() - start - paused

    @trainer.on(Events.ITERATION_COMPLETED)
    def record_loss(engine: Engine) -> None:
        losses.append(engine.state.output)  # a float, so the GPU has caught up
        if engine.state.iteration % log_interval == 0:
            record = {"iteration": engine.state.iteration, "time": elapsed()}
            history.append(record | {"loss": sum(losses) / len(losses)})
            losses.clear()

    if val_loader is not None:
        evaluator = create_evaluator(model, metrics or {}, device, amp=amp)
        evaluated: set[int] = set()

        @trainer.on(
            Events.ITERATION_COMPLETED(every=eval_interval or max_iters)
            | Events.COMPLETED
        )
        def evaluate(engine: Engine) -> None:
            nonlocal paused
            if engine.state.iteration in evaluated:
                return  # the last iteration is also a multiple of eval_interval
            evaluated.add(engine.state.iteration)
            begin = time.perf_counter()
            record = {"iteration": engine.state.iteration, "time": elapsed()}
            scores = evaluator.run(val_loader).metrics
            history.append(record | scores)
            if progress:
                summary = ", ".join(
                    f"{name} {value:.4g}" for name, value in scores.items()
                )
                bar.log_message(f"Iteration {engine.state.iteration}: {summary}")
            paused += time.perf_counter() - begin

    trainer.run(train_loader, max_epochs=1, epoch_length=max_iters)
    return history
