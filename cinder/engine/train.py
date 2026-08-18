"""Training entry point — run with ``python -m cinder.engine.train`` from project root."""

import math
import os
from typing import cast

import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LRScheduler as PyTorchLRScheduler
from ignite.handlers import LRScheduler as IgniteLRScheduler
from ignite.handlers import create_lr_scheduler_with_warmup
from ignite.utils import manual_seed
try:  # ignite >= 0.5
    from ignite.handlers.base_logger import BaseLogger
except ModuleNotFoundError:  # ignite < 0.5
    from ignite.contrib.handlers.base_logger import BaseLogger
import ignite.distributed as idist
from ignite.engine import Events, Engine

from .engine import create_trainer, create_evaluator


def train(local_rank, cfg) -> None:
    import cinder.handlers.proglogger  # noqa: F401  (silences noisy loggers early)

    device = idist.device()
    rank = idist.get_rank()
    manual_seed(cfg.seed + rank)

    # Model. sync_bn matters whenever the encoder carries BatchNorm and the run
    # is distributed: without it each rank normalises over its own shard, so a
    # 2-GPU result is not comparable to a 1-GPU one.
    # Guarded on world size: SyncBatchNorm requires an initialised process group
    # and raises on a single-GPU run, where it would be a no-op anyway.
    sync_bn = bool(cfg.trainer.sync_bn) and idist.get_world_size() > 1
    model = hydra.utils.instantiate(cfg.model)
    model = idist.auto_model(
        model,
        sync_bn=sync_bn,
        find_unused_parameters=cfg.trainer.find_unused_parameters,
    )
    raw_model = getattr(model, "module", model)

    # Data
    train_loader = hydra.utils.instantiate(cfg.dataset.train_loader)
    val_loader = hydra.utils.instantiate(cfg.dataset.val_loader)

    # Training is counted in optimizer steps. Ignite's engine is epoch-native,
    # so one engine "epoch" is defined as one evaluation interval: EPOCH_COMPLETED
    # then means "an evaluation is due", and evaluation, checkpointing and
    # progress all hang off it without any of them needing to count iterations
    # themselves. The dataloader is cycled as needed to fill the interval.
    max_iters = int(cfg.trainer.max_iters)
    epoch_length = min(int(cfg.trainer.eval_interval), max_iters)
    max_epochs, remainder = divmod(max_iters, epoch_length)
    if remainder:
        # The engine only runs whole intervals, so a remainder would silently
        # train past max_iters and carry the LR schedule beyond its horizon.
        raise ValueError(
            f"trainer.eval_interval ({epoch_length}) must divide "
            f"trainer.max_iters ({max_iters}); it currently leaves {remainder} "
            f"iterations over, which would run {max_epochs + 1} full intervals "
            f"({(max_epochs + 1) * epoch_length} iterations)."
        )
    iters_per_pass = max(len(train_loader), 1)  # one pass over the training set

    # Optimizer, scheduler, loss
    optimizer = hydra.utils.instantiate(cfg.trainer.optimizer)(params=model.parameters())
    optimizer = idist.auto_optim(optimizer)

    lr_scheduler = hydra.utils.instantiate(cfg.trainer.lr_scheduler)(optimizer=optimizer)

    base_loss = hydra.utils.instantiate(cfg.trainer.loss).to(device)
    loss_fn = cinder.SampledLoss(base_loss, raw_model)

    # Engines
    trainer = create_trainer(model, optimizer, loss_fn, device, amp=cfg.trainer.amp)

    metrics = {k: hydra.utils.instantiate(v) for k, v in cfg.metric.items()}
    train_evaluator = create_evaluator(model, metrics, device)
    inferer = hydra.utils.instantiate(cfg.inferer)
    val_evaluator = create_evaluator(model, metrics, device, inferer=inferer)

    # LR schedule, stepped per iteration, over the same horizon as the budget:
    # lr_scheduler.T_max is ${trainer.max_iters}.
    if lr_scheduler is not None:
        # >1, not >0: ignite requires at least two warmup events, and a
        # single-iteration ramp is a no-op anyway. Short debug budgets reach
        # this, since warmup is derived as a fraction of max_iters.
        warmup_iters = int(cfg.trainer.warmup_iters or 0)
        if warmup_iters > 1:
            scheduler = create_lr_scheduler_with_warmup(
                lr_scheduler,
                warmup_start_value=float(cfg.trainer.warmup_start_lr),
                warmup_duration=warmup_iters,
            )
        elif isinstance(lr_scheduler, PyTorchLRScheduler):
            scheduler = IgniteLRScheduler(cast(PyTorchLRScheduler, lr_scheduler))
        else:
            scheduler = lr_scheduler
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Reshuffle every time the loader is exhausted, not once per engine epoch:
    # an epoch here is an evaluation interval, which may span the training set
    # many times over, and reseeding only at its boundary would repeat one
    # ordering throughout.
    if isinstance(train_loader.sampler, DistributedSampler):

        @trainer.on(Events.ITERATION_COMPLETED(every=iters_per_pass))
        def reshuffle(engine: Engine) -> None:
            train_loader.sampler.set_epoch(engine.state.iteration // iters_per_pass)

    # Instantiate handlers (before eval so proglogger logs Train before Eval)
    objects = {
        "model": model,
        "trainer": trainer,
        "train_evaluator": train_evaluator,
        "val_evaluator": val_evaluator,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "experiment": cfg.experiment,
        "seed": cfg.seed,
        "output_dir": cfg.path.output_dir,
        "max_iters": max_iters,
        "iters_per_pass": iters_per_pass,
    }
    handlers = {k: hydra.utils.instantiate(v)(objects=objects) for k, v in cfg.handler.items()}

    # Evaluation
    # Scoring the training set is diagnostic, not part of the result, and it is
    # charged at the size of the training set: on ADE20K a full pass is 20210 of
    # the 22210 images an evaluation touches. eval_train_ratio sub-samples it
    # (0 disables it, 1.0 restores the full pass); the loader is shuffled, so a
    # fraction is a fresh random subset each time.
    eval_train_ratio = float(cfg.trainer.eval_train_ratio)
    eval_train_iters = (
        max(1, round(len(train_loader) * min(eval_train_ratio, 1.0)))
        if eval_train_ratio > 0
        else 0
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_eval(engine):
        torch.cuda.empty_cache()
        if eval_train_iters:
            train_evaluator.run(train_loader, epoch_length=eval_train_iters)
        val_evaluator.run(val_loader)

    # Train
    trainer.run(train_loader, max_epochs=max_epochs, epoch_length=epoch_length)

    # Cleanup loggers
    if rank == 0:
        for handler in handlers.values():
            if isinstance(handler, BaseLogger):
                handler.close()


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    backend = "nccl" if "RANK" in os.environ else None
    with idist.Parallel(backend=backend) as parallel:
        parallel.run(train, cfg)


if __name__ == "__main__":
    main()
