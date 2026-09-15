"""Train CINDER with ``scripts/train.sh`` or ``python -m cinder.engine.train``."""

import os

import hydra
import ignite.distributed as idist
import torch
from ignite.engine import Engine, Events
from ignite.handlers import LRScheduler, create_lr_scheduler_with_warmup
from ignite.utils import manual_seed
from omegaconf import DictConfig
from torch.utils.data.distributed import DistributedSampler

from ..handlers.progress import quiet_library_loggers
from .engines import create_evaluator, create_trainer
from .losses import SampledLoss


def train(local_rank: int, cfg: DictConfig) -> None:
    """Train on one process; ``idist.Parallel`` supplies ``local_rank``."""
    quiet_library_loggers()
    device = idist.device()
    rank = idist.get_rank()
    manual_seed(cfg.seed + rank)

    # auto_model applies SyncBatchNorm only when there is more than one process.
    model = idist.auto_model(
        hydra.utils.instantiate(cfg.model),
        sync_bn=cfg.trainer.sync_bn,
        find_unused_parameters=cfg.trainer.find_unused_parameters,
    )
    unwrapped_model = getattr(model, "module", model)

    train_loader = hydra.utils.instantiate(cfg.dataset.train_loader)
    val_loader = hydra.utils.instantiate(cfg.dataset.val_loader)

    # One Ignite epoch is one evaluation interval; the loader cycles as needed.
    max_iters = int(cfg.trainer.max_iters)
    eval_interval = min(int(cfg.trainer.eval_interval), max_iters)
    num_intervals, remainder = divmod(max_iters, eval_interval)
    if remainder:
        # A partial interval would run past max_iters and the LR schedule.
        raise ValueError(
            f"trainer.eval_interval ({eval_interval}) must divide "
            f"trainer.max_iters ({max_iters}); the remainder is {remainder}"
        )
    iters_per_pass = max(len(train_loader), 1)

    optimizer = hydra.utils.instantiate(cfg.trainer.optimizer)(
        params=model.parameters()
    )
    optimizer = idist.auto_optim(optimizer)
    lr_scheduler = None
    if cfg.trainer.lr_scheduler is not None:
        lr_scheduler = hydra.utils.instantiate(cfg.trainer.lr_scheduler)(
            optimizer=optimizer
        )

    loss_fn = SampledLoss(
        hydra.utils.instantiate(cfg.trainer.loss).to(device), unwrapped_model
    )
    trainer = create_trainer(
        model,
        optimizer,
        loss_fn,
        device,
        non_blocking=cfg.trainer.non_blocking,
        amp=cfg.trainer.amp,
    )

    def instantiate_metrics() -> dict:
        return {
            name: hydra.utils.instantiate(spec) for name, spec in cfg.metric.items()
        }

    train_evaluator = create_evaluator(
        model, instantiate_metrics(), device, non_blocking=cfg.trainer.non_blocking
    )
    inferer = hydra.utils.instantiate(cfg.inferer)
    val_evaluator = create_evaluator(
        model,
        instantiate_metrics(),
        device,
        non_blocking=cfg.trainer.non_blocking,
        inferer=inferer,
    )

    if lr_scheduler is not None:
        warmup_iters = int(cfg.trainer.warmup_iters or 0)
        # Ignite's warmup needs at least two steps; shorter warmups are skipped.
        if warmup_iters > 1:
            scheduler = create_lr_scheduler_with_warmup(
                lr_scheduler,
                warmup_start_value=float(cfg.trainer.warmup_start_lr),
                warmup_duration=warmup_iters,
            )
        else:
            scheduler = LRScheduler(lr_scheduler)
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # An evaluation interval can span several passes, and each needs a new order.
    if isinstance(train_loader.sampler, DistributedSampler):

        @trainer.on(Events.ITERATION_COMPLETED(every=iters_per_pass))
        def reshuffle(engine: Engine) -> None:
            train_loader.sampler.set_epoch(engine.state.iteration // iters_per_pass)

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
    # Handlers attach before the evaluation hook below, so each interval's
    # checkpoint is saved and its training timer paused before evaluation.
    for handler in cfg.handler.values():
        hydra.utils.instantiate(handler)(objects=objects)

    # Scoring a random subset of the training set is a cheaper diagnostic.
    eval_train_ratio = min(float(cfg.trainer.eval_train_ratio), 1.0)
    eval_train_iters = (
        max(1, round(len(train_loader) * eval_train_ratio))
        if eval_train_ratio > 0
        else 0
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_evaluation(_: Engine) -> None:
        torch.cuda.empty_cache()
        if eval_train_iters:
            train_evaluator.run(train_loader, epoch_length=eval_train_iters)
        val_evaluator.run(val_loader)

    trainer.run(train_loader, max_epochs=num_intervals, epoch_length=eval_interval)


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Launch training, distributed across processes when started by torchrun."""
    backend = "nccl" if "RANK" in os.environ else None
    with idist.Parallel(backend=backend) as parallel:
        parallel.run(train, cfg)


if __name__ == "__main__":
    main()
