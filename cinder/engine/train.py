"""Training entry point — run with ``python -m cinder.engine.train`` from project root."""

import os
from typing import cast

import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LRScheduler as PyTorchLRScheduler
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

    # Model
    model = hydra.utils.instantiate(cfg.model)
    model = idist.auto_model(model, find_unused_parameters=cfg.trainer.find_unused_parameters)
    raw_model = getattr(model, "module", model)

    # Data
    train_loader = hydra.utils.instantiate(cfg.dataset.train_loader)
    val_loader = hydra.utils.instantiate(cfg.dataset.val_loader)

    # Optimizer, scheduler, loss
    optimizer = hydra.utils.instantiate(cfg.trainer.optimizer)(params=model.parameters())
    optimizer = idist.auto_optim(optimizer)

    lr_scheduler = hydra.utils.instantiate(cfg.trainer.lr_scheduler)(optimizer=optimizer)

    base_loss = hydra.utils.instantiate(cfg.trainer.loss).to(device)
    loss_fn = cinder.SampledLoss(base_loss, raw_model)

    # Engines
    trainer = create_trainer(model, optimizer, loss_fn, device)

    metrics = {k: hydra.utils.instantiate(v) for k, v in cfg.metric.items()}
    train_evaluator = create_evaluator(model, metrics, device)
    inferer = hydra.utils.instantiate(cfg.inferer)
    val_evaluator = create_evaluator(model, metrics, device, inferer=inferer)

    # LR scheduler step
    if lr_scheduler is not None:
        if isinstance(lr_scheduler, PyTorchLRScheduler):
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED,
                lambda engine: cast(PyTorchLRScheduler, lr_scheduler).step(),
            )
        else:
            trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)

    # Distributed sampler epoch sync
    if isinstance(train_loader.sampler, DistributedSampler):

        @trainer.on(Events.EPOCH_STARTED)
        def set_epoch(engine: Engine) -> None:
            train_loader.sampler.set_epoch(engine.state.epoch - 1)

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
    }
    handlers = {k: hydra.utils.instantiate(v)(objects=objects) for k, v in cfg.handler.items()}

    # Evaluation
    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.trainer.eval_every_epochs))
    def run_eval(engine):
        torch.cuda.empty_cache()
        train_evaluator.run(train_loader)
        val_evaluator.run(val_loader)

    # Train
    trainer.run(train_loader, max_epochs=cfg.trainer.max_epochs)

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
