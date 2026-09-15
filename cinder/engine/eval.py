"""Evaluate CINDER with ``scripts/eval.sh`` or ``python -m cinder.engine.eval``."""

import os

import hydra
import ignite.distributed as idist
from ignite.utils import manual_seed
from omegaconf import DictConfig

from ..handlers.progress import quiet_library_loggers
from .engines import create_evaluator


def evaluate(local_rank: int, cfg: DictConfig) -> None:
    """Evaluate on one process; ``idist.Parallel`` supplies ``local_rank``."""
    quiet_library_loggers()
    device = idist.device()
    manual_seed(cfg.seed + idist.get_rank())

    model = idist.auto_model(hydra.utils.instantiate(cfg.model))
    val_loader = hydra.utils.instantiate(cfg.dataset.val_loader)
    metrics = {name: hydra.utils.instantiate(spec) for name, spec in cfg.metric.items()}
    inferer = hydra.utils.instantiate(cfg.inferer)
    evaluator = create_evaluator(
        model, metrics, device, non_blocking=cfg.trainer.non_blocking, inferer=inferer
    )

    objects = {
        "evaluator": evaluator,
        "model": model,
        "experiment": cfg.experiment,
        "seed": cfg.seed,
        "output_dir": cfg.path.output_dir,
    }
    # The checkpoint handler restores the model weights here.
    for handler in cfg.handler.values():
        hydra.utils.instantiate(handler)(objects=objects)

    evaluator.run(val_loader)


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Launch evaluation, distributed across processes when started by torchrun."""
    backend = "nccl" if "RANK" in os.environ else None
    with idist.Parallel(backend=backend) as parallel:
        parallel.run(evaluate, cfg)


if __name__ == "__main__":
    main()
