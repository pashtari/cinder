"""Evaluation entry point — run with ``python -m cinder.engine.eval`` from project root."""

import os

import hydra
from omegaconf import DictConfig
from ignite.utils import manual_seed
import ignite.distributed as idist

from .engine import create_evaluator


def evaluate(local_rank, cfg) -> None:
    device = idist.device()
    rank = idist.get_rank()
    manual_seed(cfg.seed + rank)

    model = hydra.utils.instantiate(cfg.model)
    model = idist.auto_model(model)

    val_loader = hydra.utils.instantiate(cfg.dataset.val_loader)
    metrics = {k: hydra.utils.instantiate(v) for k, v in cfg.metric.items()}

    inferer = hydra.utils.instantiate(cfg.inferer)
    evaluator = create_evaluator(model, metrics, device, inferer=inferer)

    objects = {
        "evaluator": evaluator,
        "model": model,
        "experiment": cfg.experiment,
        "seed": cfg.seed,
        "output_dir": cfg.path.output_dir,
    }
    for value in cfg.handler.values():
        hydra.utils.instantiate(value)(objects=objects)

    evaluator.run(val_loader)


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    backend = "nccl" if "RANK" in os.environ else None
    with idist.Parallel(backend=backend) as parallel:
        parallel.run(evaluate, cfg)


if __name__ == "__main__":
    main()
