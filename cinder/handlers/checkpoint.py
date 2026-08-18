from pathlib import Path

import torch
import ignite.distributed as idist
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine


def load_checkpoint(load_from):
    ckpt_path = Path(load_from)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path.as_posix(), map_location="cpu")
    return ckpt


def checkpoint(
    objects,
    load_from=None,
    score_metric=None,
    score_mode="max",
    load_checkpoint_kwargs=None,
    **kwargs,
):
    """Checkpoint handler.

    Saves once per evaluation interval (the engine's epoch, see
    ``cinder/engine/train.py``). With *score_metric* unset the latest checkpoint
    is kept; set it to a validation metric name (e.g. ``"dice"``) to keep the
    best-scoring one instead, chosen on the validation engine's metrics. Which
    of the two a paper reports has to be the same for every method compared, so
    it is recorded in the config rather than left implicit.

    Checkpoints are named by iteration, since that is the unit the training
    budget is expressed in.
    """

    load_checkpoint_kwargs = (
        {} if load_checkpoint_kwargs is None else load_checkpoint_kwargs
    )

    rank = idist.get_rank()

    if "trainer" in objects:
        trainer = objects["trainer"]
        model = objects["model"]
        optimizer = objects["optimizer"]
        lr_scheduler = objects["lr_scheduler"]

        to_save = {
            "trainer": trainer,
            "model": model,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

        if rank == 0:
            if score_metric is None:
                model_checkpoint = ModelCheckpoint(
                    global_step_transform=global_step_from_engine(
                        trainer, Events.ITERATION_COMPLETED
                    ),
                    **kwargs,
                )
                trainer.add_event_handler(
                    Events.EPOCH_COMPLETED | Events.COMPLETED,
                    model_checkpoint,
                    to_save,
                )
            else:
                # Fired on the validation engine, whose state holds the metrics.
                sign = 1.0 if score_mode == "max" else -1.0
                model_checkpoint = ModelCheckpoint(
                    score_name=score_metric,
                    score_function=lambda engine: sign
                    * float(engine.state.metrics[score_metric]),
                    global_step_transform=global_step_from_engine(
                        trainer, Events.ITERATION_COMPLETED
                    ),
                    **kwargs,
                )
                objects["val_evaluator"].add_event_handler(
                    Events.COMPLETED, model_checkpoint, to_save
                )

        if load_from is not None:
            ckpt = load_checkpoint(load_from)
            ModelCheckpoint.load_objects(
                to_load=to_save, checkpoint=ckpt, **load_checkpoint_kwargs
            )
    elif "evaluator" in objects:
        model = objects["model"]

        if load_from is None:
            raise ValueError(
                "Evaluation requires a checkpoint; set "
                "`handler.checkpoint.load_from=/path/to/checkpoint.pt`."
            )

        to_load = {"model": model}
        ckpt = load_checkpoint(load_from)
        ModelCheckpoint.load_objects(
            to_load=to_load, checkpoint=ckpt, **load_checkpoint_kwargs
        )
