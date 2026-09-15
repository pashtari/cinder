"""Save training checkpoints and restore them with Ignite."""

from pathlib import Path
from typing import Any, Literal

import ignite.distributed as idist
import torch
from ignite.engine import Events
from ignite.handlers import Checkpoint, ModelCheckpoint, global_step_from_engine

__all__ = ["load_checkpoint", "setup_checkpointing"]


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint onto the CPU, whichever device saved it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def setup_checkpointing(
    objects: dict[str, Any],
    load_from: str | Path | None = None,
    score_metric: str | None = None,
    score_mode: Literal["max", "min"] = "max",
    **kwargs: Any,
) -> None:
    """Restore a checkpoint and, during training, save one at every evaluation.

    Training saves the trainer, model, optimizer and LR scheduler, keeping the
    latest checkpoints or, with ``score_metric``, the best-scoring ones; file
    names carry the training iteration. Evaluation restores only the model and
    requires ``load_from``.

    Args:
        objects: Run objects from the entry point: ``trainer``, ``model``,
            ``optimizer``, ``lr_scheduler`` and ``val_evaluator`` for training,
            or ``evaluator`` and ``model`` for evaluation.
        load_from: Checkpoint to restore: a run to resume, or the weights to
            evaluate.
        score_metric: Validation metric that ranks the checkpoints to keep;
            ``None`` keeps the latest.
        score_mode: Whether higher (``"max"``) or lower (``"min"``) scores are
            better.
        **kwargs: Passed to :class:`ignite.handlers.ModelCheckpoint`, such as
            ``dirname`` and ``n_saved``.
    """
    if score_mode not in ("max", "min"):
        raise ValueError(f"score_mode must be 'max' or 'min', got {score_mode!r}")

    if "trainer" in objects:
        trainer = objects["trainer"]
        # A run without an LR scheduler has no scheduler state to save.
        checkpoint_objects = {
            name: objects[name]
            for name in ("trainer", "model", "optimizer", "lr_scheduler")
            if objects[name] is not None
        }
        if idist.get_rank() == 0:
            global_step = global_step_from_engine(trainer, Events.ITERATION_COMPLETED)
            if score_metric is None:
                saver = ModelCheckpoint(global_step_transform=global_step, **kwargs)
                trainer.add_event_handler(
                    Events.EPOCH_COMPLETED, saver, checkpoint_objects
                )
            else:
                # Scores live in the validation evaluator's state, so save from there.
                sign = 1.0 if score_mode == "max" else -1.0
                saver = ModelCheckpoint(
                    score_name=score_metric,
                    score_function=Checkpoint.get_default_score_fn(score_metric, sign),
                    global_step_transform=global_step,
                    **kwargs,
                )
                objects["val_evaluator"].add_event_handler(
                    Events.COMPLETED, saver, checkpoint_objects
                )
    else:
        if load_from is None:
            raise ValueError(
                "Evaluation requires a checkpoint; set "
                "`handler.checkpoint.load_from=/path/to/checkpoint.pt`."
            )
        checkpoint_objects = {"model": objects["model"]}

    if load_from is not None:
        Checkpoint.load_objects(
            to_load=checkpoint_objects, checkpoint=load_checkpoint(load_from)
        )
