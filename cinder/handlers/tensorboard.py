"""TensorBoard curves and segmentation previews for training runs."""

from pathlib import Path
from typing import Any

import ignite.distributed as idist
import torch
from ignite.engine import Engine, Events
from ignite.handlers import TensorboardLogger, global_step_from_engine
from torch import Tensor

__all__ = ["setup_tensorboard_logging"]

# Class colors for multi-class previews; classes past the end share the last color.
# Binary masks are drawn in grayscale.
_PALETTE = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.90, 0.10, 0.10],
        [0.10, 0.60, 0.10],
        [0.10, 0.30, 0.90],
        [0.90, 0.80, 0.10],
        [0.80, 0.10, 0.80],
        [0.10, 0.80, 0.80],
        [0.90, 0.50, 0.10],
        [0.50, 0.20, 0.70],
        [0.40, 0.40, 0.40],
        [0.60, 0.90, 0.30],
    ]
)


def _prediction_labels(prediction: Tensor) -> Tensor:
    """Convert ``(C, H, W)`` logits to ``(H, W)`` integer labels."""
    if prediction.shape[0] == 1:
        return (prediction[0].sigmoid() > 0.5).long()
    return prediction.argmax(dim=0)


def _target_labels(target: Tensor) -> Tensor:
    """Convert binary masks or class indices to ``(H, W)`` integer labels."""
    if target.ndim == 3:
        return (target[0] > 0.5).long()
    return target.long()


def _colorize(labels: Tensor, num_classes: int) -> Tensor:
    """Map ``(H, W)`` integer labels to ``(3, H, W)`` RGB in [0, 1]."""
    if num_classes <= 1:
        mask = labels.float().clamp(0.0, 1.0)
        return mask.unsqueeze(0).expand(3, *mask.shape)
    palette = _PALETTE.to(labels.device)
    indices = labels.clamp(0, palette.shape[0] - 1)
    return palette[indices].permute(2, 0, 1)


def _make_panel(
    image: Tensor,
    target: Tensor,
    prediction: Tensor,
    num_classes: int,
    separator_width: int = 4,
) -> Tensor:
    """Build an RGB panel showing the input, target, and prediction."""
    image = image.float().clamp(0.0, 1.0)
    if image.shape[0] == 1:
        image = image.expand(3, *image.shape[1:])
    target_rgb = _colorize(_target_labels(target), num_classes)
    prediction_rgb = _colorize(_prediction_labels(prediction), num_classes)
    separator = torch.ones(3, image.shape[1], separator_width)
    return torch.cat([image, separator, target_rgb, separator, prediction_rgb], dim=2)


def _attach_image_logging(
    tb_logger: TensorboardLogger, trainer: Engine, evaluator: Engine, num_examples: int
) -> None:
    """Log the first validation cases against the training iteration."""
    panels: list[Tensor] = []

    @evaluator.on(Events.EPOCH_STARTED)
    def _reset(_engine: Engine) -> None:
        panels.clear()

    @evaluator.on(Events.ITERATION_COMPLETED)
    def _capture(engine: Engine) -> None:
        if len(panels) >= num_examples:
            return
        inputs, targets = engine.state.batch
        predictions = engine.state.output[0]
        num_classes = predictions.shape[1]
        remaining = min(num_examples - len(panels), inputs.shape[0])
        for index in range(remaining):
            panels.append(
                _make_panel(
                    inputs[index],
                    targets[index],
                    predictions[index].detach().cpu(),
                    num_classes,
                )
            )

    @evaluator.on(Events.EPOCH_COMPLETED)
    def _write(_engine: Engine) -> None:
        step = trainer.state.iteration
        for index, panel in enumerate(panels):
            tb_logger.writer.add_image(f"val/example_{index}", panel, global_step=step)


def setup_tensorboard_logging(
    objects: dict[str, Any],
    log_dir: str | Path = "./",
    num_examples: int = 2,
    log_interval: int = 50,
    **kwargs: Any,
) -> None:
    """Log the training loss, learning rate, metrics and validation previews.

    Every curve is stepped by training iteration. The loss and learning rate are
    single-iteration values, sampled every ``log_interval`` iterations. The
    logger closes when training completes.

    Args:
        objects: Run objects from the training entry point: ``trainer``,
            ``optimizer``, ``train_evaluator`` and ``val_evaluator``.
        log_dir: TensorBoard log directory.
        num_examples: Validation cases logged as ``input | target | prediction``
            panels after each evaluation; ``0`` disables them. The validation
            loader is not shuffled, so the same cases appear every time.
        log_interval: Iterations between loss and learning-rate points.
        **kwargs: Passed to :class:`~ignite.handlers.TensorboardLogger`.
    """
    if idist.get_rank() != 0:
        return

    trainer = objects["trainer"]
    optimizer = objects["optimizer"]
    evaluators = {
        "train": objects["train_evaluator"],
        "val": objects["val_evaluator"],
    }

    tb_logger = TensorboardLogger(log_dir=log_dir, **kwargs)

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_interval),
        tag="train",
        output_transform=lambda loss: {"loss": loss},
    )
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_interval),
        optimizer=optimizer,
    )
    for tag, evaluator in evaluators.items():
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(
                trainer, Events.ITERATION_COMPLETED
            ),
        )

    if num_examples > 0:
        _attach_image_logging(
            tb_logger, trainer, objects["val_evaluator"], num_examples
        )

    trainer.add_event_handler(Events.COMPLETED, lambda _: tb_logger.close())
