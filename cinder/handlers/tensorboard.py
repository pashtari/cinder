import ignite.distributed as idist
import torch
from ignite.contrib.engines import common
from ignite.engine import Events

# Qualitative palette (RGB in [0, 1]); index 0 is background (black), 1 is white.
# Used to colorize multi-class label maps; binary masks use index 0/1 only.
_PALETTE = torch.tensor(
    [
        [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.90, 0.10, 0.10], [0.10, 0.60, 0.10],
        [0.10, 0.30, 0.90], [0.90, 0.80, 0.10], [0.80, 0.10, 0.80], [0.10, 0.80, 0.80],
        [0.90, 0.50, 0.10], [0.50, 0.20, 0.70], [0.40, 0.40, 0.40], [0.60, 0.90, 0.30],
    ]
)


def _pred_to_label(pred: torch.Tensor) -> torch.Tensor:
    """``(C, H, W)`` logits -> ``(H, W)`` integer label map."""
    if pred.shape[0] == 1:  # binary: sigmoid + threshold
        return (pred[0].sigmoid() > 0.5).long()
    return pred.argmax(dim=0)  # multi-class: argmax


def _target_to_label(target: torch.Tensor) -> torch.Tensor:
    """``(1, H, W)`` binary mask or ``(H, W)`` class indices -> ``(H, W)`` labels."""
    if target.ndim == 3:
        return (target[0] > 0.5).long()
    return target.long()


def _colorize(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``(H, W)`` integer labels -> ``(3, H, W)`` RGB in [0, 1]."""
    if num_classes <= 1:  # binary -> white foreground on black
        m = label.float().clamp(0.0, 1.0)
        return m.unsqueeze(0).expand(3, *m.shape)
    palette = _PALETTE.to(label.device)
    idx = label.clamp(0, palette.shape[0] - 1)
    return palette[idx].permute(2, 0, 1)


def _panel(
    inp: torch.Tensor,
    tgt: torch.Tensor,
    prd: torch.Tensor,
    num_classes: int,
    sep: int = 4,
) -> torch.Tensor:
    """Build an ``input | ground-truth | prediction`` RGB panel, ``(3, H, 3W + 2sep)``."""
    inp = inp.float().clamp(0.0, 1.0)
    if inp.shape[0] == 1:
        inp = inp.expand(3, *inp.shape[1:])
    gt = _colorize(_target_to_label(tgt), num_classes)
    pr = _colorize(_pred_to_label(prd), num_classes)
    bar = torch.ones(3, inp.shape[1], sep)
    return torch.cat([inp, bar, gt, bar, pr], dim=2)


def _attach_image_logging(tb_logger, trainer, evaluator, num_examples: int) -> None:
    """Log ``input | GT | prediction`` panels for the first ``num_examples``
    validation cases to TensorBoard after every evaluation (rank 0 only).

    Captures the (fixed, since the val loader is unshuffled) first few cases of
    each evaluation run and writes them under per-example tags stepped by the
    training epoch, so the TensorBoard image slider shows each case evolving.
    """
    buffer: list[torch.Tensor] = []

    @evaluator.on(Events.EPOCH_STARTED)
    def _reset(_engine) -> None:
        buffer.clear()

    @evaluator.on(Events.ITERATION_COMPLETED)
    def _capture(engine) -> None:
        if len(buffer) >= num_examples:
            return
        inputs, targets = engine.state.batch  # raw (CPU) batch
        preds = engine.state.output[0]  # logits (on device)
        num_classes = preds.shape[1]
        take = min(num_examples - len(buffer), inputs.shape[0])
        for b in range(take):
            buffer.append(
                _panel(inputs[b], targets[b], preds[b].detach().cpu(), num_classes)
            )

    @evaluator.on(Events.EPOCH_COMPLETED)
    def _write(_engine) -> None:
        step = trainer.state.epoch
        for i, panel in enumerate(buffer):
            tb_logger.writer.add_image(f"val/example_{i}", panel, global_step=step)


def tensorboard(
    objects,
    output_path="./",
    log_images=True,
    num_examples=2,
    **kwargs,
):
    rank = idist.get_rank()

    if rank != 0:
        return None

    trainer = objects["trainer"]
    optimizer = objects["optimizer"]
    evaluators = {
        "train": objects["train_evaluator"],
        "val": objects["val_evaluator"],
    }

    tb_logger = common.setup_tb_logging(
        output_path,
        trainer=trainer,
        optimizers=optimizer,
        evaluators=evaluators,
        **kwargs,
    )

    # Log a few qualitative val examples (input / ground truth / prediction)
    # after each evaluation.
    if log_images and num_examples > 0 and "val_evaluator" in objects:
        _attach_image_logging(
            tb_logger, trainer, objects["val_evaluator"], int(num_examples)
        )

    return tb_logger
