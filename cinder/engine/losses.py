"""Segmentation losses for dense and coordinate-sampled predictions."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["SampledLoss", "DiceLoss", "MaskedCrossEntropyLoss", "DiceCELoss"]


class SampledLoss:
    """Apply a loss to the pixels CINDER decoded in its last training forward.

    With ``sampling_ratio < 1``, CINDER decodes a random subset of pixels, shared
    across the batch, and stores their flat indices in ``model.sample_indices``.
    This wrapper gathers the same pixels from the target, so call it right after
    the forward pass. Pass the unwrapped model: DDP is fine, but ``DataParallel``
    replicas would each record their own indices. Dense predictions pass through.

    Args:
        loss_fn: ``(predictions, targets) -> loss``.
        model: The model whose ``sample_indices`` to read.

    Shape:
        - pred: ``(B, C, N)`` for ``N`` sampled pixels, or dense ``(B, C, H, W)``.
        - target: class indices ``(B, H, W)`` or masks ``(B, C, H, W)``.
    """

    def __init__(
        self, loss_fn: Callable[[Tensor, Tensor], Tensor], model: nn.Module
    ) -> None:
        self.loss_fn = loss_fn
        self.model = model

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        indices = self.model.sample_indices
        if indices is not None:
            # Class-index targets have no channel axis.
            if target.ndim == 3:
                target = target.flatten(1)[:, indices]
            else:
                target = target.flatten(2)[:, :, indices]
        return self.loss_fn(pred, target)


class DiceLoss(nn.Module):
    """Soft Dice loss, averaged over images and classes.

    Classes absent from an image still count in the average; ``smooth`` keeps
    their Dice defined.

    Args:
        sigmoid: Apply a sigmoid to the predictions (binary masks).
        softmax: Apply a softmax to the predictions and read targets as
            ``(B, ...)`` class indices. Takes precedence over ``sigmoid``.
        squared_pred: Use ``pred**2 + target**2`` in the denominator instead of
            ``pred + target``.
        smooth: Added to the numerator and denominator.
        ignore_index: Target label excluded from the loss; multi-class only.
    """

    def __init__(
        self,
        sigmoid: bool = True,
        softmax: bool = False,
        squared_pred: bool = True,
        smooth: float = 1e-5,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.squared_pred = squared_pred
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.softmax:
            pred = torch.softmax(pred, dim=1)
            num_classes = pred.shape[1]
            valid = target != self.ignore_index
            safe_target = target.masked_fill(~valid, 0)
            target = F.one_hot(safe_target, num_classes).movedim(-1, 1).float()
            # Ignored pixels must contribute to neither overlap nor area.
            valid = valid.unsqueeze(1).float()
            pred = pred * valid
            target = target * valid
        elif self.sigmoid:
            pred = torch.sigmoid(pred)

        pred = pred.flatten(2)
        target = target.flatten(2)

        intersection = (pred * target).sum(dim=-1)

        if self.squared_pred:
            denominator = (pred**2).sum(dim=-1) + (target**2).sum(dim=-1)
        else:
            denominator = pred.sum(dim=-1) + target.sum(dim=-1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class MaskedCrossEntropyLoss(nn.Module):
    """Mean cross-entropy over the pixels whose label is not ``ignore_index``.

    A fully ignored target gives zero loss and gradients instead of NaN, so
    unlabeled crops or coordinate samples cannot break training.
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        total = F.cross_entropy(
            pred, target, ignore_index=self.ignore_index, reduction="sum"
        )
        count = (target != self.ignore_index).sum().clamp_min(1)
        return total / count


class DiceCELoss(nn.Module):
    """Weighted sum of :class:`DiceLoss` and cross-entropy.

    Binary masks use :class:`~torch.nn.BCEWithLogitsLoss`; with ``softmax=True``,
    class indices use :class:`MaskedCrossEntropyLoss`.

    Args:
        sigmoid: Apply a sigmoid in the Dice term (binary masks).
        softmax: Apply a softmax in the Dice term and use class-index targets.
        squared_pred: Use a squared denominator in the Dice term.
        smooth: Smoothing constant of the Dice term.
        lambda_dice: Weight of the Dice term.
        lambda_ce: Weight of the cross-entropy term.
        ignore_index: Target label excluded from both terms; multi-class only.
    """

    def __init__(
        self,
        sigmoid: bool = True,
        softmax: bool = False,
        squared_pred: bool = True,
        smooth: float = 1e-5,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(
            sigmoid=sigmoid,
            softmax=softmax,
            squared_pred=squared_pred,
            smooth=smooth,
            ignore_index=ignore_index,
        )
        self.ce = (
            MaskedCrossEntropyLoss(ignore_index) if softmax else nn.BCEWithLogitsLoss()
        )
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        dice_loss = self.lambda_dice * self.dice(pred, target)
        ce_loss = self.lambda_ce * self.ce(pred, target)
        return dice_loss + ce_loss
