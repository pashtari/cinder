import torch
from torch import Tensor, nn
import torch.nn.functional as F


def make_sampled_target_fn(model: nn.Module):
    """Create a function that subsamples targets to match model's sampled predictions."""

    def fn(target: Tensor) -> Tensor:
        if model._sample_indices is not None:
            if target.ndim == 2:  # (H, W) multi-class per-sample after decollate
                return target.flatten(0)[model._sample_indices]
            elif target.ndim == 3:
                if target.shape[0] > 1 and target.dtype == torch.long:
                    # (B, H, W) multi-class batched
                    return target.flatten(1)[:, model._sample_indices]
                else:
                    # (C, H, W) binary per-sample after decollate
                    return target.flatten(1)[:, model._sample_indices]
            else:  # (B, C, H, W) binary / multi-label batched
                return target.flatten(2)[:, :, model._sample_indices]
        return target

    return fn


class SampledLoss:
    """Loss wrapper that handles coordinate-sampled outputs from CINDER.

    During training with sampling_ratio < 1.0, CINDER outputs (B, C, N_sampled)
    instead of (B, C, H, W). This wrapper subsamples the target to match.
    """

    def __init__(self, loss_fn: nn.Module, model: nn.Module):
        self.loss_fn = loss_fn
        self.model = model

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.model._sample_indices is not None:
            if target.ndim == 3:  # (B, H, W) multi-class
                target = target.flatten(1)[:, self.model._sample_indices]
            else:  # (B, C, H, W) binary / multi-label
                target = target.flatten(2)[:, :, self.model._sample_indices]
        return self.loss_fn(pred, target)


class DiceLoss(nn.Module):
    """Soft Dice loss for binary or multi-class segmentation.

    Args:
        sigmoid: Apply sigmoid to predictions (binary mode).
        softmax: Apply softmax to predictions (multi-class mode).
            When ``True``, *sigmoid* is ignored and the target is expected
            to be a ``(B, ...)`` long tensor of class indices.
        squared_pred: Square the denominator terms (pred² + target²) instead
            of (pred + target), which can stabilise training.
        smooth: Small constant added to numerator and denominator to avoid
            division by zero.
        ignore_index: Class index to ignore (multi-class only). Pixels with
            this label are excluded from the Dice computation.
    """

    def __init__(
        self,
        sigmoid: bool = True,
        softmax: bool = False,
        squared_pred: bool = True,
        smooth: float = 1e-5,
        ignore_index: int = -100,
    ):
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
            mask = target != self.ignore_index
            safe_target = target.clone()
            safe_target[~mask] = 0
            target = F.one_hot(safe_target, num_classes).movedim(-1, 1).float()
            # Zero out ignored pixels in both pred and target
            mask = mask.unsqueeze(1).float()  # (B, 1, ...)
            pred = pred * mask
            target = target * mask
        elif self.sigmoid:
            pred = torch.sigmoid(pred)

        # Flatten spatial dims: (B, C, *) -> (B, C, N)
        pred = pred.flatten(2)
        target = target.flatten(2)

        intersection = (pred * target).sum(dim=-1)

        if self.squared_pred:
            denom = (pred**2).sum(dim=-1) + (target**2).sum(dim=-1)
        else:
            denom = pred.sum(dim=-1) + target.sum(dim=-1)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss for binary or multi-class segmentation.

    Args:
        sigmoid: Apply sigmoid to predictions in the Dice term (binary).
        softmax: Apply softmax to predictions in the Dice term (multi-class).
            When ``True``, CE uses ``F.cross_entropy`` instead of BCE.
        squared_pred: Use squared denominator in the Dice term.
        lambda_dice: Weight for the Dice loss component.
        lambda_ce: Weight for the CE loss component.
        ignore_index: Class index to ignore in both CE and Dice (multi-class).
    """

    def __init__(
        self,
        sigmoid: bool = True,
        softmax: bool = False,
        squared_pred: bool = True,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.dice = DiceLoss(
            sigmoid=sigmoid,
            softmax=softmax,
            squared_pred=squared_pred,
            ignore_index=ignore_index,
        )
        self.softmax = softmax
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.ignore_index = ignore_index

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        dice_loss = self.lambda_dice * self.dice(pred, target)
        if self.softmax:
            ce_loss = self.lambda_ce * F.cross_entropy(
                pred,
                target,
                ignore_index=self.ignore_index,
            )
        else:
            ce_loss = self.lambda_ce * F.binary_cross_entropy_with_logits(pred, target)
        return dice_loss + ce_loss
