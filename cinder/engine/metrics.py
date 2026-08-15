import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class DiceMetric(Metric):
    """Computes mean Dice score for binary or multi-class segmentation.

    Expects evaluator output ``(preds, targets)`` where ``preds`` are raw logits.

    Args:
        threshold: Binarisation threshold (binary mode only).
        sigmoid: Apply sigmoid before thresholding (binary mode).
        num_classes: Number of segmentation classes.
            ``1`` → binary mode.
            ``>1`` → multi-class mode (argmax, per-class Dice, averaged).
        ignore_index: Class index to exclude from the computation
            (multi-class only, e.g. ``0`` for ADE20K unlabeled pixels).
    """

    def __init__(
        self,
        threshold=0.5,
        sigmoid=True,
        num_classes=1,
        ignore_index=-100,
        output_transform=lambda x: x,
        device="cpu",
    ):
        self.threshold = threshold
        self.sigmoid = sigmoid
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        super().__init__(output_transform=output_transform, device=device)

    def _score_fn(self, intersection, pred_sum, target_sum):
        """Dice = 2·TP / (pred + target)."""
        return (2.0 * intersection + 1e-8) / (pred_sum + target_sum + 1e-8)

    @reinit__is_reduced
    def reset(self):
        self._sum_score = torch.tensor(0.0, device=self._device)
        self._num_samples = torch.tensor(0, device=self._device)
        self._intersection = torch.zeros(max(self.num_classes, 1), device=self._device)
        self._pred_sum = torch.zeros(max(self.num_classes, 1), device=self._device)
        self._target_sum = torch.zeros(max(self.num_classes, 1), device=self._device)

    @reinit__is_reduced
    def update(self, output):
        preds, targets = output

        if self.num_classes > 1:
            preds = preds.argmax(dim=1).flatten()
            targets = targets.flatten()
            valid = targets != self.ignore_index
            preds = preds[valid]
            targets = targets[valid]
            for c in range(self.num_classes):
                p = preds == c
                t = targets == c
                self._intersection[c] += (p & t).sum().to(self._device)
                self._pred_sum[c] += p.sum().to(self._device)
                self._target_sum[c] += t.sum().to(self._device)
        else:
            if self.sigmoid:
                preds = torch.sigmoid(preds)
            preds = (preds > self.threshold).float()

            preds_flat = preds.flatten(1)
            targets_flat = targets.flatten(1)

            intersection = (preds_flat * targets_flat).sum(dim=1)
            pred_sum = preds_flat.sum(dim=1)
            target_sum = targets_flat.sum(dim=1)

            score = self._score_fn(intersection, pred_sum, target_sum)
            self._sum_score += score.sum().to(self._device)
            self._num_samples += score.shape[0]

    @sync_all_reduce(
        "_sum_score", "_num_samples", "_intersection", "_pred_sum", "_target_sum"
    )
    def compute(self):
        if self.num_classes > 1:
            return (
                self._score_fn(self._intersection, self._pred_sum, self._target_sum)
                .mean()
                .item()
            )
        else:
            if self._num_samples == 0:
                raise NotComputableError(
                    f"{type(self).__name__} must have at least one example."
                )
            return (self._sum_score / self._num_samples).item()


class IoUMetric(DiceMetric):
    """Computes mean IoU for binary or multi-class segmentation.

    Identical to :class:`DiceMetric` except for the score formula:
    IoU = TP / (pred + target − TP).
    """

    def _score_fn(self, intersection, pred_sum, target_sum):
        union = pred_sum + target_sum - intersection
        return (intersection + 1e-8) / (union + 1e-8)


class HausdorffDistanceMetric(Metric):
    """95th-percentile Hausdorff Distance (in pixels).

    The Hausdorff distance measures boundary disagreement; the 95th percentile
    (HD95) is the robust variant standard in medical image segmentation, less
    sensitive to outliers than the maximum. Lower is better.

    Computed per image (binary mode), or per foreground class then per image
    (multi-class mode), and averaged over all accumulated samples. Expects
    evaluator output ``(preds, targets)`` where ``preds`` are raw logits.

    When exactly one of prediction / ground-truth is empty the distance is
    undefined; we assign the image diagonal as a worst-case penalty (an empty
    prediction against a non-empty mask should score poorly, not be skipped).

    Args:
        threshold: Binarisation threshold (binary mode).
        sigmoid: Apply sigmoid before thresholding (binary mode).
        num_classes: ``1`` -> binary mode; ``>1`` -> multi-class, averaging
            HD95 over foreground classes ``1..num_classes-1`` (background
            class 0 excluded).
        percentile: Percentile of surface distances to report (default 95).
        ignore_index: Class index to skip (multi-class mode).
    """

    def __init__(
        self,
        threshold=0.5,
        sigmoid=True,
        num_classes=1,
        percentile=95.0,
        ignore_index=-100,
        output_transform=lambda x: x,
        device="cpu",
    ):
        self.threshold = threshold
        self.sigmoid = sigmoid
        self.num_classes = num_classes
        self.percentile = percentile
        self.ignore_index = ignore_index
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum = torch.tensor(0.0, device=self._device)
        self._count = torch.tensor(0, device=self._device)

    @staticmethod
    def _surface_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Distances from each boundary pixel of ``a`` to the boundary of ``b``."""
        a_border = a ^ binary_erosion(a)
        b_border = b ^ binary_erosion(b)
        # distance_transform_edt gives, at each pixel, the distance to the
        # nearest zero; feeding ~b_border makes those distances measure the gap
        # to the closest boundary pixel of b.
        dist_to_b = distance_transform_edt(~b_border)
        return dist_to_b[a_border]

    def _hd95(self, pred: np.ndarray, gt: np.ndarray) -> float:
        if not pred.any() and not gt.any():
            return 0.0
        if not pred.any() or not gt.any():
            h, w = pred.shape
            return float((h**2 + w**2) ** 0.5)  # image diagonal (worst case)
        distances = np.concatenate(
            [self._surface_distances(pred, gt), self._surface_distances(gt, pred)]
        )
        return float(np.percentile(distances, self.percentile))

    @reinit__is_reduced
    def update(self, output):
        preds, targets = output

        if self.num_classes > 1:
            pred_lab = preds.argmax(dim=1).cpu().numpy()
            gt_lab = targets.cpu().numpy()
            for p, g in zip(pred_lab, gt_lab):
                per_class = []
                for c in range(1, self.num_classes):
                    if c == self.ignore_index:
                        continue
                    pc, gc = (p == c), (g == c)
                    if not pc.any() and not gc.any():
                        continue  # class absent from both -> not applicable
                    per_class.append(self._hd95(pc, gc))
                if per_class:
                    self._sum += sum(per_class) / len(per_class)
                    self._count += 1
        else:
            preds = torch.sigmoid(preds) if self.sigmoid else preds
            preds = preds > self.threshold
            if preds.ndim == 4:
                preds = preds.squeeze(1)
            if targets.ndim == 4:
                targets = targets.squeeze(1)
            targets = targets > 0.5

            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()
            for p, g in zip(preds, targets):
                self._sum += self._hd95(p, g)
                self._count += 1

    @sync_all_reduce("_sum", "_count")
    def compute(self):
        if self._count == 0:
            raise NotComputableError(
                f"{type(self).__name__} must have at least one example."
            )
        return (self._sum / self._count).item()
