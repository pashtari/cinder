"""Segmentation metrics.

Two families, both exposed as Ignite metrics over evaluator output
``(preds, targets)`` with *preds* as raw logits:

**Pixel-level** — :class:`DiceMetric`, :class:`IoUMetric`,
:class:`HausdorffDistanceMetric`. They score how much area was labelled
correctly, and treat any non-zero target as foreground.

**Object-level** — :class:`DetectionF1`, :class:`ObjectDice`,
:class:`ObjectHausdorff`, the metrics the gland-segmentation literature reports.
They score whether individual objects were found and delineated, which pixel
Dice cannot see: a prediction merging two touching glands into one blob scores
well on area and badly here. These need the target to carry one label per
object, so the dataset must be built with ``instance_labels=True``.
"""

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

__all__ = [
    "DiceMetric",
    "IoUMetric",
    "HausdorffDistanceMetric",
    "DetectionF1",
    "ObjectDice",
    "ObjectHausdorff",
    "label_instances",
    "detection_f1",
    "object_dice",
    "object_hausdorff",
    "glas_metrics",
]


# ---------------------------------------------------------------------------
# Pixel-level
# ---------------------------------------------------------------------------


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

            # Any non-zero target is foreground: the target may carry per-object
            # labels (see SegmentationDataset.instance_labels) rather than {0, 1}.
            preds_flat = preds.flatten(1)
            targets_flat = (targets > 0).float().flatten(1)

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
            targets = targets > 0

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


# ---------------------------------------------------------------------------
# Object-level
#
# Following Sirinukunwattana et al., "Gland Segmentation in Colon Histology
# Images: The GlaS Challenge Contest" (Medical Image Analysis, 2017):
#
#   detection F1   a predicted object is a hit when it overlaps a ground-truth
#                  object with IoU > 0.5. At that threshold the matching is
#                  necessarily one-to-one, so no assignment step is needed.
#
#   object Dice / Hausdorff
#                  area-weighted averages over objects, symmetrised:
#                  M(G,S) = ½[Σ γ_i m(G_i, S*(G_i)) + Σ σ_j m(G*(S_j), S_j)],
#                  where S*(G_i) is the predicted object overlapping G_i most.
#
# Unmatched objects are where implementations diverge. Here an object with no
# overlapping counterpart contributes Dice 0, and for Hausdorff is paired with
# the nearest counterpart by centroid; if the other mask is empty entirely the
# image diagonal is the penalty. Cross-check these conventions against the
# reference implementation before comparing against published numbers.
#
# The functions below take integer instance maps -- 0 background, 1..N one
# value per object -- and are kept independent of Ignite so they can be tested
# directly; the Metric classes at the end are thin adapters over them.
# ---------------------------------------------------------------------------

from scipy.ndimage import label as cc_label  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402


def label_instances(mask: np.ndarray, min_size: int = 0) -> np.ndarray:
    """Label the connected components of a binary mask.

    Args:
        mask: Binary (or truthy) 2-D array.
        min_size: Drop components smaller than this many pixels. The challenge
            imposes no minimum, but a few pixels of speckle otherwise register
            as false-positive glands.

    Returns:
        Integer label map, 0 for background.
    """
    labels, n = cc_label(np.asarray(mask) > 0)
    if min_size > 0 and n > 0:
        counts = np.bincount(labels.ravel())
        too_small = np.flatnonzero(counts < min_size)
        too_small = too_small[too_small != 0]
        if too_small.size:
            labels[np.isin(labels, too_small)] = 0
            # Renumber so labels stay contiguous.
            labels, _ = cc_label(labels > 0)
    return labels


def _overlaps(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Contingency table ``C[i, j] = |G_i ∩ S_j|``, including the 0 background."""
    n_g, n_p = int(gt.max()), int(pred.max())
    flat = gt.astype(np.int64).ravel() * (n_p + 1) + pred.astype(np.int64).ravel()
    return np.bincount(flat, minlength=(n_g + 1) * (n_p + 1)).reshape(n_g + 1, n_p + 1)


def detection_f1(gt: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5):
    """Object detection F1, with precision and recall.

    Returns:
        ``(f1, precision, recall)``. An image with no ground-truth and no
        predicted objects scores 1.0 -- nothing was there and nothing was
        claimed.
    """
    counts = _overlaps(gt, pred)
    inter = counts[1:, 1:]
    n_gt, n_pred = inter.shape
    if n_gt == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0, 0.0, 0.0

    gt_area = counts.sum(axis=1)[1:]
    pred_area = counts.sum(axis=0)[1:]
    union = gt_area[:, None] + pred_area[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    tp = int((iou > iou_threshold).sum())
    precision = tp / n_pred
    recall = tp / n_gt
    f1 = 0.0 if tp == 0 else 2 * precision * recall / (precision + recall)
    return float(f1), float(precision), float(recall)


def _best_match(counts: np.ndarray) -> np.ndarray:
    """For each object in axis 0, the axis-1 object it overlaps most (0 = none)."""
    inter = counts[1:, 1:]
    if inter.size == 0:
        return np.zeros(inter.shape[0], dtype=np.int64)
    best = inter.argmax(axis=1) + 1
    return np.where(inter.max(axis=1) > 0, best, 0)


def _weighted_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """One direction of the object-level Dice: every G_i against its best S_j."""
    counts = _overlaps(gt, pred)
    gt_area = counts.sum(axis=1)[1:]
    pred_area = counts.sum(axis=0)[1:]
    if gt_area.size == 0:
        return 0.0
    match = _best_match(counts)
    total = gt_area.sum()
    acc = 0.0
    for i, j in enumerate(match):
        if j == 0:
            continue  # no overlapping counterpart -> Dice 0
        inter = counts[i + 1, j]
        denom = gt_area[i] + pred_area[j - 1]
        acc += (gt_area[i] / total) * (2.0 * inter / denom)
    return float(acc)


def object_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Symmetric, area-weighted object-level Dice in [0, 1]."""
    has_gt, has_pred = gt.max() > 0, pred.max() > 0
    if not has_gt and not has_pred:
        return 1.0
    if not has_gt or not has_pred:
        return 0.0
    return 0.5 * (_weighted_dice(gt, pred) + _weighted_dice(pred, gt))


def _boundary_points(mask: np.ndarray) -> np.ndarray:
    """Coordinates of a binary mask's boundary pixels."""
    eroded = binary_erosion(mask, border_value=0)
    pts = np.argwhere(mask & ~eroded)
    return pts if pts.size else np.argwhere(mask)


def _hausdorff(a_pts: np.ndarray, b_pts: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two point sets."""
    d = cdist(a_pts, b_pts)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def _centroids(labels: np.ndarray, ids) -> np.ndarray:
    return np.array([np.argwhere(labels == i).mean(axis=0) for i in ids])


def _weighted_hausdorff(gt: np.ndarray, pred: np.ndarray, diagonal: float) -> float:
    """One direction of the object-level Hausdorff."""
    counts = _overlaps(gt, pred)
    gt_area = counts.sum(axis=1)[1:]
    if gt_area.size == 0:
        return 0.0
    gt_ids = np.arange(1, gt.max() + 1)
    pred_ids = np.arange(1, pred.max() + 1)
    if pred_ids.size == 0:
        return diagonal

    match = _best_match(counts)
    # Unmatched objects fall back to the nearest counterpart by centroid.
    if (match == 0).any():
        gt_c = _centroids(gt, gt_ids)
        pred_c = _centroids(pred, pred_ids)
        nearest = cdist(gt_c, pred_c).argmin(axis=1) + 1
        match = np.where(match == 0, nearest, match)

    boundaries = {j: _boundary_points(pred == j) for j in np.unique(match)}
    total = gt_area.sum()
    acc = 0.0
    for i, j in enumerate(match):
        g_pts = _boundary_points(gt == gt_ids[i])
        acc += (gt_area[i] / total) * _hausdorff(g_pts, boundaries[j])
    return float(acc)


def object_hausdorff(gt: np.ndarray, pred: np.ndarray) -> float:
    """Symmetric, area-weighted object-level Hausdorff distance, in pixels."""
    diagonal = float(np.hypot(*gt.shape))
    has_gt, has_pred = gt.max() > 0, pred.max() > 0
    if not has_gt and not has_pred:
        return 0.0
    if not has_gt or not has_pred:
        return diagonal
    return 0.5 * (
        _weighted_hausdorff(gt, pred, diagonal)
        + _weighted_hausdorff(pred, gt, diagonal)
    )


def glas_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """All three challenge metrics for one image.

    Args:
        gt: Ground-truth instance label map (the ``*_anno.bmp`` contents).
        pred: Predicted instance label map, e.g. from :func:`label_instances`.
    """
    f1, precision, recall = detection_f1(gt, pred)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "object_dice": object_dice(gt, pred),
        "object_hausdorff": object_hausdorff(gt, pred),
    }


class _ObjectMetric(Metric):
    """Base for the object-level metrics: threshold, label, score, average.

    Each image contributes one score, averaged over the accumulated samples.
    The target must carry one label per object; build the dataset with
    ``instance_labels=True`` (see
    :class:`~cinder.datasets.utils.SegmentationDataset`), or every gland in an
    image arrives as a single merged blob and the scores are meaningless.

    Args:
        threshold: Probability cut applied to ``sigmoid(logits)``.
        min_object_size: Drop predicted components below this many pixels. A
            few pixels of speckle otherwise register as false-positive objects.
    """

    def __init__(
        self,
        threshold=0.5,
        min_object_size=0,
        output_transform=lambda x: x,
        device="cpu",
    ):
        self.threshold = threshold
        self.min_object_size = min_object_size
        super().__init__(output_transform=output_transform, device=device)

    def _score(self, gt: np.ndarray, pred: np.ndarray) -> float:
        raise NotImplementedError

    @reinit__is_reduced
    def reset(self):
        self._sum = torch.tensor(0.0, device=self._device)
        self._count = torch.tensor(0, device=self._device)

    @reinit__is_reduced
    def update(self, output):
        preds, targets = output
        probabilities = torch.sigmoid(preds)
        for probability, target in zip(probabilities, targets):
            gt = np.asarray(target.squeeze().cpu(), dtype=np.int64)
            if gt.max() <= 1 and gt.any():
                raise ValueError(
                    f"{type(self).__name__} needs per-object target labels, but "
                    "the target is binary. Build the dataset with "
                    "instance_labels=True (dataset.instance_labels=true)."
                )
            mask = probability.squeeze().cpu().numpy() > self.threshold
            pred = label_instances(mask, min_size=self.min_object_size)
            self._sum += self._score(gt, pred)
            self._count += 1

    @sync_all_reduce("_sum", "_count")
    def compute(self):
        if self._count == 0:
            raise NotComputableError(
                f"{type(self).__name__} must have at least one example."
            )
        return (self._sum / self._count).item()


class DetectionF1(_ObjectMetric):
    """Object detection F1 at IoU > 0.5. Higher is better."""

    def _score(self, gt, pred):
        return detection_f1(gt, pred)[0]


class ObjectDice(_ObjectMetric):
    """Symmetric, area-weighted object-level Dice in [0, 1]. Higher is better."""

    def _score(self, gt, pred):
        return object_dice(gt, pred)


class ObjectHausdorff(_ObjectMetric):
    """Symmetric, area-weighted object-level Hausdorff, in pixels. Lower is better."""

    def _score(self, gt, pred):
        return object_hausdorff(gt, pred)
