"""Pixel and object segmentation metrics for evaluator outputs ``(logits, targets)``.

Pixel metrics (Dice, IoU, HD95) threshold ``sigmoid(logits)`` for binary masks
and take the argmax for several classes. Object metrics (F1, object Dice, object
Hausdorff) score individual instances and need per-object target labels
(``instance_labels=True``); their NumPy functions take instance maps with 0 as
background.

Conventions:

- Binary Dice and IoU are averaged over images; an image whose prediction and
  target are both empty scores 1. Multi-class Dice and IoU accumulate counts per
  class over the dataset, then average the classes that occur in a prediction or
  target, skipping ``ignore_index``.
- HD95 takes the percentile of both directed boundary distances pooled together,
  as MedPy does, and counts image-border pixels as boundary. Two empty masks
  score 0; a single empty mask scores the image diagonal.
- Objects are 4-connected components of the thresholded prediction. Object F1
  counts a prediction as a hit when its IoU with a target object exceeds 0.5.
- Object Dice and Hausdorff weight each object by its area and compare it with
  its largest overlap. Unmatched objects score a Dice of 0 and, for Hausdorff,
  fall back to the nearest object by centroid. Object metrics are averaged over
  images.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.ndimage import label as label_components
from scipy.spatial.distance import cdist
from torch import Tensor

__all__ = [
    "DiceMetric",
    "IoUMetric",
    "HausdorffDistanceMetric",
    "BaseObjectMetric",
    "ObjectF1Metric",
    "ObjectDiceMetric",
    "ObjectHausdorffMetric",
    "label_instances",
    "object_f1",
    "object_dice",
    "object_hausdorff",
]


class DiceMetric(Metric):
    """Dice averaged over images (binary) or over dataset-wide classes (multi-class).

    Args:
        threshold: Binarization threshold; binary only.
        sigmoid: Apply a sigmoid before thresholding; binary only.
        num_classes: ``1`` for binary masks, or the number of argmax classes.
        ignore_index: Target label excluded from the counts; multi-class only.
            Classes absent from both predictions and targets are also left out
            of the mean.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sigmoid: bool = True,
        num_classes: int = 1,
        ignore_index: int = -100,
        output_transform: Callable[[Any], Sequence[Tensor]] = lambda x: x,
        device: str | torch.device = "cpu",
    ) -> None:
        self.threshold = threshold
        self.sigmoid = sigmoid
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        super().__init__(output_transform=output_transform, device=device)

    def score(
        self, intersection: Tensor, pred_sum: Tensor, target_sum: Tensor
    ) -> Tensor:
        """Score overlap counts; subclasses redefine the score."""
        return (2.0 * intersection + 1e-8) / (pred_sum + target_sum + 1e-8)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_score = torch.tensor(0.0, device=self._device)
        self._num_samples = torch.tensor(0, device=self._device)
        self._intersection = torch.zeros(max(self.num_classes, 1), device=self._device)
        self._pred_sum = torch.zeros(max(self.num_classes, 1), device=self._device)
        self._target_sum = torch.zeros(max(self.num_classes, 1), device=self._device)

    @reinit__is_reduced
    def update(self, output: Sequence[Tensor]) -> None:
        preds, targets = output

        if self.num_classes > 1:
            preds = preds.argmax(dim=1).flatten()
            targets = targets.flatten()
            valid = targets != self.ignore_index
            preds = preds[valid]
            targets = targets[valid]
            for class_index in range(self.num_classes):
                pred_mask = preds == class_index
                target_mask = targets == class_index
                self._intersection[class_index] += (
                    (pred_mask & target_mask).sum().to(self._device)
                )
                self._pred_sum[class_index] += pred_mask.sum().to(self._device)
                self._target_sum[class_index] += target_mask.sum().to(self._device)
        else:
            if self.sigmoid:
                preds = torch.sigmoid(preds)
            preds = (preds > self.threshold).float()

            # Instance targets may use any positive integer for foreground.
            preds_flat = preds.flatten(1)
            targets_flat = (targets > 0).float().flatten(1)

            intersection = (preds_flat * targets_flat).sum(dim=1)
            pred_sum = preds_flat.sum(dim=1)
            target_sum = targets_flat.sum(dim=1)

            score = self.score(intersection, pred_sum, target_sum)
            self._sum_score += score.sum().to(self._device)
            self._num_samples += score.shape[0]

    @sync_all_reduce(
        "_sum_score", "_num_samples", "_intersection", "_pred_sum", "_target_sum"
    )
    def compute(self) -> float:
        if self.num_classes > 1:
            included = (self._pred_sum + self._target_sum) > 0
            if 0 <= self.ignore_index < self.num_classes:
                included[self.ignore_index] = False
            if not included.any():
                raise NotComputableError(
                    f"{type(self).__name__} must have at least one valid class."
                )
            scores = self.score(self._intersection, self._pred_sum, self._target_sum)
            return scores[included].mean().item()
        if self._num_samples == 0:
            raise NotComputableError(
                f"{type(self).__name__} must have at least one example."
            )
        return (self._sum_score / self._num_samples).item()


class IoUMetric(DiceMetric):
    """IoU averaged over images (binary) or over dataset-wide classes (multi-class)."""

    def score(
        self, intersection: Tensor, pred_sum: Tensor, target_sum: Tensor
    ) -> Tensor:
        union = pred_sum + target_sum - intersection
        return (intersection + 1e-8) / (union + 1e-8)


def _surface_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Distances from each source boundary pixel to the target boundary."""
    source_border = source ^ binary_erosion(source)
    target_border = target ^ binary_erosion(target)
    # The distance transform measures distance to zeros, hence the inversion.
    distances = distance_transform_edt(~target_border)
    return distances[source_border]


def _percentile_hausdorff(pred: np.ndarray, gt: np.ndarray, percentile: float) -> float:
    """Percentile of the pooled boundary distances between two binary masks."""
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        height, width = pred.shape
        return float((height**2 + width**2) ** 0.5)
    distances = np.concatenate(
        [_surface_distances(pred, gt), _surface_distances(gt, pred)]
    )
    return float(np.percentile(distances, percentile))


class HausdorffDistanceMetric(Metric):
    """Percentile Hausdorff distance in pixels, HD95 by default; lower is better.

    Binary masks give one distance per image. With several classes, each image
    averages its foreground classes ``1 .. num_classes - 1`` that occur in the
    prediction or target, and images without any are skipped. Distances are then
    averaged over images.

    Args:
        threshold: Binarization threshold; binary only.
        sigmoid: Apply a sigmoid before thresholding; binary only.
        num_classes: ``1`` for binary masks, or the number of argmax classes.
        percentile: Percentile of the boundary distances.
        ignore_index: Target label whose pixels are ignored; multi-class only.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sigmoid: bool = True,
        num_classes: int = 1,
        percentile: float = 95.0,
        ignore_index: int = -100,
        output_transform: Callable[[Any], Sequence[Tensor]] = lambda x: x,
        device: str | torch.device = "cpu",
    ) -> None:
        self.threshold = threshold
        self.sigmoid = sigmoid
        self.num_classes = num_classes
        self.percentile = percentile
        self.ignore_index = ignore_index
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum = torch.tensor(0.0, device=self._device)
        self._count = torch.tensor(0, device=self._device)

    @reinit__is_reduced
    def update(self, output: Sequence[Tensor]) -> None:
        preds, targets = output

        if self.num_classes > 1:
            pred_labels = preds.argmax(dim=1).cpu().numpy()
            target_labels = targets.cpu().numpy()
            for pred, target in zip(pred_labels, target_labels):
                valid = target != self.ignore_index
                per_class: list[float] = []
                for class_index in range(1, self.num_classes):
                    if class_index == self.ignore_index:
                        continue
                    pred_mask = (pred == class_index) & valid
                    target_mask = (target == class_index) & valid
                    if not pred_mask.any() and not target_mask.any():
                        continue
                    per_class.append(
                        _percentile_hausdorff(pred_mask, target_mask, self.percentile)
                    )
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
            for pred, target in zip(preds, targets):
                self._sum += _percentile_hausdorff(pred, target, self.percentile)
                self._count += 1

    @sync_all_reduce("_sum", "_count")
    def compute(self) -> float:
        if self._count == 0:
            raise NotComputableError(
                f"{type(self).__name__} must have at least one example."
            )
        return (self._sum / self._count).item()


def label_instances(mask: np.ndarray, min_size: int = 0) -> np.ndarray:
    """Label the connected components of a binary mask.

    Args:
        mask: Binary (or truthy) 2-D array.
        min_size: Drop components smaller than this many pixels.

    Returns:
        Integer label map, 0 for background.
    """
    labels, num_objects = label_components(np.asarray(mask) > 0)
    if min_size > 0 and num_objects > 0:
        counts = np.bincount(labels.ravel())
        too_small = np.flatnonzero(counts < min_size)
        too_small = too_small[too_small != 0]
        if too_small.size:
            labels[np.isin(labels, too_small)] = 0
            # Keep object IDs contiguous after removing small components.
            labels, _ = label_components(labels > 0)
    return labels


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Use contiguous object IDs while preserving background and object identity."""
    ids = np.unique(labels)
    ids = ids[ids != 0]
    return np.where(labels == 0, 0, np.searchsorted(ids, labels) + 1)


def _overlaps(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Pixel counts for each (target ID, prediction ID), including background."""
    num_gt, num_pred = int(gt.max()), int(pred.max())
    pair_ids = (
        gt.astype(np.int64).ravel() * (num_pred + 1) + pred.astype(np.int64).ravel()
    )
    return np.bincount(pair_ids, minlength=(num_gt + 1) * (num_pred + 1)).reshape(
        num_gt + 1, num_pred + 1
    )


def object_f1(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Object detection F1, counting matches with IoU above 0.5.

    Above 0.5 IoU a prediction can match at most one target, so matches are
    one-to-one without an explicit assignment.

    Returns:
        ``(f1, precision, recall)``; two empty maps score 1.
    """
    gt, pred = _relabel(gt), _relabel(pred)
    counts = _overlaps(gt, pred)
    intersection = counts[1:, 1:]
    num_gt, num_pred = intersection.shape
    if num_gt == 0 and num_pred == 0:
        return 1.0, 1.0, 1.0
    if num_gt == 0 or num_pred == 0:
        return 0.0, 0.0, 0.0

    gt_area = counts.sum(axis=1)[1:]
    pred_area = counts.sum(axis=0)[1:]
    union = gt_area[:, None] + pred_area[None, :] - intersection
    iou = np.where(union > 0, intersection / np.maximum(union, 1), 0.0)

    true_positives = int((iou > 0.5).sum())
    precision = true_positives / num_pred
    recall = true_positives / num_gt
    f1 = 0.0 if true_positives == 0 else 2 * precision * recall / (precision + recall)
    return float(f1), float(precision), float(recall)


def _best_match(counts: np.ndarray) -> np.ndarray:
    """Return the largest-overlap prediction ID for each target (0 for none)."""
    intersection = counts[1:, 1:]
    best = intersection.argmax(axis=1) + 1
    return np.where(intersection.max(axis=1) > 0, best, 0)


def _weighted_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Area-weighted Dice from each target to its largest-overlap prediction."""
    counts = _overlaps(gt, pred)
    gt_area = counts.sum(axis=1)[1:]
    pred_area = counts.sum(axis=0)[1:]
    matches = _best_match(counts)
    total_area = gt_area.sum()
    score = 0.0
    for gt_index, pred_id in enumerate(matches):
        if pred_id == 0:
            continue  # Unmatched targets contribute zero Dice.
        intersection = counts[gt_index + 1, pred_id]
        denominator = gt_area[gt_index] + pred_area[pred_id - 1]
        score += (gt_area[gt_index] / total_area) * (2.0 * intersection / denominator)
    return float(score)


def object_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Symmetric, area-weighted Dice against each object's largest overlap.

    Unmatched objects score zero; two empty maps score one.
    """
    gt, pred = _relabel(gt), _relabel(pred)
    has_gt, has_pred = gt.max() > 0, pred.max() > 0
    if not has_gt and not has_pred:
        return 1.0
    if not has_gt or not has_pred:
        return 0.0
    return 0.5 * (_weighted_dice(gt, pred) + _weighted_dice(pred, gt))


def _boundary_points(mask: np.ndarray) -> np.ndarray:
    """Coordinates of a binary mask's boundary pixels."""
    return np.argwhere(mask & ~binary_erosion(mask, border_value=0))


def _hausdorff(source_points: np.ndarray, target_points: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two point sets."""
    distances = cdist(source_points, target_points)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def _centroids(labels: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Return the ``(len(ids), 2)`` centroids of the given objects."""
    return np.array([np.argwhere(labels == label_id).mean(axis=0) for label_id in ids])


def _weighted_hausdorff(gt: np.ndarray, pred: np.ndarray) -> float:
    """Area-weighted Hausdorff from each target to its matched prediction."""
    counts = _overlaps(gt, pred)
    gt_area = counts.sum(axis=1)[1:]
    gt_ids = np.arange(1, gt.max() + 1)
    pred_ids = np.arange(1, pred.max() + 1)

    matches = _best_match(counts)
    # Unmatched objects fall back to the nearest counterpart by centroid.
    if (matches == 0).any():
        gt_centroids = _centroids(gt, gt_ids)
        pred_centroids = _centroids(pred, pred_ids)
        nearest = cdist(gt_centroids, pred_centroids).argmin(axis=1) + 1
        matches = np.where(matches == 0, nearest, matches)

    boundaries = {
        pred_id: _boundary_points(pred == pred_id) for pred_id in np.unique(matches)
    }
    total_area = gt_area.sum()
    score = 0.0
    for gt_index, pred_id in enumerate(matches):
        gt_points = _boundary_points(gt == gt_ids[gt_index])
        score += (gt_area[gt_index] / total_area) * _hausdorff(
            gt_points, boundaries[pred_id]
        )
    return float(score)


def object_hausdorff(gt: np.ndarray, pred: np.ndarray) -> float:
    """Symmetric, area-weighted Hausdorff distance against each largest overlap.

    Unmatched objects use the nearest counterpart by centroid. If only one map
    is empty, return the image diagonal; two empty maps have zero distance.
    Distances are measured in pixels.
    """
    gt, pred = _relabel(gt), _relabel(pred)
    diagonal = float(np.hypot(*gt.shape))
    has_gt, has_pred = gt.max() > 0, pred.max() > 0
    if not has_gt and not has_pred:
        return 0.0
    if not has_gt or not has_pred:
        return diagonal
    return 0.5 * (_weighted_hausdorff(gt, pred) + _weighted_hausdorff(pred, gt))


class BaseObjectMetric(Metric):
    """Base class for object metrics, averaged over images.

    Subclasses implement :meth:`score`. Predictions become instances as the
    connected components of ``sigmoid(logits) > threshold``; targets must carry
    integer object labels (``instance_labels=True``).

    Args:
        threshold: Probability threshold for the prediction.
        min_object_size: Drop predicted components smaller than this many pixels.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_object_size: int = 0,
        output_transform: Callable[[Any], Sequence[Tensor]] = lambda x: x,
        device: str | torch.device = "cpu",
    ) -> None:
        self.threshold = threshold
        self.min_object_size = min_object_size
        super().__init__(output_transform=output_transform, device=device)

    def score(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Score one image from its ground-truth and predicted instance labels."""
        raise NotImplementedError

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum = torch.tensor(0.0, device=self._device)
        self._count = torch.tensor(0, device=self._device)

    @reinit__is_reduced
    def update(self, output: Sequence[Tensor]) -> None:
        preds, targets = output
        probabilities = torch.sigmoid(preds)
        for probability, target in zip(probabilities, targets):
            if target.is_floating_point() or target.dtype == torch.bool:
                raise ValueError(
                    f"{type(self).__name__} needs integer per-object target labels. "
                    "Build the dataset with instance_labels=True "
                    "(dataset.instance_labels=true)."
                )
            gt = np.asarray(target.squeeze(0).cpu(), dtype=np.int64)
            mask = probability.squeeze(0).cpu().numpy() > self.threshold
            pred = label_instances(mask, min_size=self.min_object_size)
            self._sum += self.score(gt, pred)
            self._count += 1

    @sync_all_reduce("_sum", "_count")
    def compute(self) -> float:
        if self._count == 0:
            raise NotComputableError(
                f"{type(self).__name__} must have at least one example."
            )
        return (self._sum / self._count).item()


class ObjectF1Metric(BaseObjectMetric):
    """Object detection F1 at IoU above 0.5; higher is better."""

    def score(self, gt: np.ndarray, pred: np.ndarray) -> float:
        return object_f1(gt, pred)[0]


class ObjectDiceMetric(BaseObjectMetric):
    """Symmetric, area-weighted object Dice in ``[0, 1]``; higher is better."""

    def score(self, gt: np.ndarray, pred: np.ndarray) -> float:
        return object_dice(gt, pred)


class ObjectHausdorffMetric(BaseObjectMetric):
    """Symmetric, area-weighted object Hausdorff distance in pixels; lower is better."""

    def score(self, gt: np.ndarray, pred: np.ndarray) -> float:
        return object_hausdorff(gt, pred)
