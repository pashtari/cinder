"""Unit tests for the GlaS object-level metrics.

These score whether individual glands were found and delineated, rather than
how much gland area was labelled, so the properties worth pinning down are the
ones that distinguish them from pixel Dice: a merged pair of touching objects
must be penalised, and the whole thing must be symmetric in its two arguments.
"""

import numpy as np
import pytest

from cinder.engine.metrics import (
    detection_f1,
    glas_metrics,
    label_instances,
    object_dice,
    object_hausdorff,
)


def two_squares(gap: int = 4, size: int = 10, canvas: int = 40) -> np.ndarray:
    """Two square objects separated by *gap* background pixels."""
    m = np.zeros((canvas, canvas), dtype=np.int32)
    m[5 : 5 + size, 5 : 5 + size] = 1
    m[5 : 5 + size, 5 + size + gap : 5 + 2 * size + gap] = 2
    return m


# ---------------------------------------------------------------------------
# Identity and degenerate cases
# ---------------------------------------------------------------------------


def test_identity_is_perfect():
    gt = two_squares()
    assert detection_f1(gt, gt) == (1.0, 1.0, 1.0)
    assert object_dice(gt, gt) == pytest.approx(1.0)
    assert object_hausdorff(gt, gt) == pytest.approx(0.0)


def test_empty_prediction_scores_zero_and_penalises_hausdorff():
    gt = two_squares()
    empty = np.zeros_like(gt)
    assert detection_f1(gt, empty) == (0.0, 0.0, 0.0)
    assert object_dice(gt, empty) == 0.0
    # Falls back to the image diagonal.
    assert object_hausdorff(gt, empty) == pytest.approx(np.hypot(*gt.shape))


def test_both_empty_is_perfect():
    empty = np.zeros((20, 20), dtype=np.int32)
    assert detection_f1(empty, empty) == (1.0, 1.0, 1.0)
    assert object_dice(empty, empty) == 1.0
    assert object_hausdorff(empty, empty) == 0.0


# ---------------------------------------------------------------------------
# The behaviour that pixel Dice misses
# ---------------------------------------------------------------------------


def test_merging_two_objects_is_penalised():
    """One blob covering both glands is near-perfect on pixels, not on objects."""
    gt = two_squares(gap=4)
    merged = np.zeros_like(gt)
    merged[5:15, 5:29] = 1  # single object spanning both squares and the gap

    pixel_overlap = np.logical_and(gt > 0, merged > 0).sum() / (gt > 0).sum()
    assert pixel_overlap == 1.0  # every gland pixel is covered

    f1, precision, recall = detection_f1(gt, merged)
    assert f1 < 0.7  # one prediction cannot match two ground-truth objects
    assert object_dice(gt, merged) < 0.9


def test_splitting_one_object_is_penalised():
    gt = np.zeros((40, 40), dtype=np.int32)
    gt[5:15, 5:29] = 1
    split = two_squares(gap=4)
    f1, _, _ = detection_f1(gt, split)
    assert f1 < 0.7


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------


def test_object_dice_is_symmetric():
    gt = two_squares()
    pred = two_squares()
    pred[5:15, 5:15] = 0  # drop the first object
    pred = label_instances(pred > 0)
    assert object_dice(gt, pred) == pytest.approx(object_dice(pred, gt))


def test_object_hausdorff_is_symmetric():
    gt = two_squares()
    pred = two_squares(gap=6)
    assert object_hausdorff(gt, pred) == pytest.approx(object_hausdorff(pred, gt))


def test_shrinking_boundaries_lowers_dice_and_raises_hausdorff():
    gt = two_squares()
    shrunk = np.zeros_like(gt)
    shrunk[7:13, 7:13] = 1
    shrunk[7:13, 21:27] = 2
    assert object_dice(gt, shrunk) < object_dice(gt, gt)
    assert object_hausdorff(gt, shrunk) > object_hausdorff(gt, gt)


def test_missing_one_of_two_objects_halves_recall():
    gt = two_squares()
    one = np.where(gt == 1, 1, 0).astype(np.int32)
    f1, precision, recall = detection_f1(gt, one)
    assert recall == pytest.approx(0.5)
    assert precision == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# label_instances
# ---------------------------------------------------------------------------


def test_label_instances_separates_disconnected_components():
    binary = two_squares() > 0
    labels = label_instances(binary)
    assert labels.max() == 2


def test_label_instances_drops_speckle():
    binary = two_squares() > 0
    binary[30, 30] = True  # one-pixel false positive
    assert label_instances(binary).max() == 3
    assert label_instances(binary, min_size=4).max() == 2


def test_glas_metrics_reports_all_three():
    gt = two_squares()
    keys = set(glas_metrics(gt, gt))
    assert keys == {"f1", "precision", "recall", "object_dice", "object_hausdorff"}


# ---------------------------------------------------------------------------
# Ignite adapters
#
# The unification made pixel metrics responsible for thresholding their own
# target, so that instance-labelled targets can reach the object metrics. The
# property that must hold is that pixel metrics are blind to the difference.
# ---------------------------------------------------------------------------


def _logits_and_target():
    """Two square objects, and logits that recover them imperfectly."""
    torch = pytest.importorskip("torch")
    gt = two_squares()  # labels {0, 1, 2}
    logits = torch.full((1, 1, 40, 40), -4.0)
    logits[0, 0, 6:14, 6:14] = 4.0  # slightly shrunk first object
    logits[0, 0, 5:15, 19:29] = 4.0  # second object
    target = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
    return logits, target


def test_pixel_metrics_ignore_instance_labels():
    """Dice/IoU/HD95 must score an instance-labelled target as its binary form."""
    torch = pytest.importorskip("torch")
    from cinder.engine.metrics import DiceMetric, HausdorffDistanceMetric, IoUMetric

    logits, instance_target = _logits_and_target()
    binary_target = (instance_target > 0).float()

    for cls in (DiceMetric, IoUMetric, HausdorffDistanceMetric):
        scores = []
        for target in (instance_target, binary_target):
            metric = cls()
            metric.reset()
            metric.update((logits, target))
            scores.append(metric.compute())
        assert scores[0] == pytest.approx(scores[1]), cls.__name__


def test_object_metric_adapters_score_like_the_functions():
    from cinder.engine.metrics import DetectionF1, ObjectDice, ObjectHausdorff

    logits, target = _logits_and_target()
    gt = target.squeeze().numpy()
    pred = label_instances(logits.squeeze().numpy() > 0)

    for cls, fn in (
        (DetectionF1, lambda g, p: detection_f1(g, p)[0]),
        (ObjectDice, object_dice),
        (ObjectHausdorff, object_hausdorff),
    ):
        metric = cls()
        metric.reset()
        metric.update((logits, target))
        assert metric.compute() == pytest.approx(fn(gt, pred)), cls.__name__


def test_object_metric_rejects_a_binary_target():
    """A binary target would silently score every object as one merged blob."""
    from cinder.engine.metrics import ObjectDice

    logits, target = _logits_and_target()
    metric = ObjectDice()
    metric.reset()
    with pytest.raises(ValueError, match="instance_labels"):
        metric.update((logits, (target > 0).float()))
