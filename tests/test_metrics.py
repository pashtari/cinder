"""Tests for segmentation metrics (HD95)."""

import math

import pytest
import torch
from ignite.exceptions import NotComputableError

from cinder import HausdorffDistanceMetric


def _run(metric, preds, targets):
    metric.reset()
    metric.update((preds, targets))
    return metric.compute()


def _mask_to_logits(mask: torch.Tensor) -> torch.Tensor:
    """Turn a {0,1} mask into confident logits (sigmoid -> ~mask)."""
    return torch.where(mask > 0, 10.0, -10.0)


def test_hd95_identical_is_zero():
    gt = torch.zeros(1, 1, 64, 64)
    gt[..., 16:48, 16:48] = 1
    hd = _run(HausdorffDistanceMetric(num_classes=1), _mask_to_logits(gt), gt)
    assert hd == 0.0


def test_hd95_shifted_box_is_small_positive():
    gt = torch.zeros(1, 1, 64, 64)
    gt[..., 16:48, 16:48] = 1
    pred = torch.zeros(1, 1, 64, 64)
    pred[..., 16:48, 20:52] = 1  # shifted +4 px along x
    hd = _run(HausdorffDistanceMetric(num_classes=1), _mask_to_logits(pred), gt)
    assert 0.0 < hd < 10.0


def test_hd95_empty_prediction_penalized_with_diagonal():
    gt = torch.zeros(1, 1, 10, 10)
    gt[..., 2:8, 2:8] = 1
    logits = torch.full((1, 1, 10, 10), -10.0)  # predicts all background
    hd = _run(HausdorffDistanceMetric(num_classes=1), logits, gt)
    assert math.isclose(hd, math.sqrt(10**2 + 10**2), rel_tol=1e-6)


def test_hd95_averages_over_batch():
    gt = torch.zeros(2, 1, 32, 32)
    gt[:, :, 8:24, 8:24] = 1
    preds = _mask_to_logits(gt.clone())
    # second sample shifted -> nonzero HD95, first sample perfect -> 0
    pred1 = torch.zeros(1, 1, 32, 32)
    pred1[..., 8:24, 10:26] = 1
    preds[1] = _mask_to_logits(pred1)[0]
    hd = _run(HausdorffDistanceMetric(num_classes=1), preds, gt)
    assert hd > 0.0


if __name__ == "__main__":
    test_hd95_identical_is_zero()
    test_hd95_shifted_box_is_small_positive()
    test_hd95_empty_prediction_penalized_with_diagonal()
    test_hd95_averages_over_batch()
    print("All HD95 metric tests passed.")


@pytest.mark.parametrize(
    "metric_cls, expected", [("IoUMetric", 0.25), ("DiceMetric", 1 / 3)]
)
def test_multiclass_mean_excludes_ignored_and_absent_classes(metric_cls, expected):
    import cinder.engine.metrics as metrics

    # Class 0 is ignored, class 3 is absent. Classes 1 and 2 have IoU .5 and 0.
    logits = torch.tensor([[[[0.0, 0.0]], [[5.0, 5.0]], [[0.0, 0.0]], [[0.0, 0.0]]]])
    target = torch.tensor([[[1, 2]]])
    metric = getattr(metrics, metric_cls)(num_classes=4, ignore_index=0)

    assert _run(metric, logits, target) == pytest.approx(expected)


@pytest.mark.parametrize("metric_cls", ["IoUMetric", "DiceMetric"])
def test_multiclass_mean_requires_valid_classes(metric_cls):
    import cinder.engine.metrics as metrics

    metric = getattr(metrics, metric_cls)(num_classes=3, ignore_index=0)
    with pytest.raises(NotComputableError):
        metric.compute()
    metric.update((torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long)))
    with pytest.raises(NotComputableError):
        metric.compute()


def test_multiclass_hd95_ignores_predictions_on_unlabelled_pixels():
    target = torch.zeros(1, 10, 10, dtype=torch.long)
    target[:, 2:5, 2:5] = 1
    logits = torch.zeros(1, 2, 10, 10)
    logits[:, 1] = 5  # Extra foreground is entirely in the ignored region.
    metric = HausdorffDistanceMetric(num_classes=2, ignore_index=0)

    assert _run(metric, logits, target) == 0
