"""Tests for segmentation metrics (HD95)."""

import math

import torch

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
