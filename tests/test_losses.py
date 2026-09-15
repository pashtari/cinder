"""Regression tests for ignored labels in segmentation losses."""

import pytest
import torch
import torch.nn.functional as F

from cinder.engine.losses import (
    DiceCELoss,
    DiceLoss,
    MaskedCrossEntropyLoss,
    SampledLoss,
)


def test_all_ignored_targets_produce_zero_loss_and_gradients():
    pred = torch.randn(2, 3, 4, 4, requires_grad=True)
    target = torch.zeros(2, 4, 4, dtype=torch.long)

    loss = DiceCELoss(softmax=True, ignore_index=0)(pred, target)
    loss.backward()

    assert loss.item() == 0
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


def test_ignored_targets_preserve_cross_entropy_mean():
    pred = torch.randn(2, 3, 4, 4)
    target = torch.randint(0, 3, (2, 4, 4))
    loss = DiceCELoss(softmax=True, ignore_index=0, lambda_dice=0)

    torch.testing.assert_close(
        loss(pred, target), F.cross_entropy(pred, target, ignore_index=0)
    )


@pytest.mark.parametrize("ignore_index", [-100, 0])
@pytest.mark.parametrize("ignore_pixels", [False, True])
def test_masked_cross_entropy_matches_pytorch_loss_and_gradients(
    ignore_index, ignore_pixels
):
    pred = torch.randn(2, 3, 2, 2, requires_grad=True)
    target = torch.tensor([[[1, 2], [2, 1]], [[2, 1], [1, 2]]])
    if ignore_pixels:
        target[:, 0, 0] = ignore_index

    actual = MaskedCrossEntropyLoss(ignore_index)(pred, target)
    expected = F.cross_entropy(pred, target, ignore_index=ignore_index)

    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, pred)[0]
    expected_grad = torch.autograd.grad(expected, pred)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("ignore_index", [-100, 0])
def test_masked_cross_entropy_handles_entirely_ignored_targets(ignore_index):
    pred = torch.randn(2, 3, 4, 4, requires_grad=True)
    target = torch.full((2, 4, 4), ignore_index, dtype=torch.long)

    loss = MaskedCrossEntropyLoss(ignore_index)(pred, target)
    loss.backward()

    assert loss.item() == 0
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


def test_binary_dice_ce_keeps_its_original_objective():
    pred = torch.randn(2, 1, 4, 4)
    target = torch.randint(0, 2, (2, 1, 4, 4)).float()
    loss = DiceCELoss(lambda_dice=0.5, lambda_ce=2)

    expected = 0.5 * DiceLoss()(pred, target) + 2 * F.binary_cross_entropy_with_logits(
        pred, target
    )
    torch.testing.assert_close(loss(pred, target), expected)


@pytest.mark.parametrize("multiclass", [False, True])
def test_sampled_loss_reads_the_pixels_the_model_decoded(multiclass):
    class Model(torch.nn.Module):
        sample_indices = torch.tensor([5, 0, 11])

    pred = torch.randn(2, 3 if multiclass else 1, 3)
    if multiclass:
        target = torch.randint(0, 3, (2, 3, 4))
        base = MaskedCrossEntropyLoss()
        expected = base(pred, target.flatten(1)[:, [5, 0, 11]])
    else:
        target = torch.rand(2, 1, 3, 4).round()
        base = DiceLoss()
        expected = base(pred, target.flatten(2)[:, :, [5, 0, 11]])

    model = Model()
    torch.testing.assert_close(SampledLoss(base, model)(pred, target), expected)

    # Without sampled coordinates, dense predictions reach the loss unchanged.
    model.sample_indices = None
    dense = torch.randn(2, 3 if multiclass else 1, 3, 4)
    torch.testing.assert_close(
        SampledLoss(base, model)(dense, target), base(dense, target)
    )
