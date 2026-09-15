"""Compact sparse products agree with dense values and autograd."""

import pytest
import torch

from cinder.models.rcs_matrix import RCSMatrix

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.complex64]
)
def test_sparse_inference_matches_dense_across_dtypes(device, dtype):
    torch.manual_seed(0)
    values = torch.randn(4, 2, device=device, dtype=dtype)
    starts = torch.tensor([0, 2, 1, 2], device=device)
    other = torch.randn(5, 3, device=device, dtype=dtype)
    sparse = RCSMatrix(values, starts, 5)
    with torch.no_grad():
        torch.testing.assert_close(sparse @ other, sparse.to_dense() @ other)


def test_sparse_cpu_autocast_matches_dense():
    values = torch.randn(4, 2)
    sparse = RCSMatrix(values, torch.tensor([0, 2, 1, 2]), 5)
    other = torch.randn(5, 3)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        actual = sparse @ other
        expected = sparse.to_dense() @ other
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_sparse_product_first_and_second_derivatives(device):
    values = torch.randn(3, 2, dtype=torch.float64, device=device, requires_grad=True)
    other = torch.randn(4, 2, dtype=torch.float64, device=device, requires_grad=True)
    starts = torch.tensor([0, 1, 1], device=device)

    def product(values, other):
        return RCSMatrix(values, starts, 4) @ other

    assert torch.autograd.gradcheck(product, (values, other))
    assert torch.autograd.gradgradcheck(product, (values, other))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows, outputs", [(0, 3), (3, 0)])
def test_empty_sparse_products_support_second_order_autograd(device, rows, outputs):
    values = torch.randn(rows, 2, device=device, requires_grad=True)
    other = torch.randn(4, outputs, device=device, requires_grad=True)
    sparse = RCSMatrix(values, torch.zeros(rows, dtype=torch.long, device=device), 4)
    actual = sparse @ other
    assert actual.shape == (rows, outputs)
    first = torch.autograd.grad(actual.sum(), (values, other), create_graph=True)
    second = torch.autograd.grad(
        sum(gradient.sum() for gradient in first), (values, other)
    )
    for gradient in (*first, *second):
        assert torch.count_nonzero(gradient) == 0
