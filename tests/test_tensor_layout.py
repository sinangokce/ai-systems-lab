import torch

from src.fundamentals.tensor_layout import (
    is_tensor_contiguous,
    tensor_shape,
    tensor_stride,
)

x = torch.arange(24).reshape(2, 3, 4)
x_transposed = x.transpose(1, 2)


def test_shape():
    assert tensor_shape(x) == torch.Size([2, 3, 4])
    assert tensor_shape(x_transposed) == torch.Size([2, 4, 3])


def test_stride():
    assert tensor_stride(x) == (12, 4, 1)
    # Stride of x_transposed is not (12,3,1) but (12, 1, 4)
    # Because both x and x_transposed point to the same mem allocation
    assert tensor_stride(x_transposed) == (12, 1, 4)

    x_transposed[0, 0, 0] = 963
    assert x_transposed[0, 0, 0] == x[0, 0, 0]

    # After calling contiguous(), stride of x_transposed is (12, 3, 1) as expected
    # contigous() allocates new memory for x_transposed
    x_transposed_contiguous = x_transposed.contiguous()
    assert tensor_stride(x_transposed_contiguous) == (12, 3, 1)

    x_transposed_contiguous[0, 0, 0] = 111
    assert not x_transposed_contiguous[0, 0, 0] == x[0, 0, 0]
    assert not x_transposed_contiguous[0, 0, 0] == x_transposed[0, 0, 0]


def test_is_contiguous():
    assert is_tensor_contiguous(x)
    assert not is_tensor_contiguous(x_transposed)
