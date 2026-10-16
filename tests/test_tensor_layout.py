# test_tensor_layout.py

import pytest
import torch

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture(params=DEVICES)
def x(request) -> torch.Tensor:
    return torch.arange(24, device=request.param).reshape(2, 3, 4)


def test_shape(x: torch.Tensor):
    x_transposed = x.transpose(1, 2)

    assert x.shape == torch.Size([2, 3, 4])
    assert x_transposed.shape == torch.Size([2, 4, 3])


def test_stride(x: torch.Tensor):
    x_transposed = x.transpose(1, 2)

    assert x.stride() == (12, 4, 1)

    # Stride of x_transposed is not (12,3,1) but (12, 1, 4)
    assert x_transposed.stride() == (12, 1, 4)
    # Because both x and x_transposed point to the same mem allocation
    assert x.untyped_storage().data_ptr() == x_transposed.untyped_storage().data_ptr()
    # It's alliasing so when you modify x_transposed or x, it affects x or x_transposed
    x_transposed[0, 0, 0] = 963
    assert x_transposed[0, 0, 0] == x[0, 0, 0]

    x_transposed_contiguous = x_transposed.contiguous()
    # After calling contiguous(), stride of x_transposed is (12, 3, 1) as expected
    assert x_transposed_contiguous.stride() == (12, 3, 1)
    # contigous() allocates new memory when not contiguous
    assert (
        not x_transposed_contiguous.untyped_storage().data_ptr()
        == x_transposed.untyped_storage().data_ptr()
    )

    x_transposed_contiguous[0, 0, 0] = 111
    assert not x_transposed_contiguous[0, 0, 0] == x[0, 0, 0]
    assert not x_transposed_contiguous[0, 0, 0] == x_transposed[0, 0, 0]


def test_is_contiguous(x: torch.Tensor):
    x_transposed = x.transpose(1, 2)
    assert x.is_contiguous()
    assert not x_transposed.is_contiguous()


def test_view(x: torch.Tensor):
    x_transposed = x.transpose(1, 2)
    # x_transposed.shape is (2, 4, 3)
    # x_transposed.stride is (12, 1, 4)

    # Calling view creates NEVER a new memory
    a = x.view(2, 12)
    assert a.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()

    # view() can work on non-contiguous tensors too!
    # Insert singleton dims anywhere
    b = x_transposed.view(2, 4, 1, 1, 1, 3)
    # When inserting a singleton dim:
    # * keep the strides of original dims => (12, 1, ?, ?, ?, 4)
    # * starting from the right, new stride is next dim's size * next dim's stride
    # * 3 * 4 = 12 => (12, 1, ?, ?, 12, 4)
    # * 1 * 12 = 12 => (12, 1, ?, 12, 12, 4)
    # * 1 * 12 = 12 => (12, 1, 12, 12, 12, 4)
    assert b.stride() == (12, 1, 12, 12, 12, 4)
    assert b.untyped_storage().data_ptr() == x_transposed.untyped_storage().data_ptr()

    # Split a dimension independently from other dimensions
    c = x_transposed.view(2, 2, 2, 3)
    # When split:
    # * keep the strides of original dims => (12, ?, ?, 4)
    # * put the stride of the split dim for the new rightmost dim => (12, ?, 1, 4)
    # * leftmost stride is calculated as if singleton dim => 2 * 1 = 2 => (12, 2, 1, 4)
    assert c.stride() == (12, 2, 1, 4)
    assert c.untyped_storage().data_ptr() == x_transposed.untyped_storage().data_ptr()

    # Split a dimension and insert singleton dims in the split dimension
    d = x_transposed.view(2, 2, 1, 1, 2, 3)
    # stride => (12, ?, ?, ?, ?, 4) keep the strides of original dims
    # stride => (12, ?, ?, ?, 1, 4) put the stride of the split dim for rightmost dim
    # stride => (12, ?, ?, 2, 1, 4) next dim's size * next dim's stride
    # stride => (12, ?, 2, 2, 1, 4) next dim's size * next dim's stride
    # stride => (12, 2, 2, 2, 1, 4) next dim's size * next dim's stride
    assert d.stride() == (12, 2, 2, 2, 1, 4)
    assert d.untyped_storage().data_ptr() == x_transposed.untyped_storage().data_ptr()

    # view() fails on non-contiguous tensors when not splitting a dim independently
    with pytest.raises(RuntimeError):
        x_transposed.view(2, 2, 3, 2)

    # view() fails on non-contiguous tensors when merging
    with pytest.raises(RuntimeError):
        x_transposed.view(2, 12)
    # Because stride[i] = stride[i+1] * size[i+1] must be satisfied
    # We are merging dim1 of size 4 with dim2 of size 3 so
    # stride[1] = stride[2] * size[2]
    # 1 != 4 * 3

    # Calling reshape() doesn't create a new memory BECAUSE x is contiguous
    e = x.reshape(2, 12)
    assert e.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()

    # Calling reshape() doesn't always copy data on non-contiguous tensors
    f = x_transposed.reshape(2, 2, 2, 3)
    assert f.untyped_storage().data_ptr() == x_transposed.untyped_storage().data_ptr()

    # Calling reshape() creates a copy when view() would fail
    g = x_transposed.reshape(2, 12)
    assert (
        not g.untyped_storage().data_ptr() == x_transposed.untyped_storage().data_ptr()
    )


# Both reshape() and view() are differentiable so autograd works fine with both.

# When profiling GPU workloads,
# treat unexpected reshape() copies as potential performance bottlenecks.

# Use contiguous() explicitly when the algorithm or downstream operation requires
# contiguous storage, but be aware it can allocate and copy.

# reshape() is not the same as permute() or transpose()

# Use transpose() or permute() when you intend to reorder axes,
# rather than merely repartition dimensions.
