# test_04_dimensions_broadcasting.py

import pytest
import torch


def test_dimensions():
    x = torch.arange(24).reshape(2, 3, 4)

    # x =
    # [
    #     [
    #       [0, 1, 2, 3],
    #       [4, 5, 6, 7],
    #       [8, 9, 10, 11]
    #     ],
    #
    #     [
    #       [12, 13, 14, 15],
    #       [16, 17, 18, 19],
    #       [20, 21, 22, 23]
    #     ]
    # ]

    reduced_x = x.max(dim=-1, keepdim=True).values
    assert reduced_x.shape == torch.Size([2, 3, 1])
    torch.testing.assert_close(
        reduced_x[:, :, 0],
        torch.tensor(
            [
                [3, 7, 11],
                [15, 19, 23],
            ]
        ),
    )

    reduced_x = x.max(dim=-1, keepdim=False).values
    assert reduced_x.shape == torch.Size([2, 3])
    torch.testing.assert_close(
        reduced_x[:, :],
        torch.tensor(
            [
                [3, 7, 11],
                [15, 19, 23],
            ]
        ),
    )

    reduced_x = x.sum(dim=0, keepdim=True)
    assert reduced_x.shape == torch.Size([1, 3, 4])
    torch.testing.assert_close(
        reduced_x,
        torch.tensor([[[12, 14, 16, 18], [20, 22, 24, 26], [28, 30, 32, 34]]]),
    )

    reduced_x = x.sum(dim=1, keepdim=False)
    assert reduced_x.shape == torch.Size([2, 4])
    torch.testing.assert_close(
        reduced_x, torch.tensor([[12, 15, 18, 21], [48, 51, 54, 57]])
    )

    reduced_x = x.sum(dim=-2, keepdim=True)
    assert reduced_x.shape == torch.Size([2, 1, 4])
    torch.testing.assert_close(
        reduced_x,
        torch.tensor(
            [
                [[12, 15, 18, 21]],
                [[48, 51, 54, 57]],
            ]
        ),
    )

    reduced_x = x.sum(dim=(0, 2), keepdim=True)
    assert reduced_x.shape == torch.Size([1, 3, 1])
    # (dim=(0, 2) is equivalent to 2 chained sums over the same axis
    torch.testing.assert_close(
        reduced_x, x.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)
    )
    torch.testing.assert_close(
        reduced_x, x.sum(dim=2, keepdim=True).sum(dim=0, keepdim=True)
    )
    torch.testing.assert_close(reduced_x, torch.tensor([[[60], [92], [124]]]))

    reduced_x = x.sum(dim=(0, 2), keepdim=False)
    assert reduced_x.shape == torch.Size([3])
    # Notice how the dim=1 in the second sum.
    # Because the first sum has keepdim=False
    torch.testing.assert_close(
        reduced_x, x.sum(dim=0, keepdim=False).sum(dim=1, keepdim=False)
    )
    torch.testing.assert_close(
        reduced_x, x.sum(dim=2, keepdim=False).sum(dim=0, keepdim=False)
    )
    torch.testing.assert_close(reduced_x, torch.tensor([60, 92, 124]))


def test_error_raising_broadcasting():
    x = torch.randn(2, 3, 4)
    x_max = x.max(dim=1, keepdim=False).values

    # The below operation fails because after broadcasting we have
    # x.shape               = (2, 3, 4)
    # x_max.broadcast_shape = (1, 2, 4)
    # The dim1 sizes (3 and 2) don't match
    with pytest.raises(RuntimeError):
        shifted = x - x_max


def test_silently_failing_broadcasting():
    x = torch.tensor([[[0], [2]], [[10], [12]]])
    x_max = x.max(dim=1, keepdim=False).values
    # The below subtraction doesn't raise error because after broadcasting we have
    # x.shape               = (2, 2, 1)
    # x_max.broadcast_shape = (1, 2, 1)
    # Either sizes match or one of the sizes is 1 (dim0)
    shifted = x - x_max

    # Even though the above subtraction doesn't throw any error,
    # it's semantically incorrect.
    # The intention was to subtract the max along the dim1 for each batch sample:
    # First sample:
    # 0 -> 0-2 -> -2
    # 2 -> 2-2 -> 0
    # Second sample:
    # 10 -> 10-12 -> -2
    # 12 -> 12-12 -> 0
    with pytest.raises(AssertionError):
        torch.testing.assert_close(shifted, torch.tensor([[[-2], [0]], [[-2], [0]]]))

    # But the shifted gives instead
    # First sample:
    # -2
    # -10
    # Second sample:
    # 8
    # 0
    torch.testing.assert_close(shifted, torch.tensor([[[-2], [-10]], [[8], [0]]]))

    # The fix is either to use keepdim=True during the reduce operation
    shifted = x - x.max(dim=1, keepdim=True).values
    torch.testing.assert_close(shifted, torch.tensor([[[-2], [0]], [[-2], [0]]]))
    # or unsqueze the x_max
    # The dim=1 because this the dim over which we did max
    shifted = x - x_max.unsqueeze(dim=1)
    torch.testing.assert_close(shifted, torch.tensor([[[-2], [0]], [[-2], [0]]]))
