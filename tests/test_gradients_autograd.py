# test_gradients_autograd.py

import pytest
import torch

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def initialize_tensors(
    batch_size, in_features, out_features, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(size=(batch_size, in_features), device=device, requires_grad=True)
    w = torch.randn(size=(in_features, out_features), device=device, requires_grad=True)
    b = torch.randn(size=(1, out_features), device=device, requires_grad=True)
    return x, w, b


def manual_backward(
    x: torch.Tensor, w: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_gradient = z
    input_gradient = loss_gradient @ w.T  # dL/dx
    weight_gradient = x.T @ loss_gradient  # dL/dw
    bias_gradient = loss_gradient.sum(dim=0, keepdim=True)  # dL/db
    return input_gradient, weight_gradient, bias_gradient


@pytest.mark.parametrize(
    "batch_size, in_features, out_features",
    [
        (1, 3, 2),  # i,j,k
        (8, 64, 16),
        (17, 7, 11),
        (32, 128, 64),
    ],
)
@pytest.mark.parametrize("device", DEVICES)
def test_gradients(batch_size, in_features, out_features, device):
    x, w, b = initialize_tensors(batch_size, in_features, out_features, device)

    # Forward pass
    z = x @ w + b

    # Loss
    loss = torch.sum(torch.square(z)) / 2

    # Avoid constructing an unnecessary autograd graph
    with torch.no_grad():
        input_gradient, weight_gradient, bias_gradient = manual_backward(x, w, z)

    loss.backward()
    torch.testing.assert_close(input_gradient, x.grad)
    torch.testing.assert_close(weight_gradient, w.grad)
    torch.testing.assert_close(bias_gradient, b.grad)
