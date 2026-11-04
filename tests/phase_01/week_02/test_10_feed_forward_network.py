# test_10_feed_forward_network.py

import pytest
import torch

from src.phase_01.week_02.exercise_08.feed_forward import FeedForward

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_ffn(device):
    B, T, D = 2, 5, 12
    expansion_factor = 4
    x = torch.rand(B, T, D, device=device)

    ffn = FeedForward(D, expansion_factor=expansion_factor).to(device)
    output: torch.Tensor = ffn(x)

    assert output.shape == torch.Size([B, T, D])
    assert output.device.type == device

    for name, p in ffn.first_linear_layer.named_parameters():
        if name == "weight":
            assert p.numel() == D * (D * expansion_factor)
        if name == "bias":
            assert p.numel() == D * expansion_factor

    for name, p in ffn.second_linear_layer.named_parameters():
        if name == "weight":
            assert p.numel() == (D * expansion_factor) * D
        if name == "bias":
            assert p.numel() == D


@pytest.mark.parametrize("device", DEVICES)
def test_backward(device):
    B, T, D = 2, 5, 12
    x = torch.randn(B, T, D, device=device, requires_grad=True)

    ffn = FeedForward(D).to(device)
    output = ffn(x)

    upstream = torch.randn_like(output)
    grads = torch.autograd.grad(output, (x, *ffn.parameters()), upstream)

    for grad in grads:
        assert grad is not None
        assert torch.isfinite(grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_noncontiguous_input(device):
    B, T, D = 2, 5, 12

    ffn = FeedForward(D).to(device)

    x = torch.randn(B, D, T, device=device)  # Swap T and D for the next tranpose
    x_noncontiguous = x.transpose(1, 2)
    x_contiguous = x_noncontiguous.contiguous()

    assert not x_noncontiguous.is_contiguous()

    output_a = ffn(x_noncontiguous)
    output_b = ffn(x_contiguous)

    torch.testing.assert_close(output_a, output_b)
