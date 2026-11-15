# test_11_transformer_block.py

import pytest
import torch

from src.phase_01.week_02.exercise_08.transformer_block import TransformerBlock


class ZeroModule(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.zeros_like(x)


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_transformer_block(device):
    B, T, D = 2, 5, 12
    num_heads = 4

    x = torch.randn(B, T, D, device=device)
    transformer_block = TransformerBlock(D, num_heads).to(device)

    output: torch.Tensor = transformer_block(x)

    assert output.shape == torch.Size([B, T, D])
    assert output.device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_residual_identity(device):
    B, T, D = 2, 5, 12
    num_heads = 4

    x = torch.randn(B, T, D, device=device)
    transformer_block = TransformerBlock(D, num_heads).to(device)

    transformer_block.mha = ZeroModule()
    transformer_block.ffn = ZeroModule()

    output: torch.Tensor = transformer_block(x)

    # If both transformation branches output zero, output equals input
    torch.testing.assert_close(output, x)


@pytest.mark.parametrize("device", DEVICES)
def test_causality(device):
    B, T, D = 2, 5, 12
    num_heads = 4

    x = torch.randn(B, T, D, device=device)
    transformer_block = TransformerBlock(D, num_heads).to(device)

    output: torch.Tensor = transformer_block(x)

    x_clone = x.clone()
    x_clone[0, 3] *= 4
    x_clone[1, 4] *= 2
    new_output: torch.Tensor = transformer_block(x_clone)

    for batch in range(B):
        if batch == 0:
            torch.testing.assert_close(output[batch, :3], new_output[batch, :3])
        if batch == 1:
            torch.testing.assert_close(output[batch, :4], new_output[batch, :4])


@pytest.mark.parametrize("device", DEVICES)
def test_backward(device):
    B, T, D = 2, 5, 12
    num_heads = 4

    x = torch.randn(B, T, D, device=device, requires_grad=True)
    transformer_block = TransformerBlock(D, num_heads).to(device)

    output: torch.Tensor = transformer_block(x)

    upstream = torch.rand_like(output)
    grads = torch.autograd.grad(output, (x, *transformer_block.parameters()), upstream)

    for grad in grads:
        assert grad is not None
        assert torch.isfinite(grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_noncontiguous_input(device):
    B, T, D = 2, 5, 12
    num_heads = 4

    transformer_block = TransformerBlock(D, num_heads).to(device)

    x = torch.randn(B, D, T, device=device)  # Swap T and D for the next tranpose
    x_noncontiguous = x.transpose(1, 2)
    x_contiguous = x_noncontiguous.contiguous()

    assert not x_noncontiguous.is_contiguous()

    output_a = transformer_block(x_noncontiguous)
    output_b = transformer_block(x_contiguous)

    torch.testing.assert_close(output_a, output_b)
