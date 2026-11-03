# test_08_multi_head_attention.py

import pytest
import torch
import torch.nn.functional as F

from src.phase_01.week_02.exercise_07.multi_head_attention import MultiHeadSelfAttention

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_multi_head_attention(device):
    B, T, d_model, num_heads = 2, 5, 12, 3
    model = MultiHeadSelfAttention(d_model, num_heads).to(device)

    x = torch.randn(B, T, d_model, device=device, requires_grad=True)

    y = model(x)

    assert y.shape == (2, 5, 12)
    assert y.device.type == device
    assert torch.isfinite(y).all()

    y.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_forward_against_pytorch_reference(device):
    B, T, d_model, num_heads = 2, 5, 12, 3
    model = MultiHeadSelfAttention(d_model, num_heads).to(device)

    x = torch.randn(B, T, d_model, device=device, requires_grad=True)

    custom_output = model(x)

    H = model.num_heads
    D_H = model.head_dim
    D = model.d_model

    Q = model.W_query(x).view(B, T, H, D_H).transpose(1, 2)
    K = model.W_key(x).view(B, T, H, D_H).transpose(1, 2)
    V = model.W_value(x).view(B, T, H, D_H).transpose(1, 2)

    reference_output = torch.nn.functional.scaled_dot_product_attention(
        Q,
        K,
        V,
        is_causal=True,
        dropout_p=0.0,
    )
    reference_output = reference_output.transpose(1, 2).contiguous().view(B, T, D)
    reference_output = model.W_output(reference_output)

    torch.testing.assert_close(custom_output, reference_output)


@pytest.mark.parametrize("device", DEVICES)
def test_backward_against_pytorch_reference(device):
    B, T, d_model, num_heads = 2, 5, 12, 3
    model = MultiHeadSelfAttention(d_model, num_heads).to(device)

    x = torch.randn(B, T, d_model, device=device, requires_grad=True)

    custom_output = model(x)

    H = model.num_heads
    D_H = model.head_dim
    D = model.d_model

    Q = model.W_query(x).view(B, T, H, D_H).transpose(1, 2)
    K = model.W_key(x).view(B, T, H, D_H).transpose(1, 2)
    V = model.W_value(x).view(B, T, H, D_H).transpose(1, 2)

    reference_output = F.scaled_dot_product_attention(
        Q,
        K,
        V,
        is_causal=True,
        dropout_p=0.0,
    )
    reference_output = reference_output.transpose(1, 2).contiguous().view(B, T, D)
    reference_output = model.W_output(reference_output)

    # compute gradients for each computational graph
    # without accumulating them into the parameters' .grad attributes
    # Notice how we are using the output itself
    # instead of a loss like we did in test_07_causal_attention.py
    # https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
    upstream_gradient = torch.randn_like(
        custom_output
    )  # This operation is called a vector-Jacobian product (VJP).
    custom_grads = torch.autograd.grad(
        custom_output, (x, *model.parameters()), upstream_gradient
    )
    reference_grads = torch.autograd.grad(
        reference_output, (x, *model.parameters()), upstream_gradient
    )

    for custom_grad, reference_grad in zip(custom_grads, reference_grads, strict=True):
        torch.testing.assert_close(
            custom_grad,
            reference_grad,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_causality(device):
    B, T, d_model, num_heads = 2, 5, 12, 3
    model = MultiHeadSelfAttention(d_model, num_heads).to(device)

    x = torch.randn(B, T, d_model, device=device, requires_grad=True)

    output = model(x)

    x_clone = x.clone()
    x_clone[0, 3] = x[0, 3] * 5  # Change the fourth token in batch 1
    x_clone[1, 4] = x[1, 4] / 4  # Change the fifth token in batch 2

    new_output = model(x_clone)

    # Changing a future token must not change
    # the output representation of any earlier token.
    torch.testing.assert_close(output[0, :3], new_output[0, :3])
    torch.testing.assert_close(output[1, :4], new_output[1, :4])

    # Changing a token MAY or may NOT change
    # the output of its and next token's representations.


@pytest.mark.parametrize("device", DEVICES)
def test_noncontiguous_input(device):
    B, T, d_model, num_heads = 2, 5, 12, 3

    model = MultiHeadSelfAttention(d_model, num_heads).to(device)

    x_base = torch.randn(
        B, d_model, T, device=device
    )  # Swap d_model and T for the next tranpose
    x_noncontiguous = x_base.transpose(1, 2)
    x_contiguous = x_noncontiguous.contiguous()

    assert not x_noncontiguous.is_contiguous()
    assert x_contiguous.is_contiguous()
    torch.testing.assert_close(x_noncontiguous, x_contiguous)

    output_x_noncontiguous = model(x_noncontiguous)
    output_x_contiguous = model(x_contiguous)

    assert output_x_noncontiguous.shape == output_x_contiguous.shape
    torch.testing.assert_close(output_x_noncontiguous, output_x_contiguous)
