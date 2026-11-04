# test_09_layer_norm.py

import pytest
import torch

from src.phase_01.week_02.exercise_08.layer_norm import MyLayerNorm

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_layer_norm(device):
    B, T, D = 2, 5, 12
    x = torch.rand(B, T, D, device=device)

    layer_norm = MyLayerNorm(D).to(device)
    output: torch.Tensor = layer_norm(x)

    assert output.shape == torch.Size([B, T, D])

    # Normalized?
    # Mean = 0 for each token's representation
    torch.testing.assert_close(output.mean(dim=-1), torch.zeros(B, T, device=device))
    # Var ~ 1 for each token's representation
    input_variance = x.var(dim=-1, unbiased=False)
    # Var((X-b)/a) = Var(X) / a^2
    expected_variance = input_variance / (input_variance + layer_norm.eps)
    torch.testing.assert_close(output.var(dim=-1, unbiased=False), expected_variance)


@pytest.mark.parametrize("device", DEVICES)
def test_forward_against_pytorch_reference(device):
    B, T, D = 2, 5, 12
    x = torch.rand(B, T, D, device=device)

    layer_norm = MyLayerNorm(D).to(device)
    custom_output = layer_norm(x)

    reference_model = torch.nn.LayerNorm(D).to(device)
    reference_output = reference_model(x)

    torch.testing.assert_close(custom_output, reference_output)


@pytest.mark.parametrize("device", DEVICES)
def test_backward_against_pytorch_reference(device):
    B, T, D = 2, 5, 12
    x = torch.rand(B, T, D, device=device, requires_grad=True)

    layer_norm = MyLayerNorm(D).to(device)
    custom_output = layer_norm(x)

    reference_model = torch.nn.LayerNorm(D).to(device)
    reference_output = reference_model(x)

    upstream_gradient = torch.randn_like(custom_output)
    # Calculate gradients wrt to x, γ and β
    custom_grads = torch.autograd.grad(
        custom_output, (x, *layer_norm.parameters()), upstream_gradient
    )
    reference_grads = torch.autograd.grad(
        reference_output, (x, *reference_model.parameters()), upstream_gradient
    )

    for custom_grad, reference_grad in zip(custom_grads, reference_grads, strict=True):
        torch.testing.assert_close(
            custom_grad,
            reference_grad,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_numerical_stability(device):
    B, T, D = 2, 5, 12
    # constant input -> var = 0 -> denominator = 0!
    x = torch.full(
        (B, T, D),
        3.0,
        device=device,
        requires_grad=True,
    )

    layer_norm = MyLayerNorm(D).to(device)
    custom_output: torch.Tensor = layer_norm(x)

    assert torch.isfinite(custom_output).all()

    reference_model = torch.nn.LayerNorm(D).to(device)
    reference_output = reference_model(x)

    torch.testing.assert_close(
        custom_output,
        reference_output,
    )

    upstream = torch.randn_like(custom_output)

    custom_grad = torch.autograd.grad(custom_output, x, upstream)[0]

    reference_grad = torch.autograd.grad(reference_output, x, upstream)[0]

    assert torch.isfinite(custom_grad).all()

    torch.testing.assert_close(
        custom_grad,
        reference_grad,
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_noncontiguous_input(device):
    B, T, D = 2, 5, 12
    x_base = torch.rand(
        B, D, T, device=device, requires_grad=True
    )  # Swap T and D for the next tranpose
    x_non_contiguous = x_base.transpose(1, 2)
    x_contiguous = x_non_contiguous.contiguous()

    assert not x_non_contiguous.is_contiguous()
    assert x_non_contiguous.shape == torch.Size([B, T, D])
    assert x_contiguous.is_contiguous()
    assert x_contiguous.shape == torch.Size([B, T, D])

    torch.testing.assert_close(x_non_contiguous, x_contiguous)

    layer_norm = MyLayerNorm(D).to(device)

    output_x_non_contiguous = layer_norm(x_non_contiguous)
    output_x_contiguous = layer_norm(x_contiguous)

    torch.testing.assert_close(output_x_non_contiguous, output_x_contiguous)
