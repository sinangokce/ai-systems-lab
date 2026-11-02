# test_07_causal_attention.py

import pytest
import torch
import torch.nn.functional as F

from src.phase_01.week_02.exercise_06.causal_attention import causal_attention

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_causal_attention(device):
    b, t, d_k, d_v = 2, 5, 8, 6
    q = torch.randn(b, t, d_k, dtype=torch.float32, device=device)
    k = torch.randn(b, t, d_k, dtype=torch.float32, device=device)
    v = torch.randn(b, t, d_v, dtype=torch.float32, device=device)

    output, attention_weights = causal_attention(q, k, v)

    # Test shapes
    assert output.shape == torch.Size([b, t, d_v])
    assert attention_weights.shape == torch.Size([b, t, t])

    # Attention weights are normalized
    assert torch.allclose(
        attention_weights.sum(dim=-1), torch.ones_like(attention_weights.sum(dim=-1))
    )


@pytest.mark.parametrize("device", DEVICES)
def test_causal_mask(device):
    b, t, d_k, d_v = 2, 5, 8, 6
    q = torch.randn(b, t, d_k, dtype=torch.float32, device=device)
    k = torch.randn(b, t, d_k, dtype=torch.float32, device=device)
    v = torch.randn(b, t, d_v, dtype=torch.float32, device=device)

    output, attention_weights = causal_attention(q, k, v)

    # Attention weights are causal
    causal_mask = torch.triu(
        torch.ones(b, t, t, dtype=torch.bool, device=device), diagonal=1
    )
    assert torch.all(attention_weights[causal_mask] == 0)

    # The first token attends to itself with probability 1
    for batch in range(b):
        torch.testing.assert_close(
            attention_weights[batch][0][0],
            torch.ones_like(attention_weights[batch][0][0]),
        )


@pytest.mark.parametrize("device", DEVICES)
def test_no_future_information_leakage(device):
    b, t, d_k, d_v = 2, 5, 8, 6

    q = torch.randn(b, t, d_k, dtype=torch.float32, device=device)
    k = torch.randn(b, t, d_k, dtype=torch.float32, device=device)
    v = torch.randn(b, t, d_v, dtype=torch.float32, device=device)

    output, attention_weights = causal_attention(q, k, v)

    k_clone = k.clone()
    v_clone = v.clone()

    # Change values of tokens which come after and including 3rd position
    k_clone[:, 3:] += 10.0
    v_clone[:, 3:] += 20.0

    output_new, attention_weights_new = causal_attention(q, k_clone, v_clone)
    torch.testing.assert_close(
        attention_weights_new[:, :3],
        attention_weights[:, :3],
    )
    torch.testing.assert_close(
        output_new[:, :3],
        output[:, :3],
    )


@pytest.mark.parametrize("device", DEVICES)
def test_forward_against_pytorch_reference(device):
    b, t, d_k, d_v = 2, 5, 8, 6

    q = torch.randn(b, t, d_k, dtype=torch.float32, device=device)
    k = torch.randn(b, t, d_k, dtype=torch.float32, device=device)
    v = torch.randn(b, t, d_v, dtype=torch.float32, device=device)

    output, attention_weights = causal_attention(q, k, v)

    reference_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        dropout_p=0.0,
    )

    # PyTorch's reference function returns the attention output
    torch.testing.assert_close(output, reference_output)


@pytest.mark.parametrize("test_output_index", [0, 2])
@pytest.mark.parametrize("device", DEVICES)
def test_attention_gradients(test_output_index, device):
    b, t, d_k, d_v = 1, 5, 8, 6

    q = torch.randn(b, t, d_k, dtype=torch.float32, device=device, requires_grad=True)
    k = torch.randn(b, t, d_k, dtype=torch.float32, device=device, requires_grad=True)
    v = torch.randn(b, t, d_v, dtype=torch.float32, device=device, requires_grad=True)

    output, _ = causal_attention(q, k, v)

    for batch in range(b):
        loss = output[
            batch, test_output_index, :
        ].sum()  # a dummy differentiable scalar loss function
        loss.backward()

    q_grad = q.grad
    k_grad = k.grad
    v_grad = v.grad

    assert q_grad is not None
    assert k_grad is not None
    assert v_grad is not None

    for batch in range(b):
        # The output of a given token is only dependent to its query vector
        expected_q_grad = torch.zeros_like(q_grad[batch])
        expected_q_grad[test_output_index] = q_grad[batch, test_output_index]
        torch.testing.assert_close(q_grad[batch], expected_q_grad)
        #          |0 0 0......0|
        #          |0 0 0......0|
        # q_grad = |x y z......v| --> test_index
        #          |0 0 0......0|
        #          |0 0 0......0|
        # Moreover, when the loss depends only on the first token's output
        # q_grad is full of zeros
        if test_output_index == 0:
            expected_q_grad = torch.zeros_like(q_grad[batch])
            torch.testing.assert_close(q_grad[batch], expected_q_grad)

        # The output of a given token is only dependent to it's and past's key vectors
        expected_k_grad = torch.zeros_like(k_grad[batch])
        expected_k_grad[: test_output_index + 1] = k_grad[
            batch, : test_output_index + 1, :
        ]
        torch.testing.assert_close(k_grad[batch], expected_k_grad)
        #          |a b c......d|
        #          |e f g......h|
        # k_grad = |x y z......v| --> test_index
        #          |0 0 0......0|
        #          |0 0 0......0|
        # Moreover, when the loss depends only on the first token's output
        # k_grad is full of zeros
        if test_output_index == 0:
            expected_k_grad = torch.zeros_like(k_grad[batch])
            torch.testing.assert_close(k_grad[batch], expected_k_grad)

        # The output of a given token is only dependent to it's and past's value vectors
        expected_v_grad = torch.zeros_like(v_grad[batch])
        expected_v_grad[: test_output_index + 1] = v_grad[
            batch, : test_output_index + 1, :
        ]
        torch.testing.assert_close(v_grad[batch], expected_v_grad)
        #          |a b c......d|
        #          |e f g......h|
        # v_grad = |x y z......v| --> test_index
        #          |0 0 0......0|
        #          |0 0 0......0|
        # Moreover, when the loss depends only on the first token's output
        # v_grad's first row is full of 1's and the rest is full of zeros
        if test_output_index == 0:
            expected_v_grad = torch.zeros_like(v_grad[batch])
            expected_v_grad[0] = torch.ones_like(v_grad[batch][0])
            torch.testing.assert_close(v_grad[batch], expected_v_grad)


@pytest.mark.parametrize("test_output_index", [0, 2])
@pytest.mark.parametrize("device", DEVICES)
def test_backward_against_pytorch_reference(test_output_index, device):
    b, t, d_k, d_v = 1, 5, 8, 6

    q = torch.randn(b, t, d_k, dtype=torch.float32, device=device, requires_grad=True)
    k = torch.randn(b, t, d_k, dtype=torch.float32, device=device, requires_grad=True)
    v = torch.randn(b, t, d_v, dtype=torch.float32, device=device, requires_grad=True)

    output, _ = causal_attention(q, k, v)

    reference_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        dropout_p=0.0,
    )

    loss_custom = output[:, test_output_index, :].sum()
    grads_custom = torch.autograd.grad(loss_custom, (q, k, v))

    loss_reference = reference_output[:, test_output_index, :].sum()
    grads_reference = torch.autograd.grad(loss_reference, (q, k, v))

    for grad_custom, grad_reference in zip(grads_custom, grads_reference, strict=True):
        torch.testing.assert_close(
            grad_custom,
            grad_reference,
            rtol=1e-4,
            atol=1e-5,
        )


@pytest.mark.parametrize("test_output_index", [0, 2])
@pytest.mark.parametrize("device", DEVICES)
def test_non_contiguous_tensors(test_output_index, device):
    b, t, d_k, d_v = 1, 5, 8, 6

    q = torch.randn(
        b, d_k, t, dtype=torch.float32, device=device, requires_grad=True
    ).transpose(1, 2)
    k = torch.randn(
        b, d_k, t, dtype=torch.float32, device=device, requires_grad=True
    ).transpose(1, 2)
    v = torch.randn(
        b, d_v, t, dtype=torch.float32, device=device, requires_grad=True
    ).transpose(1, 2)

    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()

    output, _ = causal_attention(q, k, v)

    reference_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        dropout_p=0.0,
    )

    # Forward pass works with non contiguous tensors
    torch.testing.assert_close(output, reference_output)

    loss_custom = output[:, test_output_index, :].sum()
    grads_custom = torch.autograd.grad(loss_custom, (q, k, v))

    loss_reference = reference_output[:, test_output_index, :].sum()
    grads_reference = torch.autograd.grad(loss_reference, (q, k, v))

    # Backward pass works with non contiguous tensors
    for grad_custom, grad_reference in zip(grads_custom, grads_reference, strict=True):
        torch.testing.assert_close(
            grad_custom,
            grad_reference,
            rtol=1e-4,
            atol=1e-5,
        )
