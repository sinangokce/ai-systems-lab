# test_13_loss.py

import pytest
import torch

from src.phase_01.week_02.exercise_09 import config
from src.phase_01.week_02.exercise_09.loss import calc_loss_batch
from src.phase_01.week_02.exercise_09.model import GPTModule

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_loss(device):
    B = 2
    T = 4
    V = config.VOCAB_SIZE
    tokens = torch.randint(
        high=V,
        size=(B, T + 1),
        device=device,
    )
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]
    assert torch.equal(input_ids[:, 1:], targets[:, :-1])  # targets are shifted by 1

    model = GPTModule(config)
    model = model.to(device)
    model.train()

    logits = model(input_ids)
    loss = calc_loss_batch(logits, targets)

    assert loss.ndim == 0  # Scalar
    assert torch.isfinite(loss).all()  # Finite
    assert loss >= 0  # Non-negative
    # Matches the official implementation
    torch.testing.assert_close(
        loss,
        torch.nn.functional.cross_entropy(
            logits.transpose(1, 2),  # Pytorch expects B * V * T instead of B * T * V
            targets,
        ),
    )

    loss.backward()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        assert param.grad is not None, f"Parameter ({name}) has no gradient."
        assert torch.isfinite(param.grad).all(), (
            f"Parameter ({name}) has no finite gradients."
        )


@pytest.mark.parametrize("device", DEVICES)
def test_grad_mathematically(device):
    B = 2
    T = 4
    V = config.VOCAB_SIZE
    logits = torch.randn(B, T, V, device=device, requires_grad=True)
    targets = torch.randint(high=V, size=(B, T), device=device)
    loss = calc_loss_batch(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape

    # logit_grads = (prob - target) / (B * T)

    # logit_grads = torch.empty(size=(B, T, V), device=device)
    # probs = torch.nn.functional.softmax(logits, dim=-1)
    # for batch in range(B):
    #     for token in range(T):
    #         vocab_idx = targets[batch, token]
    #         for vocab in range(V):
    #             prob = probs[batch, token, vocab]
    #             diff = prob - 1 if vocab == vocab_idx else prob
    #             logit_grads[batch, token, vocab] = diff / (B * T)

    # The above has a performance problem on CUDA because of if vocab == vocab_idx
    with torch.no_grad():
        probs = torch.nn.functional.softmax(logits, dim=-1)
        one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=V).to(
            logits.dtype
        )

        logit_grads = (probs - one_hot_targets) / (B * T)

    torch.testing.assert_close(logit_grads, logits.grad)


@pytest.mark.parametrize("device", DEVICES)
def test_non_contiguous_tensors(device):
    B, T, V = 2, 4, 32

    logits = torch.randn(T, B, V, device=device).transpose(0, 1)

    assert logits.shape == (B, T, V)
    assert not logits.is_contiguous()

    targets = torch.randint(high=V, size=(B, T), device=device)
    loss_non_contiguous = calc_loss_batch(logits, targets)
    loss_contiguous = calc_loss_batch(logits.contiguous(), targets)

    torch.testing.assert_close(loss_non_contiguous, loss_contiguous)


@pytest.mark.parametrize("device", DEVICES)
def test_loss_numerical_stability(device):
    logits = torch.tensor(
        [
            [
                [1000.0, -1000.0, 0.0],
                [-1000.0, 1000.0, 0.0],
            ]
        ],
        device=device,
        requires_grad=True,
    )

    targets = torch.tensor(
        [[0, 1]],
        device=device,
    )

    loss = calc_loss_batch(logits, targets)

    assert torch.isfinite(loss)

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
