# test_03_numerical_stability_precision.py


import pytest
import torch

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def initialize_logits_targets(device) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor(
        [
            [1000.0, 1001.0, 1002.0],
            [1.0, 2.0, 3.0],
            [-1000.0, -1001.0, -1002.0],
            [1000.0, 0.0, -1000.0],
        ],
        device=device,
    )
    targets = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        device=device,
    )
    return logits, targets


def naive_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # x = x - torch.max(x) is wrong when
    # x = torch.tensor([
    #   [1000.0, 1001.0, 1002.0],
    #   [   1.0,    2.0,    3.0],
    # ])
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def naive_logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logsumexp = torch.log(torch.sum(torch.exp(x), dim=dim))
    return logsumexp


def stable_logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # keepdim is True for correct subtraction
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    logsumexp = (
        torch.log(torch.sum(torch.exp(x_shifted), dim=dim, keepdim=True)) + x_max
    )
    return logsumexp.squeeze(dim)


def cross_entropy_from_logits(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    per_sample_loss = stable_logsumexp(logits) - torch.sum(targets * logits, dim=-1)
    # mean is necessary when working with batch size > 1
    return per_sample_loss.mean()


# No Subtraction of Large Numbers:
# The subtraction happens inside the exponent
def stable_cross_entropy_from_logits(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    target_logits = torch.sum(targets * logits, dim=-1, keepdim=True)
    per_sample_loss = stable_logsumexp(logits - target_logits)
    # mean is necessary when working with batch size > 1
    return per_sample_loss.mean()


@pytest.mark.parametrize("device", DEVICES)
def test_softmax(device):
    logits, _ = initialize_logits_targets(device)
    # For large x values exp(x) outputs a infinity value
    probs = naive_softmax(logits)
    assert not torch.isfinite(probs).all()
    probs = naive_softmax(-logits)
    assert not torch.isfinite(probs).all()

    probs = stable_softmax(logits)
    torch.testing.assert_close(probs, torch.softmax(logits, dim=-1))
    probs = stable_softmax(-logits)
    torch.testing.assert_close(probs, torch.softmax(-logits, dim=-1))


@pytest.mark.parametrize("device", DEVICES)
def test_logsumexp(device):
    logits, _ = initialize_logits_targets(device)
    # For large x values exp(x) outputs a infinity value
    logsumexp = naive_logsumexp(logits)
    assert not torch.isfinite(logsumexp).all()
    logsumexp = naive_logsumexp(-logits)
    assert not torch.isfinite(logsumexp).all()

    logsumexp = stable_logsumexp(logits)
    torch.testing.assert_close(logsumexp, torch.logsumexp(logits, dim=-1))
    logsumexp = stable_logsumexp(-logits)
    torch.testing.assert_close(logsumexp, torch.logsumexp(-logits, dim=-1))


@pytest.mark.parametrize("device", DEVICES)
def test_cross_entropy(device):
    logits, targets = initialize_logits_targets(device)
    cross_entropy_loss = cross_entropy_from_logits(logits, targets)
    torch.testing.assert_close(
        cross_entropy_loss, torch.nn.functional.cross_entropy(logits, targets)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_noncontiguous_dimensions(device):
    x = torch.randn((3, 4, 5), device=device)
    x = x.transpose(0, 1)

    assert not x.is_contiguous()

    for dim in (0, 1, -1):
        torch.testing.assert_close(
            stable_softmax(x, dim),
            torch.softmax(x, dim),
        )

        torch.testing.assert_close(
            stable_logsumexp(x, dim),
            torch.logsumexp(x, dim),
        )


@pytest.mark.parametrize("device", DEVICES)
def test_cross_entropy_precision(device):
    logits = torch.tensor([[1_000_000.0, 999_999.0]], device=device)
    targets = torch.tensor([[1.0, 0.0]], device=device)

    actual = cross_entropy_from_logits(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, targets)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(actual, expected)

    actual = stable_cross_entropy_from_logits(logits, targets)

    torch.testing.assert_close(actual, expected)
