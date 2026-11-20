# loss.py

import torch


def calc_loss_batch(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits:  (B, T, V), raw model outputs
        targets: (B, T), integer token IDs

    Returns:
        Scalar mean cross-entropy loss.
    """
    B, T, V = logits.shape

    loss = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2),  # Pytorch expects B * V * T instead of B * T * V
        targets,
    )
    return loss
