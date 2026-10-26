# loss.py

import torch


def calc_loss_batch(logits_batch: torch.Tensor, targets_batch: torch.Tensor, device):
    logits_batch = logits_batch.to(device)
    targets_batch = targets_batch.to(device)
    batch_size, seq_length, vocab_size = logits_batch.shape
    loss = torch.nn.functional.cross_entropy(
        logits_batch.view(batch_size * seq_length, vocab_size),
        targets_batch.view(batch_size * seq_length),
    )
    return loss
