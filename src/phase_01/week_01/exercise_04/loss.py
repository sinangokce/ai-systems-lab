# loss.py

import torch


def calc_loss_batch(logits_batch: torch.Tensor, targets_batch: torch.Tensor, device):
    logits_batch = logits_batch.to(device)
    targets_batch = targets_batch.to(device)
    batch_size, seq_length, vocab_size = logits_batch.shape

    # The below works because
    # cross-entropy expects:
    #   2D logits (samples, num_classes)
    #   1D targets (samples,)
    # vocab_size is num_classes (classification case)
    # Example:
    # logits = tensor(
    #     [
    #         [2.5, 0.1, 0.3, -1.0],  # Sample 0
    #         [0.2, 3.1, -0.5, 0.4],  # Sample 1
    #         [-0.2, 0.3, 1.8, 0.9],  # Sample 2
    #     ]
    # )
    #
    # targets = tensor([3, 1, 2])
    # In this example, after applying softmax,
    # log-probability[0][3] will be picked for the first sample
    # log-probability[1][1] will be picked for the second sample
    # log-probability[2][2] will be picked for the third sample

    loss = torch.nn.functional.cross_entropy(
        logits_batch.view(batch_size * seq_length, vocab_size),
        targets_batch.view(batch_size * seq_length),
    )
    return loss
