# model.py

import torch


class BigramModel(torch.nn.Module):
    def __init__(self, vocab_size, device) -> None:
        super().__init__()
        self.device = device
        self.embedding_layer = torch.nn.Embedding(
            vocab_size, vocab_size, device=self.device
        )

    # First argument (num_embeddings): The size of your vocabulary.
    # This tells the layer: "Expect integer IDs ranging from 0 to vocab_size - 1."
    # Second argument (embedding_dim): The size of the vector output for each token.

    # Logit is of shape (batch_size, seq_length, vocab_size)

    # Increasing seq_length has no impact to learn longer-range dependencies.

    # Embedding layer behaves like lookup
    # If batch contains only token IDs 2, 2, 5, and 5 as inputs,
    # only rows 2 and 5 are selected by the embedding lookup,
    # so only those rows can receive nonzero loss gradients.
    def forward(self, x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)
        return self.embedding_layer(x)
