# embeddings.py

import torch


class TokenPositionEmbedding(torch.nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, context_length: int, device: str
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.device = device

        self.token_embedder = torch.nn.Embedding(
            self.vocab_size, self.embedding_dim, device=self.device
        )

        self.position_embedder = torch.nn.Embedding(
            self.context_length, self.embedding_dim, device=self.device
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids.shape == (B, T)

        _, seq_length = token_ids.shape
        # Context boundary must be respected
        if seq_length > self.context_length:
            raise ValueError
        # assert seq_length <= self.context_length, (
        #     f"Number of tokens ({seq_length}) can't exceed embedding layer's "
        #     f"context_length ({self.context_length})"
        # )

        # token_embeddings.shape == (B, seq_length, embedding_dim)
        token_embeddings = self.token_embedder(token_ids)

        # Prefer device=token_ids.device to device=self.device
        # It's more robust when you first initialize the model on CPU or GPU and
        # you decide to move it to GPU or CPU.
        # With device=token_ids.device, it derives the device from the input
        position_indices = torch.arange(seq_length, device=token_ids.device)
        # position_embeddings.shape == (seq_length, embedding_dim)
        position_embeddings = self.position_embedder(position_indices)

        # token_embeddings and position_embeddings can be added thanks to broadcasting

        # Return shape: (B, T, embedding_dim)
        return token_embeddings + position_embeddings
