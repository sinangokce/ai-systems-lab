# model.py

import torch

from src.phase_01.week_02.exercise_05.embeddings import TokenPositionEmbedding
from src.phase_01.week_02.exercise_08.layer_norm import MyLayerNorm
from src.phase_01.week_02.exercise_08.transformer_block import TransformerBlock


class GPTModule(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        if config.MODEL_DIM % config.ATTENTION_HEADS != 0:
            raise ValueError(
                f"Model dimension ({config.MODEL_DIM}) "
                f"must be divisible by number of heads ({config.ATTENTION_HEADS})"
            )
        self.CONTEXT_LENGTH = config.CONTEXT_LENGTH

        self.token_embedder = TokenPositionEmbedding(
            config.VOCAB_SIZE, config.MODEL_DIM, config.CONTEXT_LENGTH
        )

        self.transformer_blocks = torch.nn.Sequential(
            # Notice the *
            *[
                TransformerBlock(
                    config.MODEL_DIM,
                    config.ATTENTION_HEADS,
                    config.FFN_EXP_FACTOR,
                )
                for _ in range(config.TRANSFORMER_BLOCKS)
            ]
        )

        self.final_LN = MyLayerNorm(config.MODEL_DIM)
        self.output_projection = torch.nn.Linear(
            config.MODEL_DIM, config.VOCAB_SIZE, bias=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("Input IDs must have shape (B,T).")

        seq_length = input_ids.shape[1]
        if not 1 <= seq_length <= self.CONTEXT_LENGTH:
            raise ValueError(
                f"Sequence length must be between 1 and "
                f"{self.CONTEXT_LENGTH}, got {seq_length}"
            )

        x: torch.Tensor = self.token_embedder(input_ids)
        x = self.transformer_blocks(x)
        x = self.final_LN(x)
        logits = self.output_projection(x)
        return logits
