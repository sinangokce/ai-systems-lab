# transformer_block.py

import torch

from src.phase_01.week_02.exercise_07.multi_head_attention import MultiHeadSelfAttention
from src.phase_01.week_02.exercise_08.feed_forward import FeedForward
from src.phase_01.week_02.exercise_08.layer_norm import MyLayerNorm


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        expansion_factor: int = 4,
    ) -> None:
        super().__init__()
        self.first_layer_norm = MyLayerNorm(d_model)
        self.second_layer_norm = MyLayerNorm(d_model)
        # The two LayerNorm modules should have independent parameters because
        # they normalize representations at different points in the block,
        # and the model benefits from learning separate scales and offsets.

        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)

        shortcut = x
        x = self.first_layer_norm(x)
        x = self.mha(x)

        x = shortcut + x  # The first residual connection

        shortcut = x
        x = self.second_layer_norm(x)
        x = self.ffn(x)

        x = shortcut + x  # The second residual connection

        return x
