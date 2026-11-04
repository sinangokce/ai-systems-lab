# layer_norm.py

import torch


class MyLayerNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.beta = torch.nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)

        mean = x.mean(dim=-1, keepdim=True)
        # population variance => biased variance
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Different hidden dimensions can carry different learned information.
        # Forcing token's representations to remain standardized after normalization
        # would restrict how subsequent layers can use those features.

        # Learnable affine transformation: γ * norm_x + β
        # The learned affine transformation gives the network flexibility
        # while preserving the benefits of normalizing its incoming activations.
        return self.gamma * norm_x + self.beta
