# feed_forward.py

import torch


class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, expansion_factor: int = 4) -> None:
        super().__init__()

        intermediate_dim = d_model * expansion_factor
        self.first_linear_layer = torch.nn.Linear(d_model, intermediate_dim)
        self.second_linear_layer = torch.nn.Linear(intermediate_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN transforms each token's hidden representation independently from others.
        # x: (B, T, D)

        expanded_input: torch.Tensor = self.first_linear_layer(x)
        activations = torch.nn.functional.gelu(expanded_input)
        reduced_activations = self.second_linear_layer(activations)

        # Expansion-Contraction allows for the exploration of a richer representation.
        # Having two linear layers without a nonlinear activation
        # is the same as having one single layer.
        return reduced_activations
