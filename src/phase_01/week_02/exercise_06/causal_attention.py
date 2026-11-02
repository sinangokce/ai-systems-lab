# causal_attention.py

import torch


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single-head scaled dot-product causal attention.

    Args:
        q: (B, T, d_k)
        k: (B, T, d_k)
        v: (B, T, d_v)

    Returns:
        output: (B, T, d_v)
        attention_weights: (B, T, T)
    """
    batch_size, seq_length, d_k = k.shape
    device = k.device

    attention_scores = q @ k.transpose(1, 2)
    attention_scores = attention_scores / d_k**0.5

    upper_mask = torch.triu(
        torch.ones(
            seq_length,
            seq_length,
            dtype=torch.bool,  # masked_fill requires bool
            device=device,
        ),
        diagonal=1,
    )
    # In-place operations can introduce autograd versioning issues
    # when a tensor's previous values are needed elsewhere in a computation graph.
    # They are not inherently faster.
    attention_scores = attention_scores.masked_fill(upper_mask, -torch.inf)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    output = attention_weights @ v
    return output, attention_weights
