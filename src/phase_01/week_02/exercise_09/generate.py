# generate.py

import torch


def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
) -> torch.Tensor:
    """
    Autoregressively generate tokens using greedy decoding.

    Args:
        model: GPT model
        input_ids: (B, T) initial token IDs
        max_new_tokens: Number of tokens to generate
        context_length: Model's maximum context window

    Returns:
        Token IDs of shape (B, T + max_new_tokens)
    """
    model.eval()

    for _ in range(max_new_tokens):
        # Consider only last context_length tokens
        input_ids_cropped = input_ids[:, -context_length:]
        with torch.no_grad():
            outputs = model(input_ids_cropped)
        last_token_logits = outputs[:, -1, :]

        next_token_ids = torch.argmax(
            last_token_logits, dim=-1, keepdim=True
        )  # greedy selection
        input_ids = torch.cat(
            [input_ids, next_token_ids],
            dim=1,
        )
    return input_ids
