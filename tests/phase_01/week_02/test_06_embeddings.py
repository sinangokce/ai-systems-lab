# test_06_embeddings.py

import pytest
import torch

from src.phase_01.week_02.exercise_05.embeddings import TokenPositionEmbedding

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize(
    "batch_size, seq_length, vocab_size, context_length, embedding_dim",
    [
        (4, 32, 128, 64, 128),  # succeeds (seq_length < context_length)
        (4, 32, 128, 32, 128),  # succeeds (seq_length = context_length)
        (4, 32, 128, 16, 128),  # raises error (seq_length > context_length)
        (4, 32, 64, 32, 128),  # raises error (vocab_size is not large enough)
    ],
)
@pytest.mark.parametrize("device", DEVICES)
def test_dimensions(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    context_length: int,
    embedding_dim: int,
    device: str,
):
    positional_embeddings_layer = TokenPositionEmbedding(
        vocab_size, embedding_dim, context_length, device
    )

    token_ids = torch.arange(batch_size * seq_length, device=device).view(
        batch_size, seq_length
    )

    if batch_size * seq_length > vocab_size:
        if device == "cpu":
            with pytest.raises(IndexError):
                positional_embeddings_layer(token_ids)
        else:
            pytest.skip("Out-of-bounds embedding tests are restricted to CPU.")
    else:
        if seq_length > context_length:
            # Context boundary is not respected
            with pytest.raises(ValueError):
                positional_embeddings_layer(token_ids)
        else:
            positional_token_embeddings = positional_embeddings_layer(token_ids)
            assert positional_token_embeddings.shape == torch.Size(
                [batch_size, seq_length, embedding_dim]
            )


@pytest.mark.parametrize("device", DEVICES)
def test_positional_embedding_importance(device: str):

    token_ids = torch.tensor(
        [
            [[0, 1, 0, 0, 0], [0, 1, 1, 0, 0]],  # -> batch 0
            [[0, 1, 0, 0, 0], [0, 1, 1, 0, 1]],  # -> batch 1
        ],
        device=device,
    )

    vocab_size = int(token_ids.max().item() + 1)
    seq_length = token_ids.shape[2]
    context_margin = 4
    context_length = seq_length + context_margin  # context_length >= seq_length
    embedding_dim = 256

    model = TokenPositionEmbedding(vocab_size, embedding_dim, context_length, device)

    batch_results = []
    for batch in token_ids:
        positional_token_embeddings = model(batch)
        for sample_idx in range(token_ids.shape[1]):
            # Same token (0) at different positions (0 and 3)
            # have different positional token embeddings
            with pytest.raises(AssertionError):
                torch.testing.assert_close(
                    positional_token_embeddings[sample_idx][0],
                    positional_token_embeddings[sample_idx][3],
                )
        batch_results.append(positional_token_embeddings)

    # Batch independence
    # Identical token sequences in two batch entries produce identical embeddings
    torch.testing.assert_close(batch_results[0][0], batch_results[1][0])
    with pytest.raises(AssertionError):
        torch.testing.assert_close(batch_results[0][1], batch_results[1][1])


@pytest.mark.parametrize("vocab_size", [2, 3, 4])
@pytest.mark.parametrize("context_length", [3, 4, 5])
@pytest.mark.parametrize("device", DEVICES)
def test_positional_gradient_accumulation(
    vocab_size: int,
    context_length: int,
    device: str,
):

    token_ids = torch.tensor(
        [
            [
                [0, 1, 0],
                [1, 0, 1],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
            ],
        ],
        device=device,
    )
    batch_size = 2
    seq_length = 3
    context_margin = context_length - seq_length
    embedding_dim = 256

    model = TokenPositionEmbedding(vocab_size, embedding_dim, context_length, device)

    for batch in token_ids:
        # Zero out gradients after each batch to avoid inter-batches accumulation
        model.zero_grad(set_to_none=True)

        output = model(batch)
        loss = output.sum()  # a dummy differentiable scalar loss function
        loss.backward()

        # Token embedding
        token_embedder_grads = model.token_embedder.weight.grad
        assert token_embedder_grads is not None
        unique_token_count = {}
        for sample in batch:
            for vocab in sample:
                if vocab.item() in unique_token_count:
                    unique_token_count[vocab.item()] += 1
                else:
                    unique_token_count[vocab.item()] = 1

        for unique_token, occurence in list(unique_token_count.items()):
            # Each used token contributes to the loss
            # as many times as it appears in the whole batch
            torch.testing.assert_close(
                token_embedder_grads[unique_token],
                torch.tensor([float(occurence)] * embedding_dim, device=device),
            )

        for vocab in range(vocab_size):
            if vocab not in unique_token_count:
                # When vocab contains token_ids that are never used during training,
                # their parameters have zero gradients.
                # It means token_embedder weight matrix uselessly consumes memory.
                not_used_token = vocab
                torch.testing.assert_close(
                    token_embedder_grads[not_used_token],
                    torch.tensor([0.0] * embedding_dim, device=device),
                )

        # Position embedding
        position_embedder_grads = model.position_embedder.weight.grad
        assert position_embedder_grads is not None
        # Each used position contributes to the loss as many times as batch size
        torch.testing.assert_close(
            position_embedder_grads[:seq_length],
            torch.full((seq_length, embedding_dim), float(batch_size), device=device),
        )

        # There are context_margin positions that are not used.
        # Their parameters have zero gradients.
        # It means position_embedder weight matrix uselessly consumes memory.
        # Choose the maximum context length according to your model's requirements,
        # ensure training adequately covers the positions you expect at inference.
        torch.testing.assert_close(
            position_embedder_grads[seq_length:],
            torch.zeros((context_margin, embedding_dim), device=device),
        )
