# test_05_bigram_language_model_train.py

import pytest
import torch

from src.phase_1.week_1.exercise_4.load_dataset import Dataset
from src.phase_1.week_1.exercise_4.loss import calc_loss_batch
from src.phase_1.week_1.exercise_4.model import BigramModel
from src.phase_1.week_1.exercise_4.tokenizer import CharacterTokenizer

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def test_char_tokenizer():
    txt = "hello! こんにちは!"
    tokenizer = CharacterTokenizer()
    tokenizer.fit(txt)
    tokens = tokenizer.encode(txt)
    assert txt == tokenizer.decode(tokens)

    txt = "what happens with unseen chars"
    tokens = tokenizer.encode(txt)
    # Character tokenizer maps unseen chars to "UNK"
    assert not txt == tokenizer.decode(tokens)

    short_txt = "A short text"
    short_txt_tokenizer = CharacterTokenizer()
    short_txt_tokenizer.fit(short_txt)
    longer_txt = "A longer text containing more chars"
    longer_txt_tokenizer = CharacterTokenizer()
    longer_txt_tokenizer.fit(longer_txt)
    # Vocab size is dependent on the training set
    assert short_txt_tokenizer.vocab_size < longer_txt_tokenizer.vocab_size


@pytest.mark.parametrize(
    "batch_size, seq_length, stride",
    [
        (4, 3, 2),
        (8, 64, 1),
        (8, 100, 1),  # should raise error ( seq_length <= len(token_ids) )
        (16, 8, 9),  # should raise error ( num_samples < batch_size )
    ],
)
@pytest.mark.parametrize("device", DEVICES)
def test_dataset_loader(batch_size, seq_length, stride, device):
    token_ids = list(range(100))
    dataset = Dataset()

    num_samples = (len(token_ids) - seq_length - 1) // stride + 1
    number_of_batches = num_samples // batch_size

    if seq_length >= len(token_ids):
        with pytest.raises(AssertionError):
            train_inputs, train_targets = dataset.get_batched_inputs_targets(
                token_ids, batch_size, seq_length, stride, device
            )
    else:
        if num_samples < batch_size:
            with pytest.raises(AssertionError):
                train_inputs, train_targets = dataset.get_batched_inputs_targets(
                    token_ids, batch_size, seq_length, stride, device
                )
        else:
            train_inputs, train_targets = dataset.get_batched_inputs_targets(
                token_ids, batch_size, seq_length, stride, device
            )
            assert train_inputs.shape == torch.Size(
                [number_of_batches, batch_size, seq_length]
            )

            assert torch.equal(train_targets, train_inputs + 1)


@pytest.mark.parametrize("vocab_size", [6, 10, 4, 2])
@pytest.mark.parametrize("device", DEVICES)
def test_bigram_language_model(vocab_size, device):
    x = torch.tensor(
        [
            [[0, 1, 2, 3], [4, 3, 2, 2], [5, 5, 5, 1]],
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
        ],
        device=device,
    )

    model = BigramModel(vocab_size, device)
    if vocab_size > x.max().item():
        for batch in x:
            logits = model(batch)
            assert logits.shape == torch.Size(
                [batch.shape[0], batch.shape[1], vocab_size]
            )
    # vocab_sizes smaller than x_max should raise an error
    # because embedding layer expects
    # integer IDs ranging from 0 to vocab_size - 1."
    else:
        if device == "cpu":
            with pytest.raises(IndexError):
                for batch in x:
                    logits = model(batch)
        else:
            # On the terminal you will see a message ending with
            # "Assertion `srcIndex < srcSelectDimSize` failed."
            pytest.skip("Out-of-bounds embedding tests are restricted to CPU.")


@pytest.mark.parametrize("device", DEVICES)
def test_bigram_language_conditional_distribution(device):
    train_txt = "ab" * 1000
    tokenizer = CharacterTokenizer()
    tokenizer.fit(train_txt, allow_unk_token=False)
    train_token_ids = tokenizer.encode(train_txt)

    dataset = Dataset()
    train_inputs, train_targets = dataset.get_batched_inputs_targets(
        train_token_ids, 32, 16, 16, device
    )

    learning_rate = 0.1
    weight_decay = 0
    epochs = 50

    model = BigramModel(tokenizer.vocab_size, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
    )

    train_loss = 0.0
    model.train()
    for _ in range(epochs):
        train_losses = []
        for train_inputs_batch, train_targets_batch in zip(
            train_inputs, train_targets, strict=True
        ):
            optimizer.zero_grad()
            logits_batch = model(
                train_inputs_batch
            )  # logits_batch shape is (batch_size * seq_length * vocab_size)
            train_loss = calc_loss_batch(logits_batch, train_targets_batch, device)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.detach())
    with torch.no_grad():
        P = torch.softmax(
            model.embedding_layer.weight,
            dim=-1,
        )

    # The model's softmaxed weights is actually equivalent to a conditional distribution.
    # In this training example, "b" comes after "a" and vice-versa.
    # The model overfits this example and becomes a conditional distribution
    # P(b|a) ~ 1 and P(a|b) ~ 1
    torch.testing.assert_close(
        P[0][1],
        torch.tensor(1, dtype=torch.float32, device=device),
        atol=0.01,
        rtol=0.0,
    )
    torch.testing.assert_close(
        P[1][0],
        torch.tensor(1, dtype=torch.float32, device=device),
        atol=0.01,
        rtol=0.0,
    )
