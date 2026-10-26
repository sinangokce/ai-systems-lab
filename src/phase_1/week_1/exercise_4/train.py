# train.py

import torch
from load_dataset import Dataset
from loss import calc_loss_batch
from model import BigramModel
from tokenizer import CharacterTokenizer

file_path = "/home/singok/Projects/ai-systems-lab/src/data/the-verdict.txt"
batch_size = 8
seq_length = 64
stride = 64
epochs = 200
train_ratio = 0.7
learning_rate = 0.01
weight_decay = 0.1
eval_freq = 10


def train(
    file_path: str,
    batch_size: int,
    seq_length: int,
    stride: int,
    epochs: int,
    train_ratio: float,
    learning_rate: float,
    weight_decay: float,
    eval_freq: int,
) -> tuple[BigramModel, CharacterTokenizer]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(file_path, encoding="utf-8") as f:
        txt_data = f.read()

    split_idx = int(train_ratio * len(txt_data))
    train_txt = txt_data[:split_idx]
    val_txt = txt_data[split_idx:]

    tokenizer = CharacterTokenizer()
    tokenizer.fit(train_txt)
    train_token_ids = tokenizer.encode(train_txt)
    val_token_ids = tokenizer.encode(val_txt)

    dataset = Dataset()
    train_inputs, train_targets = dataset.get_batched_inputs_targets(
        train_token_ids, batch_size, seq_length, stride, device
    )
    val_inputs, val_targets = dataset.get_batched_inputs_targets(
        val_token_ids, batch_size, seq_length, stride, device
    )

    model = BigramModel(tokenizer.vocab_size, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
    )

    train_loss = 0.0
    model.train()
    for epoch in range(epochs):
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
        avg_train_loss = torch.stack(train_losses).mean().item()
        print(f"Train loss is {avg_train_loss} at epoch: {epoch}")
        if epoch % eval_freq == 0:
            model.eval()
            # disable gradient recording
            with torch.no_grad():
                val_losses = []
                for val_inputs_batch, val_targets_batch in zip(
                    val_inputs, val_targets, strict=True
                ):
                    logits_batch = model(val_inputs_batch)
                    val_loss_batch = calc_loss_batch(
                        logits_batch, val_targets_batch, device
                    )
                    val_losses.append(val_loss_batch)

            val_loss = torch.stack(val_losses).mean().item()
            print(f"Validation loss is {val_loss} at epoch: {epoch}")
            model.train()

    return model, tokenizer


@torch.inference_mode()
def generate_txt(
    model: BigramModel, tokenizer: CharacterTokenizer, prompt: str, max_new_tokens: int
) -> str:
    model.eval()
    token_ids = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        logits = model([token_ids[-1]])

        # # greedy decoding
        # next_token = torch.argmax(logits).item()

        # probabilistic sampling
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        token_ids.append(int(next_token))

    return tokenizer.decode(token_ids)


if __name__ == "__main__":
    model, tokenizer = train(
        file_path,
        batch_size,
        seq_length,
        stride,
        epochs,
        train_ratio,
        learning_rate,
        weight_decay,
        eval_freq,
    )

    print(f"Training is completed after {epochs} epochs")
    generated_txt = generate_txt(model, tokenizer, "a", 20)
    print(f"Generated text is {generated_txt}")
