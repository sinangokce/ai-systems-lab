# train.py

from types import SimpleNamespace

import torch

from src.phase_01.week_01.exercise_04.load_dataset import Dataset
from src.phase_01.week_01.exercise_04.tokenizer import CharacterTokenizer
from src.phase_01.week_02.exercise_09 import config
from src.phase_01.week_02.exercise_09.generate import generate
from src.phase_01.week_02.exercise_09.loss import calc_loss_batch
from src.phase_01.week_02.exercise_09.model import GPTModule


def train_single_batch(device):
    """
    Before training on a complete dataset,
    we must verify that the model can actually learn.

    So we repeatedly train on the **same batch**
    and observe whether the loss decreases.

    For a vocabulary containing V, the uniform-prediction cross-entropy is ln(V).
    This is a useful reference.
    As training progresses, the loss is expected to decrease and be lower than ln(V).
    """
    torch.manual_seed(42)

    model = GPTModule(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    B = 2
    T = config.CONTEXT_LENGTH

    # Fixed batch with shifted targets
    tokens = torch.randint(high=config.VOCAB_SIZE, size=(B, T + 1), device=device)
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]

    model.train()

    for step in range(301):
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        logits = model(input_ids)
        # Causal LM loss
        loss = calc_loss_batch(logits, targets)
        # Backward propagation
        loss.backward()
        # Optimizer step
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")


def evaluate_loss(model, inputs, targets, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in zip(inputs, targets, strict=True):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = calc_loss_batch(logits, y)

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    return total_loss / total_tokens


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train_single_batch(device)

    BATCH_SIZE = 8
    SEQ_LENGTH = 64
    STRIDE = SEQ_LENGTH  # Avoid excessive overlap
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.01
    EPOCHS = 20
    TRAIN_SPLIT = 0.7
    FILE_PATH = "src/data/the-verdict.txt"

    with open(FILE_PATH, encoding="utf-8") as f:
        txt = f.read()

    tokenizer = CharacterTokenizer()
    tokenizer.fit(txt)
    token_ids = tokenizer.encode(txt)

    split_idx = int(len(token_ids) * TRAIN_SPLIT)
    train_token_ids = token_ids[:split_idx]
    val_token_ids = token_ids[split_idx:]

    dataset = Dataset()
    # Keep the dataset tensors on CPU initially,
    # transfer individual batches to CUDA during training
    train_inputs, train_targets = dataset.get_batched_inputs_targets(
        train_token_ids, BATCH_SIZE, SEQ_LENGTH, STRIDE, "cpu"
    )
    val_inputs, val_targets = dataset.get_batched_inputs_targets(
        val_token_ids, BATCH_SIZE, SEQ_LENGTH, STRIDE, "cpu"
    )

    model_config = SimpleNamespace(
        VOCAB_SIZE=tokenizer.vocab_size,
        MODEL_DIM=128,
        CONTEXT_LENGTH=64,
        TRANSFORMER_BLOCKS=4,
        ATTENTION_HEADS=4,
        FFN_EXP_FACTOR=4,
    )

    print("Vocabulary:", tokenizer.vocab_size)
    print("Training batches:", train_inputs.shape)
    print("Validation batches:", val_inputs.shape)
    print("Training sequence length:", train_inputs.shape[-1])

    model = GPTModule(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_state = {}
    for epoch in range(EPOCHS):
        model.train()

        # Shuffle batch order
        permutation = torch.randperm(train_inputs.shape[0])
        for batch_idx in permutation:
            train_inputs_batch = train_inputs[batch_idx].to(device)
            train_targets_batch = train_targets[batch_idx].to(device)

            optimizer.zero_grad()

            logits = model(train_inputs_batch)

            loss_batch = calc_loss_batch(logits, train_targets_batch)
            loss_batch.backward()

            optimizer.step()

        avg_train_loss = evaluate_loss(model, train_inputs, train_targets, device)
        avg_val_loss = evaluate_loss(model, val_inputs, val_targets, device)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {
                name: param.detach().cpu().clone()
                for name, param in model.state_dict().items()
            }
        print(
            f"Epoch: {epoch} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Validation: {avg_val_loss:.4f} | "
            f"Best Validation: {best_val_loss:.4f}"
        )

    model.load_state_dict(best_state)

    prompt = txt[:20]

    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    generated_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=500,
        context_length=model_config.CONTEXT_LENGTH,
    )

    generated_text = tokenizer.decode(generated_ids[0].tolist())
    # This can create repetitive sequences because
    # choosing one character changes the context, which influences the next prediction,
    # potentially creating repetitive patterns.
    # There are three contributing factors:
    # 1. Small dataset: The Verdict is a relatively small corpus.
    # 2. Character-level modeling: The model predicts individual characters rather than
    #    words or subwords.
    # 3. Greedy decoding: You always choose the highest-probability character,
    #    sacrificing diversity.
    print("\n--- GENERATED TEXT ---\n")
    print(generated_text)
