# test_14_autoregressive_text_generation.py


import pytest
import torch

from src.phase_01.week_02.exercise_09 import config
from src.phase_01.week_02.exercise_09.generate import generate
from src.phase_01.week_02.exercise_09.model import GPTModule

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_txt_generation(device):
    B, T, V = 2, 4, config.VOCAB_SIZE
    CONTEXT_LENGTH = config.CONTEXT_LENGTH
    max_new_tokens = CONTEXT_LENGTH + int(
        torch.randint(high=10, size=(), device=device)
    )

    # Generation beyond context length
    assert max_new_tokens >= CONTEXT_LENGTH

    input_ids = torch.randint(high=V, size=(B, T), device=device)
    model = GPTModule(config).to(device)
    output = generate(model, input_ids, max_new_tokens, CONTEXT_LENGTH)

    assert output.shape == torch.Size([B, T + max_new_tokens])
    assert (output[:, -max_new_tokens:] >= 0).all() and (
        output[:, -max_new_tokens:] < V
    ).all()
