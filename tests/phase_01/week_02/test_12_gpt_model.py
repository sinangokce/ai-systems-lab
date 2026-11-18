# test_12_gpt_model.py

import pytest
import torch

from src.phase_01.week_02.exercise_09 import config
from src.phase_01.week_02.exercise_09.model import GPTModule

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_gpt_variable_seq_length(device):
    B = 2
    model = GPTModule(config).to(device)
    model.eval()
    # TORCH_DISABLE_NATIVE_JIT=1 is required for T=1
    for T in range(1, config.CONTEXT_LENGTH + 1):
        input_ids = torch.randint(high=config.VOCAB_SIZE, size=(B, T), device=device)
        with torch.no_grad():
            output = model(input_ids)

        assert output.shape == torch.Size([B, T, config.VOCAB_SIZE])
        assert torch.isfinite(output).all()


@pytest.mark.parametrize("device", DEVICES)
def test_gpt_invalid_context_length(device):
    B = 2
    model = GPTModule(config).to(device)
    model.eval()

    input_ids = torch.randint(
        high=config.VOCAB_SIZE, size=(B, config.CONTEXT_LENGTH + 1), device=device
    )

    with pytest.raises(ValueError):
        model(input_ids)


@pytest.mark.parametrize("device", DEVICES)
def test_gpt_blocks_independent(device):
    model = GPTModule(config).to(device)

    blocks = model.transformer_blocks
    assert config.TRANSFORMER_BLOCKS == len(blocks)

    for i in range(config.TRANSFORMER_BLOCKS):
        for j in range(i + 1, config.TRANSFORMER_BLOCKS):
            assert blocks[i] != blocks[j]

            for p1, p2 in zip(
                blocks[i].parameters(), blocks[j].parameters(), strict=True
            ):
                assert p1 is not p2
                assert p1.data_ptr() != p2.data_ptr()


@pytest.mark.parametrize("device", DEVICES)
def test_gpt_causality(device):
    x1 = torch.tensor([3, 5, 7, 9], dtype=torch.int, device=device)
    x2 = torch.tensor([3, 5, 7, 2], dtype=torch.int, device=device)
    input_ids = torch.stack([x1, x2])

    model = GPTModule(config).to(device)
    model.eval()

    with torch.no_grad():
        output = model(input_ids)

    x1_output = output[0]
    x2_output = output[1]
    torch.testing.assert_close(x1_output[:3, :], x2_output[:3, :])


@pytest.mark.parametrize("device", DEVICES)
def test_backward_propagation(device):
    B = 2
    T = 4
    input_ids = torch.randint(
        high=config.VOCAB_SIZE,
        size=(B, T),
        device=device,
    )

    model = GPTModule(config).to(device)
    model.eval()

    output = model(input_ids)

    loss = output.square().mean()
    loss.backward()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        assert param.grad is not None, f"Parameter ({name}) has no gradient."
        assert torch.isfinite(param.grad).all(), (
            f"Parameter ({name}) has no finite gradients."
        )


@pytest.mark.parametrize("device", DEVICES)
def test_model_param_count(device):
    model = GPTModule(config).to(device)

    # Positional Token Embedding:
    # -->Token embedder: V * D
    # -->Position embedder: context_len * D

    # Transformer Block:
    # -->LayerNorm:
    # ---->gamma: D
    # ---->beta: D
    # -->MHA:
    # ---->W_Q: D * D + D
    # ---->W_K: D * D + D
    # ---->W_V: D * D + D
    # ---->W_O: D * D + D
    # -->LayerNorm:
    # ---->gamma: D
    # ---->beta: D
    # -->FeedForwardNetwork:
    # ---->expansion: D * (D * EXP_FACTOR) + D
    # ---->contraction: (D * EXP_FACTOR) * D + D

    # LayerNorm:
    # gamma: D
    # beta: D

    # Output Projection: D*V (no bias)

    V = config.VOCAB_SIZE
    D = config.MODEL_DIM
    context_len = config.CONTEXT_LENGTH
    exp_factor = config.FFN_EXP_FACTOR

    pos_embed_param_count = (V * D) + context_len * D
    ln_param_count = D + D
    mha_param_count = 4 * (D * D + D)
    ffn_param_count = (D * (D * exp_factor) + D * exp_factor) + (
        (D * exp_factor) * D + D
    )
    transformer_block_param_count = (
        ln_param_count + mha_param_count + ln_param_count + ffn_param_count
    )
    output_proj_param_count = D * V

    total_param_count = (
        pos_embed_param_count
        + transformer_block_param_count * (config.TRANSFORMER_BLOCKS)
        + ln_param_count
        + output_proj_param_count
    )

    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         print(name, "bias =", module.bias is not None)

    assert total_param_count == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("device", DEVICES)
def test_noncontiguous_input(device):
    B = 2
    T = 4
    input_ids = torch.randint(
        high=config.VOCAB_SIZE,
        size=(T, B),
        device=device,
    )

    input_ids_noncontiguous = input_ids.transpose(0, 1)
    input_ids_contiguous = input_ids_noncontiguous.contiguous()

    assert not input_ids_noncontiguous.is_contiguous()
    assert input_ids.is_contiguous()

    model = GPTModule(config).to(device)
    model.eval()

    with torch.no_grad():
        output_noncontiguous = model(input_ids_noncontiguous)
        output_contiguous = model(input_ids_contiguous)

    torch.testing.assert_close(output_noncontiguous, output_contiguous)
