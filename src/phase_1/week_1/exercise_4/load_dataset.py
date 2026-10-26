# load_dataset.py
import torch


class Dataset:
    def get_batched_inputs_targets(
        self,
        token_ids: list[int],
        batch_size: int,
        seq_length: int,
        stride: int = 1,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # stride < seq_length: overlapping samples
        # stride > seq_length: skipped tokens
        # stride = seq_length: no overlapping nor skipping
        inputs = []
        targets = []
        all_inputs = []
        all_targets = []

        assert len(token_ids) > seq_length, (
            f"token_ids length ({len(token_ids)}) must be greater than "
            f"seq_length ({seq_length}) "
            "to create at least one input-target pair."
        )

        num_samples = (len(token_ids) - seq_length - 1) // stride + 1
        assert num_samples >= batch_size, (
            "Not enough samples (len(token_ids) - seq_length - 1) // stride + 1 to form "
            f"a batch of size {batch_size}. "
            "Reduce batch_size or provide a longer token_ids sequence."
        )

        for i in range(0, len(token_ids) - seq_length, stride):
            all_inputs.append(token_ids[i : i + seq_length])
            all_targets.append(token_ids[i + 1 : i + 1 + seq_length])

        # all_inputs = [[.,.,...],[.,.,...],...[.,.,...]]
        # all_targets = [[.,.,...],[.,.,...],...[.,.,...]]

        # create batches from all inputs and targets
        for i in range(0, len(all_inputs) - batch_size + 1, batch_size):
            inputs.append(all_inputs[i : i + batch_size])
            targets.append(all_targets[i : i + batch_size])

        # inputs = [
        #     [[...], [...], ..., [...]] -> batch0
        #     [[...], [...], ..., [...]] -> batch1
        #     [[...], [...], ..., [...]] -> batch2
        # ]
        # inputs shape is (number_of_batches, batch_size, seq_length)

        return torch.tensor(inputs, device=device), torch.tensor(targets, device=device)
