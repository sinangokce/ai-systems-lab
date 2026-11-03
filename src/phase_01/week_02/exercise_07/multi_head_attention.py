# multi_head_attention.py

import torch


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q,K,V projection matrices (weights)
        self.W_query = torch.nn.Linear(d_model, d_model)
        self.W_key = torch.nn.Linear(d_model, d_model)
        self.W_value = torch.nn.Linear(d_model, d_model)

        # Output projection
        # Not strictly necessary but commonly used in many LLM architectures
        # It allows the model to mix and transform information
        # from different heads after concatenation
        self.W_output = torch.nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        device = x.device
        H = self.num_heads
        D_H = self.head_dim

        # Projected Q tensor
        Q: torch.Tensor = self.W_query(x)  # shape = (B,T,D)
        # Q = | q_00 q_01 q_02 q_03 q_04 q_05 |
        #     | q_10 q_11 q_12 q_13 q_14 q_15 |
        # Projected K tensor
        K: torch.Tensor = self.W_key(x)  # shape = (B,T,D)
        # K = | k_00 k_01 k_02 k_03 k_04 k_05 |
        #     | k_10 k_11 k_12 k_13 k_14 k_15 |
        # Projected V tensor
        V: torch.Tensor = self.W_value(x)  # shape = (B,T,D)
        # V = | v_00 v_01 v_02 v_03 v_04 v_05 |
        #     | v_10 v_11 v_12 v_13 v_14 v_15 |

        # Split into attention heads
        # Each head generates a projection matrix with D_H dimension

        # Insert the extra dimension for number of heads (H)
        Q = Q.view(B, T, H, D_H)
        K = K.view(B, T, H, D_H)
        V = V.view(B, T, H, D_H)
        # Ensure semantic meaning
        Q = Q.transpose(1, 2)  # shape = (B,H,T,D_H)
        # Q_head0 = | q_00 q_01 |, Q_head1 = | q_02 q_03 |, Q_head2 = | q_04 q_05 |
        #           | q_10 q_11 |            | q_12 q_13 |            | q_14 q_15 |
        K = K.transpose(1, 2)  # shape = (B,H,T,D_H)
        # K_head0 = | k_00 k_01 |, K_head1 = | k_02 k_03 |, K_head2 = | k_04 k_05 |
        #           | k_10 k_11 |            | k_12 k_13 |            | k_14 k_15 |
        V = V.transpose(1, 2)  # shape = (B,H,T,D_H)
        # V_head0 = | v_00 v_01 |, V_head1 = | v_02 v_03 |, Q_head2 = | v_04 v_05 |
        #           | v_10 v_11 |            | v_12 v_13 |            | v_14 v_15 |

        # Apply scaled dot-product causal attention
        attn_scores: torch.Tensor = Q @ K.transpose(-2, -1)
        attn_scores = attn_scores / D_H**0.5
        upper_mask = torch.triu(
            torch.ones(
                T,
                T,
                dtype=torch.bool,  # masked_fill requires bool
                device=device,
            ),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(upper_mask, -torch.inf)
        A: torch.Tensor = torch.softmax(attn_scores, dim=-1)  # shape = (B,H,T,T)

        # Merge attention heads
        output: torch.Tensor = (
            A @ V
        )  # shape = (B,H,T,D_H), stride = (H*T*D_H, T*D_H, D_H, 1)
        # Ensure semantic meaning
        output = output.transpose(
            1, 2
        )  # shape = (B,T,H,D_H), stride = (H*T*D_H, D_H, T*D_H , 1)
        # Merge split outputs from each head
        output = output.contiguous().view(B, T, D)  # shape = (B,T,D)
        return self.W_output(output)
