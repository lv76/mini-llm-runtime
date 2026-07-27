import torch
from torch import nn


def rope(
    d_k,
    theta,
    max_seq_len,
    in_query_or_key,
    token_positions,
):
    i = torch.arange(d_k // 2)
    freq = 1 / (theta ** (2 * i / d_k))
    all_angles = torch.outer(torch.arange(max_seq_len), freq)
    cos_cache = torch.cos(all_angles)
    sin_cache = torch.sin(all_angles)

    cos = cos_cache[token_positions]
    sin = sin_cache[token_positions]

    while cos.ndim < in_query_or_key.ndim:
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]

    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot = x_even * sin + x_odd * cos

    x_rot = torch.empty_like(in_query_or_key)
    x_rot[..., 0::2] = x_even_rot
    x_rot[..., 1::2] = x_odd_rot

    return x_rot


class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

    def forward(
        self,
        in_query_or_key: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        return rope(
            self.d_k,
            self.theta,
            self.max_seq_len,
            in_query_or_key,
            token_positions,
        )
