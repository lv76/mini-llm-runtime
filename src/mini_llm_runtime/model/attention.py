import math

import torch
from einops import rearrange
from torch import nn

from mini_llm_runtime.model.functional import linear
from mini_llm_runtime.model.layers import Linear
from mini_llm_runtime.model.rope import RoPE, rope


def scaled_dot_product_attention(q, k, v, mask):
    score = q @ k.transpose(-2, -1)
    d_k = k.size(-1)
    score = score / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(score, dim=-1)
    out = attn @ v
    return out


def multihead_self_attention(
    d_model,
    num_heads,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    in_features,
):
    Q = linear(d_model, d_model, q_proj_weight, in_features)
    K = linear(d_model, d_model, k_proj_weight, in_features)
    V = linear(d_model, d_model, v_proj_weight, in_features)

    q, k, v = (
        rearrange(
            x,
            "batch seq (head d) -> batch head seq d",
            head=num_heads,
        )
        for x in (Q, K, V)
    )

    seq_len = in_features.shape[1]

    mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            device=in_features.device,
            dtype=torch.bool,
        )
    )

    attn_out = scaled_dot_product_attention(q, k, v, mask)
    attn_out = rearrange(
        attn_out,
        "batch head seq d -> batch seq (head d)",
    )

    return linear(d_model, d_model, o_proj_weight, attn_out)


def multihead_self_attention_with_rope(
    d_model,
    num_heads,
    max_seq_len,
    theta,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    in_features,
    token_positions,
):
    Q = linear(d_model, d_model, q_proj_weight, in_features)
    K = linear(d_model, d_model, k_proj_weight, in_features)
    V = linear(d_model, d_model, v_proj_weight, in_features)

    q, k, v = (
        rearrange(
            x,
            "batch seq (head d) -> batch head seq d",
            head=num_heads,
        )
        for x in (Q, K, V)
    )

    head_dim = d_model // num_heads
    q = rope(head_dim, theta, max_seq_len, q, token_positions)
    k = rope(head_dim, theta, max_seq_len, k, token_positions)

    seq_len = in_features.shape[1]
    mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            device=in_features.device,
            dtype=torch.bool,
        ),
        diagonal=0,
    )

    attn_out = scaled_dot_product_attention(q, k, v, mask)
    attn_out = rearrange(
        attn_out,
        "batch head seq d -> batch seq (head d)",
    )

    return linear(d_model, d_model, o_proj_weight, attn_out)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(
        self,
        in_features: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.rope is None:
            return multihead_self_attention(
                self.d_model,
                self.num_heads,
                self.q_proj.weight,
                self.k_proj.weight,
                self.v_proj.weight,
                self.output_proj.weight,
                in_features,
            )

        return multihead_self_attention_with_rope(
            self.d_model,
            self.num_heads,
            self.rope.max_seq_len,
            self.rope.theta,
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.output_proj.weight,
            in_features,
            token_positions,
        )
