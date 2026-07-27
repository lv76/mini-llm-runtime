import torch
from torch import nn

from mini_llm_runtime.model.attention import (
    MultiHeadSelfAttention,
    multihead_self_attention_with_rope,
)
from mini_llm_runtime.model.functional import embedding, linear, rmsnorm, swiglu
from mini_llm_runtime.model.layers import Embedding, Linear, RMSNorm, SwiGLU
from mini_llm_runtime.model.rope import RoPE


def transformer_block(
    d_model,
    num_heads,
    d_ff,
    max_seq_len,
    theta,
    weights,
    in_features,
):
    attention_input = rmsnorm(
        d_model,
        1e-5,
        weights["ln1.weight"],
        in_features,
    )
    seq_len = in_features.shape[1]

    token_position = torch.arange(seq_len, device=in_features.device)
    attention_output = multihead_self_attention_with_rope(
        d_model,
        num_heads,
        max_seq_len,
        theta,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        attention_input,
        token_position,
    )

    x = in_features + attention_output

    ffn_input = rmsnorm(
        d_model,
        1e-5,
        weights["ln2.weight"],
        x,
    )
    ffn_out = swiglu(
        d_model,
        d_ff,
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
        ffn_input,
    )

    x = x + ffn_out
    return x


def transformer_lm(
    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    rope_theta,
    weights,
    in_indices,
):
    x = embedding(
        vocab_size,
        d_model,
        weights["token_embeddings.weight"],
        in_indices,
    )

    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."

        layer_weights = {
            "attn.q_proj.weight": weights[
                prefix + "attn.q_proj.weight"
            ],
            "attn.k_proj.weight": weights[
                prefix + "attn.k_proj.weight"
            ],
            "attn.v_proj.weight": weights[
                prefix + "attn.v_proj.weight"
            ],
            "attn.output_proj.weight": weights[
                prefix + "attn.output_proj.weight"
            ],
            "ln1.weight": weights[prefix + "ln1.weight"],
            "ffn.w1.weight": weights[prefix + "ffn.w1.weight"],
            "ffn.w2.weight": weights[prefix + "ffn.w2.weight"],
            "ffn.w3.weight": weights[prefix + "ffn.w3.weight"],
            "ln2.weight": weights[prefix + "ln2.weight"],
        }

        x = transformer_block(
            d_model,
            num_heads,
            d_ff,
            context_length,
            rope_theta,
            layer_weights,
            x,
        )

    x = rmsnorm(
        d_model,
        1e-5,
        weights["ln_final.weight"],
        x,
    )

    logits = linear(
        d_model,
        vocab_size,
        weights["lm_head.weight"],
        x,
    )
    return logits


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()

        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            RoPE(d_model // num_heads, theta, max_seq_len),
        )
        self.ln1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        attention_input = self.ln1(in_features)
        seq_len = in_features.shape[1]
        token_positions = torch.arange(
            seq_len,
            device=in_features.device,
        )
        attention_output = self.attn(
            attention_input,
            token_positions,
        )

        x = in_features + attention_output
        ffn_input = self.ln2(x)
        ffn_out = self.ffn(ffn_input)

        return x + ffn_out


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        return self.lm_head(x)
