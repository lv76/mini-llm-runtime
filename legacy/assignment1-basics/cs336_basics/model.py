import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange

from cs336_basics.nn_utils import softmax

def silu(in_features):
    return in_features * torch.sigmoid(in_features) 

def linear(
        d_in, 
        d_out, 
        weights, 
        in_features
    ):
    out = torch.matmul(in_features, weights.t())

    return out

def embedding(
        vocab_size, 
        d_model, 
        embedding_weight, 
        token_ids
    ):
    assert embedding_weight.shape == (vocab_size, d_model)

    return embedding_weight[token_ids]

def SwiGLU(
    d_model, 
    d_ff, 
    w1_weight,
    w2_weight, 
    w3_weight, 
    in_features
    ):
    x_w1 = torch.matmul(in_features, w1_weight.t())
    x_w3 = torch.matmul(in_features,w3_weight.t())

    silu_x_w1 = silu(x_w1)
    hidden = silu_x_w1 * x_w3

    output = torch.matmul(hidden, w2_weight.t())

    return output

def rmsnorm(d_model, eps, weights, in_features):
    rms = torch.sqrt((in_features ** 2).mean(dim = -1, keepdim = True)) + eps
    
    return weights * (in_features / rms)

def rope(
    d_k, 
    theta, 
    max_seq_len, 
    in_query_or_key, 
    token_positions
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
    x_rot[...,0::2] = x_even_rot
    x_rot[...,1::2] = x_odd_rot

    return x_rot

def scaled_dot_product_attention(q, k, v, mask):
    
    score = q @ k.transpose(-2,-1) 
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
        in_features
):
    Q = in_features @ q_proj_weight.t()
    K = in_features @ k_proj_weight.t()
    V = in_features @ v_proj_weight.t()

    q, k, v = (rearrange(x, "batch seq (head d) -> batch head seq d", head=num_heads) for x in (Q, K, V))

    seq_len = in_features.shape[1]

    mask = torch.tril(
        torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool)
    )

    attn_out = scaled_dot_product_attention(q, k, v, mask)
    attn_out = rearrange(attn_out, "batch head seq d -> batch seq (head d)")

    return attn_out @ o_proj_weight.t()

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
    token_positions
):
    Q = in_features @ q_proj_weight.t()
    K = in_features @ k_proj_weight.t()
    V = in_features @ v_proj_weight.t()

    q, k, v = (rearrange(x ,"batch seq (head d) -> batch head seq d", head = num_heads) for x in (Q, K ,V))

    head_dim = d_model // num_heads
    q = rope(head_dim, theta, max_seq_len, q, token_positions)
    k = rope(head_dim, theta, max_seq_len, k, token_positions)

    seq_len = in_features.shape[1]
    mask = torch.tril(
        torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool),
        diagonal=0
    )

    attn_out = scaled_dot_product_attention(q, k, v, mask)
    attn_out = rearrange(attn_out, "batch head seq d -> batch seq (head d)")

    return attn_out @ o_proj_weight.t()
    
def transformer_block(
    d_model, 
    num_heads,
    d_ff, 
    max_seq_len, 
    theta, 
    weights, 
    in_features
):
    attention_input = rmsnorm(d_model, 1e-5, weights["ln1.weight"], in_features)
    seq_len = in_features.shape[1]

    token_position = torch.arange(seq_len, device = in_features.device)
    attention_output = multihead_self_attention_with_rope(
        d_model, 
        num_heads, 
        max_seq_len, 
        theta, weights["attn.q_proj.weight"], 
        weights["attn.k_proj.weight"], 
        weights["attn.v_proj.weight"], 
        weights["attn.output_proj.weight"], 
        attention_input,
        token_position 
    )

    x = in_features + attention_output

    ffn_input = rmsnorm(d_model, 1e-5, weights["ln2.weight"], x)
    ffn_out = SwiGLU(d_model, d_ff, weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"], ffn_input)

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
    in_indices
):
    x = embedding(vocab_size, d_model, weights["token_embeddings.weight"], in_indices)

    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."

        layer_weights = {
            "attn.q_proj.weight": weights[prefix + "attn.q_proj.weight"],
            "attn.k_proj.weight": weights[prefix + "attn.k_proj.weight"],
            "attn.v_proj.weight": weights[prefix + "attn.v_proj.weight"],
            "attn.output_proj.weight": weights[prefix + "attn.output_proj.weight"],
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
    
    logits = x @ weights["lm_head.weight"].t()
    return logits







    
