import torch
from einops import rearrange

from mini_llm_runtime.model.attention import (
    MultiHeadSelfAttention,
    multihead_self_attention,
    multihead_self_attention_with_rope,
    scaled_dot_product_attention,
)
from mini_llm_runtime.model.rope import RoPE


def test_scaled_dot_product_attention(
    assert_matches_snapshot,
    q,
    k,
    v,
    mask,
):
    output = scaled_dot_product_attention(q, k, v, mask)
    assert_matches_snapshot(output, atol=1e-5)


def test_4d_scaled_dot_product_attention(
    assert_matches_snapshot,
    q,
    k,
    v,
    mask,
):
    q, k, v = (
        rearrange(
            x,
            "(batch head) seq d -> batch head seq d",
            head=2,
        )
        for x in (q, k, v)
    )
    mask = rearrange(
        mask,
        "(batch head) query key -> batch head query key",
        head=2,
    )
    output = scaled_dot_product_attention(q, k, v, mask)
    assert_matches_snapshot(output, atol=1e-5)


def test_multihead_self_attention(
    assert_matches_snapshot,
    in_embeddings,
    d_model,
    n_heads,
    state_dict,
):
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        state_dict[f"layers.0.attn.{name}_proj.weight"]
        for name in ("q", "k", "v", "output")
    ]
    output = multihead_self_attention(
        d_model=d_model,
        num_heads=n_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    assert_matches_snapshot(output, atol=1e-5)

    module = MultiHeadSelfAttention(d_model, n_heads)
    module.load_state_dict(
        {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": o_proj_weight,
        },
        strict=True,
    )
    torch.testing.assert_close(
        module(in_embeddings),
        output,
        rtol=0,
        atol=0,
    )


def test_multihead_self_attention_with_rope(
    assert_matches_snapshot,
    in_embeddings,
    d_model,
    n_heads,
    state_dict,
    n_keys,
    theta,
    pos_ids,
):
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        state_dict[f"layers.0.attn.{name}_proj.weight"]
        for name in ("q", "k", "v", "output")
    ]
    pos_ids = rearrange(pos_ids, "seq -> 1 seq")
    output = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=n_heads,
        max_seq_len=n_keys,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
        token_positions=pos_ids,
    )
    assert_matches_snapshot(output, atol=1e-5)

    module = MultiHeadSelfAttention(
        d_model,
        n_heads,
        RoPE(d_model // n_heads, theta, n_keys),
    )
    module.load_state_dict(
        {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": o_proj_weight,
        },
        strict=True,
    )
    torch.testing.assert_close(
        module(in_embeddings, pos_ids),
        output,
        rtol=0,
        atol=0,
    )
