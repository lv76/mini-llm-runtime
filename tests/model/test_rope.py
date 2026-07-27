import torch

from mini_llm_runtime.model.rope import RoPE, rope


def test_rope(
    assert_matches_snapshot,
    in_embeddings,
    d_model,
    theta,
    n_queries,
    pos_ids,
):
    output = rope(
        d_model,
        theta=theta,
        max_seq_len=n_queries,
        in_query_or_key=in_embeddings,
        token_positions=pos_ids,
    )
    assert_matches_snapshot(output, atol=1e-5)

    module = RoPE(d_model, theta, n_queries)
    torch.testing.assert_close(
        module(in_embeddings, pos_ids),
        output,
        rtol=0,
        atol=0,
    )
