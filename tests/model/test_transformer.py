import torch

from mini_llm_runtime.model.transformer import (
    TransformerBlock,
    TransformerLM,
    transformer_block,
    transformer_lm,
)


def test_transformer_lm(
    assert_matches_snapshot,
    vocab_size,
    n_keys,
    d_model,
    n_layers,
    n_heads,
    d_ff,
    theta,
    state_dict,
    in_indices,
):
    output = transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=state_dict,
        in_indices=in_indices,
    )
    assert_matches_snapshot(output, atol=1e-4, rtol=1e-2)

    module = TransformerLM(
        vocab_size,
        n_keys,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        theta,
    )
    module.load_state_dict(state_dict, strict=True)
    torch.testing.assert_close(
        module(in_indices),
        output,
        rtol=0,
        atol=0,
    )


def test_transformer_lm_truncated_input(
    assert_matches_snapshot,
    vocab_size,
    n_keys,
    d_model,
    n_layers,
    n_heads,
    d_ff,
    theta,
    state_dict,
    in_indices,
):
    in_indices_truncated = in_indices[..., : in_indices.shape[-1] // 2]
    output = transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=state_dict,
        in_indices=in_indices_truncated,
    )
    assert_matches_snapshot(output, atol=1e-4)

    module = TransformerLM(
        vocab_size,
        n_keys,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        theta,
    )
    module.load_state_dict(state_dict, strict=True)
    torch.testing.assert_close(
        module(in_indices_truncated),
        output,
        rtol=0,
        atol=0,
    )


def test_transformer_block(
    assert_matches_snapshot,
    state_dict,
    in_embeddings,
    d_model,
    n_heads,
    d_ff,
    n_keys,
    theta,
):
    block_weights = {
        key.replace("layers.0.", ""): value
        for key, value in state_dict.items()
        if "layers.0." in key
    }
    output = transformer_block(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=n_keys,
        theta=theta,
        weights=block_weights,
        in_features=in_embeddings,
    )
    assert_matches_snapshot(output, atol=1e-4)

    module = TransformerBlock(
        d_model,
        n_heads,
        d_ff,
        n_keys,
        theta,
    )
    module.load_state_dict(block_weights, strict=True)
    torch.testing.assert_close(
        module(in_embeddings),
        output,
        rtol=0,
        atol=0,
    )
