import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from mini_llm_runtime.model.qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2MLP,
    Qwen2QKVProjection,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    TinyQwen2ForCausalLM,
    TinyQwen2Model,
    TinyQwen2Config,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)
from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache


def test_tiny_qwen2_config_defaults():
    config = TinyQwen2Config()

    assert config.vocab_size == 128
    assert config.num_hidden_layers == 2
    assert config.num_attention_heads == 4
    assert config.num_key_value_heads == 2
    assert config.head_dim == 16


def test_qwen2_5_0_5b_config_from_dict():
    config_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "qwen2_5_0_5b_config.json"
    )

    with config_path.open(encoding="utf-8") as file:
        config_dict = json.load(file)

    config = TinyQwen2Config.from_dict(config_dict)

    assert config.vocab_size == 151936
    assert config.hidden_size == 896
    assert config.intermediate_size == 4864
    assert config.num_hidden_layers == 24
    assert config.num_attention_heads == 14
    assert config.num_key_value_heads == 2
    assert config.head_dim == 64
    assert config.max_position_embeddings == 32768
    assert config.rope_theta == 1_000_000.0
    assert config.rms_norm_eps == 1e-6
    assert config.tie_word_embeddings is True
    assert config.bos_token_id == 151643
    assert config.eos_token_id == 151643


def test_qwen2_lm_head_does_not_tie_embedding_weights_by_default():
    model = TinyQwen2ForCausalLM(TinyQwen2Config())

    assert model.lm_head.weight is not model.model.embed_tokens.weight
    assert (
        model.lm_head.weight.data_ptr()
        != model.model.embed_tokens.weight.data_ptr()
    )


def test_qwen2_lm_head_ties_embedding_weights_when_configured():
    model = TinyQwen2ForCausalLM(
        TinyQwen2Config(tie_word_embeddings=True)
    )

    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert (
        model.lm_head.weight.data_ptr()
        == model.model.embed_tokens.weight.data_ptr()
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_hidden_layers": 0}, "num_hidden_layers"),
        (
            {"hidden_size": 62, "num_attention_heads": 4},
            "hidden_size must be divisible",
        ),
        (
            {"num_attention_heads": 4, "num_key_value_heads": 3},
            "num_attention_heads must be divisible",
        ),
        (
            {"hidden_size": 12, "num_attention_heads": 4},
            "head_dim must be even",
        ),
    ],
)
def test_tiny_qwen2_config_rejects_invalid_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        TinyQwen2Config(**overrides)


def rms_norm_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    statistics_dtype = (
        torch.float32
        if input_dtype in (torch.float16, torch.bfloat16)
        else input_dtype
    )
    hidden_states_for_norm = hidden_states.to(statistics_dtype)
    variance = hidden_states_for_norm.square().mean(dim=-1, keepdim=True)
    output = hidden_states_for_norm * torch.rsqrt(variance + eps)
    return (output * weight.to(statistics_dtype)).to(input_dtype)


def test_qwen2_rms_norm_matches_reference():
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 3, 16)
    module = Qwen2RMSNorm(hidden_size=16, eps=1e-6)
    with torch.no_grad():
        module.weight.copy_(torch.randn(16))

    expected = rms_norm_reference(
        hidden_states,
        module.weight,
        module.variance_epsilon,
    )

    torch.testing.assert_close(module(hidden_states), expected)


@pytest.mark.parametrize("shape", [(16,), (2, 16), (2, 3, 16)])
def test_qwen2_rms_norm_preserves_shape(shape):
    hidden_states = torch.randn(shape)

    output = Qwen2RMSNorm(hidden_size=16)(hidden_states)

    assert output.shape == hidden_states.shape


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_qwen2_rms_norm_preserves_dtype_and_uses_float32_statistics(dtype):
    hidden_states = torch.tensor(
        [[[0.01, 100.0, -0.02, -50.0]]],
        dtype=dtype,
    )
    module = Qwen2RMSNorm(hidden_size=4)

    output = module(hidden_states)
    expected = rms_norm_reference(
        hidden_states,
        module.weight,
        module.variance_epsilon,
    )

    assert output.dtype == dtype
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_qwen2_rms_norm_gradients_match_reference():
    torch.manual_seed(1)
    actual_input = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)
    module = Qwen2RMSNorm(hidden_size=8, eps=1e-6).double()
    expected_weight = module.weight.detach().clone().requires_grad_(True)
    gradient = torch.randn_like(actual_input)

    actual_output = module(actual_input)
    expected_output = rms_norm_reference(
        expected_input,
        expected_weight,
        module.variance_epsilon,
    )
    actual_output.backward(gradient)
    expected_output.backward(gradient)

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_input.grad, expected_input.grad)
    torch.testing.assert_close(module.weight.grad, expected_weight.grad)


def test_qwen2_rms_norm_weight_starts_at_one():
    module = Qwen2RMSNorm(hidden_size=16)

    torch.testing.assert_close(module.weight, torch.ones(16))


def test_rotate_half_uses_contiguous_halves():
    hidden_states = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    output = rotate_half(hidden_states)

    torch.testing.assert_close(output, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def qwen2_rope_reference(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    head_dim = hidden_states.shape[-1]
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    frequencies = position_ids.float().unsqueeze(-1) * inv_freq
    angles = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(-2)
    first_half, second_half = hidden_states.float().chunk(2, dim=-1)
    rotated_half = torch.cat((-second_half, first_half), dim=-1)
    output = hidden_states.float() * angles.cos() + rotated_half * angles.sin()
    return output.to(hidden_states.dtype)


def test_qwen2_rope_matches_reference_with_gqa_and_non_contiguous_positions():
    torch.manual_seed(2)
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 2, 8)
    position_ids = torch.tensor([[0, 2, 7], [1, 4, 9]])
    rope = Qwen2RotaryEmbedding(head_dim=8, theta=10_000.0)

    cos, sin = rope(query, position_ids)
    actual_query, actual_key = apply_rotary_pos_emb(query, key, cos, sin)

    torch.testing.assert_close(
        actual_query,
        qwen2_rope_reference(query, position_ids, theta=10_000.0),
    )
    torch.testing.assert_close(
        actual_key,
        qwen2_rope_reference(key, position_ids, theta=10_000.0),
    )


def test_qwen2_rope_position_zero_is_identity_and_does_not_modify_inputs():
    torch.manual_seed(3)
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 2, 8)
    original_query = query.clone()
    original_key = key.clone()
    rope = Qwen2RotaryEmbedding(head_dim=8)

    cos, sin = rope(query, torch.zeros(2, dtype=torch.long))
    rotated_query, rotated_key = apply_rotary_pos_emb(query, key, cos, sin)

    torch.testing.assert_close(rotated_query, original_query)
    torch.testing.assert_close(rotated_key, original_key)
    torch.testing.assert_close(query, original_query)
    torch.testing.assert_close(key, original_key)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_qwen2_rope_preserves_shape_dtype_and_head_norm(dtype):
    torch.manual_seed(4)
    query = torch.randn(2, 3, 4, 8).to(dtype)
    key = torch.randn(2, 3, 2, 8).to(dtype)
    rope = Qwen2RotaryEmbedding(head_dim=8)

    cos, sin = rope(query, torch.tensor([[0, 3, 8], [1, 5, 13]]))
    rotated_query, rotated_key = apply_rotary_pos_emb(query, key, cos, sin)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert rotated_query.dtype == dtype
    assert rotated_key.dtype == dtype
    torch.testing.assert_close(
        rotated_query.float().norm(dim=-1),
        query.float().norm(dim=-1),
        rtol=2e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(
        rotated_key.float().norm(dim=-1),
        key.float().norm(dim=-1),
        rtol=2e-3,
        atol=2e-3,
    )


def test_qwen2_rope_gradients_flow_through_query_and_key():
    torch.manual_seed(5)
    query = torch.randn(2, 3, 4, 8, requires_grad=True)
    key = torch.randn(2, 3, 2, 8, requires_grad=True)
    rope = Qwen2RotaryEmbedding(head_dim=8)
    cos, sin = rope(query, torch.arange(3).expand(2, -1))

    rotated_query, rotated_key = apply_rotary_pos_emb(query, key, cos, sin)
    (rotated_query.square().sum() + rotated_key.square().sum()).backward()

    assert query.grad is not None
    assert key.grad is not None
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(key.grad).all()


@pytest.mark.parametrize("head_dim", [0, 7])
def test_qwen2_rope_rejects_invalid_head_dim(head_dim):
    with pytest.raises(ValueError, match="positive even"):
        Qwen2RotaryEmbedding(head_dim=head_dim)


def test_qwen2_qkv_projection_parameter_shapes():
    config = TinyQwen2Config()
    module = Qwen2QKVProjection(config)
    query_size = config.num_attention_heads * config.head_dim
    key_value_size = config.num_key_value_heads * config.head_dim

    assert module.q_proj.weight.shape == (query_size, config.hidden_size)
    assert module.k_proj.weight.shape == (key_value_size, config.hidden_size)
    assert module.v_proj.weight.shape == (key_value_size, config.hidden_size)
    assert module.q_proj.bias.shape == (query_size,)
    assert module.k_proj.bias.shape == (key_value_size,)
    assert module.v_proj.bias.shape == (key_value_size,)


def test_qwen2_qkv_projection_matches_linear_reference():
    torch.manual_seed(6)
    config = TinyQwen2Config()
    module = Qwen2QKVProjection(config)
    hidden_states = torch.randn(2, 3, config.hidden_size)

    query, key, value = module(hidden_states)
    leading_shape = hidden_states.shape[:-1]
    expected_query = F.linear(
        hidden_states,
        module.q_proj.weight,
        module.q_proj.bias,
    ).reshape(*leading_shape, config.num_attention_heads, config.head_dim)
    expected_key = F.linear(
        hidden_states,
        module.k_proj.weight,
        module.k_proj.bias,
    ).reshape(*leading_shape, config.num_key_value_heads, config.head_dim)
    expected_value = F.linear(
        hidden_states,
        module.v_proj.weight,
        module.v_proj.bias,
    ).reshape(*leading_shape, config.num_key_value_heads, config.head_dim)

    torch.testing.assert_close(query, expected_query)
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)


@pytest.mark.parametrize("leading_shape", [(), (5,), (2, 3)])
def test_qwen2_qkv_projection_preserves_leading_dimensions(leading_shape):
    config = TinyQwen2Config()
    module = Qwen2QKVProjection(config)
    hidden_states = torch.randn(*leading_shape, config.hidden_size)

    query, key, value = module(hidden_states)

    assert query.shape == leading_shape + (
        config.num_attention_heads,
        config.head_dim,
    )
    assert key.shape == leading_shape + (
        config.num_key_value_heads,
        config.head_dim,
    )
    assert value.shape == leading_shape + (
        config.num_key_value_heads,
        config.head_dim,
    )


def test_qwen2_qkv_projection_accepts_non_contiguous_input():
    config = TinyQwen2Config()
    module = Qwen2QKVProjection(config)
    hidden_states = torch.randn(2, config.hidden_size, 3).transpose(1, 2)
    assert not hidden_states.is_contiguous()

    query, key, value = module(hidden_states)

    assert query.shape == (2, 3, config.num_attention_heads, config.head_dim)
    assert key.shape == (2, 3, config.num_key_value_heads, config.head_dim)
    assert value.shape == (2, 3, config.num_key_value_heads, config.head_dim)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_qwen2_qkv_projection_preserves_dtype(dtype):
    config = TinyQwen2Config()
    module = Qwen2QKVProjection(config).to(dtype)
    hidden_states = torch.randn(2, 3, config.hidden_size, dtype=dtype)

    query, key, value = module(hidden_states)

    assert query.dtype == dtype
    assert key.dtype == dtype
    assert value.dtype == dtype


def test_qwen2_qkv_projection_gradients_are_finite():
    torch.manual_seed(7)
    config = TinyQwen2Config()
    module = Qwen2QKVProjection(config)
    hidden_states = torch.randn(2, 3, config.hidden_size, requires_grad=True)

    query, key, value = module(hidden_states)
    loss = query.square().mean() + key.square().mean() + value.square().mean()
    loss.backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    for projection in (module.q_proj, module.k_proj, module.v_proj):
        assert projection.weight.grad is not None
        assert projection.bias.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert torch.isfinite(projection.bias.grad).all()


def test_qwen2_qkv_projection_rejects_wrong_hidden_size():
    config = TinyQwen2Config()
    module = Qwen2QKVProjection(config)
    hidden_states = torch.randn(2, 3, config.hidden_size + 1)

    with pytest.raises(ValueError, match="input shape"):
        module(hidden_states)


def test_repeat_kv_repeats_each_head_contiguously():
    hidden_states = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]]
    )

    output = repeat_kv(hidden_states, num_key_value_groups=2)

    expected = torch.tensor(
        [[[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]]
    )
    torch.testing.assert_close(output, expected)


def test_repeat_kv_preserves_leading_shape_dtype_and_device():
    hidden_states = torch.randn(2, 3, 2, 8, dtype=torch.float64)

    output = repeat_kv(hidden_states, num_key_value_groups=3)

    assert output.shape == (2, 3, 6, 8)
    assert output.dtype == hidden_states.dtype
    assert output.device == hidden_states.device


def test_repeat_kv_one_group_returns_original_tensor():
    hidden_states = torch.randn(2, 3, 2, 8)

    output = repeat_kv(hidden_states, num_key_value_groups=1)

    assert output is hidden_states


@pytest.mark.parametrize("num_key_value_groups", [0, -1])
def test_repeat_kv_rejects_non_positive_group_count(num_key_value_groups):
    with pytest.raises(ValueError):
        repeat_kv(torch.randn(2, 8), num_key_value_groups)


@pytest.mark.parametrize("shape", [(), (8,)])
def test_repeat_kv_rejects_tensors_without_head_dimensions(shape):
    with pytest.raises(ValueError):
        repeat_kv(torch.randn(shape), num_key_value_groups=2)


def test_repeat_kv_gradient_accumulates_across_repeated_heads():
    hidden_states = torch.randn(2, 3, 2, 8, requires_grad=True)

    repeat_kv(hidden_states, num_key_value_groups=3).sum().backward()

    torch.testing.assert_close(
        hidden_states.grad,
        torch.full_like(hidden_states, 3.0),
    )


def qwen2_attention_reference(
    module: Qwen2Attention,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    query = F.linear(
        hidden_states,
        module.qkv_proj.q_proj.weight,
        module.qkv_proj.q_proj.bias,
    ).reshape(
        batch_size,
        seq_len,
        module.num_attention_heads,
        module.head_dim,
    )
    key = F.linear(
        hidden_states,
        module.qkv_proj.k_proj.weight,
        module.qkv_proj.k_proj.bias,
    ).reshape(
        batch_size,
        seq_len,
        module.num_key_value_heads,
        module.head_dim,
    )
    value = F.linear(
        hidden_states,
        module.qkv_proj.v_proj.weight,
        module.qkv_proj.v_proj.bias,
    ).reshape(
        batch_size,
        seq_len,
        module.num_key_value_heads,
        module.head_dim,
    )
    query = qwen2_rope_reference(query, position_ids, module.rotary_embedding.theta)
    key = qwen2_rope_reference(key, position_ids, module.rotary_embedding.theta)
    key = torch.repeat_interleave(key, module.num_key_value_groups, dim=-2)
    value = torch.repeat_interleave(value, module.num_key_value_groups, dim=-2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    scores = query @ key.transpose(-2, -1) * module.scale
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device)
    )
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = (weights @ value).transpose(1, 2).reshape(
        batch_size,
        seq_len,
        module.hidden_size,
    )
    return F.linear(output, module.o_proj.weight, bias=None)


def test_qwen2_attention_initialization():
    config = TinyQwen2Config()
    module = Qwen2Attention(config)

    assert module.num_key_value_groups == 2
    assert module.scale == config.head_dim**-0.5
    assert module.o_proj.weight.shape == (config.hidden_size, config.hidden_size)
    assert module.o_proj.bias is None


def test_qwen2_attention_matches_dense_reference():
    torch.manual_seed(8)
    config = TinyQwen2Config()
    module = Qwen2Attention(config)
    hidden_states = torch.randn(2, 5, config.hidden_size)
    position_ids = torch.tensor([[0, 1, 2, 3, 4], [2, 4, 5, 8, 9]])

    actual = module(hidden_states, position_ids)
    expected = qwen2_attention_reference(module, hidden_states, position_ids)

    torch.testing.assert_close(actual, expected)


def test_qwen2_attention_is_causal():
    torch.manual_seed(9)
    config = TinyQwen2Config()
    module = Qwen2Attention(config)
    hidden_states = torch.randn(1, 6, config.hidden_size)
    changed_hidden_states = hidden_states.clone()
    changed_hidden_states[:, 3:] = torch.randn_like(changed_hidden_states[:, 3:])
    position_ids = torch.arange(6).unsqueeze(0)

    original = module(hidden_states, position_ids)
    changed = module(changed_hidden_states, position_ids)

    torch.testing.assert_close(original[:, :3], changed[:, :3])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_qwen2_attention_preserves_shape_and_dtype(dtype):
    config = TinyQwen2Config()
    module = Qwen2Attention(config).to(dtype)
    hidden_states = torch.randn(2, 4, config.hidden_size, dtype=dtype)
    position_ids = torch.arange(4).expand(2, -1)

    output = module(hidden_states, position_ids)

    assert output.shape == hidden_states.shape
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_qwen2_attention_supports_single_token_sequence():
    config = TinyQwen2Config()
    module = Qwen2Attention(config)
    hidden_states = torch.randn(3, 1, config.hidden_size)
    position_ids = torch.tensor([[0], [3], [10]])

    output = module(hidden_states, position_ids)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()


def test_qwen2_attention_gradients_are_finite():
    torch.manual_seed(10)
    config = TinyQwen2Config()
    module = Qwen2Attention(config)
    hidden_states = torch.randn(2, 4, config.hidden_size, requires_grad=True)
    position_ids = torch.arange(4).expand(2, -1)

    module(hidden_states, position_ids).square().mean().backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_qwen2_attention_prefill_with_cache_matches_dense():
    torch.manual_seed(0)
    config = TinyQwen2Config()
    module = Qwen2Attention(config)
    seq_len = 6
    block_size = 4
    layer_idx = 1
    seq_id = "request-0"
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    kv_cache.add_sequence(seq_id)
    assert kv_cache.reserve_tokens(seq_id, seq_len)

    block_table = kv_cache.block_tables[seq_id]
    slot_mapping = torch.tensor(
        [
            block_table[position // block_size] * block_size
            + position % block_size
            for position in range(seq_len)
        ],
        dtype=torch.long,
    )
    hidden_states = torch.randn(1, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    dense_output = module(hidden_states, position_ids)
    cached_output = module.forward_prefill_with_cache(
        hidden_states,
        position_ids,
        kv_cache,
        layer_idx,
        slot_mapping,
    )

    torch.testing.assert_close(cached_output, dense_output)

    query, expected_key, expected_value = module.qkv_proj(hidden_states)
    cos, sin = module.rotary_embedding(query, position_ids)
    _, expected_key = apply_rotary_pos_emb(
        query,
        expected_key,
        cos,
        sin,
    )
    layer_k_cache, layer_v_cache = kv_cache.get_layer_cache(layer_idx)

    for token_index, slot in enumerate(slot_mapping.tolist()):
        physical_block = slot // block_size
        block_offset = slot % block_size
        torch.testing.assert_close(
            layer_k_cache[physical_block, block_offset],
            expected_key[0, token_index],
        )
        torch.testing.assert_close(
            layer_v_cache[physical_block, block_offset],
            expected_value[0, token_index],
        )


def test_qwen2_attention_paged_decode_matches_dense_last_token():
    torch.manual_seed(1)
    config = TinyQwen2Config()
    module = Qwen2Attention(config)
    prefill_len = 4
    context_len = prefill_len + 1
    block_size = 4
    layer_idx = 0
    seq_id = "request-0"
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    kv_cache.add_sequence(seq_id)
    assert kv_cache.reserve_tokens(seq_id, prefill_len)

    prefill_hidden_states = torch.randn(
        1, prefill_len, config.hidden_size
    )
    decode_hidden_states = torch.randn(1, 1, config.hidden_size)
    prefill_positions = torch.arange(prefill_len).unsqueeze(0)
    prefill_block_table = kv_cache.block_tables[seq_id]
    prefill_slots = torch.tensor(
        [
            prefill_block_table[position // block_size] * block_size
            + position % block_size
            for position in range(prefill_len)
        ],
        dtype=torch.long,
    )
    module.forward_prefill_with_cache(
        prefill_hidden_states,
        prefill_positions,
        kv_cache,
        layer_idx,
        prefill_slots,
    )

    assert kv_cache.reserve_tokens(seq_id, context_len)
    block_table = kv_cache.block_tables[seq_id]
    decode_position = prefill_len
    decode_slot = torch.tensor(
        [
            block_table[decode_position // block_size] * block_size
            + decode_position % block_size
        ],
        dtype=torch.long,
    )
    incremental_output = module.forward_decode_with_cache(
        decode_hidden_states,
        torch.tensor([[decode_position]]),
        kv_cache,
        layer_idx,
        decode_slot,
        block_table,
        context_len,
    )

    full_hidden_states = torch.cat(
        [prefill_hidden_states, decode_hidden_states], dim=1
    )
    dense_last_token = module(
        full_hidden_states,
        torch.arange(context_len).unsqueeze(0),
    )[:, -1:]

    torch.testing.assert_close(incremental_output, dense_last_token)


@pytest.mark.parametrize(
    ("hidden_states", "position_ids", "message"),
    [
        (torch.randn(2, 64), torch.arange(2), "hidden_states must have shape"),
        (
            torch.randn(2, 3, 65),
            torch.arange(3).expand(2, -1),
            "last dimension must match hidden_size",
        ),
        (
            torch.randn(2, 3, 64),
            torch.arange(3).unsqueeze(0),
            "position_ids shape must match",
        ),
    ],
)
def test_qwen2_attention_rejects_invalid_inputs(
    hidden_states,
    position_ids,
    message,
):
    with pytest.raises(ValueError, match=message):
        Qwen2Attention(TinyQwen2Config())(hidden_states, position_ids)


def test_qwen2_mlp_parameter_shapes_and_biases():
    config = TinyQwen2Config()
    module = Qwen2MLP(config)

    assert module.gate_proj.weight.shape == (
        config.intermediate_size,
        config.hidden_size,
    )
    assert module.up_proj.weight.shape == (
        config.intermediate_size,
        config.hidden_size,
    )
    assert module.down_proj.weight.shape == (
        config.hidden_size,
        config.intermediate_size,
    )
    assert module.gate_proj.bias is None
    assert module.up_proj.bias is None
    assert module.down_proj.bias is None


def test_qwen2_mlp_matches_swiglu_reference():
    torch.manual_seed(11)
    config = TinyQwen2Config()
    module = Qwen2MLP(config)
    hidden_states = torch.randn(2, 3, config.hidden_size)

    expected_gate = F.silu(
        F.linear(hidden_states, module.gate_proj.weight, bias=None)
    )
    expected_up = F.linear(hidden_states, module.up_proj.weight, bias=None)
    expected = F.linear(
        expected_gate * expected_up,
        module.down_proj.weight,
        bias=None,
    )

    torch.testing.assert_close(module(hidden_states), expected)


@pytest.mark.parametrize("leading_shape", [(), (5,), (2, 3)])
def test_qwen2_mlp_preserves_leading_dimensions(leading_shape):
    config = TinyQwen2Config()
    module = Qwen2MLP(config)
    hidden_states = torch.randn(*leading_shape, config.hidden_size)

    output = module(hidden_states)

    assert output.shape == hidden_states.shape


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_qwen2_mlp_preserves_dtype(dtype):
    config = TinyQwen2Config()
    module = Qwen2MLP(config).to(dtype)
    hidden_states = torch.randn(2, 3, config.hidden_size, dtype=dtype)

    output = module(hidden_states)

    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_qwen2_mlp_applies_silu_to_gate_projection():
    config = TinyQwen2Config()
    module = Qwen2MLP(config)
    hidden_states = torch.ones(1, config.hidden_size)
    with torch.no_grad():
        module.gate_proj.weight.fill_(0.25)
        module.up_proj.weight.fill_(-0.125)
        module.down_proj.weight.fill_(0.01)

    actual = module(hidden_states)
    gate = F.linear(hidden_states, module.gate_proj.weight)
    up = F.linear(hidden_states, module.up_proj.weight)
    expected = F.linear(F.silu(gate) * up, module.down_proj.weight)
    wrong_projection_activated = F.linear(
        gate * F.silu(up),
        module.down_proj.weight,
    )

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, wrong_projection_activated)


def test_qwen2_mlp_gradients_are_finite():
    torch.manual_seed(12)
    config = TinyQwen2Config()
    module = Qwen2MLP(config)
    hidden_states = torch.randn(2, 3, config.hidden_size, requires_grad=True)

    module(hidden_states).square().mean().backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def qwen2_mlp_reference(
    module: Qwen2MLP,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    gate = F.silu(F.linear(hidden_states, module.gate_proj.weight))
    up = F.linear(hidden_states, module.up_proj.weight)
    return F.linear(gate * up, module.down_proj.weight)


def test_qwen2_decoder_layer_matches_pre_norm_reference():
    torch.manual_seed(13)
    config = TinyQwen2Config()
    layer = Qwen2DecoderLayer(config)
    hidden_states = torch.randn(2, 5, config.hidden_size)
    position_ids = torch.arange(5).expand(2, -1)

    normalized = rms_norm_reference(
        hidden_states,
        layer.input_layernorm.weight,
        layer.input_layernorm.variance_epsilon,
    )
    after_attention = hidden_states + qwen2_attention_reference(
        layer.self_attn,
        normalized,
        position_ids,
    )
    normalized_after_attention = rms_norm_reference(
        after_attention,
        layer.post_attention_layernorm.weight,
        layer.post_attention_layernorm.variance_epsilon,
    )
    expected = after_attention + qwen2_mlp_reference(
        layer.mlp,
        normalized_after_attention,
    )

    torch.testing.assert_close(layer(hidden_states, position_ids), expected)


def test_qwen2_decoder_layer_prefill_with_cache_matches_dense():
    torch.manual_seed(13)
    config = TinyQwen2Config()
    layer = Qwen2DecoderLayer(config)
    seq_len = 6
    block_size = 4
    layer_idx = 1
    seq_id = "request-0"
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    kv_cache.add_sequence(seq_id)
    assert kv_cache.reserve_tokens(seq_id, seq_len)
    block_table = kv_cache.block_tables[seq_id]
    slot_mapping = torch.tensor(
        [
            block_table[position // block_size] * block_size
            + position % block_size
            for position in range(seq_len)
        ],
        dtype=torch.long,
    )
    hidden_states = torch.randn(1, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    dense_output = layer(hidden_states, position_ids)
    cached_output = layer.forward_prefill_with_cache(
        hidden_states,
        position_ids,
        kv_cache,
        layer_idx,
        slot_mapping,
    )

    torch.testing.assert_close(cached_output, dense_output)

    normalized = layer.input_layernorm(hidden_states)
    query, expected_key, _ = layer.self_attn.qkv_proj(normalized)
    cos, sin = layer.self_attn.rotary_embedding(query, position_ids)
    _, expected_key = apply_rotary_pos_emb(
        query,
        expected_key,
        cos,
        sin,
    )
    layer_k_cache, _ = kv_cache.get_layer_cache(layer_idx)
    for token_index, slot in enumerate(slot_mapping.tolist()):
        torch.testing.assert_close(
            layer_k_cache[slot // block_size, slot % block_size],
            expected_key[0, token_index],
        )


def test_qwen2_decoder_layer_paged_decode_matches_dense_last_token():
    torch.manual_seed(14)
    config = TinyQwen2Config()
    layer = Qwen2DecoderLayer(config)
    prefill_len = 4
    context_len = prefill_len + 1
    block_size = 4
    layer_idx = 1
    seq_id = "request-0"
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    kv_cache.add_sequence(seq_id)
    assert kv_cache.reserve_tokens(seq_id, prefill_len)

    prefill_hidden_states = torch.randn(
        1, prefill_len, config.hidden_size
    )
    decode_hidden_states = torch.randn(1, 1, config.hidden_size)
    prefill_positions = torch.arange(prefill_len).unsqueeze(0)
    prefill_block_table = kv_cache.block_tables[seq_id]
    prefill_slots = torch.tensor(
        [
            prefill_block_table[position // block_size] * block_size
            + position % block_size
            for position in range(prefill_len)
        ],
        dtype=torch.long,
    )
    layer.forward_prefill_with_cache(
        prefill_hidden_states,
        prefill_positions,
        kv_cache,
        layer_idx,
        prefill_slots,
    )

    assert kv_cache.reserve_tokens(seq_id, context_len)
    block_table = kv_cache.block_tables[seq_id]
    decode_position = prefill_len
    decode_slot = torch.tensor(
        [
            block_table[decode_position // block_size] * block_size
            + decode_position % block_size
        ],
        dtype=torch.long,
    )
    incremental_output = layer.forward_decode_with_cache(
        decode_hidden_states,
        torch.tensor([[decode_position]]),
        kv_cache,
        layer_idx,
        decode_slot,
        block_table,
        context_len,
    )

    full_hidden_states = torch.cat(
        [prefill_hidden_states, decode_hidden_states], dim=1
    )
    dense_last_token = layer(
        full_hidden_states,
        torch.arange(context_len).unsqueeze(0),
    )[:, -1:]

    torch.testing.assert_close(incremental_output, dense_last_token)


def test_qwen2_decoder_layer_residual_path_is_identity_when_branches_are_zero():
    config = TinyQwen2Config()
    layer = Qwen2DecoderLayer(config)
    with torch.no_grad():
        for parameter in layer.self_attn.parameters():
            parameter.zero_()
        for parameter in layer.mlp.parameters():
            parameter.zero_()
    hidden_states = torch.randn(2, 4, config.hidden_size)
    position_ids = torch.arange(4).expand(2, -1)

    output = layer(hidden_states, position_ids)

    torch.testing.assert_close(output, hidden_states, rtol=0, atol=0)


def test_qwen2_decoder_layer_is_causal():
    torch.manual_seed(14)
    config = TinyQwen2Config()
    layer = Qwen2DecoderLayer(config)
    hidden_states = torch.randn(1, 6, config.hidden_size)
    changed_hidden_states = hidden_states.clone()
    changed_hidden_states[:, 3:] = torch.randn_like(changed_hidden_states[:, 3:])
    position_ids = torch.arange(6).unsqueeze(0)

    original = layer(hidden_states, position_ids)
    changed = layer(changed_hidden_states, position_ids)

    torch.testing.assert_close(original[:, :3], changed[:, :3])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_qwen2_decoder_layer_preserves_shape_and_dtype(dtype):
    config = TinyQwen2Config()
    layer = Qwen2DecoderLayer(config).to(dtype)
    hidden_states = torch.randn(2, 4, config.hidden_size, dtype=dtype)
    position_ids = torch.arange(4).expand(2, -1)

    output = layer(hidden_states, position_ids)

    assert output.shape == hidden_states.shape
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_qwen2_decoder_layer_gradients_are_finite():
    torch.manual_seed(15)
    config = TinyQwen2Config()
    layer = Qwen2DecoderLayer(config)
    hidden_states = torch.randn(2, 4, config.hidden_size, requires_grad=True)
    position_ids = torch.arange(4).expand(2, -1)

    layer(hidden_states, position_ids).square().mean().backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_tiny_qwen2_model_initialization_and_parameter_registration():
    config = TinyQwen2Config()
    model = TinyQwen2Model(config)

    assert model.config is config
    assert model.embed_tokens.weight.shape == (
        config.vocab_size,
        config.hidden_size,
    )
    assert len(model.layers) == config.num_hidden_layers
    assert model.norm.weight.shape == (config.hidden_size,)
    parameter_names = dict(model.named_parameters())
    for layer_idx in range(config.num_hidden_layers):
        assert f"layers.{layer_idx}.self_attn.o_proj.weight" in parameter_names
        assert f"layers.{layer_idx}.mlp.down_proj.weight" in parameter_names


def test_tiny_qwen2_model_matches_explicit_layer_composition():
    torch.manual_seed(16)
    config = TinyQwen2Config()
    model = TinyQwen2Model(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    position_ids = torch.tensor([[0, 1, 2, 3, 4], [3, 4, 7, 8, 10]])

    hidden_states = F.embedding(input_ids, model.embed_tokens.weight)
    for layer in model.layers:
        hidden_states = layer(hidden_states, position_ids)
    expected = rms_norm_reference(
        hidden_states,
        model.norm.weight,
        model.norm.variance_epsilon,
    )

    torch.testing.assert_close(model(input_ids, position_ids), expected)


def test_tiny_qwen2_model_prefill_with_cache_matches_dense():
    torch.manual_seed(16)
    config = TinyQwen2Config()
    model = TinyQwen2Model(config)
    seq_len = 6
    block_size = 4
    seq_id = "request-0"
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    kv_cache.k_cache.fill_(torch.nan)
    kv_cache.v_cache.fill_(torch.nan)
    kv_cache.add_sequence(seq_id)
    assert kv_cache.reserve_tokens(seq_id, seq_len)
    block_table = kv_cache.block_tables[seq_id]
    slot_mapping = torch.tensor(
        [
            block_table[position // block_size] * block_size
            + position % block_size
            for position in range(seq_len)
        ],
        dtype=torch.long,
    )
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)

    dense_output = model(input_ids, position_ids)
    cached_output = model.forward_prefill_with_cache(
        input_ids,
        position_ids,
        kv_cache,
        slot_mapping,
    )

    torch.testing.assert_close(cached_output, dense_output)
    for layer_idx in range(config.num_hidden_layers):
        layer_k_cache, layer_v_cache = kv_cache.get_layer_cache(layer_idx)
        for slot in slot_mapping.tolist():
            physical_block = slot // block_size
            block_offset = slot % block_size
            assert torch.isfinite(
                layer_k_cache[physical_block, block_offset]
            ).all()
            assert torch.isfinite(
                layer_v_cache[physical_block, block_offset]
            ).all()


def test_tiny_qwen2_model_paged_decode_matches_dense_last_token():
    torch.manual_seed(17)
    config = TinyQwen2Config()
    model = TinyQwen2Model(config)
    prefill_len = 4
    context_len = prefill_len + 1
    block_size = 4
    seq_id = "request-0"
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    kv_cache.add_sequence(seq_id)
    assert kv_cache.reserve_tokens(seq_id, prefill_len)
    input_ids = torch.randint(0, config.vocab_size, (1, context_len))
    prefill_input_ids = input_ids[:, :prefill_len]
    prefill_positions = torch.arange(prefill_len).unsqueeze(0)
    prefill_block_table = kv_cache.block_tables[seq_id]
    prefill_slots = torch.tensor(
        [
            prefill_block_table[position // block_size] * block_size
            + position % block_size
            for position in range(prefill_len)
        ],
        dtype=torch.long,
    )
    model.forward_prefill_with_cache(
        prefill_input_ids,
        prefill_positions,
        kv_cache,
        prefill_slots,
    )

    assert kv_cache.reserve_tokens(seq_id, context_len)
    block_table = kv_cache.block_tables[seq_id]
    decode_position = prefill_len
    decode_slot = torch.tensor(
        [
            block_table[decode_position // block_size] * block_size
            + decode_position % block_size
        ],
        dtype=torch.long,
    )
    incremental_output = model.forward_decode_with_cache(
        input_ids[:, -1:],
        torch.tensor([[decode_position]]),
        kv_cache,
        decode_slot,
        block_table,
        context_len,
    )
    dense_last_token = model(
        input_ids,
        torch.arange(context_len).unsqueeze(0),
    )[:, -1:]

    torch.testing.assert_close(incremental_output, dense_last_token)


def test_tiny_qwen2_model_auto_positions_match_explicit_positions():
    config = TinyQwen2Config()
    model = TinyQwen2Model(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    position_ids = torch.arange(5).expand(2, -1)

    automatic = model(input_ids)
    explicit = model(input_ids, position_ids)

    torch.testing.assert_close(automatic, explicit, rtol=0, atol=0)


def test_tiny_qwen2_model_executes_every_decoder_layer():
    config = TinyQwen2Config(num_hidden_layers=3)
    model = TinyQwen2Model(config)
    call_counts = [0] * config.num_hidden_layers
    handles = []

    for layer_idx, layer in enumerate(model.layers):
        def count_call(_module, _inputs, _output, index=layer_idx):
            call_counts[index] += 1

        handles.append(layer.register_forward_hook(count_call))

    try:
        model(torch.randint(0, config.vocab_size, (2, 4)))
    finally:
        for handle in handles:
            handle.remove()

    assert call_counts == [1, 1, 1]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tiny_qwen2_model_preserves_parameter_dtype(dtype):
    config = TinyQwen2Config()
    model = TinyQwen2Model(config).to(dtype)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))

    output = model(input_ids)

    assert output.shape == (2, 4, config.hidden_size)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_tiny_qwen2_model_is_causal():
    config = TinyQwen2Config()
    model = TinyQwen2Model(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    changed_input_ids = input_ids.clone()
    changed_input_ids[:, 3:] = torch.randint(0, config.vocab_size, (1, 3))

    original = model(input_ids)
    changed = model(changed_input_ids)

    torch.testing.assert_close(original[:, :3], changed[:, :3])


def test_tiny_qwen2_model_gradients_are_finite():
    torch.manual_seed(17)
    config = TinyQwen2Config()
    model = TinyQwen2Model(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))

    model(input_ids).square().mean().backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    ("input_ids", "position_ids", "exception", "message"),
    [
        (torch.ones(4, dtype=torch.long), None, ValueError, "input_ids must have shape"),
        (
            torch.ones(2, 4, dtype=torch.float32),
            None,
            TypeError,
            "input_ids must have dtype",
        ),
        (
            torch.empty(2, 0, dtype=torch.long),
            None,
            ValueError,
            "sequence length must be greater than 0",
        ),
        (
            torch.ones(2, 4, dtype=torch.long),
            torch.arange(4).unsqueeze(0),
            ValueError,
            "position_ids shape must match",
        ),
    ],
)
def test_tiny_qwen2_model_rejects_invalid_inputs(
    input_ids,
    position_ids,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        TinyQwen2Model(TinyQwen2Config())(input_ids, position_ids)


def test_tiny_qwen2_causal_lm_initialization():
    config = TinyQwen2Config()
    causal_lm = TinyQwen2ForCausalLM(config)

    assert causal_lm.lm_head.weight.shape == (
        config.vocab_size,
        config.hidden_size,
    )
    assert causal_lm.lm_head.bias is None
    assert causal_lm.lm_head.weight is not causal_lm.model.embed_tokens.weight


def test_tiny_qwen2_causal_lm_logits_match_linear_reference():
    torch.manual_seed(18)
    config = TinyQwen2Config()
    causal_lm = TinyQwen2ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    position_ids = torch.tensor([[0, 1, 2, 3, 4], [3, 4, 6, 9, 10]])

    hidden_states = causal_lm.model(input_ids, position_ids)
    expected = F.linear(hidden_states, causal_lm.lm_head.weight, bias=None)
    logits = causal_lm(input_ids, position_ids)

    assert logits.shape == (2, 5, config.vocab_size)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def test_tiny_qwen2_causal_lm_auto_positions_match_explicit_positions():
    config = TinyQwen2Config()
    causal_lm = TinyQwen2ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    position_ids = torch.arange(5).expand(2, -1)

    torch.testing.assert_close(
        causal_lm(input_ids),
        causal_lm(input_ids, position_ids),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tiny_qwen2_causal_lm_preserves_parameter_dtype(dtype):
    config = TinyQwen2Config()
    causal_lm = TinyQwen2ForCausalLM(config).to(dtype)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))

    logits = causal_lm(input_ids)

    assert logits.shape == (2, 4, config.vocab_size)
    assert logits.dtype == dtype
    assert torch.isfinite(logits).all()


def test_tiny_qwen2_causal_lm_logits_are_causal():
    config = TinyQwen2Config()
    causal_lm = TinyQwen2ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    changed_input_ids = input_ids.clone()
    changed_input_ids[:, 3:] = torch.randint(0, config.vocab_size, (1, 3))

    original = causal_lm(input_ids)
    changed = causal_lm(changed_input_ids)

    torch.testing.assert_close(original[:, :3], changed[:, :3])


def test_tiny_qwen2_causal_lm_gradients_cover_backbone_and_lm_head():
    torch.manual_seed(19)
    config = TinyQwen2Config()
    causal_lm = TinyQwen2ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))

    causal_lm(input_ids).square().mean().backward()

    assert causal_lm.model.embed_tokens.weight.grad is not None
    assert causal_lm.lm_head.weight.grad is not None
    for parameter in causal_lm.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_tiny_qwen2_causal_lm_paged_decode_logits_match_dense():
    torch.manual_seed(20)
    config = TinyQwen2Config()
    causal_lm = TinyQwen2ForCausalLM(config)
    prefill_len = 4
    context_len = prefill_len + 1
    block_size = 4
    seq_id = "request-0"
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    kv_cache.add_sequence(seq_id)
    assert kv_cache.reserve_tokens(seq_id, prefill_len)
    input_ids = torch.randint(0, config.vocab_size, (1, context_len))
    prefill_input_ids = input_ids[:, :prefill_len]
    prefill_positions = torch.arange(prefill_len).unsqueeze(0)
    prefill_block_table = kv_cache.block_tables[seq_id]
    prefill_slots = torch.tensor(
        [
            prefill_block_table[position // block_size] * block_size
            + position % block_size
            for position in range(prefill_len)
        ],
        dtype=torch.long,
    )

    prefill_logits = causal_lm.forward_prefill_with_cache(
        prefill_input_ids,
        prefill_positions,
        kv_cache,
        prefill_slots,
    )
    dense_prefill_logits = causal_lm(
        prefill_input_ids,
        prefill_positions,
    )
    torch.testing.assert_close(prefill_logits, dense_prefill_logits)

    assert kv_cache.reserve_tokens(seq_id, context_len)
    block_table = kv_cache.block_tables[seq_id]
    decode_position = prefill_len
    decode_slot = torch.tensor(
        [
            block_table[decode_position // block_size] * block_size
            + decode_position % block_size
        ],
        dtype=torch.long,
    )
    incremental_logits = causal_lm.forward_decode_with_cache(
        input_ids[:, -1:],
        torch.tensor([[decode_position]]),
        kv_cache,
        decode_slot,
        block_table,
        context_len,
    )
    dense_last_token_logits = causal_lm(
        input_ids,
        torch.arange(context_len).unsqueeze(0),
    )[:, -1:]

    torch.testing.assert_close(
        incremental_logits,
        dense_last_token_logits,
    )
