import copy

import pytest
import torch

from mini_llm_runtime.model.qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    TinyQwen2Config,
    TinyQwen2ForCausalLM,
)
from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache


def make_decode_case(num_kv_heads=2, dtype=torch.float32, module_type=Qwen2Attention):
    torch.manual_seed(42)
    config = TinyQwen2Config(num_key_value_heads=num_kv_heads)
    attention = module_type(config).to(dtype=dtype)
    cache = PagedKVCache(2, 6, 2, num_kv_heads, config.head_dim, dtype=dtype)
    cache.k_cache.normal_()
    cache.v_cache.normal_()
    inputs = dict(
        hidden_states=torch.randn(2, 1, config.hidden_size, dtype=dtype),
        position_ids=torch.tensor([[2], [4]]),
        layer_idx=1,
        slot_mapping=torch.tensor([0, 8]),
        block_tables=torch.tensor([[2, 0, -1], [1, 3, 4]]),
        context_lens=torch.tensor([3, 5]),
    )
    return attention, cache, inputs


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("module_type", [Qwen2Attention, Qwen2DecoderLayer])
def test_batch_decode_matches_single_requests(num_kv_heads, dtype, batch_size, module_type):
    attention, cache, inputs = make_decode_case(num_kv_heads, dtype, module_type)
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[name] = value[:batch_size]
    reference_cache = copy.deepcopy(cache)
    old_k = cache.k_cache.clone()
    old_v = cache.v_cache.clone()

    with torch.no_grad():
        expected = torch.cat([
            attention.forward_decode_with_cache(
                hidden_states=inputs["hidden_states"][i:i + 1],
                position_ids=inputs["position_ids"][i:i + 1],
                kv_cache=reference_cache,
                layer_idx=inputs["layer_idx"],
                slot_mapping=inputs["slot_mapping"][i:i + 1],
                block_table=inputs["block_tables"][i].tolist(),
                context_len=int(inputs["context_lens"][i]),
            )
            for i in range(batch_size)
        ])
        actual = attention.forward_decode_batch_with_cache(kv_cache=cache, **inputs)

    assert actual.shape == inputs["hidden_states"].shape
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(cache.k_cache, reference_cache.k_cache)
    torch.testing.assert_close(cache.v_cache, reference_cache.v_cache)

    # All other slots and the other layer must remain untouched.
    unchanged = torch.ones(cache.k_cache.shape[:3], dtype=torch.bool)
    slots = inputs["slot_mapping"]
    unchanged[1, slots // cache.block_size, slots % cache.block_size] = False
    assert torch.equal(cache.k_cache[unchanged], old_k[unchanged])
    assert torch.equal(cache.v_cache[unchanged], old_v[unchanged])


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("hidden_states", torch.zeros(2, 2, 64), "hidden_states must have shape"),
        ("position_ids", torch.tensor([[1], [4]]), "decode positions"),
        ("context_lens", torch.tensor([0, 5]), "greater than 0"),
        ("block_tables", torch.tensor([[2, 0], [1, 3]]), "insufficient blocks"),
        ("block_tables", torch.tensor([[2, -1, -1], [1, 3, 4]]), "invalid physical block"),
        ("slot_mapping", torch.tensor([1, 8]), "slot_mapping must match"),
        ("context_lens", torch.tensor([3., 5.]), "torch.long dtype"),
    ],
)
def test_attention_batch_decode_rejects_metadata_before_cache_write(field, value, message):
    attention, cache, inputs = make_decode_case()
    old_k = cache.k_cache.clone()
    old_v = cache.v_cache.clone()
    inputs[field] = value
    with pytest.raises(ValueError, match=message):
        attention.forward_decode_batch_with_cache(kv_cache=cache, **inputs)
    assert torch.equal(cache.k_cache, old_k)
    assert torch.equal(cache.v_cache, old_v)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tie_word_embeddings", [False, True])
@torch.no_grad()
def test_model_batch_decode_multistep_matches_single_and_dense(dtype, tie_word_embeddings):
    torch.manual_seed(17)
    config = TinyQwen2Config(tie_word_embeddings=tie_word_embeddings)
    model = TinyQwen2ForCausalLM(config).to(dtype=dtype).eval()
    cache = PagedKVCache(2, 16, 2, 2, config.head_dim, dtype=dtype)
    cache.k_cache.zero_()
    cache.v_cache.zero_()
    histories = [[10, 11], [20, 21, 22, 23]]
    pending = []

    # Separate prefill establishes genuine, different-length histories.
    for request_id, tokens in enumerate(histories):
        cache.add_sequence(request_id)
        assert cache.reserve_tokens(request_id, len(tokens))
        table = cache.block_tables[request_id]
        slots = torch.tensor([table[p // 2] * 2 + p % 2 for p in range(len(tokens))])
        logits = model.forward_prefill_with_cache(
            torch.tensor([tokens]), torch.arange(len(tokens))[None, :], cache, slots
        )
        pending.append(int(logits[0, -1].argmax()))
    reference_cache = copy.deepcopy(cache)
    initial_blocks = [len(cache.block_tables[i]) for i in range(2)]

    for _ in range(5):
        positions = torch.tensor([[len(tokens)] for tokens in histories])
        lengths = positions[:, 0] + 1
        for request_id, length in enumerate(lengths.tolist()):
            assert cache.reserve_tokens(request_id, length)
            assert reference_cache.reserve_tokens(request_id, length)
        tables = [cache.block_tables[i] for i in range(2)]
        width = max(map(len, tables))
        block_tables = torch.tensor([table + [-1] * (width - len(table)) for table in tables])
        slots = torch.tensor([
            tables[i][int(positions[i, 0]) // 2] * 2 + int(positions[i, 0]) % 2
            for i in range(2)
        ])
        input_ids = torch.tensor(pending)[:, None]
        old_k, old_v = cache.k_cache.clone(), cache.v_cache.clone()
        actual = model.forward_decode_batch_with_cache(
            input_ids, positions, cache, slots, block_tables, lengths
        )
        expected = torch.cat([
            model.forward_decode_with_cache(
                input_ids[i:i + 1], positions[i:i + 1], reference_cache,
                slots[i:i + 1], tables[i], int(lengths[i]),
            )
            for i in range(2)
        ])
        assert actual.shape == (2, 1, config.vocab_size)
        # Multi-step comparisons include FP32 softmax rounding, even when
        # model weights are FP64; batch GEMMs can propagate different rounding.
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))
        torch.testing.assert_close(cache.k_cache, reference_cache.k_cache, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cache.v_cache, reference_cache.v_cache, atol=1e-5, rtol=1e-5)

        unchanged = torch.ones(cache.k_cache.shape[:3], dtype=torch.bool)
        unchanged[:, slots // 2, slots % 2] = False
        assert torch.equal(cache.k_cache[unchanged], old_k[unchanged])
        assert torch.equal(cache.v_cache[unchanged], old_v[unchanged])
        for i in range(2):
            histories[i].append(pending[i])
            dense = model(torch.tensor([histories[i]]))[:, -1:]
            # Paged attention uses FP32 softmax even for FP64 inputs,
            # whereas dense attention preserves FP64 softmax precision.
            torch.testing.assert_close(actual[i:i + 1], dense, atol=1e-5, rtol=1e-5)
            assert torch.equal(actual[i:i + 1].argmax(-1), dense.argmax(-1))
        pending = actual[:, 0].argmax(-1).tolist()

    for i in range(2):
        assert len(cache.block_tables[i]) >= initial_blocks[i] + 2


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("input_ids", torch.zeros(2, 2, dtype=torch.long), "shape"),
        ("input_ids", torch.zeros(2, 1), "dtype"),
        ("input_ids", torch.empty(0, 1, dtype=torch.long), "empty"),
        ("position_ids", torch.tensor([2, 4]), "shape"),
        ("context_lens", torch.tensor([0, 5]), "greater than 0"),
        ("slot_mapping", torch.tensor([1, 8]), "slot_mapping must match"),
    ],
)
def test_model_batch_decode_rejects_invalid_inputs_before_write(field, value, message):
    _, cache, inputs = make_decode_case()
    model = TinyQwen2ForCausalLM(TinyQwen2Config())
    inputs.pop("hidden_states")
    inputs.pop("layer_idx")
    inputs["input_ids"] = torch.tensor([[10], [20]])
    inputs[field] = value
    old_k, old_v = cache.k_cache.clone(), cache.v_cache.clone()
    with pytest.raises((ValueError, TypeError), match=message):
        model.forward_decode_batch_with_cache(kv_cache=cache, **inputs)
    assert torch.equal(cache.k_cache, old_k)
    assert torch.equal(cache.v_cache, old_v)
