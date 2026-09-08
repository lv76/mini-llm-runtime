import math

import torch

from mini_llm_runtime.kernels.reference.paged_attention import (
    paged_attention_decode,
)

from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache

def test_paged_attention_decode_matches_dense():
    torch.manual_seed(0)

    num_blocks = 4
    block_size = 4
    num_heads = 2
    head_dim = 8
    seq_len = 6

    # 当前 decode token 的 query
    # [H, D]
    q = torch.randn(num_heads, head_dim)

    # 按逻辑 token 顺序生成连续 K/V，用于构造 reference
    # [S, H, D]
    logical_k = torch.randn(seq_len, num_heads, head_dim)
    logical_v = torch.randn(seq_len, num_heads, head_dim)

    # 物理 KV Cache
    # [num_blocks, block_size, H, D]
    k_cache = torch.empty(
        num_blocks,
        block_size,
        num_heads,
        head_dim,
    )
    v_cache = torch.empty_like(k_cache)

    # 故意使用非连续且乱序的 physical blocks
    #
    # logical block 0 -> physical block 2
    # logical block 1 -> physical block 0
    block_table = [2, 0]

    # 把逻辑连续的 K/V 写入 paged cache
    for position in range(seq_len):
        logical_block = position // block_size
        offset = position % block_size
        physical_block = block_table[logical_block]

        k_cache[physical_block, offset].copy_(logical_k[position])
        v_cache[physical_block, offset].copy_(logical_v[position])

    actual = paged_attention_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        block_size=block_size,
    )

    # Dense attention reference
    #
    # [S, H, D] -> [H, S, D]
    dense_k = logical_k.permute(1, 0, 2)
    dense_v = logical_v.permute(1, 0, 2)

    # [H, D] -> [H, 1, D]
    q_matrix = q.unsqueeze(1)

    # [H, 1, D] @ [H, D, S] -> [H, 1, S]
    scores = torch.matmul(
        q_matrix,
        dense_k.transpose(-2, -1),
    ) / math.sqrt(head_dim)

    probs = torch.softmax(scores, dim=-1)

    # [H, 1, S] @ [H, S, D] -> [H, 1, D]
    expected = torch.matmul(probs, dense_v).squeeze(1)

    assert actual.shape == (num_heads, head_dim)
    torch.testing.assert_close(actual, expected)


def test_paged_attention_decode_with_paged_kv_cache():
    torch.manual_seed(0)

    num_heads = 2
    head_dim = 8
    seq_len = 6

    cache = PagedKVCache(
        num_layers=1,
        num_blocks=6,
        block_size=4,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
    )

    seq_id = 0
    cache.add_sequence(seq_id)

    logical_k = torch.randn(seq_len, num_heads, head_dim)
    logical_v = torch.randn(seq_len, num_heads, head_dim)

    cache.reserve_tokens(seq_id, seq_len)
    slot_mapping = []

    for position in range(seq_len):
        logical_block = position // cache.block_size
        block_offset = position % cache.block_size
        physical_block = cache.block_tables[seq_id][logical_block]
        slot = physical_block * cache.block_size + block_offset
        slot_mapping.append(slot) 

    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long)             
    cache.write_slots(
        slot_mapping,
        logical_k,
        logical_v,
        layer_idx=0,
    )

    q = torch.randn(num_heads, head_dim)
    k_cache,v_cache = cache.get_layer_cache(0)
    
    actual = paged_attention_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=cache.block_tables[seq_id],
        seq_len=seq_len,
        block_size=cache.block_size,
    )

    # Dense reference
    dense_k, dense_v = logical_k, logical_v

    dense_k = dense_k.permute(1, 0, 2)  # [H, S, D]
    dense_v = dense_v.permute(1, 0, 2)  # [H, S, D]

    q_matrix = q.unsqueeze(1)  # [H, 1, D]

    scores = torch.matmul(
        q_matrix,
        dense_k.transpose(-2, -1),
    ) / math.sqrt(head_dim)

    probs = torch.softmax(scores, dim=-1)

    expected = torch.matmul(probs, dense_v).squeeze(1)

    torch.testing.assert_close(actual, expected)


def test_paged_attention_decode_gqa_matches_dense():
    torch.manual_seed(0)

    num_blocks = 4
    block_size = 4
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 8
    seq_len = 6

    q = torch.randn(num_query_heads, head_dim)

    logical_k = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
    )
    logical_v = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
    )

    k_cache = torch.empty(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
    )
    v_cache = torch.empty_like(k_cache)

    # 故意使用乱序 physical blocks
    block_table = [2, 0]

    for position in range(seq_len):
        logical_block = position // block_size
        offset = position % block_size
        physical_block = block_table[logical_block]

        k_cache[physical_block, offset].copy_(
            logical_k[position]
        )
        v_cache[physical_block, offset].copy_(
            logical_v[position]
        )

    actual = paged_attention_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        block_size=block_size,
    )

    # Dense GQA reference
    dense_k = logical_k.permute(1, 0, 2)
    dense_v = logical_v.permute(1, 0, 2)

    group_size = num_query_heads // num_kv_heads

    dense_k = dense_k.repeat_interleave(
        group_size,
        dim=0,
    )
    dense_v = dense_v.repeat_interleave(
        group_size,
        dim=0,
    )

    q_matrix = q.unsqueeze(1)

    scores = torch.matmul(
        q_matrix,
        dense_k.transpose(-2, -1),
    ) / math.sqrt(head_dim)

    probs = torch.softmax(scores, dim=-1)

    expected = torch.matmul(
        probs,
        dense_v,
    ).squeeze(1)

    assert actual.shape == (num_query_heads, head_dim)
    torch.testing.assert_close(actual, expected)
