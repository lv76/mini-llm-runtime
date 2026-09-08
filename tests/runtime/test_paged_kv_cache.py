import torch
import pytest
from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache


def test_append_single_block():
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    cache.add_sequence(0)

    k = torch.randn(2, 8)
    v = torch.randn(2, 8)

    cache.append(0, k, v)

    assert cache.seq_lens[0] == 1
    assert len(cache.block_tables[0]) == 1

    physical_block = cache.block_tables[0][0]

    torch.testing.assert_close(
        cache.k_cache[0, physical_block, 0],
        k,
    )

    torch.testing.assert_close(
        cache.v_cache[0, physical_block, 0],
        v,
    )

def test_append_across_block_boundary():
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    cache.add_sequence(0)

    for _ in range(5):
        k = torch.randn(2, 8)
        v = torch.randn(2, 8)

        cache.append(0, k, v)

    assert cache.seq_lens[0] == 5
    assert len(cache.block_tables[0]) == 2


def test_free_and_reuse():
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    cache.add_sequence(0)

    # seq0 用满两个 physical blocks
    for _ in range(5):
        k = torch.randn(2, 8)
        v = torch.randn(2, 8)
        cache.append(0, k, v)

    assert len(cache.block_tables[0]) == 2
    assert len(cache.free_blocks) == 0

    used_blocks = set(cache.block_tables[0])

    # 释放 seq0
    cache.free_sequence(0)

    assert 0 not in cache.block_tables
    assert 0 not in cache.seq_lens
    assert len(cache.free_blocks) == 2

    # 新 sequence 再申请
    cache.add_sequence(1)

    k = torch.randn(2, 8)
    v = torch.randn(2, 8)

    cache.append(1, k, v)

    new_block = cache.block_tables[1][0]

    # 新申请的 block 必须来自刚才释放的 block
    assert new_block in used_blocks


def test_multiple_sequences():
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    cache.add_sequence(0)
    cache.add_sequence(1)

    for _ in range(6):
        cache.append(
            0,
            torch.randn(2, 8),
            torch.randn(2, 8),
        )

    for _ in range(3):
        cache.append(
            1,
            torch.randn(2, 8),
            torch.randn(2, 8),
        )

    assert cache.seq_lens[0] == 6
    assert cache.seq_lens[1] == 3

    assert len(cache.block_tables[0]) == 2
    assert len(cache.block_tables[1]) == 1

    blocks0 = set(cache.block_tables[0])
    blocks1 = set(cache.block_tables[1])

    assert blocks0.isdisjoint(blocks1)


def test_get_sequence_across_blocks():
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    cache.add_sequence(0)

    expected_k = []
    expected_v = []

    for _ in range(6):
        k = torch.randn(2, 8)
        v = torch.randn(2, 8)

        expected_k.append(k)
        expected_v.append(v)

        cache.append(0, k, v)

    actual_k, actual_v = cache.get_sequence(0)

    expected_k = torch.stack(expected_k)
    expected_v = torch.stack(expected_v)

    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)

def test_multiple_sequences_data():
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=6,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    cache.add_sequence(0)
    cache.add_sequence(1)

    k0 = torch.randn(6, 2, 8)
    v0 = torch.randn(6, 2, 8)

    k1 = torch.randn(3, 2, 8)
    v1 = torch.randn(3, 2, 8)

    for i in range(6):
        cache.append(0, k0[i], v0[i])

    for i in range(3):
        cache.append(1, k1[i], v1[i])

    actual_k0, actual_v0 = cache.get_sequence(0)
    actual_k1, actual_v1 = cache.get_sequence(1)

    torch.testing.assert_close(actual_k0, k0)
    torch.testing.assert_close(actual_v0, v0)

    torch.testing.assert_close(actual_k1, k1)
    torch.testing.assert_close(actual_v1, v1)

def test_prefill_then_decode():
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=8,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    cache.add_sequence(0)

    # Prefill
    prompt_k = torch.randn(8, 2, 8)
    prompt_v = torch.randn(8, 2, 8)

    cache.append_tokens(0, prompt_k, prompt_v)

    assert cache.seq_lens[0] == 8
    assert len(cache.block_tables[0]) == 2

    # Decode one token
    new_k = torch.randn(2, 8)
    new_v = torch.randn(2, 8)

    cache.append(0, new_k, new_v)

    assert cache.seq_lens[0] == 9
    assert len(cache.block_tables[0]) == 3

    actual_k, actual_v = cache.get_sequence(0)

    expected_k = torch.cat(
        [prompt_k, new_k.unsqueeze(0)],
        dim=0,
    )

    expected_v = torch.cat(
        [prompt_v, new_v.unsqueeze(0)],
        dim=0,
    )

    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)

def test_prefill_oom_does_not_modify_cache():

    cache = PagedKVCache(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    cache.add_sequence(0)

    prompt_k = torch.randn(10, 2, 8)
    prompt_v = torch.randn(10, 2, 8)

    with pytest.raises(RuntimeError):
        cache.append_tokens(0, prompt_k, prompt_v)

    assert cache.seq_lens[0] == 0
    assert cache.block_tables[0] == []
    assert len(cache.free_blocks) == 2

def test_reserve_tokens_is_atomic_and_reusable():
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=3,
        block_size=2,
        num_kv_heads=1,
        head_dim=4,
        device="cpu",
    )

    cache.add_sequence("A")

    # 3 tokens 需要两个 blocks。
    assert cache.reserve_tokens("A", total_num_tokens=3) is True
    assert len(cache.block_tables["A"]) == 2
    assert len(cache.free_blocks) == 1
    assert cache.seq_lens["A"] == 0

    # 从容量 3 扩展到 4，仍然只需要两个 blocks。
    assert cache.reserve_tokens("A", total_num_tokens=4) is True
    assert len(cache.block_tables["A"]) == 2
    assert len(cache.free_blocks) == 1

    cache.add_sequence("B")

    # B 也需要两个 blocks，但现在只有一个，必须整体失败。
    assert cache.reserve_tokens("B", total_num_tokens=3) is False
    assert cache.block_tables["B"] == []
    assert len(cache.free_blocks) == 1

    # A 释放后，B 可以复用这些 blocks。
    cache.free_sequence("A")

    assert cache.reserve_tokens("B", total_num_tokens=3) is True
    assert len(cache.block_tables["B"]) == 2
    assert len(cache.free_blocks) == 1


def test_write_slots_writes_kv_to_physical_locations():
      cache = PagedKVCache(
          num_layers=1,
          num_blocks=3,
          block_size=4,
          num_kv_heads=2,
          head_dim=8,
          dtype=torch.float32,
          device="cpu",
      )

      slot_mapping = torch.tensor([1, 6, 11], dtype=torch.long)

      k = torch.randn(3, 2, 8)
      v = torch.randn(3, 2, 8)

      cache.write_slots(slot_mapping, k, v)

      expected_locations = [
          (0, 1),
          (1, 2),
          (2, 3),
      ]

      for token_index, (physical_block, block_offset) in enumerate(
          expected_locations
      ):
          torch.testing.assert_close(
              cache.k_cache[0, physical_block, block_offset],
              k[token_index],
          )
          torch.testing.assert_close(
              cache.v_cache[0, physical_block, block_offset],
              v[token_index],
          )


def test_write_slots_rejects_non_3d_kv():
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    slot_mapping = torch.tensor([0, 1], dtype=torch.long)
    k = torch.randn(2, 8)
    v = torch.randn(2, 8)

    with pytest.raises(ValueError, match="Expected K/V shape"):
        cache.write_slots(slot_mapping, k, v, 0)


def test_write_slots_writes_to_independent_layers():
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    slot_mapping = torch.tensor([1, 6], dtype=torch.long)

    layer_0_k = torch.randn(2, 2, 8)
    layer_0_v = torch.randn(2, 2, 8)
    layer_1_k = torch.randn(2, 2, 8)
    layer_1_v = torch.randn(2, 2, 8)

    cache.write_slots(
        slot_mapping,
        layer_0_k,
        layer_0_v,
        layer_idx=0,
    )
    cache.write_slots(
        slot_mapping,
        layer_1_k,
        layer_1_v,
        layer_idx=1,
    )

    assert cache.k_cache.shape == (2, 2, 4, 2, 8)
    assert cache.v_cache.shape == (2, 2, 4, 2, 8)

    expected_locations = [(0, 1), (1, 2)]

    for token_index, (physical_block, block_offset) in enumerate(
        expected_locations
    ):
        torch.testing.assert_close(
            cache.k_cache[0, physical_block, block_offset],
            layer_0_k[token_index],
        )
        torch.testing.assert_close(
            cache.v_cache[0, physical_block, block_offset],
            layer_0_v[token_index],
        )
        torch.testing.assert_close(
            cache.k_cache[1, physical_block, block_offset],
            layer_1_k[token_index],
        )
        torch.testing.assert_close(
            cache.v_cache[1, physical_block, block_offset],
            layer_1_v[token_index],
        )


@pytest.mark.parametrize("layer_idx", [-1, 2])
def test_write_slots_rejects_out_of_range_layer_idx(layer_idx):
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )
    cache.k_cache.zero_()
    cache.v_cache.zero_()

    slot_mapping = torch.tensor([1], dtype=torch.long)
    k = torch.randn(1, 2, 8)
    v = torch.randn(1, 2, 8)
    k_before = cache.k_cache.clone()
    v_before = cache.v_cache.clone()

    with pytest.raises(IndexError, match="layer_idx"):
        cache.write_slots(
            slot_mapping,
            k,
            v,
            layer_idx=layer_idx,
        )

    torch.testing.assert_close(cache.k_cache, k_before)
    torch.testing.assert_close(cache.v_cache, v_before)


def test_get_layer_cache_returns_four_dimensional_views():
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=3,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )
    cache.k_cache[0].fill_(1.0)
    cache.v_cache[0].fill_(2.0)
    cache.k_cache[1].fill_(3.0)
    cache.v_cache[1].fill_(4.0)

    default_k, default_v = cache.get_layer_cache()
    layer_0_k, layer_0_v = cache.get_layer_cache(layer_idx=0)
    layer_1_k, layer_1_v = cache.get_layer_cache(layer_idx=1)

    expected_shape = (3, 4, 2, 8)
    assert default_k.shape == expected_shape
    assert default_v.shape == expected_shape
    assert layer_1_k.shape == expected_shape
    assert layer_1_v.shape == expected_shape

    torch.testing.assert_close(default_k, layer_0_k)
    torch.testing.assert_close(default_v, layer_0_v)
    torch.testing.assert_close(layer_0_k, torch.ones_like(layer_0_k))
    torch.testing.assert_close(layer_0_v, torch.full_like(layer_0_v, 2.0))
    torch.testing.assert_close(layer_1_k, torch.full_like(layer_1_k, 3.0))
    torch.testing.assert_close(layer_1_v, torch.full_like(layer_1_v, 4.0))

    layer_1_k[0, 0, 0, 0] = 9.0
    layer_1_v[0, 0, 0, 0] = 10.0
    assert cache.k_cache[1, 0, 0, 0, 0].item() == 9.0
    assert cache.v_cache[1, 0, 0, 0, 0].item() == 10.0

    for invalid_layer_idx in (-1, cache.num_layers):
        with pytest.raises(IndexError, match="layer_idx"):
            cache.get_layer_cache(layer_idx=invalid_layer_idx)
