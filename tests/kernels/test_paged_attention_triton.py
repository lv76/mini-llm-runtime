import pytest
import torch

from mini_llm_runtime.kernels.reference.paged_attention import (
    paged_attention_decode,
)
from mini_llm_runtime.kernels.triton.paged_attention import (
    paged_attention_decode_triton,
)


CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def _make_paged_attention_inputs(
    *,
    seq_len,
    block_size,
    num_query_heads,
    num_kv_heads,
    head_dim,
):
    torch.manual_seed(0)

    required_blocks = (seq_len + block_size - 1) // block_size
    num_blocks = required_blocks + 3
    block_table = torch.randperm(num_blocks)[:required_blocks].tolist()

    q = torch.randn(
        num_query_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    logical_k = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    logical_v = torch.randn_like(logical_k)
    k_cache = torch.empty(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    v_cache = torch.empty_like(k_cache)

    for position in range(seq_len):
        logical_block = position // block_size
        offset = position % block_size
        physical_block = block_table[logical_block]
        k_cache[physical_block, offset].copy_(logical_k[position])
        v_cache[physical_block, offset].copy_(logical_v[position])

    return q, k_cache, v_cache, block_table


def _assert_triton_matches_reference(
    *,
    seq_len,
    block_size,
    num_query_heads,
    num_kv_heads,
    head_dim,
):
    q, k_cache, v_cache, block_table = _make_paged_attention_inputs(
        seq_len=seq_len,
        block_size=block_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    expected = paged_attention_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        block_size=block_size,
    )
    actual = paged_attention_decode_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
    )

    assert actual.shape == (num_query_heads, head_dim)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@CUDA_ONLY
def test_paged_attention_decode_triton_matches_reference():
    _assert_triton_matches_reference(
        seq_len=6,
        block_size=4,
        num_query_heads=2,
        num_kv_heads=2,
        head_dim=8,
    )


@CUDA_ONLY
def test_paged_attention_decode_triton_multiple_tiles():
    _assert_triton_matches_reference(
        seq_len=70,
        block_size=8,
        num_query_heads=2,
        num_kv_heads=2,
        head_dim=16,
    )


@CUDA_ONLY
def test_paged_attention_decode_triton_gqa():
    _assert_triton_matches_reference(
        seq_len=70,
        block_size=8,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
    )


@CUDA_ONLY
def test_paged_attention_decode_triton_mqa():
    _assert_triton_matches_reference(
        seq_len=70,
        block_size=8,
        num_query_heads=4,
        num_kv_heads=1,
        head_dim=16,
    )


@CUDA_ONLY
@pytest.mark.parametrize("seq_len", [1, 31, 32, 33, 63, 64, 65])
def test_paged_attention_decode_triton_seq_len_tile_boundaries(seq_len):
    _assert_triton_matches_reference(
        seq_len=seq_len,
        block_size=8,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
    )


@CUDA_ONLY
@pytest.mark.parametrize("head_dim", [40, 80, 96])
def test_paged_attention_decode_triton_non_power_of_two_head_dim(head_dim):
    _assert_triton_matches_reference(
        seq_len=70,
        block_size=8,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=head_dim,
    )


def _cpu_inputs(*, num_query_heads=4, num_kv_heads=2, q_head_dim=16):
    q = torch.randn(num_query_heads, q_head_dim)
    k_cache = torch.randn(2, 8, num_kv_heads, 16)
    return q, k_cache, torch.randn_like(k_cache)


def test_paged_attention_decode_triton_rejects_invalid_head_ratio():
    q, k_cache, v_cache = _cpu_inputs(
        num_query_heads=3,
        num_kv_heads=2,
    )

    with pytest.raises(ValueError, match="must be divisible"):
        paged_attention_decode_triton(q, k_cache, v_cache, [0], 8)


def test_paged_attention_decode_triton_rejects_head_dim_mismatch():
    q, k_cache, v_cache = _cpu_inputs(q_head_dim=8)

    with pytest.raises(ValueError, match="same head_dim"):
        paged_attention_decode_triton(q, k_cache, v_cache, [0], 8)


def test_paged_attention_decode_triton_rejects_kv_shape_mismatch():
    q, k_cache, _ = _cpu_inputs()
    v_cache = torch.randn(2, 8, 2, 8)

    with pytest.raises(ValueError, match="must have the same shape"):
        paged_attention_decode_triton(q, k_cache, v_cache, [0], 8)


def test_paged_attention_decode_triton_rejects_short_block_table():
    q, k_cache, v_cache = _cpu_inputs()

    with pytest.raises(ValueError, match="does not contain enough blocks"):
        paged_attention_decode_triton(q, k_cache, v_cache, [0], 9)


@pytest.mark.parametrize("seq_len", [0, -1])
def test_paged_attention_decode_triton_rejects_non_positive_seq_len(seq_len):
    q, k_cache, v_cache = _cpu_inputs()

    with pytest.raises(ValueError, match="must be greater than 0"):
        paged_attention_decode_triton(q, k_cache, v_cache, [0], seq_len)


