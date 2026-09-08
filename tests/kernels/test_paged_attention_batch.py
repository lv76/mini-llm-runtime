import pytest
import torch

from mini_llm_runtime.kernels.reference.paged_attention import (
    paged_attention_decode_batch_parallel,
    paged_attention_decode_batch_reference,
)


@pytest.mark.parametrize(
    "num_query_heads,num_kv_heads,seq_lens,block_tables",
    [
        (4, 2, [3, 5], [[2, 0, -1], [1, 3, 4]]),
        (2, 2, [3, 5], [[2, 0, -1], [1, 3, 4]]),
        (4, 1, [3, 5], [[2, 0, -1], [1, 3, 4]]),
        (4, 2, [3, 4], [[2, 0], [1, 3]]),
    ],
    ids=["gqa-ragged", "mha-ragged", "mqa-ragged", "seq-length-equals-heads"],
)
def test_batch_decode_matches_request_reference(
    num_query_heads, num_kv_heads, seq_lens, block_tables
):
    torch.manual_seed(42)
    block_size = 2
    head_dim = 8
    q = torch.randn(len(seq_lens), num_query_heads, head_dim)
    k_cache = torch.randn(5, block_size, num_kv_heads, head_dim)
    v_cache = torch.randn_like(k_cache)

    expected = paged_attention_decode_batch_reference(
        q, k_cache, v_cache, block_tables, seq_lens, block_size
    )
    actual = paged_attention_decode_batch_parallel(
        q,
        k_cache,
        v_cache,
        torch.tensor(block_tables),
        torch.tensor(seq_lens),
        block_size,
    )

    assert actual.shape == q.shape
    torch.testing.assert_close(actual, expected)
