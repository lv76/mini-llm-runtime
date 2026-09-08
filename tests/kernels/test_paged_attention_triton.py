import pytest
import torch

from mini_llm_runtime.kernels.reference.paged_attention import (
    paged_attention_decode,
)
from mini_llm_runtime.kernels.triton.paged_attention import (
    paged_attention_decode_split_kv_block_local_triton,
    paged_attention_decode_split_kv_gqa_reuse_2head_triton,
    paged_attention_decode_split_kv_gqa_reuse_v2_triton,
    paged_attention_decode_split_kv_gqa_reuse_triton,
    paged_attention_decode_split_kv_triton,
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
    block_table=None,
):
    torch.manual_seed(0)

    required_blocks = (seq_len + block_size - 1) // block_size
    num_blocks = required_blocks + 3
    if block_table is None:
        block_table = torch.randperm(num_blocks)[:required_blocks].tolist()
    elif len(block_table) != required_blocks:
        raise ValueError("block_table must contain exactly required_blocks entries")

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


def _assert_split_kv_matches_reference_and_single_pass(
    *,
    seq_len,
    block_size,
    num_query_heads,
    num_kv_heads,
    head_dim,
    partition_size,
    block_n,
    block_table=None,
):
    q, k_cache, v_cache, block_table = _make_paged_attention_inputs(
        seq_len=seq_len,
        block_size=block_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_table=block_table,
    )

    reference_output = paged_attention_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        block_size=block_size,
    )
    single_pass_output = paged_attention_decode_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
    )
    split_output = paged_attention_decode_split_kv_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        partition_size=partition_size,
        block_n=block_n,
    )
    block_local_output = paged_attention_decode_split_kv_block_local_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        partition_size=partition_size,
        block_n=block_n,
    )

    assert split_output.shape == (num_query_heads, head_dim)
    assert torch.isfinite(split_output).all()
    assert block_local_output.shape == (num_query_heads, head_dim)
    assert torch.isfinite(block_local_output).all()
    torch.testing.assert_close(
        single_pass_output,
        reference_output,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        split_output,
        reference_output,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        split_output,
        single_pass_output,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        block_local_output,
        reference_output,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        block_local_output,
        split_output,
        rtol=1e-2,
        atol=1e-2,
    )


@CUDA_ONLY
@pytest.mark.parametrize(
    (
        "seq_len",
        "partition_size",
        "block_n",
        "num_query_heads",
        "num_kv_heads",
        "head_dim",
    ),
    [
        (65, 128, 64, 4, 2, 80),
        (4096, 512, 256, 4, 2, 80),
        (600, 512, 256, 4, 2, 80),
        (513, 512, 256, 4, 2, 80),
        (128, 512, 256, 4, 2, 80),
        (700, 384, 256, 4, 2, 80),
        (600, 512, 256, 4, 1, 80),
        (600, 512, 256, 4, 4, 80),
    ],
)
def test_paged_attention_decode_split_kv_matches_reference_and_single_pass(
    seq_len,
    partition_size,
    block_n,
    num_query_heads,
    num_kv_heads,
    head_dim,
):
    _assert_split_kv_matches_reference_and_single_pass(
        seq_len=seq_len,
        block_size=16,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        partition_size=partition_size,
        block_n=block_n,
    )


@CUDA_ONLY
def test_paged_attention_decode_split_kv_shuffled_physical_blocks():
    seq_len = 600
    block_size = 16
    required_blocks = (seq_len + block_size - 1) // block_size
    block_table = list(reversed(range(required_blocks)))

    _assert_split_kv_matches_reference_and_single_pass(
        seq_len=seq_len,
        block_size=block_size,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=80,
        partition_size=512,
        block_n=256,
        block_table=block_table,
    )


def _assert_gqa_reuse_matches_original_and_reference(
    *,
    seq_len,
    head_dim,
    block_table=None,
):
    block_size = 16
    q, k_cache, v_cache, block_table = _make_paged_attention_inputs(
        seq_len=seq_len,
        block_size=block_size,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=head_dim,
        block_table=block_table,
    )
    expected = paged_attention_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        block_size=block_size,
    )
    original = paged_attention_decode_split_kv_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        partition_size=128,
        block_n=256,
        num_warps=4,
    )
    actual = paged_attention_decode_split_kv_gqa_reuse_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        partition_size=128,
        block_n=256,
        num_warps=4,
    )
    actual_v2 = paged_attention_decode_split_kv_gqa_reuse_v2_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        partition_size=128,
        block_n=256,
        num_warps=4,
    )
    actual_2head = paged_attention_decode_split_kv_gqa_reuse_2head_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        partition_size=128,
        block_n=256,
        num_warps=4,
    )

    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert actual_v2.shape == expected.shape
    assert torch.isfinite(actual_v2).all()
    assert actual_2head.shape == expected.shape
    assert torch.isfinite(actual_2head).all()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual, original, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_v2, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_v2, original, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_2head, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_2head, original, rtol=1e-2, atol=1e-2)


@CUDA_ONLY
@pytest.mark.parametrize("seq_len", [127, 513])
@pytest.mark.parametrize("head_dim", [128, 160])
def test_paged_attention_decode_split_kv_gqa_reuse_matches_original_and_reference(
    seq_len,
    head_dim,
):
    _assert_gqa_reuse_matches_original_and_reference(
        seq_len=seq_len,
        head_dim=head_dim,
    )


@CUDA_ONLY
def test_paged_attention_decode_split_kv_gqa_reuse_shuffled_physical_blocks():
    seq_len = 529
    required_blocks = (seq_len + 15) // 16
    _assert_gqa_reuse_matches_original_and_reference(
        seq_len=seq_len,
        head_dim=128,
        block_table=list(reversed(range(required_blocks))),
    )


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


@pytest.mark.parametrize("partition_size", [0, -1])
def test_paged_attention_decode_split_kv_rejects_invalid_partition_size(
    partition_size,
):
    q, k_cache, v_cache = _cpu_inputs()

    with pytest.raises(
        ValueError,
        match="partition_size must be greater than 0",
    ):
        paged_attention_decode_split_kv_triton(
            q,
            k_cache,
            v_cache,
            [0],
            8,
            partition_size=partition_size,
        )


def test_paged_attention_decode_split_kv_rejects_zero_block_n():
    q, k_cache, v_cache = _cpu_inputs()

    with pytest.raises(ValueError, match="block_n must be greater than 0"):
        paged_attention_decode_split_kv_triton(
            q,
            k_cache,
            v_cache,
            [0],
            8,
            block_n=0,
        )


def test_paged_attention_decode_split_kv_rejects_invalid_block_shape():
    q, k_cache, v_cache = _cpu_inputs()

    with pytest.raises(ValueError, match="block_n must be a power of 2"):
        paged_attention_decode_split_kv_triton(
            q,
            k_cache,
            v_cache,
            [0],
            8,
            block_n=192,
        )


@pytest.mark.parametrize("num_warps", [0, 3, 16])
def test_paged_attention_decode_split_kv_rejects_invalid_num_warps(num_warps):
    q, k_cache, v_cache = _cpu_inputs()

    with pytest.raises(ValueError, match="num_warps must be one of"):
        paged_attention_decode_split_kv_triton(
            q,
            k_cache,
            v_cache,
            [0],
            8,
            num_warps=num_warps,
        )


@pytest.mark.parametrize(
    ("partition_size", "block_n", "message"),
    [
        (512, 4, "block_n must be divisible by block_size"),
        (500, 256, "partition_size must be divisible by block_size"),
    ],
)
def test_paged_attention_decode_split_kv_block_local_rejects_unaligned_tiles(
    partition_size,
    block_n,
    message,
):
    q, k_cache, v_cache = _cpu_inputs()

    with pytest.raises(ValueError, match=message):
        paged_attention_decode_split_kv_block_local_triton(
            q,
            k_cache,
            v_cache,
            [0],
            8,
            partition_size=partition_size,
            block_n=block_n,
        )


def test_paged_attention_decode_split_kv_gqa_reuse_rejects_other_group_size():
    q, k_cache, v_cache = _cpu_inputs(
        num_query_heads=4,
        num_kv_heads=2,
    )

    with pytest.raises(ValueError, match="requires group_size == 4"):
        paged_attention_decode_split_kv_gqa_reuse_triton(
            q,
            k_cache,
            v_cache,
            [0],
            8,
        )

    with pytest.raises(ValueError, match="requires group_size == 4"):
        paged_attention_decode_split_kv_gqa_reuse_v2_triton(
            q,
            k_cache,
            v_cache,
            [0],
            8,
        )

    with pytest.raises(ValueError, match="requires group_size == 4"):
        paged_attention_decode_split_kv_gqa_reuse_2head_triton(
            q,
            k_cache,
            v_cache,
            [0],
            8,
        )


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
