import triton.language as tl
import triton
import math
import torch


def paged_attention_decode_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table,
    seq_len: int,
) -> torch.Tensor:
    if q.ndim != 2:
        raise ValueError("q must have shape [Hq, D]")

    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "k_cache and v_cache must have shape "
            "[num_blocks, block_size, Hkv, D]"
        )

    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")

    num_query_heads, query_head_dim = q.shape
    _, block_size, num_kv_heads, head_dim = k_cache.shape

    if query_head_dim != head_dim:
        raise ValueError("q and KV cache must have the same head_dim")

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num_query_heads must be divisible by num_kv_heads"
        )

    if seq_len <= 0:
        raise ValueError("seq_len must be greater than 0")

    required_blocks = (seq_len + block_size - 1) // block_size
    if len(block_table) < required_blocks:
        raise ValueError(
            "block_table does not contain enough blocks for seq_len"
        )

    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()

    assert q.is_cuda
    assert k_cache.is_cuda and v_cache.is_cuda

    if not isinstance(block_table, torch.Tensor):
        block_table = torch.tensor(
            block_table,
            dtype=torch.int32,
            device=q.device,
        )
    else:
        block_table = block_table.to(
            device=q.device,
            dtype=torch.int32,
        ).contiguous()

    output = torch.empty_like(q)

    BLOCK_N = 32
    BLOCK_D = triton.next_power_of_2(head_dim)

    # 一个 program 负责一个 query head
    grid = (num_query_heads,)

    scale = 1.0 / math.sqrt(head_dim)

    _paged_attention_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_table,
        output,
        seq_len,
        num_query_heads,
        num_kv_heads,
        head_dim,
        block_size,
        scale,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )

    return output

@triton.jit
def _paged_attention_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    output_ptr,
    seq_len,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_head = tl.program_id(0)

    group_size = num_query_heads // num_kv_heads
    kv_head = query_head // group_size

    offs_d = tl.arange(0, BLOCK_D)
    dim_mask = offs_d < head_dim

    q = tl.load(
        q_ptr + query_head * head_dim + offs_d,
        mask=dim_mask,
        other=0.0,
    )

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start_n in tl.range(0, seq_len, BLOCK_N):
        positions = start_n + tl.arange(0, BLOCK_N)
        token_mask = positions < seq_len

        logical_blocks = positions // block_size
        offsets_in_block = positions % block_size

        physical_blocks = tl.load(
            block_table_ptr + logical_blocks,
            mask=token_mask,
            other=0,
        )

        token_slots = (
            physical_blocks[:, None] * block_size
            + offsets_in_block[:, None]
        )

        head_slots = (
            token_slots * num_kv_heads
            + kv_head
        )

        kv_offsets = (
            head_slots * head_dim
            + offs_d[None, :]
        )

        kv_mask = (
            token_mask[:, None]
            & dim_mask[None, :]
        )

        k_tile = tl.load(
            k_cache_ptr + kv_offsets,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)

        v_tile = tl.load(
            v_cache_ptr + kv_offsets,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)

        # [BLOCK_N, BLOCK_D] × [BLOCK_D]
        # -> [BLOCK_N]
        scores = tl.sum(
            k_tile * q[None, :],
            axis=1,
        ) * scale

        scores = tl.where(
            token_mask,
            scores,
            -float("inf"),
        )

        # Online softmax
        m_tile = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_tile)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        l_new = (
            alpha * l_i
            + tl.sum(p, axis=0)
        )

        acc = (
            alpha * acc
            + tl.sum(
                p[:, None] * v_tile,
                axis=0,
            )
        )

        m_i = m_new
        l_i = l_new

    output = acc / l_i

    tl.store(
        output_ptr
        + query_head * head_dim
        + offs_d,
        output,
        mask=dim_mask,
    )

