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
    return _paged_attention_decode_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        block_n=64,
        num_warps=4,
    )

def paged_attention_decode_split_kv_triton(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_len,
    partition_size=512,
    block_n=256,
    num_warps=4,
    _block_local=False,
    _gqa_reuse=False,
    _gqa_reuse_v2=False,
    _gqa_reuse_2head=False,
):
    if partition_size <= 0:
        raise ValueError("partition_size must be greater than 0")

    if block_n <= 0:
        raise ValueError("block_n must be greater than 0")

    if block_n & (block_n - 1):
        raise ValueError("block_n must be a power of 2")

    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")

    num_partitions = triton.cdiv(
        seq_len,
        partition_size,
    )

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

    if _block_local and block_n % block_size:
        raise ValueError("block_n must be divisible by block_size")

    if _block_local and partition_size % block_size:
        raise ValueError("partition_size must be divisible by block_size")

    if query_head_dim != head_dim:
        raise ValueError("q and KV cache must have the same head_dim")

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num_query_heads must be divisible by num_kv_heads"
        )

    group_size = num_query_heads // num_kv_heads
    if (_gqa_reuse or _gqa_reuse_v2 or _gqa_reuse_2head) and group_size != 4:
        raise ValueError("gqa-reuse stage1 requires group_size == 4")

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


    BLOCK_D = triton.next_power_of_2(head_dim)

    partial_m = torch.empty(
        (num_query_heads, num_partitions),
        dtype=torch.float32,
        device=q.device,
    )

    partial_l = torch.empty_like(partial_m)

    partial_acc = torch.empty(
        (
            num_query_heads,
            num_partitions,
            head_dim,
        ),
        dtype=torch.float32,
        device=q.device,
    )

    output = torch.empty_like(q)

    grid_stage1 = (
        (
            num_kv_heads * 2
            if _gqa_reuse_2head
            else (
                num_kv_heads
                if (_gqa_reuse or _gqa_reuse_v2)
                else num_query_heads
            )
        ),
        num_partitions,
    )

    stage1_kernel = (
        _paged_attention_split_kv_stage1_gqa_reuse_2head_kernel
        if _gqa_reuse_2head
        else (
            _paged_attention_split_kv_stage1_gqa_reuse_v2_kernel
            if _gqa_reuse_v2
            else (
                _paged_attention_split_kv_stage1_gqa_reuse_kernel
                if _gqa_reuse
                else (
                    _paged_attention_split_kv_stage1_block_local_kernel
                    if _block_local
                    else _paged_attention_split_kv_stage1_kernel
                )
            )
        )
    )

    stage1_kernel[
        grid_stage1
    ](
        q,
        k_cache,
        v_cache,
        block_table,
        partial_m,
        partial_l,
        partial_acc,
        seq_len,
        num_partitions,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        scale=1.0 / math.sqrt(head_dim),
        PARTITION_SIZE=partition_size,
        BLOCK_N=block_n,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )

    BLOCK_P = triton.next_power_of_2(
        num_partitions
    )

    grid_merge = (num_query_heads,)

    _paged_attention_split_kv_merge_kernel[
        grid_merge
    ](
        partial_m,
        partial_l,
        partial_acc,
        output,
        num_partitions,
        head_dim=head_dim,
        BLOCK_P=BLOCK_P,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )

    return output


def paged_attention_decode_split_kv_block_local_triton(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_len,
    partition_size=512,
    block_n=256,
    num_warps=4,
):
    """Run split-KV PagedAttention with block-local stage-1 traversal."""
    return paged_attention_decode_split_kv_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        partition_size=partition_size,
        block_n=block_n,
        num_warps=num_warps,
        _block_local=True,
    )


def paged_attention_decode_split_kv_gqa_reuse_triton(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_len,
    partition_size=512,
    block_n=256,
    num_warps=4,
):
    """Run split-KV PagedAttention with group-size-4 GQA KV reuse."""
    return paged_attention_decode_split_kv_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        partition_size=partition_size,
        block_n=block_n,
        num_warps=num_warps,
        _gqa_reuse=True,
    )


def paged_attention_decode_split_kv_gqa_reuse_v2_triton(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_len,
    partition_size=512,
    block_n=256,
    num_warps=4,
):
    """Run split-KV PagedAttention with scalarized group-size-4 GQA reuse."""
    return paged_attention_decode_split_kv_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        partition_size=partition_size,
        block_n=block_n,
        num_warps=num_warps,
        _gqa_reuse_v2=True,
    )


def paged_attention_decode_split_kv_gqa_reuse_2head_triton(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_len,
    partition_size=512,
    block_n=256,
    num_warps=4,
):
    """Run split-KV PagedAttention with two-query-head GQA KV reuse."""
    return paged_attention_decode_split_kv_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_len=seq_len,
        partition_size=partition_size,
        block_n=block_n,
        num_warps=num_warps,
        _gqa_reuse_2head=True,
    )


def _paged_attention_decode_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table,
    seq_len: int,
    *,
    block_n: int,
    num_warps: int,
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
        BLOCK_N=block_n,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
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



@triton.jit
def _paged_attention_split_kv_stage1_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    seq_len,
    num_partitions,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    scale: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_head = tl.program_id(0)
    partition_id = tl.program_id(1)

    group_size = num_query_heads // num_kv_heads
    kv_head = query_head // group_size

    # 当前 program 负责的 sequence 范围
    partition_start = partition_id * PARTITION_SIZE
    partition_end = partition_start + PARTITION_SIZE

    offs_d = tl.arange(0, BLOCK_D)
    dim_mask = offs_d < head_dim

    q = tl.load(
        q_ptr + query_head * head_dim + offs_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0

    acc = tl.zeros(
        (BLOCK_D,),
        dtype=tl.float32,
    )

    for start_n in tl.range(
        0,
        PARTITION_SIZE,
        BLOCK_N,
    ):
        positions = (
            partition_start
            + start_n
            + tl.arange(0, BLOCK_N)
        )

        token_mask = (
            (positions < seq_len)
            & (positions < partition_end)
        )

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

        scores = tl.sum(
            k_tile * q[None, :],
            axis=1,
        ) * scale

        scores = tl.where(
            token_mask,
            scores,
            -float("inf"),
        )

        m_tile = tl.max(scores, axis=0)

        m_new = tl.maximum(
            m_i,
            m_tile,
        )

        alpha = tl.exp(
            m_i - m_new
        )

        p = tl.exp(
            scores - m_new
        )

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

    # [Hq, num_partitions]
    partial_scalar_offset = (
        query_head * num_partitions
        + partition_id
    )

    tl.store(
        partial_m_ptr + partial_scalar_offset,
        m_i,
    )

    tl.store(
        partial_l_ptr + partial_scalar_offset,
        l_i,
    )

    # [Hq, num_partitions, D]
    partial_acc_offset = (
        partial_scalar_offset * head_dim
        + offs_d
    )

    tl.store(
        partial_acc_ptr + partial_acc_offset,
        acc,
        mask=dim_mask,
    )


@triton.jit
def _paged_attention_split_kv_stage1_block_local_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    seq_len,
    num_partitions,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    scale: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_head = tl.program_id(0)
    partition_id = tl.program_id(1)

    group_size = num_query_heads // num_kv_heads
    kv_head = query_head // group_size

    # 当前 program 负责的 sequence 范围
    partition_start = partition_id * PARTITION_SIZE
    partition_end = partition_start + PARTITION_SIZE

    offs_d = tl.arange(0, BLOCK_D)
    dim_mask = offs_d < head_dim

    q = tl.load(
        q_ptr + query_head * head_dim + offs_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0

    acc = tl.zeros(
        (BLOCK_D,),
        dtype=tl.float32,
    )

    NUM_BLOCKS_PER_TILE: tl.constexpr = BLOCK_N // block_size

    for start_n in tl.range(
        0,
        PARTITION_SIZE,
        BLOCK_N,
    ):
        tile_start = partition_start + start_n
        block_offs = tl.arange(0, NUM_BLOCKS_PER_TILE)
        logical_blocks = tile_start // block_size + block_offs

        block_positions = logical_blocks * block_size
        block_mask = (
            (block_positions < seq_len)
            & (block_positions < partition_end)
        )

        physical_blocks = tl.load(
            block_table_ptr + logical_blocks,
            mask=block_mask,
            other=0,
        )

        slots = tl.arange(0, block_size)
        positions_2d = (
            logical_blocks[:, None] * block_size
            + slots[None, :]
        )
        token_slots_2d = (
            physical_blocks[:, None] * block_size
            + slots[None, :]
        )
        token_mask_2d = (
            (positions_2d < seq_len)
            & (positions_2d < partition_end)
        )

        positions = tl.reshape(positions_2d, (BLOCK_N,))
        token_slots = tl.reshape(token_slots_2d, (BLOCK_N,))
        token_mask = tl.reshape(token_mask_2d, (BLOCK_N,))

        head_slots = (
            token_slots[:, None] * num_kv_heads
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

        scores = tl.sum(
            k_tile * q[None, :],
            axis=1,
        ) * scale

        scores = tl.where(
            token_mask,
            scores,
            -float("inf"),
        )

        m_tile = tl.max(scores, axis=0)

        m_new = tl.maximum(
            m_i,
            m_tile,
        )

        alpha = tl.exp(
            m_i - m_new
        )

        p = tl.exp(
            scores - m_new
        )

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

    # [Hq, num_partitions]
    partial_scalar_offset = (
        query_head * num_partitions
        + partition_id
    )

    tl.store(
        partial_m_ptr + partial_scalar_offset,
        m_i,
    )

    tl.store(
        partial_l_ptr + partial_scalar_offset,
        l_i,
    )

    # [Hq, num_partitions, D]
    partial_acc_offset = (
        partial_scalar_offset * head_dim
        + offs_d
    )

    tl.store(
        partial_acc_ptr + partial_acc_offset,
        acc,
        mask=dim_mask,
    )


@triton.jit
def _paged_attention_split_kv_stage1_gqa_reuse_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    seq_len,
    num_partitions,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    scale: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    kv_head = tl.program_id(0)
    partition_id = tl.program_id(1)

    GROUP_SIZE: tl.constexpr = num_query_heads // num_kv_heads
    query_head_base = kv_head * GROUP_SIZE
    query_heads = query_head_base + tl.arange(0, GROUP_SIZE)

    # 当前 program 负责的 sequence 范围
    partition_start = partition_id * PARTITION_SIZE
    partition_end = partition_start + PARTITION_SIZE

    offs_d = tl.arange(0, BLOCK_D)
    dim_mask = offs_d < head_dim
    query_head_mask = query_heads < num_query_heads

    q_offsets = (
        query_heads[:, None] * head_dim
        + offs_d[None, :]
    )
    q_mask = (
        query_head_mask[:, None]
        & dim_mask[None, :]
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full(
        (GROUP_SIZE,),
        -float("inf"),
        tl.float32,
    )
    l_i = tl.zeros(
        (GROUP_SIZE,),
        dtype=tl.float32,
    )
    acc = tl.zeros(
        (GROUP_SIZE, BLOCK_D),
        dtype=tl.float32,
    )

    for start_n in tl.range(
        0,
        PARTITION_SIZE,
        BLOCK_N,
    ):
        positions = (
            partition_start
            + start_n
            + tl.arange(0, BLOCK_N)
        )

        token_mask = (
            (positions < seq_len)
            & (positions < partition_end)
        )

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

        scores = tl.sum(
            q[:, None, :] * k_tile[None, :, :],
            axis=2,
        ) * scale

        scores = tl.where(
            token_mask[None, :],
            scores,
            -float("inf"),
        )

        m_tile = tl.max(scores, axis=1)

        m_new = tl.maximum(
            m_i,
            m_tile,
        )

        alpha = tl.exp(
            m_i - m_new
        )

        p = tl.exp(
            scores - m_new[:, None]
        )

        l_new = (
            alpha * l_i
            + tl.sum(p, axis=1)
        )

        acc = (
            alpha[:, None] * acc
            + tl.sum(
                p[:, :, None] * v_tile[None, :, :],
                axis=1,
            )
        )

        m_i = m_new
        l_i = l_new

    # [Hq, num_partitions]
    partial_scalar_offsets = (
        query_heads * num_partitions
        + partition_id
    )

    tl.store(
        partial_m_ptr + partial_scalar_offsets,
        m_i,
        mask=query_head_mask,
    )

    tl.store(
        partial_l_ptr + partial_scalar_offsets,
        l_i,
        mask=query_head_mask,
    )

    # [Hq, num_partitions, D]
    partial_acc_offsets = (
        partial_scalar_offsets[:, None] * head_dim
        + offs_d[None, :]
    )

    tl.store(
        partial_acc_ptr + partial_acc_offsets,
        acc,
        mask=q_mask,
    )


@triton.jit
def _paged_attention_split_kv_stage1_gqa_reuse_v2_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    seq_len,
    num_partitions,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    scale: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    kv_head = tl.program_id(0)
    partition_id = tl.program_id(1)

    GROUP_SIZE: tl.constexpr = num_query_heads // num_kv_heads
    query_head_base = kv_head * GROUP_SIZE
    query_head0 = query_head_base
    query_head1 = query_head_base + 1
    query_head2 = query_head_base + 2
    query_head3 = query_head_base + 3

    # 当前 program 负责的 sequence 范围
    partition_start = partition_id * PARTITION_SIZE
    partition_end = partition_start + PARTITION_SIZE

    offs_d = tl.arange(0, BLOCK_D)
    dim_mask = offs_d < head_dim

    q0 = tl.load(
        q_ptr + query_head0 * head_dim + offs_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    q1 = tl.load(
        q_ptr + query_head1 * head_dim + offs_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    q2 = tl.load(
        q_ptr + query_head2 * head_dim + offs_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    q3 = tl.load(
        q_ptr + query_head3 * head_dim + offs_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    m0 = -float("inf")
    m1 = -float("inf")
    m2 = -float("inf")
    m3 = -float("inf")
    l0 = 0.0
    l1 = 0.0
    l2 = 0.0
    l3 = 0.0
    acc0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start_n in tl.range(
        0,
        PARTITION_SIZE,
        BLOCK_N,
    ):
        positions = (
            partition_start
            + start_n
            + tl.arange(0, BLOCK_N)
        )

        token_mask = (
            (positions < seq_len)
            & (positions < partition_end)
        )

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

        scores0 = tl.sum(
            k_tile * q0[None, :],
            axis=1,
        ) * scale
        scores0 = tl.where(
            token_mask,
            scores0,
            -float("inf"),
        )
        m_tile0 = tl.max(scores0, axis=0)
        m_new0 = tl.maximum(m0, m_tile0)
        alpha0 = tl.exp(m0 - m_new0)
        p0 = tl.exp(scores0 - m_new0)
        l_new0 = alpha0 * l0 + tl.sum(p0, axis=0)
        acc0 = (
            alpha0 * acc0
            + tl.sum(
                p0[:, None] * v_tile,
                axis=0,
            )
        )
        m0 = m_new0
        l0 = l_new0

        scores1 = tl.sum(
            k_tile * q1[None, :],
            axis=1,
        ) * scale
        scores1 = tl.where(
            token_mask,
            scores1,
            -float("inf"),
        )
        m_tile1 = tl.max(scores1, axis=0)
        m_new1 = tl.maximum(m1, m_tile1)
        alpha1 = tl.exp(m1 - m_new1)
        p1 = tl.exp(scores1 - m_new1)
        l_new1 = alpha1 * l1 + tl.sum(p1, axis=0)
        acc1 = (
            alpha1 * acc1
            + tl.sum(
                p1[:, None] * v_tile,
                axis=0,
            )
        )
        m1 = m_new1
        l1 = l_new1

        scores2 = tl.sum(
            k_tile * q2[None, :],
            axis=1,
        ) * scale
        scores2 = tl.where(
            token_mask,
            scores2,
            -float("inf"),
        )
        m_tile2 = tl.max(scores2, axis=0)
        m_new2 = tl.maximum(m2, m_tile2)
        alpha2 = tl.exp(m2 - m_new2)
        p2 = tl.exp(scores2 - m_new2)
        l_new2 = alpha2 * l2 + tl.sum(p2, axis=0)
        acc2 = (
            alpha2 * acc2
            + tl.sum(
                p2[:, None] * v_tile,
                axis=0,
            )
        )
        m2 = m_new2
        l2 = l_new2

        scores3 = tl.sum(
            k_tile * q3[None, :],
            axis=1,
        ) * scale
        scores3 = tl.where(
            token_mask,
            scores3,
            -float("inf"),
        )
        m_tile3 = tl.max(scores3, axis=0)
        m_new3 = tl.maximum(m3, m_tile3)
        alpha3 = tl.exp(m3 - m_new3)
        p3 = tl.exp(scores3 - m_new3)
        l_new3 = alpha3 * l3 + tl.sum(p3, axis=0)
        acc3 = (
            alpha3 * acc3
            + tl.sum(
                p3[:, None] * v_tile,
                axis=0,
            )
        )
        m3 = m_new3
        l3 = l_new3

    partial_scalar_offset0 = query_head0 * num_partitions + partition_id
    partial_scalar_offset1 = query_head1 * num_partitions + partition_id
    partial_scalar_offset2 = query_head2 * num_partitions + partition_id
    partial_scalar_offset3 = query_head3 * num_partitions + partition_id

    tl.store(partial_m_ptr + partial_scalar_offset0, m0)
    tl.store(partial_m_ptr + partial_scalar_offset1, m1)
    tl.store(partial_m_ptr + partial_scalar_offset2, m2)
    tl.store(partial_m_ptr + partial_scalar_offset3, m3)

    tl.store(partial_l_ptr + partial_scalar_offset0, l0)
    tl.store(partial_l_ptr + partial_scalar_offset1, l1)
    tl.store(partial_l_ptr + partial_scalar_offset2, l2)
    tl.store(partial_l_ptr + partial_scalar_offset3, l3)

    partial_acc_offset0 = partial_scalar_offset0 * head_dim + offs_d
    partial_acc_offset1 = partial_scalar_offset1 * head_dim + offs_d
    partial_acc_offset2 = partial_scalar_offset2 * head_dim + offs_d
    partial_acc_offset3 = partial_scalar_offset3 * head_dim + offs_d

    tl.store(
        partial_acc_ptr + partial_acc_offset0,
        acc0,
        mask=dim_mask,
    )
    tl.store(
        partial_acc_ptr + partial_acc_offset1,
        acc1,
        mask=dim_mask,
    )
    tl.store(
        partial_acc_ptr + partial_acc_offset2,
        acc2,
        mask=dim_mask,
    )
    tl.store(
        partial_acc_ptr + partial_acc_offset3,
        acc3,
        mask=dim_mask,
    )


@triton.jit
def _paged_attention_split_kv_stage1_gqa_reuse_2head_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    seq_len,
    num_partitions,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    scale: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program_group = tl.program_id(0)
    partition_id = tl.program_id(1)

    kv_head = program_group // 2
    subgroup = program_group % 2
    query_head_base = kv_head * 4 + subgroup * 2
    query_head0 = query_head_base
    query_head1 = query_head_base + 1

    # 当前 program 负责的 sequence 范围
    partition_start = partition_id * PARTITION_SIZE
    partition_end = partition_start + PARTITION_SIZE

    offs_d = tl.arange(0, BLOCK_D)
    dim_mask = offs_d < head_dim

    q0 = tl.load(
        q_ptr + query_head0 * head_dim + offs_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    q1 = tl.load(
        q_ptr + query_head1 * head_dim + offs_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    m0 = -float("inf")
    m1 = -float("inf")
    l0 = 0.0
    l1 = 0.0
    acc0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start_n in tl.range(
        0,
        PARTITION_SIZE,
        BLOCK_N,
    ):
        positions = (
            partition_start
            + start_n
            + tl.arange(0, BLOCK_N)
        )

        token_mask = (
            (positions < seq_len)
            & (positions < partition_end)
        )

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

        scores0 = tl.sum(
            k_tile * q0[None, :],
            axis=1,
        ) * scale
        scores0 = tl.where(
            token_mask,
            scores0,
            -float("inf"),
        )
        m_tile0 = tl.max(scores0, axis=0)
        m_new0 = tl.maximum(m0, m_tile0)
        alpha0 = tl.exp(m0 - m_new0)
        p0 = tl.exp(scores0 - m_new0)
        l_new0 = alpha0 * l0 + tl.sum(p0, axis=0)
        acc0 = (
            alpha0 * acc0
            + tl.sum(
                p0[:, None] * v_tile,
                axis=0,
            )
        )
        m0 = m_new0
        l0 = l_new0

        scores1 = tl.sum(
            k_tile * q1[None, :],
            axis=1,
        ) * scale
        scores1 = tl.where(
            token_mask,
            scores1,
            -float("inf"),
        )
        m_tile1 = tl.max(scores1, axis=0)
        m_new1 = tl.maximum(m1, m_tile1)
        alpha1 = tl.exp(m1 - m_new1)
        p1 = tl.exp(scores1 - m_new1)
        l_new1 = alpha1 * l1 + tl.sum(p1, axis=0)
        acc1 = (
            alpha1 * acc1
            + tl.sum(
                p1[:, None] * v_tile,
                axis=0,
            )
        )
        m1 = m_new1
        l1 = l_new1

    partial_scalar_offset0 = query_head0 * num_partitions + partition_id
    partial_scalar_offset1 = query_head1 * num_partitions + partition_id

    tl.store(partial_m_ptr + partial_scalar_offset0, m0)
    tl.store(partial_m_ptr + partial_scalar_offset1, m1)
    tl.store(partial_l_ptr + partial_scalar_offset0, l0)
    tl.store(partial_l_ptr + partial_scalar_offset1, l1)

    partial_acc_offset0 = partial_scalar_offset0 * head_dim + offs_d
    partial_acc_offset1 = partial_scalar_offset1 * head_dim + offs_d

    tl.store(
        partial_acc_ptr + partial_acc_offset0,
        acc0,
        mask=dim_mask,
    )
    tl.store(
        partial_acc_ptr + partial_acc_offset1,
        acc1,
        mask=dim_mask,
    )


@triton.jit
def _paged_attention_split_kv_merge_kernel(
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    output_ptr,
    num_partitions,
    head_dim: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_head = tl.program_id(0)

    offs_p = tl.arange(0, BLOCK_P)
    offs_d = tl.arange(0, BLOCK_D)

    partition_mask = offs_p < num_partitions
    dim_mask = offs_d < head_dim

    scalar_offsets = (
        query_head * num_partitions
        + offs_p
    )

    partial_m = tl.load(
        partial_m_ptr + scalar_offsets,
        mask=partition_mask,
        other=-float("inf"),
    )

    partial_l = tl.load(
        partial_l_ptr + scalar_offsets,
        mask=partition_mask,
        other=0.0,
    )

    # 全局 max
    m_global = tl.max(
        partial_m,
        axis=0,
    )

    alpha = tl.exp(
        partial_m - m_global
    )

    alpha = tl.where(
        partition_mask,
        alpha,
        0.0,
    )

    # 全局 softmax denominator
    l_global = tl.sum(
        alpha * partial_l,
        axis=0,
    )

    acc_offsets = (
        (
            query_head * num_partitions
            + offs_p[:, None]
        ) * head_dim
        + offs_d[None, :]
    )

    partial_acc = tl.load(
        partial_acc_ptr + acc_offsets,
        mask=(
            partition_mask[:, None]
            & dim_mask[None, :]
        ),
        other=0.0,
    )

    acc_global = tl.sum(
        alpha[:, None] * partial_acc,
        axis=0,
    )

    output = acc_global / l_global

    tl.store(
        output_ptr
        + query_head * head_dim
        + offs_d,
        output,
        mask=dim_mask,
    )
