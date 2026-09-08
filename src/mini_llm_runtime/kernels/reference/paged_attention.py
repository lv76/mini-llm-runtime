import torch
import math

from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache

def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: list[int],
    seq_len: int,
    block_size: int,
):
    if q.ndim != 2:
        raise ValueError("q must have shape [Hq, D]")

    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "k_cache and v_cache must have shape "
            "[num_blocks, block_size, Hkv, D]"
        )

    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")

    num_query_heads = q.shape[0]
    num_kv_heads = k_cache.shape[2]
    head_dim = q.shape[-1]

    if head_dim != k_cache.shape[-1]:
        raise ValueError(
            "q and KV cache must have the same head_dim"
        )

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num_query_heads must be divisible by num_kv_heads"
        )

    if seq_len <= 0:
        raise ValueError("seq_len must be greater than 0")

    required_blocks = (
        seq_len + block_size - 1
    ) // block_size

    if len(block_table) < required_blocks:
        raise ValueError(
            "block_table does not contain enough blocks "
            "for seq_len"
    )

    if block_size != k_cache.shape[1]:
        raise ValueError(
            "block_size does not match KV cache layout"
        )

    scores= torch.empty(
        q.shape[0],
        seq_len,
        dtype=q.dtype,
        device=q.device
    )
    scale = 1.0 / math.sqrt(q.shape[-1])
    logical_k = []
    logical_v = []

    group_size = num_query_heads // num_kv_heads

    for position in range(seq_len):

        logical_block = position // block_size
        offset = position % block_size
        physical_block = block_table[logical_block]

        k = k_cache[physical_block][offset] #[H, D]
        v = v_cache[physical_block][offset]

        logical_k.append(k)
        logical_v.append(v)

    k = torch.stack(logical_k, dim=0)
    v = torch.stack(logical_v, dim=0) #[S, H, D]

    k = torch.einsum('SHD -> HSD',k) #[H, S, D]
    v = torch.einsum('SHD -> HSD',v)


    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=0)
        v = v.repeat_interleave(group_size, dim=0)

    q = q.unsqueeze(dim=1) #[H, 1, D]

    scores = q @ k.transpose(-2,-1) * scale

    probs = torch.softmax(
      scores,
      dim=-1,
      dtype=torch.float32,
    ).to(v.dtype) #[H, 1, S]
    
    output = probs @ v ##[H, 1, D]

    return output.squeeze(1)


#reference
def paged_attention_decode_batch_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: list[list[int]],
    seq_lens: list[int],
    block_size: int,
):
    if q.ndim != 3:
        raise ValueError("q must have shape [B, Hq, D]")

    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "k_cache and v_cache must have shape "
            "[num_blocks, block_size, Hkv, D]"
        )

    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")

    num_query_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = q.shape[-1]

    if head_dim != k_cache.shape[-1]:
        raise ValueError(
            "q and KV cache must have the same head_dim"
        )

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num_query_heads must be divisible by num_kv_heads"
        )

    if len(block_tables) != q.shape[0]:
        raise ValueError(
            "block_tables length must match batch size of q"
        )

    if len(seq_lens) != q.shape[0]:
        raise ValueError(
            "seq_lens length must match batch size of q"
        )

    for i in range(q.shape[0]):
        required_blocks = (
            seq_lens[i] + block_size - 1
        ) // block_size

        if len(block_tables[i]) < required_blocks:
            raise ValueError(
                f"block_table for batch {i} does not contain enough blocks "
                f"for seq_len {seq_lens[i]}"
            )

    outputs = []
    for i in range(q.shape[0]):
        output = paged_attention_decode(
            q=q[i],
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_tables[i],
            seq_len=seq_lens[i],
            block_size=block_size,
        )
        outputs.append(output)

    return torch.stack(outputs, dim=0)

def paged_attention_decode_batch_parallel(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
):
    if q.ndim != 3:
            raise ValueError("q must have shape [B, Hq, D]")
    
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "k_cache and v_cache must have shape "
            "[num_blocks, block_size, Hkv, D]"
        )
    
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    
    num_query_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = q.shape[-1]
    
    if head_dim != k_cache.shape[-1]:
        raise ValueError(
            "q and KV cache must have the same head_dim"
        )
    
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num_query_heads must be divisible by num_kv_heads"
        )
    
    if len(block_tables) != q.shape[0]:
        raise ValueError(
            "block_tables length must match batch size of q"
        )
    
    if len(seq_lens) != q.shape[0]:
        raise ValueError(
            "seq_lens length must match batch size of q"
        )

    max_seq_len = seq_lens.max().item()

    position_ids = torch.arange(max_seq_len, device=block_tables.device)#(max_seq_len)
    logical_block_ids = position_ids // block_size
    offsets = position_ids % block_size

    batch_indices = torch.arange(q.shape[0], device=q.device).unsqueeze(1)#(B,1)
    physical_block = block_tables[batch_indices, logical_block_ids]#(B,max_seq_len)
    valid_mask = position_ids.unsqueeze(0) < seq_lens.unsqueeze(1)#(B,max_seq_len)

    safe_physical_block = physical_block.masked_fill(~valid_mask, 0)

    gathered_k = k_cache[safe_physical_block, offsets.unsqueeze(0)] #(B,max_seq_len,Hkv,D)
    gathered_v = v_cache[safe_physical_block, offsets.unsqueeze(0)] #(B,max_seq_len,Hkv,D)

    #处理GQA
    group_size = num_query_heads // num_kv_heads
    gathered_k = gathered_k.transpose(1, 2).repeat_interleave(group_size, dim=1) #(B,Hq,max_seq_len,D)
    gathered_v = gathered_v.transpose(1, 2).repeat_interleave(group_size, dim=1) #(B,Hq,max_seq_len,D)

    k = gathered_k #(B,Hq,max_seq_len,D)
    v = gathered_v 

    Query = q.unsqueeze(2) #(B,Hq,1,D)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(Query, k.transpose(-2, -1)) * scale #(B,Hq,1,max_seq_len)

    scores = scores.masked_fill(~valid_mask[:,None,None,:], float('-inf'))

    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype) #(B,Hq,1,max_seq_len)

    output = torch.matmul(probs, v) #(B,Hq,1,D)
    return output.squeeze(2) #(B,Hq,D)


    

    


    
def test_paged_attention_decode_rejects_invalid_head_ratio():
    q = torch.randn(3, 8)

    k_cache = torch.randn(2, 4, 2, 8)
    v_cache = torch.randn_like(k_cache)

    with pytest.raises(
        ValueError,
        match="num_query_heads must be divisible",
    ):
        paged_attention_decode(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=[0],
            seq_len=4,
            block_size=4,
        )
