from dataclasses import dataclass
from typing import Any
import torch.nn.functional as F

import torch
import torch.nn as nn
from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache
from mini_llm_runtime.kernels.reference.paged_attention import (
    paged_attention_decode,
    paged_attention_decode_batch_parallel,
    paged_attention_prefill_chunk_tiled,
)

@dataclass(frozen=True)
class TinyQwen2Config:
    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 128
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TinyQwen2Config":
        supported_fields = {
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "tie_word_embeddings",
            "bos_token_id",
            "eos_token_id",
        }

        config_kwargs = {
              key: value
              for key, value in config_dict.items()
              if key in supported_fields
        }
        return cls(**config_kwargs)

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
        )
        for field_name in positive_integer_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than 0")

        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be greater than 0")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be greater than 0")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_key_value_heads"
            )

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class Qwen2RMSNorm(nn.Module):
    """Root mean square normalization used by Qwen2."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than 0")
        if eps <= 0:
            raise ValueError("eps must be greater than 0")

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                "hidden_states last dimension must match hidden_size "
                f"({self.weight.shape[0]}), got {hidden_states.shape[-1]}"
            )

        input_dtype = hidden_states.dtype
        statistics_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        hidden_states_for_norm = hidden_states.to(statistics_dtype)
        variance = hidden_states_for_norm.square().mean(dim=-1, keepdim=True)
        normalized = hidden_states_for_norm * torch.rsqrt(
            variance + self.variance_epsilon
        )
        #return (self.weight.to(statistics_dtype) * normalized).to(input_dtype)
        return (self.weight.to(input_dtype) * normalized.to(input_dtype))

class Qwen2RotaryEmbedding(nn.Module):
    """Generate Qwen2 rotary position cosine and sine tensors."""

    def __init__(self, head_dim: int, theta: float = 1_000_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if theta <= 0:
            raise ValueError("theta must be greater than 0")

        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.head_dim = head_dim
        self.theta = theta
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be a floating-point tensor")
        if position_ids.ndim == 0:
            raise ValueError("position_ids must have at least one dimension")
        if tuple(hidden_states.shape[:-2]) != tuple(position_ids.shape):
            raise ValueError(
                "position_ids shape must match hidden_states dimensions before "
                f"the head dimensions, got {tuple(position_ids.shape)} and "
                f"{tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.head_dim:
            raise ValueError(
                "hidden_states last dimension must match head_dim "
                f"({self.head_dim}), got {hidden_states.shape[-1]}"
            )

        frequencies = position_ids.to(
            device=hidden_states.device,
            dtype=torch.float32,
        ).unsqueeze(-1) * self.inv_freq.to(hidden_states.device)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return (
            embeddings.cos().to(hidden_states.dtype),
            embeddings.sin().to(hidden_states.dtype),
        )


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Rotate the two contiguous halves of the last dimension."""

    if hidden_states.shape[-1] % 2 != 0:
        raise ValueError("hidden_states last dimension must be even")
    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to Q and K in ``[..., heads, head_dim]`` layout."""

    if query.ndim < 2 or key.ndim < 2:
        raise ValueError("query and key must include head and head_dim dimensions")
    if query.shape[:-2] != key.shape[:-2]:
        raise ValueError("query and key token dimensions must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head_dim must match")
    expected_rotary_shape = query.shape[:-2] + (query.shape[-1],)
    if cos.shape != expected_rotary_shape or sin.shape != expected_rotary_shape:
        raise ValueError(
            "cos and sin shape must match token dimensions plus head_dim, "
            f"expected {tuple(expected_rotary_shape)}"
        )

    cos = cos.to(query.dtype).unsqueeze(-2)
    sin = sin.to(query.dtype).unsqueeze(-2)

    query_embed = query * cos + rotate_half(query) * sin
    key_embed = key * cos + rotate_half(key) * sin

    return query_embed, key_embed

class Qwen2QKVProjection(nn.Module):
    def __init__(self, config:TinyQwen2Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim) 
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim) # 64 -> 2* (64//4)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim)


    def forward(
    self,
    hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"Unexpect input shape")
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        leading_shape = hidden_states.shape[:-1]
        query = query.reshape(leading_shape + (self.num_attention_heads, self.hidden_dim))
        key = key.reshape(leading_shape + (self.num_key_value_heads, self.hidden_dim))
        value = value.reshape(leading_shape + (self.num_key_value_heads, self.hidden_dim))

        return query, key, value

def repeat_kv(
    hidden_states: torch.Tensor,
    num_key_value_groups: int,     
):
    if num_key_value_groups <= 0:
        raise ValueError
    if hidden_states.ndim < 2:
        raise ValueError
    if num_key_value_groups == 1:
      return hidden_states

    return torch.repeat_interleave(
      hidden_states,
      repeats=num_key_value_groups,
      dim=-2,
    )
    

class Qwen2Attention(nn.Module):
    def __init__(self, config: TinyQwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5 
        self.qkv_proj = Qwen2QKVProjection(config)
        self.rotary_embedding = Qwen2RotaryEmbedding(
            head_dim=config.head_dim,
            theta=config.rope_theta,
        )

        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "[batch_size, seq_len, hidden_size], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states last dimension must match hidden_size "
                f"({self.hidden_size}), got {hidden_states.shape[-1]}"
            )
        if position_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                "position_ids shape must match hidden_states batch and "
                f"sequence dimensions {tuple(hidden_states.shape[:2])}, "
                f"got {tuple(position_ids.shape)}"
            )
        query, key, value = self.qkv_proj(hidden_states)
        cos, sin = self.rotary_embedding(
            query,
            position_ids,
        )

        query, key = apply_rotary_pos_emb(
            query,
            key,
            cos,
            sin,
        )

        attention_output = self._dense_attention(query, key, value)

        attention_output = self.o_proj(attention_output)
        return attention_output

    #reference
    def _dense_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        seq_len = query.shape[1]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = query @ key.transpose(-2, -1)
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=query.device,
            )
        )
        scores = scores * self.scale
        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf"),
        )
            
        softmax_dtype = (
            torch.float32
            if scores.dtype in (torch.float16, torch.bfloat16)
            else scores.dtype
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1,
            dtype=softmax_dtype,
        ).to(value.dtype) 

        attention_output = attention_weights @ value
        attention_output = attention_output.transpose(1, 2)
        batch_size, seq_len, _, _ = attention_output.shape
        attention_output = attention_output.reshape(
            batch_size,
            seq_len,
            self.hidden_size,
        )
        return attention_output

    def forward_prefill_with_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: PagedKVCache,
        layer_idx: int,
        slot_mapping: torch.Tensor,
    )-> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "[1, seq_len, hidden_size], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] != 1:
            raise ValueError(
                "prefill with cache currently supports exactly one request, "
                f"got batch_size={hidden_states.shape[0]}"
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states last dimension must match hidden_size "
                f"({self.hidden_size}), got {hidden_states.shape[-1]}"
            )
        if position_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                "position_ids shape must match hidden_states batch and "
                f"sequence dimensions {tuple(hidden_states.shape[:2])}, "
                f"got {tuple(position_ids.shape)}"
            )
        if slot_mapping.ndim != 1:
            raise ValueError(
                "slot_mapping must have shape [seq_len], "
                f"got {tuple(slot_mapping.shape)}"
            )
        if slot_mapping.shape[0] != hidden_states.shape[1]:
            raise ValueError(
                "slot_mapping length must match sequence length "
                f"({hidden_states.shape[1]}), got {slot_mapping.shape[0]}"
            )
        query, key, value = self.qkv_proj(hidden_states)
        cos, sin = self.rotary_embedding(
            query,
            position_ids,
        )

        query, key = apply_rotary_pos_emb(
            query,
            key,
            cos,
            sin
        )   #hidden_states: [1, S, hidden_size]
            #query: [1, S, num_attention_heads, head_dim]
            #key:   [1, S, num_key_value_heads, head_dim]
            #value: [1, S, num_key_value_heads, head_dim]

        kv_cache.write_slots(
            slot_mapping=slot_mapping,
            k = key.squeeze(0),
            v = value.squeeze(0),
            layer_idx=layer_idx
        )
        attention_output = self._dense_attention(
            query,
            key,
            value,
        )
        return self.o_proj(attention_output)

    def forward_decode_with_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: PagedKVCache,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        block_table: list[int],
        context_len: int,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [1, 1, hidden_size], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] != 1:
            raise ValueError(
                "decode with cache currently supports exactly one request, "
                f"got batch_size={hidden_states.shape[0]}"
            )
        if hidden_states.shape[1] != 1:
            raise ValueError(
                "decode with cache requires exactly one input token, "
                f"got {hidden_states.shape[1]}"
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states last dimension must match hidden_size "
                f"({self.hidden_size}), got {hidden_states.shape[-1]}"
            )
        if position_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                "position_ids shape must match hidden_states batch and "
                f"sequence dimensions {tuple(hidden_states.shape[:2])}, "
                f"got {tuple(position_ids.shape)}"
            )
        if slot_mapping.ndim != 1 or slot_mapping.shape[0] != 1:
            raise ValueError(
                "slot_mapping must have shape [1] for decode, "
                f"got {tuple(slot_mapping.shape)}"
            )
            
        if context_len <= 0:
            raise ValueError("context_len must be greater than 0")
        if position_ids.item() != context_len - 1:
            raise ValueError(
                "decode position must equal context_len - 1, "
                f"got position={position_ids.item()} and "
                f"context_len={context_len}"
            )

        query, key, value = self.qkv_proj(hidden_states)

        cos, sin = self.rotary_embedding(
            query,
            position_ids,
        )

        query, key = apply_rotary_pos_emb(
            query,
            key,
            cos,
            sin,
        )

        kv_cache.write_slots(
            slot_mapping=slot_mapping,
            k=key.squeeze(1),
            v=value.squeeze(1),
            layer_idx=layer_idx,
        )

        layer_k_cache, layer_v_cache = kv_cache.get_layer_cache(
            layer_idx
        )
        decode_query = query[0,0]
        attention_output = paged_attention_decode(
            q=decode_query,
            k_cache=layer_k_cache,
            v_cache=layer_v_cache,
            block_table=block_table,
            seq_len=context_len,
            block_size=kv_cache.block_size,
        )

        attention_output = attention_output.reshape(1, 1, self.hidden_size,)
        """
        query
        [1, 1, 4, 16]
            ↓ 取 query[0, 0]
        [4, 16]
            ↓ paged_attention_decode
        [4, 16]
            ↓ reshape,合并 heads 并恢复 B/S 维
        [1, 1, 64]
            ↓ o_proj
        [1, 1, 64]
        """
        return self.o_proj(attention_output)

    
    def forward_decode_batch_with_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: PagedKVCache,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one token per request; context lengths include this token."""
        if hidden_states.ndim != 3 or hidden_states.shape[1:] != (1, self.hidden_size):
            raise ValueError("hidden_states must have shape [B, 1, hidden_size]")
        batch_size = hidden_states.shape[0]
        if batch_size == 0:
            raise ValueError("decode batch must not be empty")
        if position_ids.shape != (batch_size, 1):
            raise ValueError("position_ids must have shape [B, 1]")
        if slot_mapping.shape != (batch_size,):
            raise ValueError("slot_mapping must have shape [B]")
        if context_lens.shape != (batch_size,):
            raise ValueError("context_lens must have shape [B]")
        if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
            raise ValueError("block_tables must have shape [B, max_num_blocks]")
        for name, tensor in (
            ("position_ids", position_ids),
            ("slot_mapping", slot_mapping),
            ("context_lens", context_lens),
            ("block_tables", block_tables),
        ):
            if tensor.dtype != torch.long:
                raise ValueError(f"{name} must use torch.long dtype")
            if tensor.device != hidden_states.device:
                raise ValueError(f"{name} must be on the hidden_states device")
        if kv_cache.k_cache.device != hidden_states.device:
            raise ValueError("KV cache must be on the hidden_states device")
        if kv_cache.dtype != hidden_states.dtype:
            raise ValueError("KV cache dtype must match hidden_states dtype")
        if torch.any(context_lens <= 0):
            raise ValueError("context_lens must be greater than 0")
        if torch.any(position_ids[:, 0] != context_lens - 1):
            raise ValueError("decode positions must equal context_lens - 1")

        # Check page metadata before mutating the shared cache.
        required_blocks = (context_lens + kv_cache.block_size - 1) // kv_cache.block_size
        if torch.any(required_blocks > block_tables.shape[1]):
            raise ValueError("block_tables have insufficient blocks for context_lens")
        columns = torch.arange(block_tables.shape[1], device=block_tables.device)
        valid_blocks = columns[None, :] < required_blocks[:, None]
        if torch.any(valid_blocks & ((block_tables < 0) | (block_tables >= kv_cache.num_blocks))):
            raise ValueError("block_tables contain an invalid physical block")
        rows = torch.arange(batch_size, device=block_tables.device)
        positions = position_ids[:, 0]
        expected_slots = (
            block_tables[rows, positions // kv_cache.block_size] * kv_cache.block_size
            + positions % kv_cache.block_size
        )
        if torch.any(slot_mapping != expected_slots):
            raise ValueError("slot_mapping must match block_tables and positions")
        if slot_mapping.unique().numel() != batch_size:
            raise ValueError("decode requests must use distinct write slots")
        layer_k_cache, layer_v_cache = kv_cache.get_layer_cache(layer_idx)

        query, key, value = self.qkv_proj(hidden_states)
        cos, sin = self.rotary_embedding(query, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        kv_cache.write_slots(
            slot_mapping=slot_mapping,
            k=key.squeeze(1),
            v=value.squeeze(1),
            layer_idx=layer_idx,
        )
        attention_output = paged_attention_decode_batch_parallel(
            q=query.squeeze(1),
            k_cache=layer_k_cache,
            v_cache=layer_v_cache,
            block_tables=block_tables,
            seq_lens=context_lens,
            block_size=kv_cache.block_size,
        )
        return self.o_proj(attention_output.reshape(batch_size, 1, self.hidden_size))

    def forward_prefill_chunk_with_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: PagedKVCache,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        block_table: list[int],
        prefix_len: int,
        context_len: int,
        tile_size: int,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "[1, chunk_len, hidden_size], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] != 1:
            raise ValueError(
                "chunk prefill currently supports exactly one request, "
                f"got batch_size={hidden_states.shape[0]}"
            )
        chunk_len = hidden_states.shape[1]
        if chunk_len == 0:
            raise ValueError("chunk_len must be greater than 0")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states last dimension must match hidden_size "
                f"({self.hidden_size}), got {hidden_states.shape[-1]}"
            )
        if position_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                "position_ids shape must match hidden_states batch and "
                f"chunk dimensions {tuple(hidden_states.shape[:2])}, "
                f"got {tuple(position_ids.shape)}"
            )
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("position_ids must use an integer dtype")
        if slot_mapping.ndim != 1 or slot_mapping.shape[0] != chunk_len:
            raise ValueError(
                "slot_mapping must have shape [chunk_len], "
                f"got {tuple(slot_mapping.shape)}"
            )
        if prefix_len < 0:
            raise ValueError("prefix_len cannot be negative")
        if context_len != prefix_len + chunk_len:
            raise ValueError(
                "context_len must equal prefix_len + chunk_len, "
                f"got prefix_len={prefix_len}, chunk_len={chunk_len}, "
                f"and context_len={context_len}"
            )
        if tile_size <= 0:
            raise ValueError("tile_size must be greater than 0")

        expected_positions = torch.arange(
            prefix_len,
            context_len,
            dtype=position_ids.dtype,
            device=position_ids.device,
        ).unsqueeze(0)
        if not torch.equal(position_ids, expected_positions):
            raise ValueError(
                "position_ids must contain the contiguous global positions "
                "from prefix_len through context_len - 1"
            )

        required_blocks = (
            context_len + kv_cache.block_size - 1
        ) // kv_cache.block_size
        if len(block_table) < required_blocks:
            raise ValueError(
                "block_table does not contain enough blocks for context_len"
            )
        used_blocks = block_table[:required_blocks]
        if any(
            physical_block < 0 or physical_block >= kv_cache.num_blocks
            for physical_block in used_blocks
        ):
            raise ValueError(
                "block_table contains an out-of-range physical block"
            )

        chunk_positions = torch.arange(
            prefix_len,
            context_len,
            dtype=torch.long,
            device=slot_mapping.device,
        )
        block_table_tensor = torch.as_tensor(
            block_table,
            dtype=torch.long,
            device=slot_mapping.device,
        )
        expected_slots = (
            block_table_tensor[chunk_positions // kv_cache.block_size]
            * kv_cache.block_size
            + chunk_positions % kv_cache.block_size
        )
        if not torch.equal(slot_mapping.to(torch.long), expected_slots):
            raise ValueError(
                "slot_mapping does not match block_table for the chunk positions"
            )

        query, key, value = self.qkv_proj(hidden_states)
        cos, sin = self.rotary_embedding(
            query,
            position_ids,
        )

        query, key = apply_rotary_pos_emb(
            query,
            key,
            cos,
            sin,
        )

        kv_cache.write_slots(
            slot_mapping=slot_mapping,
            k=key.squeeze(0),
            v=value.squeeze(0),
            layer_idx=layer_idx,
        )

        layer_k_cache, layer_v_cache = kv_cache.get_layer_cache(
            layer_idx
        )
        attention_output = paged_attention_prefill_chunk_tiled(
            q=query.squeeze(0),
            k_cache=layer_k_cache,
            v_cache=layer_v_cache,
            block_table=block_table,
            prefix_len=prefix_len,
            context_len=context_len,
            block_size=kv_cache.block_size,
            tile_size=tile_size,
        )

        attention_output = attention_output.reshape(
            1,
            chunk_len,
            self.hidden_size,
        )
        return self.o_proj(attention_output)

class Qwen2MLP(nn.Module):
    def __init__(self, config: TinyQwen2Config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up_output = self.up_proj(hidden_states)
        gate = F.silu(self.gate_proj(hidden_states))
        hidden = gate * up_output
        output = self.down_proj(hidden)
        return output

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: TinyQwen2Config):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
      self,
      hidden_states: torch.Tensor,
      position_ids: torch.Tensor,
    ) -> torch.Tensor:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

    def forward_prefill_with_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: PagedKVCache,
        layer_idx: int,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_prefill_with_cache(
            hidden_states = hidden_states,
            position_ids=position_ids,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            slot_mapping=slot_mapping,
        )
        hidden_states = residual + hidden_states
 
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward_decode_with_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: PagedKVCache,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        block_table: list[int],
        context_len: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_decode_with_cache(
            hidden_states=hidden_states,
            position_ids=position_ids,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            slot_mapping=slot_mapping,
            block_table=block_table,
            context_len=context_len,
        )
        hidden_states = hidden_states + residual

        residual = hidden_states 
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states



    def forward_decode_batch_with_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: PagedKVCache,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Run the decoder layer for one token per request in a batch."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_decode_batch_with_cache(
            hidden_states=hidden_states,
            position_ids=position_ids,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class TinyQwen2Model(nn.Module):
    """Tiny multi-layer Qwen2 backbone returning normalized hidden states."""

    def __init__(self, config: TinyQwen2Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch_size, seq_len], "
                f"got {tuple(input_ids.shape)}"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "input_ids must have dtype torch.int32 or torch.int64, "
                f"got {input_ids.dtype}"
            )

        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            raise ValueError("input_ids sequence length must be greater than 0")

        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0).expand(batch_size, -1)
        elif position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids shape must match input_ids shape "
                f"{tuple(input_ids.shape)}, got {tuple(position_ids.shape)}"
            )

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)

        return self.norm(hidden_states)

    def forward_prefill_with_cache(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: PagedKVCache,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [1, seq_len], "
                f"got {tuple(input_ids.shape)}"
            )
        if input_ids.shape[0] != 1:
            raise ValueError(
                "prefill with cache currently supports exactly one request, "
                f"got batch_size={input_ids.shape[0]}"
            )
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids sequence length must be greater than 0")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "input_ids must have dtype torch.int32 or torch.int64, "
                f"got {input_ids.dtype}"
            )
        if position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids shape must match input_ids shape "
                f"{tuple(input_ids.shape)}, got {tuple(position_ids.shape)}"
            )
        if slot_mapping.ndim != 1:
            raise ValueError(
                "slot_mapping must have shape [seq_len], "
                f"got {tuple(slot_mapping.shape)}"
            )
        if slot_mapping.shape[0] != input_ids.shape[1]:
            raise ValueError(
                "slot_mapping length must match sequence length "
                f"({input_ids.shape[1]}), got {slot_mapping.shape[0]}"
            )
        if kv_cache.num_layers != len(self.layers):
            raise ValueError(
                "KV cache num_layers must match model layers "
                f"({len(self.layers)}), got {kv_cache.num_layers}"
            )
        if kv_cache.num_kv_heads != self.config.num_key_value_heads:
            raise ValueError(
                "KV cache num_kv_heads must match model num_key_value_heads "
                f"({self.config.num_key_value_heads}), "
                f"got {kv_cache.num_kv_heads}"
            )
        if kv_cache.head_dim != self.config.head_dim:
            raise ValueError(
                "KV cache head_dim must match model head_dim "
                f"({self.config.head_dim}), got {kv_cache.head_dim}"
            )

        hidden_states = self.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.forward_prefill_with_cache(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                slot_mapping=slot_mapping,
            )
        return self.norm(hidden_states)

    def forward_decode_with_cache(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: PagedKVCache,
        slot_mapping: torch.Tensor,
        block_table: list[int],
        context_len: int,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [1, 1] for decode, "
                f"got {tuple(input_ids.shape)}"
            )
        if input_ids.shape != (1, 1):
            raise ValueError(
                "decode with cache requires exactly one request and one token, "
                f"got {tuple(input_ids.shape)}"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "input_ids must have dtype torch.int32 or torch.int64, "
                f"got {input_ids.dtype}"
            )
        if position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids shape must match input_ids shape "
                f"{tuple(input_ids.shape)}, got {tuple(position_ids.shape)}"
            )
        if slot_mapping.ndim != 1 or slot_mapping.shape[0] != 1:
            raise ValueError(
                "slot_mapping must have shape [1] for decode, "
                f"got {tuple(slot_mapping.shape)}"
            )
        if context_len <= 0:
            raise ValueError("context_len must be greater than 0")
        if position_ids.item() != context_len - 1:
            raise ValueError(
                "decode position must equal context_len - 1, "
                f"got position={position_ids.item()} and "
                f"context_len={context_len}"
            )
        required_blocks = (
            context_len + kv_cache.block_size - 1
        ) // kv_cache.block_size
        if len(block_table) < required_blocks:
            raise ValueError(
                "block_table does not contain enough blocks for context_len"
            )
        if kv_cache.num_layers != len(self.layers):
            raise ValueError(
                "KV cache num_layers must match model layers "
                f"({len(self.layers)}), got {kv_cache.num_layers}"
            )
        if kv_cache.num_kv_heads != self.config.num_key_value_heads:
            raise ValueError(
                "KV cache num_kv_heads must match model num_key_value_heads "
                f"({self.config.num_key_value_heads}), "
                f"got {kv_cache.num_kv_heads}"
            )
        if kv_cache.head_dim != self.config.head_dim:
            raise ValueError(
                "KV cache head_dim must match model head_dim "
                f"({self.config.head_dim}), got {kv_cache.head_dim}"
            )

        hidden_states = self.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.forward_decode_with_cache(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                slot_mapping=slot_mapping,
                block_table=block_table,
                context_len=context_len,
            )
        return self.norm(hidden_states)



    def forward_decode_batch_with_cache(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: PagedKVCache,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Return [B, 1, hidden_size] for one decode token per request."""
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("input_ids must have shape [B, 1] for decode")
        if input_ids.shape[0] == 0:
            raise ValueError("decode batch must not be empty")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must have dtype torch.int32 or torch.int64")
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids shape must match input_ids shape")
        if input_ids.device != self.embed_tokens.weight.device:
            raise ValueError("input_ids and model must be on the same device")
        if kv_cache.num_layers != len(self.layers):
            raise ValueError("KV cache num_layers must match model layers")
        if kv_cache.num_kv_heads != self.config.num_key_value_heads:
            raise ValueError("KV cache num_kv_heads must match model num_key_value_heads")
        if kv_cache.head_dim != self.config.head_dim:
            raise ValueError("KV cache head_dim must match model head_dim")

        hidden_states = self.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.forward_decode_batch_with_cache(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                context_lens=context_lens,
            )
        return self.norm(hidden_states)


class TinyQwen2ForCausalLM(nn.Module):
    def __init__(self, config: TinyQwen2Config):
        super().__init__()
        self.model = TinyQwen2Model(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward_decode_batch_with_cache(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: PagedKVCache,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Return [B, 1, vocab_size] logits from the batched decode backbone."""
        hidden_states = self.model.forward_decode_batch_with_cache(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        return self.lm_head(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        hidden_states = self.model(
            input_ids,
            position_ids,
        )

        logits = self.lm_head(hidden_states)
        return logits

    def forward_prefill_with_cache(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: PagedKVCache,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model.forward_prefill_with_cache(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
        )

        return self.lm_head(hidden_states)

    def forward_decode_with_cache(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: PagedKVCache,
        slot_mapping: torch.Tensor,
        block_table: list[int],
        context_len: int,
    ) -> torch.Tensor:
        hidden_states = self.model.forward_decode_with_cache(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            block_table=block_table,
            context_len=context_len,
        )

        return self.lm_head(hidden_states)
