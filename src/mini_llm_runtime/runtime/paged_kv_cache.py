from __future__ import annotations

import torch


class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
    
        self.k_cache = torch.empty(
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )

        self.v_cache = torch.empty_like(self.k_cache)

        self.free_blocks = list(range(num_blocks - 1, -1, -1))

        self.block_tables: dict[int, list[int]] = {}
        self.seq_lens: dict[int, int] = {}

    def add_sequence(self, seq_id):
        if seq_id in self.block_tables:
            raise KeyError(f"Sequence {seq_id} is already exists")

        self.block_tables[seq_id] = []
        self.seq_lens[seq_id] = 0

    def _allocate_block(self):
        if not self.free_blocks:
            raise RuntimeError("KV cache out of physical blocks")

        return self.free_blocks.pop()

    def _get_physical_localtion(self, seq_id, position):

        logical_block = position // self.block_size
        offset = position % self.block_size

        physical_block = self.block_tables[seq_id][logical_block]

        return physical_block, offset

    def append_tokens(
        self,
        seq_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if k.ndim != 3 or v.ndim != 3:
            raise ValueError("Expected K/V shape [T, H_kv, D]")

        if k.shape != v.shape:
            raise ValueError("K and V must have the same shape")

        expected_tail = (self.num_kv_heads, self.head_dim)

        if tuple(k.shape[1:]) != expected_tail:
            raise ValueError(
                f"Expected K/V shape [T, {self.num_kv_heads}, {self.head_dim}]"
            )

        num_new_tokens = k.shape[0]
        if not self._check_capacity(seq_id, num_new_tokens):
            raise RuntimeError("out of memory")

        for i in range(k.shape[0]):
            self.append(seq_id, k[i], v[i])

        

    def free_sequence(self, seq_id: int) -> None:
        if seq_id not in self.block_tables:
            raise KeyError(f"Unknown sequence {seq_id}")

        physical_blocks = self.block_tables[seq_id]

        self.free_blocks.extend(physical_blocks) ##把 block 编号放回 allocator
        physical_blocks.clear()  ##清空 block_table
        del self.block_tables[seq_id]
        del self.seq_lens[seq_id]


    #out of memory check
    def _check_capacity(
        self,
        seq_id: int,
        num_new_tokens: int,
        ):

        current_len = self.seq_lens[seq_id]

        final_len = current_len + num_new_tokens

        required_new_blocks = (
            (final_len + self.block_size - 1) // self.block_size
        ) - len(self.block_tables[seq_id])

        return len(self.free_blocks) >= required_new_blocks
            
    def get_sequence(
        self,
        seq_id
    ):
        seq_len = self.seq_lens[seq_id]

        k_out = torch.empty(
            seq_len,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        v_out = torch.empty_like(k_out)

        for position in range(seq_len):
            physical_block, offset = self._get_physical_localtion(seq_id, position)

            k_out[position].copy_(self.k_cache[0,physical_block, offset])
            v_out[position].copy_(self.v_cache[0, physical_block, offset])

        return k_out,v_out


    def append(self, seq_id:int, k:torch.Tensor, v:torch.Tensor):

        if seq_id not in self.block_tables:
            raise KeyError(f"Unknow sequence {seq_id}")

        expected_shape = (self.num_kv_heads, self.head_dim)

        if k.shape != expected_shape:
            raise ValueError(
                f"Expected k shape {expected_shape}, got {tuple(k.shape)}"
            )

        if v.shape != expected_shape:
            raise ValueError(
                f"Expected v shape {expected_shape}, got {tuple(v.shape)}"
            )

        position = self.seq_lens[seq_id]

        logical_address = position // self.block_size
        offset = position % self.block_size

        if logical_address == len(self.block_tables[seq_id]):
            self.block_tables[seq_id].append(self._allocate_block())

        physical_address = self.block_tables[seq_id][logical_address]

        self.k_cache[0, physical_address, offset].copy_(k)
        self.v_cache[0, physical_address, offset].copy_(v)

        self.seq_lens[seq_id] += 1

    def reserve_tokens(self, seq_id: int, total_num_tokens: int) -> bool:

        if total_num_tokens < 0:
            raise ValueError("total_num_tokens cannot be negative")
        
        required_new_blocks = (
            (total_num_tokens + self.block_size - 1) // self.block_size
        ) - len(self.block_tables[seq_id])

        required_new_blocks = max(required_new_blocks, 0)
        
        if len(self.free_blocks) >= required_new_blocks:
            self.block_tables[seq_id].extend(self._allocate_block() for _ in range(required_new_blocks))
            return True
        return False

    def write_slots(
        self,
        slot_mapping: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Write [T, Hkv, D] K/V to distinct flat slots in one layer.

        Empty writes are allowed. Duplicate slots are rejected to avoid
        ambiguous parallel writes. Source dtype conversion matches copy_.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError("Unexpected layer_idx")
        expected_tail = (self.num_kv_heads, self.head_dim)
        if k.ndim != 3 or v.ndim != 3:
            raise ValueError(f"Expected K/V shape [T, H_kv, D]")

        if k.shape != v.shape:
            raise ValueError("K and V must have the same shape")

        if tuple(k.shape[1:]) != expected_tail:
            raise ValueError(
                f"Expected K/V shape [T, {self.num_kv_heads}, {self.head_dim}]"
            )

        if slot_mapping.ndim != 1:
            raise ValueError("Expected slot_mapping shape [T]")

        if slot_mapping.shape[0] != k.shape[0]:
            raise ValueError(
                "slot_mapping length must match the number of K/V tokens"
            )

        if slot_mapping.dtype not in (torch.int32, torch.int64):
            raise TypeError("slot_mapping must use an integer index dtype")
        cache_device = self.k_cache.device
        if slot_mapping.device != cache_device or k.device != cache_device or v.device != cache_device:
            raise ValueError("slot_mapping and K/V must be on the KV cache device")

        num_slots = self.num_blocks * self.block_size
        if torch.any(slot_mapping < 0) or torch.any(slot_mapping >= num_slots):
            raise IndexError("slot_mapping contains an out-of-range slot")

        if slot_mapping.unique().numel() != slot_mapping.numel():
            raise ValueError("slot_mapping must contain distinct slots")

        # Flatten only block/offset dimensions; slot = block * block_size + offset.
        # These are views into the selected layer, so writes update the cache.
        layer_k = self.k_cache[layer_idx].flatten(0, 1)
        layer_v = self.v_cache[layer_idx].flatten(0, 1)
        indices = slot_mapping.to(dtype=torch.long)
        source_k = k.to(dtype=layer_k.dtype)
        source_v = v.to(dtype=layer_v.dtype)
        layer_k.index_copy_(0, indices, source_k)
        layer_v.index_copy_(0, indices, source_v)

    def get_layer_cache(
        self, 
        layer_idx: int = 0,
    ):
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"Unexpect layer_idx")
        layer_k = self.k_cache[layer_idx]
        layer_v = self.v_cache[layer_idx]
        return layer_k, layer_v





        
