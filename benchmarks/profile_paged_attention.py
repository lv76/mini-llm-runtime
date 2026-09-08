"""Minimal Nsight Compute harness for Triton Split-KV PagedAttention."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable

import torch
import triton

from mini_llm_runtime.kernels.reference.paged_attention import (
    paged_attention_decode,
)
from mini_llm_runtime.kernels.triton.paged_attention import (
    _paged_attention_split_kv_merge_kernel,
    _paged_attention_split_kv_stage1_block_local_kernel,
    _paged_attention_split_kv_stage1_gqa_reuse_kernel,
    _paged_attention_split_kv_stage1_gqa_reuse_2head_kernel,
    _paged_attention_split_kv_stage1_gqa_reuse_v2_kernel,
    _paged_attention_split_kv_stage1_kernel,
)


NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
BLOCK_SIZE = 16


@dataclass(frozen=True)
class ProfileInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_table: torch.Tensor
    block_table_list: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Split-KV PagedAttention kernels with Nsight Compute."
    )
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--partition-size", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--stage", choices=("stage1", "full"), default="stage1")
    parser.add_argument(
        "--addressing",
        choices=(
            "original",
            "block-local",
            "gqa-reuse",
            "gqa-reuse-v2",
            "gqa-reuse-2head",
        ),
        default="original",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PagedAttention profiling")
    for name in (
        "seq_len",
        "head_dim",
        "partition_size",
        "block_n",
        "num_warps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be greater than 0")
    if args.block_n & (args.block_n - 1):
        raise ValueError("--block-n must be a power of 2")
    if args.num_warps not in (1, 2, 4, 8):
        raise ValueError("--num-warps must be one of 1, 2, 4, or 8")
    if args.addressing == "block-local" and args.block_n % BLOCK_SIZE:
        raise ValueError("--block-n must be divisible by block size")
    if args.addressing == "block-local" and args.partition_size % BLOCK_SIZE:
        raise ValueError("--partition-size must be divisible by block size")


def create_inputs(seq_len: int, head_dim: int) -> ProfileInputs:
    logical_blocks = triton.cdiv(seq_len, BLOCK_SIZE)
    num_physical_blocks = logical_blocks + 3
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    block_table_list = torch.randperm(
        num_physical_blocks,
        generator=generator,
    )[:logical_blocks].tolist()
    block_table = torch.tensor(
        block_table_list,
        dtype=torch.int32,
        device="cuda",
    )

    torch.cuda.manual_seed(0)
    q = torch.randn(
        NUM_QUERY_HEADS,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    logical_k = torch.randn(
        seq_len,
        NUM_KV_HEADS,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    logical_v = torch.randn_like(logical_k)
    k_cache = torch.empty(
        num_physical_blocks,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    v_cache = torch.empty_like(k_cache)

    positions = torch.arange(seq_len, device="cuda")
    physical_blocks = block_table[positions // BLOCK_SIZE].long()
    offsets_in_block = positions % BLOCK_SIZE
    k_cache[physical_blocks, offsets_in_block] = logical_k
    v_cache[physical_blocks, offsets_in_block] = logical_v
    return ProfileInputs(q, k_cache, v_cache, block_table, block_table_list)


def create_kernel_paths(
    inputs: ProfileInputs,
    *,
    seq_len: int,
    head_dim: int,
    partition_size: int,
    block_n: int,
    num_warps: int,
    addressing: str,
) -> tuple[Callable[[], None], Callable[[], torch.Tensor]]:
    num_partitions = triton.cdiv(seq_len, partition_size)
    block_d = triton.next_power_of_2(head_dim)
    block_p = triton.next_power_of_2(num_partitions)
    partial_m = torch.empty(
        (NUM_QUERY_HEADS, num_partitions),
        dtype=torch.float32,
        device="cuda",
    )
    partial_l = torch.empty_like(partial_m)
    partial_acc = torch.empty(
        (NUM_QUERY_HEADS, num_partitions, head_dim),
        dtype=torch.float32,
        device="cuda",
    )
    output = torch.empty_like(inputs.q)
    if addressing == "gqa-reuse-2head":
        stage1_kernel = _paged_attention_split_kv_stage1_gqa_reuse_2head_kernel
        stage1_grid_heads = NUM_KV_HEADS * 2
    elif addressing == "gqa-reuse-v2":
        stage1_kernel = _paged_attention_split_kv_stage1_gqa_reuse_v2_kernel
        stage1_grid_heads = NUM_KV_HEADS
    elif addressing == "gqa-reuse":
        stage1_kernel = _paged_attention_split_kv_stage1_gqa_reuse_kernel
        stage1_grid_heads = NUM_KV_HEADS
    elif addressing == "block-local":
        stage1_kernel = _paged_attention_split_kv_stage1_block_local_kernel
        stage1_grid_heads = NUM_QUERY_HEADS
    else:
        stage1_kernel = _paged_attention_split_kv_stage1_kernel
        stage1_grid_heads = NUM_QUERY_HEADS

    def stage1() -> None:
        stage1_kernel[
            (stage1_grid_heads, num_partitions)
        ](
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.block_table,
            partial_m,
            partial_l,
            partial_acc,
            seq_len,
            num_partitions,
            num_query_heads=NUM_QUERY_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=head_dim,
            block_size=BLOCK_SIZE,
            scale=1.0 / math.sqrt(head_dim),
            PARTITION_SIZE=partition_size,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )

    def full() -> torch.Tensor:
        stage1()
        _paged_attention_split_kv_merge_kernel[(NUM_QUERY_HEADS,)](
            partial_m,
            partial_l,
            partial_acc,
            output,
            num_partitions,
            head_dim=head_dim,
            BLOCK_P=block_p,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
        return output

    return stage1, full


def main() -> None:
    args = parse_args()
    validate_args(args)
    inputs = create_inputs(args.seq_len, args.head_dim)
    stage1, full = create_kernel_paths(
        inputs,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        partition_size=args.partition_size,
        block_n=args.block_n,
        num_warps=args.num_warps,
        addressing=args.addressing,
    )

    reference = paged_attention_decode(
        q=inputs.q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_table=inputs.block_table_list,
        seq_len=args.seq_len,
        block_size=BLOCK_SIZE,
    )
    output = full()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
    torch.cuda.synchronize()

    profile_launch = stage1 if args.stage == "stage1" else full
    for _ in range(args.warmup):
        profile_launch()
    torch.cuda.synchronize()

    print(
        f"Profiling addressing={args.addressing}, stage={args.stage}, "
        f"S={args.seq_len}, D={args.head_dim}, "
        f"PARTITION_SIZE={args.partition_size}, BLOCK_N={args.block_n}, "
        f"num_warps={args.num_warps}, iters={args.iters}"
    )
    cuda_runtime = torch.cuda.cudart()
    cuda_runtime.cudaProfilerStart()
    for _ in range(args.iters):
        profile_launch()
    torch.cuda.synchronize()
    cuda_runtime.cudaProfilerStop()


if __name__ == "__main__":
    main()
