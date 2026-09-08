"""Basic correctness and latency benchmark for PagedAttention decode.

The dense path is a dense PyTorch baseline.  The paged PyTorch path is a
correctness/reference baseline; its Python gather/stack work makes it an
intentionally unfair comparison with high-performance kernels.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import triton

from mini_llm_runtime.kernels.reference.paged_attention import (
    paged_attention_decode,
)
from mini_llm_runtime.kernels.triton.paged_attention import (
    _paged_attention_decode_triton,
    _paged_attention_split_kv_merge_kernel,
    _paged_attention_split_kv_stage1_kernel,
    paged_attention_decode_split_kv_triton,
    paged_attention_decode_triton,
)


DEFAULT_SEQ_LENS = [128, 256, 512, 1024, 2048, 4096]
DEFAULT_TUNING_SEQ_LENS = [512, 2048, 4096]
DEFAULT_BLOCK_NS = [16, 32, 64, 128]
DEFAULT_NUM_WARPS_VALUES = [1, 2, 4, 8]
SPLIT_KV_SEQ_LENS = [512, 1024, 2048, 4096, 8192]
SPLIT_KV_PARTITION_SIZE = 512
SPLIT_KV_BLOCK_N = 256
SPLIT_KV_NUM_WARPS = 4
SINGLE_PASS_BLOCK_N = 512
SINGLE_PASS_NUM_WARPS = 8
DTYPES = {
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class BenchmarkInputs:
    q: torch.Tensor
    logical_k: torch.Tensor
    logical_v: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_table: torch.Tensor
    block_table_list: list[int]
    seq_len: int
    block_size: int


@dataclass(frozen=True)
class BenchmarkResult:
    seq_len: int
    dense_ms: float
    paged_reference_ms: float | None
    triton_ms: float
    triton_vs_dense: float
    triton_vs_paged_reference: float | None


@dataclass(frozen=True)
class TuningResult:
    seq_len: int
    BLOCK_N: int
    num_warps: int
    triton_ms: float | None
    dense_ms: float | None
    triton_vs_dense: float | None
    status: str
    error: str | None


@dataclass(frozen=True)
class SplitKVBenchmarkResult:
    seq_len: int
    num_partitions: int
    single_pass_ms: float
    split_kernel_ms: float
    split_e2e_ms: float
    split_kernel_vs_single: float
    split_e2e_vs_single: float


@dataclass(frozen=True)
class PartitionScanResult:
    partition_size: int
    num_partitions: int
    stage1_ms: float
    merge_ms: float
    split_kernel_ms: float
    single_pass_ms: float
    split_vs_single: float


@dataclass(frozen=True)
class SplitConfigScanResult:
    partition_size: int
    block_n: int
    num_warps: int
    num_partitions: int
    stage1_ms: float
    merge_ms: float
    split_kernel_ms: float
    single_pass_ms: float
    split_vs_single: float


@dataclass(frozen=True)
class SeqSplitConfigScanResult:
    seq_len: int
    partition_size: int
    block_n: int
    num_warps: int
    num_partitions: int
    split_kernel_ms: float
    single_pass_ms: float
    split_vs_single: float


@dataclass(frozen=True)
class ShapeSplitConfigScanResult:
    seq_len: int
    head_dim: int
    partition_size: int
    block_n: int
    num_warps: int
    num_partitions: int
    split_kernel_ms: float
    single_pass_ms: float
    split_vs_single: float


@dataclass(frozen=True)
class SingleConfigScanResult:
    seq_len: int
    head_dim: int
    block_n: int
    num_warps: int
    single_pass_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PagedAttention decode against PyTorch baselines.",
    )
    parser.add_argument("--seq-lens", nargs="+", type=int)
    parser.add_argument("--block-ns", nargs="+", type=int, default=DEFAULT_BLOCK_NS)
    parser.add_argument(
        "--num-warps-values",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_WARPS_VALUES,
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="scan BLOCK_N and num_warps combinations",
    )
    parser.add_argument(
        "--split-kv",
        action="store_true",
        help="benchmark tuned single-pass against Split-KV paths",
    )
    parser.add_argument(
        "--partition-sizes",
        nargs="+",
        type=int,
        help="scan Split-KV PARTITION_SIZE at fixed seq_len=4096",
    )
    parser.add_argument(
        "--split-configs",
        nargs="+",
        metavar="PARTITION_SIZE:BLOCK_N[:NUM_WARPS]",
        help="scan explicit Split-KV configs at fixed seq_len=4096",
    )
    parser.add_argument(
        "--seq-split-configs",
        nargs="+",
        metavar="SEQ_LEN:PARTITION_SIZE:BLOCK_N:NUM_WARPS",
        help="scan explicit Split-KV configs across sequence lengths",
    )
    parser.add_argument(
        "--shape-split-configs",
        nargs="+",
        metavar="SEQ_LEN:HEAD_DIM:PARTITION_SIZE:BLOCK_N:NUM_WARPS",
        help="scan explicit Split-KV configs across sequence lengths and head dims",
    )
    parser.add_argument(
        "--single-configs",
        nargs="+",
        metavar="SEQ_LEN:HEAD_DIM:BLOCK_N:NUM_WARPS",
        help="scan explicit single-pass configs across sequence lengths and head dims",
    )
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=DTYPES, default="fp16")
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="do_bench warmup duration in milliseconds (ms; default: 25)",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="do_bench measurement duration in milliseconds (ms; default: 100)",
    )
    parser.add_argument(
        "--benchmark-reference",
        action="store_true",
        help=(
            "benchmark the Python PagedAttention reference with synchronized "
            "end-to-end wall-clock timing (default: disabled)"
        ),
    )
    parser.add_argument(
        "--reference-reps",
        type=int,
        default=1,
        help="fixed number of Python reference calls to time (default: 1)",
    )
    parser.add_argument("--csv", type=Path, help="optional CSV output path")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton PagedAttention benchmark")
    partition_scan = args.partition_sizes is not None
    config_scan = args.split_configs is not None
    seq_config_scan = args.seq_split_configs is not None
    shape_config_scan = args.shape_split_configs is not None
    single_config_scan = args.single_configs is not None
    if sum(
        (
            args.tune,
            args.split_kv,
            partition_scan,
            config_scan,
            seq_config_scan,
            shape_config_scan,
            single_config_scan,
        )
    ) > 1:
        raise ValueError(
            "--tune, --split-kv, --partition-sizes, and --split-configs "
            "and explicit shape scan modes are mutually exclusive"
        )
    if single_config_scan:
        if args.seq_lens is not None:
            raise ValueError("--seq-lens cannot be used with --single-configs")
        parsed_single_configs = []
        for config in args.single_configs:
            try:
                fields = config.split(":")
                if len(fields) != 4:
                    raise ValueError
                seq_len, head_dim, block_n, num_warps = map(int, fields)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    "--single-configs entries must use "
                    "SEQ_LEN:HEAD_DIM:BLOCK_N:NUM_WARPS"
                ) from error
            if min(seq_len, head_dim, block_n, num_warps) <= 0:
                raise ValueError("single-pass config values must be greater than 0")
            if block_n & (block_n - 1):
                raise ValueError("single-pass BLOCK_N must be a power of 2")
            if num_warps not in (1, 2, 4, 8):
                raise ValueError(
                    "single-pass num_warps must be one of 1, 2, 4, or 8"
                )
            parsed_single_configs.append((seq_len, head_dim, block_n, num_warps))
        args.single_configs = parsed_single_configs
    if shape_config_scan:
        if args.seq_lens is not None:
            raise ValueError("--seq-lens cannot be used with --shape-split-configs")
        parsed_shape_configs = []
        for config in args.shape_split_configs:
            try:
                fields = config.split(":")
                if len(fields) != 5:
                    raise ValueError
                seq_len, head_dim, partition_size, block_n, num_warps = map(
                    int, fields
                )
            except (ValueError, TypeError) as error:
                raise ValueError(
                    "--shape-split-configs entries must use "
                    "SEQ_LEN:HEAD_DIM:PARTITION_SIZE:BLOCK_N:NUM_WARPS"
                ) from error
            if min(seq_len, head_dim, partition_size, block_n, num_warps) <= 0:
                raise ValueError("Split-KV config values must be greater than 0")
            if block_n & (block_n - 1):
                raise ValueError("Split-KV BLOCK_N must be a power of 2")
            if num_warps not in (1, 2, 4, 8):
                raise ValueError("Split-KV num_warps must be one of 1, 2, 4, or 8")
            parsed_shape_configs.append(
                (seq_len, head_dim, partition_size, block_n, num_warps)
            )
        args.shape_split_configs = parsed_shape_configs
    if seq_config_scan:
        if args.seq_lens is not None:
            raise ValueError("--seq-lens cannot be used with --seq-split-configs")
        parsed_seq_configs = []
        for config in args.seq_split_configs:
            try:
                fields = config.split(":")
                if len(fields) != 4:
                    raise ValueError
                seq_len, partition_size, block_n, num_warps = map(int, fields)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    "--seq-split-configs entries must use "
                    "SEQ_LEN:PARTITION_SIZE:BLOCK_N:NUM_WARPS"
                ) from error
            if min(seq_len, partition_size, block_n, num_warps) <= 0:
                raise ValueError("Split-KV config values must be greater than 0")
            if block_n & (block_n - 1):
                raise ValueError("Split-KV BLOCK_N must be a power of 2")
            if num_warps not in (1, 2, 4, 8):
                raise ValueError("Split-KV num_warps must be one of 1, 2, 4, or 8")
            parsed_seq_configs.append(
                (seq_len, partition_size, block_n, num_warps)
            )
        args.seq_split_configs = parsed_seq_configs
    if args.seq_lens is None:
        if single_config_scan:
            args.seq_lens = [config[0] for config in args.single_configs]
        elif shape_config_scan:
            args.seq_lens = [config[0] for config in args.shape_split_configs]
        elif seq_config_scan:
            args.seq_lens = [config[0] for config in args.seq_split_configs]
        elif partition_scan or config_scan:
            args.seq_lens = [4096]
        elif args.split_kv:
            args.seq_lens = SPLIT_KV_SEQ_LENS
        else:
            args.seq_lens = DEFAULT_TUNING_SEQ_LENS if args.tune else DEFAULT_SEQ_LENS
    if any(seq_len <= 0 for seq_len in args.seq_lens):
        raise ValueError("all --seq-lens values must be greater than 0")
    if (partition_scan or config_scan) and args.seq_lens != [4096]:
        raise ValueError("Split-KV scan modes require seq_len=4096")
    if partition_scan and any(size <= 0 for size in args.partition_sizes):
        raise ValueError("all --partition-sizes values must be greater than 0")
    if config_scan:
        parsed_configs = []
        for config in args.split_configs:
            try:
                fields = config.split(":")
                if len(fields) == 2:
                    partition_text, block_n_text = fields
                    num_warps_text = str(SPLIT_KV_NUM_WARPS)
                elif len(fields) == 3:
                    partition_text, block_n_text, num_warps_text = fields
                else:
                    raise ValueError
                partition_size = int(partition_text)
                block_n = int(block_n_text)
                num_warps = int(num_warps_text)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    "--split-configs entries must use "
                    "PARTITION_SIZE:BLOCK_N[:NUM_WARPS]"
                ) from error
            if partition_size <= 0 or block_n <= 0 or num_warps <= 0:
                raise ValueError("Split-KV config values must be greater than 0")
            if block_n & (block_n - 1):
                raise ValueError("Split-KV BLOCK_N must be a power of 2")
            if num_warps not in (1, 2, 4, 8):
                raise ValueError("Split-KV num_warps must be one of 1, 2, 4, or 8")
            parsed_configs.append((partition_size, block_n, num_warps))
        args.split_configs = parsed_configs
    if args.num_query_heads <= 0 or args.num_kv_heads <= 0:
        raise ValueError("head counts must be greater than 0")
    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    if args.head_dim <= 0 or args.block_size <= 0:
        raise ValueError("head_dim and block_size must be greater than 0")
    if args.warmup <= 0 or args.rep <= 0:
        raise ValueError("warmup and rep must be greater than 0")
    if args.reference_reps <= 0:
        raise ValueError("reference-reps must be greater than 0")
    if any(block_n <= 0 for block_n in args.block_ns):
        raise ValueError("all --block-ns values must be greater than 0")
    if any(num_warps not in (1, 2, 4, 8) for num_warps in args.num_warps_values):
        raise ValueError("--num-warps-values must contain only 1, 2, 4, or 8")
    split_scan = args.split_kv or partition_scan or config_scan or seq_config_scan
    if (split_scan or shape_config_scan) and (
        args.dtype != "fp16"
        or args.num_query_heads != 32
        or args.num_kv_heads != 8
        or args.block_size != 16
    ):
        raise ValueError(
            "Split-KV benchmarks require dtype=fp16, Hq=32, Hkv=8, "
            "and block_size=16"
        )
    if split_scan and args.head_dim != 128:
        raise ValueError("this Split-KV benchmark mode requires D=128")
    if single_config_scan and (
        args.dtype != "fp16"
        or args.num_query_heads != 32
        or args.num_kv_heads != 8
        or args.block_size != 16
    ):
        raise ValueError(
            "single-pass config scan requires dtype=fp16, Hq=32, Hkv=8, "
            "and block_size=16"
        )


def prepare_csv_output(path: Path) -> None:
    if path.exists() and path.is_dir():
        raise ValueError(
            f"--csv must include a CSV filename, but points to a directory: {path}"
        )
    if path.parent.exists() and not path.parent.is_dir():
        raise ValueError(f"--csv parent path is not a directory: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=True)


def create_benchmark_inputs(
    *,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    seed: int = 0,
) -> BenchmarkInputs:
    logical_blocks = (seq_len + block_size - 1) // block_size
    num_physical_blocks = logical_blocks + 3

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    block_table_list = torch.randperm(
        num_physical_blocks,
        generator=cpu_generator,
    )[:logical_blocks].tolist()
    block_table = torch.tensor(
        block_table_list,
        device="cuda",
        dtype=torch.int32,
    )

    torch.cuda.manual_seed(seed)

    before_allocated_bytes = torch.cuda.memory_allocated()

    q = torch.randn(
        num_query_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    logical_k = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    logical_v = torch.randn_like(logical_k)
    k_cache = torch.empty(
        num_physical_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    v_cache = torch.empty_like(k_cache)

    after_allocated_bytes = torch.cuda.memory_allocated()
    allocated_bytes = after_allocated_bytes - before_allocated_bytes
    print(
        f"Allocated {allocated_bytes / 1_000_000:.2f} MB"
        )

    positions = torch.arange(seq_len, device="cuda")
    physical_blocks = block_table[positions // block_size].long()
    offsets = positions % block_size
    k_cache[physical_blocks, offsets] = logical_k
    v_cache[physical_blocks, offsets] = logical_v

    return BenchmarkInputs(
        q=q,
        logical_k=logical_k,
        logical_v=logical_v,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        block_table_list=block_table_list,
        seq_len=seq_len,
        block_size=block_size,
    )


def dense_attention_decode(
    q: torch.Tensor,
    logical_k: torch.Tensor,
    logical_v: torch.Tensor,
) -> torch.Tensor:
    """Dense PyTorch SDPA baseline over logically contiguous K/V."""
    q_bhsd = q[None, :, None, :]
    k_bhsd = logical_k.permute(1, 0, 2)[None, :, :, :]
    v_bhsd = logical_v.permute(1, 0, 2)[None, :, :, :]
    return F.scaled_dot_product_attention(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        enable_gqa=True,
    )[0, :, 0, :]


def _paths(inputs: BenchmarkInputs) -> tuple[Callable[[], torch.Tensor], ...]:
    dense = lambda: dense_attention_decode(
        inputs.q,
        inputs.logical_k,
        inputs.logical_v,
    )
    paged_reference = lambda: paged_attention_decode(
        q=inputs.q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_table=inputs.block_table_list,
        seq_len=inputs.seq_len,
        block_size=inputs.block_size,
    )
    triton_paged = lambda: paged_attention_decode_triton(
        q=inputs.q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_table=inputs.block_table,
        seq_len=inputs.seq_len,
    )
    return dense, paged_reference, triton_paged


def verify_outputs(inputs: BenchmarkInputs) -> None:
    dense, paged_reference, triton_paged = _paths(inputs)
    dense_output = dense()
    reference_output = paged_reference()
    triton_output = triton_paged()  # Also performs the first Triton JIT compile.

    torch.testing.assert_close(reference_output, dense_output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(triton_output, dense_output, rtol=1e-2, atol=1e-2)


def _bench(fn: Callable[[], torch.Tensor], warmup: int, rep: int) -> float:
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="median",
        )
    )


def _bench_reference(
    fn: Callable[[], torch.Tensor], reference_reps: int
) -> float:
    """Return synchronized end-to-end wall time per reference call."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reference_reps):
        fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1_000
    return elapsed_ms / reference_reps


def benchmark_one_configuration(
    inputs: BenchmarkInputs,
    *,
    warmup: int,
    rep: int,
    benchmark_reference: bool = False,
    reference_reps: int = 1,
) -> BenchmarkResult:
    verify_outputs(inputs)
    dense, paged_reference, triton_paged = _paths(inputs)

    # do_bench performs its own warmup.  The correctness call above ensures the
    # first Triton JIT compile is complete before any formal measurement.
    dense_ms = _bench(dense, warmup, rep)
    paged_reference_ms = (
        _bench_reference(paged_reference, reference_reps)
        if benchmark_reference
        else None
    )
    triton_ms = _bench(triton_paged, warmup, rep)

    return BenchmarkResult(
        seq_len=inputs.seq_len,
        dense_ms=dense_ms,
        paged_reference_ms=paged_reference_ms,
        triton_ms=triton_ms,
        triton_vs_dense=dense_ms / triton_ms,
        triton_vs_paged_reference=(
            paged_reference_ms / triton_ms
            if paged_reference_ms is not None
            else None
        ),
    )


def print_results(results: list[BenchmarkResult]) -> None:
    headers = (
        "seq_len",
        "dense_ms",
        "paged_reference_ms",
        "triton_ms",
        "triton_vs_dense",
        "triton_vs_paged_reference",
    )
    rows = [
        (
            str(result.seq_len),
            f"{result.dense_ms:.4f}",
            (
                f"{result.paged_reference_ms:.4f}"
                if result.paged_reference_ms is not None
                else "N/A"
            ),
            f"{result.triton_ms:.4f}",
            f"{result.triton_vs_dense:.2f}x",
            (
                f"{result.triton_vs_paged_reference:.2f}x"
                if result.triton_vs_paged_reference is not None
                else "N/A"
            ),
        )
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.rjust(width) for value, width in zip(row, widths)))


def save_csv(results: list[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BenchmarkResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    field: (
                        getattr(result, field)
                        if getattr(result, field) is not None
                        else "N/A"
                    )
                    for field in fieldnames
                }
            )


def _split_kv_kernel_paths(
    inputs: BenchmarkInputs,
    partition_size: int,
    block_n: int,
    num_warps: int,
) -> tuple[
    Callable[[], None],
    Callable[[], torch.Tensor],
    Callable[[], torch.Tensor],
    int,
]:
    num_query_heads, head_dim = inputs.q.shape
    _, block_size, num_kv_heads, _ = inputs.k_cache.shape
    num_partitions = triton.cdiv(inputs.seq_len, partition_size)
    block_d = triton.next_power_of_2(head_dim)
    block_p = triton.next_power_of_2(num_partitions)

    partial_m = torch.empty(
        (num_query_heads, num_partitions),
        dtype=torch.float32,
        device=inputs.q.device,
    )
    partial_l = torch.empty_like(partial_m)
    partial_acc = torch.empty(
        (num_query_heads, num_partitions, head_dim),
        dtype=torch.float32,
        device=inputs.q.device,
    )
    output = torch.empty_like(inputs.q)

    def stage1() -> None:
        _paged_attention_split_kv_stage1_kernel[
            (num_query_heads, num_partitions)
        ](
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.block_table,
            partial_m,
            partial_l,
            partial_acc,
            inputs.seq_len,
            num_partitions,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            scale=1.0 / math.sqrt(head_dim),
            PARTITION_SIZE=partition_size,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )

    def merge() -> torch.Tensor:
        _paged_attention_split_kv_merge_kernel[(num_query_heads,)](
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

    def run() -> torch.Tensor:
        stage1()
        return merge()

    return stage1, merge, run, num_partitions


def benchmark_split_kv_configuration(
    inputs: BenchmarkInputs,
    *,
    warmup: int,
    rep: int,
) -> SplitKVBenchmarkResult:
    reference = lambda: paged_attention_decode(
        q=inputs.q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_table=inputs.block_table_list,
        seq_len=inputs.seq_len,
        block_size=inputs.block_size,
    )
    single_pass = lambda: _paged_attention_decode_triton(
        q=inputs.q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_table=inputs.block_table,
        seq_len=inputs.seq_len,
        block_n=SINGLE_PASS_BLOCK_N,
        num_warps=SINGLE_PASS_NUM_WARPS,
    )
    _, _, split_kernel, num_partitions = _split_kv_kernel_paths(
        inputs,
        SPLIT_KV_PARTITION_SIZE,
        SPLIT_KV_BLOCK_N,
        SPLIT_KV_NUM_WARPS,
    )
    split_e2e = lambda: paged_attention_decode_split_kv_triton(
        q=inputs.q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_table=inputs.block_table,
        seq_len=inputs.seq_len,
        partition_size=SPLIT_KV_PARTITION_SIZE,
        block_n=SPLIT_KV_BLOCK_N,
        num_warps=SPLIT_KV_NUM_WARPS,
    )

    reference_output = reference()
    single_pass_output = single_pass()
    split_kernel_output = split_kernel()
    split_e2e_output = split_e2e()
    assert torch.isfinite(split_e2e_output).all()
    torch.testing.assert_close(
        single_pass_output, reference_output, rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        split_kernel_output, reference_output, rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        split_e2e_output, reference_output, rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        split_e2e_output, single_pass_output, rtol=1e-2, atol=1e-2
    )

    single_pass_ms = _bench(single_pass, warmup, rep)
    split_kernel_ms = _bench(split_kernel, warmup, rep)
    split_e2e_ms = _bench(split_e2e, warmup, rep)
    return SplitKVBenchmarkResult(
        seq_len=inputs.seq_len,
        num_partitions=num_partitions,
        single_pass_ms=single_pass_ms,
        split_kernel_ms=split_kernel_ms,
        split_e2e_ms=split_e2e_ms,
        split_kernel_vs_single=single_pass_ms / split_kernel_ms,
        split_e2e_vs_single=single_pass_ms / split_e2e_ms,
    )


def print_split_kv_results(results: list[SplitKVBenchmarkResult]) -> None:
    headers = tuple(SplitKVBenchmarkResult.__dataclass_fields__)
    rows = [
        (
            str(result.seq_len),
            str(result.num_partitions),
            f"{result.single_pass_ms:.4f}",
            f"{result.split_kernel_ms:.4f}",
            f"{result.split_e2e_ms:.4f}",
            f"{result.split_kernel_vs_single:.2f}x",
            f"{result.split_e2e_vs_single:.2f}x",
        )
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.rjust(width) for value, width in zip(row, widths)))


def save_split_kv_csv(
    results: list[SplitKVBenchmarkResult], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SplitKVBenchmarkResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fieldnames})


def benchmark_partition_size(
    inputs: BenchmarkInputs,
    *,
    partition_size: int,
    reference_output: torch.Tensor,
    single_pass_output: torch.Tensor,
    single_pass_ms: float,
    warmup: int,
    rep: int,
) -> PartitionScanResult:
    stage1, merge, split_kernel, num_partitions = _split_kv_kernel_paths(
        inputs,
        partition_size,
        SPLIT_KV_BLOCK_N,
        SPLIT_KV_NUM_WARPS,
    )

    split_output = split_kernel()
    assert torch.isfinite(split_output).all()
    torch.testing.assert_close(
        single_pass_output, reference_output, rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        split_output, reference_output, rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        split_output, single_pass_output, rtol=1e-2, atol=1e-2
    )

    stage1_ms = _bench(stage1, warmup, rep)
    stage1()
    merge_ms = _bench(merge, warmup, rep)
    split_kernel_ms = _bench(split_kernel, warmup, rep)
    return PartitionScanResult(
        partition_size=partition_size,
        num_partitions=num_partitions,
        stage1_ms=stage1_ms,
        merge_ms=merge_ms,
        split_kernel_ms=split_kernel_ms,
        single_pass_ms=single_pass_ms,
        split_vs_single=single_pass_ms / split_kernel_ms,
    )


def benchmark_partition_scan(
    inputs: BenchmarkInputs,
    *,
    partition_sizes: list[int],
    warmup: int,
    rep: int,
) -> list[PartitionScanResult]:
    reference_output = paged_attention_decode(
        q=inputs.q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_table=inputs.block_table_list,
        seq_len=inputs.seq_len,
        block_size=inputs.block_size,
    )
    single_pass = lambda: _paged_attention_decode_triton(
        q=inputs.q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_table=inputs.block_table,
        seq_len=inputs.seq_len,
        block_n=SINGLE_PASS_BLOCK_N,
        num_warps=SINGLE_PASS_NUM_WARPS,
    )
    single_pass_output = single_pass()
    torch.testing.assert_close(
        single_pass_output, reference_output, rtol=1e-2, atol=1e-2
    )
    single_pass_ms = _bench(single_pass, warmup, rep)

    return [
        benchmark_partition_size(
            inputs,
            partition_size=partition_size,
            reference_output=reference_output,
            single_pass_output=single_pass_output,
            single_pass_ms=single_pass_ms,
            warmup=warmup,
            rep=rep,
        )
        for partition_size in partition_sizes
    ]


def print_partition_scan_results(results: list[PartitionScanResult]) -> None:
    headers = tuple(PartitionScanResult.__dataclass_fields__)
    rows = [
        (
            str(result.partition_size),
            str(result.num_partitions),
            f"{result.stage1_ms:.4f}",
            f"{result.merge_ms:.4f}",
            f"{result.split_kernel_ms:.4f}",
            f"{result.single_pass_ms:.4f}",
            f"{result.split_vs_single:.2f}x",
        )
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.rjust(width) for value, width in zip(row, widths)))


def save_partition_scan_csv(
    results: list[PartitionScanResult], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PartitionScanResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fieldnames})


def benchmark_split_config_scan(
    inputs: BenchmarkInputs,
    *,
    configs: list[tuple[int, int, int]],
    warmup: int,
    rep: int,
) -> list[SplitConfigScanResult]:
    reference_output = paged_attention_decode(
        q=inputs.q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_table=inputs.block_table_list,
        seq_len=inputs.seq_len,
        block_size=inputs.block_size,
    )
    single_pass = lambda: _paged_attention_decode_triton(
        q=inputs.q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_table=inputs.block_table,
        seq_len=inputs.seq_len,
        block_n=SINGLE_PASS_BLOCK_N,
        num_warps=SINGLE_PASS_NUM_WARPS,
    )
    single_pass_output = single_pass()
    torch.testing.assert_close(
        single_pass_output, reference_output, rtol=1e-2, atol=1e-2
    )
    single_pass_ms = _bench(single_pass, warmup, rep)

    results = []
    for partition_size, block_n, num_warps in configs:
        stage1, merge, split_kernel, num_partitions = _split_kv_kernel_paths(
            inputs,
            partition_size,
            block_n,
            num_warps,
        )
        split_output = split_kernel()
        assert torch.isfinite(split_output).all()
        torch.testing.assert_close(
            split_output, reference_output, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            split_output, single_pass_output, rtol=1e-2, atol=1e-2
        )

        stage1_ms = _bench(stage1, warmup, rep)
        stage1()
        merge_ms = _bench(merge, warmup, rep)
        split_kernel_ms = _bench(split_kernel, warmup, rep)
        results.append(
            SplitConfigScanResult(
                partition_size=partition_size,
                block_n=block_n,
                num_warps=num_warps,
                num_partitions=num_partitions,
                stage1_ms=stage1_ms,
                merge_ms=merge_ms,
                split_kernel_ms=split_kernel_ms,
                single_pass_ms=single_pass_ms,
                split_vs_single=single_pass_ms / split_kernel_ms,
            )
        )
    return results


def print_split_config_scan_results(
    results: list[SplitConfigScanResult],
) -> None:
    headers = tuple(SplitConfigScanResult.__dataclass_fields__)
    rows = [
        (
            str(result.partition_size),
            str(result.block_n),
            str(result.num_warps),
            str(result.num_partitions),
            f"{result.stage1_ms:.4f}",
            f"{result.merge_ms:.4f}",
            f"{result.split_kernel_ms:.4f}",
            f"{result.single_pass_ms:.4f}",
            f"{result.split_vs_single:.2f}x",
        )
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.rjust(width) for value, width in zip(row, widths)))


def save_split_config_scan_csv(
    results: list[SplitConfigScanResult], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SplitConfigScanResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fieldnames})


def benchmark_seq_split_config(
    inputs: BenchmarkInputs,
    *,
    partition_size: int,
    block_n: int,
    num_warps: int,
    reference_output: torch.Tensor,
    single_pass_output: torch.Tensor,
    single_pass_ms: float,
    warmup: int,
    rep: int,
) -> SeqSplitConfigScanResult:
    _, _, split_kernel, num_partitions = _split_kv_kernel_paths(
        inputs,
        partition_size,
        block_n,
        num_warps,
    )

    split_output = split_kernel()
    assert torch.isfinite(split_output).all()
    torch.testing.assert_close(
        single_pass_output, reference_output, rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        split_output, reference_output, rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        split_output, single_pass_output, rtol=1e-2, atol=1e-2
    )

    split_kernel_ms = _bench(split_kernel, warmup, rep)
    return SeqSplitConfigScanResult(
        seq_len=inputs.seq_len,
        partition_size=partition_size,
        block_n=block_n,
        num_warps=num_warps,
        num_partitions=num_partitions,
        split_kernel_ms=split_kernel_ms,
        single_pass_ms=single_pass_ms,
        split_vs_single=single_pass_ms / split_kernel_ms,
    )


def print_seq_split_config_scan_results(
    results: list[SeqSplitConfigScanResult],
) -> None:
    headers = tuple(SeqSplitConfigScanResult.__dataclass_fields__)
    rows = [
        (
            str(result.seq_len),
            str(result.partition_size),
            str(result.block_n),
            str(result.num_warps),
            str(result.num_partitions),
            f"{result.split_kernel_ms:.4f}",
            f"{result.single_pass_ms:.4f}",
            f"{result.split_vs_single:.2f}x",
        )
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.rjust(width) for value, width in zip(row, widths)))


def save_seq_split_config_scan_csv(
    results: list[SeqSplitConfigScanResult], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SeqSplitConfigScanResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fieldnames})


def benchmark_shape_split_config(
    inputs: BenchmarkInputs,
    *,
    partition_size: int,
    block_n: int,
    num_warps: int,
    reference_output: torch.Tensor,
    single_pass_output: torch.Tensor,
    single_pass_ms: float,
    warmup: int,
    rep: int,
) -> ShapeSplitConfigScanResult:
    result = benchmark_seq_split_config(
        inputs,
        partition_size=partition_size,
        block_n=block_n,
        num_warps=num_warps,
        reference_output=reference_output,
        single_pass_output=single_pass_output,
        single_pass_ms=single_pass_ms,
        warmup=warmup,
        rep=rep,
    )
    return ShapeSplitConfigScanResult(
        seq_len=result.seq_len,
        head_dim=inputs.q.shape[1],
        partition_size=result.partition_size,
        block_n=result.block_n,
        num_warps=result.num_warps,
        num_partitions=result.num_partitions,
        split_kernel_ms=result.split_kernel_ms,
        single_pass_ms=result.single_pass_ms,
        split_vs_single=result.split_vs_single,
    )


def print_shape_split_config_scan_results(
    results: list[ShapeSplitConfigScanResult],
) -> None:
    headers = tuple(ShapeSplitConfigScanResult.__dataclass_fields__)
    rows = [
        (
            str(result.seq_len),
            str(result.head_dim),
            str(result.partition_size),
            str(result.block_n),
            str(result.num_warps),
            str(result.num_partitions),
            f"{result.split_kernel_ms:.4f}",
            f"{result.single_pass_ms:.4f}",
            f"{result.split_vs_single:.2f}x",
        )
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.rjust(width) for value, width in zip(row, widths)))


def save_shape_split_config_scan_csv(
    results: list[ShapeSplitConfigScanResult], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ShapeSplitConfigScanResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fieldnames})


def benchmark_single_config(
    inputs: BenchmarkInputs,
    *,
    block_n: int,
    num_warps: int,
    reference_output: torch.Tensor,
    warmup: int,
    rep: int,
) -> SingleConfigScanResult:
    single_pass = lambda: _paged_attention_decode_triton(
        q=inputs.q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_table=inputs.block_table,
        seq_len=inputs.seq_len,
        block_n=block_n,
        num_warps=num_warps,
    )
    output = single_pass()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference_output, rtol=1e-2, atol=1e-2)
    single_pass_ms = _bench(single_pass, warmup, rep)
    return SingleConfigScanResult(
        seq_len=inputs.seq_len,
        head_dim=inputs.q.shape[1],
        block_n=block_n,
        num_warps=num_warps,
        single_pass_ms=single_pass_ms,
    )


def print_single_config_scan_results(
    results: list[SingleConfigScanResult],
) -> None:
    headers = tuple(SingleConfigScanResult.__dataclass_fields__)
    rows = [
        (
            str(result.seq_len),
            str(result.head_dim),
            str(result.block_n),
            str(result.num_warps),
            f"{result.single_pass_ms:.4f}",
        )
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.rjust(width) for value, width in zip(row, widths)))

    best_by_shape = {}
    for result in results:
        shape = (result.seq_len, result.head_dim)
        current = best_by_shape.get(shape)
        if current is None or result.single_pass_ms < current.single_pass_ms:
            best_by_shape[shape] = result
    print("\nBest single-pass config per shape:")
    for (seq_len, head_dim), result in best_by_shape.items():
        print(
            f"seq_len={seq_len}, D={head_dim}: BLOCK_N={result.block_n}, "
            f"num_warps={result.num_warps}, "
            f"latency={result.single_pass_ms:.4f} ms"
        )


def save_single_config_scan_csv(
    results: list[SingleConfigScanResult], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SingleConfigScanResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fieldnames})


def benchmark_tuning_configuration(
    inputs: BenchmarkInputs,
    *,
    block_n: int,
    num_warps: int,
    warmup: int,
    rep: int,
) -> TuningResult:
    dense = lambda: dense_attention_decode(
        inputs.q, inputs.logical_k, inputs.logical_v
    )
    triton_paged = lambda: _paged_attention_decode_triton(
        q=inputs.q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_table=inputs.block_table,
        seq_len=inputs.seq_len,
        block_n=block_n,
        num_warps=num_warps,
    )

    try:
        dense_output = dense()
        triton_output = triton_paged()  # Compile before timing this configuration.
        torch.testing.assert_close(
            triton_output, dense_output, rtol=1e-2, atol=1e-2
        )
        dense_ms = _bench(dense, warmup, rep)
        triton_ms = _bench(triton_paged, warmup, rep)
        return TuningResult(
            seq_len=inputs.seq_len,
            BLOCK_N=block_n,
            num_warps=num_warps,
            triton_ms=triton_ms,
            dense_ms=dense_ms,
            triton_vs_dense=dense_ms / triton_ms,
            status="OK",
            error=None,
        )
    except Exception as error:
        return TuningResult(
            seq_len=inputs.seq_len,
            BLOCK_N=block_n,
            num_warps=num_warps,
            triton_ms=None,
            dense_ms=None,
            triton_vs_dense=None,
            status="FAILED",
            error=f"{type(error).__name__}: {error}",
        )


def print_tuning_results(results: list[TuningResult]) -> None:
    headers = (
        "seq_len", "BLOCK_N", "num_warps", "triton_ms", "dense_ms",
        "triton_vs_dense",
    )
    rows = []
    for result in results:
        if result.status == "OK":
            rows.append((
                str(result.seq_len),
                str(result.BLOCK_N),
                str(result.num_warps),
                f"{result.triton_ms:.4f}",
                f"{result.dense_ms:.4f}",
                f"{result.triton_vs_dense:.2f}x",
            ))
        else:
            rows.append((
                str(result.seq_len),
                str(result.BLOCK_N),
                str(result.num_warps),
                "FAILED", "FAILED", "FAILED",
            ))
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.rjust(width) for value, width in zip(row, widths)))
    for result in results:
        if result.status == "FAILED":
            print(
                f"FAILED seq_len={result.seq_len}, BLOCK_N={result.BLOCK_N}, "
                f"num_warps={result.num_warps}: {result.error}"
            )


def save_tuning_csv(results: list[TuningResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TuningResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                field: (
                    getattr(result, field)
                    if getattr(result, field) is not None
                    else "N/A"
                )
                for field in fieldnames
            })


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.csv is not None:
        prepare_csv_output(args.csv)
    dtype = DTYPES[args.dtype]

    print(
        "Configuration: "
        f"dtype={args.dtype}, Hq={args.num_query_heads}, "
        f"Hkv={args.num_kv_heads}, D={args.head_dim}, "
        f"block_size={args.block_size}"
    )
    print("Dense PyTorch is the dense baseline over contiguous logical K/V.")
    if args.tune:
        print(
            f"Tuning BLOCK_N={args.block_ns}, "
            f"num_warps={args.num_warps_values}"
        )
    if args.split_kv:
        print(
            "Split-KV benchmark: "
            f"single-pass BLOCK_N={SINGLE_PASS_BLOCK_N}, "
            f"num_warps={SINGLE_PASS_NUM_WARPS}; "
            f"Split-KV PARTITION_SIZE={SPLIT_KV_PARTITION_SIZE}, "
            f"BLOCK_N={SPLIT_KV_BLOCK_N}, num_warps={SPLIT_KV_NUM_WARPS}"
        )
    if args.partition_sizes is not None:
        print(
            "Split-KV PARTITION_SIZE scan: "
            f"seq_len=4096, partition_sizes={args.partition_sizes}, "
            f"BLOCK_N={SPLIT_KV_BLOCK_N}, num_warps={SPLIT_KV_NUM_WARPS}; "
            f"single-pass BLOCK_N={SINGLE_PASS_BLOCK_N}, "
            f"num_warps={SINGLE_PASS_NUM_WARPS}"
        )
    if args.split_configs is not None:
        print(
            "Split-KV explicit config scan: "
            f"seq_len=4096, configs={args.split_configs}, "
            f"single-pass BLOCK_N={SINGLE_PASS_BLOCK_N}, "
            f"num_warps={SINGLE_PASS_NUM_WARPS}"
        )
    if args.seq_split_configs is not None:
        print(
            "Split-KV cross-sequence config scan: "
            f"configs={args.seq_split_configs}, "
            f"single-pass BLOCK_N={SINGLE_PASS_BLOCK_N}, "
            f"num_warps={SINGLE_PASS_NUM_WARPS}"
        )
    if args.shape_split_configs is not None:
        print(
            "Split-KV shape config scan: "
            f"configs={args.shape_split_configs}, "
            f"single-pass BLOCK_N={SINGLE_PASS_BLOCK_N}, "
            f"num_warps={SINGLE_PASS_NUM_WARPS}"
        )
    if args.single_configs is not None:
        print(f"Single-pass explicit config scan: configs={args.single_configs}")
    if args.benchmark_reference:
        print(
            "paged_reference_ms includes Python dispatch/gather/stack overhead; "
            "it is not a fair comparison with dense/Triton operator latency."
        )
    else:
        print(
            "Paged PyTorch is used for correctness only; pass "
            "--benchmark-reference to time it."
        )
    results = []
    if args.single_configs is not None:
        configs_by_shape = {}
        for seq_len, head_dim, block_n, num_warps in args.single_configs:
            configs_by_shape.setdefault((seq_len, head_dim), []).append(
                (block_n, num_warps)
            )

        for (seq_len, head_dim), configs in configs_by_shape.items():
            inputs = create_benchmark_inputs(
                seq_len=seq_len,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=head_dim,
                block_size=args.block_size,
                dtype=dtype,
            )
            reference_output = paged_attention_decode(
                q=inputs.q,
                k_cache=inputs.k_cache,
                v_cache=inputs.v_cache,
                block_table=inputs.block_table_list,
                seq_len=inputs.seq_len,
                block_size=inputs.block_size,
            )
            for block_n, num_warps in configs:
                try:
                    results.append(
                        benchmark_single_config(
                            inputs,
                            block_n=block_n,
                            num_warps=num_warps,
                            reference_output=reference_output,
                            warmup=args.warmup,
                            rep=args.rep,
                        )
                    )
                except Exception as error:
                    raise RuntimeError(
                        "single-pass config scan failed for "
                        f"seq_len={seq_len}, head_dim={head_dim}, "
                        f"BLOCK_N={block_n}, num_warps={num_warps}"
                    ) from error
        print_single_config_scan_results(results)
        if args.csv is not None:
            save_single_config_scan_csv(results, args.csv)
            print(f"Saved CSV to {args.csv}")
        return

    if args.shape_split_configs is not None:
        configs_by_shape = {}
        for config in args.shape_split_configs:
            seq_len, head_dim, partition_size, block_n, num_warps = config
            configs_by_shape.setdefault((seq_len, head_dim), []).append(
                (partition_size, block_n, num_warps)
            )

        for (seq_len, head_dim), configs in configs_by_shape.items():
            inputs = create_benchmark_inputs(
                seq_len=seq_len,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=head_dim,
                block_size=args.block_size,
                dtype=dtype,
            )
            reference_output = paged_attention_decode(
                q=inputs.q,
                k_cache=inputs.k_cache,
                v_cache=inputs.v_cache,
                block_table=inputs.block_table_list,
                seq_len=inputs.seq_len,
                block_size=inputs.block_size,
            )
            single_pass = lambda: _paged_attention_decode_triton(
                q=inputs.q,
                k_cache=inputs.k_cache,
                v_cache=inputs.v_cache,
                block_table=inputs.block_table,
                seq_len=inputs.seq_len,
                block_n=SINGLE_PASS_BLOCK_N,
                num_warps=SINGLE_PASS_NUM_WARPS,
            )
            single_pass_output = single_pass()
            torch.testing.assert_close(
                single_pass_output,
                reference_output,
                rtol=1e-2,
                atol=1e-2,
            )
            single_pass_ms = _bench(single_pass, args.warmup, args.rep)

            for partition_size, block_n, num_warps in configs:
                try:
                    results.append(
                        benchmark_shape_split_config(
                            inputs,
                            partition_size=partition_size,
                            block_n=block_n,
                            num_warps=num_warps,
                            reference_output=reference_output,
                            single_pass_output=single_pass_output,
                            single_pass_ms=single_pass_ms,
                            warmup=args.warmup,
                            rep=args.rep,
                        )
                    )
                except Exception as error:
                    raise RuntimeError(
                        "Split-KV shape config scan failed for "
                        f"seq_len={seq_len}, head_dim={head_dim}, "
                        f"partition_size={partition_size}, BLOCK_N={block_n}, "
                        f"num_warps={num_warps}"
                    ) from error
        print_shape_split_config_scan_results(results)
        if args.csv is not None:
            save_shape_split_config_scan_csv(results, args.csv)
            print(f"Saved CSV to {args.csv}")
        return

    if args.seq_split_configs is not None:
        configs_by_seq_len = {}
        for seq_len, partition_size, block_n, num_warps in args.seq_split_configs:
            configs_by_seq_len.setdefault(seq_len, []).append(
                (partition_size, block_n, num_warps)
            )

        for seq_len, configs in configs_by_seq_len.items():
            inputs = create_benchmark_inputs(
                seq_len=seq_len,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                block_size=args.block_size,
                dtype=dtype,
            )
            reference_output = paged_attention_decode(
                q=inputs.q,
                k_cache=inputs.k_cache,
                v_cache=inputs.v_cache,
                block_table=inputs.block_table_list,
                seq_len=inputs.seq_len,
                block_size=inputs.block_size,
            )
            single_pass = lambda: _paged_attention_decode_triton(
                q=inputs.q,
                k_cache=inputs.k_cache,
                v_cache=inputs.v_cache,
                block_table=inputs.block_table,
                seq_len=inputs.seq_len,
                block_n=SINGLE_PASS_BLOCK_N,
                num_warps=SINGLE_PASS_NUM_WARPS,
            )
            single_pass_output = single_pass()
            torch.testing.assert_close(
                single_pass_output,
                reference_output,
                rtol=1e-2,
                atol=1e-2,
            )
            single_pass_ms = _bench(single_pass, args.warmup, args.rep)

            for partition_size, block_n, num_warps in configs:
                try:
                    results.append(
                        benchmark_seq_split_config(
                            inputs,
                            partition_size=partition_size,
                            block_n=block_n,
                            num_warps=num_warps,
                            reference_output=reference_output,
                            single_pass_output=single_pass_output,
                            single_pass_ms=single_pass_ms,
                            warmup=args.warmup,
                            rep=args.rep,
                        )
                    )
                except Exception as error:
                    raise RuntimeError(
                        "Split-KV cross-sequence config scan failed for "
                        f"seq_len={seq_len}, partition_size={partition_size}, "
                        f"BLOCK_N={block_n}, num_warps={num_warps}"
                    ) from error
        print_seq_split_config_scan_results(results)
        if args.csv is not None:
            save_seq_split_config_scan_csv(results, args.csv)
            print(f"Saved CSV to {args.csv}")
        return

    for seq_len in args.seq_lens:
        inputs = create_benchmark_inputs(
            seq_len=seq_len,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            block_size=args.block_size,
            dtype=dtype,
        )
        if args.split_configs is not None:
            try:
                results.extend(
                    benchmark_split_config_scan(
                        inputs,
                        configs=args.split_configs,
                        warmup=args.warmup,
                        rep=args.rep,
                    )
                )
            except Exception as error:
                raise RuntimeError(
                    "Split-KV explicit config scan failed for seq_len=4096"
                ) from error
            continue
        if args.partition_sizes is not None:
            try:
                results.extend(
                    benchmark_partition_scan(
                        inputs,
                        partition_sizes=args.partition_sizes,
                        warmup=args.warmup,
                        rep=args.rep,
                    )
                )
            except Exception as error:
                raise RuntimeError(
                    "Split-KV PARTITION_SIZE scan failed for seq_len=4096"
                ) from error
            continue
        if args.split_kv:
            try:
                result = benchmark_split_kv_configuration(
                    inputs,
                    warmup=args.warmup,
                    rep=args.rep,
                )
            except Exception as error:
                raise RuntimeError(
                    "Split-KV PagedAttention benchmark failed for "
                    f"seq_len={seq_len}"
                ) from error
            results.append(result)
            continue
        if args.tune:
            for block_n in args.block_ns:
                for num_warps in args.num_warps_values:
                    result = benchmark_tuning_configuration(
                        inputs,
                        block_n=block_n,
                        num_warps=num_warps,
                        warmup=args.warmup,
                        rep=args.rep,
                    )
                    results.append(result)
                    if result.status == "FAILED":
                        print(
                            f"FAILED seq_len={seq_len}, BLOCK_N={block_n}, "
                            f"num_warps={num_warps}: {result.error}"
                        )
            continue
        try:
            result = benchmark_one_configuration(
                inputs,
                warmup=args.warmup,
                rep=args.rep,
                benchmark_reference=args.benchmark_reference,
                reference_reps=args.reference_reps,
            )
        except Exception as error:
            raise RuntimeError(
                "PagedAttention benchmark failed for "
                f"seq_len={seq_len}, dtype={args.dtype}, "
                f"Hq={args.num_query_heads}, Hkv={args.num_kv_heads}, "
                f"D={args.head_dim}, block_size={args.block_size}"
            ) from error
        results.append(result)

    if args.split_configs is not None:
        print_split_config_scan_results(results)
    elif args.partition_sizes is not None:
        print_partition_scan_results(results)
    elif args.split_kv:
        print_split_kv_results(results)
    elif args.tune:
        print_tuning_results(results)
    else:
        print_results(results)
    if args.csv is not None:
        if args.split_configs is not None:
            save_split_config_scan_csv(results, args.csv)
        elif args.partition_sizes is not None:
            save_partition_scan_csv(results, args.csv)
        elif args.split_kv:
            save_split_kv_csv(results, args.csv)
        elif args.tune:
            save_tuning_csv(results, args.csv)
        else:
            save_csv(results, args.csv)
        print(f"Saved CSV to {args.csv}")


if __name__ == "__main__":
    main()
