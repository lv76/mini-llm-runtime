"""Compare real CUDA storage for paged and dense-padded KV caches."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from mini_llm_runtime.model.qwen2 import TinyQwen2Config


DTYPES = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
}


@dataclass(frozen=True)
class KVLayout:
    max_num_seqs: int
    request_capacity_tokens: list[int]
    selected_capacity_tokens: list[int]
    useful_token_slots: int
    paged_blocks: int
    paged_token_slots: int
    dense_token_slots: int
    paged_packing_efficiency: float
    dense_packing_efficiency: float
    paged_saving_vs_dense: float


def parse_positive_int_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('expected comma-separated integers') from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError('values must be positive integers')
    return values


def expand_values(values: list[int], count: int, name: str) -> list[int]:
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f'{name} must contain either 1 or {count} values')
    return values


def compute_layout(
    prompt_lens: list[int],
    output_lens: list[int],
    block_size: int,
    max_num_seqs: int,
) -> KVLayout:
    if not prompt_lens or any(value <= 0 for value in prompt_lens):
        raise ValueError('prompt_lens must be non-empty and positive')
    output_lens = expand_values(output_lens, len(prompt_lens), 'output_lens')
    if any(value <= 0 for value in output_lens):
        raise ValueError('output_lens must be positive')
    if block_size <= 0 or max_num_seqs <= 0:
        raise ValueError('block_size and max_num_seqs must be positive')
    concurrency = min(max_num_seqs, len(prompt_lens))
    # The final sampled token is returned to the user but is never consumed by
    # the model, so it does not need a KV entry.
    capacities = [
        prompt_len + output_len - 1
        for prompt_len, output_len in zip(prompt_lens, output_lens)
    ]
    selected = sorted(capacities, reverse=True)[:concurrency]
    useful_slots = sum(selected)
    paged_blocks = sum(math.ceil(value / block_size) for value in selected)
    paged_slots = paged_blocks * block_size
    dense_slots = concurrency * max(capacities)
    return KVLayout(
        max_num_seqs=concurrency,
        request_capacity_tokens=capacities,
        selected_capacity_tokens=selected,
        useful_token_slots=useful_slots,
        paged_blocks=paged_blocks,
        paged_token_slots=paged_slots,
        dense_token_slots=dense_slots,
        paged_packing_efficiency=useful_slots / paged_slots,
        dense_packing_efficiency=useful_slots / dense_slots,
        paged_saving_vs_dense=1 - paged_slots / dense_slots,
    )


def tensor_storage_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


def measure_cuda_allocation(shape: tuple[int, ...], dtype: torch.dtype) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    k_cache = torch.empty(shape, dtype=dtype, device='cuda')
    v_cache = torch.empty_like(k_cache)
    torch.cuda.synchronize()
    result = {
        'allocated_bytes': torch.cuda.memory_allocated() - allocated_before,
        'reserved_bytes': torch.cuda.memory_reserved() - reserved_before,
        'tensor_bytes': k_cache.numel() * k_cache.element_size() * 2,
    }
    del k_cache, v_cache
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model-config',
        type=Path,
        default=Path('checkpoints/Qwen2.5-0.5B/config.json'),
    )
    parser.add_argument('--prompt-lens', type=parse_positive_int_list, required=True)
    parser.add_argument('--output-lens', type=parse_positive_int_list, required=True)
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument('--max-num-seqs', type=int)
    parser.add_argument('--dtype', choices=DTYPES, default='bfloat16')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmarks/results/kv_cache_memory.csv'),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for allocation measurements')
    config = TinyQwen2Config.from_dict(
        json.loads(args.model_config.read_text())
    )
    output_lens = expand_values(
        args.output_lens, len(args.prompt_lens), 'output_lens'
    )
    max_num_seqs = (
        len(args.prompt_lens) if args.max_num_seqs is None else args.max_num_seqs
    )
    layout = compute_layout(
        args.prompt_lens, output_lens, args.block_size, max_num_seqs
    )
    dtype = DTYPES[args.dtype]
    common_tail = (config.num_key_value_heads, config.head_dim)
    paged_shape = (
        config.num_hidden_layers,
        layout.paged_blocks,
        args.block_size,
        *common_tail,
    )
    dense_shape = (
        config.num_hidden_layers,
        layout.max_num_seqs,
        max(layout.request_capacity_tokens),
        *common_tail,
    )
    paged = measure_cuda_allocation(paged_shape, dtype)
    dense = measure_cuda_allocation(dense_shape, dtype)
    bytes_per_kv_token = (
        2
        * config.num_hidden_layers
        * config.num_key_value_heads
        * config.head_dim
        * torch.empty((), dtype=dtype).element_size()
    )
    row = {
        'dtype': args.dtype,
        'block_size': args.block_size,
        'max_num_seqs': layout.max_num_seqs,
        'num_layers': config.num_hidden_layers,
        'num_kv_heads': config.num_key_value_heads,
        'head_dim': config.head_dim,
        'bytes_per_kv_token': bytes_per_kv_token,
        'useful_token_slots': layout.useful_token_slots,
        'paged_blocks': layout.paged_blocks,
        'paged_token_slots': layout.paged_token_slots,
        'dense_token_slots': layout.dense_token_slots,
        'paged_packing_efficiency': layout.paged_packing_efficiency,
        'dense_packing_efficiency': layout.dense_packing_efficiency,
        'paged_saving_vs_dense': layout.paged_saving_vs_dense,
        'paged_tensor_mib': paged['tensor_bytes'] / 2**20,
        'dense_tensor_mib': dense['tensor_bytes'] / 2**20,
        'measured_paged_allocated_mib': paged['allocated_bytes'] / 2**20,
        'measured_dense_allocated_mib': dense['allocated_bytes'] / 2**20,
        'measured_paged_reserved_mib': paged['reserved_bytes'] / 2**20,
        'measured_dense_reserved_mib': dense['reserved_bytes'] / 2**20,
        'measured_allocated_saving_vs_dense': (
            1 - paged['allocated_bytes'] / dense['allocated_bytes']
        ),
        'measured_reserved_saving_vs_dense': (
            1 - paged['reserved_bytes'] / dense['reserved_bytes']
        ),
    }
    metadata = {
        'args': {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        'layout': asdict(layout),
        'paged_shape_per_k_or_v': paged_shape,
        'dense_shape_per_k_or_v': dense_shape,
        'paged_measurement': paged,
        'dense_measurement': dense,
        'result': row,
        'baseline_note': (
            'Dense cache uses max_num_seqs rows padded to the largest request '
            'capacity. Paged cache holds the largest max_num_seqs requests in '
            'whole blocks. Both measurements allocate one K and one V tensor.'
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    args.output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2) + '\n'
    )
    print(json.dumps(row, indent=2))
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
