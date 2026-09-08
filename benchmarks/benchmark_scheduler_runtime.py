"""Benchmark Scheduler-driven generation with fixed or staggered traffic.

The timed region starts at workload arrival time zero and ends when every
request finishes. Model loading and dense correctness generation are excluded.
Requests can use ragged prompt/output lengths, staggered arrivals, constrained
Scheduler concurrency, and a constrained physical KV-cache pool.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from mini_llm_runtime.model.qwen2 import TinyQwen2Config, TinyQwen2ForCausalLM
from mini_llm_runtime.model.qwen2_loader import load_qwen2_safetensors
from mini_llm_runtime.runtime.model_runner import run_generation_step
from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache
from mini_llm_runtime.runtime.request import Request
from mini_llm_runtime.runtime.scheduler import Scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('checkpoints/Qwen2.5-0.5B'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--num-requests', type=int, default=4)
    parser.add_argument('--prompt-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=16)
    parser.add_argument(
        '--prompt-lens',
        type=parse_positive_int_list,
        help='Comma-separated random prompt lengths; overrides --num-requests and --prompt-len.',
    )
    parser.add_argument(
        '--output-lens',
        type=parse_positive_int_list,
        help='Comma-separated per-request output lengths; one value broadcasts.',
    )
    arrival_group = parser.add_mutually_exclusive_group()
    arrival_group.add_argument(
        '--arrival-times-ms',
        type=parse_nonnegative_float_list,
        help='Comma-separated per-request arrival offsets in milliseconds.',
    )
    arrival_group.add_argument(
        '--arrival-interval-ms',
        type=float,
        default=0.0,
        help='Fixed interval between consecutive request arrivals.',
    )
    parser.add_argument(
        '--max-num-seqs',
        type=int,
        help='Maximum concurrent running requests; defaults to the request count.',
    )
    parser.add_argument(
        '--max-num-batched-tokens',
        type=int,
        help='Scheduler token budget; defaults to the sum of prompt lengths.',
    )
    parser.add_argument(
        '--kv-cache-blocks',
        type=int,
        help='Physical KV blocks; defaults to enough blocks for all requests.',
    )
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--prompt-text-file',
        type=Path,
        help='JSON list of real prompt strings; overrides random prompt generation.',
    )
    reduction_group = parser.add_mutually_exclusive_group()
    reduction_group.add_argument(
        '--disable-bf16-reduced-precision-reduction',
        dest='disable_bf16_reduced_precision_reduction',
        action='store_true',
        help='Use the default accuracy-oriented mixed-precision path.',
    )
    reduction_group.add_argument(
        '--allow-bf16-reduced-precision-reduction',
        dest='disable_bf16_reduced_precision_reduction',
        action='store_false',
        help='Opt into faster but less accurate BF16 GEMM reductions.',
    )
    parser.set_defaults(disable_bf16_reduced_precision_reduction=True)
    parser.add_argument(
        '--strict-greedy-validation',
        action='store_true',
        help='Fail if Scheduler output token IDs differ from dense greedy generation.',
    )
    parser.add_argument(
        '--strict-runtime-validation',
        action='store_true',
        help='Fail unless Batched Scheduler exactly matches Isolated Paged.',
    )
    parser.add_argument(
        '--capture-top2',
        action='store_true',
        help='Save per-step top-2 logits and the first three-path divergence.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmarks/results/scheduler_runtime.csv'),
    )
    args = parser.parse_args()
    positive = [
        args.num_requests,
        args.prompt_len,
        args.output_len,
        args.block_size,
        args.samples,
        args.threads,
    ]
    optional_positive = [
        args.max_num_seqs,
        args.max_num_batched_tokens,
        args.kv_cache_blocks,
    ]
    if (
        min(positive) <= 0
        or args.warmup < 0
        or args.arrival_interval_ms < 0
        or any(value is not None and value <= 0 for value in optional_positive)
    ):
        parser.error('sizes, samples, and threads must be positive; warmup must be non-negative')
    return args


def parse_positive_int_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('expected comma-separated integers') from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError('values must be positive integers')
    return values


def parse_nonnegative_float_list(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('expected comma-separated numbers') from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError('values must be non-negative')
    return values


def expand_request_values(values, count, name):
    """Broadcast one value or validate one value per request."""
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f'{name} must contain either 1 or {count} values')
    return list(values)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError('percentile requires at least one value')
    if not 0 < probability <= 1:
        raise ValueError('probability must be in (0, 1]')
    ordered = sorted(values)
    return ordered[math.ceil(probability * len(ordered)) - 1]


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {'mean': 0.0, 'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0}
    return {
        'mean': statistics.mean(values),
        'p50': percentile(values, 0.50),
        'p95': percentile(values, 0.95),
        'p99': percentile(values, 0.99),
        'max': max(values),
    }


def summarize_samples(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('cannot summarize an empty sample list')
    metrics = [
        'ttft_mean_ms',
        'ttft_p50_ms',
        'ttft_p95_ms',
        'ttft_p99_ms',
        'queue_mean_ms',
        'queue_p95_ms',
        'tpot_mean_ms',
        'itl_p95_ms',
        'itl_p99_ms',
        'e2e_max_ms',
        'output_tokens_per_second',
        'request_throughput_per_second',
        'incremental_peak_allocated_mib',
        'peak_kv_utilization',
        'kv_pool_allocated_mib',
        'kv_pool_saving_vs_dense_static',
        'peak_paged_committed_mib',
        'peak_block_packing_efficiency',
        'peak_saving_vs_dense_padded',
    ]
    summary = {'num_samples': len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        summary[metric] = {
            'p50': statistics.median(values),
            'p95': percentile(values, 0.95),
            'min': min(values),
            'max': max(values),
        }
    summary['all_greedy_match'] = all(row['greedy_match'] for row in rows)
    summary['all_runtime_match'] = all(
        row.get('runtime_match', row['greedy_match']) for row in rows
    )
    return summary


def top2_snapshot(logits: torch.Tensor) -> dict:
    """Return JSON-safe top-2 token/logit data for one vocabulary row."""
    if logits.ndim != 1 or logits.numel() < 2:
        raise ValueError('logits must be one-dimensional with at least 2 entries')
    if not torch.isfinite(logits).all():
        raise ValueError('logits must contain only finite values')
    float_logits = logits.float()
    # Match greedy_sample's argmax tie-breaking exactly. torch.topk may return
    # a different token first when two BF16 logits are bitwise equal.
    top1_token = int(torch.argmax(float_logits).item())
    top1_logit = float(float_logits[top1_token].item())
    runner_up_logits = float_logits.clone()
    runner_up_logits[top1_token] = -torch.inf
    top2_token = int(torch.argmax(runner_up_logits).item())
    top2_logit = float(runner_up_logits[top2_token].item())
    return {
        'top1_token_id': top1_token,
        'top1_logit': top1_logit,
        'top2_token_id': top2_token,
        'top2_logit': top2_logit,
        'top2_margin': top1_logit - top2_logit,
    }


def build_first_divergences(
    dense_tokens,
    isolated_tokens,
    batched_tokens,
    dense_top2,
    isolated_top2,
    batched_top2,
):
    """Describe only the first meaningful divergence for each request."""
    reports = []
    paths = {
        'dense': (dense_tokens, dense_top2),
        'isolated_paged': (isolated_tokens, isolated_top2),
        'batched_scheduler': (batched_tokens, batched_top2),
    }
    for request_index in range(len(dense_tokens)):
        num_steps = min(
            len(tokens[request_index]) for tokens, _ in paths.values()
        )
        for output_index in range(num_steps):
            chosen = {
                name: tokens[request_index][output_index]
                for name, (tokens, _) in paths.items()
            }
            if len(set(chosen.values())) == 1:
                continue
            reports.append({
                'request_index': request_index,
                'output_index': output_index,
                'output_position_1_based': output_index + 1,
                'shared_prefix_token_ids': dense_tokens[request_index][:output_index],
                'paths': {
                    name: {
                        'chosen_token_id': chosen[name],
                        **snapshots[request_index][output_index],
                    }
                    for name, (_, snapshots) in paths.items()
                },
            })
            break
    return reports


def kv_memory_snapshot(step, scheduler, kv_cache) -> dict:
    """Measure live KV payload, block fragmentation, and dense padding.

    The step has already released finished requests, so its ModelBatch is also
    consulted to reconstruct the just-executed pre-update peak.
    """
    contexts = {}
    batch = step.model_batch
    for request_index, request_id in enumerate(batch.request_ids):
        contexts[request_id] = int(batch.context_lens[request_index].item())
    used_blocks = int((batch.block_tables >= 0).sum().item())
    batched_request_ids = set(batch.request_ids)
    for request in scheduler.running:
        contexts.setdefault(request.request_id, request.num_computed_tokens)
        if request.request_id not in batched_request_ids:
            used_blocks += len(request.block_table)

    allocated_slots = used_blocks * kv_cache.block_size
    useful_slots = sum(contexts.values())
    dense_padded_slots = (
        len(contexts) * max(contexts.values()) if contexts else 0
    )
    bytes_per_token = (
        2
        * kv_cache.num_layers
        * kv_cache.num_kv_heads
        * kv_cache.head_dim
        * kv_cache.k_cache.element_size()
    )
    return {
        'live_requests': len(contexts),
        'live_kv_tokens': useful_slots,
        'used_blocks': used_blocks,
        'allocated_slots': allocated_slots,
        'internal_fragmentation_slots': allocated_slots - useful_slots,
        'block_packing_efficiency': (
            useful_slots / allocated_slots if allocated_slots else 1.0
        ),
        'dense_padded_slots': dense_padded_slots,
        'paged_committed_mib': allocated_slots * bytes_per_token / 2**20,
        'useful_kv_mib': useful_slots * bytes_per_token / 2**20,
        'internal_fragmentation_mib': (
            (allocated_slots - useful_slots) * bytes_per_token / 2**20
        ),
        'dense_padded_mib': dense_padded_slots * bytes_per_token / 2**20,
        'saving_vs_dense_padded': (
            1 - allocated_slots / dense_padded_slots
            if dense_padded_slots else 0.0
        ),
    }


@torch.inference_mode()
def dense_greedy_trace(model, prompt_token_ids, output_lens):
    device = next(model.parameters()).device
    output_lens = expand_request_values(
        [output_lens] if isinstance(output_lens, int) else output_lens,
        len(prompt_token_ids),
        'output_lens',
    )
    histories = [list(tokens) for tokens in prompt_token_ids]
    outputs = [[] for _ in histories]
    snapshots = [[] for _ in histories]
    for index, (history, output_len) in enumerate(zip(histories, output_lens)):
        for _ in range(output_len):
            input_ids = torch.tensor([history], dtype=torch.long, device=device)
            snapshot = top2_snapshot(model(input_ids)[0, -1])
            token = snapshot['top1_token_id']
            outputs[index].append(token)
            snapshots[index].append(snapshot)
            history.append(token)
    return {'token_ids': outputs, 'top2': snapshots}


def dense_greedy_tokens(model, prompt_token_ids, output_lens):
    return dense_greedy_trace(model, prompt_token_ids, output_lens)['token_ids']


@torch.inference_mode()
def isolated_paged_greedy_trace(
    model,
    config,
    prompt_token_ids,
    output_lens,
    block_size,
):
    """Generate each request alone through the paged Scheduler path."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    output_lens = expand_request_values(
        [output_lens] if isinstance(output_lens, int) else output_lens,
        len(prompt_token_ids),
        'output_lens',
    )
    outputs = []
    snapshots = []
    for index, (tokens, output_len) in enumerate(
        zip(prompt_token_ids, output_lens)
    ):
        num_blocks = math.ceil((len(tokens) + output_len) / block_size)
        kv_cache = PagedKVCache(
            num_layers=config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=dtype,
            device=device,
        )
        scheduler = Scheduler(
            max_num_batched_tokens=len(tokens),
            max_num_seqs=1,
            kv_cache=kv_cache,
        )
        request = Request(
            request_id=f'isolated-{index}',
            prompt_token_ids=list(tokens),
            max_new_tokens=output_len,
            arrival_time=0.0,
        )
        scheduler.add_request(request)
        request_snapshots = []
        while scheduler.waiting or scheduler.running:
            step = run_generation_step(model, scheduler)
            end = int(step.model_batch.query_start_loc[1].item())
            request_snapshots.append(top2_snapshot(step.packed_logits[end - 1]))
        outputs.append(request.output_token_ids)
        snapshots.append(request_snapshots)
    return {'token_ids': outputs, 'top2': snapshots}


def isolated_paged_greedy_tokens(
    model,
    config,
    prompt_token_ids,
    output_lens,
    block_size,
):
    return isolated_paged_greedy_trace(
        model,
        config,
        prompt_token_ids,
        output_lens,
        block_size,
    )['token_ids']


@torch.inference_mode()
def run_workload(
    model,
    config,
    prompt_token_ids,
    output_lens,
    block_size,
    dense_expected_tokens,
    isolated_expected_tokens,
    dense_top2=None,
    isolated_top2=None,
    capture_top2=False,
    arrival_times_ms=None,
    max_num_seqs=None,
    max_num_batched_tokens=None,
    kv_cache_blocks=None,
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_requests = len(prompt_token_ids)
    output_lens = expand_request_values(
        [output_lens] if isinstance(output_lens, int) else output_lens,
        num_requests,
        'output_lens',
    )
    arrival_times_ms = expand_request_values(
        [0.0] if arrival_times_ms is None else arrival_times_ms,
        num_requests,
        'arrival_times_ms',
    )
    if any(value < 0 for value in arrival_times_ms):
        raise ValueError('arrival_times_ms must be non-negative')
    num_prompt_tokens = sum(len(tokens) for tokens in prompt_token_ids)
    required_blocks = [
        math.ceil((len(tokens) + max(output_len - 1, 0)) / block_size)
        for tokens, output_len in zip(prompt_token_ids, output_lens)
    ]
    num_blocks = (
        sum(required_blocks) if kv_cache_blocks is None else kv_cache_blocks
    )
    if num_blocks < max(required_blocks):
        raise ValueError(
            'kv_cache_blocks must fit the largest request through completion'
        )
    max_num_seqs = num_requests if max_num_seqs is None else max_num_seqs
    max_num_batched_tokens = (
        num_prompt_tokens
        if max_num_batched_tokens is None
        else max_num_batched_tokens
    )
    if max_num_batched_tokens < max(map(len, prompt_token_ids)):
        raise ValueError(
            'max_num_batched_tokens must fit the largest prompt'
        )
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=dtype,
        device=device,
    )
    scheduler = Scheduler(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        kv_cache=kv_cache,
    )
    requests = []
    for index, (tokens, output_len, arrival_ms) in enumerate(
        zip(prompt_token_ids, output_lens, arrival_times_ms)
    ):
        request = Request(
            request_id=f'request-{index}',
            prompt_token_ids=list(tokens),
            max_new_tokens=output_len,
            arrival_time=arrival_ms / 1000,
        )
        requests.append(request)

    pending_requests = sorted(requests, key=lambda request: request.arrival_time)
    token_timestamps = {request.request_id: [] for request in requests}
    batched_top2 = {request.request_id: [] for request in requests}
    scheduler_enqueue_times = {}
    first_scheduled_times = {}
    iterations = []
    kv_memory_snapshots = []
    peak_used_blocks = 0
    diagnostic_seconds = 0.0
    memory_metrics_seconds = 0.0
    top2_seconds = 0.0
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    while pending_requests or scheduler.waiting or scheduler.running:
        elapsed = time.perf_counter() - start - diagnostic_seconds
        while pending_requests and pending_requests[0].arrival_time <= elapsed:
            request = pending_requests.pop(0)
            scheduler.add_request(request)
            scheduler_enqueue_times[request.request_id] = elapsed

        if not scheduler.waiting and not scheduler.running:
            wait_seconds = pending_requests[0].arrival_time - elapsed
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            continue

        iteration_start = time.perf_counter()
        iteration_start_elapsed = iteration_start - start - diagnostic_seconds
        step = run_generation_step(model, scheduler)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        iteration_end = time.perf_counter()
        elapsed = iteration_end - start - diagnostic_seconds
        for request in step.scheduler_output.prefill_requests:
            first_scheduled_times.setdefault(
                request.request_id, iteration_start_elapsed
            )
        for request_id in step.sampled_token_ids:
            token_timestamps[request_id].append(elapsed)
        metrics_start = time.perf_counter()
        memory_snapshot = kv_memory_snapshot(step, scheduler, kv_cache)
        metrics_elapsed = time.perf_counter() - metrics_start
        memory_metrics_seconds += metrics_elapsed
        diagnostic_seconds += metrics_elapsed
        kv_memory_snapshots.append(memory_snapshot)
        used_blocks = kv_cache.num_blocks - len(kv_cache.free_blocks)
        peak_used_blocks = max(
            peak_used_blocks, memory_snapshot['used_blocks']
        )
        iterations.append({
            'index': len(iterations),
            'latency_ms': (iteration_end - iteration_start) * 1000,
            'num_prefill_requests': len(step.scheduler_output.prefill_requests),
            'num_decode_requests': len(step.scheduler_output.decode_requests),
            'num_pending_arrivals': len(pending_requests),
            'num_waiting_requests': len(scheduler.waiting),
            'num_running_requests': len(scheduler.running),
            'used_kv_blocks_after_update': used_blocks,
            'kv_memory_before_request_release': memory_snapshot,
        })
        if capture_top2:
            diagnostic_start = time.perf_counter()
            for request_index, request_id in enumerate(step.model_batch.request_ids):
                end = int(step.model_batch.query_start_loc[request_index + 1].item())
                batched_top2[request_id].append(
                    top2_snapshot(step.packed_logits[end - 1])
                )
            top2_elapsed = time.perf_counter() - diagnostic_start
            top2_seconds += top2_elapsed
            diagnostic_seconds += top2_elapsed

    actual_tokens = [request.output_token_ids for request in requests]
    dense_greedy_match = actual_tokens == dense_expected_tokens
    isolated_paged_greedy_match = actual_tokens == isolated_expected_tokens
    dense_isolated_greedy_match = (
        dense_expected_tokens == isolated_expected_tokens
    )
    greedy_match = (
        dense_greedy_match
        and isolated_paged_greedy_match
        and dense_isolated_greedy_match
    )
    runtime_match = isolated_paged_greedy_match
    all_kv_blocks_released = len(kv_cache.free_blocks) == kv_cache.num_blocks
    request_metrics = []
    for request in requests:
        request_id = request.request_id
        timestamps = token_timestamps[request_id]
        arrival = request.arrival_time
        enqueue = scheduler_enqueue_times[request_id]
        first_scheduled = first_scheduled_times[request_id]
        request_itls_ms = [
            (right - left) * 1000
            for left, right in zip(timestamps, timestamps[1:])
        ]
        request_metrics.append({
            'request_id': request_id,
            'prompt_len': request.prompt_len,
            'output_len': request.max_new_tokens,
            'arrival_ms': arrival * 1000,
            'scheduler_enqueue_ms': enqueue * 1000,
            'arrival_delivery_delay_ms': (enqueue - arrival) * 1000,
            'first_scheduled_ms': first_scheduled * 1000,
            'queue_ms': (first_scheduled - arrival) * 1000,
            'scheduler_queue_ms': (first_scheduled - enqueue) * 1000,
            'prefill_to_first_token_ms': (timestamps[0] - first_scheduled) * 1000,
            'ttft_ms': (timestamps[0] - arrival) * 1000,
            'e2e_ms': (timestamps[-1] - arrival) * 1000,
            'itl_ms': request_itls_ms,
            'itl_summary_ms': latency_summary(request_itls_ms),
        })

    ttfts_ms = [metrics['ttft_ms'] for metrics in request_metrics]
    queue_times_ms = [metrics['queue_ms'] for metrics in request_metrics]
    e2e_ms = [metrics['e2e_ms'] for metrics in request_metrics]
    itls_ms = [
        (right - left) * 1000
        for timestamps in token_timestamps.values()
        for left, right in zip(timestamps, timestamps[1:])
    ]
    total_seconds = max(timestamps[-1] for timestamps in token_timestamps.values())
    total_output_tokens = sum(output_lens)
    ttft_summary = latency_summary(ttfts_ms)
    queue_summary = latency_summary(queue_times_ms)
    itl_summary = latency_summary(itls_ms)
    peak_paged_snapshot = max(
        kv_memory_snapshots, key=lambda snapshot: snapshot['paged_committed_mib']
    )
    peak_dense_snapshot = max(
        kv_memory_snapshots, key=lambda snapshot: snapshot['dense_padded_mib']
    )
    kv_pool_bytes = (
        kv_cache.k_cache.numel() * kv_cache.k_cache.element_size()
        + kv_cache.v_cache.numel() * kv_cache.v_cache.element_size()
    )
    bytes_per_token = kv_pool_bytes / (num_blocks * block_size)
    max_request_kv_tokens = max(
        len(tokens) + max(output_len - 1, 0)
        for tokens, output_len in zip(prompt_token_ids, output_lens)
    )
    dense_static_slots = min(max_num_seqs, num_requests) * max_request_kv_tokens
    dense_static_mib = dense_static_slots * bytes_per_token / 2**20
    kv_pool_allocated_mib = kv_pool_bytes / 2**20
    result = {
        'ttft_mean_ms': ttft_summary['mean'],
        'ttft_p50_ms': ttft_summary['p50'],
        'ttft_p95_ms': ttft_summary['p95'],
        'ttft_p99_ms': ttft_summary['p99'],
        'ttft_max_ms': ttft_summary['max'],
        'queue_mean_ms': queue_summary['mean'],
        'queue_p50_ms': queue_summary['p50'],
        'queue_p95_ms': queue_summary['p95'],
        'queue_p99_ms': queue_summary['p99'],
        'tpot_mean_ms': itl_summary['mean'],
        'tpot_p95_ms': itl_summary['p95'],
        'itl_p50_ms': itl_summary['p50'],
        'itl_p95_ms': itl_summary['p95'],
        'itl_p99_ms': itl_summary['p99'],
        'e2e_max_ms': max(e2e_ms),
        'output_tokens_per_second': total_output_tokens / total_seconds,
        'request_throughput_per_second': num_requests / total_seconds,
        'peak_used_kv_blocks': peak_used_blocks,
        'total_kv_blocks': num_blocks,
        'peak_kv_utilization': peak_used_blocks / num_blocks,
        'kv_bytes_per_token': int(bytes_per_token),
        'kv_pool_allocated_mib': kv_pool_allocated_mib,
        'dense_static_baseline_mib': dense_static_mib,
        'kv_pool_saving_vs_dense_static': (
            1 - kv_pool_allocated_mib / dense_static_mib
            if dense_static_mib else 0.0
        ),
        'peak_paged_committed_mib': peak_paged_snapshot['paged_committed_mib'],
        'peak_useful_kv_mib': peak_paged_snapshot['useful_kv_mib'],
        'peak_internal_fragmentation_mib': (
            peak_paged_snapshot['internal_fragmentation_mib']
        ),
        'peak_block_packing_efficiency': (
            peak_paged_snapshot['block_packing_efficiency']
        ),
        'dense_padded_at_peak_paged_mib': (
            peak_paged_snapshot['dense_padded_mib']
        ),
        'peak_saving_vs_dense_padded': (
            peak_paged_snapshot['saving_vs_dense_padded']
        ),
        'max_dense_padded_mib': peak_dense_snapshot['dense_padded_mib'],
        'paged_at_max_dense_padded_mib': (
            peak_dense_snapshot['paged_committed_mib']
        ),
        'saving_at_max_dense_padded': (
            peak_dense_snapshot['saving_vs_dense_padded']
        ),
        'memory_metric_collection_overhead_ms': memory_metrics_seconds * 1000,
        'num_iterations': len(iterations),
        'greedy_match': greedy_match,
        'runtime_match': runtime_match,
        'all_kv_blocks_released': all_kv_blocks_released,
        'dense_greedy_match': dense_greedy_match,
        'isolated_paged_greedy_match': isolated_paged_greedy_match,
        'dense_isolated_greedy_match': dense_isolated_greedy_match,
        'actual_token_ids': actual_tokens,
        'dense_expected_token_ids': dense_expected_tokens,
        'isolated_expected_token_ids': isolated_expected_tokens,
        'iterations': iterations,
        'request_metrics': request_metrics,
    }
    if capture_top2:
        batched_top2_list = [
            batched_top2[request.request_id] for request in requests
        ]
        result.update({
            'top2_capture_overhead_ms': top2_seconds * 1000,
            'dense_top2': dense_top2,
            'isolated_paged_top2': isolated_top2,
            'batched_scheduler_top2': batched_top2_list,
            'first_divergences': build_first_divergences(
                dense_expected_tokens,
                isolated_expected_tokens,
                actual_tokens,
                dense_top2,
                isolated_top2,
                batched_top2_list,
            ),
        })
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    device = torch.device('cuda', torch.cuda.current_device()) if args.device == 'cuda' else torch.device('cpu')
    dtype = getattr(torch, args.dtype)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not (
        args.disable_bf16_reduced_precision_reduction
    )
    config = TinyQwen2Config.from_dict(
        json.loads((args.model_dir / 'config.json').read_text())
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = TinyQwen2ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    load_qwen2_safetensors(
        model,
        args.model_dir / 'model.safetensors',
        tie_word_embeddings=config.tie_word_embeddings,
        device='cpu',
    )
    model.eval()

    tokenizer = None
    prompt_texts = None
    if args.prompt_text_file is not None:
        if args.prompt_lens is not None:
            raise ValueError('--prompt-lens cannot be used with --prompt-text-file')
        from transformers import AutoTokenizer

        prompt_texts = json.loads(args.prompt_text_file.read_text())
        if (
            not isinstance(prompt_texts, list)
            or not prompt_texts
            or not all(isinstance(text, str) and text for text in prompt_texts)
        ):
            raise ValueError('prompt text file must contain a non-empty JSON string list')
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir,
            local_files_only=True,
        )
        prompt_token_ids = [tokenizer.encode(text) for text in prompt_texts]
    else:
        generator = torch.Generator(device='cpu').manual_seed(args.seed)
        prompt_lens = (
            args.prompt_lens
            if args.prompt_lens is not None
            else [args.prompt_len] * args.num_requests
        )
        prompt_token_ids = [
            torch.randint(
                10,
                min(1000, config.vocab_size),
                (prompt_len,),
                generator=generator,
            ).tolist()
            for prompt_len in prompt_lens
        ]
    num_requests = len(prompt_token_ids)
    output_lens = expand_request_values(
        args.output_lens if args.output_lens is not None else [args.output_len],
        num_requests,
        'output_lens',
    )
    if args.arrival_times_ms is not None:
        arrival_times_ms = expand_request_values(
            args.arrival_times_ms,
            num_requests,
            'arrival_times_ms',
        )
    else:
        arrival_times_ms = [
            index * args.arrival_interval_ms for index in range(num_requests)
        ]
    max_num_seqs = (
        num_requests if args.max_num_seqs is None else args.max_num_seqs
    )
    max_num_batched_tokens = (
        sum(map(len, prompt_token_ids))
        if args.max_num_batched_tokens is None
        else args.max_num_batched_tokens
    )
    if args.capture_top2:
        dense_trace = dense_greedy_trace(
            model, prompt_token_ids, output_lens
        )
        isolated_trace = isolated_paged_greedy_trace(
            model,
            config,
            prompt_token_ids,
            output_lens,
            args.block_size,
        )
        expected_tokens = dense_trace['token_ids']
        isolated_expected_tokens = isolated_trace['token_ids']
        dense_top2 = dense_trace['top2']
        isolated_top2 = isolated_trace['top2']
    else:
        expected_tokens = dense_greedy_tokens(
            model, prompt_token_ids, output_lens
        )
        isolated_expected_tokens = isolated_paged_greedy_tokens(
            model,
            config,
            prompt_token_ids,
            output_lens,
            args.block_size,
        )
        dense_top2 = None
        isolated_top2 = None

    sources = [Path(__file__), Path('src/mini_llm_runtime/runtime/model_runner.py'),
               Path('src/mini_llm_runtime/runtime/scheduler.py'),
               Path('src/mini_llm_runtime/runtime/paged_kv_cache.py'),
               Path('src/mini_llm_runtime/model/qwen2.py')]
    metadata = {
        'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'platform': platform.platform(),
        'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        'allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'timed_region': (
            'workload arrival time zero -> Scheduler/model/sample/cache lifecycle '
            'through completion, including configured idle and queueing time'
        ),
        'excluded': 'model loading and dense greedy correctness reference',
        'memory_metrics_note': (
            'KV tensor bytes are exact tensor storage sizes. Dense baselines '
            'model a rectangular cache padded to the longest live/request '
            'length; metric collection overhead is subtracted from latency.'
        ),
        'top2_capture_note': (
            'top-2 extraction overhead is subtracted from lifecycle latency'
            if args.capture_top2 else None
        ),
        'prompt_token_ids': prompt_token_ids,
        'prompt_texts': prompt_texts,
        'prompt_lengths': [len(tokens) for tokens in prompt_token_ids],
        'output_lengths': output_lens,
        'arrival_times_ms': arrival_times_ms,
        'max_num_seqs': max_num_seqs,
        'max_num_batched_tokens': max_num_batched_tokens,
        'dense_expected_token_ids': expected_tokens,
        'isolated_expected_token_ids': isolated_expected_tokens,
        'dense_isolated_greedy_match': expected_tokens == isolated_expected_tokens,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')

    for _ in range(args.warmup):
        run_workload(
            model,
            config,
            prompt_token_ids,
            output_lens,
            args.block_size,
            expected_tokens,
            isolated_expected_tokens,
            dense_top2,
            isolated_top2,
            args.capture_top2,
            arrival_times_ms,
            max_num_seqs,
            max_num_batched_tokens,
            args.kv_cache_blocks,
        )

    rows = []
    detailed_samples = []
    for sample_index in range(args.samples):
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline_allocated = torch.cuda.memory_allocated()
        result = run_workload(
            model,
            config,
            prompt_token_ids,
            output_lens,
            args.block_size,
            expected_tokens,
            isolated_expected_tokens,
            dense_top2,
            isolated_top2,
            args.capture_top2,
            arrival_times_ms,
            max_num_seqs,
            max_num_batched_tokens,
            args.kv_cache_blocks,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
            peak_mib = (
                torch.cuda.max_memory_allocated() - baseline_allocated
            ) / 2**20
        else:
            peak_mib = 0.0
        result['incremental_peak_allocated_mib'] = peak_mib
        if tokenizer is not None:
            result['actual_texts'] = tokenizer.batch_decode(
                result['actual_token_ids'], skip_special_tokens=True
            )
            result['dense_expected_texts'] = tokenizer.batch_decode(
                result['dense_expected_token_ids'], skip_special_tokens=True
            )
            result['isolated_expected_texts'] = tokenizer.batch_decode(
                result['isolated_expected_token_ids'], skip_special_tokens=True
            )
        detailed_samples.append(result)
        row = {
            key: value
            for key, value in result.items()
            if key not in {
                'actual_token_ids',
                'dense_expected_token_ids',
                'isolated_expected_token_ids',
                'iterations',
                'request_metrics',
                'dense_top2',
                'isolated_paged_top2',
                'batched_scheduler_top2',
                'first_divergences',
                'actual_texts',
                'dense_expected_texts',
                'isolated_expected_texts',
            }
        }
        row['sample'] = sample_index
        rows.append(row)
        with args.output.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        args.output.with_suffix('.samples.json').write_text(
            json.dumps(detailed_samples, indent=2) + '\n'
        )
        print(
            f"sample={sample_index} TTFT-p50/p95="
            f"{row['ttft_p50_ms']:.2f}/{row['ttft_p95_ms']:.2f} ms "
            f"queue-p95={row['queue_p95_ms']:.2f} ms "
            f"ITL-p95={row['itl_p95_ms']:.2f} ms "
            f"E2E={row['e2e_max_ms']:.2f} ms "
            f"throughput={row['output_tokens_per_second']:.1f} tok/s "
            f"greedy_match={row['greedy_match']}",
            flush=True,
        )
        if args.strict_greedy_validation and not result['greedy_match']:
            raise AssertionError(
                'Dense, isolated paged, and batched Scheduler token IDs differ'
            )
        if args.strict_runtime_validation and (
            not result['runtime_match']
            or not result['all_kv_blocks_released']
        ):
            raise AssertionError(
                'Batched Scheduler differs from Isolated Paged or leaked KV blocks'
            )

    summary = summarize_samples(rows)
    metadata['summary'] = summary
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
