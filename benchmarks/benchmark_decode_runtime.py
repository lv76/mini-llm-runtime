"""Benchmark sequential and batched fixed-context full-model decode.

Excludes checkpoint loading, prefill, scheduling, and sampling. Uses real prefill
KV but repeatedly overwrites the same current-token slots (no context growth).
The sequential baseline is this repository's Python reference, not HF/vLLM.
"""
import argparse
import copy
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
from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    parser.add_argument('--context-lens', type=int, nargs='+', default=[128])
    parser.add_argument('--ragged', action='store_true', help='Lengths range from half to full context')
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--strict-validation', action='store_true',
                        help='Abort on logit tolerance failures, including FP16/BF16. '
                             'FP32 and non-finite values always fail fast.')
    parser.add_argument(
        '--input-validation',
        type=Path,
        help='Replay prompt/decode token IDs from a saved validation JSON.',
    )
    parser.add_argument(
        '--disable-bf16-reduced-precision-reduction',
        action='store_true',
        help='Disable reduced-precision BF16 GEMM reductions for accuracy/performance A/B tests.',
    )
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/decode_runtime.csv'))
    args = parser.parse_args()
    if min(args.batch_sizes + args.context_lens + [args.block_size, args.samples, args.threads]) <= 0:
        parser.error('sizes, samples and threads must be positive')
    if min(args.context_lens) < 2 or args.warmup < 0:
        parser.error('context lengths must be >= 2 and warmup >= 0')
    return args


def percentile_95(values: list[float]) -> float:
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def compare_logits(actual, reference, *, atol, rtol):
    """Keep per-request failures visible; aggregate averages cannot hide outliers."""
    if actual.shape != reference.shape or actual.ndim != 2 or actual.shape[1] < 2:
        raise ValueError('logits must have matching [B, vocab_size >= 2] shapes')
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise AssertionError('decode logits contain non-finite values')
    actual, reference = actual.float(), reference.float()
    error = actual - reference
    abs_error = error.abs()
    violations = abs_error > (atol + rtol * reference.abs())
    rmse = error.square().mean(-1).sqrt()
    ref_rms = reference.square().mean(-1).sqrt().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(actual, reference, dim=-1)
    top2 = reference.topk(2, dim=-1)
    per_request = []
    for i in range(actual.shape[0]):
        per_request.append({
            'request_index': i,
            'violating_elements': int(violations[i].sum()),
            'max_abs_error': float(abs_error[i].max()),
            'mean_abs_error': float(abs_error[i].mean()),
            'normalized_rmse': float(rmse[i] / ref_rms[i]),
            'cosine_similarity': float(cosine[i]),
            'actual_top1': int(actual[i].argmax()),
            'reference_top1': int(reference[i].argmax()),
            'reference_top2_margin': float(top2.values[i, 0] - top2.values[i, 1]),
        })
    return {
        'allclose': not bool(violations.any()),
        'violating_elements': int(violations.sum()),
        'total_elements': actual.numel(),
        'atol': atol, 'rtol': rtol,
        'per_request': per_request,
    }


def validation_status(comparisons, greedy_agreements):
    if not all(report['allclose'] for report in comparisons.values()):
        return 'failed_logit_tolerance'
    if any(agreement != 1.0 for agreement in greedy_agreements):
        return 'logits_pass_greedy_mismatch'
    return 'logits_and_greedy_pass'


def select_replay_case(cases, batch_size, context_lens):
    matches = [
        case
        for case in cases
        if case.get('batch_size') == batch_size
        and case.get('context_lens') == context_lens
    ]
    if len(matches) != 1:
        raise ValueError(
            'input validation must contain exactly one case for '
            f'B={batch_size}, context_lens={context_lens}; found {len(matches)}'
        )
    case = matches[0]
    if len(case.get('prompt_token_ids', [])) != batch_size:
        raise ValueError('saved prompt_token_ids count must match batch_size')
    if len(case.get('decode_token_ids', [])) != batch_size:
        raise ValueError('saved decode_token_ids count must match batch_size')
    for tokens, context_len in zip(case['prompt_token_ids'], context_lens):
        if len(tokens) + 1 != context_len:
            raise ValueError('saved prompt length must equal context_len - 1')
    return case


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not (
        args.disable_bf16_reduced_precision_reduction
    )
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if args.device == "cuda" and torch.cuda.is_available()
        else torch.device(args.device)
    )
    dtype = getattr(torch, args.dtype)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    replay_cases = None
    replay_sha256 = None
    if args.input_validation:
        replay_bytes = args.input_validation.read_bytes()
        replay_cases = json.loads(replay_bytes)
        if not isinstance(replay_cases, list):
            raise ValueError('input validation must contain a JSON list')
        replay_sha256 = hashlib.sha256(replay_bytes).hexdigest()
    config = (TinyQwen2Config.from_dict(json.loads((args.model_dir / 'config.json').read_text()))
              if args.model_dir else TinyQwen2Config())
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = TinyQwen2ForCausalLM(config)
    finally:
        torch.set_default_dtype(old_dtype)
    if args.model_dir:
        load_qwen2_safetensors(model, args.model_dir / 'model.safetensors',
                              tie_word_embeddings=config.tie_word_embeddings)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path('src/mini_llm_runtime/model/qwen2.py'),
               Path('src/mini_llm_runtime/runtime/model_runner.py'),
               Path('src/mini_llm_runtime/runtime/paged_kv_cache.py'),
               Path('src/mini_llm_runtime/kernels/reference/paged_attention.py'), Path(__file__)]
    metadata = {
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'platform': platform.platform(),
        'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'config': vars(config), 'seed': 42,
        'timing': 'synchronized wall-clock; fixed decode step; excludes prefill, scheduling, and sampling',
        'baseline': 'local single-request Python paged reference; not HF or vLLM',
        'model': str(args.model_dir) if args.model_dir else 'random tiny model',
        'validation_policy': 'FP32/strict: fail on tolerance; FP16/BF16: report '
                             'failed tolerance and continue timing; non-finite: always abort',
        'allow_bf16_reduced_precision_reduction': (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        'input_validation_sha256': replay_sha256,
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    rows = []
    validation_reports = []
    for context_len in args.context_lens:
        for batch_size in args.batch_sizes:
            lengths = ([max(2, context_len // 2 + (context_len - context_len // 2) * (i + 1) // batch_size)
                        for i in range(batch_size)] if args.ragged else [context_len] * batch_size)
            replay_case = (
                select_replay_case(replay_cases, batch_size, lengths)
                if replay_cases is not None
                else None
            )
            blocks = sum(math.ceil(s / args.block_size) for s in lengths)
            cache = PagedKVCache(config.num_hidden_layers, blocks, args.block_size,
                                 config.num_key_value_heads, config.head_dim, dtype, device)
            cache.k_cache.zero_()
            cache.v_cache.zero_()
            pending = []
            prompt_token_ids = []
            for i, length in enumerate(lengths):
                cache.add_sequence(i)
                assert cache.reserve_tokens(i, length)
                table = cache.block_tables[i]
                ids = (
                    torch.tensor(
                        [replay_case['prompt_token_ids'][i]],
                        device=device,
                    )
                    if replay_case is not None
                    else torch.randint(
                        10,
                        min(1000, config.vocab_size),
                        (1, length - 1),
                        device=device,
                    )
                )
                prompt_token_ids.append(ids.clone())
                positions = torch.arange(length - 1, device=device)
                slots = torch.tensor([table[p // args.block_size] * args.block_size + p % args.block_size
                                      for p in range(length - 1)], device=device)
                logits = model.forward_prefill_with_cache(ids, positions[None, :], cache, slots)
                pending.append(
                    int(replay_case['decode_token_ids'][i])
                    if replay_case is not None
                    else int(logits[0, -1].argmax())
                )
            reference_cache = copy.deepcopy(cache)
            tables = [cache.block_tables[i] for i in range(batch_size)]
            width = max(map(len, tables))
            table_tensor = torch.tensor([t + [-1] * (width - len(t)) for t in tables], device=device)
            contexts = torch.tensor(lengths, device=device)
            positions = (contexts - 1)[:, None]
            ids = torch.tensor(pending, device=device)[:, None]
            slots = torch.tensor([tables[i][(s - 1) // args.block_size] * args.block_size
                                  + (s - 1) % args.block_size for i, s in enumerate(lengths)], device=device)
            def sequential():
                return torch.cat([model.forward_decode_with_cache(
                    ids[i:i+1], positions[i:i+1], reference_cache, slots[i:i+1], tables[i], lengths[i])
                    for i in range(batch_size)])[:, 0]

            def batched():
                return model.forward_decode_batch_with_cache(ids, positions, cache, slots, table_tensor, contexts)[:, 0]

            expected, actual = sequential(), batched()
            dense = torch.cat([
                model(torch.cat([prompt_token_ids[i], ids[i:i + 1]], dim=1))[:, -1]
                for i in range(batch_size)
            ])
            max_error = (expected.float() - actual.float()).abs().max().item()
            agreement = (expected.argmax(-1) == actual.argmax(-1)).float().mean().item()
            sequential_dense_agreement = (
                expected.argmax(-1) == dense.argmax(-1)
            ).float().mean().item()
            batched_dense_agreement = (
                actual.argmax(-1) == dense.argmax(-1)
            ).float().mean().item()
            dense_top2 = dense.float().topk(2, dim=-1)
            min_dense_top2_margin = (
                dense_top2.values[:, 0] - dense_top2.values[:, 1]
            ).min().item()
            if not all(torch.isfinite(value).all() for value in (expected, actual, dense)):
                raise AssertionError("decode logits contain non-finite values")
            atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.5, 0.1)
            comparisons = {
                'batched_vs_sequential': compare_logits(actual, expected, atol=atol, rtol=rtol),
                'sequential_vs_dense': compare_logits(expected, dense, atol=atol, rtol=rtol),
                'batched_vs_dense': compare_logits(actual, dense, atol=atol, rtol=rtol),
            }
            status = validation_status(comparisons, [
                agreement, sequential_dense_agreement, batched_dense_agreement,
            ])
            validation_reports.append({
                'batch_size': batch_size, 'context_lens': lengths,
                'validation_status': status,
                'prompt_token_ids': [prompt[0].tolist() for prompt in prompt_token_ids],
                'decode_token_ids': pending,
                'comparisons': comparisons,
            })
            args.output.with_suffix('.validation.json').write_text(
                json.dumps(validation_reports, indent=2) + '\n'
            )
            if status == 'failed_logit_tolerance':
                failures = {name: report['violating_elements']
                            for name, report in comparisons.items() if not report['allclose']}
                message = f'B={batch_size} S={lengths}: failed_logit_tolerance {failures}'
                if dtype == torch.float32 or args.strict_validation:
                    raise AssertionError(message)
                print(f'WARNING: {message}; timings are diagnostic, not correctness-qualified', flush=True)
            if any(value != 1.0 for value in (
                agreement, sequential_dense_agreement, batched_dense_agreement,
            )):
                print(
                    f"WARNING: {args.dtype} greedy token disagreement: "
                    f"agreement={agreement:.3f}, "
                    f"sequential={expected.argmax(-1).tolist()}, "
                    f"batched={actual.argmax(-1).tolist()}, "
                    f"dense={dense.argmax(-1).tolist()}, "
                    f"dense_top2={dense_top2.indices.tolist()}, "
                    f"dense_top2_margin="
                    f"{(dense_top2.values[:, 0] - dense_top2.values[:, 1]).tolist()}, "
                    f"max_logit_abs_error={max_error}",
                    flush=True,
                )
            max_cache_error = max(
                (cache.k_cache.float() - reference_cache.k_cache.float()).abs().max().item(),
                (cache.v_cache.float() - reference_cache.v_cache.float()).abs().max().item(),
            )
            if not torch.isfinite(cache.k_cache).all() or not torch.isfinite(cache.v_cache).all():
                raise AssertionError("batched KV cache contains non-finite values")
            functions = {'sequential': sequential, 'batched': batched}
            for _ in range(args.warmup):
                for fn in functions.values():
                    fn()
            timings = {name: [] for name in functions}
            # Rotate order to reduce warmup/thermal bias between implementations.
            names = list(functions)
            for sample in range(args.samples):
                offset = sample % len(names)
                for name in names[offset:] + names[:offset]:
                    sync()
                    start = time.perf_counter()
                    functions[name]()
                    sync()
                    timings[name].append((time.perf_counter() - start) * 1000)
            for name, values in timings.items():
                sync()
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    functions[name]()
                    sync()
                    extra_mib = (torch.cuda.max_memory_allocated() - allocated) / 2**20
                    extra_reserved_mib = (
                        torch.cuda.max_memory_reserved() - reserved
                    ) / 2**20
                else:
                    extra_mib = None
                    extra_reserved_mib = None
                median = statistics.median(values)
                row = dict(batch_size=batch_size, context_min=min(lengths), context_max=max(lengths),
                           mode=name, median_ms=median, p95_ms=percentile_95(values),
                           tokens_per_second=batch_size * 1000 / median,
                           speedup_vs_sequential=statistics.median(timings['sequential']) / median,
                           incremental_peak_allocated_mib=extra_mib,
                           incremental_peak_reserved_mib=extra_reserved_mib,
                           kv_cache_mib=(cache.k_cache.numel() + cache.v_cache.numel())
                           * cache.k_cache.element_size() / 2**20,
                           max_logit_abs_error=max_error,
                           max_kv_cache_abs_error=max_cache_error,
                           sequential_batched_greedy_agreement=agreement,
                           sequential_dense_greedy_agreement=sequential_dense_agreement,
                           batched_dense_greedy_agreement=batched_dense_agreement,
                           min_dense_top2_margin=min_dense_top2_margin,
                           validation_status=status,
                           comparison_metrics=json.dumps(comparisons),
                           correctness_atol=atol, correctness_rtol=rtol,
                           samples_ms=json.dumps(values))
                rows.append(row)
                print(f'B={batch_size} S={min(lengths)}..{max(lengths)} {name}: {median:.3f} ms, '
                      f'{row["tokens_per_second"]:.1f} tok/s, {row["speedup_vs_sequential"]:.2f}x '
                      f'[{status}]', flush=True)
            with args.output.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            del reference_cache, cache
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
