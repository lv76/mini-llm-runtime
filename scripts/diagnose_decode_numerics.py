"""Replay saved benchmark inputs with fresh FP32/BF16 models and layer traces.

This is a correctness diagnostic: CPU copies from hooks make it unsuitable for
timing. No model kernels or benchmark acceptance thresholds are changed.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

import mini_llm_runtime.model.qwen2 as qwen2
from mini_llm_runtime.model.qwen2_loader import load_qwen2_safetensors
from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache


def metrics(actual, reference, *, atol=1e-4, rtol=1e-4):
    actual, reference = actual.float(), reference.float()
    if actual.shape != reference.shape:
        raise ValueError(f'trace shape mismatch: {actual.shape} vs {reference.shape}')
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError('non-finite trace')
    difference = actual - reference
    rms = reference.square().mean().sqrt().clamp_min(1e-12)
    return dict(
        max_abs=float(difference.abs().max()),
        mean_abs=float(difference.abs().mean()),
        nrmse=float(difference.square().mean().sqrt() / rms),
        different_elements=int((difference != 0).sum()),
        violating_elements=int(
            (difference.abs() > (atol + rtol * reference.abs())).sum()
        ),
        atol=atol,
        rtol=rtol,
    )


class Trace:
    def __init__(self, model, request_index):
        self.model = model
        self.request_index = request_index
        self.enabled = False
        self.batched = False
        self.active_layer = None
        self.values = {}
        self.handles = []
        self.patches = []

    def record(self, name, value):
        if self.enabled:
            row = self.request_index if self.batched else 0
            self.values[name] = value[row, -1].detach().float().cpu().clone()

    def __enter__(self):
        def output_hook(name):
            return lambda module, args, result: self.record(name, result)

        def input_hook(name):
            return lambda module, args: self.record(name, args[0])

        for index, layer in enumerate(self.model.model.layers):
            prefix = f'layer.{index:02d}'
            for suffix, module in [
                ('input_norm', layer.input_layernorm),
                ('q_proj', layer.self_attn.qkv_proj.q_proj),
                ('k_proj', layer.self_attn.qkv_proj.k_proj),
                ('v_proj', layer.self_attn.qkv_proj.v_proj),
                ('o_proj', layer.self_attn.o_proj),
                ('post_norm', layer.post_attention_layernorm),
                ('mlp.gate_proj', layer.mlp.gate_proj),
                ('mlp.up_proj', layer.mlp.up_proj),
                ('mlp.down_proj', layer.mlp.down_proj),
                ('mlp', layer.mlp),
            ]:
                self.handles.append(module.register_forward_hook(output_hook(f'{prefix}.{suffix}')))
            self.handles.append(layer.input_layernorm.register_forward_pre_hook(input_hook(f'{prefix}.input')))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(input_hook(f'{prefix}.attention')))
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(input_hook(f'{prefix}.attention_residual')))
            # Cached paths call named methods directly, bypassing Module hooks.
            for method in ('forward', 'forward_decode_with_cache', 'forward_decode_batch_with_cache'):
                original = getattr(layer, method)

                def wrapped(*args, _original=original, _prefix=prefix, **kwargs):
                    self.active_layer = _prefix
                    result = _original(*args, **kwargs)
                    self.record(f'{_prefix}.output', result)
                    return result

                replacement = patch.object(layer, method, wrapped)
                replacement.start()
                self.patches.append(replacement)
        original_rope = qwen2.apply_rotary_pos_emb

        def traced_rope(*args, **kwargs):
            query, key = original_rope(*args, **kwargs)
            self.record(f'{self.active_layer}.q_rope', query)
            self.record(f'{self.active_layer}.k_rope', key)
            return query, key

        replacement = patch.object(qwen2, 'apply_rotary_pos_emb', traced_rope)
        replacement.start()
        self.patches.append(replacement)
        self.handles.append(self.model.model.norm.register_forward_hook(output_hook('final_norm')))
        self.handles.append(self.model.lm_head.register_forward_hook(output_hook('logits')))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        for replacement in reversed(self.patches):
            replacement.stop()


def controlled_shape_experiment(
    model,
    traces,
    dtype,
    batch_size,
    context_len,
    request_index,
):
    """Hold one token input fixed while changing only the GEMM outer shape."""
    layer = model.model.layers[0]
    device = next(model.parameters()).device
    atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.5, 0.1)
    cases = [
        (
            'q_proj',
            layer.self_attn.qkv_proj.q_proj,
            'layer.00.input_norm',
            'layer.00.q_proj',
            True,
        ),
        (
            'o_proj',
            layer.self_attn.o_proj,
            'layer.00.attention',
            'layer.00.o_proj',
            False,
        ),
        (
            'mlp.up_proj',
            layer.mlp.up_proj,
            'layer.00.post_norm',
            'layer.00.mlp.up_proj',
            False,
        ),
        (
            'mlp.gate_proj',
            layer.mlp.gate_proj,
            'layer.00.post_norm',
            'layer.00.mlp.gate_proj',
            False,
        ),
        (
            'mlp',
            layer.mlp,
            'layer.00.post_norm',
            'layer.00.mlp',
            False,
        ),
    ]
    report = {}
    for name, module, input_stage, output_stage, check_dense in cases:
        token_input = traces['sequential'][input_stage].to(
            device=device,
            dtype=dtype,
        ).reshape(1, 1, -1)
        single = module(token_input)[0, -1].detach().float().cpu()
        batch_input = token_input.expand(batch_size, 1, -1).contiguous()
        batched = module(batch_input)[request_index, -1].detach().float().cpu()
        row = {
            'controlled_batch_vs_single': metrics(
                batched, single, atol=atol, rtol=rtol
            ),
            'controlled_single_vs_actual_sequential': metrics(
                single,
                traces['sequential'][output_stage],
                atol=atol,
                rtol=rtol,
            ),
            'controlled_batch_vs_actual_batched': metrics(
                batched,
                traces['batched'][output_stage],
                atol=atol,
                rtol=rtol,
            ),
        }
        if check_dense:
            dense_input = token_input.expand(1, context_len, -1).contiguous()
            dense = module(dense_input)[0, -1].detach().float().cpu()
            row['controlled_dense_vs_single'] = metrics(
                dense, single, atol=atol, rtol=rtol
            )
            row['controlled_dense_vs_actual_dense'] = metrics(
                dense,
                traces['dense'][output_stage],
                atol=atol,
                rtol=rtol,
            )
        report[name] = row
    return report


@torch.inference_mode()
def replay(case, model_dir, dtype, block_size, request_index, device):
    # Reload checkpoint into a fresh model and prefill in this precision.
    config = qwen2.TinyQwen2Config.from_dict(json.loads((model_dir / 'config.json').read_text()))
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = qwen2.TinyQwen2ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    load_qwen2_safetensors(model, model_dir / 'model.safetensors', tie_word_embeddings=config.tie_word_embeddings)
    model.eval()
    lengths = case['context_lens']
    count = len(lengths)
    cache = PagedKVCache(config.num_hidden_layers,
                        sum((length + block_size - 1) // block_size for length in lengths),
                        block_size, config.num_key_value_heads, config.head_dim, dtype, device)
    cache.k_cache.zero_()
    cache.v_cache.zero_()
    prompts = [torch.tensor([tokens], device=device) for tokens in case['prompt_token_ids']]
    for i, prompt in enumerate(prompts):
        if prompt.shape[1] + 1 != lengths[i]:
            raise ValueError('saved prompt length does not match context length')
        cache.add_sequence(i)
        assert cache.reserve_tokens(i, lengths[i])
        table = cache.block_tables[i]
        positions = torch.arange(prompt.shape[1], device=device)[None, :]
        slots = torch.tensor([table[p // block_size] * block_size + p % block_size
                              for p in range(prompt.shape[1])], device=device)
        model.forward_prefill_with_cache(prompt, positions, cache, slots)
    reference_cache = copy.deepcopy(cache)
    tables = [cache.block_tables[i] for i in range(count)]
    width = max(map(len, tables))
    block_tables = torch.tensor([t + [-1] * (width - len(t)) for t in tables], device=device)
    contexts = torch.tensor(lengths, device=device)
    positions = (contexts - 1)[:, None]
    ids = torch.tensor(case['decode_token_ids'], device=device)[:, None]
    slots = torch.tensor([tables[i][(s - 1) // block_size] * block_size + (s - 1) % block_size
                          for i, s in enumerate(lengths)], device=device)
    traces, logits = {}, {}
    with Trace(model, request_index) as trace:
        for mode in ('sequential', 'batched', 'dense'):
            trace.values = {}
            trace.batched = mode == 'batched'
            if mode == 'batched':
                trace.enabled = True
                output = model.forward_decode_batch_with_cache(ids, positions, cache, slots, block_tables, contexts)[:, 0]
            else:
                parts = []
                for i in range(count):
                    trace.enabled = i == request_index
                    if mode == 'sequential':
                        result = model.forward_decode_with_cache(ids[i:i+1], positions[i:i+1], reference_cache,
                                                                 slots[i:i+1], tables[i], lengths[i])[:, 0]
                    else:
                        result = model(torch.cat([prompts[i], ids[i:i+1]], dim=1))[:, -1]
                    parts.append(result)
                output = torch.cat(parts)
            logits[mode] = output.float().cpu()
            traces[mode] = trace.values
        trace.enabled = False
    atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.5, 0.1)

    def metric_pair(left, right):
        return {
            'acceptance': metrics(left, right, atol=atol, rtol=rtol),
            'strict_diagnostic': metrics(left, right),
        }

    cache_metrics = {
        kind: dict(
            **metric_pair(left, right),
            per_layer=[
                metric_pair(left[layer_idx], right[layer_idx])
                for layer_idx in range(left.shape[0])
            ],
        )
        for kind, left, right in [
            ('k', cache.k_cache, reference_cache.k_cache),
            ('v', cache.v_cache, reference_cache.v_cache),
        ]
    }
    controlled_shapes = controlled_shape_experiment(
        model=model,
        traces=traces,
        dtype=dtype,
        batch_size=count,
        context_len=max(lengths),
        request_index=request_index,
    )
    return logits, traces, cache_metrics, controlled_shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=Path('checkpoints/Qwen2.5-0.5B'))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--context-len', type=int, default=32)
    parser.add_argument('--request-index', type=int, default=2)
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument(
        '--disable-bf16-reduced-precision-reduction',
        action='store_true',
        help='Disable reduced-precision BF16 GEMM reductions for an A/B replay',
    )
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/decode_numerics.json'))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for real-checkpoint diagnostics')
    cases = json.loads(args.validation.read_text())
    matches = [c for c in cases if c['batch_size'] == args.batch_size and max(c['context_lens']) == args.context_len]
    if len(matches) != 1 or not 0 <= args.request_index < args.batch_size:
        raise ValueError('select exactly one case and a valid request index')
    case = matches[0]
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not (
        args.disable_bf16_reduced_precision_reduction
    )
    device = torch.device('cuda', torch.cuda.current_device())
    result = {'validation_file': str(args.validation),
              'validation_sha256': hashlib.sha256(args.validation.read_bytes()).hexdigest(),
              'case': case, 'request_index': args.request_index,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'tf32': False,
              'allow_bf16_reduced_precision_reduction': (
                  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
              ),
              'precisions': {}}
    layer_rows = []
    fp32_logits = None
    pairs = [('batched', 'sequential'), ('sequential', 'dense'), ('batched', 'dense')]
    for dtype in (torch.float32, torch.bfloat16):
        name = str(dtype).split('.')[-1]
        atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.5, 0.1)
        logits, traces, cache_metrics, controlled_shapes = replay(
            case,
            args.model_dir,
            dtype,
            args.block_size,
            args.request_index,
            device,
        )
        report = {'comparisons': {}, 'strict_diagnostics': {},
                  'greedy': {key: value.argmax(-1).tolist() for key, value in logits.items()},
                  'cache_batched_vs_sequential': cache_metrics,
                  'controlled_shape_experiment': controlled_shapes}
        for left, right in pairs:
            key = f'{left}_vs_{right}'
            report['comparisons'][key] = metrics(
                logits[left], logits[right], atol=atol, rtol=rtol
            )
            report['comparisons'][key]['per_request'] = [
                metrics(a, b, atol=atol, rtol=rtol)
                for a, b in zip(logits[left], logits[right])
            ]
            report['strict_diagnostics'][key] = metrics(logits[left], logits[right])
            report['strict_diagnostics'][key]['per_request'] = [
                metrics(a, b) for a, b in zip(logits[left], logits[right])
            ]
            for stage in traces[left]:
                diagnostic = metrics(traces[left][stage], traces[right][stage])
                acceptance = metrics(
                    traces[left][stage], traces[right][stage], atol=atol, rtol=rtol
                )
                layer_rows.append(dict(
                    dtype=name,
                    comparison=key,
                    stage=stage,
                    **diagnostic,
                    acceptance_violating_elements=acceptance['violating_elements'],
                    acceptance_atol=atol,
                    acceptance_rtol=rtol,
                ))
            summary = {
                k: v for k, v in report['comparisons'][key].items()
                if k != 'per_request'
            }
            print(name, key, summary, flush=True)
        if fp32_logits is None:
            fp32_logits = logits
        else:
            report['vs_fp32_dense'] = {
                key: metrics(value, fp32_logits['dense'], atol=atol, rtol=rtol)
                for key, value in logits.items()
            }
        result['precisions'][name] = report
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        with args.output.with_suffix('.layers.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(layer_rows[0]))
            writer.writeheader()
            writer.writerows(layer_rows)
        torch.cuda.empty_cache()
    print(f'Diagnostics: {args.output}', flush=True)


if __name__ == '__main__':
    main()
