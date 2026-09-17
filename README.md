# Mini LLM Systems Runtime

Mini LLM Systems Runtime is a compact learning implementation of an LLM model,
optimized attention kernels, and a paged inference runtime. The production
package uses the `src/mini_llm_runtime/` layout and is managed with `uv`; the
original coursework implementations remain under `legacy/`.

## Current status

The repository now contains an end-to-end Tiny Qwen2 reference generation
path:

```text
Request → Scheduler → ModelBatch → cache-aware Qwen2
        → greedy sampling → request/KV-cache lifecycle update
```

Implemented and numerically verified:

- Qwen2 RMSNorm, RoPE, GQA, causal attention, SwiGLU, decoder layers and LM head;
- multilayer paged KV cache with block tables and slot mappings;
- cache-aware prefill and paged single-token decode;
- FP32/BF16 tiled paged chunk-prefill attention with GQA/MQA support,
  causal masking by absolute token position, and online-softmax accumulation;
- cache-aware chunk-prefill integration at the `Qwen2Attention` layer, aligned
  with dense full-context attention across multiple KV tile sizes;
- incremental logits aligned with dense full-sequence recomputation;
- mixed prefill/decode `ModelBatch` execution;
- single-step generation API and greedy multi-step generation;
- dynamic request admission/exit and physical KV block reuse;
- reference-executor validation for packed metadata, block tables, dtype/device,
  and scheduler no-progress conditions;
- Qwen2.5-0.5B config parsing, strict safetensors key/shape validation, and
  Hugging Face-to-local checkpoint loading;
- exact BF16 dense-logits alignment with Transformers for a real
  Qwen2.5-0.5B checkpoint;
- exact real-model alignment for Paged Prefill against HF dense prompt and
  cross-block Paged Decode against HF cached incremental decoding;
- offline Qwen tokenizer integration and multi-step real-text generation through
  the local Paged Prefill/Decode path;
- staggered-arrival and ragged Scheduler benchmarking with request-level TTFT,
  queueing, ITL, end-to-end latency, throughput, and KV utilization metrics;
- Triton attention kernels and CUDA/benchmark experiments.

The current `ModelBatch` executor runs all decode requests in one batched Qwen2
call and scatters their logits back into the original packed token order.
Prefill still runs per request. Batched decode uses tensorized PyTorch paged
attention with batched KV writes. A batched Triton kernel has not yet been
connected. Chunked Prefill is currently complete through the reference kernel
and `Qwen2Attention`; it has not yet been propagated through DecoderLayer,
the full model, ModelRunner, Request state, or Scheduler. Real-model generation
and the Scheduler benchmark default to a
mixed-precision mode: BF16 weights/activations/KV cache, FP32 RMSNorm
statistics and attention softmax, with reduced-precision BF16 GEMM reductions
disabled. FP32 remains the strict correctness reference; BF16 is not required
to reproduce FP32 greedy tokens at near-tied decision boundaries.

## Next milestones

1. Propagate Chunked Prefill through `Qwen2DecoderLayer`, the full Qwen2 model,
   ModelRunner, Request state, and Scheduler; intermediate chunks must not
   sample, and only the final prompt chunk may produce the first output token.
2. Compare multi-step paged greedy token IDs against Hugging Face `generate()`
   while exercising repeated block growth.
3. Benchmark sequential and batched full-model decode on Qwen2.5-0.5B BF16,
   including TPOT, throughput, peak memory, and ragged-batch overhead.
4. Sweep Scheduler arrival rates and concurrency limits to locate throughput
   saturation and the onset of TTFT tail-latency growth.
5. Profile the resulting bottlenecks, then connect a batched Triton paged
   decode kernel and add packed varlen prefill.
6. Add broader FP16/BF16 end-to-end coverage for the serving path.
7. Reduce FIFO head-of-line blocking when a large waiting request cannot be
   admitted but smaller requests behind it could make progress.

See `docs/single_request_paged_qwen2_walkthrough.md` for the current execution
walkthrough and `docs/qwen2_paged_attention_report.md` for the design report.
