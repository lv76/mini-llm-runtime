"""Compare the baseline Triton reduction with the CUDA reference kernel."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from mini_llm_runtime.kernels.cuda.accumulator import (
    paged_attention_accumulator_cuda,
)


N = 64
D = 256


@triton.jit
def _triton_accumulator_kernel(p_ptr, v_ptr, out_ptr, BLOCK_D: tl.constexpr):
    batch = tl.program_id(0)
    offs_n = tl.arange(0, 64)
    offs_d = tl.arange(0, BLOCK_D)
    p = tl.load(p_ptr + batch * 64 + offs_n)
    v = tl.load(
        v_ptr + batch * 64 * 256 + offs_n[:, None] * 256 + offs_d[None, :]
    )
    acc = tl.sum(p[:, None] * v, axis=0)
    tl.store(out_ptr + batch * 256 + offs_d, acc)


def triton_accumulator(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    output = torch.empty((p.shape[0], D), device=p.device, dtype=torch.float32)
    _triton_accumulator_kernel[(p.shape[0],)](p, v, output, BLOCK_D=D, num_warps=4)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=500)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    p = torch.randn(args.batch, N, device="cuda", dtype=torch.float32)
    v = torch.randn(args.batch, N, D, device="cuda", dtype=torch.float32)

    cuda_output = paged_attention_accumulator_cuda(p, v)
    triton_output = triton_accumulator(p, v)
    expected = torch.sum(p[:, :, None] * v, dim=1)
    torch.testing.assert_close(cuda_output, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(triton_output, expected, rtol=2e-5, atol=2e-5)

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
        paged_attention_accumulator_cuda(p, v)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    cuda_ms = triton.testing.do_bench(
        lambda: paged_attention_accumulator_cuda(p, v),
        warmup=args.warmup,
        rep=args.rep,
    )
    triton_ms = triton.testing.do_bench(
        lambda: triton_accumulator(p, v),
        warmup=args.warmup,
        rep=args.rep,
    )
    print(f"batch={args.batch}, N={N}, D={D}")
    print(f"Triton baseline: {triton_ms * 1000:.3f} us")
    print(f"CUDA reference:  {cuda_ms * 1000:.3f} us")
    print("CUDA static shared memory: 4096 bytes/block")


if __name__ == "__main__":
    main()
