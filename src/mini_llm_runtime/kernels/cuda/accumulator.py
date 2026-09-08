from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def _extension():
    source = Path(__file__).with_name("paged_attention_accumulator.cu")
    return load(
        name="mini_llm_runtime_accumulator_cuda",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def paged_attention_accumulator_cuda(
    p: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute batched ``sum_n p[n] * v[n, d]`` for N=64 and D=256."""
    if not p.is_cuda or not v.is_cuda:
        raise ValueError("p and v must be CUDA tensors")
    if p.dtype != torch.float32 or v.dtype != torch.float32:
        raise TypeError("p and v must have dtype torch.float32")
    if p.ndim != 2 or p.shape[1] != 64:
        raise ValueError("p must have shape [batch, 64]")
    if v.ndim != 3 or v.shape[1:] != (64, 256):
        raise ValueError("v must have shape [batch, 64, 256]")
    if p.shape[0] != v.shape[0]:
        raise ValueError("p and v must have the same batch dimension")

    return _extension().forward(p.contiguous(), v.contiguous())
