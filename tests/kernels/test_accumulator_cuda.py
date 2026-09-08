import pytest
import torch

from mini_llm_runtime.kernels.cuda.accumulator import (
    paged_attention_accumulator_cuda,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch", [1, 7, 1024])
def test_paged_attention_accumulator_cuda(batch: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    p = torch.randn(batch, 64, device="cuda", dtype=torch.float32, generator=generator)
    v = torch.randn(
        batch,
        64,
        256,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )

    actual = paged_attention_accumulator_cuda(p, v)
    expected = torch.sum(p[:, :, None] * v, dim=1)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
