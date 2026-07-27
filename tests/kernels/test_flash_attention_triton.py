import pytest
import torch
from einops import einsum


pytest.importorskip("triton")

from mini_llm_runtime.kernels.triton.flash_attention import (  # noqa: E402
    flash_attention,
)


def _attention_and_lse(q, k, v, is_causal=False):
    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    d = q.shape[-1]
    scale = 1 / (d**0.5)
    scores = einsum(
        q,
        k,
        "... q d, ... k d -> ... q k",
    ) * scale
    if is_causal:
        scores = torch.where(
            torch.arange(
                n_queries,
                device=scores.device,
            )[None, :, None]
            >= torch.arange(
                n_keys,
                device=scores.device,
            )[None, None, :],
            scores,
            -1e6,
        )
    probabilities = torch.softmax(scores, dim=-1)
    output = einsum(
        probabilities,
        v,
        "... q k, ... k d -> ... q d",
    )
    logsumexp = torch.logsumexp(scores, dim=-1)
    return output, logsumexp


def _make_attn_inputs(device=None):
    torch.random.manual_seed(0)
    batch_size = 4
    n_queries = 128
    n_keys = 128
    d = 64
    q = torch.randn(
        batch_size,
        n_queries,
        d,
        device=device,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        n_keys,
        d,
        device=device,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        n_keys,
        d,
        device=device,
        requires_grad=True,
    )
    do = torch.randn(
        batch_size,
        n_queries,
        d,
        device=device,
    )

    return q, k, v, do


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attention_triton_forward(is_causal):
    q, k, v, _do = _make_attn_inputs("cuda")
    output = flash_attention(q, k, v, is_causal)

    assert output.grad_fn.saved_tensors is not None, (
        "No saved tensors found in the output tensor. Make sure "
        "the autograd forward saves them with ctx.save_for_backward."
    )
    maybe_lse = [
        tensor
        for tensor in output.grad_fn.saved_tensors
        if tensor.shape == (q.shape[0], q.shape[1])
    ]

    assert len(maybe_lse) == 1, (
        f"Expected one tensor of shape {q.shape[0], q.shape[1]} "
        f"in saved tensors, but found {len(maybe_lse)}."
    )
    lse = maybe_lse[0]

    expected_output, expected_lse = _attention_and_lse(
        q,
        k,
        v,
        is_causal,
    )

    torch.testing.assert_close(
        output,
        expected_output,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        lse,
        expected_lse,
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attention_triton_backward(is_causal):
    q, k, v, do = _make_attn_inputs("cuda")
    _attention_and_lse(q, k, v, is_causal)[0].backward(do)
    expected_dq = q.grad
    expected_dk = k.grad
    expected_dv = v.grad

    q, k, v, do = _make_attn_inputs("cuda")
    flash_attention(q, k, v, is_causal).backward(do)

    torch.testing.assert_close(
        expected_dq,
        q.grad,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        expected_dk,
        k.grad,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        expected_dv,
        v.grad,
        rtol=1e-2,
        atol=1e-2,
    )
