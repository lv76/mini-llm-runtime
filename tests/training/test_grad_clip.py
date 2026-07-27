import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from mini_llm_runtime.training.grad_clip import gradient_clipping


def test_gradient_clipping():
    tensors = [torch.randn((5, 5)) for _ in range(6)]
    max_norm = 1e-2

    parameters = tuple(
        torch.nn.Parameter(torch.clone(tensor))
        for tensor in tensors
    )
    parameters[-1].requires_grad_(False)

    loss = torch.cat(parameters).sum()
    loss.backward()
    clip_grad_norm_(parameters, max_norm)
    expected_grads = [
        torch.clone(parameter.grad)
        for parameter in parameters
        if parameter.grad is not None
    ]

    actual_parameters = tuple(
        torch.nn.Parameter(torch.clone(tensor))
        for tensor in tensors
    )
    actual_parameters[-1].requires_grad_(False)
    actual_loss = torch.cat(actual_parameters).sum()
    actual_loss.backward()
    gradient_clipping(actual_parameters, max_norm)
    actual_grads = [
        torch.clone(parameter.grad)
        for parameter in actual_parameters
        if parameter.grad is not None
    ]

    assert len(expected_grads) == len(actual_grads)

    for expected_grad, actual_grad in zip(
        expected_grads,
        actual_grads,
    ):
        np.testing.assert_allclose(
            expected_grad.detach().numpy(),
            actual_grad.detach().numpy(),
            atol=1e-5,
        )
