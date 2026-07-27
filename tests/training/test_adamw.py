from pathlib import Path

import numpy as np
import torch

from mini_llm_runtime.training.adamw import AdamW


SNAPSHOT_PATH = (
    Path(__file__).resolve().parent
    / "snapshots"
    / "test_adamw.npz"
)


def _optimize(optimizer_class) -> torch.Tensor:
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    optimizer = optimizer_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    for _ in range(1000):
        optimizer.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        optimizer.step()
    return model.weight.detach()


def test_adamw():
    pytorch_weights = _optimize(torch.optim.AdamW)
    actual_weights = _optimize(AdamW)

    if torch.allclose(actual_weights, pytorch_weights, atol=1e-4):
        return

    with np.load(SNAPSHOT_PATH) as snapshot:
        expected = snapshot["array"]
    np.testing.assert_allclose(
        actual_weights.numpy(),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )
