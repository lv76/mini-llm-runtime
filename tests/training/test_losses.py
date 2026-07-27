import numpy as np
import torch
import torch.nn.functional as torch_functional

from mini_llm_runtime.training.losses import cross_entropy


def test_cross_entropy():
    inputs = torch.tensor(
        [
            [
                [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
            ],
            [
                [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
            ],
        ]
    )
    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])

    flattened_inputs = inputs.view(-1, inputs.size(-1))
    flattened_targets = targets.view(-1)
    expected = torch_functional.cross_entropy(
        flattened_inputs,
        flattened_targets,
    )
    actual = cross_entropy(flattened_inputs, flattened_targets)
    np.testing.assert_allclose(
        actual.detach().numpy(),
        expected.detach().numpy(),
        atol=1e-4,
    )

    large_inputs = 1000.0 * flattened_inputs
    large_expected = torch_functional.cross_entropy(
        large_inputs,
        flattened_targets,
    )
    np.testing.assert_allclose(
        cross_entropy(
            large_inputs,
            flattened_targets,
        ).detach().numpy(),
        large_expected.detach().numpy(),
        atol=1e-4,
    )
