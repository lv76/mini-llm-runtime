from copy import deepcopy
from typing import Type

import numpy
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mini_llm_runtime.distributed.sharded_optimizer import (
    ShardedOptimizer,
)

from .common import (
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
)


def _assert_parameters_identical_across_ranks(model, world_size):
    for parameter in model.parameters():
        gathered = [
            torch.zeros_like(parameter)
            for _ in range(world_size)
        ]
        dist.all_gather(gathered, parameter)
        for rank_parameter in gathered:
            assert torch.allclose(rank_parameter, parameter)


def _assert_optimizer_state_is_sharded(
    optimizer,
    rank,
    world_size,
):
    expected_local_params = [
        parameter
        for index, parameter in enumerate(optimizer.all_params)
        if index % world_size == rank
    ]
    assert len(optimizer.local_params) == len(
        expected_local_params
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            optimizer.local_params,
            expected_local_params,
        )
    )

    local_optimizer_params = [
        parameter
        for group in optimizer.local_optimzer.param_groups
        for parameter in group["params"]
    ]
    assert len(local_optimizer_params) == len(
        expected_local_params
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            local_optimizer_params,
            expected_local_params,
        )
    )

    state_params = set(optimizer.local_optimzer.state)
    expected_state_params = {
        parameter
        for parameter in expected_local_params
        if parameter.grad is not None
    }
    assert state_params == expected_state_params

    owned_indices = [
        index
        for index in range(len(optimizer.all_params))
        if index % world_size == rank
    ]
    state_indices = [
        index
        for index, parameter in enumerate(optimizer.all_params)
        if parameter in state_params
    ]

    all_owned_indices = [None] * world_size
    all_state_indices = [None] * world_size
    dist.all_gather_object(all_owned_indices, owned_indices)
    dist.all_gather_object(all_state_indices, state_indices)

    for owner_rank in range(world_size):
        expected_owned_indices = [
            index
            for index in range(len(optimizer.all_params))
            if index % world_size == owner_rank
        ]
        assert (
            all_owned_indices[owner_rank]
            == expected_owned_indices
        )
        assert set(all_state_indices[owner_rank]).issubset(
            expected_owned_indices
        )


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_sharded_optimizer(model_class):
    world_size = 2
    mp.spawn(
        _test_sharded_optimizer,
        args=(world_size, model_class),
        nprocs=world_size,
        join=True,
    )


def _test_sharded_optimizer(rank: int, world_size: int, model_class: Type[torch.nn.Module]):
    # Use gloo backend for CPU
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(42)
    optimizer_cls = torch.optim.AdamW
    # Since we've seeded, model states should be the same across ranks without having to broadcast.
    non_sharded_model = model_class().to(device)

    non_sharded_optimizer = optimizer_cls(
        non_sharded_model.parameters(),
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    sharded_model = deepcopy(non_sharded_model)
    sharded_optimizer = ShardedOptimizer(
        sharded_model.parameters(),
        optimizer_cls,
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    if model_class is ToyModelWithTiedWeights:
        assert sharded_model.fc4.weight is sharded_model.fc2.weight

    for _ in range(10):
        non_sharded_optimizer.zero_grad()
        sharded_optimizer.zero_grad()
        assert all(
            parameter.grad is None
            for parameter in sharded_model.parameters()
        )

        # batch size 32, 10 input features, 5 output features
        input_ = torch.rand((32, 10)).to(device)
        labels = torch.rand((32, 10)).to(device)
        non_sharded_input = deepcopy(input_)
        sharded_input = deepcopy(input_)
        non_sharded_labels = deepcopy(labels)
        sharded_labels = deepcopy(labels)

        non_sharded_model_logits = non_sharded_model(non_sharded_input)
        sharded_model_logits = sharded_model(sharded_input)

        non_sharded_model_loss = ((non_sharded_labels - non_sharded_model_logits) ** 2).sum()
        sharded_model_loss = ((sharded_labels - sharded_model_logits) ** 2).sum()

        non_sharded_model_loss.backward()
        sharded_model_loss.backward()

        non_sharded_optimizer.step()
        sharded_optimizer.step()

        _assert_parameters_identical_across_ranks(
            sharded_model,
            world_size,
        )
        _assert_optimizer_state_is_sharded(
            sharded_optimizer,
            rank,
            world_size,
        )

    # Check that the final model weights are the same regardless of if we're using
    # the sharded or non-sharded optimizer.
    for non_sharded_parameters, sharded_parameters in zip(non_sharded_model.parameters(), sharded_model.parameters()):
        numpy.testing.assert_allclose(
            non_sharded_parameters.detach().cpu().numpy(),
            sharded_parameters.detach().cpu().numpy(),
        )

    if model_class is ToyModelWithTiedWeights:
        assert sharded_model.fc4.weight is sharded_model.fc2.weight

    _cleanup_process_group()
