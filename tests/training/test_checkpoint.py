import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_functional

from mini_llm_runtime.training.adamw import AdamW
from mini_llm_runtime.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


class _TestNet(nn.Module):
    def __init__(self, d_input: int = 100, d_output: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(d_input, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, d_output)

    def forward(self, x):
        x = torch_functional.relu(self.fc1(x))
        x = torch_functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _optimizers_are_equal(
    first_state_dict,
    second_state_dict,
    atol=1e-8,
    rtol=1e-5,
):
    if set(first_state_dict.keys()) != set(second_state_dict.keys()):
        return False

    if (
        first_state_dict["param_groups"]
        != second_state_dict["param_groups"]
    ):
        return False

    first_state = first_state_dict["state"]
    second_state = second_state_dict["state"]
    if set(first_state.keys()) != set(second_state.keys()):
        return False

    for key in first_state:
        if set(first_state[key].keys()) != set(
            second_state[key].keys()
        ):
            return False

        for sub_key in first_state[key]:
            first_item = first_state[key][sub_key]
            second_item = second_state[key][sub_key]

            if torch.is_tensor(first_item) and torch.is_tensor(
                second_item
            ):
                if not torch.allclose(
                    first_item,
                    second_item,
                    atol=atol,
                    rtol=rtol,
                ):
                    return False
            elif first_item != second_item:
                return False
    return True


def test_checkpointing(tmp_path):
    torch.manual_seed(42)
    d_input = 100
    d_output = 10
    num_iters = 10

    model = _TestNet(d_input=d_input, d_output=d_output)
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    iteration = 0
    for _ in range(num_iters):
        optimizer.zero_grad()
        x = torch.rand(d_input)
        y = torch.rand(d_output)
        y_hat = model(x)
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        optimizer.step()
        iteration += 1

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        model,
        optimizer,
        iteration=iteration,
        out=checkpoint_path,
    )

    new_model = _TestNet(d_input=d_input, d_output=d_output)
    new_optimizer = AdamW(
        new_model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    loaded_iteration = load_checkpoint(
        src=checkpoint_path,
        model=new_model,
        optimizer=new_optimizer,
    )
    assert iteration == loaded_iteration

    original_model_state = model.state_dict()
    original_optimizer_state = optimizer.state_dict()
    new_model_state = new_model.state_dict()
    new_optimizer_state = new_optimizer.state_dict()

    assert set(original_model_state) == set(new_model_state)
    assert set(original_optimizer_state) == set(new_optimizer_state)

    for key in original_model_state:
        np.testing.assert_allclose(
            original_model_state[key].detach().numpy(),
            new_model_state[key].detach().numpy(),
        )

    assert _optimizers_are_equal(
        original_optimizer_state,
        new_optimizer_state,
    )
