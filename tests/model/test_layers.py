import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as torch_functional
from torch import nn

from mini_llm_runtime.model.functional import embedding, linear, rmsnorm, silu, swiglu
from mini_llm_runtime.model.layers import Embedding, Linear, RMSNorm, SiLU, SwiGLU


SNAPSHOTS_PATH = Path(__file__).resolve().parent / "snapshots"
MODEL_PATH = Path(__file__).resolve().parent / "fixtures" / "model.pt"


def assert_matches_snapshot(
    actual: torch.Tensor,
    test_name: str,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-2,
) -> None:
    with np.load(SNAPSHOTS_PATH / f"{test_name}.npz") as snapshot:
        expected = snapshot["array"]

    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        expected,
        rtol=rtol,
        atol=atol,
    )


@pytest.fixture(scope="module")
def state_dict() -> dict[str, torch.Tensor]:
    weights = torch.load(MODEL_PATH, map_location="cpu")
    return {key.replace("_orig_mod.", ""): value for key, value in weights.items()}


@pytest.fixture
def in_embeddings() -> torch.Tensor:
    torch.manual_seed(4)
    return torch.randn(4, 12, 64)


@pytest.fixture
def in_indices() -> torch.Tensor:
    torch.manual_seed(6)
    return torch.randint(0, 10_000, (4, 12))


def test_linear(state_dict, in_embeddings):
    d_model = 64
    d_ff = 128
    weight = state_dict["layers.0.ffn.w1.weight"]

    output = linear(
        d_in=d_model,
        d_out=d_ff,
        weights=weight,
        in_features=in_embeddings,
    )
    assert_matches_snapshot(output, "test_linear")

    module = Linear(d_model, d_ff)
    with torch.no_grad():
        module.weight.copy_(weight)
    torch.testing.assert_close(module(in_embeddings), output, rtol=0, atol=0)


def test_embedding(state_dict, in_indices):
    vocab_size = 10_000
    d_model = 64
    weight = state_dict["token_embeddings.weight"]

    output = embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        embedding_weight=weight,
        token_ids=in_indices,
    )
    assert_matches_snapshot(output, "test_embedding")

    module = Embedding(vocab_size, d_model)
    with torch.no_grad():
        module.weight.copy_(weight)
    torch.testing.assert_close(module(in_indices), output, rtol=0, atol=0)


def test_swiglu(state_dict, in_embeddings):
    d_model = 64
    d_ff = 128
    w1_weight, w2_weight, w3_weight = [
        state_dict[f"layers.0.ffn.{name}.weight"]
        for name in ("w1", "w2", "w3")
    ]

    output = swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=in_embeddings,
    )
    assert_matches_snapshot(output, "test_swiglu", atol=1e-5)

    module = SwiGLU(d_model, d_ff)
    with torch.no_grad():
        module.w1.weight.copy_(w1_weight)
        module.w2.weight.copy_(w2_weight)
        module.w3.weight.copy_(w3_weight)
    torch.testing.assert_close(module(in_embeddings), output, rtol=0, atol=0)


def test_rmsnorm(state_dict, in_embeddings):
    weight = state_dict["layers.1.ln1.weight"]
    d_model = weight.shape[0]

    output = rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weight,
        in_features=in_embeddings,
    )
    assert_matches_snapshot(output, "test_rmsnorm", atol=1e-4)

    module = RMSNorm(d_model, eps=1e-5)
    with torch.no_grad():
        module.weight.copy_(weight)
    torch.testing.assert_close(module(in_embeddings), output, rtol=0, atol=0)


def test_silu_matches_pytorch():
    in_features = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )

    expected_output = torch_functional.silu(in_features)
    actual_output = silu(in_features)
    np.testing.assert_allclose(
        actual_output.detach().numpy(),
        expected_output.detach().numpy(),
        atol=1e-5,
    )
    torch.testing.assert_close(SiLU()(in_features), actual_output, rtol=0, atol=0)


def test_parameter_shapes_and_initialization():
    d_model = 64
    d_ff = 128
    vocab_size = 10_000

    torch.manual_seed(7)
    linear_module = Linear(d_model, d_ff)
    torch.manual_seed(7)
    linear_std = math.sqrt(2 / (d_model + d_ff))
    expected_linear_weight = nn.init.trunc_normal_(
        torch.empty(d_ff, d_model),
        std=linear_std,
        a=-3 * linear_std,
        b=3 * linear_std,
    )
    torch.testing.assert_close(
        linear_module.weight,
        expected_linear_weight,
        rtol=0,
        atol=0,
    )

    torch.manual_seed(8)
    embedding_module = Embedding(vocab_size, d_model)
    torch.manual_seed(8)
    expected_embedding_weight = nn.init.trunc_normal_(
        torch.empty(vocab_size, d_model),
        std=1.0,
        a=-3.0,
        b=3.0,
    )
    torch.testing.assert_close(
        embedding_module.weight,
        expected_embedding_weight,
        rtol=0,
        atol=0,
    )

    swiglu_module = SwiGLU(d_model, d_ff)
    assert swiglu_module.w1.weight.shape == (d_ff, d_model)
    assert swiglu_module.w2.weight.shape == (d_model, d_ff)
    assert swiglu_module.w3.weight.shape == (d_ff, d_model)

    rmsnorm_module = RMSNorm(d_model)
    torch.testing.assert_close(
        rmsnorm_module.weight,
        torch.ones(d_model),
        rtol=0,
        atol=0,
    )
