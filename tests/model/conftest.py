from pathlib import Path

import numpy as np
import pytest
import torch


SNAPSHOTS_PATH = Path(__file__).resolve().parent / "snapshots"
MODEL_PATH = Path(__file__).resolve().parent / "fixtures" / "model.pt"


@pytest.fixture
def assert_matches_snapshot(request):
    def assert_match(
        actual: torch.Tensor,
        *,
        rtol: float = 1e-4,
        atol: float = 1e-2,
    ) -> None:
        snapshot_path = SNAPSHOTS_PATH / f"{request.node.name}.npz"
        with np.load(snapshot_path) as snapshot:
            expected = snapshot["array"]

        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=rtol,
            atol=atol,
        )

    return assert_match


@pytest.fixture(scope="session")
def state_dict() -> dict[str, torch.Tensor]:
    weights = torch.load(MODEL_PATH, map_location="cpu")
    return {
        key.replace("_orig_mod.", ""): value
        for key, value in weights.items()
    }


@pytest.fixture
def n_layers():
    return 3


@pytest.fixture
def vocab_size():
    return 10_000


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def n_queries():
    return 12


@pytest.fixture
def n_keys():
    return 16


@pytest.fixture
def n_heads():
    return 4


@pytest.fixture
def d_head():
    return 16


@pytest.fixture
def d_model(n_heads, d_head):
    return n_heads * d_head


@pytest.fixture
def d_ff():
    return 128


@pytest.fixture
def q(batch_size, n_queries, d_model):
    torch.manual_seed(1)
    return torch.randn(batch_size, n_queries, d_model)


@pytest.fixture
def k(batch_size, n_keys, d_model):
    torch.manual_seed(2)
    return torch.randn(batch_size, n_keys, d_model)


@pytest.fixture
def v(batch_size, n_keys, d_model):
    torch.manual_seed(3)
    return torch.randn(batch_size, n_keys, d_model)


@pytest.fixture
def in_embeddings(batch_size, n_queries, d_model):
    torch.manual_seed(4)
    return torch.randn(batch_size, n_queries, d_model)


@pytest.fixture
def mask(batch_size, n_queries, n_keys):
    torch.manual_seed(5)
    return torch.randn(batch_size, n_queries, n_keys) > 0.5


@pytest.fixture
def in_indices(batch_size, n_queries):
    torch.manual_seed(6)
    return torch.randint(0, 10_000, (batch_size, n_queries))


@pytest.fixture
def theta():
    return 10_000.0


@pytest.fixture
def pos_ids(n_queries):
    return torch.arange(0, n_queries)
