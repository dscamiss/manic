"""Test configuration."""

import random

import numpy as np
import pytest
import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.satanic import MechanicOptimizer


def set_seed(seed: int) -> None:
    """
    Set random seeds etc. to attempt reproducibility.

    Args:
        seed: Random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="session", autouse=True)
def setup_session() -> None:
    """Set up for tests."""
    set_seed(11)
    # Working in float64 avoids numerical issues in tests
    torch.set_default_dtype(torch.float64)


@pytest.fixture(name="dims")
def fixture_dims() -> dict[str, int]:
    """Dimensions."""
    return {
        "batch": 2,
        "input": 3,
        "output": 4,
    }


@pytest.fixture(name="model")
def fixture_model(dims: dict[str, int]) -> nn.Module:
    """Network model."""
    return nn.Sequential(
        nn.Linear(dims["input"], 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, dims["output"]),
    )


@pytest.fixture(name="sgd_store_delta")
def fixture_sgd_store_delta(model: nn.Module) -> MechanicOptimizer:
    """`MechanicOptimizer` with SGD base optimizer."""
    base_optimizer = torch.optim.SGD(model.parameters())
    return MechanicOptimizer(base_optimizer, store_delta=True)


@pytest.fixture(name="sgd_compute_delta")
def fixture_sgd_compute_delta(model: nn.Module) -> MechanicOptimizer:
    """`MechanicOptimizer` with SGD base optimizer."""
    base_optimizer = torch.optim.SGD(model.parameters())
    return MechanicOptimizer(base_optimizer, store_delta=False)


@pytest.fixture()
def sgd(sgd_store_delta: MechanicOptimizer) -> MechanicOptimizer:
    """Aliased text fixture for brevity."""
    return sgd_store_delta


@pytest.fixture(name="x")
def fixture_x(dims: dict[str, int]) -> Float[Tensor, "..."]:
    """Input tensor."""
    return torch.randn(dims["batch"], dims["input"]).requires_grad_(False)


@pytest.fixture(name="y")
def fixture_y(dims: dict[str, int]) -> Float[Tensor, "..."]:
    """Output tensor."""
    return torch.randn(dims["batch"], dims["output"]).requires_grad_(False)
