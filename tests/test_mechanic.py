"""Test code for `MechanicOptimizer` class."""

# flake8: noqa=D401
# pylint: disable=protected-access

import pytest
import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.satanic import MechanicOptimizer, make_mechanic_optimizer


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


@pytest.fixture(name="sgd")
def fixture_sgd(model: nn.Module) -> MechanicOptimizer:
    """Instance of `MechanicOptimizer` with vanilla SGD base optimizer class."""
    return make_mechanic_optimizer(torch.optim.SGD, model.parameters())


@pytest.fixture(name="x")
def fixture_x(dims: dict[str, int]) -> Float[Tensor, "..."]:
    """Input tensor."""
    return torch.randn(dims["batch"], dims["input"]).requires_grad_(False)


@pytest.fixture(name="y")
def fixture_y(dims: dict[str, int]) -> Float[Tensor, "..."]:
    """Output tensor."""
    return torch.randn(dims["batch"], dims["output"]).requires_grad_(False)


def test_inheritance_hierarchy(sgd: MechanicOptimizer) -> None:
    """Test `MechanicOptimizer` inheritance hierarchy."""
    assert isinstance(sgd, MechanicOptimizer), "Invalid class type"
    assert len(type(sgd).__bases__) == 2, "Invalid inheritance pattern"
    assert type(sgd).__bases__[0] == MechanicOptimizer, "Invalid superclass type"
    assert type(sgd).__bases__[1] == torch.optim.SGD, "Invalid superclass type"


def test_refresh_param_cache(model: nn.Module, sgd: MechanicOptimizer) -> None:
    """Test `_refresh_param_cache()` behavior."""
    # Cache parameter values for comparisons
    param_cache = {}
    for param in model.parameters():
        param_cache[param] = param.clone().detach()

    # Cache parameter values
    sgd._refresh_param_cache()

    # Check for common parameters
    err_str = "Parameter cache keys differ from expected"
    assert param_cache.keys() == sgd._param_cache.keys(), err_str

    # Check parameter values
    err_str = "Error in parameter values"
    for param in param_cache:  # pylint: disable=consider-using-dict-items
        assert torch.equal(param_cache[param], sgd._param_cache[param]), err_str


def test_refresh_update_cache(
    model: nn.Module, sgd: MechanicOptimizer, x: Float[Tensor, "..."], y: Float[Tensor, "..."]
) -> None:
    """Test `refresh_update_cache()` behavior."""
    # Compute gradients
    sgd.zero_grad()
    torch.nn.MSELoss()(model(x), y).backward()

    # Cache updates
    sgd._refresh_update_cache()

    # Check updates
    err_str = "Error in updates"
    for group in sgd.param_groups:
        for param in group["params"]:
            if "maximize" not in group or not group["maximize"]:
                expected_update = param.grad
            else:
                expected_update = -1.0 * param.grad
            update = sgd.get_update(param)
            assert torch.allclose(update, expected_update), err_str


def test_refresh_update_cache_side_effects(model: nn.Module, sgd: MechanicOptimizer) -> None:
    """Test `_refresh_update_cache()` for side effects."""
    # Cache parameter values for comparisons
    param_cache = {}
    for param in model.parameters():
        param_cache[param] = param.clone().detach()

    # Cache updates
    sgd._refresh_update_cache()

    # Check if parameter values were modified
    err_str = "Parameter values were modified"
    for param in model.parameters():
        assert torch.all(param == param_cache[param]), err_str


def test_restore_params(model: nn.Module, sgd: MechanicOptimizer) -> None:
    """Test `_restore_params()` behavior."""
    # Cache parameter values for comparisons
    param_cache = {}
    for param in model.parameters():
        param_cache[param] = param.clone().detach()

    # Cache parameter values
    sgd._refresh_param_cache()

    # Modify parameter values
    for param in model.parameters():
        param.data = torch.randn_like(param)

    # Restore parameter values
    sgd._restore_params()

    # Check for common parameters
    err_str = "Parameter cache keys differ from expected"
    assert param_cache.keys() == sgd._param_cache.keys(), err_str

    # Check parameter values
    err_str = "Error in parameter values"
    for param in model.parameters():
        assert torch.equal(param_cache[param], sgd._param_cache[param]), err_str


def test_get_update(
    model: nn.Module, sgd: MechanicOptimizer, x: Float[Tensor, "..."], y: Float[Tensor, "..."]
) -> None:
    """Test `get_update()` behavior."""
    # Check for expected failure
    for param in model.parameters():
        with pytest.raises(ValueError):
            sgd.get_update(param)

    # Compute gradients
    sgd.zero_grad()
    torch.nn.MSELoss()(model(x), y).backward()

    # Cache updates
    sgd._refresh_update_cache()

    # Check for expected failure
    with pytest.raises(ValueError):
        sgd.get_update(torch.randn(1))
