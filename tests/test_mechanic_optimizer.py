"""Test code for `MechanicOptimizer` class."""

# flake8: noqa=D401
# pylint: disable=protected-access

import pytest
import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.satanic import MechanicOptimizer
from src.satanic.types import ParamTensorDict


def test_class_types(sgd: MechanicOptimizer) -> None:
    """Test `MechanicOptimizer` class types."""
    err_str = "Invalid class type"
    assert isinstance(sgd, MechanicOptimizer), err_str
    assert isinstance(sgd._base_optimizer, torch.optim.SGD), err_str


def test_refresh_params(model: nn.Module, sgd: MechanicOptimizer) -> None:
    """Test `_refresh_params()` behavior."""
    # Make local parameter cache for comparisons
    params = ParamTensorDict()
    for x in model.parameters():
        params[x] = x.clone().detach()

    # Refresh parameter cache with current values
    sgd._refresh_params()

    # Check for common parameters
    err_str = "Parameter cache keys differ"
    assert params.keys() == sgd._params.keys(), err_str

    # Check parameter values
    err_str = "Error in parameter values"
    for x in params:  # pylint: disable=consider-using-dict-items
        assert torch.equal(params[x], sgd._params[x]), err_str


def test_refresh_updates(
    model: nn.Module, sgd: MechanicOptimizer, x: Float[Tensor, "..."], y: Float[Tensor, "..."]
) -> None:
    """Test `refresh_updates()` behavior."""
    # Compute gradients
    sgd.zero_grad()
    torch.nn.MSELoss()(model(x), y).backward()

    # Refresh update cache with current values
    sgd._refresh_updates()

    # Check updates
    err_str = "Error in updates"
    for group in sgd.param_groups:
        for p in group["params"]:
            update = sgd.get_update(p)
            if "maximize" not in group or not group["maximize"]:
                expected_update = p.grad
            else:
                expected_update = -1.0 * p.grad
            assert torch.allclose(update, expected_update), err_str


def test_refresh_updates_side_effects(model: nn.Module, sgd: MechanicOptimizer) -> None:
    """Test `_refresh_updates()` for side effects."""
    # Make local parameter cache for comparisons
    params = ParamTensorDict()
    for x in model.parameters():
        params[x] = x.clone().detach()

    # Refresh update cache with current values
    sgd._refresh_updates()

    # Check if parameter values were modified
    err_str = "Parameter values were modified"
    for x in model.parameters():
        assert torch.all(x == params[x]), err_str


def test_restore_params(model: nn.Module, sgd: MechanicOptimizer) -> None:
    """Test `_restore_params()` behavior."""
    # Make local parameter cache for comparisons
    params = ParamTensorDict()
    for x in model.parameters():
        params[x] = x.clone().detach()

    # Refresh parameter cache with current values
    sgd._refresh_params()

    # Modify parameter values
    for x in model.parameters():
        x.data = torch.randn_like(x)

    # Restore parameter values
    sgd._restore_params()

    # Check for common parameters
    err_str = "Parameter cache keys differ"
    assert params.keys() == sgd._params.keys(), err_str

    # Check parameter values
    err_str = "Error in parameter values"
    for x in model.parameters():
        assert torch.equal(params[x], sgd._params[x]), err_str


def test_get_update(
    model: nn.Module, sgd: MechanicOptimizer, x: Float[Tensor, "..."], y: Float[Tensor, "..."]
) -> None:
    """Test `get_update()` behavior."""
    # Check for expected failure
    for p in model.parameters():
        with pytest.raises(ValueError):
            sgd.get_update(p)

    # Compute gradients
    sgd.zero_grad()
    torch.nn.MSELoss()(model(x), y).backward()

    # Refresh update cache with current values
    sgd._refresh_updates()

    # Check for expected failure
    with pytest.raises(ValueError):
        sgd.get_update(nn.Parameter(torch.randn(1)))
