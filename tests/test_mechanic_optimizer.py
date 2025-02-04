"""Test code for `MechanicOptimizer` class."""

# flake8: noqa=D401
# pylint: disable=protected-access

import pytest
import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.satanic import MechanicOptimizer


def test_class_types(sgd: MechanicOptimizer) -> None:
    """Test `MechanicOptimizer` class types."""
    err_str = "Invalid class type"
    assert isinstance(sgd, MechanicOptimizer), err_str
    assert isinstance(sgd._base_optimizer, torch.optim.SGD), err_str


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
