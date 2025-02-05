"""Test code for `MechanicOptimizer` class."""

# flake8: noqa=D401
# pylint: disable=protected-access

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.manic import MechanicOptimizer
from src.manic.types import ParamTensorDict

_SGD_OPTIMIZERS = ["sgd_store_delta", "sgd_compute_delta"]


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


@jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("_sgd", _SGD_OPTIMIZERS)
def test_refresh_updates(
    request, model: nn.Module, _sgd: str, x: Float[Tensor, "..."], y: Float[Tensor, "..."]
) -> None:
    """Test `_refresh_updates()` behavior."""
    # Get test fixture
    sgd = request.getfixturevalue(_sgd)

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
            assert torch.allclose(update, p.grad), err_str


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


@jaxtyped(typechecker=typechecker)
def test_delta(
    model: nn.Module, sgd: MechanicOptimizer, x: Float[Tensor, "..."], y: Float[Tensor, "..."]
) -> None:
    """
    Test "delta" cache behavior.

    The following behavior is expected:
    - Initial "delta" values should be zeroes.
    - After one optimizer step, "delta" values should be update values.
    """
    # Check "delta" values
    for p in model.parameters():
        delta = sgd.get_delta(p)
        assert torch.all(delta == 0.0), "Error in initial delta values"

    # Compute gradients
    sgd.zero_grad()
    torch.nn.MSELoss()(model(x), y).backward()

    # Run one optimizer step (this refreshes update cache)
    sgd.step()

    # Check "delta" values
    for p in model.parameters():
        delta = sgd.get_delta(p)
        update = sgd.get_update(p)
        assert torch.all(delta == update), "Error in delta values after step"


@jaxtyped(typechecker=typechecker)
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
