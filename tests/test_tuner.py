"""Test code for `Tuner` class."""

# flake8: noqa=D401
# pylint: disable=protected-access

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.manic import Tuner
from src.manic.types import ParamTensorDict

_SGD_TUNERS = ["sgd_store_delta", "sgd_compute_delta"]


def test_class_types(sgd: Tuner) -> None:
    """Test `Tuner` class types."""
    err_str = "Invalid class type"
    assert isinstance(sgd, Tuner), err_str
    assert isinstance(sgd.base_optimizer, torch.optim.SGD), err_str


def test_refresh_model_params(model: nn.Module, sgd: Tuner) -> None:
    """Test `_refresh_model_params()` behavior."""
    # Make alias for brevity
    state = sgd._tuner_state

    # Record model parameter values for comparisons
    model_params = ParamTensorDict()
    for x in model.parameters():
        model_params[x] = x.clone().detach()

    # Refresh model parameter values
    sgd._refresh_model_params()

    # Check for common parameters
    err_str = "Parameters differ"
    assert model_params.keys() == state.model_params.keys(), err_str

    # Check parameter values
    err_str = "Error in parameter values"
    for x in model_params:  # pylint: disable=consider-using-dict-items
        assert torch.equal(model_params[x], state.model_params[x]), err_str


@jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("_sgd", _SGD_TUNERS)
def test_refresh_updates(
    request, model: nn.Module, _sgd: str, x: Float[Tensor, "..."], y: Float[Tensor, "..."]
) -> None:
    """Test `_refresh_updates()` behavior."""
    # Get test fixture
    sgd = request.getfixturevalue(_sgd)

    # Compute gradients
    sgd.base_optimizer.zero_grad()
    torch.nn.MSELoss()(model(x), y).backward()

    # Refresh model parameter update values
    sgd._refresh_updates()

    # Check update values
    err_str = "Error in update values"
    for p in model.parameters():
        update = sgd.get_update(p)
        assert torch.allclose(update, p.grad), err_str


def test_refresh_updates_side_effects(model: nn.Module, sgd: Tuner) -> None:
    """Test `_refresh_updates()` for side effects."""
    # Record model parameter values for comparisons
    model_params = ParamTensorDict()
    for x in model.parameters():
        model_params[x] = x.clone().detach()

    # Refresh model parameter update values
    sgd._refresh_updates()

    # Check parameter values
    err_str = "Parameter values were modified"
    for x in model.parameters():
        assert torch.all(x == model_params[x]), err_str


def test_restore_params(model: nn.Module, sgd: Tuner) -> None:
    """Test `_restore_params()` behavior."""
    # Make alias for brevity
    state = sgd._tuner_state

    # Record model parameter values for comparisons
    model_params = ParamTensorDict()
    for x in model.parameters():
        model_params[x] = x.clone().detach()

    # Refresh model parameter update values
    sgd._refresh_model_params()

    # Modify parameter values
    for x in model.parameters():
        x.data = torch.randn_like(x)

    # Restore parameter values
    sgd._restore_model_params()

    # Check for common parameters
    err_str = "Parameters differ"
    assert model_params.keys() == state.model_params.keys(), err_str

    # Check parameter values
    err_str = "Error in parameter values"
    for x in model.parameters():
        assert torch.equal(model_params[x], state.model_params[x]), err_str


@jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("_sgd", _SGD_TUNERS)
def test_delta(
    request, model: nn.Module, _sgd: str, x: Float[Tensor, "..."], y: Float[Tensor, "..."]
) -> None:
    """
    Test `get_delta()` behavior.

    The following behavior is expected:
        - Initial "delta" values should be zeroes.
        - After one step, the "delta" values should be the update values.
    """
    # Get test fixture
    sgd = request.getfixturevalue(_sgd)

    # Check "delta" values
    for p in model.parameters():
        delta = sgd.get_delta(p)
        assert torch.all(delta == 0.0), "Error in delta values"

    # Compute gradients
    sgd.base_optimizer.zero_grad()
    torch.nn.MSELoss()(model(x), y).backward()

    # Set sum of components value
    # - Note: This test is not sensible when this value is zero
    sgd.s_sum = 1.0

    # Run one optimizer step (this refreshes update cache)
    sgd.step()

    # Check "delta" values
    for p in model.parameters():
        delta = sgd.get_delta(p)
        update = sgd.get_update(p)
        assert torch.allclose(delta, update), "Error in delta values"


@jaxtyped(typechecker=typechecker)
def test_get_update(
    model: nn.Module, sgd: Tuner, x: Float[Tensor, "..."], y: Float[Tensor, "..."]
) -> None:
    """Test `get_update()` behavior."""
    # Check for expected failure
    for p in model.parameters():
        with pytest.raises(ValueError):
            sgd.get_update(p)

    # Compute gradients
    sgd.base_optimizer.zero_grad()
    torch.nn.MSELoss()(model(x), y).backward()

    # Refresh model parameter update values
    sgd._refresh_updates()

    # Check for expected failure
    with pytest.raises(ValueError):
        sgd.get_update(nn.Parameter(torch.randn(1)))
