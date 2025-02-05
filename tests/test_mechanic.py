"""Test code for `Mechanic` class."""

# flake8: noqa=D401
# pylint: disable=protected-access

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from typeguard import typechecked as typechecker

from src.manic import Mechanic, MechanicOptimizer, MechanicParams


@pytest.fixture(name="mechanic_params")
def fixture_mechanic_params() -> MechanicParams:
    """Core `Mechanic` parameters."""
    return MechanicParams()


@pytest.fixture(name="mechanic")
def fixture_mechanic(sgd: MechanicOptimizer, mechanic_params: MechanicParams) -> Mechanic:
    """Mechanic learning rate scheduler."""
    return Mechanic(sgd, -1, mechanic_params)


def test_creation(mechanic_params: MechanicParams, mechanic: Mechanic) -> None:
    """Test `Mechanic` class types."""
    assert isinstance(mechanic, Mechanic), "Invalid class type"
    assert len(type(mechanic).__bases__) == 1, "Invalid inheritance pattern"
    assert type(mechanic).__bases__[0] == LRScheduler, "Invalid superclass type"

    assert isinstance(mechanic._mechanic_params, MechanicParams), "Invalid class type"
    assert mechanic._mechanic_params == mechanic_params, "Invalid core parameters"


@jaxtyped(typechecker=typechecker)
def test_step_side_effects(
    model: nn.Module, mechanic: Mechanic, x: Float[Tensor, "..."], y: Float[Tensor, "..."]
) -> None:
    """
    Test `step()` for side effects.

    The following behavior is expected:
    - The base optimizer's learning rates are not modified.
    -
    """
    optimizer = mechanic._mechanic_optimizer
    base_optimizer = optimizer.base_optimizer

    # Get learning rate(s) from base optimizer
    base_lrs: dict[int, float] = {}
    for idx, group in enumerate(base_optimizer.param_groups):
        base_lrs[idx] = group["lr"]

    # Compute gradients
    optimizer.zero_grad()
    torch.nn.MSELoss()(model(x), y).backward()

    # Run one scheduler step
    mechanic.step()

    # Run one optimizer step
    optimizer.step()

    # Check learning rate(s) in base optimizer
    for idx, group in enumerate(base_optimizer.param_groups):
        assert group["lr"] == base_lrs[idx]
