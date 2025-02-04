"""Test code for `Mechanic` class."""

# flake8: noqa=D401
# pylint: disable=protected-access

import pytest
from torch.optim.lr_scheduler import LRScheduler

from src.satanic import Mechanic, MechanicOptimizer, MechanicParams


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
