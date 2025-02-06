"""/src/manic/__init__.py."""

from src.manic.mechanic import Mechanic, MechanicParams, MechanicState
from src.manic.mechanic_optimizer import MechanicOptimizer

__all__ = ["MechanicOptimizer", "Mechanic", "MechanicParams", "MechanicState"]
__doc__ = "A PyTorch implementation of the Mechanic learning rate scheduler."
