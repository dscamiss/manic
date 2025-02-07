"""/src/manic/__init__.py."""

from src.manic.mechanic import Mechanic, MechanicParams, MechanicState
from src.manic.tuner import Tuner, TunerParams

__all__ = ["Tuner", "TunerParams", "Mechanic", "MechanicParams", "MechanicState"]
__doc__ = "A PyTorch implementation of the Mechanic learning rate scheduler."
