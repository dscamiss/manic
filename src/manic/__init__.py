"""/src/manic/__init__.py."""

from src.manic.mechanic import Mechanic, MechanicParams, MechanicState
from src.manic.updater import Updater, UpdaterParams

__all__ = ["Updater", "UpdaterParams", "Mechanic", "MechanicParams", "MechanicState"]
__doc__ = "A PyTorch implementation of the Mechanic learning rate scale tuner."
