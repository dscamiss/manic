"""Mechanic learning rate scheduler."""

from typing import Any, Type

import torch
from jaxtyping import Num, jaxtyped
from torch import Tensor
from typeguard import typechecked as typechecker

from src.satanic.types import Optimizer

# flake8: noqa=DCO010
# pylint: disable=too-few-public-methods


class MechanicOptimizer:
    """Empty class to facilitate wrapper class creation."""


def make_mechanic_optimizer(
    base_optimizer_class: Type[Optimizer], *args: Any, **kwargs: dict[str, Any]
) -> MechanicOptimizer:
    """
    Make wrapped optimizer for use by `Mechanic`.

    Args:
        base_optimizer_class: Base optimizer class.

    Returns:
        Instance of `MechanicOptimizer` wrapping the base optimizer class.
    """

    class _MechanicOptimizer(MechanicOptimizer, base_optimizer_class):
        def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
            super().__init__(*args, **kwargs)
            self._param_cache: dict[Tensor, Tensor] = {}
            self._update_cache: dict[Tensor, Tensor] = {}
            self._updates_available = False

        def _refresh_param_cache(self) -> None:
            """Refresh parameter cache with current values."""
            for group in self.param_groups:
                for param in group["params"]:
                    self._param_cache[param] = param.clone().detach()

        def _refresh_update_cache(self) -> None:
            """Refresh update cache with current values."""
            # Refresh parameter values
            self._refresh_param_cache()

            # Run base optimizer step
            super().step()

            # Derive parameter updates
            for group in self.param_groups:
                for param in group["params"]:
                    prev_param = self._param_cache[param]
                    self._update_cache[param] = param.clone().detach().sub_(prev_param)
                    self._update_cache[param].div_(-1.0 * group["lr"])

            # Restore previous parameters
            self._restore_params()

            # Indicate that parameter updates are available
            self._updates_available = True

        def _restore_params(self) -> None:
            """Restore parameters from parameter cache."""
            with torch.no_grad():
                for group in self.param_groups:
                    for param in group["params"]:
                        param.copy_(self._param_cache[param])

        @jaxtyped(typechecker=typechecker)
        def get_update(self, param: Num[Tensor, "..."]) -> Num[Tensor, "..."]:
            """
            Get update for a particular parameter.

            Args:
                param: Parameter.

            Returns:
                The update associated with `param`.

            Raises:
                ValueError: If update is not available.
            """
            if not self._updates_available or param not in self._update_cache:
                raise ValueError("Update is not available.")
            return self._update_cache[param]

    return _MechanicOptimizer(*args, **kwargs)
