"""Wrapped optimizer for Mechanic learning rate scheduler."""

from typing import Any

import torch
from jaxtyping import Num, jaxtyped
from torch import Tensor
from typeguard import typechecked as typechecker

from src.satanic.types import Optimizer

# flake8: noqa=DCO010
# pylint: disable=too-few-public-methods


class MechanicOptimizer:
    """
    Wrapped optimizer for use by the Mechanic learning rate scheduler.

    Args:
        base_optimizer: Instance of base optimizer.

    Example:
        model = make_model(...)
        sgd = MechanicOptimizer(torch.optim.SGD(model.parameters(), ...))
    """

    def __init__(self, base_optimizer: Optimizer) -> None:
        self._base_optimizer = base_optimizer
        self._param_cache: dict[Tensor, Tensor] = {}
        self._update_cache: dict[Tensor, Tensor] = {}
        self._updates_available = False

    def __getattr__(self, attr_name: str) -> Any:
        """
        Redirect undefined attributes to base optimizer.

        Args:
            attr_name: Attribute name.

        Returns:
            Result of accessing attribute `attr_name` in the base optimizer.
        """
        attr_obj = getattr(self._base_optimizer, attr_name)
        if callable(attr_obj):

            def redirect(*args: Any, **kwargs: dict[str, Any]) -> Any:
                attr_obj(*args, **kwargs)

            return redirect
        return attr_obj

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
        self._base_optimizer.step()

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
