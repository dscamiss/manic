"""
Wrapped optimizer for Mechanic learning rate scheduler.

References:

[1] A.Cutkosky, Aaron Defazio, and Harsh Mehta,
    Mechanic: A Learning Rate Tuner, arXiv:2306.00144.
"""

from typing import Any

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.satanic.types import Optimizer, ParamTensorDict

# flake8: noqa=DCO010
# pylint: disable=too-few-public-methods


class MechanicOptimizer:
    r"""
    Wrapped optimizer for use by the Mechanic learning rate scheduler.

    Correspondence with the variables of Algorithm 1 in [1]:
        ref_params: \mathbf{x}_{ref}.
        delta     : \mathbf{\Delta}_t

    Args:
        base_optimizer: Instance of base optimizer.
        epsilon: Small value for numerical stability (default = 1e-8).
        store_delta: Store "delta" values between iterations (default = True).

    Example:
        model = make_model(...)
        sgd = MechanicOptimizer(torch.optim.SGD(model.parameters(), ...))
    """

    @torch.no_grad()
    def __init__(
        self, base_optimizer: Optimizer, epsilon: float = 1e-8, store_delta: bool = True
    ) -> None:
        self._base_optimizer = base_optimizer
        self._deltas = ParamTensorDict()
        self._epsilon = epsilon
        self._params = ParamTensorDict()
        self._ref_params = ParamTensorDict()
        self._s_sum = 0.0
        self._s_sum_prev = 0.0
        self._store_delta = store_delta
        self._updates = ParamTensorDict()
        self._updates_available = False

        # Sanity check on base optimizer objective
        for group in base_optimizer.param_groups:
            if "maximize" in group and group["maximize"]:
                raise ValueError("Maximization objective is not supported")

        # Sanity check on `epsilon`
        if epsilon <= 0.0:
            raise ValueError("Epsilon must be positive")

        # Initialize "delta" values and reference parameters
        for group in base_optimizer.param_groups:
            for x in group["params"]:
                if self._store_delta:
                    self._deltas[x] = torch.zeros_like(x)
                self._ref_params[x] = x.clone()

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

    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def get_delta(self, x: nn.Parameter) -> Float[Tensor, "..."]:
        """
        Get current "delta" value for a particular parameter.

        Args:
            x: Parameter.

        Returns:
            The "delta" value for parameter `x`.
        """
        if self._store_delta:
            delta = self._deltas[x]
        else:
            x_ref = self.ref_params[x]
            denom = self.s_sum + self.epsilon
            delta = x.clone().sub_(x_ref).div_(denom)
        return delta

    @torch.no_grad()
    def _refresh_params(self) -> None:
        """Refresh parameter cache with current values."""
        for group in self.param_groups:
            for x in group["params"]:
                self._params[x] = x.clone()

    @torch.no_grad()
    def _refresh_updates(self) -> None:
        """Refresh update cache with current values."""
        # Refresh parameter cache with current values
        self._refresh_params()

        # Run base optimizer step (this modifies parameters)
        self._base_optimizer.step()

        # Derive updates
        for group in self.param_groups:
            for x in group["params"]:
                x_prev = self._params[x]
                self._updates[x] = x.clone().sub_(x_prev)
                self._updates[x].div_(-1.0 * group["lr"])

        # Restore parameters
        self._restore_params()

        # Indicate that updates are available
        self._updates_available = True

    @torch.no_grad()
    def _restore_params(self) -> None:
        """Restore parameters from parameter cache."""
        for group in self.param_groups:
            for x in group["params"]:
                x.copy_(self._params[x])

    @property
    def base_optimizer(self) -> Optimizer:
        """Getter for `_base_optimizer`."""
        return self._base_optimizer

    def set_s_sum(self, s_sum: float) -> float:
        """Set sum of components value."""
        self._s_sum_prev = self._s_sum
        self._s_sum = s_sum

    @jaxtyped(typechecker=typechecker)
    def get_update(self, x: nn.Parameter) -> Float[Tensor, "..."]:
        """
        Get current update for a particular parameter.

        Args:
            x: Parameter.

        Returns:
            The update for parameter `x`.

        Raises:
            ValueError: If the update is not available.
        """
        if not self._updates_available or x not in self._updates:
            raise ValueError("Update is not available.")
        return self._updates[x]

    @torch.no_grad()
    def step(self) -> None:
        """
        Run a single optimizer step.

        This implements line 17 of Algorithm 1 in [1].
        """
        # Compute updates
        self._refresh_updates()

        # Adjust parameters
        for group in self.param_groups:
            for x in group["params"]:
                x_ref = self._ref_params[x]
                update = self._updates[x]
                if self._store_delta:
                    # Add update to stored "delta" value
                    self._deltas[x].add_(update)
                    new_delta = self._deltas[x]
                else:
                    # Add update to computed "delta" value
                    # - Note denominator uses data from previous iteration
                    denom = self._s_sum_prev + self._epsilon
                    new_delta = (x - x_ref).div_(denom).add_(update)
                x.copy_(x_ref + self._s_sum * new_delta)
