"""
Updater: A wrapped optimizer and LR scheduler for use by `Mechanic`.

References:

[1] A.Cutkosky, Aaron Defazio, and Harsh Mehta,
    Mechanic: A Learning Rate Tuner, arXiv:2306.00144.
"""

# flake8: noqa=DCO010
# pylint: disable=too-few-public-methods

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from typeguard import typechecked as typechecker

from src.manic.constants import EPSILON
from src.manic.types import Optimizer, ParamTensorDict


class _StaticLRScheduler(LRScheduler):
    """
    Static LR scheduler.

    This maintains constant learning rate(s) in the optimizer.

    Args:
        optimizer: Optimizer instance.
        last_epoch: Index of last epoch (default = -1).
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Return base learning rate(s)."""
        return self.base_lrs


@dataclass
class UpdaterParams:
    """
    Dataclass for `Updater` parameters.

    Args:
        epsilon: Small value for numerical stability (default = `EPSILON`).
        store_delta: Store "delta" values between iterations (default = `True`).
    """

    epsilon: float = EPSILON
    store_delta: bool = True


@dataclass
class UpdaterState:
    r"""
    Dataclass for `Updater` state variables.

    Args:
        model_params: Model parameter values.
        ref_model_params: Reference (i.e., initial) model parameter values.
        deltas: The "delta" values.
        updates: Model parameter update values.
        s_sum: Sum of `s` components.
        s_sum_prev: Previous sum of `s` components.

    Note:
        The correspondence with the variables of Algorithm 1 in [1] is:
            model_params: \mathbf{x}_t
            ref_model_params: \mathbf{x}_{ref}
            deltas: \mathbf{\Delta}_t
            updates: \mathbf{u}_t
            s_sum: \sum_{i=1}^n s_{t,i}
            s_sum_prev: \sum_{i=1}^n s_{t-1,i}
    """

    model_params: ParamTensorDict = field(default_factory=dict)
    ref_model_params: ParamTensorDict = field(default_factory=dict)
    deltas: ParamTensorDict = field(default_factory=dict)
    updates: ParamTensorDict = field(default_factory=dict)
    s_sum: float = 0.0
    s_sum_prev: float = 0.0


class Updater:
    """
    A wrapped optimizer and LR scheduler for use by `Mechanic`.

    Args:
        base_optimizer: Instance of base optimizer.
        base_lr_scheduler: Instance of base LR scheduler.
        updater_params: `Updater` parameters.

    Raises:
        ValueError: If any arguments are invalid.

    Note:
        In principle, this class could be extended to any optimization process
        that updates model parameters, not just one that uses the standard
        "optimizer step and LR scheduler step" paradigm.  To do so, we would
        need to refactor `Mechanic` since it is currently implemented as an
        `LRScheduler` whose associated optimizer is the base optimizer.

        This extension is probably unnecessary since the "optimizer step and
        LR scheduler step" paradigm is enough to cover all normal use cases.
    """

    @torch.no_grad()
    def __init__(
        self,
        base_optimizer: Optimizer,
        base_lr_scheduler: Optional[LRScheduler] = None,
        updater_params: UpdaterParams = UpdaterParams(),
    ) -> None:
        # Default to static LR scheduler
        if base_lr_scheduler is None:
            base_lr_scheduler = _StaticLRScheduler(base_optimizer)

        # Sanity check on base optimizer's objective
        for group in base_optimizer.param_groups:
            if "maximize" in group and group["maximize"]:
                raise ValueError("Maximization objective is not supported")

        self._base_optimizer = base_optimizer
        self._base_lr_scheduler = base_lr_scheduler
        self._updater_params = updater_params
        self._updater_state = UpdaterState()

        # Initialize reference model parameter values and "delta" values
        for group in self._base_optimizer.param_groups:
            for x in group["params"]:
                self._updater_state.ref_model_params[x] = x.clone()
                if self._updater_params.store_delta:
                    self._updater_state.deltas[x] = torch.zeros_like(x)

    @property
    @torch.no_grad()
    def base_optimizer(self) -> Optimizer:
        """Get base optimizer."""
        return self._base_optimizer

    @property
    @torch.no_grad()
    def s_sum(self) -> float:
        """Get sum of `s` components value."""
        return self._updater_state.s_sum

    @s_sum.setter
    @torch.no_grad()
    def s_sum(self, s_sum: float) -> None:
        """Set sum of `s` components value."""
        state = self._updater_state
        state.s_sum_prev = state.s_sum
        state.s_sum = s_sum

    @torch.no_grad()
    def _refresh_model_params(self) -> None:
        """Refresh model parameter values."""
        for group in self._base_optimizer.param_groups:
            for x in group["params"]:
                self._updater_state.model_params[x] = x.clone()

    @torch.no_grad()
    def _refresh_updates(self) -> None:
        """Refresh model parameter update values."""
        # Refresh model parameter values
        self._refresh_model_params()

        # Run base optimizer step plus base LR scheduler step
        # - Calling the optimizer step before the LR scheduler step is the
        #   expected order of operations for PyTorch >= 1.1.0.
        self._base_optimizer.step()
        self._base_lr_scheduler.step()

        # Derive model parameter update values
        state = self._updater_state
        for group in self._base_optimizer.param_groups:
            for x in group["params"]:
                x_prev = state.model_params[x]
                state.updates[x] = x.clone().sub_(x_prev)

        # Restore previous model parameters
        self._restore_model_params()

    @torch.no_grad()
    def _restore_model_params(self) -> None:
        """Restore model parameter values."""
        for group in self._base_optimizer.param_groups:
            for x in group["params"]:
                x.copy_(self._updater_state.model_params[x])

    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def get_delta(self, x: nn.Parameter, s_sum: Optional[float] = None) -> Float[Tensor, "..."]:
        """
        Get "delta" value for a particular model parameter.

        Args:
            x: Model parameter.
            s_sum: Sum of `s` components value to use (default = None).  This
                is only effective when the `store_delta` parameter is `True`.

        Returns:
            The "delta" value for model parameter `x`.
        """
        state = self._updater_state
        params = self._updater_params

        if params.store_delta:
            return state.deltas[x]

        if s_sum is None:
            s_sum = state.s_sum_prev

        s_sum = state.s_sum_prev if s_sum is None else state.s_sum
        x_ref = state.ref_model_params[x]
        denom = s_sum + params.epsilon
        return x.clone().sub_(x_ref).div_(denom)

    @jaxtyped(typechecker=typechecker)
    def get_update(self, x: nn.Parameter) -> Float[Tensor, "..."]:
        """
        Get update for a particular parameter.

        Args:
            x: Parameter.

        Returns:
            The update for parameter `x`.

        Raises:
            ValueError: If the update is not available.
        """
        if x not in self._updater_state.updates:
            raise ValueError("Update is not available.")
        return self._updater_state.updates[x]

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load `Updater` state."""
        self.__dict__.update(state_dict)

    def state_dict(self) -> dict[str, Any]:
        """Return `Updater` state as a dict."""
        return self.__dict__

    @torch.no_grad()
    def step(self) -> None:
        """
        Run one updater step.

        This implements line 17 of Algorithm 1 in [1].
        """
        # Refresh model parameter update values
        self._refresh_updates()

        # Adjust model parameters
        state = self._updater_state
        for group in self._base_optimizer.param_groups:
            for x in group["params"]:
                x_ref = state.ref_model_params[x]
                update = state.updates[x]
                # Here the "delta" values are computed using the *previous*
                # sum of `s` components value.  This is out of order compared
                # to the description in [1] but is efficient, since the
                # "delta" values are computed just in time.
                new_delta = self.get_delta(x, state.s_sum_prev).add_(update)
                x.copy_(x_ref + state.s_sum * new_delta)
