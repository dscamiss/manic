"""
Mechanic learning rate scale tuner.

References:

[1] A.Cutkosky, Aaron Defazio, and Harsh Mehta,
    Mechanic: A Learning Rate Tuner, arXiv:2306.00144.
"""

from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import jaxtyped
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from typeguard import typechecked as typechecker

from src.manic.constants import EPSILON
from src.manic.updater import Updater

# flake8: noqa=DCO010
# pylint: disable=invalid-name,not-callable


@dataclass
class MechanicParams:
    r"""
    Dataclass for core `Mechanic` parameters.

    Correspondence with the parameters of Algorithm 1 in [1]:
        beta   : \beta (n-dimensional)
        decay  : \lambda (scalar)
        s_init : s_{init} (scalar)
        epsilon: \epsilon (scalar)
    """

    beta: Tensor = Tensor([0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999])
    decay: float = 1e-2
    s_init: float = 1e-8
    epsilon: float = EPSILON


@dataclass
class MechanicState:
    r"""
    Dataclass for `Mechanic` internal state variables.

    Correspondence with the variables of Algorithm 1 in [1]:
        h: h_t (scalar)
        m: m_t (n-dimensional)
        v: v_t (n-dimensional)
        r: r_t (n-dimensional)
        W: W_t (n-dimensional)
        s: s_t (n-dimensional)
    """

    h: Tensor = Tensor()
    m: Tensor = Tensor()
    v: Tensor = Tensor()
    r: Tensor = Tensor()
    W: Tensor = Tensor()
    s: Tensor = Tensor()


class Mechanic(LRScheduler):
    """
    Mechanic learning rate scale tuner.

    Args:
        updater: Wrapped optimizer and LR scheduler.
        last_epoch: Index of the last epoch (default = -1).
        mechanic_params: Core mechanic parameters.
    """

    @torch.no_grad()
    def __init__(
        self,
        updater: Updater,
        last_epoch: int = -1,
        mechanic_params: MechanicParams = MechanicParams(),
    ) -> None:
        self._updater = updater
        self._mechanic_params = mechanic_params
        self._mechanic_state = MechanicState()
        self._last_lr = None

        # Constructor is here since initial `get_lr()` needs `self._updater`
        super().__init__(updater.base_optimizer, last_epoch)

        # Initialize internal state variables
        self._initialize_state()

        # Compute derived parameter(s)
        beta = self._mechanic_params.beta
        self._mechanic_params.beta_sq = torch.pow(beta, 2.0)

    @torch.no_grad()
    def _initialize_state(self) -> None:
        """Initialize internal state variables."""
        state = self._mechanic_state
        beta = self._mechanic_params.beta

        state.h = torch.as_tensor(0.0)
        state.m = torch.zeros_like(beta)
        state.v = torch.zeros_like(beta)
        state.r = torch.zeros_like(beta)
        state.W = torch.zeros_like(beta)
        state.s = torch.zeros_like(beta)

    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def _compute_inner_product_state(self) -> None:
        """
        Compute the inner product state `h`.

        This implements line 10 of Algorithm 1 in [1].
        """
        updater = self._updater
        base_optimizer = updater.base_optimizer
        params = self._mechanic_params
        state = self._mechanic_state

        # Get current sum of `s` components
        s_sum = updater.s_sum

        # Compute current inner product state
        state.h.zero_()
        for group in base_optimizer.param_groups:
            for x in group["params"]:
                delta_flat = updater.get_delta(x).flatten()
                grad_flat = x.grad.flatten()
                grad_norm = torch.norm(x.grad)
                x_norm = torch.norm(x)
                x_scale = (params.decay * s_sum * grad_norm) / x_norm
                inner_product = torch.inner(delta_flat, grad_flat + x_scale * x.flatten())
                state.h.add_(inner_product)

    @torch.no_grad()
    def _compute_aux_states(self) -> None:
        """
        Compute the current auxiliary states `m`, `V`, `r`, and `W`.

        This implements lines 11 through 15 of Algorithm 1 in [1].
        """
        params = self._mechanic_params
        state = self._mechanic_state

        # Compute current auxiliary states
        state.m = torch.max(params.beta * state.m, state.h)
        state.v = params.beta_sq * state.v + state.h * state.h
        state.r = torch.clamp(params.beta * state.r - state.h * state.s, 0.0)
        state.W = (params.s_init / state.m.numel()) * state.m + state.r

    @torch.no_grad()
    def _compute_s_state(self) -> None:
        """
        Compute the current `s` state.

        This implements line 10 through 16 of Algorithm 1 in [1].
        """
        params = self._mechanic_params
        state = self._mechanic_state

        # Compute inner product and auxiliary states
        self._compute_inner_product_state()
        self._compute_aux_states()

        # Compute current `s` state
        state.s = state.W / (torch.sqrt(state.v) + params.epsilon)

    @torch.no_grad()
    def get_lr(self) -> list[float]:
        """
        Compute the current sum of `s` components.

        This is the learning rate scale used by `Mechanic`.

        The learning rate scale is replicated for each parameter group to
        maintain consistency with the `LRScheduler` implementation.
        """
        updater = self._updater
        base_optimizer = updater.base_optimizer

        # No gradients available on first call, so return default value
        if self.last_epoch == 0:
            return [updater.s_sum for _ in base_optimizer.param_groups]

        self._compute_s_state()
        s_sum = torch.sum(self._mechanic_state.s).item()
        return [s_sum for _ in base_optimizer.param_groups]

    # Note: Pylint complains about omitting deprecated `epoch` argument
    @torch.no_grad()
    def step(self) -> Any:  # pylint: disable=arguments-differ
        """Run one Mechanic step."""
        # Increment last epoch index
        # - This is a misnomer, since for Mechanic this is the batch index
        self.last_epoch += 1

        # Compute current sum of `s` components
        s_sum = self.get_lr()[0]

        # Send it to the updater
        self._updater.s_sum = s_sum

        # Record last LR for consistency with the `LRScheduler` implementation
        base_optimizer = self._updater.base_optimizer
        self._last_lr = [s_sum for group in base_optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        """Make `Mechanic` state dict."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["optimizer", "_updater"]
        }
