"""
Mechanic learning rate scheduler.

References:

[1] A.Cutkosky, Aaron Defazio, and Harsh Mehta,
    Mechanic: A Learning Rate Tuner, arXiv:2306.00144.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from jaxtyping import jaxtyped
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from typeguard import typechecked as typechecker

from src.satanic.mechanic_optimizer import MechanicOptimizer

# flake8: noqa=DCO010
# pylint: disable=invalid-name,not-callable

_DEFAULT_BETA = Tensor([0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999])
_DEFAULT_S_SUM = 0.0


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

    beta: Tensor = _DEFAULT_BETA
    decay: float = 1e-2
    s_init: float = 1e-8
    epsilon: float = 1e-8


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
    Mechanic learning rate scheduler.

    Args:
        mechanic_optimizer: Wrapped optimizer.
        last_epoch: Index of the last epoch (default = -1).
        mechanic_params: Core mechanic parameters.
    """

    @torch.no_grad()
    def __init__(
        self,
        mechanic_optimizer: MechanicOptimizer,
        last_epoch: int = -1,
        mechanic_params: MechanicParams = MechanicParams(),
    ) -> None:
        self._mechanic_optimizer = mechanic_optimizer
        self._mechanic_params = mechanic_params
        self._mechanic_state = MechanicState()

        super().__init__(mechanic_optimizer.base_optimizer, last_epoch)

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
        Compute the current inner product state.

        This implements line 10 of Algorithm 1 in [1].
        """
        optimizer = self._mechanic_optimizer
        params = self._mechanic_params
        state = self._mechanic_state

        # Compute current sum of components
        s_sum = torch.sum(state.s)

        # Compute current inner product state
        state.h.zero_()
        for group in optimizer.param_groups:
            for x in group["params"]:
                delta_flat = optimizer.get_delta(x).flatten()
                grad_flat = x.grad.flatten()
                grad_norm = torch.norm(x.grad)
                x_norm = torch.norm(x)
                x_scale = (params.decay * s_sum * grad_norm) / x_norm
                inner_product = torch.inner(delta_flat, grad_flat + x_scale * x.flatten())
                state.h.add_(inner_product)

    @torch.no_grad()
    def _compute_aux_states(self) -> None:
        """
        Compute the current auxiliary states.

        This implements lines 11 through 16 of Algorithm 1 in [1].
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

        This implements lines 10 through 16 of Algorithm 1 in [1].
        """
        params = self._mechanic_params
        state = self._mechanic_state

        # Compute inner product and auxiliary states
        self._compute_inner_product_state()
        self._compute_aux_states()

        # Compute current `s` state
        state.s = state.W / (torch.sqrt(state.v) + params.epsilon)

    @torch.no_grad()
    def get_lr(self) -> float:
        """
        Compute the current sum of components.

        This is (effectively) the learning rate used by `Mechanic`.

        Calling this function should *not* change the learning rates
        used by the base optimizer.
        """
        if self.last_epoch == 0:
            return _DEFAULT_S_SUM
        self._compute_s_state()
        return torch.sum(self._mechanic_state.s).item()

    # Pylint complains about omitting deprecated `epoch` argument
    @torch.no_grad()
    def step(self) -> None:  # pylint: disable=arguments-differ
        """Run one scheduler step."""
        # Increment last epoch index (really, the batch index)
        self.last_epoch += 1

        # Compute current sum of components
        s_sum = self.get_lr()

        # Send it to the optimizer
        self._mechanic_optimizer.set_s_sum(s_sum)

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state dict."""

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load scheduler state dict.

        Args:
            state_dict: State dict to load.
        """


def main():
    """Test Mechanic."""
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 1),
        torch.nn.ReLU(),
    )
    optimizer = MechanicOptimizer(torch.optim.SGD(model.parameters()))
    sgd = optimizer.base_optimizer
    print(sgd.param_groups[0]["lr"])
    for group in sgd.param_groups:
        for param in group["params"]:
            print(param)
    mechanic = Mechanic(optimizer)


if __name__ == "__main__":
    main()
