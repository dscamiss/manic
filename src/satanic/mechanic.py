"""
Mechanic learning rate scheduler.

References:

[1] A.Cutkosky, Aaron Defazio, and Harsh Mehta,
    Mechanic: A Learning Rate Tuner, arXiv:2306.00144.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from jaxtyping import Num
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler

from src.satanic import MechanicOptimizer

# flake8: noqa=DCO010
# pylint: disable=invalid-name

_TensorDict = dict[str, Num[Tensor, "..."]]

_DEFAULT_BETA = Tensor([0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999])
_DEFAULT_LR = 1e-3

@dataclass
class MechanicParams:
    r"""
    Dataclass for `Mechanic` base parameters.

    Correspondence with the parameters of Algorithm 1 in [1]:

        beta        : $\beta$ (n-dimensional)
        lambda_decay: $\lambda$ (scalar)
        s_init      : $s_{init}$ (scalar)
        epsilon     : $\epsilon$ (scalar).

    """

    beta: Tensor = _DEFAULT_BETA
    lambda_decay: int = 1e-2
    s_init: float = 1e-8
    epsilon: float = 1e-8


@dataclass
class MechanicState:
    r"""
    Dataclass for `Mechanic` internal state.

    Correspondence with the variables of Algorithm 1 in [1]:

        ref_params: $\mathbf{x}_{ref}$.
        delta     : $\Delta_t$
        delta_next: $\Delta_{t+1}$
        h         : $h_t$ (scalar)
        m         : $m_t$ (n-dimensional)
        v         : $v_t$ (n-dimensional)
        r         : $r_t$ (n-dimensional)
        W         : $W_t$ (n-dimensional)
        s_next    : $s_{t+1}$ (n-dimensional)

    """

    ref_params: _TensorDict = field(default_factory=dict)
    delta: _TensorDict = field(default_factory=dict)
    delta_next: _TensorDict = field(default_factory=dict)
    h: Tensor = torch.as_tensor(0.0)
    m: Tensor = Tensor()
    v: Tensor = Tensor()
    r: Tensor = Tensor()
    W: Tensor = Tensor()
    s_next = Tensor()


class Mechanic(LRScheduler):
    """
    Mechanic learning rate scheduler.

    Args:
        optimizer: Wrapped optimizer.
        last_epoch: Index of the last epoch (default = -1).
        mechanic_params: Mechanic base parameters.
        track_state: Track internal state (default = False).  This must be
            set to `True` to resume the scheduler after interruption.
    """

    def __init__(
        self,
        optimizer: MechanicOptimizer,
        last_epoch: int = -1,
        mechanic_params: MechanicParams = MechanicParams(),
        track_state: bool = False,
    ) -> None:
        super().__init__(optimizer.base_optimizer, last_epoch)
        self._mechanic_params = mechanic_params
        self._mechanic_state = MechanicState()
        self._track_state = track_state
        self._initialize_state()

    @torch.no_grad()
    def _initialize_state(self) -> None:
        state = self._mechanic_state
        beta = self._mechanic_params.beta

        state.h = torch.as_tensor(0.0)

        state.m = torch.zeros_like(beta)
        state.v = torch.zeros_like(beta)
        state.r = torch.zeros_like(beta)
        state.W = torch.zeros_like(beta)
        state.s_next = torch.zeros_like(beta)

        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    state.delta[param] = torch.zeros_like(param)
                    state.delta_next[param] = torch.zeros_like(param)
                    state.ref_params[param] = param.clone()

    def get_lr(self) -> list[float]:
        """Compute learning rate."""
        if self.last_epoch == 0:
            return [_DEFAULT_LR for group in self.optimizer.param_groups]
        return [_DEFAULT_LR for group in self.optimizer.param_groups]  # TODO

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
