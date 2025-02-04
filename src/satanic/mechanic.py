"""
Mechanic learning rate scheduler.

References:

[1] A.Cutkosky, Aaron Defazio, and Harsh Mehta,
    Mechanic: A Learning Rate Tuner, arXiv:2306.00144.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from typeguard import typechecked as typechecker

from src.satanic.mechanic_optimizer import MechanicOptimizer

# flake8: noqa=DCO010
# pylint: disable=invalid-name,not-callable

_TensorDict = dict[str, Float[Tensor, "..."]]

_DEFAULT_BETA = Tensor([0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999])
_DEFAULT_LR = 1e-3


@dataclass
class MechanicParams:
    r"""
    Dataclass for core `Mechanic` parameters.

    Correspondence with the parameters of Algorithm 1 in [1]:

        beta   : $$\beta$$ (n-dimensional)
        decay  : $$\lambda$$ (scalar)
        s_init : $$s_{init}$$ (scalar)
        epsilon: $$\epsilon$$ (scalar)

    Extra parameters:
        store_delta: Determines if $$\mathbf{\Delta}_t$$ is stored between iterations.
    """

    beta: Tensor = _DEFAULT_BETA
    decay: float = 1e-2
    s_init: float = 1e-8
    epsilon: float = 1e-8
    store_delta: bool = True


@dataclass
class MechanicState:
    r"""
    Dataclass for `Mechanic` internal state variables.

    Correspondence with the variables of Algorithm 1 in [1]:

        ref_params: $$\mathbf{x}_{ref}$$.
        delta     : $$\mathbf{\Delta}_t$$
        h         : $$h_t$$ (scalar)
        m         : $$m_t$$ (n-dimensional)
        v         : $$v_t$$ (n-dimensional)
        r         : $$r_t$$ (n-dimensional)
        W         : $$W_t$$ (n-dimensional)
        s         : $$s_t$$ (n-dimensional)

    Extra states:
        s_sum: Sum of components of `s`.
    """

    ref_params: _TensorDict = field(default_factory=dict)
    delta: _TensorDict = field(default_factory=dict)

    h: Tensor = torch.as_tensor(0.0)
    s_sum: Tensor = torch.as_tensor(0.0)

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

    def __init__(
        self,
        mechanic_optimizer: MechanicOptimizer,
        last_epoch: int = -1,
        mechanic_params: MechanicParams = MechanicParams(),
    ) -> None:
        super().__init__(mechanic_optimizer.base_optimizer, last_epoch)
        self._mechanic_optimizer = mechanic_optimizer
        self._mechanic_params = mechanic_params
        self._mechanic_state = MechanicState()
        self._initialize_state()

        # Compute derived parameters
        self._mechanic_params.beta_squared = torch.pow(self._mechanic_params.beta, 2)

    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def _get_delta(self, x: nn.Parameter) -> Float[Tensor, "..."]:
        params = self._mechanic_params
        state = self._mechanic_state

        if params.store_delta:
            delta = state.delta[x]
        else:
            # Recompute $$\Delta_t$$
            x_ref = state.ref_params[x]
            denom = state.s_sum.add_(params.epsilon)
            delta = (x - x_ref).div_(denom)
        return delta

    @torch.no_grad()
    def _initialize_state(self) -> None:
        """Initialize internal state variables."""
        state = self._mechanic_state

        for group in self.optimizer.param_groups:
            for x in group["params"]:
                if self._mechanic_params.store_delta:
                    state.delta[x] = torch.zeros_like(x)
                state.ref_params[x] = x.clone()

        state.h = torch.as_tensor(0.0)
        state.s_sum = torch.as_tensor(0.0)

        beta = self._mechanic_params.beta
        state.m = torch.zeros_like(beta)
        state.v = torch.zeros_like(beta)
        state.r = torch.zeros_like(beta)
        state.s = torch.zeros_like(beta)

    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def _compute_delta(self) -> None:
        """Compute next "delta" terms."""
        # Make aliases for brevity
        optimizer = self._mechanic_optimizer
        state = self._mechanic_state

        # 9: $$\mathbf{\Delta}_{t+1} = \mathbf{\Delta}_t +\mathbf{u}_t$$
        for group in optimizer.param_groups:
            for x in group["params"]:
                update = optimizer.get_update(x)
                state.delta[x].add_(update)

    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def _compute_inner_product(self) -> None:
        """Compute the current inner product term."""
        # Make aliases for brevity
        optimizer = self._mechanic_optimizer
        params = self._mechanic_params
        state = self._mechanic_state

        # Compute inner product term
        state.h.zero_()
        for group in optimizer.param_groups:
            for x in group["params"]:
                delta_flat = self._get_delta(x).flatten()
                grad_flat = x.grad.flatten()
                grad_norm = torch.norm(x.grad)
                x_norm = torch.norm(x)
                x_scale = (params.decay * state.s_sum * grad_norm) / x_norm  # TODO: Check OOO
                inner_product = torch.inner(delta_flat, grad_flat + x_scale * x.flatten())
                state.h.add_(inner_product)

    @torch.no_grad()
    def _compute_remaining_states(self) -> None:
        """Compute remaining internal states."""
        # Make aliases for brevity
        params = self._mechanic_params
        state = self._mechanic_state

        # $$m_t \leftarrow \mathrm{max}(\beta m_{t-1}, h_t)$$
        state.m = torch.max(params.beta * state.m, state.h)

        # $$v_t \leftarrow \beta^2 v_{t-1} + h_t^2$$
        state.v = params.beta_squared * state.v + state.h * state.h

        # $$r_t \leftarrow \mathrm{max}(\beta r_{t-1} - h_t s_t, 0_n)$$
        state.r = params.beta * state.r - state.h * state.s
        state.r = torch.clamp(state.r, 0.0, None)

        # $$W_t \leftarrow (s_{init} / n) m_t + r_t$$
        state.W = (params.s_init / state.m.numel()) * state.m + state.r

        # $$s_{t+1} \leftarrow W_t / (\sqrt{v_t}+ \epsilon)$$
        state.s = state.W / (torch.sqrt(state.v) + params.epsilon)

        # Compute sum of components of `s`
        state.s_sum = torch.sum(state.s)

    def _update_learning_rates(self) -> None:
        """Update learning rates in optimizer."""
        for group in self._mechanic_optimizer.param_groups:
            group["lr"] = self._mechanic_state.s_sum

    def get_lr(self) -> list[float]:
        """Compute learning rate."""
        if self.last_epoch == 0:
            return [_DEFAULT_LR for group in self.optimizer.param_groups]

        # 7: Send $$\mathbf{g}_t$$ to BASE, receive update $$\mathbf{u}_t$$
        self._mechanic_optimizer._refresh_update_cache()

        # 10: $$h_t \leftarrow \langle \Delta_t, g_t + S_t \rangle$$
        #     $$S_t = \frac{\lambda (\sum_{i=1}^n s_{t,i}) \| g_t \| x_t}{\| x_t \|}$$
        self._compute_inner_product()

        # Compute "delta" state, if needed
        # 9: $$\Delta_{t+1} \leftarrow = \Delta_t + u_t$$
        if self._mechanic_params.store_delta:
            self._compute_delta()

        # Compute remaining states
        self._compute_remaining_states()

        # 16: $$x_{t+1} \leftarrow x_{ref} + \left( \sum_{i=1}^n s_{t+1,i} \right) \Delta_{t+1}$$
        self._update_learning_rates()

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
