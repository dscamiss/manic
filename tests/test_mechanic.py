"""Test code for `Mechanic` class."""

# flake8: noqa=D401
# pylint: disable=protected-access,invalid-name

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from typeguard import typechecked as typechecker

from src.manic.mechanic import Mechanic, MechanicParams
from src.manic.tuner import Tuner
from src.manic.types import ParamTensorDict

_MECHANIC_LR_SCHEDULERS = ["mechanic_store_delta", "mechanic_compute_delta"]


@pytest.fixture(name="mechanic_params")
def fixture_mechanic_params() -> MechanicParams:
    """Core `Mechanic` parameters."""
    return MechanicParams()


@pytest.fixture(name="mechanic_store_delta")
def fixture_mechanic_store_delta(
    sgd_store_delta: Tuner, mechanic_params: MechanicParams
) -> Mechanic:
    """`Mechanic` learning rate scheduler."""
    return Mechanic(sgd_store_delta, -1, mechanic_params)


@pytest.fixture(name="mechanic_compute_delta")
def fixture_mechanic_compute_delta(
    sgd_compute_delta: Tuner, mechanic_params: MechanicParams
) -> Mechanic:
    """`Mechanic` learning rate scheduler."""
    return Mechanic(sgd_compute_delta, -1, mechanic_params)


@pytest.fixture(name="mechanic")
def fixture_mechanic(mechanic_store_delta: Mechanic) -> Mechanic:
    """Aliased test fixture for brevity."""
    return mechanic_store_delta


def test_creation(mechanic_params: MechanicParams, mechanic: Mechanic) -> None:
    """Test `Mechanic` class types."""
    assert isinstance(mechanic, Mechanic), "Invalid class type"
    assert len(type(mechanic).__bases__) == 1, "Invalid inheritance pattern"
    assert type(mechanic).__bases__[0] == LRScheduler, "Invalid superclass type"

    assert isinstance(mechanic._mechanic_params, MechanicParams), "Invalid class type"
    assert mechanic._mechanic_params == mechanic_params, "Invalid core parameters"


@jaxtyped(typechecker=typechecker)
def test_step_side_effects(
    model: nn.Module, mechanic: Mechanic, x: Float[Tensor, "..."], y: Float[Tensor, "..."]
) -> None:
    """
    Test `step()` for side effects.

    The base optimizer's learning rates should not be modified.
    """
    tuner = mechanic._tuner
    base_optimizer = tuner._base_optimizer

    # Get learning rate(s) from base optimizer
    base_lrs: dict[int, float] = {}
    for idx, group in enumerate(base_optimizer.param_groups):
        base_lrs[idx] = group["lr"]

    # Compute gradients
    base_optimizer.zero_grad()
    torch.nn.MSELoss()(model(x), y).backward()

    # Run one scheduler step
    mechanic.step()

    # Run one tuner step
    tuner.step()

    # Check learning rate(s) in base optimizer
    for idx, group in enumerate(tuner._base_optimizer.param_groups):
        assert group["lr"] == base_lrs[idx]


@jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("_mechanic", _MECHANIC_LR_SCHEDULERS)
def test_step(
    request, model: nn.Module, _mechanic: str, x: Float[Tensor, "..."], y: Float[Tensor, "..."]
) -> None:
    """Test `step()` behavior over two iterations."""
    # Get test fixture
    mechanic = request.getfixturevalue(_mechanic)

    # Aliases for brevity
    tuner = mechanic._tuner
    base_optimizer = tuner._base_optimizer
    params = mechanic._mechanic_params
    state = mechanic._mechanic_state

    # Make local parameter cache for comparisons
    ref_params = ParamTensorDict()
    for p in model.parameters():
        ref_params[p] = p.clone().detach()

    # --- Iteration 1 ---

    # Compute gradients
    base_optimizer.zero_grad()
    torch.nn.MSELoss()(model(x), y).backward()

    # Run one scheduler step
    mechanic.step()

    # Check scheduler internal state
    err_str = "Error in internal state: "
    assert state.h == 0.0, err_str + " h"
    assert torch.all(state.m == 0.0), err_str + "m"
    assert torch.all(state.v == 0.0), err_str + "v"
    assert torch.all(state.r == 0.0), err_str + "r"
    assert torch.all(state.W == 0.0), err_str + "W"
    assert torch.all(state.s == 0.0), err_str + "s"

    # Run one tuner step
    tuner.step()

    # Check parameter values
    for p in model.parameters():
        assert torch.equal(p, ref_params[p]), "Error in parameter values"

    # --- Iteration 2 ---

    # Compute gradients
    # - For convenience, we reuse the previous batch
    base_optimizer.zero_grad()
    torch.nn.MSELoss()(model(x), y).backward()

    # Run one scheduler step
    mechanic.step()

    # Check scheduler internal state
    err_str = "Error in internal state: "
    beta = params.beta
    epsilon = params.epsilon

    h_expected = torch.as_tensor(0.0)
    for p in model.parameters():
        delta_flat = tuner.get_delta(p).flatten()
        grad_flat = p.grad.flatten()
        h_expected.add_(torch.inner(delta_flat, grad_flat))

    assert state.h == h_expected, err_str + "h"

    # The following computations use expected values from iteration 1
    m_expected = torch.max(torch.zeros_like(beta), h_expected)
    v_expected = torch.pow(h_expected, 2.0) * torch.ones_like(beta)
    r_expected = torch.zeros_like(beta)
    W_expected = (params.s_init / beta.numel()) * m_expected
    s_expected = W_expected / (torch.sqrt(v_expected) + epsilon)

    assert torch.all(state.m == m_expected), err_str + "m"
    assert torch.all(state.v == v_expected), err_str + "v"
    assert torch.all(state.r == r_expected), err_str + "r"
    assert torch.all(state.W == W_expected), err_str + "W"
    assert torch.all(state.s == s_expected), err_str + "s"

    # Run one tuner step
    tuner.step()

    # Check parameter values
    for p in model.parameters():
        p_expected = ref_params[p] + torch.sum(s_expected) * p.grad
        assert torch.allclose(p, p_expected), "Error in parameter values"
