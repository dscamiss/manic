"""Test configuration."""

import random

import numpy as np
import pytest
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds etc. to attempt reproducibility.

    Args:
        seed: Random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="session", autouse=True)
def setup_session() -> None:
    """Set up for tests."""
    set_seed(11)
    # Working in float64 avoids numerical issues in tests
    torch.set_default_dtype(torch.float64)
