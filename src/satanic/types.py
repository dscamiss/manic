"""Custom types."""

import torch
from jaxtyping import Float
from torch import Tensor, nn

Optimizer = torch.optim.SGD
ParamTensorDict = dict[nn.Parameter, Float[Tensor, "..."]]
