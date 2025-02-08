"""Fully-connected neural network."""

# flake8: noqa=DCO010
# pylint: disable=missing-function-docstring

from typing import Any

from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker


class FullyConnected(nn.Module):
    """Fully-connected neural network with ReLU activations.

    Args:
        input_dim: Input dimension.
        hidden_layer_dims: Hidden layer dimensions.
        output_dim: Output dimension.
        negative_slope: Negative slope for leaky ReLU (default = 0.0).
        final_activation: Apply activation at final layer (default = True)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_dims: list[int],
        output_dim: int,
        negative_slope: float = 0.0,
        final_activation: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layer_dims = hidden_layer_dims
        self.output_dim = output_dim
        self.negative_slope = negative_slope
        self.final_activation = final_activation

        layers: list[Any] = []

        if not hidden_layer_dims:
            # Edge case: No hidden layers
            layers.append(nn.Linear(input_dim, output_dim))
            if final_activation:
                layers.append(nn.LeakyReLU(negative_slope))
        else:
            # Generic case: At least one hidden layer
            layers = []
            layers.append(nn.Linear(input_dim, hidden_layer_dims[0]))
            layers.append(nn.LeakyReLU(negative_slope))

            for i in range(1, len(hidden_layer_dims)):
                layers.append(nn.Linear(hidden_layer_dims[i - 1], hidden_layer_dims[i]))
                layers.append(nn.LeakyReLU(negative_slope))

            layers.append(nn.Linear(hidden_layer_dims[-1], output_dim))
            if final_activation:
                layers.append(nn.LeakyReLU(negative_slope))

        self.layers = nn.Sequential(*layers)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "b input_dim"]) -> Float[Tensor, "b output_dim"]:
        return self.layers(x)
